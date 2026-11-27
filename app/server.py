"""Local server behind Clipper.app: runs clipper and upload.py, and serves the page.

Only listens on 127.0.0.1, and every API call must carry the per-launch token that is
written into the page, so other websites open in a browser can't drive it.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent          # the clipper folder
CLIPS = ROOT / "clips"
COMPLETED = CLIPS / "#COMPLETED"
LOG = CLIPS / "posting_log.csv"
STATE = CLIPS / "app_state.json"                        # skipped clips, upload settings
SKIP_DIRS = {".work", "#COMPLETED", "TikTok queue"}
TOKEN = secrets.token_urlsafe(24)

sys.path.insert(0, str(ROOT))
import upload  # noqa: E402  (light: doesn't load clipper's heavy imports)

app = Flask(__name__)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# ------------------------------------------------------------------ helpers

def notify(title: str, text: str):
    """A macOS notification (best effort)."""
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{esc(text)}" with title "{esc(title)}"'],
                   capture_output=True)


def in_shell(args):
    """Run with the user's shell setup (~/.zshrc has the Claude API key), since an app
    opened from the Dock doesn't get it. The key is never read here."""
    # -l loads the login profile too (Homebrew's PATH for ffmpeg is usually set there)
    return ["/bin/zsh", "-l", "-c",
            'export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"; '
            'source ~/.zshrc >/dev/null 2>&1; exec "$@"', "_", *args]


def state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(s: dict):
    STATE.write_text(json.dumps(s, indent=2))


def safe_folder(name: str) -> Path:
    p = (CLIPS / name).resolve()
    if p.parent != CLIPS.resolve() or not (p / "clips.json").exists():
        abort(404)
    return p


def stage_of(line: str, stage: str) -> str:
    l = line.lower()
    for key, name in [("downloading", "Downloading"), ("fetching chat", "Downloading"),
                      ("measuring loudness", "Finding the best moments"),
                      ("whisper", "Transcribing"), ("transcrib", "Transcribing"),
                      ("captions", "Writing captions"), ("rendering", "Rendering clips"),
                      ("uploading", "Uploading"), ("done:", "Done")]:
        if key in l:
            return name
    return stage


def run_job(kind: str, title: str, args, on_done=None) -> str:
    jid = uuid.uuid4().hex[:8]
    job = dict(id=jid, kind=kind, title=title, status="running", stage="Starting",
               lines=[], started=time.time(), folder=None, rendered=0)
    with LOCK:
        JOBS[jid] = job

    def work():
        p = subprocess.Popen(in_shell(args), cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if not line:
                continue
            with LOCK:
                job["lines"] = (job["lines"] + [line])[-200:]
                job["stage"] = stage_of(line, job["stage"])
                if "output folder:" in line:
                    job["folder"] = Path(line.split("output folder:", 1)[1].strip()).name
                if "rendering " in line:
                    job["rendered"] += 1
        code = p.wait()
        with LOCK:
            job["status"] = "done" if code == 0 else "failed"
            job["stage"] = "Done" if code == 0 else "Failed"
            job["ended"] = time.time()
        if on_done:
            on_done(job, code)

    threading.Thread(target=work, daemon=True).start()
    return jid


def log_rows():
    return upload.read_log(LOG)


def clip_list(folder: Path, rows, skipped):
    posted = {(r["video"], r["file"]): r for r in rows}
    out = []
    for e in upload.clip_entries(folder):
        cap = upload.read_caption(e["txt"])
        r = posted.get((e["folder"], e["file"]), {})
        out.append(dict(
            file=e["file"], title=cap["title"], description=cap["description"],
            hashtags=" ".join(cap["hashtags"]),
            video=f"{folder.name}/{e['file']}",
            thumb=f"{folder.name}/{e['thumb'].name}" if e["thumb"] else None,
            youtube_url=r.get("youtube_url", ""), posted_at=r.get("posted_at", ""),
            views=r.get("views", ""),
            skipped=f"{folder.name}/{e['file']}" in skipped))
    return out


# ------------------------------------------------------------------ API

@app.before_request
def check_token():
    if request.path.startswith("/api/") and request.headers.get("X-Clipper-Token") != TOKEN:
        abort(403)


@app.get("/")
def index():
    html = (Path(__file__).parent / "index.html").read_text()
    return html.replace("__TOKEN__", TOKEN)


@app.get("/icon.png")
def icon():
    return send_file(ROOT / "assets" / "icon.png")


@app.get("/media/<path:rel>")
def media(rel):
    p = (CLIPS / rel).resolve()
    if CLIPS.resolve() not in p.parents or not p.is_file():
        abort(404)
    return send_file(p, conditional=True)


@app.get("/api/jobs")
def jobs():
    with LOCK:
        js = [dict(j, lines=j["lines"][-8:]) for j in JOBS.values()]
    return jsonify(sorted(js, key=lambda j: -j["started"]))


@app.post("/api/jobs")
def new_job():
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not re.match(r"^https?://\S+$", url):
        return jsonify(error="Paste a full video link (https://...)"), 400
    fmt = d.get("format") if d.get("format") in ("portrait", "landscape", "both") else "landscape"
    args = [sys.executable, str(ROOT / "clipper.py"), url, "--format", fmt]
    if d.get("count"):
        args += ["-n", str(int(d["count"]))]
    for key in ("start", "end"):   # times like 1:30:00, 5400 or 1h30m
        t = str(d.get(key) or "").strip()
        if t:
            if not re.fullmatch(r"[0-9:hms. ]{1,12}", t):
                return jsonify(error=f"'{t}' isn't a time. Use 1:30:00 or 90:00"), 400
            args += [f"--{key}", t]

    def done(job, code):
        if code == 0:
            notify("Clipper", f"Clips are ready: {job.get('folder') or url}")
        else:
            notify("Clipper", "Making clips failed. Open Clipper to see why.")

    return jsonify(id=run_job("clip", url, args, done))


@app.get("/api/folders")
def folders():
    rows, skipped = log_rows(), set(state().get("skipped", []))
    out = []
    for f in sorted(CLIPS.iterdir(), key=lambda p: -p.stat().st_mtime):
        if f.is_dir() and f.name not in SKIP_DIRS and (f / "clips.json").exists():
            clips = clip_list(f, rows, skipped)
            out.append(dict(name=f.name, clips=clips,
                            waiting=sum(1 for c in clips if not c["youtube_url"] and not c["skipped"])))
    return jsonify(out)


@app.post("/api/caption")
def edit_caption():
    d = request.get_json(force=True)
    folder = safe_folder(d["folder"])
    e = next((e for e in upload.clip_entries(folder) if e["file"] == d["file"]), None)
    if not e:
        abort(404)
    title = re.sub(r"\s+", " ", d.get("title", "")).strip()
    desc = d.get("description", "").strip()
    if not title:
        return jsonify(error="The title can't be empty"), 400
    if len(title) > 100:
        return jsonify(error="YouTube titles can be at most 100 characters"), 400
    blocks = [b for b in e["txt"].read_text().split("\n\n") if b.strip()]
    rest = [b for b in blocks[1:] if b.strip().startswith(("#", "Tags:", "TikTok"))]
    e["txt"].write_text("\n\n".join([title] + ([desc] if desc else []) + [b.strip() for b in rest]) + "\n")
    return jsonify(ok=True)


@app.post("/api/skip")
def skip():
    d = request.get_json(force=True)
    safe_folder(d["folder"])
    key = f"{d['folder']}/{d['file']}"
    s = state()
    sk = set(s.get("skipped", []))
    sk.add(key) if d.get("skip") else sk.discard(key)
    s["skipped"] = sorted(sk)
    save_state(s)
    return jsonify(ok=True)


def upload_settings(d=None):
    s = state().get("upload", {})
    s = {"per_day": "2-3", "times": "13:00,16:00,19:00", "limit": 6, **s, **(d or {})}
    s["limit"] = max(1, min(int(s["limit"]), 6))
    return s


@app.post("/api/plan")
def plan():
    """The slots the next upload would use (same rules as upload.py)."""
    d = request.get_json(force=True)
    cfg = upload_settings(d.get("settings"))
    names = d.get("folders") or []
    rows, skipped = log_rows(), set(state().get("skipped", []))
    done = {(r["video"], r["file"]) for r in rows if r.get("youtube_url")}
    groups = []
    for n in names:
        f = safe_folder(n)
        groups.append([e for e in upload.clip_entries(f) if (e["folder"], e["file"]) not in done
                       and f"{e['folder']}/{e['file']}" not in skipped])
    entries = upload.interleave(groups)[:cfg["limit"]]
    times = [t.strip() for t in cfg["times"].split(",") if t.strip()]
    slots = upload.next_slots(len(entries), rows, times, cfg["per_day"]) if entries else []
    return jsonify(settings=cfg, plan=[dict(
        at=s.isoformat(timespec="minutes"), folder=e["folder"], file=e["file"],
        title=upload.read_caption(e["txt"])["title"],
        thumb=f"{e['folder']}/{e['thumb'].name}" if e["thumb"] else None)
        for e, s in zip(entries, slots)])


@app.post("/api/upload")
def do_upload():
    d = request.get_json(force=True)
    cfg = upload_settings(d.get("settings"))
    names = d.get("folders") or []
    if not names:
        return jsonify(error="Pick at least one video"), 400
    if any(j["kind"] == "upload" and j["status"] == "running" for j in JOBS.values()):
        return jsonify(error="An upload is already running"), 409
    s = state()
    s["upload"] = {k: cfg[k] for k in ("per_day", "times", "limit")}
    save_state(s)
    args = [sys.executable, str(ROOT / "upload.py"), *[str(safe_folder(n)) for n in names],
            "--per-day", str(cfg["per_day"]), "--times", cfg["times"], "--limit", str(cfg["limit"])]
    for k in s.get("skipped", []):
        args += ["--exclude", k]

    def done(job, code):
        n = sum(1 for l in job["lines"] if "done: https://" in l)
        notify("Clipper", f"Uploaded {n} clip{'s' * (n != 1)} to YouTube (private). Set their times in Studio."
               if code == 0 else f"Upload stopped after {n} clip{'s' * (n != 1)}. Open Clipper to see why.")

    return jsonify(id=run_job("upload", f"Upload to YouTube ({len(names)} video{'s' * (len(names) != 1)})",
                              args, done))


@app.get("/api/calendar")
def calendar():
    """Posts in the log with a time: past week and everything ahead."""
    today = dt.date.today()
    days: dict[str, list] = {}
    for r in log_rows():
        for col, url_col, where in (("posted_at", "youtube_url", "YouTube"),
                                    ("tiktok_posted_at", "tiktok_url", "TikTok"),
                                    ("ig_posted_at", "ig_url", "Instagram")):
            raw = (r.get(col) or "").strip()
            if not raw:
                continue
            try:
                t = dt.datetime.fromisoformat(raw.replace(" ", "T"))
            except ValueError:
                continue
            if t.date() < today - dt.timedelta(days=7):
                continue
            title = r.get("title") or re.sub(r"^\d+ - | \(landscape\)\.mp4$|\.mp4$", "", r["file"])
            days.setdefault(t.date().isoformat(), []).append(dict(
                time=t.strftime("%H:%M"), title=title, where=where, url=r.get(url_col, ""),
                views=r.get("views", "") if where == "YouTube" else r.get(f"{'tiktok' if where == 'TikTok' else 'ig'}_views", ""),
                video=r["video"], needs_studio="set this publish time in Studio" in (r.get("notes") or "")
                and t > dt.datetime.now(t.tzinfo)))
    for v in days.values():
        v.sort(key=lambda x: x["time"])
    return jsonify(today=today.isoformat(), days=dict(sorted(days.items())))


@app.post("/api/complete")
def complete():
    """Move a video's folder into clips/#COMPLETED (all its clips are posted or scheduled)."""
    d = request.get_json(force=True)
    f = safe_folder(d["folder"])
    COMPLETED.mkdir(exist_ok=True)
    dest = COMPLETED / f.name
    if dest.exists():
        return jsonify(error="A folder with that name is already in #COMPLETED"), 409
    shutil.move(str(f), str(dest))
    return jsonify(ok=True)


def serve(port: int):
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # don't log every request
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    print(f"Clipper: http://127.0.0.1:{port}/")
    serve(port)
