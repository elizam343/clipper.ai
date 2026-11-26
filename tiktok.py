#!/usr/bin/env python3
"""TikTok and Instagram Reels helpers for clipper's output folders.

    python tiktok.py captions "clips/<video folder>" ["clips/<another>" ...]

New clipper runs already write a caption for TikTok and Instagram Reels into each
clip's .txt (the "TikTok / Reels:" block at the end). `captions` adds that block to folders made before that, without
touching the rest of the .txt (so your edits there are kept). It asks Claude when
ANTHROPIC_API_KEY is set, otherwise it builds one from the title and hashtags.

    python tiktok.py queue "clips/<video folder>" [...] [--per-day 2] [--days 7]

`queue` picks the next clips to post to TikTok (one source video's clips spread out,
clips already on YouTube first so the two can be compared) and puts them in
clips/TikTok queue/ named by day and time, with captions.txt holding each caption.
AirDrop the folder to your phone and post them in order.

    python tiktok.py login                 # once: sign in to TikTok in the browser
    python tiktok.py post --queue 1 2 --privacy SELF_ONLY

`post` sends clips to TikTok through the Content Posting API (Direct Post), using the
"TikTok / Reels" caption. TikTok keeps posts from an app it hasn't audited private
(SELF_ONLY) and only for accounts set to private; after the audit, PUBLIC_TO_EVERYONE
works. Setup: a TikTok developer app with Login Kit (Desktop, redirect URI
http://127.0.0.1:3455/callback/) and the Content Posting API (Direct Post on), its
client key and secret saved as secrets/tiktok_client.json:
{"client_key": "...", "client_secret": "..."}. secrets/ is git-ignored.

Without the API, posting is manual: copy the block when you post, then fill the tiktok_*
or ig_* columns (posted_at, url, views, avg_watch_s) in clips/posting_log.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import re
import sys
from pathlib import Path

import clipper


def log(*a):
    print("[tiktok]", *a, file=sys.stderr)


def folder_meta(folder: Path, js: dict, clips) -> dict:
    """The source video's title/creator/url from clips.json, plus the @handle from the
    credit line clipper wrote into the YouTube description."""
    parts = folder.name.rsplit(" - ", 2)
    meta = {"title": parts[0], "creator": parts[1] if len(parts) == 3 else ""}
    meta.update({k: v for k, v in (js.get("video") or {}).items() if v})
    for c in clips:
        m = re.search(r"Full video(?: by (\S+))?: (\S+)", (c.get("caption") or {}).get("caption", ""))
        if m:
            meta["handle"], meta["url"] = m.group(1) or "", m.group(2)
            break
    return meta


def claude_lines(clips, meta: dict):
    """{index: (line, hashtags)} from one Claude call, or None."""
    try:
        import anthropic
    except ImportError:
        return None
    items = [dict(id=i, title=c["caption"]["title"],
                  youtube_caption=c["caption"]["caption"].split("\n")[0],
                  transcript=(c.get("text") or "")[:1200]) for i, c in enumerate(clips)]
    prompt = (
        f"These clips are cut from \"{meta['title']}\" by {meta['creator'] or 'unknown'} "
        "and are being posted to TikTok and Instagram Reels. For each clip write one caption:\n"
        "- line: ONE punchy line, at most 100 characters, worded differently from the "
        "YouTube caption, based on what happens in the transcript. An emoji is fine. "
        "No hashtags or links\n"
        "- hashtags: 3 to 5 hashtags: the game/topic, the creator, and 1 to 2 broad ones "
        "people browse on TikTok and Reels (e.g. #gaming, #funny). No #shorts\n"
        "No profanity or swear words (even censored) and no sexual references. "
        "Don't invent events.\n\n" + json.dumps(items))
    schema = {"type": "object", "properties": {"captions": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "line": {"type": "string"},
                       "hashtags": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "line", "hashtags"], "additionalProperties": False}}},
        "required": ["captions"], "additionalProperties": False}
    try:
        resp = anthropic.Anthropic().beta.messages.create(
            model=clipper.CAPTION_MODEL, max_tokens=3000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}])
        text = next(b.text for b in resp.content if b.type == "text")
        return {d["id"]: (d["line"], d["hashtags"]) for d in json.loads(text)["captions"]}
    except Exception as ex:   # no key, API error, bad reply: fall back to basic
        log("Claude not used:", type(ex).__name__)
        return None


def set_tiktok_block(txt: Path, tiktok: str):
    """Replace (or add) the 'TikTok / Reels:' block at the end of a clip's .txt."""
    body = txt.read_text()
    body = re.split(r"\n*" + clipper.SOCIAL_HEADER_RE + r"\n", body, maxsplit=1,
                    flags=re.M)[0].rstrip("\n")
    txt.write_text(f"{body}\n\n{clipper.SOCIAL_HEADER}\n{tiktok}\n")


def captions(folder: Path, force: bool):
    js_path = folder / "clips.json"
    js = json.loads(js_path.read_text())
    clips = [c for c in js["clips"] if c.get("caption")]
    todo = [c for c in clips if force or not c["caption"].get("tiktok")]
    if not todo:
        log(f"{folder.name}: already has TikTok / Reels captions")
        return
    meta = folder_meta(folder, js, clips)
    got = claude_lines(todo, meta) or {}
    for i, c in enumerate(todo):
        cap = c["caption"]
        line, tags = got.get(i, (cap["title"], cap.get("hashtags") or []))
        cap["tiktok"] = clipper.clean_tiktok(clipper.tiktok_caption(line, tags, meta))
        for f in c["files"]:
            stem = f.replace(" (landscape).mp4", "").removesuffix(".mp4")
            txt = folder / f"{stem}.txt"
            if txt.exists():
                set_tiktok_block(txt, cap["tiktok"])
    js_path.write_text(json.dumps(js, indent=2))
    log(f"{folder.name}: {len(todo)} TikTok / Reels captions ({'Claude' if got else 'basic'})")


# ---------------------------------------------------------------- TikTok API

SECRETS = Path(__file__).resolve().parent / "secrets"
CLIENT = SECRETS / "tiktok_client.json"
TOKEN = SECRETS / "tiktok_token.json"
REDIRECT = "http://127.0.0.1:3455/callback/"
SCOPES = "user.info.basic,video.publish"
API = "https://open.tiktokapis.com"
PRIVACY = ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"]


def _save_secret(path: Path, data: dict):
    SECRETS.mkdir(mode=0o700, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def _client() -> dict:
    if not CLIENT.exists():
        sys.exit(f"[tiktok] missing {CLIENT}: save your TikTok app's client key and secret "
                 'there as {"client_key": "...", "client_secret": "..."}')
    c = json.loads(CLIENT.read_text())
    if not c.get("client_key") or not c.get("client_secret"):
        sys.exit(f"[tiktok] {CLIENT} needs client_key and client_secret")
    return c


def _request(method: str, url: str, *, json_body=None, form=None, token=None, data=None,
             headers=None):
    """Small urllib wrapper; returns parsed JSON (or {} for an empty body)."""
    import urllib.error
    import urllib.parse
    import urllib.request
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    if json_body is not None:
        data = json.dumps(json_body).encode()
        h["Content-Type"] = "application/json; charset=UTF-8"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
    except urllib.error.HTTPError as ex:
        body = ex.read()
        try:
            return json.loads(body)
        except ValueError:
            raise RuntimeError(f"HTTP {ex.code} from {url.split('?')[0]}") from None
    return json.loads(body) if body.strip() else {}


def _api(path: str, body: dict, token: str) -> dict:
    """POST to the Content Posting API; raises with TikTok's error code on failure."""
    r = _request("POST", API + path, json_body=body, token=token)
    err = r.get("error") or {}
    if err.get("code") not in (None, "ok"):
        raise RuntimeError(f"{err.get('code')}: {err.get('message', '')} (log id {err.get('logid', '?')})")
    return r.get("data") or {}


def _store_token(r: dict):
    import time
    if "access_token" not in r:
        raise RuntimeError(f"TikTok sign-in failed: {r.get('error', '')} "
                           f"{r.get('error_description', '')}".strip())
    now = time.time()
    _save_secret(TOKEN, dict(access_token=r["access_token"], refresh_token=r["refresh_token"],
                             open_id=r.get("open_id", ""), scope=r.get("scope", ""),
                             expires_at=now + r.get("expires_in", 0) - 60,
                             refresh_expires_at=now + r.get("refresh_expires_in", 0) - 60))


def login():
    """Desktop sign-in with PKCE: opens the browser, catches the redirect on 127.0.0.1."""
    import hashlib
    import http.server
    import secrets
    import urllib.parse
    import webbrowser
    c = _client()
    verifier = secrets.token_urlsafe(64)[:96]
    # TikTok's desktop docs ask for the hex SHA-256 of the verifier (not base64url)
    challenge = hashlib.sha256(verifier.encode()).hexdigest()
    state = secrets.token_urlsafe(16)
    url = "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode(dict(
        client_key=c["client_key"], scope=SCOPES, response_type="code",
        redirect_uri=REDIRECT, state=state, code_challenge=challenge,
        code_challenge_method="S256"))
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("Done. You can close this tab and go back to the terminal.".encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 3455), H)
    log("opening TikTok sign-in in your browser...")
    webbrowser.open(url)
    while not got:
        srv.handle_request()
    srv.server_close()
    if got.get("state") != state:
        sys.exit("[tiktok] sign-in reply didn't match (state); try again")
    if "code" not in got:
        sys.exit(f"[tiktok] sign-in cancelled: {got.get('error', '')} {got.get('error_description', '')}")
    r = _request("POST", API + "/v2/oauth/token/", form=dict(
        client_key=c["client_key"], client_secret=c["client_secret"], code=got["code"],
        grant_type="authorization_code", redirect_uri=REDIRECT, code_verifier=verifier))
    _store_token(r)
    log(f"signed in; token saved in {TOKEN}")


def access_token() -> str:
    """A valid access token, refreshed when it has expired (they last 2 hours)."""
    import time
    if not TOKEN.exists():
        sys.exit("[tiktok] not signed in: run  python3 tiktok.py login")
    t = json.loads(TOKEN.read_text())
    if time.time() < t["expires_at"]:
        return t["access_token"]
    if time.time() >= t["refresh_expires_at"]:
        sys.exit("[tiktok] sign-in expired: run  python3 tiktok.py login")
    c = _client()
    r = _request("POST", API + "/v2/oauth/token/", form=dict(
        client_key=c["client_key"], client_secret=c["client_secret"],
        grant_type="refresh_token", refresh_token=t["refresh_token"]))
    _store_token(r)   # TikTok returns a new refresh token every time
    return json.loads(TOKEN.read_text())["access_token"]


def video_seconds(path: Path) -> float:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def resolve_clips(queue_dir: Path, numbers, files):
    """[(video, txt, folder name, file name)] from queue numbers and/or clip paths."""
    out = []
    if numbers:
        items = {i["n"]: i for i in json.loads((queue_dir / "queue.json").read_text())}
        for n in numbers:
            if n not in items:
                sys.exit(f"[tiktok] #{n} isn't in {queue_dir}/queue.json")
            i = items[n]
            folder = Path(i["folder"])
            out.append((folder / i["file"], i["file"], folder))
    for f in files or []:
        out.append((f, f.name, f.parent))
    res = []
    for video, fname, folder in out:
        stem = fname.replace(" (landscape).mp4", "").removesuffix(".mp4")
        txt = folder / f"{stem}.txt"
        if not video.exists() or not txt.exists():
            sys.exit(f"[tiktok] can't find {video.name} and its .txt")
        res.append((video, txt, folder.name, fname))
    return res


def post(clips, privacy: str, allow_comments: bool, allow_duet: bool, allow_stitch: bool,
         yes: bool, log_path: Path):
    import time
    import upload
    token = access_token()
    info = _api("/v2/post/publish/creator_info/query/", {}, token)
    who = info.get("creator_nickname") or info.get("creator_username") or "?"
    options = info.get("privacy_level_options") or []
    print(f"\nPosting to TikTok as: {who} (@{info.get('creator_username', '?')})")
    if privacy not in options:
        sys.exit(f"[tiktok] this account can post as: {', '.join(options) or 'nothing right now'}. "
                 f"Pick one with --privacy.")
    max_s = info.get("max_video_post_duration_sec") or 0
    plan = []
    for video, txt, folder, fname in clips:
        caption = social_block(txt) or upload.read_caption(txt)["title"]
        secs = video_seconds(video)
        if max_s and secs > max_s:
            sys.exit(f"[tiktok] {video.name} is {secs:.0f}s; this account allows {max_s}s")
        plan.append((video, caption, folder, fname))
        print(f"\n  {video.name}  ({secs:.0f}s, {video.stat().st_size / 1e6:.1f} MB)\n"
              + "\n".join("    " + l for l in caption.splitlines()))
    print(f"\n  privacy: {privacy}; comments {'on' if allow_comments else 'off'}, duet "
          f"{'on' if allow_duet and not info.get('duet_disabled') else 'off'}, stitch "
          f"{'on' if allow_stitch and not info.get('stitch_disabled') else 'off'}; "
          "not promotional or branded content")
    print("  By posting, you agree to TikTok's Music Usage Confirmation "
          "(https://www.tiktok.com/legal/page/global/music-usage-confirmation/en).")
    if not yes and input("\nPost these to TikTok? [y/N] ").strip().lower() != "y":
        sys.exit("[tiktok] nothing posted")
    for video, caption, folder, fname in plan:
        size = video.stat().st_size
        # clips are well under 64 MB, which TikTok takes as one chunk
        if size > 64 * 1024 * 1024:
            sys.exit(f"[tiktok] {video.name} is over 64 MB; chunked upload isn't supported yet")
        log(f"uploading {video.name}")
        d = _api("/v2/post/publish/video/init/", dict(
            post_info=dict(title=caption, privacy_level=privacy,
                           disable_comment=not allow_comments or bool(info.get("comment_disabled")),
                           disable_duet=not allow_duet or bool(info.get("duet_disabled")),
                           disable_stitch=not allow_stitch or bool(info.get("stitch_disabled")),
                           video_cover_timestamp_ms=1000),
            source_info=dict(source="FILE_UPLOAD", video_size=size, chunk_size=size,
                             total_chunk_count=1)), token)
        _request("PUT", d["upload_url"], data=video.read_bytes(), headers={
            "Content-Type": "video/mp4", "Content-Length": str(size),
            "Content-Range": f"bytes 0-{size - 1}/{size}"})
        log("  processing (this may take a few minutes)...")
        st = {}
        for _ in range(60):
            time.sleep(10)
            st = _api("/v2/post/publish/status/fetch/", {"publish_id": d["publish_id"]}, token)
            if st.get("status") in ("PUBLISH_COMPLETE", "FAILED"):
                break
        if st.get("status") != "PUBLISH_COMPLETE":
            log(f"  not finished: {st.get('status')} {st.get('fail_reason', '')}")
            continue
        ids = st.get("publicaly_available_post_id") or []
        url = (f"https://www.tiktok.com/@{info.get('creator_username')}/video/{ids[0]}"
               if ids else "")
        _log_post(log_path, folder, fname, url, privacy)
        log(f"  done: {url or 'posted (' + privacy + '; link shows once it is public)'}")


def _log_post(log_path: Path, folder: str, fname: str, url: str, privacy: str):
    import csv
    import upload
    rows = upload.read_log(log_path)
    now = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    hit = False
    for r in rows:
        if r["video"] == folder and r["file"] == fname:
            r["tiktok_posted_at"], r["tiktok_url"], hit = now, url, True
            if privacy != "PUBLIC_TO_EVERYONE":
                r["notes"] = (r.get("notes", "") + "; " if r.get("notes") else "") + \
                             f"TikTok posted {privacy}"
    if not hit:
        rows.append({**{k: "" for k in clipper.LOG_FIELDS}, "video": folder, "file": fname,
                     "tiktok_posted_at": now, "tiktok_url": url})
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=clipper.LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def social_block(txt: Path) -> str:
    """The caption under the 'TikTok / Reels:' header of a clip's .txt ('' if none)."""
    parts = re.split(r"^" + clipper.SOCIAL_HEADER_RE[1:] + r"\n", txt.read_text(),
                     maxsplit=1, flags=re.M)
    return parts[1].strip() if len(parts) > 1 else ""


def queue(folders, per_day: int, days: int, times, start: dt.date, log_path: Path,
          out: Path):
    import upload
    rows = upload.read_log(log_path)
    done = {(r["video"], r["file"]) for r in rows if r.get("tiktok_url") or r.get("tiktok_posted_at")}
    yt = {(r["video"], r["file"]): r.get("posted_at") or "" for r in rows}
    groups = []
    for folder in folders:
        es = [e for e in upload.clip_entries(folder) if (e["folder"], e["file"]) not in done]
        # clips already on YouTube first (in their YouTube order), then the rest in order
        es.sort(key=lambda e: (not yt.get((e["folder"], e["file"])),
                               yt.get((e["folder"], e["file"])) or ""))
        if es:
            groups.append(es)
    picks = upload.interleave(groups)[:per_day * days]
    if not picks:
        log("nothing left to queue")
        return
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():   # rebuild: only files this command made
        if f.suffix in (".mp4", ".txt") or f.name == "queue.json":
            f.unlink()
    lines, items = [], []
    for k, e in enumerate(picks):
        day = start + dt.timedelta(days=k // per_day)
        at = dt.datetime.combine(day, times[k % per_day])
        label = f"{at:%a %b %d} {at:%-I%p}".replace("AM", "am").replace("PM", "pm")
        name = f"{k + 1:02d} - {label} - {e['video'].stem.replace(' (landscape)', '')}.mp4"
        try:
            os.link(e["video"], out / name)   # hard link: no extra disk space
        except OSError:
            shutil.copy2(e["video"], out / name)
        items.append(dict(n=k + 1, name=name, at=at.isoformat(timespec="minutes"),
                          folder=str(e["video"].parent.resolve()), file=e["file"]))
        cap = social_block(e["txt"]) or upload.read_caption(e["txt"])["title"]
        on_yt = yt.get((e["folder"], e["file"]))
        lines.append(f"=== {k + 1:02d}  {label}  ({e['folder'].split(' - ')[0][:40]}"
                     f"{'; on YouTube ' + on_yt[:10] if on_yt else ''})\n{cap}\n")
    (out / "captions.txt").write_text(
        "Post in order. After posting, put the time and link in clips/posting_log.csv\n"
        "(tiktok_posted_at, tiktok_url); add views and average watch time later.\n\n"
        + "\n".join(lines))
    (out / "queue.json").write_text(json.dumps(items, indent=2))
    log(f"{len(picks)} clips queued in {out} ({start:%b %d} to {day:%b %d}, {per_day} a day)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("captions", help="add TikTok / Reels captions to existing clip folders")
    c.add_argument("folders", nargs="+", type=Path)
    c.add_argument("--force", action="store_true", help="rewrite captions that exist")
    q = sub.add_parser("queue", help="put the next clips to post on TikTok in one folder")
    q.add_argument("folders", nargs="+", type=Path)
    q.add_argument("--per-day", type=int, default=2)
    q.add_argument("--days", type=int, default=7)
    q.add_argument("--times", default="12:00,19:00", help="post times, one per clip of a day")
    q.add_argument("--start", default=None, help="first day, YYYY-MM-DD (default: tomorrow)")
    q.add_argument("--log", type=Path, default=Path("clips/posting_log.csv"))
    q.add_argument("--out", type=Path, default=Path("clips/TikTok queue"))
    sub.add_parser("login", help="sign in to TikTok (once; the token is saved in secrets/)")
    pp = sub.add_parser("post", help="post clips to TikTok through the API")
    pp.add_argument("files", nargs="*", type=Path, help="clip .mp4 files (or use --queue)")
    pp.add_argument("--queue", nargs="+", type=int, metavar="N",
                    help="numbers from clips/TikTok queue (e.g. --queue 1 2)")
    pp.add_argument("--queue-dir", type=Path, default=Path("clips/TikTok queue"))
    # TikTok's rules: no default privacy, interactions off unless turned on
    pp.add_argument("--privacy", required=True, choices=PRIVACY)
    pp.add_argument("--allow-comments", action="store_true")
    pp.add_argument("--allow-duet", action="store_true")
    pp.add_argument("--allow-stitch", action="store_true")
    pp.add_argument("-y", "--yes", action="store_true", help="don't ask before posting")
    pp.add_argument("--log", type=Path, default=Path("clips/posting_log.csv"))
    a = ap.parse_args()
    if a.cmd == "login":
        login()
        return
    if a.cmd == "post":
        if not a.queue and not a.files:
            ap.error("give clip files or --queue N ...")
        post(resolve_clips(a.queue_dir, a.queue, a.files), a.privacy, a.allow_comments,
             a.allow_duet, a.allow_stitch, a.yes, a.log)
        return
    if a.cmd == "queue":
        times = [dt.time.fromisoformat(t.strip()) for t in a.times.split(",")]
        if len(times) < a.per_day:
            ap.error(f"--times needs {a.per_day} times")
        start = dt.date.fromisoformat(a.start) if a.start else dt.date.today() + dt.timedelta(days=1)
        folders = [f for f in a.folders if (f / "clips.json").exists()]
        queue(folders, a.per_day, a.days, times, start, a.log, a.out)
        return
    for folder in a.folders:
        if not (folder / "clips.json").exists():
            log(f"skipping {folder}: no clips.json")
            continue
        captions(folder, a.force)


if __name__ == "__main__":
    main()
