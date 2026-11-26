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

Posting there is manual for now: copy the block when you post, then fill the tiktok_*
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
        if f.suffix in (".mp4", ".txt"):
            f.unlink()
    lines = []
    for k, e in enumerate(picks):
        day = start + dt.timedelta(days=k // per_day)
        at = dt.datetime.combine(day, times[k % per_day])
        label = f"{at:%a %b %d} {at:%-I%p}".replace("AM", "am").replace("PM", "pm")
        name = f"{k + 1:02d} - {label} - {e['video'].stem.replace(' (landscape)', '')}.mp4"
        try:
            os.link(e["video"], out / name)   # hard link: no extra disk space
        except OSError:
            shutil.copy2(e["video"], out / name)
        cap = social_block(e["txt"]) or upload.read_caption(e["txt"])["title"]
        on_yt = yt.get((e["folder"], e["file"]))
        lines.append(f"=== {k + 1:02d}  {label}  ({e['folder'].split(' - ')[0][:40]}"
                     f"{'; on YouTube ' + on_yt[:10] if on_yt else ''})\n{cap}\n")
    (out / "captions.txt").write_text(
        "Post in order. After posting, put the time and link in clips/posting_log.csv\n"
        "(tiktok_posted_at, tiktok_url); add views and average watch time later.\n\n"
        + "\n".join(lines))
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
    a = ap.parse_args()
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
