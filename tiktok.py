#!/usr/bin/env python3
"""TikTok helpers for clipper's output folders.

    python tiktok.py captions "clips/<video folder>" ["clips/<another>" ...]

New clipper runs already write a TikTok caption into each clip's .txt (the "TikTok:"
block at the end). `captions` adds that block to folders made before that, without
touching the rest of the .txt (so your edits there are kept). It asks Claude when
ANTHROPIC_API_KEY is set, otherwise it builds one from the title and hashtags.

Posting to TikTok is manual for now: copy the TikTok block when you post, then fill
tiktok_posted_at / tiktok_url / tiktok_views in clips/posting_log.csv.
"""
from __future__ import annotations

import argparse
import json
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
        "and are being posted to TikTok. For each clip write a TikTok caption:\n"
        "- line: ONE punchy line, at most 100 characters, worded differently from the "
        "YouTube caption, based on what happens in the transcript. An emoji is fine. "
        "No hashtags or links\n"
        "- hashtags: 3 to 5 hashtags: the game/topic, the creator, and 1 to 2 broad ones "
        "people browse on TikTok (e.g. #gaming, #funny). No #shorts\n"
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
    """Replace (or add) the 'TikTok:' block at the end of a clip's .txt."""
    body = txt.read_text()
    body = re.split(r"\n*^TikTok:\n", body, maxsplit=1, flags=re.M)[0].rstrip("\n")
    txt.write_text(f"{body}\n\nTikTok:\n{tiktok}\n")


def captions(folder: Path, force: bool):
    js_path = folder / "clips.json"
    js = json.loads(js_path.read_text())
    clips = [c for c in js["clips"] if c.get("caption")]
    todo = [c for c in clips if force or not c["caption"].get("tiktok")]
    if not todo:
        log(f"{folder.name}: already has TikTok captions")
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
    log(f"{folder.name}: {len(todo)} TikTok captions ({'Claude' if got else 'basic'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("captions", help="add TikTok captions to existing clip folders")
    c.add_argument("folders", nargs="+", type=Path)
    c.add_argument("--force", action="store_true", help="rewrite TikTok captions that exist")
    a = ap.parse_args()
    for folder in a.folders:
        if not (folder / "clips.json").exists():
            log(f"skipping {folder}: no clips.json")
            continue
        captions(folder, a.force)


if __name__ == "__main__":
    main()
