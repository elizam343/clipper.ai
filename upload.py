#!/usr/bin/env python3
"""Upload clipper's finished clips to YouTube, spaced out.

    python upload.py "clips/<video folder>" ["clips/<another>" ...] [--per-day 3] [--dry-run]

For each clip it sends the video, the title / description / hashtags / tags from its .txt
(so your edits there are used) and its thumbnail. Clips from several folders are
interleaved so one video's clips aren't posted back to back. Each upload gets a slot
(by default 1pm, 4pm and 7pm, starting after the last slot already in the posting log)
and the slot and link are written to clips/posting_log.csv.

YouTube keeps videos uploaded through a new, unaudited API project private, so by default
clips are uploaded as private and you set the publish time in Studio (the slot is in
the log). If your API project has passed YouTube's audit, --schedule makes YouTube
publish each clip at its slot by itself.

Setup (once): create an OAuth client ("Desktop app") for the YouTube Data API v3 in Google
Cloud Console and save its JSON as secrets/client_secret.json in this folder. The first
run opens a browser to sign in to your channel; the token is saved next to it. secrets/
is git-ignored, so neither file is ever committed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

# OAuth client secret and token live in clipper/secrets/ (git-ignored, never committed)
CONFIG = Path(__file__).resolve().parent / "secrets"
SECRET = CONFIG / "client_secret.json"
TOKEN = CONFIG / "youtube_token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]   # the second is needed to set thumbnails
LOG_FIELDS = None   # taken from clipper so both stay in sync
# same as clipper.SOCIAL_HEADER_RE (clipper is imported lazily, it loads slowly)
SOCIAL_HEADER_RE = r"^(?:TikTok / Reels|TikTok):$"


def log(*a):
    print("[upload]", *a, file=sys.stderr)


def read_caption(txt: Path) -> dict:
    """Parse a clipper .txt: title, description, hashtags line, 'Tags:' line."""
    blocks = [b.strip() for b in txt.read_text().split("\n\n") if b.strip()]
    title = blocks[0] if blocks else txt.stem
    tags, hashtags, desc = [], [], []
    for b in blocks[1:]:
        if re.match(SOCIAL_HEADER_RE, b.split("\n", 1)[0]):
            continue   # the TikTok / Reels caption isn't part of the YouTube post
        elif b.startswith("Tags:"):
            tags = [t.strip() for t in b[5:].split(",") if t.strip()]
        elif all(w.startswith("#") for w in b.split()):
            hashtags = b.split()
        else:
            desc.append(b)
    return dict(title=title, description="\n\n".join(desc), hashtags=hashtags, tags=tags)


def clip_entries(folder: Path):
    """(video file, thumbnail, caption) for each clip in a clipper output folder, in order."""
    js = json.loads((folder / "clips.json").read_text())
    out = []
    for cl in js["clips"]:
        for f in cl["files"]:
            video = folder / f
            stem = f.replace(" (landscape).mp4", "").removesuffix(".mp4")
            txt = folder / f"{stem}.txt"
            thumb = video.with_name(video.stem + " thumbnail.jpg")
            if video.exists() and txt.exists():
                out.append(dict(folder=folder.name, file=f, video=video, txt=txt,
                                thumb=thumb if thumb.exists() else None))
    return out


def interleave(groups):
    """Spread each folder's clips evenly over the whole plan, so one source video's clips
    are never bunched together even when it has many more clips than the others."""
    total = sum(len(g) for g in groups)
    placed = []
    for gi, g in enumerate(groups):
        for i, e in enumerate(g):
            placed.append(((i + 0.5) * total / len(g), gi, e))
    return [e for _, _, e in sorted(placed, key=lambda x: (x[0], x[1]))]

def read_log(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def day_quota(per_day: str):
    """'3' -> always 3 a day; '2-3' -> alternate 3 and 2 a day."""
    lo, _, hi = str(per_day).partition("-")
    lo, hi = int(lo), int(hi or lo)
    return lambda k: hi if k % 2 == 0 else lo


def next_slots(n, rows, times, per_day, now=None, min_gap_h=2.0):
    """The next n posting slots: `times` each day (per_day of them, e.g. '3' or '2-3').
    Posts already in the log (published or scheduled) count toward their day's total,
    and no new slot is closer than min_gap_h hours to one of them."""
    quota = day_quota(per_day)
    now = now or dt.datetime.now().astimezone()
    taken = []
    for r in rows:
        try:
            taken.append(dt.datetime.fromisoformat(r["posted_at"]).astimezone())
        except (KeyError, ValueError, TypeError):
            pass
    start = now + dt.timedelta(hours=1)
    count = {}
    for t in taken:
        count[t.date()] = count.get(t.date(), 0) + 1
    gap = dt.timedelta(hours=min_gap_h)
    slots, day, k = [], start.date(), 0
    while len(slots) < n:
        q = quota(k)
        # spread the day's posts over the listed times: 2 a day -> first and last
        day_times = times if q != 2 or len(times) <= 2 else [times[0], times[-1]]
        if count.get(day, 0) >= 2 and q != 2:   # a day with scheduled posts: try every time
            day_times = times
        for hhmm in day_times:
            if count.get(day, 0) >= q or len(slots) == n:
                break
            h, m = map(int, hhmm.split(":"))
            t = dt.datetime.combine(day, dt.time(h, m)).astimezone()
            if t < start or any(abs(t - x) < gap for x in taken + slots):
                continue
            slots.append(t)
            count[day] = count.get(day, 0) + 1
        day += dt.timedelta(days=1)
        k += 1
    return slots

def youtube_client():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not SECRET.exists():
                sys.exit(f"Missing {SECRET}. Create an OAuth client (Desktop app) for the YouTube "
                         "Data API v3 in Google Cloud Console and save its JSON there. See README.")
            creds = InstalledAppFlow.from_client_secrets_file(str(SECRET), SCOPES).run_local_server(port=0)
        CONFIG.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(creds.to_json())
        TOKEN.chmod(0o600)
    return build("youtube", "v3", credentials=creds)


def upload_one(yt, e, cap, slot, args):
    from googleapiclient.http import MediaFileUpload
    desc = cap["description"]
    if cap["hashtags"]:
        desc += "\n\n" + " ".join(cap["hashtags"])
    status = {"privacyStatus": args.privacy, "selfDeclaredMadeForKids": False}
    if args.schedule:
        status = {"privacyStatus": "private", "publishAt": slot.astimezone(dt.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"), "selfDeclaredMadeForKids": False}
    body = {"snippet": {"title": cap["title"][:100], "description": desc[:5000],
                        "tags": cap["tags"], "categoryId": args.category},
            "status": status}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(e["video"]), chunksize=8 << 20,
                                                        resumable=True, mimetype="video/mp4"))
    resp = None
    while resp is None:
        prog, resp = req.next_chunk()
        if prog:
            log(f"  {int(prog.progress() * 100)}%")
    vid = resp["id"]
    if e["thumb"]:
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(e["thumb"]),
                                                                        mimetype="image/jpeg")).execute()
        except Exception as ex:   # e.g. custom thumbnails need a phone-verified channel
            log(f"  thumbnail not set ({ex.__class__.__name__}): {str(ex)[:160]}")
    return vid


def thumb_data_uri(path: Path, width=360) -> str:
    """A small JPEG copy of the thumbnail inside the page, so the preview works anywhere."""
    import base64
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, width * 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_preview(out: Path, entries, slots, args):
    """A page laid out like the posting plan: one row per day, a card per Short."""
    import html
    days = {}
    for e, slot in zip(entries, slots):
        days.setdefault(slot.date(), []).append((e, slot, read_caption(e["txt"])))
    esc = html.escape
    cards = []
    for day, items in days.items():
        row = []
        for e, slot, cap in items:
            img = (f'<img src="{thumb_data_uri(e["thumb"])}" alt="">'
                   if e["thumb"] else '<div class="noimg">no thumbnail</div>')
            desc = esc(cap["description"]).replace("\n", "<br>")
            row.append(f"""<article><div class="time">{slot:%-I:%M %p}</div>{img}
<h3>{esc(cap['title'])}</h3><p class="desc">{desc}</p>
<p class="tags">{esc(' '.join(cap['hashtags']))}</p>
<p class="meta">{len(cap['tags'])} tags · {esc(e['folder'][:60])}</p></article>""")
        cards.append(f'<section><h2>{day:%A, %B %-d}</h2><div class="row">{"".join(row)}</div></section>')
    mode = "scheduled by YouTube" if args.schedule else f"uploaded {args.privacy}; set times in Studio"
    out.write_text(f"""<!doctype html><meta charset="utf-8"><title>Upload plan</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font:15px/1.4 -apple-system,system-ui,sans-serif;background:#0f0f0f;color:#f1f1f1;margin:0;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#aaa;margin:0 0 24px}}
h2{{font-size:16px;color:#aaa;font-weight:600;margin:28px 0 12px}}
.row{{display:flex;gap:16px;overflow-x:auto;padding-bottom:8px}}
article{{flex:0 0 220px;background:#1f1f1f;border-radius:12px;padding:10px}}
img,.noimg{{width:100%;aspect-ratio:9/16;object-fit:cover;border-radius:8px;background:#000;display:block}}
.noimg{{display:flex;align-items:center;justify-content:center;color:#777}}
.time{{font-weight:700;color:#3ea6ff;margin:0 0 8px}}
h3{{font-size:15px;margin:10px 0 6px}} .desc{{color:#ddd;font-size:13px;margin:0 0 6px;word-break:break-word}}
.tags{{color:#3ea6ff;font-size:13px;margin:0 0 6px}} .meta{{color:#888;font-size:12px;margin:0}}
</style><h1>Upload plan: {len(entries)} Shorts over {len(days)} days</h1>
<p class="sub">{esc(mode)}</p>{"".join(cards)}""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="clipper output folders (clips/<video>)")
    ap.add_argument("--per-day", default="3", help="clips per day: 3, or a range like 2-3 "
                    "(alternates 3 and 2)")
    ap.add_argument("--times", default="13:00,16:00,19:00", help="local posting times, comma-separated")
    ap.add_argument("--limit", type=int, default=6,
                    help="upload at most this many now (default 6, about YouTube's daily API quota)")
    ap.add_argument("--privacy", choices=["private", "unlisted"], default="private")
    ap.add_argument("--schedule", action="store_true",
                    help="have YouTube publish at each slot (needs an audited API project)")
    ap.add_argument("--category", default="20", help="YouTube category id (20 Gaming, 24 Entertainment)")
    ap.add_argument("--log", default=None, help="posting log (default: clips/posting_log.csv)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, upload nothing")
    ap.add_argument("--exclude", action="append", default=[], metavar="FOLDER/FILE",
                    help="don't upload this clip (its folder name / video file name); repeatable")
    ap.add_argument("--preview", nargs="?", const="", default=None,
                    help="also write an HTML preview of the schedule (thumbnails, titles, "
                         "descriptions); default clips/upload_preview.html")
    args = ap.parse_args()

    folders = [Path(f) for f in args.folders]
    log_path = Path(args.log) if args.log else folders[0].parent / "posting_log.csv"
    rows = read_log(log_path)
    done = {(r["video"], r["file"]) for r in rows if r.get("youtube_url")}
    done |= {tuple(x.split("/", 1)) for x in args.exclude if "/" in x}
    entries = interleave([[e for e in clip_entries(f) if (e["folder"], e["file"]) not in done]
                          for f in folders])[:args.limit]
    if not entries:
        log("nothing to upload (everything here already has a youtube_url in the log)")
        return
    times = [t.strip() for t in args.times.split(",") if t.strip()]
    slots = next_slots(len(entries), rows, times, args.per_day)

    for e, slot in zip(entries, slots):
        cap = read_caption(e["txt"])
        print(f"{slot:%a %b %d %H:%M}  {cap['title']}  [{e['folder'][:30]}]"
              f"{'' if e['thumb'] else '  (no thumbnail)'}")
    if args.preview is not None:
        out = Path(args.preview) if args.preview else log_path.parent / "upload_preview.html"
        write_preview(out, entries, slots, args)
        log(f"preview: {out}")
    if args.dry_run:
        log("dry run: nothing uploaded")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clipper
    yt = youtube_client()
    for e, slot in zip(entries, slots):
        cap = read_caption(e["txt"])
        log(f"uploading {e['file']}")
        try:
            vid = upload_one(yt, e, cap, slot, args)
        except Exception as ex:
            if "quotaExceeded" in str(ex) or "uploadLimitExceeded" in str(ex):
                log("YouTube's daily limit reached; run again tomorrow to continue")
                break
            raise
        url = f"https://youtube.com/shorts/{vid}"
        rows = read_log(log_path)
        hit = False
        for r in rows:
            if r["video"] == e["folder"] and r["file"] == e["file"]:
                r["posted_at"], r["youtube_url"], hit = slot.isoformat(timespec="minutes"), url, True
                if not args.schedule:
                    r["notes"] = (r.get("notes", "") + "; " if r.get("notes") else "") + \
                                 "uploaded private: set this publish time in Studio"
        if not hit:
            rows.append({**{k: "" for k in clipper.LOG_FIELDS}, "video": e["folder"], "file": e["file"],
                         "title": cap["title"], "posted_at": slot.isoformat(timespec="minutes"),
                         "youtube_url": url})
        with log_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=clipper.LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        log(f"  done: {url} (slot {slot:%a %H:%M})")


if __name__ == "__main__":
    main()
