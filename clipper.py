#!/usr/bin/env python3
"""
clipper — turn a long video into vertical short clips.

Inspired by ClipsAI, rebuilt to fix its weak spots:
  * Ranks moments (loudness spikes, chat spikes, hooks, payoff, dead air)
    instead of cutting at every topic change, so it works for streams.
  * Reads chat replays (YouTube live chat, Twitch) — when chat explodes,
    something happened. Audio + chat agreeing scores highest.
  * You choose how many clips and how long they are.
  * Fast on long streams: it scores the whole video first, then transcribes
    only the candidate regions (no 3-hour Whisper pass).
  * Picks a layout per clip: follow one face (smooth "camera operator"),
    split two speakers top/bottom, facecam-over-gameplay, or blurred background.
  * Renders finished 1080x1920 MP4s plus a clips.json report.
  * Optional: let Claude re-rank candidates from their transcripts (--llm).

Usage:
  python clipper.py VIDEO_OR_URL [-o out] [-n 5] [--min 15] [--max 60] [--chat FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SR = 16000          # analysis sample rate
HOP = 0.1           # loudness hop (s)
RES = 0.5           # scoring resolution (s)
LAYOUTS = ["auto", "track", "split", "screen", "blur"]
SCREEN_MOTION = 4.0  # mean pixel change (0-255) outside the facecam that counts as gameplay


def log(*a):
    print("[clipper]", *a, file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def even(x):
    return int(round(x / 2) * 2)


# --------------------------------------------------------------------------- ingest
def ingest(src: str, workdir: Path, want_chat: bool):
    """Return (video_path, chat_path_or_None, metadata with title / creator / date)."""
    if re.match(r"^https?://", src):
        if not shutil.which("yt-dlp"):
            sys.exit("yt-dlp is not installed: pip install yt-dlp")
        log("downloading", src)
        # one file per video id, so different URLs never reuse each other's download
        out = workdir / "%(id)s.%(ext)s"
        res = subprocess.run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
                              "--merge-output-format", "mp4", "--write-info-json",
                              "--no-simulate", "--print", "after_move:filepath",
                              "-o", str(out), src], stdout=subprocess.PIPE, text=True)
        lines = [l for l in res.stdout.splitlines() if l.strip()]
        if res.returncode or not lines or not Path(lines[-1]).exists():
            sys.exit("download failed")
        vid = Path(lines[-1])
        meta = {}
        info = vid.with_suffix(".info.json")
        if info.exists():
            j = json.loads(info.read_text())
            d = j.get("upload_date") or ""
            meta = dict(title=j.get("title"), creator=j.get("channel") or j.get("uploader"),
                        date=f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else None,
                        url=j.get("webpage_url") or src, id=j.get("id"))
        chat = None
        if want_chat and "youtu" in src:
            log("fetching chat replay (if the video has one)")
            subprocess.run(["yt-dlp", "--skip-download", "--write-subs", "--sub-langs", "live_chat",
                            "-o", str(out), src],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            found = sorted(workdir.glob(f"{vid.stem}*.live_chat.json"))
            chat = found[0] if found else None
        return vid, chat, meta
    p = Path(src).expanduser().resolve()
    if not p.exists():
        sys.exit(f"not found: {p}")
    chat = None
    if want_chat:  # pick up a chat file saved next to the video
        for cand in (p.with_suffix(".live_chat.json"), p.with_suffix(".chat.json"),
                     p.with_suffix(".json"), p.with_suffix(".chat.txt")):
            if cand.exists():
                chat = cand
                break
    return p, chat, local_meta(p)


def local_meta(p: Path) -> dict:
    """Title / creator / date for a local file, from its tags or the file itself."""
    from datetime import date
    tags = {}
    try:
        tags = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(p)],
            capture_output=True, text=True).stdout).get("format", {}).get("tags", {})
        tags = {k.lower(): v for k, v in tags.items()}
    except Exception:
        pass
    created = (tags.get("creation_time") or "")[:10] or \
        date.fromtimestamp(p.stat().st_mtime).isoformat()
    return dict(title=tags.get("title") or p.stem,
                creator=tags.get("artist") or tags.get("author"), date=created)


def folder_name(meta: dict) -> str:
    """'<title> - <creator> - <YYYY-MM-DD>', safe for macOS/Windows/Linux paths."""
    parts = [meta.get("title"), meta.get("creator"), meta.get("date")]
    name = " - ".join(str(x).strip() for x in parts if x and str(x).strip())
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:150].rstrip(" .") or "clips"


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in j["streams"])
    if v is None:
        sys.exit("no video stream found")
    num, den = v["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    if not (5 < fps < 121):
        fps = 30.0
    return dict(duration=float(j["format"]["duration"]), width=int(v["width"]),
                height=int(v["height"]), fps=fps, has_audio=has_audio)


# --------------------------------------------------------------------------- shared stats
def rolling_stat(x: np.ndarray, win: int, fn) -> np.ndarray:
    """Cheap rolling statistic: compute on blocks, then interpolate."""
    blk = max(1, win // 12)
    nb = max(1, len(x) // blk)
    centers, vals = [], []
    for i in range(nb):
        lo = max(0, i * blk - win // 2)
        hi = min(len(x), i * blk + win // 2)
        centers.append(i * blk + blk / 2)
        vals.append(fn(x[lo:hi]))
    return np.interp(np.arange(len(x)), centers, vals)


def mad(s):
    return np.median(np.abs(s - np.median(s)))


# --------------------------------------------------------------------------- audio
def loudness_curve(path: Path) -> np.ndarray:
    """RMS loudness in dBFS every HOP seconds, streamed (works on 5h VODs)."""
    hop = int(SR * HOP)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    out, buf = [], b""
    while True:
        chunk = p.stdout.read(hop * 2 * 600)
        if not chunk:
            break
        buf += chunk
        n = len(buf) // (hop * 2)
        if n:
            a = np.frombuffer(buf[: n * hop * 2], dtype=np.int16).astype(np.float32) / 32768
            rms = np.sqrt((a.reshape(n, hop) ** 2).mean(1))
            out.append(20 * np.log10(rms + 1e-6))
            buf = buf[n * hop * 2:]
    p.wait()
    return np.concatenate(out) if out else np.zeros(1)


def audio_features(db: np.ndarray):
    """Return excitement (z-score above local baseline) and silence mask at RES."""
    base = rolling_stat(db, int(60 / HOP), np.median)
    spread = rolling_stat(db, int(60 / HOP), mad)
    z = (db - base) / np.maximum(spread * 1.4826, 2.0)
    k = int(RES / HOP)
    n = len(db) // k
    zr = z[: n * k].reshape(n, k)
    dbr = db[: n * k].reshape(n, k)
    excite = np.clip(zr.max(1), 0, 8)
    excite = np.convolve(excite, np.ones(3) / 3, mode="same")
    silent = (dbr.mean(1) < np.maximum(base[: n * k].reshape(n, k).mean(1) - 12, -60)) | (dbr.mean(1) < -50)
    return excite, silent


# --------------------------------------------------------------------------- chat
HYPE = re.compile(
    r"(pog|kekw|lul|lmf?ao+|lo+l|omg|omfg|wtf|holy|no ?way|clip ?(it|that)|\bw+\b|\bw (stream|speed|rizz)|"
    r"insane|crazy|bro+|ayo+|sheesh|dead|😂|🤣|💀|😭|🔥|‼|!!!)", re.I)


def load_chat(path: Path):
    """Return list of (seconds, text) from common chat-replay formats."""
    msgs = []
    raw = path.read_text(encoding="utf-8", errors="ignore")

    def yt_runs(r):
        out = []
        for run_ in r.get("message", {}).get("runs", []):
            if "text" in run_:
                out.append(run_["text"])
            elif "emoji" in run_:
                sc = run_["emoji"].get("shortcuts") or [run_["emoji"].get("emojiId", "")]
                out.append(sc[0])
        return "".join(out)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and "comments" in data:  # TwitchDownloader JSON
        for c in data["comments"]:
            t = c.get("content_offset_seconds")
            body = (c.get("message") or {}).get("body", "")
            if t is not None:
                msgs.append((float(t), body))
    elif isinstance(data, list):  # chat-downloader / generic list
        for c in data:
            t = c.get("time_in_seconds", c.get("content_offset_seconds", c.get("t")))
            body = c.get("message", c.get("text", ""))
            if isinstance(body, dict):
                body = body.get("body", "")
            if t is not None:
                msgs.append((float(t), str(body)))
    elif raw.lstrip().startswith("{"):  # yt-dlp live_chat.json (JSON lines)
        for line in raw.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = obj.get("replayChatItemAction", obj)
            off = r.get("videoOffsetTimeMsec", obj.get("videoOffsetTimeMsec"))
            if off is None:
                continue
            for act in r.get("actions", []):
                item = act.get("addChatItemAction", {}).get("item", {})
                rend = (item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer"))
                if rend:
                    msgs.append((int(off) / 1000, yt_runs(rend)))
    else:  # text log: "[1:02:03] user: message" or "01:02:03 message"
        pat = re.compile(r"^\s*\[?((?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)\]?\s*(.*)$")
        for line in raw.splitlines():
            m = pat.match(line)
            if m:
                parts = [float(x) for x in m.group(1).split(":")]
                t = sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts))
                msgs.append((t, m.group(2)))
    return msgs


def chat_features(msgs, n_bins, delay):
    """Chat excitement per RES bin: message-rate + hype spikes vs. a rolling baseline,
    shifted back by `delay` seconds because chat reacts after the moment."""
    w = np.zeros(n_bins)
    for t, text in msgs:
        i = int((t - delay) / RES)
        if 0 <= i < n_bins:
            hype = 1.0 if HYPE.search(text or "") else 0.0
            letters = [ch for ch in (text or "") if ch.isalpha()]
            if len(letters) >= 4 and sum(ch.isupper() for ch in letters) / len(letters) > 0.7:
                hype += 0.5
            w[i] += 1.0 + hype
    w = np.convolve(w, np.ones(4) / 4, mode="same") * 4 / (4 * RES)  # weighted msgs per second
    win = int(120 / RES)
    base = rolling_stat(w, win, np.median)
    spread = rolling_stat(w, win, mad)
    z = (w - base) / np.maximum(spread * 1.4826, np.sqrt(base + 0.25))
    return np.clip(z, 0, 8)


# --------------------------------------------------------------------------- scoring
@dataclass
class Clip:
    start: float
    end: float
    score: float
    parts: dict = field(default_factory=dict)
    text: str = ""
    llm: dict | None = None
    layout: str = ""
    file: str = ""
    files: list = field(default_factory=list)
    caption: dict | None = None


def grow_moments(excite, silent, min_len, max_len, k_seeds=60, gap=5.0, lead=2.0, tail=1.5):
    """Find clips by growing outward from excitement peaks, so each moment picks its own length.

    A clip keeps extending while exciting stretches keep coming back within `gap` seconds
    (a pause between laughs doesn't end it) and stops once things stay ordinary. Backward it
    takes in the build-up. Scoring credits every second above the video's ordinary level and
    charges for flat stretches, so a long clip only wins if it stays interesting.
    """
    n = len(excite)
    valid = ~(silent & (excite == 0))          # outside --start/--end is zeroed and silent
    vidx = np.flatnonzero(valid)
    if len(vidx) == 0:
        return []
    lo, hi = int(vidx[0]), int(vidx[-1]) + 1
    sm = max(1, int(3 / RES))
    heat = np.convolve(excite, np.ones(sm) / sm, mode="same")
    ref = heat[valid]
    # thresholds are relative to this video: a calm podcast and a screaming stream both work
    base = float(np.median(ref))               # an ordinary second
    hot_t = float(np.percentile(ref, 72))      # clearly above ordinary
    top = float(np.percentile(ref, 99.5))
    hot = (heat >= hot_t) & valid
    g = int(gap / RES)
    minb, maxb = int(min_len / RES), int(max_len / RES)

    seeds = []                                  # highest local peaks, at least 10 s apart
    for i in np.argsort(-heat):
        if not hot[i]:
            break
        if all(abs(i - s) > int(10 / RES) for s in seeds):
            seeds.append(int(i))
            if len(seeds) >= k_seeds:
                break

    cands = []
    for s in seeds:
        e, cold, j = s, 0, s                    # forward: ride the moment until it goes flat
        while j < hi - 1 and (j - s) < maxb:
            j += 1
            if hot[j]:
                e, cold = j, 0
            else:
                cold += 1
                if cold >= g:
                    break
        b, cold, j = s, 0, s                    # backward: include the build-up
        while j > lo and (s - j) < maxb:
            j -= 1
            if hot[j]:
                b, cold = j, 0
            else:
                cold += 1
                if cold >= g // 2 + 1:
                    break
        a0 = max(lo, b - int(lead / RES))
        a1 = min(hi, e + 1 + int(tail / RES))
        if a1 - a0 < minb:                      # too short: pad, mostly after (reactions come after)
            a0 = max(lo, a0 - (minb - (a1 - a0)) // 3)
            a1 = min(hi, a0 + minb)
            a0 = max(lo, a1 - minb)
        if a1 - a0 > maxb:                      # too long: keep the strongest max-length stretch
            cs = np.concatenate([[0], np.cumsum(np.clip(heat[a0:a1] - base, 0, None))])
            o = int(np.argmax(cs[maxb:] - cs[:-maxb]))
            a0, a1 = a0 + o, a0 + o + maxb
        seg = heat[a0:a1]
        cands.append(Clip(a0 * RES, a1 * RES, 0.0, dict(
            peak=round(min(float(seg.max() - base) / max(top - base, 1e-6), 1.0), 3),
            excess=round(float(np.clip(seg - base, 0, None).sum() * RES), 2),
            dull=round(float((seg < base).mean()), 3),
            hook=round(min(float(np.clip(heat[a0:a0 + int(3 / RES)] - base, 0, None).mean())
                           / max(hot_t - base, 1e-6), 1.0), 3),
            dead_air=round(float(silent[a0:a1].mean()), 3))))
    if not cands:
        return []
    eref = float(np.percentile([c.parts["excess"] for c in cands], 90)) or 1.0
    for c in cands:
        p = c.parts
        c.score = (0.35 * p["peak"] + 0.4 * min(p["excess"] / eref, 1.2) + 0.1 * p["hook"]
                   - 0.3 * p["dull"] - 0.35 * p["dead_air"])
    return cands


def pick(cands, k, gap=5.0):
    chosen = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(c.end + gap <= o.start or c.start >= o.end + gap for o in chosen):
            chosen.append(c)
            if len(chosen) >= k:
                break
    return chosen


# --------------------------------------------------------------------------- transcription
_whisper = None


def whisper_model(name):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        log(f"loading Whisper '{name}' (first run downloads it)")
        _whisper = WhisperModel(name, device="auto", compute_type="int8")
    return _whisper


def transcribe_range(path: Path, start: float, end: float, model: str, lang, tmp: Path):
    """Transcribe only [start, end]; returns segments with absolute times."""
    # Decode with ffmpeg and hand Whisper the samples directly. Passing a file path makes
    # faster-whisper decode it with PyAV, which breaks whenever PyAV changes its API.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
                          "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    segs, _ = whisper_model(model).transcribe(audio, language=lang, vad_filter=True,
                                              word_timestamps=True)
    out = []
    for s in segs:
        words = [(w.start + start, w.end + start, w.word) for w in (s.words or [])]
        out.append(dict(start=s.start + start, end=s.end + start, text=s.text.strip(), words=words))
    return out


def snap(clip: Clip, segs, silent, min_len, max_len, duration):
    """Move clip edges to sentence boundaries (or silences) so nothing is cut mid-word."""
    s, e = clip.start, clip.end
    if segs:
        starts = [g["start"] for g in segs if s - 4 <= g["start"] <= s + 2.5]
        if starts:
            s = min(starts, key=lambda t: abs(t - s))
        ends = [g["end"] for g in segs if e - 3 <= g["end"] <= e + 4 and min_len <= g["end"] - s <= max_len]
        if ends:
            e = min(ends, key=lambda t: abs(t - e))
        else:
            for g in segs:  # never end inside a word
                for ws, we, _ in g["words"]:
                    if ws < e < we:
                        e = we if we - s <= max_len else ws
        s, e = s - 0.25, e + 0.45
        clip.text = " ".join(g["text"] for g in segs if g["end"] > s and g["start"] < e)
    else:
        def nearest_quiet(t):
            i = int(t / RES)
            for d in range(0, 4):
                for j in (i - d, i + d):
                    if 0 <= j < len(silent) and silent[j]:
                        return j * RES
            return t
        s, e = nearest_quiet(s), nearest_quiet(e)
    s = max(0.0, s)
    e = min(duration, max(e, s + min_len))
    if e - s > max_len:
        e = s + max_len
    clip.start, clip.end = round(s, 2), round(e, 2)
    return clip


# --------------------------------------------------------------------------- post captions
CAPTION_MODEL = "claude-opus-5-5"
STOP = set("a an and are as at be but by for from has have i if in is it its just like me my "
           "of on or our so that the their them then there this to up was we what when with "
           "you your yeah okay oh um uh gonna got get go these those where which while would "
           "could should about actually really thing things stuff because doing going know "
           "think right some been being here they into only even still every other something "
           "anyway always never maybe pretty kind sort little much very well call".split())


def _tag(s: str) -> str:
    return "#" + re.sub(r"[^0-9A-Za-z]", "", s).lower()


def basic_caption(clip: Clip, meta: dict) -> dict:
    """Offline caption: the clip's opening line as a hook plus simple hashtags."""
    text = re.sub(r"\s+", " ", clip.text).strip()
    # the clip often opens mid-sentence, so use the first complete-looking sentence
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    full = [s for s in sents if len(s.split()) >= 5 and s[0].isupper()]
    hook = (full or sents or [""])[0]
    if len(hook) > 100:
        hook = hook[:97].rsplit(" ", 1)[0] + "..."
    # title: the clip's punchiest short line (exclamations and questions first), so every
    # clip gets its own title; fall back to the video title when there's no transcript
    short = [s for s in sents if 3 <= len(s.split()) <= 9 and len(s) <= 60 and s[0].isalpha()]
    short.sort(key=lambda s: (s[-1] not in "!?", -len(s.split())))
    if short:
        title = short[0].rstrip(".")
        title = title[0].upper() + title[1:]
    elif hook:  # one long run-on line: use its opening words, not ending on a filler word
        words = hook.rstrip(".").split()[:7]
        while len(words) > 3 and words[-1].lower().strip(",") in STOP:
            words.pop()
        title = " ".join(words).rstrip(",")
        title = title[0].upper() + title[1:]
    else:
        title = re.sub(r"\s*[|•-]\s*.*$", "", meta.get("title") or "").strip() or hook[:60]
    if hook.rstrip(".") == title and len(full) > 1:
        hook = full[1] if len(full[1]) <= 100 else full[1][:97].rsplit(" ", 1)[0] + "..."
    # hashtags: words from the title, then words the clip keeps repeating
    tw = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", meta.get("title") or "")
          if w.lower() not in STOP]
    cw = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", text) if w.lower() not in STOP]
    rep = sorted({w for w in cw if cw.count(w) >= 2}, key=lambda w: -cw.count(w))
    tags = ["#shorts"]
    if meta.get("creator"):
        tags.append(_tag(meta["creator"]))
    for w in (rep + tw)[:3]:
        tags.append(_tag(w))
    return dict(title=title, caption=hook or title, hashtags=list(dict.fromkeys(tags)),
                source="basic")


def claude_captions(clips, meta: dict):
    """One Claude call for all clips. Returns a list of dicts, or None if Claude isn't usable."""
    try:
        import anthropic
    except ImportError:
        log("captions: 'anthropic' not installed (pip install anthropic); using basic captions")
        return None
    items = [dict(id=i, seconds=round(c.end - c.start), transcript=c.text[:1500])
             for i, c in enumerate(clips)]
    prompt = (
        "These are clips cut from one video to post as YouTube Shorts / TikToks / Reels.\n"
        f"Video title: {meta.get('title') or 'unknown'}\n"
        f"Creator: {meta.get('creator') or 'unknown'}\n\n"
        "For each clip write a post caption based on what actually happens in its transcript:\n"
        "- title: a short, punchy title, at most 60 characters\n"
        "- caption: one or two sentences that make people want to watch, at most 150 "
        "characters, in the creator's tone; no hashtags in this field\n"
        "- hashtags: 3 to 6 relevant hashtags, each starting with #\n"
        "Don't invent events that aren't in the transcript.\n\n" + json.dumps(items))
    schema = {
        "type": "object",
        "properties": {"captions": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "title": {"type": "string"},
                           "caption": {"type": "string"},
                           "hashtags": {"type": "array", "items": {"type": "string"}}},
            "required": ["id", "title", "caption", "hashtags"],
            "additionalProperties": False}}},
        "required": ["captions"],
        "additionalProperties": False}
    try:
        resp = anthropic.Anthropic().beta.messages.create(
            model=CAPTION_MODEL, max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}])
    except (anthropic.AuthenticationError, TypeError):
        # TypeError: the SDK found no API key / auth token / profile at all
        log("captions: no Claude credentials (set ANTHROPIC_API_KEY); using basic captions")
        return None
    except anthropic.APIError as ex:
        log("captions: Claude request failed; using basic captions:", ex)
        return None
    if resp.stop_reason != "end_turn":
        log(f"captions: Claude stopped early ({resp.stop_reason}); using basic captions")
        return None
    try:
        text = next(b.text for b in resp.content if b.type == "text")
        by_id = {d["id"]: d for d in json.loads(text)["captions"]}
    except (StopIteration, ValueError, KeyError, TypeError):
        log("captions: couldn't read Claude's reply; using basic captions")
        return None
    out = []
    for i, c in enumerate(clips):
        d = by_id.get(i)
        if d is None:
            out.append(basic_caption(c, meta))
            continue
        tags = [t if t.startswith("#") else "#" + t for t in d["hashtags"]]
        out.append(dict(title=d["title"], caption=d["caption"], hashtags=tags, source="claude"))
    return out


def make_captions(clips, meta: dict, use_claude: bool):
    """Attach a caption (title, text, hashtags) to every clip."""
    caps = claude_captions(clips, meta) if use_claude else None
    if caps is None:
        caps = [basic_caption(c, meta) for c in clips]
    for c, cap in zip(clips, caps):
        c.caption = cap
    log(f"captions: {len(clips)} written ({caps[0]['source'] if caps else 'none'})")


def clip_stem(k: int, c: Clip) -> str:
    """'01 - <caption title>' when the clip has a caption, else 'clip_01_12m34s'."""
    title = (c.caption or {}).get("title") or ""
    title = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title)
    title = re.sub(r"\s+", " ", title).strip(" .#")
    if not title:
        return f"clip_{k:02d}_{int(c.start // 60):02d}m{int(c.start % 60):02d}s"
    if len(title) > 80:
        title = title[:80].rsplit(" ", 1)[0].rstrip(" .,")
    return f"{k:02d} - {title}"


def write_caption_file(path: Path, cap: dict):
    path.write_text(f"{cap['title']}\n\n{cap['caption']}\n\n{' '.join(cap['hashtags'])}\n")


# --------------------------------------------------------------------------- LLM re-rank
def llm_rerank(clips, model="claude-sonnet-5-5"):
    try:
        import anthropic
    except ImportError:
        log("anthropic not installed; skipping --llm")
        return clips
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("ANTHROPIC_API_KEY not set; skipping --llm")
        return clips
    items = [dict(id=i, seconds=round(c.end - c.start), transcript=c.text[:1500])
             for i, c in enumerate(clips)]
    prompt = (
        "You pick moments from a long video to post as YouTube Shorts / TikToks. "
        "For each candidate, rate 0-10 how well it works as a standalone short: "
        "strong first 3 seconds, a clear payoff, makes sense without context, "
        "emotion/humour/surprise. Reply with JSON only: "
        '[{"id":0,"score":7,"reason":"..."}].\n\n' + json.dumps(items))
    try:
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=2000, messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text
        data = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
    except Exception as ex:  # keep going with heuristic scores
        log("LLM re-rank failed:", ex)
        return clips
    hs = np.array([c.score for c in clips])
    hn = (hs - hs.min()) / max(1e-6, hs.max() - hs.min())
    by_id = {d["id"]: d for d in data if isinstance(d, dict) and "id" in d}
    for i, c in enumerate(clips):
        d = by_id.get(i)
        if d:
            c.llm = dict(score=d.get("score"), reason=d.get("reason"))
            c.score = round(0.4 * float(hn[i]) + 0.6 * float(d.get("score", 5)) / 10, 4)
    return clips


# --------------------------------------------------------------------------- framing: analysis
def get_detector(model_dir: Path):
    import cv2
    path = model_dir / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        log("downloading face model")
        urllib.request.urlretrieve(YUNET_URL, path)
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 50)


def decode(path, start, dur, w=None, h=None):
    vf = ["-vf", f"scale={w}:{h}"] if w else []
    return subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
                             "-t", f"{dur:.3f}", *vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE, bufsize=10 ** 8)


def analyze_faces(path, clip, info, detector):
    """One pass over the clip at 640px: scene cuts every frame, faces ~6x/second.
    Faces are stored as rows [x, y, w, h, conf] in analysis pixels."""
    import cv2
    aw = 640
    ah = even(info["height"] * aw / info["width"])
    detector.setInputSize((aw, ah))
    fps = info["fps"]
    step = max(1, round(fps / 6))
    p = decode(path, clip.start, clip.end - clip.start, aw, ah)
    fsz = aw * ah * 3
    cuts, samples, prev_hist = [0], {}, None
    motion, prev_gray = [], None  # coarse motion grids, to tell gameplay from a static room
    i = 0
    while True:
        raw = p.stdout.read(fsz)
        if len(raw) < fsz:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(ah, aw, 3)
        small = cv2.resize(frame, (64, 36))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.5:
            cuts.append(i)
        prev_hist = hist
        if i % step == 0:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None and i not in cuts:
                motion.append(np.abs(gray - prev_gray))
            prev_gray = gray
            _, faces = detector.detect(frame)
            arr = np.zeros((0, 5))
            if faces is not None and len(faces):
                arr = faces[:, [0, 1, 2, 3, 14]].astype(np.float64)
                arr = arr[arr[:, 2] >= aw * 0.02]  # ignore tiny background faces
            samples[i] = arr
        i += 1
    p.wait()
    mgrid = np.mean(motion, axis=0) if motion else np.zeros((36, 64))
    return dict(aw=aw, ah=ah, fps=fps, n=i, cuts=sorted(set(cuts)), samples=samples, motion=mgrid)


def _largest(f):
    return f[np.argmax(f[:, 2] * f[:, 3])]


def detect_screen(A):
    """Gameplay/screen with a facecam: one small face that barely moves, off-centre."""
    aw, ah, S = A["aw"], A["ah"], list(A["samples"].values())
    pts = [_largest(f) for f in S if len(f)]
    pts = [p for p in pts if p[2] < aw * 0.12]
    if len(pts) < 0.6 * len(S) or len(pts) < 3:
        return None
    P = np.array(pts)
    cx, cy = P[:, 0] + P[:, 2] / 2, P[:, 1] + P[:, 3] / 2
    if cx.std() > aw * 0.03 or cy.std() > ah * 0.03:
        return None
    mx, my = np.median(cx), np.median(cy)
    if abs(mx - aw / 2) < aw * 0.2 and abs(my - ah / 2) < ah * 0.2:
        return None  # a small face in the middle is a wide shot, not a facecam
    # the rest of the frame must be moving (gameplay / screen), not a static room
    g = A["motion"].copy()
    gx, gy = 64 / aw, 36 / ah
    fw, fh = np.median(P[:, 2]), np.median(P[:, 3])
    x0, x1 = int((mx - 1.5 * fw) * gx), int((mx + 1.5 * fw) * gx) + 1
    y0, y1 = int((my - 1.5 * fh) * gy), int((my + 1.5 * fh) * gy) + 1
    g[max(0, y0):y1, max(0, x0):x1] = np.nan
    A["screen_motion"] = float(np.nanmean(g))
    if A["screen_motion"] < SCREEN_MOTION:
        return None
    return np.array([mx, my, np.median(P[:, 2]), np.median(P[:, 3])])


def detect_split(A):
    """Two people who can't both fit in one vertical crop (podcast two-shot)."""
    aw, ah, S = A["aw"], A["ah"], list(A["samples"].values())
    cw = ah * 9 / 16
    pairs = []
    for f in S:
        big = f[f[:, 2] >= aw * 0.04]
        if len(big) < 2:
            continue
        two = big[np.argsort(-(big[:, 2] * big[:, 3]))[:2]]
        two = two[np.argsort(two[:, 0])]
        c = two[:, 0] + two[:, 2] / 2
        if c[1] - c[0] > cw * 0.8:
            pairs.append(two)
    if len(pairs) < 0.5 * len(S) or len(pairs) < 3:
        return None
    P = np.array(pairs)  # (k, 2, 5)
    med = np.median(P, axis=0)
    return [np.array([m[0] + m[2] / 2, m[1] + m[3] / 2, m[2], m[3]]) for m in med]


def track_path(A):
    """Per-frame crop centre for a single subject, like a calm camera operator."""
    aw, ah, fps, n = A["aw"], A["ah"], A["fps"], A["n"]
    cw = ah * 9 / 16
    half = cw / 2
    cutset = set(A["cuts"])
    tgt = np.full(n, np.nan)
    prev = None
    for i in sorted(A["samples"]):
        if any(c <= i and c > i - max(1, round(fps / 6)) for c in cutset if c):
            prev = None  # new shot: free to choose a new subject
        best, bs = None, -1
        for x, y, w, h, conf in A["samples"][i]:
            cx = x + w / 2
            sc = w * h * conf
            if prev is not None and abs(cx - prev) < aw * 0.12:
                sc *= 2.0  # stickiness: don't hop between faces
            if sc > bs:
                bs, best = sc, cx
        if best is not None:
            tgt[i] = best
            prev = best
    cuts = A["cuts"] + [n]
    out = np.zeros(n)
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = tgt[a:b].copy()
        idx = np.where(~np.isnan(seg))[0]
        if len(idx) == 0:
            seg[:] = aw / 2
        else:
            # hold the last detection (no sliding across a jump), back-fill the start
            j = np.where(~np.isnan(seg), np.arange(len(seg)), 0)
            np.maximum.accumulate(j, out=j)
            seg = seg[j]
            seg[np.isnan(seg)] = seg[idx[0]]
        # camera operator: hold still inside a dead zone, glide when subject leaves it
        cam = np.empty_like(seg)
        c, moving = seg[0], False
        dz, vmax = cw * 0.18, cw * 1.5 / fps
        look = max(1, int(fps * 0.5))
        jumps = [0]
        for j, t in enumerate(seg):
            # subject jumped far and stays there (camera switch / new speaker): cut, don't whip-pan
            if abs(t - c) > cw * 0.8 and abs(np.mean(seg[j:j + look]) - c) > cw * 0.8:
                c, moving = t, False
                jumps.append(j)
            elif abs(t - c) > dz:
                moving = True
            if moving:
                c += np.clip((t - c) * 0.12, -vmax, vmax)
                if abs(t - c) < cw * 0.03:
                    moving = False
            cam[j] = c
        # look-ahead smoothing within each shot (never across a cut or jump)
        sig = max(1, int(fps * 0.2))
        ker = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
        ker /= ker.sum()
        jumps.append(len(cam))
        for u, v in zip(jumps[:-1], jumps[1:]):
            if v > u:
                pad = np.pad(cam[u:v], (3 * sig, 3 * sig), mode="edge")
                cam[u:v] = np.convolve(pad, ker, mode="valid")
        out[a:b] = np.clip(cam, half, aw - half)
    return out


def plan_layout(A, info, forced):
    """Decide how to frame a clip. Returns (layout, data, face_coverage)."""
    S = list(A["samples"].values())
    cov = sum(1 for f in S if len(f)) / max(1, len(S))
    sx = info["width"] / A["aw"]
    if forced in ("auto", "screen"):
        cam = detect_screen(A)
        if cam is not None:
            return "screen", cam * sx, cov
        if forced == "screen":
            pts = [_largest(f) for f in S if len(f)]
            if pts:
                m = np.median(np.array(pts), axis=0)
                return "screen", np.array([m[0] + m[2] / 2, m[1] + m[3] / 2, m[2], m[3]]) * sx, cov
            log("no facecam found; using blur layout")
            return "blur", None, cov
    if forced in ("auto", "split"):
        two = detect_split(A)
        if two is not None:
            return "split", [t * sx for t in two], cov
        if forced == "split":
            log("couldn't find two people; falling back to tracking")
    if forced in ("auto", "track", "split"):
        if cov >= 0.35 or forced in ("track", "split"):
            return "track", track_path(A) * sx, cov
    return "blur", None, cov


# --------------------------------------------------------------------------- framing: render
def enc_args(out, crf):
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]


def face_crop(face, aspect, info, scale, up=0.42):
    """A crop box (x, y, w, h) of the given aspect around a face (cx, cy, fw, fh)."""
    sw, sh = info["width"], info["height"]
    cx, cy, fw, fh = face
    h = min(sh, max(fh * scale, sh * 0.3))
    w = h * aspect
    if w > sw:
        w, h = sw, sw / aspect
    x = np.clip(cx - w / 2, 0, sw - w)
    y = np.clip(cy - h * up, 0, sh - h)
    return even(x), even(y), even(w), even(h)


def render_static(path, clip, info, panels, out, W, crf):
    """Stack fixed crops vertically. panels = [((x, y, w, h), panel_height), ...]"""
    k = len(panels)
    fc = f"[0:v]split={k}" + "".join(f"[s{i}]" for i in range(k)) + ";"
    for i, ((x, y, w, h), ph) in enumerate(panels):
        fc += f"[s{i}]crop={w}:{h}:{x}:{y},scale={W}:{ph}:force_original_aspect_ratio=increase,crop={W}:{ph}[p{i}];"
    fc += "".join(f"[p{i}]" for i in range(k)) + f"vstack={k},setsar=1[v]"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start:.3f}", "-i", str(path),
         "-t", f"{clip.end - clip.start:.3f}", "-filter_complex", fc, "-map", "[v]",
         *(["-map", "0:a"] if info["has_audio"] else []), *enc_args(out, crf)])


def render_split(path, clip, info, faces, out, W, H, crf):
    ph = even(H / 2)
    boxes = [face_crop(f, W / ph, info, scale=3.5) for f in faces]
    render_static(path, clip, info, [(boxes[0], ph), (boxes[1], H - ph)], out, W, crf)


def render_screen(path, clip, info, cam, out, W, H, crf):
    sw, sh = info["width"], info["height"]
    top = even(H * 0.35)
    bottom = H - top
    cam_box = face_crop(cam, W / top, info, scale=3.0, up=0.5)
    gh = sh
    gw = min(sw, gh * W / bottom)
    gx = even((sw - gw) / 2)
    render_static(path, clip, info, [(cam_box, top), ((gx, 0, even(gw), even(gh)), bottom)], out, W, crf)


def render_track(path, clip, info, centers, out, W, H, crf):
    import cv2
    sw, sh, fps = info["width"], info["height"], info["fps"]
    cw = even(sh * 9 / 16)
    dur = clip.end - clip.start
    dec = decode(path, clip.start, dur)
    audio = ["-ss", f"{clip.start:.3f}", "-t", f"{dur:.3f}", "-i", str(path)] if info["has_audio"] else []
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{fps:.5f}", "-i", "-", *audio,
                            "-map", "0:v", *(["-map", "1:a"] if audio else []), "-shortest",
                            *enc_args(out, crf)], stdin=subprocess.PIPE)
    fsz = sw * sh * 3
    i = 0
    while True:
        raw = dec.stdout.read(fsz)
        if len(raw) < fsz:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
        cx = centers[min(i, len(centers) - 1)]
        x0 = int(np.clip(round(cx - cw / 2), 0, sw - cw))
        enc.stdin.write(cv2.resize(frame[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA).tobytes())
        i += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()


def render_blur(path, clip, info, out, W, H, crf):
    fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H},gblur=sigma=40,eq=brightness=-0.12[bg];"
          f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start:.3f}", "-i", str(path),
         "-t", f"{clip.end - clip.start:.3f}", "-filter_complex", fc, "-map", "[v]",
         *(["-map", "0:a"] if info["has_audio"] else []), *enc_args(out, crf)])


def render_landscape(path, clip, info, out, W, H, crf):
    """Keep the whole frame (no crop): fit into WxH, letterbox if the aspect differs."""
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start:.3f}", "-i", str(path),
         "-t", f"{clip.end - clip.start:.3f}", "-vf", vf, "-map", "0:v:0",
         *(["-map", "0:a:0"] if info["has_audio"] else []), *enc_args(out, crf)])


def ask_format() -> str:
    """Ask which orientation to render. Only called when stdin is a terminal."""
    print("\nWhere is the funny part of these clips?\n"
          "  1) portrait   - one person/face; crop to vertical 9:16 (Shorts/TikTok/Reels)\n"
          "  2) landscape  - the joke needs the whole frame; keep 16:9, no crop\n"
          "  3) both       - render each clip both ways and pick later",
          file=sys.stderr)
    choices = {"1": "portrait", "2": "landscape", "3": "both",
               "p": "portrait", "l": "landscape", "b": "both"}
    while True:
        try:
            ans = input("Choose 1/2/3 [1]: ").strip().lower()
        except EOFError:
            return "portrait"
        if not ans:
            return "portrait"
        if ans in choices or ans in choices.values():
            return choices.get(ans, ans)
        print("Please type 1, 2 or 3.", file=sys.stderr)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Turn a long video into vertical short clips.")
    ap.add_argument("input", help="video file or URL (URLs need yt-dlp)")
    ap.add_argument("-o", "--out", default="clips", help="output folder")
    ap.add_argument("-n", "--count", type=int, default=5, help="number of clips")
    ap.add_argument("--min", type=float, default=15, help="min clip length (s)")
    ap.add_argument("--max", type=float, default=90, help="max clip length (s); each clip is as long as its moment, up to this")
    ap.add_argument("--layout", choices=LAYOUTS, default="auto",
                    help="auto picks per clip: screen (facecam+gameplay), split (two people), "
                         "track (follow one face) or blur")
    ap.add_argument("--chat", default=None, help="chat replay file (YouTube live_chat.json, "
                    "TwitchDownloader JSON, or '[hh:mm:ss] msg' text). Auto-fetched for YouTube URLs")
    ap.add_argument("--no-chat", action="store_true", help="ignore chat even if available")
    ap.add_argument("--chat-delay", type=float, default=4.0, help="seconds chat lags behind the moment")
    ap.add_argument("--chat-weight", type=float, default=0.55, help="0-1, how much chat counts vs audio")
    ap.add_argument("--whisper", default="small", help="Whisper model (tiny/base/small/medium/large-v3)")
    ap.add_argument("--lang", default=None, help="language code, e.g. en (default: detect)")
    ap.add_argument("--no-transcribe", action="store_true", help="skip transcription (fastest)")
    ap.add_argument("--llm", action="store_true", help="re-rank candidates with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--start", type=float, default=0, help="only look after this time (s)")
    ap.add_argument("--end", type=float, default=None, help="only look before this time (s)")
    ap.add_argument("--format", choices=["portrait", "landscape", "both"], default=None,
                    help="portrait: 9:16 crop; landscape: full 16:9 frame; both: render each "
                         "clip twice (default: ask when run in a terminal, else portrait)")
    ap.add_argument("--size", default="1080x1920", help="portrait output size WxH "
                    "(landscape uses the same size turned sideways, e.g. 1920x1080)")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    ap.add_argument("--analyze-only", action="store_true", help="write clips.json, don't render")
    ap.add_argument("--captions", choices=["auto", "basic", "off"], default="auto",
                    help="post caption (title, text, hashtags) per clip. auto: written by Claude "
                         "when credentials are set, else basic; basic: offline, from the transcript")
    a = ap.parse_args()
    if a.format is None:
        a.format = ask_format() if (sys.stdin.isatty() and not a.analyze_only) else "portrait"
    log(f"format: {a.format}")
    portrait = a.format in ("portrait", "both")
    landscape = a.format in ("landscape", "both")

    base = Path(a.out).expanduser().resolve()
    work = base / ".work"
    work.mkdir(parents=True, exist_ok=True)
    W, H = (int(v) for v in a.size.lower().split("x"))
    LW, LH = max(W, H), min(W, H)

    src, chat_file, meta = ingest(a.input, work, want_chat=not a.no_chat and not a.chat)
    # every video gets its own folder: "<title> - <creator> - <upload date>"
    out = base / folder_name(meta)
    out.mkdir(parents=True, exist_ok=True)
    log(f"output folder: {out}")
    if a.chat and not a.no_chat:
        chat_file = Path(a.chat).expanduser()
        if not chat_file.exists():
            sys.exit(f"chat file not found: {chat_file}")
    info = probe(src)
    log(f"{src.name}: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, {info['duration'] / 60:.1f} min")
    if info["duration"] < a.min:
        sys.exit("video is shorter than --min")

    # 1) score the whole video from audio (+ chat)
    n = int(info["duration"] / RES)
    if info["has_audio"]:
        log("measuring loudness")
        audio_ex, silent = audio_features(loudness_curve(src))
        audio_ex, silent = np.resize(audio_ex, n), np.resize(silent, n)
    else:
        audio_ex, silent = np.ones(n), np.zeros(n, bool)
    excite, chat_ex = audio_ex, None
    if chat_file:
        msgs = load_chat(chat_file)
        log(f"chat: {len(msgs)} messages from {chat_file.name}")
        if len(msgs) >= 50:
            chat_ex = chat_features(msgs, n, a.chat_delay)
            cw_ = min(max(a.chat_weight, 0), 1)
            # both spiking together is the strongest signal
            excite = np.clip((1 - cw_) * audio_ex + cw_ * chat_ex + 0.4 * np.minimum(audio_ex, chat_ex), 0, 8)
            if not info["has_audio"]:
                excite = chat_ex
        else:
            log("too few chat messages to use")
    lo, hi = int(a.start / RES), int((a.end or info["duration"]) / RES)
    mask = np.zeros(n, bool)
    mask[lo:hi] = True
    excite = np.where(mask, excite, 0)
    silent = np.where(mask, silent, True)

    cands = grow_moments(excite, silent, a.min, a.max)
    pool = pick(cands, a.count * 3 if (a.llm and not a.no_transcribe) else a.count)
    log(f"{len(pool)} candidate moments")

    # 2) transcribe only the candidate regions, snap to sentence boundaries
    transcribe = not a.no_transcribe and info["has_audio"]
    for c in pool:
        segs = []
        if transcribe:
            try:
                segs = transcribe_range(src, max(0, c.start - 5), min(info["duration"], c.end + 6),
                                        a.whisper, a.lang, work)
            except Exception as ex:
                log("transcription unavailable, snapping to silences instead:", ex)
                transcribe = False
        snap(c, segs, silent, a.min, a.max, info["duration"])
        i0, i1 = int(c.start / RES), max(int(c.start / RES) + 1, int(c.end / RES))
        c.parts["audio_peak"] = round(float(audio_ex[i0:i1].max()) / 8, 3)
        if chat_ex is not None:
            c.parts["chat_peak"] = round(float(chat_ex[i0:i1].max()) / 8, 3)

    if a.llm and not a.no_transcribe:
        pool = llm_rerank(pool)
    clips = sorted(pool, key=lambda c: -c.score)[: a.count]
    clips.sort(key=lambda c: c.start)

    # captions first: each clip's file is named after its caption title
    if a.captions != "off" and clips:
        if not any(c.text for c in clips):
            log("captions: no transcript, so captions are based on the video title only")
        make_captions(clips, meta, use_claude=a.captions == "auto")

    # 3) frame + render
    detector = None
    vertical_src = info["width"] / info["height"] <= 0.75
    for k, c in enumerate(clips, 1):
        stem = clip_stem(k, c)
        name = f"{stem}.mp4"
        lname = f"{stem} (landscape).mp4" if c.caption else f"{stem}_landscape.mp4"
        c.files = ([name] if portrait else []) + ([lname] if landscape else [])
        c.file = c.files[0]
        if c.caption:
            write_caption_file(out / f"{stem}.txt", c.caption)
        if landscape and not a.analyze_only:
            log(f"rendering {lname} ({c.end - c.start:.0f}s, landscape, score {c.score:.2f})")
            render_landscape(src, c, info, out / lname, LW, LH, a.crf)
        if not portrait:
            c.layout = "landscape"
            continue
        layout, data = "blur", None
        if not vertical_src and a.layout != "blur":
            detector = detector or get_detector(Path(__file__).resolve().parent / "models")
            A = analyze_faces(src, c, info, detector)
            layout, data, cov = plan_layout(A, info, a.layout)
            c.parts["face_coverage"] = round(cov, 2)
        c.layout = layout
        if a.analyze_only:
            continue
        log(f"rendering {name} ({c.end - c.start:.0f}s, {layout}, score {c.score:.2f})")
        if layout == "track":
            render_track(src, c, info, data, out / name, W, H, a.crf)
        elif layout == "split":
            render_split(src, c, info, data, out / name, W, H, a.crf)
        elif layout == "screen":
            render_screen(src, c, info, data, out / name, W, H, a.crf)
        else:
            render_blur(src, c, info, out / name, W, H, a.crf)

    report = dict(source=str(src), video=meta, chat=str(chat_file) if chat_file else None,
                  settings=vars(a), clips=[{**asdict(c), "score": round(c.score, 4)} for c in clips])
    (out / "clips.json").write_text(json.dumps(report, indent=2))
    # .work is shared and keeps downloaded videos for re-runs; only drop it if empty
    try:
        work.rmdir()
    except OSError:
        pass
    log(f"done: {len(clips)} clips in {out}")
    for c in clips:
        print(f"{', '.join(c.files)}\t{c.start:.1f}-{c.end:.1f}s\tscore {c.score:.2f}\t{c.layout}")


if __name__ == "__main__":
    main()
