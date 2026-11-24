#!/usr/bin/env python3
"""
clipper — turn a long video into vertical short clips.

Inspired by ClipsAI, rebuilt to fix its weak spots:
  * Ranks moments (loudness spikes, hooks, payoff, dead air) instead of
    cutting at every topic change, so it works for streams, not just podcasts.
  * You choose how many clips and how long they are.
  * Fast on long streams: it scores the whole audio first, then transcribes
    only the candidate regions (no 3-hour Whisper pass).
  * Smooth "camera operator" face tracking with a dead zone, look-ahead
    smoothing and hard jumps at scene cuts; falls back to a blurred-background
    layout when there's no clear face (crowds, gameplay, screens).
  * Renders finished 1080x1920 MP4s plus a clips.json report.
  * Optional: let Claude re-rank candidates from their transcripts (--llm).

Usage:
  python clipper.py VIDEO_OR_URL [-o out] [--count 5] [--min 15] [--max 60]
"""
from __future__ import annotations

import argparse
import json
import math
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


def log(*a):
    print("[clipper]", *a, file=sys.stderr, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- ingest
def ingest(src: str, workdir: Path) -> Path:
    if re.match(r"^https?://", src):
        if not shutil.which("yt-dlp"):
            sys.exit("yt-dlp is not installed: pip install yt-dlp")
        log("downloading", src)
        out = workdir / "source.%(ext)s"
        run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4", "-o", str(out), src])
        files = sorted(workdir.glob("source.*"))
        if not files:
            sys.exit("download failed")
        return files[0]
    p = Path(src).expanduser().resolve()
    if not p.exists():
        sys.exit(f"not found: {p}")
    return p


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


def audio_features(db: np.ndarray):
    """Return excitement (z-score above local baseline) and silence mask at RES."""
    base = rolling_stat(db, int(60 / HOP), np.median)
    mad = rolling_stat(db, int(60 / HOP), lambda s: np.median(np.abs(s - np.median(s))))
    z = (db - base) / np.maximum(mad * 1.4826, 2.0)
    k = int(RES / HOP)
    n = len(db) // k
    zr = z[: n * k].reshape(n, k)
    dbr = db[: n * k].reshape(n, k)
    excite = np.clip(zr.max(1), 0, 8)
    excite = np.convolve(excite, np.ones(3) / 3, mode="same")
    silent = (dbr.mean(1) < np.maximum(base[: n * k].reshape(n, k).mean(1) - 12, -60)) | (dbr.mean(1) < -50)
    return excite, silent


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


def score_windows(excite, silent, min_len, max_len, words=None):
    """Score every window; returns list of Clip candidates."""
    n = len(excite)
    lens = sorted({int(round(L / RES)) for L in np.linspace(min_len, max_len, 4)})
    cands = []
    hook_bins = int(3 / RES)
    wd_med = None
    if words is not None:
        nz = words[words > 0]
        wd_med = float(np.median(nz)) if len(nz) else 1.0
    for L in lens:
        if L >= n:
            continue
        win = np.lib.stride_tricks.sliding_window_view(excite, L)
        peak = np.minimum(win.max(1), 6) / 6
        top = np.sort(win, axis=1)[:, -max(1, L // 6):].mean(1)
        intensity = np.minimum(top, 4) / 4
        hook = np.minimum(win[:, :hook_bins].mean(1), 3) / 3
        pos = win.argmax(1) / L
        # best shorts build up to the peak and end soon after it
        payoff = np.clip(1 - np.abs(pos - 0.6) / 0.45, 0, 1)
        dead = np.lib.stride_tricks.sliding_window_view(silent.astype(np.float32), L).mean(1)
        lenpen = (L * RES - min_len) / max(1, max_len - min_len) * 0.05
        s = 0.3 * peak + 0.2 * intensity + 0.1 * hook + 0.2 * payoff - 0.35 * dead - lenpen
        sp = np.zeros_like(s)
        if words is not None:
            wv = np.lib.stride_tricks.sliding_window_view(words, L).mean(1)
            sp = np.minimum(wv / max(wd_med, 1e-3), 1.5) / 1.5
            s = s + 0.1 * sp
        for i in range(0, len(s), 2):  # 1s stride is plenty
            cands.append(Clip(i * RES, (i + L) * RES, float(s[i]), dict(
                peak=round(float(peak[i]), 3), intensity=round(float(intensity[i]), 3),
                hook=round(float(hook[i]), 3), payoff=round(float(payoff[i]), 2),
                dead_air=round(float(dead[i]), 3), speech=round(float(sp[i]), 3))))
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
    wav = tmp / f"seg_{start:.1f}.wav"
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
         "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    segs, _ = whisper_model(model).transcribe(str(wav), language=lang, vad_filter=True,
                                              word_timestamps=True)
    out = []
    for s in segs:
        words = [(w.start + start, w.end + start, w.word) for w in (s.words or [])]
        out.append(dict(start=s.start + start, end=s.end + start, text=s.text.strip(), words=words))
    wav.unlink(missing_ok=True)
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
            # never end inside a word
            for g in segs:
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


# --------------------------------------------------------------------------- framing
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
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
                          "-t", f"{dur:.3f}", *vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, bufsize=10 ** 8)
    return p


def analyze_framing(path, clip, info, detector):
    """Return per-frame crop centers (source px) or None if a face layout won't work."""
    import cv2
    aw = 640
    ah = int(round(info["height"] * aw / info["width"] / 2) * 2)
    detector.setInputSize((aw, ah))
    fps = info["fps"]
    step = max(1, round(fps / 6))
    dur = clip.end - clip.start
    p = decode(path, clip.start, dur, aw, ah)
    fsz = aw * ah * 3
    cuts, samples, prev_hist, prev_target = [0], {}, None, None
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
            prev_target = None
        prev_hist = hist
        if i % step == 0:
            _, faces = detector.detect(frame)
            best = None
            if faces is not None:
                bs = -1
                for f in faces:
                    x, y, fw, fh, conf = f[0], f[1], f[2], f[3], f[14]
                    cx = x + fw / 2
                    sc = fw * fh * conf
                    if prev_target is not None and abs(cx - prev_target) < aw * 0.12:
                        sc *= 2.0  # stickiness: don't hop between faces
                    if fw < aw * 0.025:  # ignore tiny background faces
                        continue
                    if sc > bs:
                        bs, best = sc, cx
            samples[i] = best
            if best is not None:
                prev_target = best
        i += 1
    p.wait()
    nframes = i
    if nframes == 0:
        return None, 0.0
    found = [v for v in samples.values() if v is not None]
    coverage = len(found) / max(1, len(samples))
    if coverage < 0.35:
        return None, coverage

    # build per-frame target, holding last position across short gaps
    tgt = np.full(nframes, np.nan)
    for k, v in samples.items():
        if v is not None:
            tgt[k] = v
    cuts = sorted(set(cuts)) + [nframes]
    cw = info["height"] * 9 / 16 * aw / info["width"]  # crop width in analysis px
    half = cw / 2
    out = np.zeros(nframes)
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
    return out * info["width"] / aw, coverage


# --------------------------------------------------------------------------- render
def enc_args(out, crf):
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]


def render_track(path, clip, info, centers, out, W, H, crf):
    import cv2
    sw, sh, fps = info["width"], info["height"], info["fps"]
    cw = int(round(sh * 9 / 16 / 2) * 2)
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
        crop = frame[:, x0:x0 + cw]
        enc.stdin.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA).tobytes())
        i += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()


def render_blur(path, clip, info, out, W, H, crf):
    dur = clip.end - clip.start
    fc = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H},gblur=sigma=40,eq=brightness=-0.12[bg];"
          f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{clip.start:.3f}", "-i", str(path),
         "-t", f"{dur:.3f}", "-filter_complex", fc, "-map", "[v]",
         *(["-map", "0:a"] if info["has_audio"] else []), *enc_args(out, crf)])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Turn a long video into vertical short clips.")
    ap.add_argument("input", help="video file or URL (URLs need yt-dlp)")
    ap.add_argument("-o", "--out", default="clips", help="output folder")
    ap.add_argument("-n", "--count", type=int, default=5, help="number of clips")
    ap.add_argument("--min", type=float, default=15, help="min clip length (s)")
    ap.add_argument("--max", type=float, default=60, help="max clip length (s)")
    ap.add_argument("--layout", choices=["auto", "track", "blur"], default="auto",
                    help="auto: follow the face, blurred background if no clear face")
    ap.add_argument("--whisper", default="small", help="Whisper model (tiny/base/small/medium/large-v3)")
    ap.add_argument("--lang", default=None, help="language code, e.g. en (default: detect)")
    ap.add_argument("--no-transcribe", action="store_true", help="audio-only scoring (fastest)")
    ap.add_argument("--llm", action="store_true", help="re-rank candidates with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--start", type=float, default=0, help="only look after this time (s)")
    ap.add_argument("--end", type=float, default=None, help="only look before this time (s)")
    ap.add_argument("--size", default="1080x1920", help="output size WxH")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    ap.add_argument("--analyze-only", action="store_true", help="write clips.json, don't render")
    a = ap.parse_args()

    out = Path(a.out).expanduser().resolve()
    work = out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    W, H = (int(v) for v in a.size.lower().split("x"))

    src = ingest(a.input, work)
    info = probe(src)
    log(f"{src.name}: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, {info['duration'] / 60:.1f} min")
    if info["duration"] < a.min:
        sys.exit("video is shorter than --min")

    # 1) score the whole video from audio
    if info["has_audio"]:
        log("measuring loudness")
        excite, silent = audio_features(loudness_curve(src))
    else:
        n = int(info["duration"] / RES)
        excite, silent = np.ones(n), np.zeros(n, bool)
    lo, hi = int(a.start / RES), int((a.end or info["duration"]) / RES)
    mask = np.zeros(len(excite), bool)
    mask[lo:hi] = True
    excite = np.where(mask, excite, 0)
    silent = np.where(mask, silent, True)

    cands = score_windows(excite, silent, a.min, a.max)
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

    if a.llm and not a.no_transcribe:
        pool = llm_rerank(pool)
    clips = sorted(pool, key=lambda c: -c.score)[: a.count]
    clips.sort(key=lambda c: c.start)

    # 3) frame + render
    detector = None
    for k, c in enumerate(clips, 1):
        name = f"clip_{k:02d}_{int(c.start // 60):02d}m{int(c.start % 60):02d}s.mp4"
        c.file = name
        centers = None
        if a.layout != "blur" and not a.analyze_only:
            detector = detector or get_detector(Path(__file__).resolve().parent / "models")
            centers, cov = analyze_framing(src, c, info, detector)
            c.parts["face_coverage"] = round(cov, 2)
            if centers is None and a.layout == "track":
                centers = np.full(1, info["width"] / 2)
        if info["width"] / info["height"] <= 0.75:  # already vertical
            centers = None
        c.layout = "track" if centers is not None else "blur"
        if a.analyze_only:
            continue
        log(f"rendering {name} ({c.end - c.start:.0f}s, {c.layout}, score {c.score:.2f})")
        if centers is not None:
            render_track(src, c, info, centers, out / name, W, H, a.crf)
        else:
            render_blur(src, c, info, out / name, W, H, a.crf)

    report = dict(source=str(src), settings=vars(a),
                  clips=[{**asdict(c), "score": round(c.score, 4)} for c in clips])
    (out / "clips.json").write_text(json.dumps(report, indent=2))
    if not re.match(r"^https?://", a.input):
        shutil.rmtree(work, ignore_errors=True)  # keep downloaded sources for re-runs
    log(f"done: {len(clips)} clips in {out}")
    for c in clips:
        print(f"{c.file}\t{c.start:.1f}-{c.end:.1f}s\tscore {c.score:.2f}\t{c.layout}")


if __name__ == "__main__":
    main()
