# clipper

Turn a long video (stream VOD, podcast, vlog) into vertical 1080×1920 clips for
YouTube Shorts, TikTok and Reels. Inspired by ClipsAI, rebuilt to fix its gaps.

| | ClipsAI | clipper |
|---|---|---|
| How moments are picked | Cuts at every topic change (no ranking) | Ranks moments: loudness spikes vs. the local baseline, build-up → peak → payoff, hook, dead-air penalty; optional Claude re-rank |
| Works on streams / IRL / gaming | Weak (transcript only) | Yes (audio energy works with no talking) |
| Clip length / count | Not configurable | `--min`, `--max`, `-n` |
| Long videos | Transcribes everything | Scores the whole audio first, transcribes only the candidate clips |
| Cut boundaries | Topic boundaries | Sentence boundaries; never mid-word |
| Reframing | Follow the active speaker (Pyannote, needs HF token) | Smooth "camera operator" face tracking: dead zone, glide, hard jumps at scene cuts; blurred-background layout when there's no clear face |
| Output | Timestamps + crop data | Finished MP4s + `clips.json` |
| Maintained | Last release Jan 2024 | Current libraries (faster-whisper, OpenCV YuNet) |

## Install (macOS)

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the Whisper model (~500 MB for `small`) and a 230 KB face model.

## Use

```bash
# a local file
python clipper.py stream.mp4 -o clips

# a URL (uses yt-dlp; only clip videos you have the rights to)
python clipper.py "https://www.youtube.com/watch?v=..." -o clips -n 8 --min 20 --max 45

# fastest: audio-only scoring, no transcription
python clipper.py stream.mp4 --no-transcribe

# only look at part of a long VOD (seconds)
python clipper.py vod.mp4 --start 3600 --end 7200

# let Claude re-rank the best candidates by transcript
export ANTHROPIC_API_KEY=sk-ant-...
python clipper.py podcast.mp4 --llm
```

Options: `--layout auto|track|blur`, `--whisper tiny|base|small|medium|large-v3`,
`--lang en`, `--size 1080x1920`, `--crf 20`, `--analyze-only` (just write `clips.json`).

## Output

`clips/clip_01_12m34s.mp4 …` plus `clips/clips.json` with each clip's times, score,
score breakdown (peak, intensity, hook, payoff, dead air, face coverage), transcript
and layout. Use it to see why a moment was picked.

## How it works

1. **Loudness.** Measures RMS loudness every 0.1 s and compares it with a rolling
   60 s baseline, so a quiet podcast and a loud stream are both judged against
   themselves.
2. **Scoring.** Slides windows of several lengths across the video and scores each
   one. A strong peak, sustained intensity and a peak about 60% of the way through
   (build-up, then payoff) all raise the score. Dead air lowers it.
3. **Picking.** Chooses the best non-overlapping windows.
4. **Snapping.** Transcribes just those regions and moves the start and end to
   sentence boundaries. With `--no-transcribe` it snaps to nearby silences.
5. **Framing.** Detects faces about 6×/s, sticks with one subject, and holds the
   frame still until they leave a dead zone, then glides. It cuts instantly on
   scene changes or speaker switches. When no face is found in less than 35% of
   the clip, it uses the full frame over a blurred background.

## Tuning ideas

- If clips feel too short or too long, change `--min` and `--max`.
- For talk-heavy podcasts, use `--llm`. Loudness alone favours shouting.
- For gameplay with a facecam, use `--layout blur`. The face tracker will follow
  the facecam otherwise.
