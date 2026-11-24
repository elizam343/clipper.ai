# clipper

Turn a long video (stream VOD, podcast, vlog, gameplay) into vertical 1080×1920
clips for YouTube Shorts, TikTok and Reels. Inspired by ClipsAI, with ideas from
OpenShorts and stream-clipper.

| | ClipsAI | clipper |
|---|---|---|
| How moments are picked | Cuts at every topic change (no ranking) | Ranks moments: **chat spikes** + loudness spikes against the local baseline, build-up → peak → payoff, hook, dead-air penalty; optional Claude re-rank |
| Chat replays | No | YouTube live chat (auto-fetched), Twitch, text logs |
| Works on streams / IRL / gaming | Weak (transcript only) | Yes |
| Clip length / count | Not configurable | `--min`, `--max`, `-n` |
| Long videos | Transcribes everything | Scores the whole video first, transcribes only the candidate clips |
| Cut boundaries | Topic boundaries | Sentence boundaries; never mid-word |
| Layouts | Follow the active speaker | Picked per clip: **track** one face, **split** two people top/bottom, **screen** (facecam over gameplay), **blur** background |
| Output | Timestamps + crop data | Finished MP4s + `clips.json` |

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

# a YouTube URL; its chat replay is fetched automatically if it has one
python clipper.py "https://www.youtube.com/watch?v=..." -o clips -n 8 --min 20 --max 45

# a Twitch VOD with chat exported by TwitchDownloader
python clipper.py vod.mp4 --chat vod_chat.json

# fastest: no transcription
python clipper.py stream.mp4 --no-transcribe

# only look at part of a long VOD (seconds)
python clipper.py vod.mp4 --start 3600 --end 7200

# force a layout
python clipper.py gameplay.mp4 --layout screen

# let Claude re-rank the best candidates by transcript
export ANTHROPIC_API_KEY=sk-ant-...
python clipper.py podcast.mp4 --llm

# keep the whole 16:9 frame (the joke needs both sides of the screen)
python clipper.py "https://www.youtube.com/watch?v=..." --format landscape

# render every clip both ways and decide later
python clipper.py stream.mp4 --format both
```

Only clip videos you have the rights to.

### Portrait or landscape (`--format portrait|landscape|both`)

If you run clipper in a terminal without `--format`, it asks you first:

```
Where is the funny part of these clips?
  1) portrait   - one person/face; crop to vertical 9:16 (Shorts/TikTok/Reels)
  2) landscape  - the joke needs the whole frame; keep 16:9, no crop
  3) both       - render each clip both ways and pick later
Choose 1/2/3 [1]:
```

In scripts or other non-interactive runs it uses portrait without asking. Portrait
clips use the layouts below. Landscape clips are 1920×1080 with nothing cropped
(letterboxed if the source isn't 16:9) and end in `_landscape.mp4`.

### Chat

Chat spikes are the strongest signal for streams. When chat explodes, something
happened. clipper reads:

- **YouTube** `*.live_chat.json`. It's downloaded automatically for YouTube URLs,
  or fetch it yourself with `yt-dlp --skip-download --write-subs --sub-langs live_chat URL`.
- **Twitch** JSON from [TwitchDownloader](https://github.com/lay295/TwitchDownloader)
  (`TwitchDownloaderCLI chatdownload --id VOD_ID -o chat.json`).
- **chat-downloader** JSON, or plain text lines like `[1:02:03] user: message`.

For a local video, a chat file saved next to it is used automatically
(`video.live_chat.json`, `video.chat.json`, `video.json` or `video.chat.txt`).

Each half-second gets a score from its message rate, with extra weight for hype
messages (POG, KEKW, LMAO, OMG, "clip it", 💀😂, ALL CAPS). That score is compared
with a rolling 2-minute baseline and shifted back `--chat-delay` seconds (default 4),
because chat reacts after the moment. It's then blended with audio using
`--chat-weight` (default 0.55). Moments where chat and audio both spike get a bonus.

### Layouts (`--layout auto|track|split|screen|blur`)

`auto` analyzes each clip and picks one of these:

- **screen**: a small face that stays in one spot off-centre, while the rest of the
  frame is moving (gameplay or a screen share). The facecam goes on top (35%) and
  the game, centre-cropped, goes below.
- **split**: two people too far apart to fit in one vertical crop (a podcast
  two-shot). Each person gets their own half, stacked.
- **track**: follows one face like a calm camera operator. It holds still inside a
  dead zone, glides when the person moves, and cuts instantly on scene changes.
- **blur**: no clear face. The full frame sits over a blurred copy of itself.

## Output

Each video gets its own folder inside `-o`, named after the video's title, creator
and upload date:

```
clips/
  OPERATE ON BILLY! Tonsil, Ear and Heart Surgery - jacksepticeye - 2015-03-05/
    clip_01_00m21s.mp4
    clip_01_00m21s_landscape.mp4     (with --format landscape or both)
    clips.json
  .work/                             (downloaded videos, reused on re-runs)
```

For a local file the folder uses the file's title tag (or its file name), the artist
tag if it has one, and the creation date.

`clips.json` has the video's title, creator, date and URL. It also lists each clip's
times, score, score breakdown (peak, intensity, hook, payoff, dead air, audio peak,
chat peak, face coverage), transcript, layout and files. Use it to see why a moment
was picked.

## How it works

1. **Signals.** Audio loudness every 0.1 s is compared with a rolling 60 s baseline.
   Chat rate and hype are compared with a rolling 2 min baseline. The two are blended
   into one excitement curve.
2. **Scoring.** Windows of several lengths slide across the video. A strong peak,
   sustained intensity and a peak about 60% of the way through (build-up, then
   payoff) all raise the score. Dead air lowers it. The best non-overlapping windows
   are chosen.
3. **Snapping.** Just those regions are transcribed, and each clip's start and end
   move to sentence boundaries. With `--no-transcribe`, they snap to nearby silences.
4. **Framing.** For each clip, faces are detected about 6 times a second and scene
   cuts on every frame, along with how much the frame moves. That's used to choose
   a layout and render it.

## Tuning ideas

- If clips feel too short or too long, change `--min` and `--max`.
- If chat seems to lag more or less than 4 s on your streams, adjust `--chat-delay`.
- For talk-heavy podcasts, use `--llm`. Loudness alone favours shouting.
- Facecam detection needs the gameplay to be moving. For slow games, force
  `--layout screen`.
