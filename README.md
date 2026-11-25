# clipper

Turn a long video (stream VOD, podcast, vlog, gameplay) into vertical 1080×1920
clips for YouTube Shorts, TikTok and Reels. Inspired by ClipsAI, with ideas from
OpenShorts and stream-clipper.

| | ClipsAI | clipper |
|---|---|---|
| How moments are picked | Cuts at every topic change (no ranking) | Ranks moments: **chat spikes** + loudness spikes against the local baseline, build-up → peak → payoff, hook, dead-air penalty; optional Claude re-rank |
| Chat replays | No | YouTube live chat (auto-fetched), Twitch, text logs |
| Works on streams / IRL / gaming | Weak (transcript only) | Yes |
| Clip length / count | Not configurable | Each clip as long as its moment (15–90 s by default, `--min`/`--max`); as many clips as are good enough (2–10 by default), or `-n` for an exact number |
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
python clipper.py "https://www.youtube.com/watch?v=..." -o clips --min 20 --max 45

# exactly 8 clips instead of the automatic count
python clipper.py stream.mp4 -n 8

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

# keep the whole frame inside the Short (the joke needs both sides of the screen)
python clipper.py "https://www.youtube.com/watch?v=..." --format landscape

# render every clip both ways and decide later
python clipper.py stream.mp4 --format both
```

Only clip videos you have the rights to.

### How many clips

By default clipper keeps every clip that's good enough, at least `--min-clips` (2)
and at most `--max-clips` (10), so a short video with two good moments gets two clips
and a long, funny stream can get ten.

- With `--llm`, Claude rates a wider pool of candidates (up to 20) from 0 to 10 as
  standalone Shorts, and clips rated 6 or higher make the cut.
- Offline, clips scoring within 75% of the best clip's score make the cut.
- `-n 5` gives exactly 5 instead.

### Portrait or landscape (`--format portrait|landscape|both`)

If you run clipper in a terminal without `--format`, it asks you first:

```
Where is the funny part of these clips?
  1) portrait   - one person/face; crop to vertical 9:16 (Shorts/TikTok/Reels)
  2) landscape  - the joke needs the whole frame; keep all of it inside the 9:16 Short,
                  with black bars above and below
  3) both       - render each clip both ways and pick later
Choose 1/2/3 [1]:
```

In scripts or other non-interactive runs it uses portrait without asking. Portrait
clips use the layouts below. Landscape clips are also vertical 1080×1920 Shorts, but
with nothing cropped: the whole frame sits in the middle with black bars above and
below. Their files end in ` (landscape).mp4`.

### On-screen captions and title (`--overlay all|captions|title|none`)

Every clip gets two things burned into the video, so it isn't just a straight reupload:

- **A title card**: the clip's title on a white card. In landscape clips it sits in the
  black bar above the video; in portrait clips, near the top.
- **Word-by-word captions**: big white words in short phrases, with the word being spoken
  in yellow. In landscape clips they go in the black bar under the video, so they never
  cover the game; in portrait clips, in the lower middle. Swears are swapped or dropped
  here too.

Each clip also gets a `… thumbnail.jpg`: a clean frame (no captions) with the title card
on top and the clip's **hook** underneath, its most quotable line in big yellow text
(“IT BROKE THE GAME!”). Claude picks the hook; offline it's the clip's next punchiest
short line. Use it if Studio lets you upload a Shorts thumbnail. `--overlay title` or
`captions` keeps just one overlay; `none` turns both off; `--no-thumbnail` skips the
image. Needs Pillow
(`pip install pillow`).

### Cold open (`--no-teaser` to turn off)

The first second decides whether people swipe away, so each clip opens with a 2–3.5 s
preview of its best moment, then plays from the top. The preview is the clip's **hook**
(the same line as on the thumbnail) when it can be found in the transcript, otherwise
the clip's most exciting moment. Clips whose best moment is already in the first few
seconds don't get one. The burned-in captions follow along.

### Post captions (`--captions auto|basic|off`)

Every clip gets a caption you can paste when you post it: a title, a description,
and hashtags. **The clip's file is named after its title**, numbered in
the order the clips appear in the video (`04 - God what are we doing.mp4`). The
caption is saved as a `.txt` with the same name and in `clips.json`.

- **auto** (default): Claude writes the captions from each clip's transcript in one
  request. Needs `pip install anthropic` and an API key
  (`export ANTHROPIC_API_KEY=sk-ant-...`). Without them, clipper falls back to basic.
- **basic**: offline. The title is the clip's punchiest short line (exclamations and
  questions first). The description quotes the clip's best other line, says which part
  of the video it's from (using YouTube chapters when the video has them) and credits
  the creator with a link to the full video. Hashtags come from the creator, the video
  title, and words the clip repeats.

Claude also gets the creator's own summary of the video and each clip's chapter, so its
descriptions say what's happening, not just what was said.

**Captions never swear.** YouTube shows Shorts with profanity in the title or description
to fewer people, so Claude is told not to swear or quote lines that do, and basic captions
pick clean lines first. Every caption then goes through a filter either way: swears are
swapped for mild words ("shit" → "stuff", "what the fuck" → "what the heck"), intensifiers
like "fucking" are dropped, and hashtags or tags that swear are removed. The clip's audio
is untouched.

The creator is credited by their YouTube @handle (`@jacksepticeye`), which links to
their channel and notifies them.

**Tags** for YouTube Studio's Tags box go on a `Tags:` line at the bottom of each `.txt`,
comma-separated and ready to paste, at most 15 and under YouTube's 500-character limit.
They're the creator's name, the clip's chapter, the original video's own tags that fit
this clip (tags for the whole video like "walkthrough" or "ending" are dropped), a few
broad tags about the game or topic, and "funny moments" / "shorts". With Claude, its
tags for the clip come first.
- **off**: no captions. Files are named `clip_01_12m34s.mp4` instead.

```
God what are we doing?

“I'm drilling a hole right into your head That's disgusting.”
Ear Surgery, from @jacksepticeye's “OPERATE ON BILLY! | Tonsil, Ear and Heart Surgery”.
Full video: https://www.youtube.com/watch?v=zFk_JjsTMKc

#shorts #jacksepticeye #drilling #operate #billy

Tags: jacksepticeye, jacksepticeye funny moments, jacksepticeye shorts, ear surgery, ...
```

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
    04 - God what are we doing.mp4
    04 - God what are we doing (landscape).mp4    (with --format landscape or both)
    04 - God what are we doing thumbnail.jpg
    04 - God what are we doing.txt                (post caption)
    clips.json
  .work/                             (downloaded videos, reused on re-runs)
```

For a local file the folder uses the file's title tag (or its file name), the artist
tag if it has one, and the creation date.

### Posting log

Every run adds its clips to `clips/posting_log.csv`: title, hook, length, cold-open
length, Claude's rating, score, and where in the source video it's from. Fill in
`posted_at`, `youtube_url`, `views`, `stayed_to_watch_pct` (Studio's "viewed vs. swiped
away") and `avg_pct_viewed` as you post. Once there are a few dozen clips with numbers,
it shows which kinds of clips actually do well. Re-runs replace their own rows but keep
anything you typed in.

`clips.json` has the video's title, creator, date and URL. It also lists each clip's
times, score, score breakdown (peak, intensity, hook, payoff, dead air, audio peak,
chat peak, face coverage), transcript, layout and files. Use it to see why a moment
was picked.

## How it works

1. **Signals.** Audio loudness every 0.1 s is compared with a rolling 60 s baseline.
   Chat rate and hype are compared with a rolling 2 min baseline. The two are blended
   into one excitement curve.
2. **Finding moments (dynamic length).** Each clip grows outward from an excitement
   peak, so the moment decides how long it is. Going forward, the clip keeps
   extending while exciting stretches keep coming back within 5 s, so a pause between
   laughs doesn't end it. It stops once things stay ordinary. Going backward, it takes
   in the build-up. "Exciting" is measured against the video's own ordinary level, so
   it works for a calm podcast and a screaming stream alike. Clips stay between
   `--min` and `--max` (default 15–90 s). A moment longer than `--max` is trimmed to
   its strongest stretch.
3. **Scoring.** A clip gets credit for its peak and for every second above the
   video's ordinary level, and loses points for flat stretches and dead air. A long
   clip only wins if it stays interesting. The best non-overlapping clips are chosen.
4. **Snapping.** Just those regions are transcribed, and each clip's start and end
   move to sentence boundaries. With `--no-transcribe`, they snap to nearby silences.
5. **Framing.** For each clip, faces are detected about 6 times a second and scene
   cuts on every frame, along with how much the frame moves. That's used to choose
   a layout and render it.

## Tuning ideas

- Clip length follows the moment. To cap it, use `--max` (for example `--max 60`);
  to never go below a length, use `--min`.
- If chat seems to lag more or less than 4 s on your streams, adjust `--chat-delay`.
- For talk-heavy podcasts, use `--llm`. Loudness alone favours shouting.
- Facecam detection needs the gameplay to be moving. For slow games, force
  `--layout screen`.
