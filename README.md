# Sign Language → Text (Streamlit)

Upload a video of someone signing; get back a text transcript. Built with
MediaPipe (hand/pose landmark tracking) + Dynamic Time Warping template
matching, wrapped in a Streamlit UI.

## Honest expectations — read this first

There is no giant pretrained "sign language → English" model behind this
app (nothing like that exists off-the-shelf for general continuous
signing — it's a genuinely hard, active research problem). What this app
actually does:

1. **Tracks** hand & upper-body landmarks in the video (MediaPipe).
2. **Segments** the video into candidate signs by detecting pauses
   between hand movements.
3. **Matches** each segment against a library of *example clips you
   provide* (one short clip per word, via the "Build Vocabulary" tab)
   using Dynamic Time Warping — essentially "which word in my library
   does this movement most resemble?"
4. **Assembles** the recognized words into a sentence (basic cleanup,
   or optionally polished into fluent English via the Claude API).

This means: **it only recognizes words you've taught it**, accuracy
depends heavily on how many/varied examples you add per word, it works
best with deliberate signing with clear pauses between words, and it
will not out-of-the-box understand arbitrary continuous ASL/ISL/BSL
sentences the way a trained deaf interpreter would.

**If you want higher, more general accuracy**, the natural upgrade path
is to replace `recognizer.py`'s DTW matcher with a trained sequence
classifier (e.g. an LSTM/Transformer over the same landmark vectors)
trained on a real dataset such as [WLASL](https://dxli94.github.io/WLASL/)
(ASL, word-level) or [How2Sign](https://how2sign.github.io/) (ASL,
continuous, with English translations). `landmarks.py` already produces
the per-frame feature vectors you'd feed such a model.

A lighter-weight way to use WLASL right now, without training anything:
the **"Import from WLASL" tab** pulls real signed example clips straight
from WLASL's source links and runs them through the existing landmark
pipeline to seed your template library — much faster than filming every
word yourself. See "Importing vocabulary from WLASL" below.

## How it works

```
video.mp4
   │
   ▼
landmarks.py        → per-frame [left hand | right hand | pose] vectors
   │                   (MediaPipe HandLandmarker + PoseLandmarker)
   ▼
segmentation.py      → splits into candidate sign segments by motion energy
   │
   ▼
recognizer.py         → DTW distance vs. each word's example(s) in templates/
   │                    → best-matching word + confidence per segment
   ▼
sentence.py           → join words → clean sentence (rule-based, or
                          optionally polished by the Claude API)
```

## Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads two small MediaPipe model files (~15MB total)
from Google's servers and caches them in `models/`. This requires
internet access once; after that it works fully offline.

### `OSError: libEGL.so.1: cannot open shared object file`

MediaPipe's native runtime needs a handful of OpenGL/EGL system
libraries even when nothing gets displayed on screen (common on
headless Linux servers, minimal Docker base images, and some WSL
setups). Install them via your system package manager:

**Debian/Ubuntu:**
```bash
sudo apt-get update && sudo apt-get install -y \
    libgl1 libegl1 libgles2 ffmpeg
```
(On older Debian/Ubuntu releases you may also need `libglib2.0-0` —
newer releases renamed it to `libglib2.0-0t64` as part of a 64-bit
`time_t` transition, and pinning the wrong name can make `apt-get`
refuse the *entire* install with "none of the choices are installable."
If you hit that, just drop the glib package — headless OpenCV doesn't
need it.)

**Fedora/RHEL:**
```bash
sudo dnf install -y mesa-libGL mesa-libEGL mesa-libGLES ffmpeg
```

**macOS:** this error is Linux-specific; it shouldn't occur.

**Docker:** already handled — the provided `Dockerfile` installs these.

**Streamlit Community Cloud:** already handled — the provided
`packages.txt` tells Streamlit Cloud to apt-install these automatically
on deploy (make sure it stays in the repo root next to `app.py`).

### Offline / no-internet setup

If the machine running the app has no internet access, download the
model files on another machine and copy them into `models/` (or the
directory pointed to by the `SIGN_MODEL_DIR` env var) before starting:

- `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`

### Building a vocabulary

The app starts with **zero words** — you have to teach it. Two ways:

- **One clip per word** ("Build Vocabulary" tab): upload a few-second
  clip of one sign at a time (hands at rest just before/after the sign),
  label it, click "Add to vocabulary."
- **Batch, from one video** ("Batch Add" tab, or `batch_add.py` from the
  command line): record yourself signing a *list of words in a row*,
  with a clear pause between each one, then give the app that same word
  list (in order). It auto-detects each sign's start/end from the pauses
  and saves them all as templates in one pass — much faster than
  recording separate clips. Preview the detected segment count against
  your word count before committing; if they don't match, adjust the
  motion threshold (lower catches smaller movements but may over-split
  a single sign; higher requires bigger movements but may merge two
  signs together) and re-preview.

  CLI equivalent:
  ```bash
  python batch_add.py my_video.mp4 "hello,thank you,water,please"
  ```

Either way, add 2–3 examples per word for noticeably better accuracy.
Templates are stored as `.npy` landmark files + a `manifest.json` under
`templates/` — back that folder up if you invest time building a
vocabulary.

### Importing vocabulary from WLASL

The **"Import from WLASL" tab** (`wlasl.py`) is a third, much faster way
to seed your vocabulary: search WLASL's ~2,000-word list, pick a word,
and the app downloads a few real signed example clips and adds them as
templates automatically — no filming required.

Setup:
1. Get `WLASL_v0.3.json` from the
   [WLASL repo](https://github.com/dxli94/WLASL) or
   [project site](https://dxli94.github.io/WLASL/).
2. Place it in this folder (next to `app.py`), or point the
   `SIGN_WLASL_JSON` env var at it. *(A copy is already bundled in this
   project.)*

What to expect:
- WLASL is **metadata only** — a list of source URLs per word, not the
  video files themselves. About two-thirds of instances are direct
  `.mp4` links the app can download; the rest are YouTube links or old
  Flash `.swf` animations, which are automatically skipped (not
  downloadable this way — YouTube would need a separate tool like
  `yt-dlp`, and `.swf` isn't real video).
- It's a several-year-old academic dataset, so **expect some dead
  links** — the tab downloads a few candidates per word and reports
  which ones succeeded/failed rather than stopping at the first
  failure.
- Each clip is run through the same `landmarks.py` extraction used
  everywhere else in the app, optionally cropped to WLASL's signer
  bounding box first (toggle in the tab) to help MediaPipe on clips
  where the signer is small or off-center.
- This bootstraps vocabulary fast but doesn't replace the recognizer;
  it's still DTW matching against whatever examples you have (WLASL or
  self-filmed) — see "Honest expectations" above.

### Optional: fluent-sentence polishing

Recognized words are joined into a rough sentence by default (e.g.
"Hello water." from `["hello", "water"]`). Turning on "Smooth transcript
with Claude" in the sidebar and supplying an Anthropic API key sends
that word sequence (not the video) to Claude to produce a more natural
sentence.

## Deployment

### Option A — Streamlit Community Cloud (easiest, free)

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the
   repo, set the main file to `app.py`, and deploy. The included
   `packages.txt` makes it auto-install the system libraries MediaPipe
   needs (see the `libEGL.so.1` troubleshooting note above) — no
   manual step required.
3. The `templates/` directory on Streamlit Cloud is **ephemeral** —
   it resets on redeploy/restart. For a persistent vocabulary, either
   commit a starter `templates/` folder to the repo, or point
   `SIGN_TEMPLATE_DIR` at a mounted persistent volume / object storage
   you sync to (see Option B/C for more control).

### Option B — Docker (any host: Fly.io, Render, EC2, a home server...)

```bash
docker build -t sign-language-app .
docker run -p 8501:8501 -v $(pwd)/templates:/app/templates sign-language-app
```

The `-v` flag mounts a local `templates/` folder into the container so
your vocabulary survives container restarts. Open `http://localhost:8501`.

### Option C — Hugging Face Spaces

1. Create a new Space, SDK = Docker.
2. Push this folder's contents (including the `Dockerfile`) to the
   Space's repo.
3. Space builds and serves it automatically on port 7860 — either
   change `EXPOSE`/`--server.port` in the `Dockerfile` to `7860`, or
   set the Space's port setting to `8501`.
4. Use a Space's **persistent storage** (or an attached dataset repo)
   for `templates/` if you want the vocabulary to survive rebuilds.

## Project files

| File               | Purpose                                              |
|--------------------|-------------------------------------------------------|
| `app.py`           | Streamlit UI (upload video, translate, manage vocab)  |
| `landmarks.py`     | Video → per-frame hand/pose landmark vectors          |
| `models.py`        | Downloads/caches the MediaPipe model files            |
| `segmentation.py`  | Splits continuous signing into candidate sign chunks  |
| `templates.py`     | Vocabulary storage (add/remove/load example signs)    |
| `batch_add.py`     | Split one multi-word video into many labeled templates at once |
| `wlasl.py`          | Search WLASL metadata + download real example clips per word |
| `WLASL_v0.3.json`   | Bundled WLASL dataset metadata (source links, not videos) |
| `recognizer.py`    | DTW nearest-neighbor matching                         |
| `sentence.py`      | Gloss sequence → sentence (rule-based or Claude-polished) |
| `requirements.txt` | Python dependencies                                    |
| `packages.txt`     | System (apt) libraries MediaPipe needs — used by Streamlit Cloud |
| `Dockerfile`       | Container build for deployment                        |

## Known limitations

- **Single signer, upper body, front-facing camera** works best —
  the pose/hand model isn't robust to extreme angles or heavy occlusion.
- **No facial-grammar recognition** (eyebrow raises, mouth shapes, etc.)
  which carry real grammatical meaning in ASL/ISL — this app only
  tracks hands + arm/shoulder pose.
- **DTW matching is O(n·m) per template**, so recognition time grows
  with both segment length and vocabulary size; fine for a few dozen
  words, will need optimization (e.g. FastDTW, or replacing DTW with a
  trained classifier) for a large vocabulary.
- **Sentence assembly is intentionally simple** — sign language grammar
  doesn't map word-for-word onto English word order, so even perfect
  word recognition won't automatically produce grammatically ideal
  English without the optional Claude polishing step (or a proper
  seq2seq translation model trained for the language pair).
