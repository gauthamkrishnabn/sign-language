"""
wlasl.py
--------
Bridges the WLASL (Word-Level American Sign Language) dataset into this
app's vocabulary system, so you can bootstrap the template library from
real signed clips instead of filming every word yourself.

IMPORTANT — what WLASL_v0.3.json actually is:
It's METADATA ONLY (~2000 words x several instances each) -- for every
word it lists where a video of that sign lives, not the video itself.
Roughly:
  - ~67% are direct .mp4 links (various dictionary/university sites) --
    these we CAN download and feed straight into the existing landmark
    pipeline.
  - ~24% are YouTube links -- need a separate tool (e.g. yt-dlp), not
    handled here.
  - ~8% are old Flash .swf animations -- not real video, can't be
    decoded by OpenCV, skipped.
This module only ever attempts the direct-link instances, and it does
so per-word, on demand, when you ask for a word in the app -- it never
bulk-downloads the whole dataset.

Also worth knowing: this is a several-year-old academic dataset. Many
of those direct links have since gone dead (hosts restructured, sites
shut down, etc). download_instance() raises on failure so the caller
can just try the next instance / report it and move on -- expect some
failure rate, it's normal.
"""

import functools
import os

import requests

DEFAULT_JSON_PATH = os.environ.get(
    "SIGN_WLASL_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "WLASL_v0.3.json"),
)


def available(json_path=None):
    return os.path.exists(json_path or DEFAULT_JSON_PATH)


@functools.lru_cache(maxsize=1)
def _load_raw(json_path):
    import json
    with open(json_path, "r") as f:
        return json.load(f)


def _entries(json_path=None):
    return _load_raw(json_path or DEFAULT_JSON_PATH)


def is_direct_downloadable(url):
    """True for a plain video file we can fetch with a normal HTTP GET.
    False for YouTube (needs a dedicated downloader) and .swf (Flash,
    not decodable video)."""
    low = url.lower()
    if "youtube.com" in low or "youtu.be" in low:
        return False
    if low.endswith(".swf"):
        return False
    return True


def all_glosses(json_path=None):
    """Sorted list of every word WLASL has at least one example for."""
    return sorted(entry["gloss"] for entry in _entries(json_path))


def search_glosses(query, json_path=None, limit=40):
    """Prefix matches first, then substring matches, capped at `limit`."""
    glosses = all_glosses(json_path)
    q = query.strip().lower()
    if not q:
        return glosses[:limit]
    starts = [g for g in glosses if g.lower().startswith(q)]
    contains = [g for g in glosses if q in g.lower() and g not in starts]
    return (starts + contains)[:limit]


def get_instances(gloss, json_path=None):
    """All known example clips for one word (case-insensitive exact
    gloss match). Each instance dict gets a `downloadable` flag added."""
    for entry in _entries(json_path):
        if entry["gloss"].lower() == gloss.lower():
            out = []
            for inst in entry["instances"]:
                inst = dict(inst)
                inst["downloadable"] = is_direct_downloadable(inst["url"])
                out.append(inst)
            return out
    return []


def download_instance(instance, dest_dir, timeout=20):
    """
    Downloads one instance's clip to dest_dir.

    Returns the local file path on success.
    Raises (ValueError / requests exceptions) on failure -- callers
    should catch broadly and just skip to the next instance, since dead
    links are expected and not a bug.
    """
    url = instance["url"]
    if not is_direct_downloadable(url):
        raise ValueError(
            "Not a direct video link (YouTube or .swf) -- can't auto-download this one."
        )

    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    fname = f"wlasl_{instance.get('video_id', instance.get('instance_id'))}{ext}"
    dest_path = os.path.join(dest_dir, fname)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; sign-language-app/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()

    size = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                size += len(chunk)

    if size < 1024:  # almost certainly an error page, not a video
        os.remove(dest_path)
        raise ValueError("Downloaded file was suspiciously small -- likely a dead/broken link.")

    return dest_path


def instance_bbox(instance):
    """Returns (x1, y1, x2, y2) if this instance has a usable bounding
    box for the signer, else None."""
    bbox = instance.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return tuple(bbox)
