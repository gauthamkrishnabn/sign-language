"""
models.py
---------
MediaPipe's current Python API (Tasks API) ships without bundled models --
you point it at a small `.task` model file. This downloads the two files
we need (hand + pose landmark detectors) once and caches them locally, so
the app doesn't need internet access on every run (only the first time,
or if you predownload them -- see README "Offline / no-internet setup").
"""

import os
import urllib.request

CACHE_DIR = os.environ.get("SIGN_MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def get_model_path(name):
    """Returns a local path to the requested model, downloading it into
    CACHE_DIR first if it isn't already there."""
    if name not in MODEL_URLS:
        raise ValueError(f"Unknown model '{name}'. Known: {list(MODEL_URLS)}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = MODEL_URLS[name]
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Could not download required model '{name}' from {url}. "
            f"If this machine has no internet access, download it manually "
            f"on another machine and place it at {path}. Original error: {e}"
        ) from e
    return path


def ensure_models():
    """Downloads/caches every model the app needs. Call once at startup so
    failures surface early with a clear message instead of mid-video."""
    return {name: get_model_path(name) for name in MODEL_URLS}
