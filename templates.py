"""
templates.py
------------
Manages the "vocabulary" the app can recognize: a folder of example
landmark sequences, each labeled with the English word it represents.

Since we don't have a large pre-trained sign-classification model,
recognition works by nearest-neighbor matching (see recognizer.py)
against these examples. Add more examples per word for better accuracy.

Storage layout on disk:
    templates/
        manifest.json          -> {"hello": ["hello_0.npy", "hello_1.npy"], ...}
        hello_0.npy
        hello_1.npy
        ...
"""

import json
import os
import re
import numpy as np

MANIFEST_NAME = "manifest.json"


def _manifest_path(template_dir):
    return os.path.join(template_dir, MANIFEST_NAME)


def _slugify(word):
    return re.sub(r"[^a-z0-9]+", "_", word.strip().lower()).strip("_")


# Public alias -- other modules (e.g. wlasl import UI) need this to look
# up existing vocabulary entries without reaching into a "private" name.
slugify = _slugify


def load_manifest(template_dir):
    path = _manifest_path(template_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(template_dir, manifest):
    os.makedirs(template_dir, exist_ok=True)
    with open(_manifest_path(template_dir), "w") as f:
        json.dump(manifest, f, indent=2)


def add_template(template_dir, gloss, landmark_seq):
    """Save one labeled example sequence for a word/gloss."""
    os.makedirs(template_dir, exist_ok=True)
    manifest = load_manifest(template_dir)
    slug = _slugify(gloss)
    existing = manifest.get(slug, [])
    fname = f"{slug}_{len(existing)}.npy"
    np.save(os.path.join(template_dir, fname), landmark_seq)
    existing.append(fname)
    manifest[slug] = existing
    save_manifest(template_dir, manifest)
    return fname


def remove_word(template_dir, gloss):
    manifest = load_manifest(template_dir)
    slug = _slugify(gloss)
    for fname in manifest.pop(slug, []):
        fpath = os.path.join(template_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    save_manifest(template_dir, manifest)


def load_templates(template_dir):
    """Returns {gloss: [np.ndarray, np.ndarray, ...]} for every word
    currently in the library."""
    manifest = load_manifest(template_dir)
    out = {}
    for slug, files in manifest.items():
        seqs = []
        for fname in files:
            fpath = os.path.join(template_dir, fname)
            if os.path.exists(fpath):
                seqs.append(np.load(fpath))
        if seqs:
            out[slug] = seqs
    return out


def vocab_size(template_dir):
    return len(load_manifest(template_dir))
