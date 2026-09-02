"""
segmentation.py
----------------
Splits a continuous stream of landmark frames into candidate "sign"
segments. Continuous signing has natural pauses/slowdowns between
words (hands resting, transitioning) -- we detect those low-motion
gaps and cut there. This is a simple heuristic, not a learned model,
but works reasonably well for clearly-signed, deliberate video.
"""

import numpy as np

N_HAND = 21
HAND_DIMS = N_HAND * 3


def _motion_energy(frames):
    """Per-frame motion = how much the two hands moved since the last
    frame. Pose points are ignored here since hands carry the signal."""
    hands = frames[:, : 2 * HAND_DIMS]
    if len(hands) < 2:
        return np.zeros(len(hands))
    diffs = np.linalg.norm(np.diff(hands, axis=0), axis=1)
    return np.concatenate([[0.0], diffs])


def segment_signs(frames, timestamps, motion_threshold=0.02,
                   min_gap_frames=3, min_segment_frames=4,
                   pad_frames=1):
    """
    Args:
        frames: (n, D) landmark array from landmarks.extract_landmarks_from_video
        timestamps: (n,) seconds per frame
        motion_threshold: energy below this counts as "still"
        min_gap_frames: how many consecutive still frames end a segment
        min_segment_frames: discard segments shorter than this (likely noise)
        pad_frames: expand each segment slightly so we don't clip the sign

    Returns:
        list of (start_idx, end_idx, start_time, end_time) tuples
    """
    if len(frames) == 0:
        return []

    energy = _motion_energy(frames)
    moving = energy > motion_threshold

    segments = []
    seg_start = None
    still_run = 0

    for i, m in enumerate(moving):
        if m:
            if seg_start is None:
                seg_start = i
            still_run = 0
        else:
            if seg_start is not None:
                still_run += 1
                if still_run >= min_gap_frames:
                    seg_end = i - still_run
                    segments.append((seg_start, seg_end))
                    seg_start = None
                    still_run = 0

    if seg_start is not None:
        segments.append((seg_start, len(frames) - 1))

    out = []
    for s, e in segments:
        s = max(0, s - pad_frames)
        e = min(len(frames) - 1, e + pad_frames)
        if e - s + 1 >= min_segment_frames:
            out.append((s, e, float(timestamps[s]), float(timestamps[e])))

    return out
