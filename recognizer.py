"""
recognizer.py
-------------
Classifies a landmark sequence (one candidate sign segment) by comparing
it against labeled example sequences in the template library, using
Dynamic Time Warping (DTW). DTW lets two sequences of different lengths
and slightly different speeds be compared frame-for-frame along the best
alignment path -- important because no two people (or takes) sign a word
at exactly the same speed.

This is a k-NN-style classifier: no training step required, just examples.
Accuracy scales directly with how many/varied examples you add per word.
"""

import numpy as np
from scipy.spatial.distance import cdist


def dtw_distance(seq_a, seq_b):
    """Standard O(n*m) DTW over per-frame Euclidean distance, normalized
    by path length so longer sequences aren't unfairly penalized."""
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    cost = cdist(seq_a, seq_b, metric="euclidean")
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = cost[i - 1, j - 1] + best_prev

    path_len = n + m
    return dp[n, m] / path_len


def recognize_segment(segment_frames, templates, top_k=3, max_examples_per_word=6):
    """
    Args:
        segment_frames: (n, D) landmark array for one candidate sign
        templates: {gloss: [example_seq, ...]} from templates.load_templates
        top_k: how many candidate matches to return
        max_examples_per_word: cap how many stored examples are checked
            per word, to keep recognition fast as the library grows

    Returns:
        list of (gloss, distance) sorted best-first, or [] if no templates
    """
    if not templates:
        return []

    scores = []
    for gloss, examples in templates.items():
        best = min(
            dtw_distance(segment_frames, ex)
            for ex in examples[:max_examples_per_word]
        )
        scores.append((gloss, best))

    scores.sort(key=lambda x: x[1])
    return scores[:top_k]


def recognize_video(frames, segments, templates, confidence_gap=0.15):
    """
    Runs recognition over every segment found by segmentation.segment_signs.

    Returns:
        list of dicts: {start_time, end_time, gloss, distance, confident}
        "confident" is a rough heuristic: True if the best match is
        meaningfully closer than the second-best match.
    """
    results = []
    for (s, e, t_start, t_end) in segments:
        seg = frames[s : e + 1]
        matches = recognize_segment(seg, templates)
        if not matches:
            results.append(
                {"start_time": t_start, "end_time": t_end, "gloss": None,
                 "distance": None, "confident": False, "alternatives": []}
            )
            continue

        best_gloss, best_dist = matches[0]
        confident = True
        if len(matches) > 1:
            second_dist = matches[1][1]
            if second_dist > 0:
                rel_gap = (second_dist - best_dist) / second_dist
                confident = rel_gap > confidence_gap

        results.append({
            "start_time": t_start,
            "end_time": t_end,
            "gloss": best_gloss,
            "distance": best_dist,
            "confident": confident,
            "alternatives": matches[1:],
        })
    return results
