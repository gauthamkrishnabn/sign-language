"""
batch_add.py
------------
Lets you build vocabulary much faster than one-clip-per-word: record a
single video where you sign a list of words in order, with a brief
pause between each one, then match each auto-detected segment up to its
word from a list you provide (in the same order).

Two ways to use it:
  - From Streamlit: the "Batch Add" tab in app.py calls
    `preview_segments()` to show what was detected before you confirm,
    then `commit_segments()` to actually save them as templates.
  - From the command line: see `main()` below.
"""

import argparse
import sys

from landmarks import extract_landmarks_from_video
from segmentation import segment_signs
import templates as tpl


def preview_segments(video_path, sample_fps=15, motion_threshold=0.02,
                      min_gap_frames=3, min_segment_frames=4):
    """
    Runs landmark extraction + segmentation on a video and returns what
    it found, WITHOUT saving anything -- lets you sanity-check that the
    number of detected segments matches your word list before committing.

    Returns:
        frames: (n, D) landmark array (pass to commit_segments -- avoids
                 re-running extraction)
        segments: list of (start_idx, end_idx, start_time, end_time)
    """
    frames, timestamps = extract_landmarks_from_video(video_path, sample_fps=sample_fps)
    if len(frames) == 0:
        raise ValueError("Couldn't read any frames from that video.")

    segments = segment_signs(
        frames, timestamps,
        motion_threshold=motion_threshold,
        min_gap_frames=min_gap_frames,
        min_segment_frames=min_segment_frames,
    )
    return frames, segments


def commit_segments(template_dir, frames, segments, words):
    """
    Saves each (segment, word) pair as a template. `words` must be the
    same length as `segments`, in the same order the signs occur in
    the video.

    Returns: list of (word, saved_filename) for what was added.
    """
    if len(words) != len(segments):
        raise ValueError(
            f"Got {len(segments)} detected segment(s) but {len(words)} word(s) "
            f"-- these must match 1:1. Adjust the motion threshold and re-preview, "
            f"or fix your word list."
        )

    saved = []
    for (s, e, _, _), word in zip(segments, words):
        seg_frames = frames[s : e + 1]
        fname = tpl.add_template(template_dir, word, seg_frames)
        saved.append((word, fname))
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Batch-add sign templates from one video containing "
                     "multiple signs performed in sequence, with pauses between them."
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument(
        "words",
        help="Comma-separated words IN THE ORDER they're signed in the video, "
             "e.g. \"hello,thank you,water,please\"",
    )
    parser.add_argument("--template-dir", default="templates")
    parser.add_argument("--sample-fps", type=int, default=15)
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--min-gap-frames", type=int, default=3)
    parser.add_argument("--min-segment-frames", type=int, default=4)
    args = parser.parse_args()

    words = [w.strip() for w in args.words.split(",") if w.strip()]

    print(f"Extracting landmarks and detecting sign boundaries in {args.video}...")
    frames, segments = preview_segments(
        args.video,
        sample_fps=args.sample_fps,
        motion_threshold=args.motion_threshold,
        min_gap_frames=args.min_gap_frames,
        min_segment_frames=args.min_segment_frames,
    )

    print(f"\nDetected {len(segments)} segment(s), you gave {len(words)} word(s):")
    for i, (s, e, t0, t1) in enumerate(segments):
        label = words[i] if i < len(words) else "??? (no matching word)"
        print(f"  [{i}] {t0:.2f}s - {t1:.2f}s  ->  {label}")

    if len(segments) != len(words):
        print(
            "\nCounts don't match -- nothing was saved. Try adjusting "
            "--motion-threshold (lower = more sensitive, catches smaller "
            "movements; higher = requires bigger movements) and re-run, "
            "or fix your word list.",
            file=sys.stderr,
        )
        sys.exit(1)

    confirm = input("\nSave these as templates? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled, nothing saved.")
        return

    saved = commit_segments(args.template_dir, frames, segments, words)
    print(f"\nSaved {len(saved)} template(s) to '{args.template_dir}':")
    for word, fname in saved:
        print(f"  {word} -> {fname}")


if __name__ == "__main__":
    main()
