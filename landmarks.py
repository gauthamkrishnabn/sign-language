"""
landmarks.py
------------
Extracts hand + pose landmarks from a video using MediaPipe's Tasks API
(HandLandmarker + PoseLandmarker -- the current, supported API; the older
`mp.solutions.holistic` API has been removed from recent mediapipe
releases), and normalizes them so the same sign looks (roughly) the same
regardless of where the signer is standing or how big they appear on
camera.

Each frame is turned into a single flat numpy vector:
    [left_hand(21*3), right_hand(21*3), upper_pose(11*3)]
Missing landmarks (e.g. a hand that's out of frame) are filled with zeros.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import models

# Upper-body pose landmarks we care about (MediaPipe Pose indices).
# Shoulders/elbows/wrists/hips/face give us posture + arm position
# without wasting dimensions on legs/feet, which don't matter for signing.
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 0, 9, 10]

N_HAND = 21
N_POSE = len(POSE_IDX)
VECTOR_SIZE = (N_HAND + N_HAND + N_POSE) * 3

_landmarkers = {}  # lazy-initialized, cached across calls in the same process


def _get_landmarkers():
    """Creates (once) and returns the (hand_landmarker, pose_landmarker)
    pair, running in VIDEO mode so MediaPipe can use frame-to-frame
    tracking instead of redetecting from scratch every frame."""
    if "hand" not in _landmarkers:
        paths = models.ensure_models()

        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["pose_landmarker_lite.task"]),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarkers["hand"] = mp_vision.HandLandmarker.create_from_options(hand_options)
        _landmarkers["pose"] = mp_vision.PoseLandmarker.create_from_options(pose_options)

    return _landmarkers["hand"], _landmarkers["pose"]


def _hand_arrays(hand_result):
    """Returns (left_hand_vec, right_hand_vec), each length N_HAND*3,
    zero-filled if that hand wasn't detected in this frame.

    Note: MediaPipe's handedness label assumes a mirrored/selfie-view
    image. For non-mirrored video the Left/Right label is the opposite
    of the signer's actual hand -- harmless here since we only need the
    same physical hand to land in the same vector slot consistently
    across frames, which handedness still gives us.
    """
    left = np.zeros(N_HAND * 3, dtype=np.float32)
    right = np.zeros(N_HAND * 3, dtype=np.float32)

    if not hand_result or not hand_result.hand_landmarks:
        return left, right

    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        vec = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32).flatten()
        label = handedness[0].category_name if handedness else None
        if label == "Left":
            left = vec
        elif label == "Right":
            right = vec

    return left, right


def _pose_array(pose_result):
    if not pose_result or not pose_result.pose_landmarks:
        return np.zeros(N_POSE * 3, dtype=np.float32)
    landmarks = pose_result.pose_landmarks[0]
    pts = [landmarks[i] for i in POSE_IDX]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32).flatten()


def _normalize_frame(vec):
    """Make the frame translation/scale invariant using the shoulders as
    an anchor, so it doesn't matter whether the signer is close to the
    camera or standing slightly left/right of center."""
    pose_start = (N_HAND + N_HAND) * 3
    pose = vec[pose_start:].reshape(N_POSE, 3)
    left_shoulder, right_shoulder = pose[0], pose[1]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = scale if scale > 1e-6 else 1.0

    out = vec.copy().reshape(-1, 3)
    out = (out - center) / scale
    return out.flatten()


def extract_landmarks_from_video(video_path, sample_fps=15, max_seconds=None,
                                  crop_box=None):
    """
    Args:
        crop_box: optional (x1, y1, x2, y2) in source-pixel coordinates.
            If given, each frame is cropped to this region before
            landmark detection -- useful for dataset clips (e.g. WLASL)
            that ship a signer bounding box, where the signer may
            otherwise be small/off-center in the raw frame.

    Returns:
        frames: np.ndarray of shape (n_frames, VECTOR_SIZE)
        timestamps: np.ndarray of shape (n_frames,) -- seconds from start
    """
    hand_landmarker, pose_landmarker = _get_landmarkers()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(src_fps / sample_fps))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    crop = None
    if crop_box is not None and frame_w and frame_h:
        x1, y1, x2, y2 = crop_box
        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(x1 + 1, min(int(x2), frame_w))
        y2 = max(y1 + 1, min(int(y2), frame_h))
        crop = (x1, y1, x2, y2)

    frames = []
    timestamps = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            t = idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break

            if crop is not None:
                x1, y1, x2, y2 = crop
                frame = frame[y1:y2, x1:x2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(t * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            left, right = _hand_arrays(hand_result)
            pose = _pose_array(pose_result)

            vec = np.concatenate([left, right, pose])
            vec = _normalize_frame(vec)

            frames.append(vec)
            timestamps.append(t)
        idx += 1
    cap.release()

    if not frames:
        return np.zeros((0, VECTOR_SIZE), dtype=np.float32), np.zeros(0)

    return np.stack(frames), np.array(timestamps)
