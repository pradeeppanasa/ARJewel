"""
MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10) landmark detection.
Auto-downloads face_landmarker.task model (~6 MB) on first run.
"""

import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path(__file__).parent.parent / "assets" / "face_landmarker.task"


def _ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading FaceLandmarker model to {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# Landmark indices
LEFT_EAR_BOUNDARY  = 234   # left face boundary near ear (gives correct x)
RIGHT_EAR_BOUNDARY = 454   # right face boundary near ear
LEFT_EAR_TOP       = 227
RIGHT_EAR_TOP      = 447
CHIN_IDX           = 152
FOREHEAD_IDX       = 10
LEFT_CHEEK_IDX     = 356   # outer left cheek (for face width)
RIGHT_CHEEK_IDX    = 127   # outer right cheek


def detect_landmarks(image_rgb: np.ndarray) -> dict | None:
    _ensure_model()

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Upscale tiny images so MediaPipe has enough pixels to find the face.
    h, w = image_rgb.shape[:2]
    if w < 300:
        import cv2
        scale = 300 / w
        image_rgb = cv2.resize(image_rgb, (300, int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    with mp_vision.FaceLandmarker.create_from_options(opts) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result   = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = image_rgb.shape[:2]
    lm = result.face_landmarks[0]

    def px(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    forehead    = px(FOREHEAD_IDX)
    chin        = px(CHIN_IDX)
    left_cheek  = px(LEFT_CHEEK_IDX)
    right_cheek = px(RIGHT_CHEEK_IDX)
    left_top    = px(LEFT_EAR_TOP)
    right_top   = px(RIGHT_EAR_TOP)

    face_width  = abs(left_cheek[0] - right_cheek[0])
    face_height = abs(chin[1] - forehead[1])

    # ── Ear tragion position — raw MediaPipe landmarks 234 (left) and 454 (right)
    # overlay.py applies the 15 px downward offset to reach the earlobe
    left_ear  = px(LEFT_EAR_BOUNDARY)
    right_ear = px(RIGHT_EAR_BOUNDARY)

    # ── Neck center ────────────────────────────────────────────────────────────
    # Necklace: chin bottom + 20px, centred on image width
    neck_center = (w // 2, chin[1] + 20)

    return {
        "left_ear":      left_ear,
        "right_ear":     right_ear,
        "left_ear_top":  left_top,
        "right_ear_top": right_top,
        "chin":          chin,
        "neck_center":   neck_center,
        "face_width":    face_width,
        "face_height":   face_height,
        "image_h":       h,
        "image_w":       w,
    }
