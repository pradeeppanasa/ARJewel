"""
MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10) landmark detection.
Auto-downloads face_landmarker.task model (~6 MB) on first run.
"""

import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image as _PILImage

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

    orig_h, orig_w = image_rgb.shape[:2]

    # Upscale small images using PIL (no cv2 dependency here).
    # MediaPipe returns normalised (0-1) coords so landmarks always map back
    # to orig_h / orig_w regardless of what size was used for detection.
    if orig_w < 640:
        scale     = 640 / orig_w
        pil_img   = _PILImage.fromarray(image_rgb).resize(
            (640, int(orig_h * scale)), _PILImage.LANCZOS
        )
        detect_img = np.ascontiguousarray(np.array(pil_img), dtype=np.uint8)
    else:
        detect_img = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    with mp_vision.FaceLandmarker.create_from_options(opts) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_img)
        result   = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    # Use original dimensions — landmarks are normalised so scale doesn't matter
    h, w = orig_h, orig_w
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

    left_ear  = px(LEFT_EAR_BOUNDARY)
    right_ear = px(RIGHT_EAR_BOUNDARY)

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
