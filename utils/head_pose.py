"""
Real-time head pose estimation from MediaPipe FaceLandmarker + OpenCV solvePnP.
Provides a long-lived FacePoseDetector class for WebRTC use (create once per thread).
"""

import cv2
import numpy as np

from utils.detector import (
    _ensure_model, MODEL_PATH,
    LEFT_EAR_BOUNDARY, RIGHT_EAR_BOUNDARY,
    LEFT_EAR_TOP, RIGHT_EAR_TOP,
    CHIN_IDX, FOREHEAD_IDX,
    LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX,
)

# Generic 3-D face model (canonical face, in mm) matched to MediaPipe landmark indices
_FACE_3D = np.array([
    [   0.0,    0.0,    0.0],   # Nose tip        — lm 1
    [   0.0, -330.0,  -65.0],   # Chin            — lm 152
    [-225.0,  170.0, -135.0],   # Left eye outer  — lm 33
    [ 225.0,  170.0, -135.0],   # Right eye outer — lm 263
    [-150.0, -150.0, -125.0],   # Left mouth      — lm 61
    [ 150.0, -150.0, -125.0],   # Right mouth     — lm 291
], dtype=np.float64)

_POSE_LM = [1, 152, 33, 263, 61, 291]


class FacePoseDetector:
    """
    Persistent MediaPipe FaceLandmarker wrapper.
    Create once per thread, call detect() per frame.
    """

    def __init__(self):
        _ensure_model()
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.05,
            min_face_presence_confidence=0.05,
            min_tracking_confidence=0.05,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image_rgb: np.ndarray) -> tuple:
        """
        Returns (landmarks_dict, yaw_deg, pitch_deg, roll_deg) or (None, 0, 0, 0).
        landmarks_dict keys match utils.detector.detect_landmarks output.
        """
        from PIL import Image as _PILImage

        orig_h, orig_w = image_rgb.shape[:2]

        # Upscale to ≥480 px wide for reliable landmark detection on small frames
        target_w = max(orig_w, 480)
        scale    = target_w / orig_w
        det_img  = np.ascontiguousarray(
            np.array(
                _PILImage.fromarray(image_rgb).resize(
                    (target_w, int(orig_h * scale)), _PILImage.LANCZOS
                )
            ),
            dtype=np.uint8,
        )

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=det_img)
        result   = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, 0.0, 0.0, 0.0

        h, w = orig_h, orig_w
        lm   = result.face_landmarks[0]

        def px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        forehead    = px(FOREHEAD_IDX)
        chin_pt     = px(CHIN_IDX)
        left_cheek  = px(LEFT_CHEEK_IDX)
        right_cheek = px(RIGHT_CHEEK_IDX)
        face_width  = abs(left_cheek[0] - right_cheek[0])

        landmarks = {
            "left_ear":      px(LEFT_EAR_BOUNDARY),
            "right_ear":     px(RIGHT_EAR_BOUNDARY),
            "left_ear_top":  px(LEFT_EAR_TOP),
            "right_ear_top": px(RIGHT_EAR_TOP),
            "chin":          chin_pt,
            "neck_center":   (w // 2, chin_pt[1] + 20),
            "face_width":    face_width,
            "face_height":   abs(chin_pt[1] - forehead[1]),
            "image_h":       h,
            "image_w":       w,
        }

        yaw, pitch, roll = _head_pose_solvepnp(lm, h, w)
        return landmarks, yaw, pitch, roll

    def close(self):
        self._landmarker.close()


def _head_pose_solvepnp(lm, image_h: int, image_w: int) -> tuple:
    """Estimate yaw, pitch, roll (degrees) using OpenCV solvePnP."""
    face_2d = np.array(
        [[lm[i].x * image_w, lm[i].y * image_h] for i in _POSE_LM],
        dtype=np.float64,
    )
    focal = float(image_w)
    cam   = np.array([
        [focal, 0.0,   image_w / 2.0],
        [0.0,   focal, image_h / 2.0],
        [0.0,   0.0,   1.0          ],
    ], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, face_2d, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
        roll  = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))

    return yaw, pitch, roll
