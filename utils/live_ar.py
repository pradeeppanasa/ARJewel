"""
WebRTC video processor for real-time jewellery AR.
Detects face landmarks + head pose each frame, perspective-warps jewellery, composites.
"""

import math
import threading

import av
import cv2
import numpy as np
from PIL import Image

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    class VideoProcessorBase:  # stub so module loads without streamlit-webrtc
        pass

from utils.overlay import overlay_earrings, overlay_necklace


def _load_img(path) -> Image.Image | None:
    if not path:
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def _warp_for_pose(
    img: Image.Image,
    yaw: float,
    pitch: float,
    roll: float,
    kind: str = "earring",
) -> Image.Image:
    """
    Perspective-warp a flat jewellery PNG to match head orientation.

    Earring  — horizontal squish by cos(yaw) simulates face turning left/right.
    Necklace — vertical scale by cos(pitch) simulates nodding up/down.
    Both     — roll tilts the piece in sync with the head tilt.
    """
    img = img.convert("RGBA")
    w, h = img.size

    if kind == "earring":
        # As face turns, the earring foreshortens horizontally
        cos_yaw = max(0.15, abs(math.cos(math.radians(yaw))))
        new_w   = max(1, int(w * cos_yaw))
        img     = img.resize((new_w, h), Image.LANCZOS)
    else:
        # As face tilts up/down, the necklace foreshortens vertically
        cos_pitch = max(0.4, 1.0 - abs(pitch) / 120.0)
        new_h     = max(1, int(h * cos_pitch))
        img       = img.resize((w, new_h), Image.LANCZOS)

    # Roll: tilt the jewellery with the head
    if abs(roll) > 1.5:
        img = img.rotate(-roll, expand=True, resample=Image.BICUBIC)

    return img


class JewelleryARProcessor(VideoProcessorBase):
    """
    streamlit-webrtc processor: MediaPipe face detection + pose-aware overlay.

    Attributes are written from the Streamlit main thread after ctx is created;
    recv() runs in the WebRTC worker thread.  A threading.Lock guards the shared
    settings snapshot so there is no data race.
    """

    def __init__(self):
        self._lock     = threading.Lock()
        self._detector = None   # created lazily on first recv() call

        # Settings — written by main thread, read by recv() thread
        self.category      = "Earring"
        self.earring_path  = None
        self.necklace_path = None
        self.size_factor   = 1.0
        self.v_offset_ear  = 0
        self.h_offset_ear  = 0
        self.v_offset_nec  = 0
        self.h_offset_nec  = 0
        self.opacity       = 1.0
        self.is_pair       = False

        # Image cache: avoid re-loading the same file every frame
        self._ear_cache = (None, None)   # (path, PIL image)
        self._nec_cache = (None, None)

    # ── image helpers ─────────────────────────────────────────────────────────

    def _get_ear(self) -> Image.Image | None:
        p = self.earring_path
        if p != self._ear_cache[0]:
            self._ear_cache = (p, _load_img(p))
        return self._ear_cache[1]

    def _get_nec(self) -> Image.Image | None:
        p = self.necklace_path
        if p != self._nec_cache[0]:
            self._nec_cache = (p, _load_img(p))
        return self._nec_cache[1]

    # ── WebRTC entry point ────────────────────────────────────────────────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            out_pil = self._process(img_rgb)
        except Exception:
            out_pil = Image.fromarray(img_rgb)

        out_bgr = cv2.cvtColor(np.array(out_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    # ── per-frame processing ──────────────────────────────────────────────────

    def _process(self, img_rgb: np.ndarray) -> Image.Image:
        # Lazy init of detector (runs in recv() thread so it stays on one thread)
        if self._detector is None:
            from utils.head_pose import FacePoseDetector
            self._detector = FacePoseDetector()

        # Snapshot settings under lock (fast — no I/O)
        with self._lock:
            cat  = self.category
            ear  = self._get_ear()
            nec  = self._get_nec()
            sf   = self.size_factor
            voe  = self.v_offset_ear
            hoe  = self.h_offset_ear
            von  = self.v_offset_nec
            hon  = self.h_offset_nec
            op   = self.opacity
            pair = self.is_pair

        # Detect face + pose (slow MediaPipe call — outside lock)
        landmarks, yaw, pitch, roll = self._detector.detect(img_rgb)
        base = Image.fromarray(img_rgb)
        if landmarks is None:
            return base

        if cat in ("Earring", "Both") and ear:
            warped = _warp_for_pose(ear, yaw, pitch, roll, kind="earring")
            base   = overlay_earrings(base, warped, landmarks, sf, voe, hoe, op, pair)

        if cat in ("Necklace", "Both") and nec:
            warped = _warp_for_pose(nec, yaw, pitch, roll, kind="necklace")
            base   = overlay_necklace(base, warped, landmarks, sf, von, hon, op)

        return base
