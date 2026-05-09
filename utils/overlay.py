"""
PIL/OpenCV utilities for compositing jewellery PNGs onto photos.
"""

import math
import numpy as np
from PIL import Image, ImageFilter


def paste_with_alpha(base: Image.Image, overlay: Image.Image, position: tuple[int, int]) -> Image.Image:
    base = base.copy().convert("RGBA")
    overlay = overlay.convert("RGBA")
    x, y = position
    bw, bh = base.size
    ow, oh = overlay.size

    src_x0 = max(0, -x);  src_y0 = max(0, -y)
    src_x1 = min(ow, bw - x);  src_y1 = min(oh, bh - y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return base

    overlay_crop = overlay.crop((src_x0, src_y0, src_x1, src_y1))
    base.paste(overlay_crop, (max(0, x), max(0, y)), overlay_crop)
    return base


def rotate_image(img: Image.Image, angle_deg: float) -> Image.Image:
    return img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)


def resize_image(img: Image.Image, new_w: int, new_h: int) -> Image.Image:
    return img.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def _tight_crop(img: Image.Image) -> Image.Image:
    """Crop away transparent borders using getbbox() so sizing is based on actual jewellery pixels."""
    img = img.convert("RGBA")
    bbox = img.getbbox()          # (left, upper, right, lower) or None if all transparent
    return img.crop(bbox) if bbox else img


def _soften_edges(img: Image.Image, radius: float = 1.5) -> Image.Image:
    """Blur only the alpha channel border so earring edges blend naturally into skin."""
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a))


# ─── Earring placement ────────────────────────────────────────────────────────

_EARLOBE_OFFSET_PX = 15   # px below MediaPipe tragion landmark


def overlay_earring(
    base_img: Image.Image,
    earring_img: Image.Image,
    ear_lobe: tuple[int, int],
    ear_top: tuple[int, int],
    face_width: int,
    size_factor: float = 1.0,
    v_offset: int = 0,
    h_offset: int = 0,
    opacity: float = 1.0,
    side: str = "left",
) -> Image.Image:
    """
    Place one earring at the earlobe.
    Steps:
      1. tight-crop to actual jewellery bounding box (getbbox)
      2. normalise width to 17.6 % of face_width × size_factor
      3. position: inner edge of earring at landmark x (ears are outside face contour),
         y = landmark + 15 px + v_offset
    """
    # 1 — crop to actual content
    earring = _tight_crop(earring_img)

    # 2 — normalise to consistent size relative to face
    target_w = int(face_width * 0.176 * size_factor)
    orig_w, orig_h = earring.size
    target_h = int(target_w * (orig_h / max(orig_w, 1)))
    earring = resize_image(earring, target_w, target_h)

    # Slight tilt correction (dampened so earrings stay mostly upright)
    dx = ear_lobe[0] - ear_top[0]
    dy = ear_lobe[1] - ear_top[1]
    angle = math.degrees(math.atan2(dx, dy)) * 0.3
    if side == "right":
        angle = -angle
    earring = rotate_image(earring, angle)
    earring = apply_opacity(earring, opacity)

    # Fix 1 — soft edge blending: blur alpha border so edges melt into skin
    earring = _soften_edges(earring, radius=1.5)

    # 3 — position: landmark 234/454 is at the face-ear junction (face contour).
    # Earrings sit OUTSIDE the face boundary, so place inner edge ~15% inside landmark.
    # Fine-tune: 5px toward face center + 5px upward for better earlobe alignment.
    ew, eh = earring.size
    overlap = int(ew * 0.15)
    if side == "left":
        x = ear_lobe[0] - ew + overlap - h_offset + 5   # +5 toward center
    else:
        x = ear_lobe[0] - overlap + h_offset - 5         # -5 toward center
    y = ear_lobe[1] + _EARLOBE_OFFSET_PX + v_offset - 5  # 5px upward

    # Fix 2 — drop shadow: blurred black silhouette offset 2px down for depth
    _r, _g, _b, alpha = earring.split()
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=3))
    shadow_alpha = shadow_alpha.point(lambda p: int(p * 0.30))
    shadow = Image.new("RGBA", earring.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    base_img = paste_with_alpha(base_img, shadow, (x + 2, y + 2))

    return paste_with_alpha(base_img, earring, (x, y))


def _split_pair(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split a side-by-side pair product image into left / right halves at the natural gap."""
    arr   = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    w     = img.size[0]
    lo, hi = w // 4, 3 * w // 4
    col_sums = alpha[:, lo:hi].sum(axis=0)
    gap_col  = int(col_sums.argmin()) + lo
    return img.crop((0, 0, gap_col, img.size[1])), img.crop((gap_col, 0, w, img.size[1]))


def overlay_earrings(
    base_img: Image.Image,
    earring_img: Image.Image,
    landmarks: dict,
    size_factor: float = 1.0,
    v_offset: int = 0,
    h_offset: int = 0,
    opacity: float = 1.0,
    is_pair: bool = False,
) -> Image.Image:
    """
    Overlay earrings on both ears.
    is_pair=True  → split image at natural gap; left half → left ear, right half → right ear.
    is_pair=False → same image mirrored onto each ear.
    """
    earring = earring_img.convert("RGBA")
    if is_pair:
        left_img, right_img = _split_pair(earring)
    else:
        left_img  = earring
        right_img = earring.transpose(Image.FLIP_LEFT_RIGHT)

    result = overlay_earring(
        base_img, left_img,
        landmarks["left_ear"], landmarks["left_ear_top"],
        landmarks["face_width"], size_factor, v_offset, h_offset, opacity, side="left",
    )
    result = overlay_earring(
        result, right_img,
        landmarks["right_ear"], landmarks["right_ear_top"],
        landmarks["face_width"], size_factor, v_offset, h_offset, opacity, side="right",
    )
    return result


# ─── Necklace placement ───────────────────────────────────────────────────────

def overlay_necklace(
    base_img: Image.Image,
    necklace_img: Image.Image,
    landmarks: dict,
    size_factor: float = 1.0,
    v_offset: int = 0,
    opacity: float = 1.0,
) -> Image.Image:
    necklace = _tight_crop(necklace_img)

    target_w = int(landmarks["face_width"] * 1.10 * size_factor)
    orig_w, orig_h = necklace.size
    target_h = int(target_w * (orig_h / max(orig_w, 1)))
    necklace = resize_image(necklace, target_w, target_h)
    necklace = apply_opacity(necklace, opacity)

    nx, ny = landmarks["neck_center"]
    nw, nh = necklace.size
    return paste_with_alpha(base_img, necklace, (nx - nw // 2, ny + v_offset))
