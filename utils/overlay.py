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


def _add_specular_highlight(img: Image.Image, intensity: float = 0.45) -> Image.Image:
    """Add a top-to-bottom white gradient highlight + bottom darkening for a dramatic 3D sheen effect."""
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.float32)
    h = arr.shape[0]

    # Top highlight: bright specular shine on upper portion
    grad = np.clip(1.0 - np.arange(h) / max(h * 0.6, 1), 0.0, 1.0).reshape(h, 1)
    shine = grad * intensity * arr[:, :, 3]
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shine, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + shine, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shine, 0, 255)

    # Bottom darkening: subtle tint on lower 40% for contrast / depth
    dark_intensity = 0.15
    rows = np.arange(h, dtype=np.float32)
    cutoff = h * 0.60
    dark_grad = np.clip((rows - cutoff) / max(h * 0.40, 1), 0.0, 1.0).reshape(h, 1)
    shadow = dark_grad * dark_intensity * arr[:, :, 3]
    arr[:, :, 0] = np.clip(arr[:, :, 0] - shadow, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] - shadow, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - shadow, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), "RGBA")


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
    # 1 — tight-crop transparent padding (getbbox)
    earring = _tight_crop(earring_img)

    # 2 — size: 9% of face width × size_factor
    target_w = int(face_width * 0.09 * size_factor)
    orig_w, orig_h = earring.size
    target_h = int(target_w * (orig_h / max(orig_w, 1)))
    earring = resize_image(earring, target_w, target_h)

    # 5 — flip right earring so it mirrors left naturally
    if side == "right":
        earring = earring.transpose(Image.FLIP_LEFT_RIGHT)

    earring = apply_opacity(earring, opacity)
    earring = _soften_edges(earring, radius=1.5)
    earring = _add_specular_highlight(earring, intensity=0.45)

    # position: centre on landmark x, 10 px below landmark y
    ew, eh = earring.size
    if side == "left":
        x = ear_lobe[0] - ew // 2 - h_offset
    else:
        x = ear_lobe[0] - ew // 2 + h_offset
    y = ear_lobe[1] + 10 + v_offset

    # drop shadow — larger blur + offset for 3D depth
    _r, _g, _b, alpha = earring.split()
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=5))
    shadow_alpha = shadow_alpha.point(lambda p: int(p * 0.65))
    shadow = Image.new("RGBA", earring.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    base_img = paste_with_alpha(base_img, shadow, (x + 3, y + 3))

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
        right_img = earring   # flip handled inside overlay_earring for side="right"

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
    h_offset: int = 0,
    opacity: float = 1.0,
) -> Image.Image:
    necklace = _tight_crop(necklace_img)

    target_w = int(landmarks["face_width"] * 1.20 * size_factor)
    orig_w, orig_h = necklace.size
    target_h = int(target_w * (orig_h / max(orig_w, 1)))
    necklace = resize_image(necklace, target_w, target_h)
    necklace = apply_opacity(necklace, opacity)
    necklace = _soften_edges(necklace, radius=1.5)
    necklace = _add_specular_highlight(necklace, intensity=0.38)

    nx, ny = landmarks["neck_center"]
    nw, nh = necklace.size
    x = nx - nw // 2 + h_offset
    y = ny + v_offset

    # drop shadow for depth
    _r, _g, _b, nec_alpha = necklace.split()
    shadow_alpha = nec_alpha.filter(ImageFilter.GaussianBlur(radius=6))
    shadow_alpha = shadow_alpha.point(lambda p: int(p * 0.60))
    shadow = Image.new("RGBA", necklace.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    base_img = paste_with_alpha(base_img, shadow, (x + 4, y + 4))

    return paste_with_alpha(base_img, necklace, (x, y))
