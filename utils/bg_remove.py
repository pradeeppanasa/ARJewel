"""
Background removal for jewellery JPG images using rembg (U2Net model).
Processed PNGs are cached next to the source file as <stem>_nobg.png.
Model (~170 MB) is downloaded automatically on first use.
"""

from pathlib import Path
from PIL import Image


def get_nobg_path(source: Path) -> Path:
    return source.with_name(source.stem + "_nobg.png")


def remove_background(source: Path, force: bool = False) -> Path:
    """
    Remove background from `source` image.
    Returns path to the transparent PNG (cached after first run).
    """
    out = get_nobg_path(source)
    if out.exists() and not force:
        return out

    from rembg import remove

    img    = Image.open(source).convert("RGBA")
    result = remove(img)
    result.save(out, format="PNG")
    return out


def ensure_nobg(source: Path) -> Path:
    """Return nobg PNG path, creating it if needed. For JPG/JPEG inputs only."""
    if source.suffix.lower() in (".jpg", ".jpeg"):
        return remove_background(source)
    return source   # PNG already assumed transparent
