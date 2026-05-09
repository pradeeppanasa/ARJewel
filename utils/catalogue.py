"""
Auto-discovers jewellery images from assets/jewellery/earrings/ and necklaces/.

Each catalogue entry:
  {
    "id":       str,          # file stem
    "name":     str,          # display label
    "path":     Path,         # original file (for gallery preview)
    "nobg_path": Path | None, # transparent PNG path (set lazily on first overlay)
  }
"""

from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets" / "jewellery"
_EXTS      = {".png", ".jpg", ".jpeg"}


def _label(stem: str) -> str:
    # Strip _nobg suffix if present, then humanise
    stem = stem.replace("_nobg", "")
    return stem.replace("_", " ").title()


def load_catalogue() -> dict[str, list[dict]]:
    """
    Returns:
        {
          "Earring":  [{"id", "name", "path", "nobg_path"}, ...],
          "Necklace": [...],
        }
    Skips *_nobg.png files from the listing (they are internal cache files).
    """
    cat = {}
    for category, folder in [("Earring", "earrings"), ("Necklace", "necklaces")]:
        folder_path = ASSETS_DIR / folder
        if not folder_path.exists():
            cat[category] = []
            continue

        items = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in _EXTS and "_nobg" not in p.stem
        )

        cat[category] = [
            {
                "id":        p.stem,
                "name":      _label(p.stem),
                "path":      p,
                "nobg_path": None,   # populated lazily in app.py on first use
            }
            for p in items
        ]
    return cat
