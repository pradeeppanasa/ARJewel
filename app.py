"""
Jewellery AR Try-On — Streamlit App

Stack:
  • Streamlit          — UI + visual jewellery gallery
  • MediaPipe          — ear / neck landmark detection
  • OpenCV + PIL       — local pixel-level overlay (fallback)
  • Flux Kontext Pro   — AI photorealistic overlay (BFL API)
  • Azure OpenAI GPT-4o — jewellery style recommendation
"""

import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.catalogue     import load_catalogue
from utils.detector      import detect_landmarks
from utils.overlay       import overlay_earrings, overlay_necklace
from utils.flux_overlay  import overlay_with_flux
from utils.recommender   import get_recommendation
from utils.bg_remove     import ensure_nobg

# ── helpers ────────────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def _bfl_key():
    return os.environ.get("BFL_API_KEY", "").strip() or None

def _azure_ready():
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_API_KEY"))


def resolve_overlay_image(item: dict) -> Image.Image:
    """
    Return a transparent-background RGBA image for overlay.
    For JPGs: runs rembg on first use, caches result as _nobg.png beside source.
    Shows a one-time spinner so the user knows it's processing.
    """
    from pathlib import Path
    if item.get("nobg_path") and Path(item["nobg_path"]).exists():
        return Image.open(item["nobg_path"]).convert("RGBA")

    source = Path(item["path"])
    if source.suffix.lower() in (".jpg", ".jpeg"):
        with st.spinner(f"Removing background from {item['name']} (first use only)…"):
            try:
                nobg_path = ensure_nobg(source)
            except Exception:
                nobg_path = source  # rembg model unavailable; use original
        item["nobg_path"] = str(nobg_path)
        return Image.open(nobg_path).convert("RGBA")

    return Image.open(source).convert("RGBA")


@st.cache_data(show_spinner=False)
def _load_gallery_preview(path: str) -> bytes:
    """Load and composite a jewellery thumbnail once; cached across reruns."""
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (245, 245, 245, 255))
    bg.paste(img, mask=img)
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Jewellery AR Try-On", page_icon="💍", layout="wide")

st.markdown("""
<style>
/* Main area: fixed viewport height + always-visible scrollbar.
   overflow-y:scroll (not auto) keeps the scrollbar track permanently in the
   layout so its 8 px width never appears/disappears — that change was stealing
   column width and forcing use_container_width images to resize on every
   content-height change (spinner ↔ text), causing the visible image flicker. */
section[data-testid="stMain"] {
    overflow-y: scroll !important;
    height: 100vh !important;
    overflow-anchor: none !important;
    scrollbar-gutter: stable;
}
/* Always-visible custom scrollbar (Windows 11 hides OS scrollbars) */
section[data-testid="stMain"]::-webkit-scrollbar       { width: 8px; }
section[data-testid="stMain"]::-webkit-scrollbar-track { background: #e8e8e8; border-radius: 4px; }
section[data-testid="stMain"]::-webkit-scrollbar-thumb { background: #aaa; border-radius: 4px; }
section[data-testid="stMain"]::-webkit-scrollbar-thumb:hover { background: #777; }

/* Sidebar: scrollable for long content (Fine-tune below gallery) */
section[data-testid="stSidebar"] > div:first-child {
    overflow-y: scroll !important;
    height: 100vh !important;
    scrollbar-gutter: stable;
}
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar { width: 5px; }
section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb { background: #ccc; border-radius: 4px; }

/* Prevent layout shift and white flash while images load */
[data-testid="stImage"] {
    background-color: #f5f5f5;
    min-height: 40px;
}
[data-testid="stImage"] img {
    display: block;
    min-height: 40px;
}

/* Keep result columns at a stable height so rerenders don't shift layout */
[data-testid="stHorizontalBlock"] > div {
    min-height: 200px;
}

/* Prevent column widths from shifting on rerender */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}

/* Stop Streamlit's internal scroll-to-top on rerun */
html { scroll-behavior: auto !important; }

/* Reserve space for the recommendation box so page height is stable
   whether the GPT-4o spinner or the success text is showing */
[data-testid="stAlert"], [data-testid="stNotification"] { min-height: 80px; }

.block-container { padding-bottom: 4rem; }
</style>
""", unsafe_allow_html=True)

# ── session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("selected_earring",   None),
    ("selected_necklace",  None),
    ("page_earring",       0),
    ("page_necklace",      0),
    ("active_type",        "Earring"),
    ("source_image",       None),
    ("source_image_hash",  ""),
    ("source_image_bytes", None),
    ("v_off_ear",          0),
    ("h_off_ear",          0),
    ("v_off_nec",          0),
    ("h_off_nec",          0),
    ("video_result_bytes", None),
    ("video_result_key",   ""),
    ("global_opacity",      1.0),
    ("global_size_factor",  1.0),
    ("use_flux_toggle",     False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Apply any click-derived positions BEFORE widgets render (Streamlit forbids
# setting a widget's key after it has been instantiated in the same run).
for _k in ("v_off_ear", "h_off_ear", "v_off_nec", "h_off_nec"):
    _pending = f"_pending_{_k}"
    if _pending in st.session_state:
        st.session_state[_k] = st.session_state[_pending]
        del st.session_state[_pending]

CATALOGUE = load_catalogue()
ITEMS_PER_PAGE = 4   # 2 rows × 2 cols in sidebar


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Jewellery Gallery
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("💍 Jewellery Gallery")

    # ── category selector ──────────────────────────────────────────────────────
    category = st.radio(
        "Category",
        ["Earring", "Necklace", "Both"],
        horizontal=True,
        key="active_type",
        label_visibility="collapsed",
    )

    # ── jewellery type toggle  ─────────────────────────────────────────────────
    use_flux = st.toggle(
        "AI Overlay (Flux Kontext Pro)",
        value=bool(_bfl_key()),   # on by default when API key is present
        disabled=not _bfl_key(),
        help="Uses Flux Kontext Pro for photorealistic AI jewellery overlay (requires BFL_API_KEY).",
        key="use_flux_toggle",
    )

    st.divider()

    def gallery_panel(cat_key: str):
        """Render paginated 2×2 gallery for a category, return selected item."""
        items   = CATALOGUE.get(cat_key, [])
        page_k  = f"page_{cat_key.lower()}"
        sel_k   = f"selected_{cat_key.lower()}"
        total   = len(items)
        n_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        page    = st.session_state[page_k]
        page    = max(0, min(page, n_pages - 1))

        slice_  = items[page * ITEMS_PER_PAGE : (page + 1) * ITEMS_PER_PAGE]

        if not slice_:
            st.caption(f"No {cat_key.lower()} assets found in assets/jewellery/{cat_key.lower()}s/")
            return

        # ── 2-column grid ──────────────────────────────────────────────────────
        for row_start in range(0, len(slice_), 2):
            c1, c2 = st.columns(2)
            for col, item in zip([c1, c2], slice_[row_start:row_start+2]):
                with col:
                    st.image(_load_gallery_preview(item["path"]), use_container_width=True)
                    selected = st.session_state[sel_k]
                    is_sel   = selected is not None and selected["id"] == item["id"]
                    btn_label = "✅ Selected" if is_sel else "Select"
                    if st.button(btn_label, key=f"btn_{cat_key}_{item['id']}", use_container_width=True):
                        st.session_state[sel_k] = item
                        st.rerun()
                    st.caption(item["name"])

        # ── pagination ─────────────────────────────────────────────────────────
        prev_col, info_col, next_col = st.columns([1, 2, 1])
        with prev_col:
            if st.button("←", key=f"prev_{cat_key}", disabled=page == 0):
                st.session_state[page_k] -= 1
                st.rerun()
        with info_col:
            st.caption(f"{page+1} / {n_pages}  ({total} items)")
        with next_col:
            if st.button("→", key=f"next_{cat_key}", disabled=page >= n_pages - 1):
                st.session_state[page_k] += 1
                st.rerun()

    # ── show correct panel(s) ──────────────────────────────────────────────────
    if category == "Both":
        st.subheader("Earrings")
        gallery_panel("Earring")
        st.divider()
        st.subheader("Necklace")
        gallery_panel("Necklace")
    else:
        gallery_panel(category)

    # ── show current selection summary ────────────────────────────────────────
    st.divider()
    sel_ear = st.session_state.selected_earring
    sel_nec = st.session_state.selected_necklace
    if category in ("Earring", "Both") and sel_ear:
        st.success(f"Earring: {sel_ear['name']}")
    if category in ("Necklace", "Both") and sel_nec:
        st.success(f"Necklace: {sel_nec['name']}")
    if category == "Earring" and not sel_ear:
        st.info("Select an earring above.")
    if category == "Necklace" and not sel_nec:
        st.info("Select a necklace above.")
    if category == "Both" and not (sel_ear and sel_nec):
        st.info("Select both an earring and a necklace.")

    # ── global + per-earring adjustments ─────────────────────────────────────
    st.divider()
    st.subheader("Fine-tune")

    opacity           = st.slider("Opacity (how solid the jewellery looks)",    0.1, 1.0, 1.0, 0.05, key="global_opacity")
    v_offset_earring  = st.slider("Earring Up / Down position (px)",            -60,  60,   0,    2, key="v_off_ear")
    h_offset_earring  = st.slider("Earring Left / Right position (px)",         -60,  60,   0,    2, key="h_off_ear")
    v_offset_necklace = st.slider("Necklace Up / Down position (px)",           -60, 300,   0,    2, key="v_off_nec")
    h_offset_necklace = st.slider("Necklace Left / Right position (px)",       -200, 200,   0,    2, key="h_off_nec")

    # Global size (applies to all)
    size_factor = st.slider("Jewellery Size (makes all jewellery bigger/smaller)", 0.3, 3.0, 1.0, 0.05, key="global_size_factor")

    # Per-earring size offset (persists per earring design)
    ear_size_factor = size_factor
    if sel_ear:
        sk = f"earsize_{sel_ear['id']}"
        if sk not in st.session_state:
            st.session_state[sk] = 1.0
        per_ear = st.slider(
            f"{sel_ear['name']} — individual size adjustment",
            0.3, 3.0,
            st.session_state[sk],
            0.05,
            key=f"sl_{sk}",
        )
        st.session_state[sk] = per_ear
        ear_size_factor = size_factor * per_ear
    st.session_state["_ear_size_factor"] = ear_size_factor


# ══════════════════════════════════════════════════════════════════════════════
#  Core processing
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _detect_cached(source_bytes: bytes, _v: int = 5) -> dict | None:
    """Cache face landmarks. _v bumped to bust stale None results on redeploy."""
    img_rgb = np.array(Image.open(io.BytesIO(source_bytes)).convert("RGB"))
    return detect_landmarks(img_rgb)


@st.cache_data(show_spinner=False)
def _compute_overlay(
    source_bytes: bytes,
    cat: str,
    ear_path: str | None,
    nec_path: str | None,
    eff_size: float,
    size_factor: float,
    v_off_ear: int,
    h_off_ear: int,
    v_off_nec: int,
    h_off_nec: int,
    opacity: float,
    is_pair: bool,
) -> bytes | None:
    """Pure function — all inputs explicit so @st.cache_data can hash them."""
    landmarks = _detect_cached(source_bytes)   # cache hit on slider changes
    if landmarks is None:
        return None
    source_img = Image.open(io.BytesIO(source_bytes))

    result = source_img.convert("RGBA")

    if cat in ("Earring", "Both") and ear_path:
        earring = Image.open(ear_path).convert("RGBA")
        result  = overlay_earrings(result, earring, landmarks, eff_size,
                                   v_off_ear, h_off_ear, opacity, is_pair=is_pair)

    if cat in ("Necklace", "Both") and nec_path:
        necklace = Image.open(nec_path).convert("RGBA")
        result   = overlay_necklace(result, necklace, landmarks, size_factor,
                                    v_off_nec, h_off_nec, opacity)

    return pil_to_bytes(result.convert("RGB"))


def _overlay_paths(sel_ear, sel_nec):
    """Resolve nobg paths for both jewellery items; return (ear_path, nec_path)."""
    ear_path = None
    if sel_ear:
        resolve_overlay_image(sel_ear)
        ear_path = sel_ear.get("nobg_path") or str(sel_ear["path"])
    nec_path = None
    if sel_nec:
        resolve_overlay_image(sel_nec)
        nec_path = sel_nec.get("nobg_path") or str(sel_nec["path"])
    return ear_path, nec_path


# ══════════════════════════════════════════════════════════════════════════════
#  Shared result renderer
# ══════════════════════════════════════════════════════════════════════════════

@st.dialog("Try-On Result", width="large")
def _show_enlarged(img: Image.Image):
    st.image(img, use_container_width=True)


def _draw_position_markers(img: Image.Image, landmarks: dict, cat: str,
                            v_off_ear: int, h_off_ear: int,
                            v_off_nec: int, h_off_nec: int) -> Image.Image:
    """Draw gold/cyan dots showing where jewellery will be placed."""
    marked = img.copy().convert("RGB")
    draw   = ImageDraw.Draw(marked)
    r = max(10, landmarks["face_width"] // 18)

    if cat in ("Earring", "Both"):
        lx = landmarks["left_ear"][0]  - h_off_ear
        ly = landmarks["left_ear"][1]  + 10 + v_off_ear
        rx = landmarks["right_ear"][0] + h_off_ear
        ry = landmarks["right_ear"][1] + 10 + v_off_ear
        for cx, cy in [(lx, ly), (rx, ry)]:
            draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                         fill=(255, 215, 0), outline=(180, 120, 0), width=3)

    if cat in ("Necklace", "Both"):
        nx = landmarks["neck_center"][0] + h_off_nec
        ny = landmarks["neck_center"][1] + v_off_nec
        draw.ellipse([nx-r, ny-r, nx+r, ny+r],
                     fill=(0, 200, 255), outline=(0, 120, 180), width=3)
    return marked


@st.cache_data(show_spinner=False)
def _cached_markers(source_bytes: bytes, cat: str,
                    v_off_ear: int, h_off_ear: int,
                    v_off_nec: int, h_off_nec: int) -> bytes | None:
    """Cached version of position-marker drawing; avoids redraw on every rerun."""
    landmarks = _detect_cached(source_bytes)
    if landmarks is None:
        return None
    source_img = Image.open(io.BytesIO(source_bytes))
    marked = _draw_position_markers(source_img, landmarks, cat,
                                    v_off_ear, h_off_ear, v_off_nec, h_off_nec)
    return pil_to_bytes(marked)


def _result_session_key(
    v_off_ear: int | None = None,
    h_off_ear: int | None = None,
    v_off_nec: int | None = None,
    h_off_nec: int | None = None,
) -> str:
    img_hash = st.session_state.get("source_image_hash", "no_img")
    cat      = st.session_state.active_type
    sel_ear  = st.session_state.selected_earring
    sel_nec  = st.session_state.selected_necklace
    eff_sz   = st.session_state.get("_ear_size_factor", 1.0)
    size_f   = st.session_state.get("global_size_factor", 1.0)
    opacity  = st.session_state.get("global_opacity", 1.0)
    voe = v_off_ear if v_off_ear is not None else st.session_state.get("v_off_ear", 0)
    hoe = h_off_ear if h_off_ear is not None else st.session_state.get("h_off_ear", 0)
    von = v_off_nec if v_off_nec is not None else st.session_state.get("v_off_nec", 0)
    hon = h_off_nec if h_off_nec is not None else st.session_state.get("h_off_nec", 0)
    parts = [
        img_hash, cat,
        sel_ear["id"] if sel_ear else "",
        sel_nec["id"] if sel_nec else "",
        str(round(eff_sz, 3)),
        str(round(size_f, 3)),
        str(voe), str(hoe),
        str(von), str(hon),
        str(round(opacity, 3)),
    ]
    return "res_" + hashlib.md5("|".join(parts).encode()).hexdigest()


@st.fragment
def show_result(source_img: Image.Image, _download_key: str = "download_result"):
    # Read all state from session_state so fragment reruns work independently
    cat      = st.session_state.active_type
    sel_ear  = st.session_state.selected_earring
    sel_nec  = st.session_state.selected_necklace
    use_flux = st.session_state.get("use_flux_toggle", False)
    size_f   = st.session_state.get("global_size_factor", 1.0)
    opacity  = st.session_state.get("global_opacity", 1.0)
    eff_size = st.session_state.get("_ear_size_factor", size_f)

    if cat == "Earring"  and not sel_ear:
        st.info("Please select an earring from the gallery.")
        return
    if cat == "Necklace" and not sel_nec:
        st.info("Please select a necklace from the gallery.")
        return
    if cat == "Both"     and not (sel_ear and sel_nec):
        st.info("Please select both an earring and a necklace from the gallery.")
        return

    src_bytes = st.session_state.get("source_image_bytes")
    if not src_bytes:
        return

    # Sync click-based overrides: if slider moved since last fragment run, discard override
    for adj_k, slider_k in [
        ("_fadj_voe", "v_off_ear"), ("_fadj_hoe", "h_off_ear"),
        ("_fadj_von", "v_off_nec"), ("_fadj_hon", "h_off_nec"),
    ]:
        last_k     = f"_flast_{slider_k}"
        slider_val = st.session_state.get(slider_k, 0)
        if st.session_state.get(last_k, slider_val) != slider_val:
            st.session_state.pop(adj_k, None)
        st.session_state[last_k] = slider_val

    # ── Click-to-position (BEFORE result so coords update in the same rendering pass) ──
    # When the user clicks, streamlit_image_coordinates returns new coords, we update
    # _fadj_* immediately, then read them below for the overlay — no second st.rerun() needed.
    landmarks = _detect_cached(src_bytes)
    if landmarks:
        show_repo = st.checkbox(
            "🎯 Adjust jewellery position by clicking on the photo",
            key=f"show_repo_{_download_key}",
        )
        if show_repo:
            if cat == "Both":
                pos_mode = st.radio(
                    "What to reposition:",
                    ["Earring", "Necklace"],
                    horizontal=True,
                    key=f"pos_mode_{_download_key}",
                )
            else:
                pos_mode = cat

            if pos_mode == "Earring":
                st.caption("🟡 Click where you want the **earring** to sit.")
            else:
                st.caption("🔵 Click where you want the **necklace** centre to sit.")

            # Marker uses pre-click offsets (shows current placement)
            pre_voe = st.session_state.get("_fadj_voe", st.session_state.get("v_off_ear", 0))
            pre_hoe = st.session_state.get("_fadj_hoe", st.session_state.get("h_off_ear", 0))
            pre_von = st.session_state.get("_fadj_von", st.session_state.get("v_off_nec", 0))
            pre_hon = st.session_state.get("_fadj_hon", st.session_state.get("h_off_nec", 0))
            marked_bytes = _cached_markers(src_bytes, pos_mode, pre_voe, pre_hoe, pre_von, pre_hon)
            coords = None
            if marked_bytes:
                coords = streamlit_image_coordinates(
                    Image.open(io.BytesIO(marked_bytes)),
                    key=f"click_{_download_key}_{pos_mode}",
                )

            seen_key = f"_coords_seen_{_download_key}_{pos_mode}"
            if coords and coords != st.session_state.get(seen_key):
                st.session_state[seen_key] = coords
                if pos_mode == "Earring":
                    rx, ry = landmarks["right_ear"]
                    st.session_state["_fadj_voe"] = max(-60, min(60, int(coords["y"]) - ry - 10))
                    st.session_state["_fadj_hoe"] = max(-60, min(60, int(coords["x"]) - rx))
                    st.session_state["_flast_v_off_ear"] = st.session_state.get("v_off_ear", 0)
                    st.session_state["_flast_h_off_ear"] = st.session_state.get("h_off_ear", 0)
                else:
                    nx, ny = landmarks["neck_center"]
                    st.session_state["_fadj_von"] = max(-60,  min(300, int(coords["y"]) - ny))
                    st.session_state["_fadj_hon"] = max(-200, min(200, int(coords["x"]) - nx))
                    st.session_state["_flast_v_off_nec"] = st.session_state.get("v_off_nec", 0)
                    st.session_state["_flast_h_off_nec"] = st.session_state.get("h_off_nec", 0)
                # No st.rerun() — offsets updated above; result computed below in the same pass

    # Read effective offsets (may have just been updated by click above)
    voe = st.session_state.get("_fadj_voe", st.session_state.get("v_off_ear", 0))
    hoe = st.session_state.get("_fadj_hoe", st.session_state.get("h_off_ear", 0))
    von = st.session_state.get("_fadj_von", st.session_state.get("v_off_nec", 0))
    hon = st.session_state.get("_fadj_hon", st.session_state.get("h_off_nec", 0))

    # ── Flux path (explicit button — never auto-calls) ────────────────────────
    flux_result: bytes | None = None
    if use_flux and _bfl_key():
        flux_key = _result_session_key()
        flux_result = st.session_state.get(flux_key)

    # ── Local overlay (instant, always computed) ──────────────────────────────
    is_pair  = bool(sel_ear) and Path(sel_ear["path"]).suffix.lower() in (".jpg", ".jpeg")
    ear_path, nec_path = _overlay_paths(sel_ear, sel_nec)
    res_key = "local_" + _result_session_key(voe, hoe, von, hon)
    if res_key not in st.session_state:
        with st.spinner("Applying jewellery…"):
            rb = _compute_overlay(
                src_bytes, cat, ear_path, nec_path, eff_size, size_f,
                voe, hoe, von, hon, opacity, is_pair,
            )
        if rb:
            st.session_state[res_key] = rb
    local_result = st.session_state.get(res_key)

    result_bytes = flux_result or local_result

    if result_bytes is None:
        img_dbg = Image.open(io.BytesIO(src_bytes))
        w_dbg, h_dbg = img_dbg.size
        st.error(
            f"No face detected in {w_dbg}×{h_dbg}px image. "
            "Please use a clear, front-facing photo (minimum ~300px wide)."
        )
        return

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Original")
        st.image(src_bytes, use_container_width=True)
    with col_out:
        st.subheader("Try-On Result")
        if flux_result:
            st.caption("✨ Flux Kontext Pro result")
        st.image(result_bytes, use_container_width=True)
        if st.button("🔍 Enlarge", key=f"enlarge_{_download_key}", use_container_width=True):
            _show_enlarged(Image.open(io.BytesIO(result_bytes)))

    st.download_button(
        label="⬇️ Download Result",
        data=result_bytes,
        file_name="jewellery_tryon.png",
        mime="image/png",
        use_container_width=True,
        key=_download_key,
    )

    # ── Flux action buttons ───────────────────────────────────────────────────
    if use_flux and _bfl_key():
        flux_key = _result_session_key()
        if not flux_result:
            if st.button("✨ Apply with Flux Kontext Pro", key=f"flux_apply_{_download_key}",
                         use_container_width=True):
                design_name = (sel_ear["name"] if sel_ear else "") or (sel_nec["name"] if sel_nec else "")
                ref_img = resolve_overlay_image(sel_ear if sel_ear else sel_nec)
                with st.spinner("Sending to Flux Kontext Pro… (30–60 s)"):
                    try:
                        r = overlay_with_flux(source_img, ref_img, cat, design_name, _bfl_key())
                        st.session_state[flux_key] = pil_to_bytes(r.convert("RGB"))
                    except Exception as exc:
                        st.error(f"Flux failed: {exc}")
                        st.session_state[flux_key] = None
                st.rerun()
        else:
            if st.button("🔄 Re-generate with Flux", key=f"flux_regen_{_download_key}",
                         use_container_width=True):
                del st.session_state[flux_key]
                st.rerun()

    # ── GPT-4o auto recommendation ────────────────────────────────────────────
    st.divider()
    st.subheader("✨ AI Style Recommendation")

    if not _azure_ready():
        st.info("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY in .env to enable.")
        return

    desc    = " + ".join(filter(None, [sel_ear["name"] if sel_ear else None,
                                       sel_nec["name"] if sel_nec else None]))
    rec_key = f"rec_{desc}"
    if rec_key not in st.session_state:
        with st.spinner("GPT-4o is analysing your try-on…"):
            try:
                st.session_state[rec_key] = get_recommendation(
                    Image.open(io.BytesIO(result_bytes)), st.session_state.active_type, desc
                )
            except Exception as exc:
                st.session_state[rec_key] = f"⚠️ Could not get recommendation: {exc}"

    st.success(st.session_state[rec_key])


# ══════════════════════════════════════════════════════════════════════════════
#  Video processing
# ══════════════════════════════════════════════════════════════════════════════

def _reencode_h264(raw_path: str) -> str:
    """Re-encode a raw mp4v file to H.264 (browser-compatible). Returns output path."""
    import subprocess
    out_tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tf.name
    out_tf.close()
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path,
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-movflags", "+faststart",
         out_path],
        capture_output=True,
    )
    if r.returncode == 0:
        os.unlink(raw_path)
        return out_path
    os.unlink(out_path)
    return raw_path   # fall back to original if ffmpeg unavailable


def _process_video(
    video_bytes: bytes,
    cat: str,
    ear_path: str | None,
    nec_path: str | None,
    eff_size: float,
    size_factor: float,
    v_off_ear: int,
    h_off_ear: int,
    v_off_nec: int,
    h_off_nec: int,
    opacity: float,
    is_pair: bool,
    progress_bar,
) -> tuple[bytes | None, str]:
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(video_bytes)
        in_path = tf.name

    cap = cv2.VideoCapture(in_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tf.name
    out_tf.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Pre-load jewellery images once (avoid re-opening on every frame)
    earring_img  = Image.open(ear_path).convert("RGBA")  if ear_path  else None
    necklace_img = Image.open(nec_path).convert("RGBA")  if nec_path  else None

    landmarks  = None
    frame_idx  = 0
    face_found = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect once on the first frame; reuse for all subsequent frames
        if landmarks is None:
            landmarks = detect_landmarks(frame_rgb)
            if landmarks is None:
                face_found = False

        if landmarks is not None:
            pil_frame = Image.fromarray(frame_rgb).convert("RGBA")
            result    = pil_frame

            if cat in ("Earring", "Both") and earring_img is not None:
                result = overlay_earrings(result, earring_img, landmarks,
                                          eff_size, v_off_ear, h_off_ear,
                                          opacity, is_pair=is_pair)
            if cat in ("Necklace", "Both") and necklace_img is not None:
                result = overlay_necklace(result, necklace_img, landmarks,
                                          size_factor, v_off_nec, h_off_nec, opacity)

            out_rgb = np.array(result.convert("RGB"))
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = frame

        writer.write(out_bgr)
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    os.unlink(in_path)

    out_path = _reencode_h264(out_path)   # mp4v → H.264 for browser playback

    with open(out_path, "rb") as f:
        out_bytes = f.read()
    os.unlink(out_path)

    if not face_found:
        return out_bytes, "⚠️ No face detected — jewellery not applied. Video returned as-is."
    return out_bytes, "OK"


def _video_result_key(video_hash: str) -> str:
    cat      = st.session_state.active_type
    sel_ear  = st.session_state.selected_earring
    sel_nec  = st.session_state.selected_necklace
    eff_sz   = st.session_state.get("_ear_size_factor", 1.0)
    size_f   = st.session_state.get("global_size_factor", 1.0)
    opacity  = st.session_state.get("global_opacity", 1.0)
    parts    = [
        video_hash, cat,
        sel_ear["id"] if sel_ear else "",
        sel_nec["id"] if sel_nec else "",
        str(round(eff_sz, 3)),
        str(round(size_f, 3)),
        str(st.session_state.get("v_off_ear", 0)),
        str(st.session_state.get("h_off_ear", 0)),
        str(st.session_state.get("v_off_nec", 0)),
        str(st.session_state.get("h_off_nec", 0)),
        str(round(opacity, 3)),
    ]
    return "vid_" + hashlib.md5("|".join(parts).encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Main — tabs
# ══════════════════════════════════════════════════════════════════════════════

st.title("💍 Jewellery AR Try-On")
st.caption("Powered by MediaPipe · Flux Kontext Pro · Azure OpenAI GPT-4o")

tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Photo", "📷 Camera Capture", "🎬 Video Try-On", "📡 Live AR"])

with tab1:
    uploaded = st.file_uploader(
        "Upload a front-facing photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="upload",
    )
    if uploaded:
        new_hash = hashlib.md5(uploaded.getvalue()).hexdigest()[:10]
        if new_hash != st.session_state.source_image_hash:
            img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGBA")
            st.session_state.source_image       = img
            st.session_state.source_image_bytes = pil_to_bytes(img)
            st.session_state.source_image_hash  = new_hash
            for k in [k for k in st.session_state if k.startswith("rec_")]:
                del st.session_state[k]

    if st.session_state.source_image is not None:
        show_result(st.session_state.source_image, _download_key="download_upload")
    else:
        st.info("Upload your photo once — then browse and select any jewellery to instantly apply it.")

with tab2:
    camera_img = st.camera_input("Take a photo", key="camera")
    if camera_img:
        new_hash = hashlib.md5(camera_img.getvalue()).hexdigest()[:10]
        if new_hash != st.session_state.source_image_hash:
            img = ImageOps.exif_transpose(Image.open(camera_img)).convert("RGBA")
            st.session_state.source_image       = img
            st.session_state.source_image_bytes = pil_to_bytes(img)
            st.session_state.source_image_hash  = new_hash
            for k in [k for k in st.session_state if k.startswith("rec_")]:
                del st.session_state[k]

    if st.session_state.source_image is not None:
        show_result(st.session_state.source_image, _download_key="download_camera")
    else:
        st.info("Capture your photo once — then browse and select any jewellery to instantly apply it.")

with tab3:
    st.subheader("🎬 Video Try-On")
    st.caption("Upload a video and the selected jewellery will be overlaid on every frame.")

    # ── Sample video generator ────────────────────────────────────────────────
    src_bytes_for_vid = st.session_state.get("source_image_bytes")
    if src_bytes_for_vid:
        if st.button("🎞️ Generate sample test video from your photo", use_container_width=True):
            import cv2, tempfile, numpy as np
            frame = np.array(Image.open(io.BytesIO(src_bytes_for_vid)).convert("RGB"))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tmp_path = tf.name
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            for _ in range(75):   # 3 seconds at 25 fps
                writer.write(frame_bgr)
            writer.release()
            tmp_path = _reencode_h264(tmp_path)   # mp4v → H.264 for browser playback
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "⬇️ Download sample_test.mp4",
                    data=f.read(),
                    file_name="sample_test.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
    else:
        st.info("Upload a photo first (Upload Photo tab), then use this button to generate a test video.")

    st.divider()

    uploaded_video = st.file_uploader(
        "Upload a video (MP4 / MOV / AVI)",
        type=["mp4", "mov", "avi"],
        key="video_upload",
    )

    if not uploaded_video:
        st.info("Upload a video, select jewellery from the sidebar, then click **Process Video**.")
    else:
        video_hash = hashlib.md5(uploaded_video.getvalue()).hexdigest()[:10]

        cat     = st.session_state.active_type
        sel_ear = st.session_state.selected_earring
        sel_nec = st.session_state.selected_necklace
        ready   = (
            (cat == "Earring"  and sel_ear) or
            (cat == "Necklace" and sel_nec) or
            (cat == "Both"     and sel_ear and sel_nec)
        )

        if not ready:
            st.warning("Select jewellery from the sidebar gallery first, then click **Process Video**.")

        vkey = _video_result_key(video_hash) if ready else ""

        # ── Process button (shown when no valid cached result) ─────────────────
        has_stale = (
            bool(st.session_state.video_result_bytes)
            and st.session_state.video_result_key != vkey
        )
        if ready and not (st.session_state.video_result_bytes
                          and st.session_state.video_result_key == vkey):
            if has_stale:
                st.info("⚙️ Fine-tune settings changed — click **Process Video** to apply.")
            if st.button("▶️ Process Video", key="btn_process_video",
                         use_container_width=True,
                         help="Apply selected jewellery to every frame"):
                is_pair  = bool(sel_ear) and Path(sel_ear["path"]).suffix.lower() in (".jpg", ".jpeg")
                eff_size = st.session_state.get("_ear_size_factor", size_factor)

                ear_path = None
                if sel_ear:
                    resolve_overlay_image(sel_ear)
                    ear_path = sel_ear.get("nobg_path") or str(sel_ear["path"])
                nec_path = None
                if sel_nec:
                    resolve_overlay_image(sel_nec)
                    nec_path = sel_nec.get("nobg_path") or str(sel_nec["path"])

                prog = st.progress(0.0, text="Processing frames…")
                try:
                    out_bytes, msg = _process_video(
                        uploaded_video.getvalue(),
                        cat, ear_path, nec_path,
                        eff_size, size_factor,
                        v_offset_earring, h_offset_earring,
                        v_offset_necklace, h_offset_necklace,
                        opacity, is_pair,
                        prog,
                    )
                    prog.empty()
                    st.session_state.video_result_bytes = out_bytes
                    st.session_state.video_result_key   = vkey
                    if msg != "OK":
                        st.warning(msg)
                except Exception as exc:
                    prog.empty()
                    st.error(f"Video processing failed: {exc}")

        # ── Display original + result side by side ─────────────────────────────
        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.video(uploaded_video.getvalue())

        if st.session_state.video_result_bytes and st.session_state.video_result_key == vkey:
            with col_result:
                st.markdown("**Try-On Result**")
                st.video(st.session_state.video_result_bytes)

            st.download_button(
                label="⬇️ Download Video",
                data=st.session_state.video_result_bytes,
                file_name="jewellery_tryon_video.mp4",
                mime="video/mp4",
                use_container_width=True,
                key="download_video",
            )
            if st.button("🔄 Re-process with updated settings", key="btn_reprocess_video",
                         use_container_width=True):
                st.session_state.video_result_bytes = None
                st.session_state.video_result_key   = ""
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — Live AR (WebRTC real-time camera with head-pose-aware jewellery)
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📡 Live AR Try-On")
    st.caption(
        "Your webcam streams live — jewellery follows your face in real time. "
        "Turn your head left/right or tilt and the pieces warp to match."
    )

    try:
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration
        from utils.live_ar import JewelleryARProcessor
        _webrtc_ok = True
    except ImportError:
        _webrtc_ok = False
        st.error(
            "streamlit-webrtc is not installed. "
            "Add `streamlit-webrtc` and `aiortc` to requirements.txt and redeploy."
        )

    if _webrtc_ok:
        # Resolve jewellery paths for the processor
        _live_ear_path = None
        _live_nec_path = None
        if sel_ear:
            resolve_overlay_image(sel_ear)
            _live_ear_path = sel_ear.get("nobg_path") or str(sel_ear["path"])
        if sel_nec:
            resolve_overlay_image(sel_nec)
            _live_nec_path = sel_nec.get("nobg_path") or str(sel_nec["path"])

        if not (_live_ear_path or _live_nec_path):
            st.info("Select jewellery from the sidebar gallery, then start the camera below.")

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        ctx = webrtc_streamer(
            key="live-ar",
            video_processor_factory=JewelleryARProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            _is_pair = bool(sel_ear) and Path(sel_ear["path"]).suffix.lower() in (".jpg", ".jpeg")
            with ctx.video_processor._lock:
                ctx.video_processor.category      = category
                ctx.video_processor.earring_path  = _live_ear_path
                ctx.video_processor.necklace_path = _live_nec_path
                ctx.video_processor.size_factor   = ear_size_factor
                ctx.video_processor.v_offset_ear  = v_offset_earring
                ctx.video_processor.h_offset_ear  = h_offset_earring
                ctx.video_processor.v_offset_nec  = v_offset_necklace
                ctx.video_processor.h_offset_nec  = h_offset_necklace
                ctx.video_processor.opacity       = opacity
                ctx.video_processor.is_pair       = _is_pair

        st.info(
            "**Tips:** Use the Fine-tune sliders in the sidebar to adjust size and position "
            "while the camera is live. Works best in Chrome with good lighting."
        )
