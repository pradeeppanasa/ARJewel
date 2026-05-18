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
            nobg_path = ensure_nobg(source)
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
/* Main area: fixed viewport height + scrollable = right-side scrollbar */
section[data-testid="stMain"] {
    overflow-y: auto !important;
    height: 100vh !important;
    overflow-anchor: none !important;
}
/* Always-visible custom scrollbar (Windows 11 hides OS scrollbars) */
section[data-testid="stMain"]::-webkit-scrollbar       { width: 8px; }
section[data-testid="stMain"]::-webkit-scrollbar-track { background: #e8e8e8; border-radius: 4px; }
section[data-testid="stMain"]::-webkit-scrollbar-thumb { background: #aaa; border-radius: 4px; }
section[data-testid="stMain"]::-webkit-scrollbar-thumb:hover { background: #777; }

/* Sidebar: scrollable for long content (Fine-tune below gallery) */
section[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    height: 100vh !important;
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
                        # Clear recommendation cache so it refreshes for new jewellery
                        for k in list(st.session_state.keys()):
                            if k.startswith("rec_"):
                                del st.session_state[k]
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

    opacity           = st.slider("Opacity (how solid the jewellery looks)",    0.1, 1.0, 1.0, 0.05)
    v_offset_earring  = st.slider("Earring Up / Down position (px)",            -60,  60,   0,    2, key="v_off_ear")
    h_offset_earring  = st.slider("Earring Left / Right position (px)",         -60,  60,   0,    2, key="h_off_ear")
    v_offset_necklace = st.slider("Necklace Up / Down position (px)",           -60, 300,   0,    2, key="v_off_nec")
    h_offset_necklace = st.slider("Necklace Left / Right position (px)",       -200, 200,   0,    2, key="h_off_nec")

    # Global size (applies to all)
    size_factor = st.slider("Jewellery Size (makes all jewellery bigger/smaller)", 0.3, 3.0, 1.0, 0.05)

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
def _detect_cached(source_bytes: bytes) -> dict | None:
    """Cache face landmarks separately so slider changes skip re-detection."""
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


def _result_session_key() -> str:
    # Use pre-stored image hash — avoids expensive pil_to_bytes on every rerun
    img_hash = st.session_state.get("source_image_hash", "no_img")
    cat      = st.session_state.active_type
    sel_ear  = st.session_state.selected_earring
    sel_nec  = st.session_state.selected_necklace
    eff_sz   = st.session_state.get("_ear_size_factor", 1.0)
    parts    = [
        img_hash, cat,
        sel_ear["id"] if sel_ear else "",
        sel_nec["id"] if sel_nec else "",
        str(round(eff_sz, 3)),
        str(round(size_factor, 3)),
        str(v_offset_earring), str(h_offset_earring),
        str(v_offset_necklace), str(h_offset_necklace),
        str(round(opacity, 3)),
    ]
    return "res_" + hashlib.md5("|".join(parts).encode()).hexdigest()


def show_result(source_img: Image.Image, _download_key: str = "download_result"):
    cat     = st.session_state.active_type
    sel_ear = st.session_state.selected_earring
    sel_nec = st.session_state.selected_necklace

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

    result_bytes: bytes | None = None

    # ── Flux path (slow — always cache result in session state) ───────────────
    if use_flux and _bfl_key():
        flux_key = _result_session_key()
        if flux_key not in st.session_state:
            design_name = (sel_ear["name"] if sel_ear else "") or (sel_nec["name"] if sel_nec else "")
            ref_img = resolve_overlay_image(sel_ear if sel_ear else sel_nec)
            try:
                with st.spinner("Sending to Flux Kontext Pro… (30–60 s)"):
                    r = overlay_with_flux(source_img, ref_img, cat, design_name, _bfl_key())
                st.session_state[flux_key] = pil_to_bytes(r.convert("RGB"))
            except Exception:
                st.session_state[flux_key] = None
        result_bytes = st.session_state.get(flux_key)

    # ── Local path ────────────────────────────────────────────────────────────
    if result_bytes is None:
        is_pair  = bool(sel_ear) and Path(sel_ear["path"]).suffix.lower() in (".jpg", ".jpeg")
        eff_size = st.session_state.get("_ear_size_factor", size_factor)
        ear_path, nec_path = _overlay_paths(sel_ear, sel_nec)

        # Cache result bytes in session state keyed by ALL parameters.
        # @st.cache_data returns cache hits instantly so no spinner is needed.
        # Storing in session state means the exact same bytes object is served
        # on every re-render — React sees an unchanged data-URL and skips DOM
        # update, eliminating the right-column flicker.
        res_key = _result_session_key()
        if res_key not in st.session_state:
            rb = _compute_overlay(
                src_bytes, cat, ear_path, nec_path, eff_size, size_factor,
                v_offset_earring, h_offset_earring,
                v_offset_necklace, h_offset_necklace,
                opacity, is_pair,
            )
            if rb:
                st.session_state[res_key] = rb
        result_bytes = st.session_state.get(res_key)

    if result_bytes is None:
        st.error("No face detected. Please use a clear, front-facing photo.")
        return

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Original")
        st.image(src_bytes, use_container_width=True)
    with col_out:
        st.subheader("Try-On Result")
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

    # ── Click-to-position ─────────────────────────────────────────────────────
    # The component is only instantiated when the checkbox is ON so it never
    # fires its initial {"x":0,"y":0} value silently and causes an infinite
    # rerun loop.  Coordinate deduplication is a second guard.
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
                st.caption("🟡 Click where you want the **earring** to sit — both sides update together.")
            else:
                st.caption("🔵 Click where you want the **necklace** centre to sit.")

            marked_bytes = _cached_markers(
                src_bytes, pos_mode,
                v_offset_earring, h_offset_earring,
                v_offset_necklace, h_offset_necklace,
            )
            coords = None
            if marked_bytes:
                coords = streamlit_image_coordinates(
                    Image.open(io.BytesIO(marked_bytes)),
                    key=f"click_{_download_key}_{pos_mode}",
                )

            # Only act on genuinely new coordinates — prevents the initial
            # {"x":0,"y":0} value (or any repeated value) from looping.
            seen_key = f"_coords_seen_{_download_key}_{pos_mode}"
            if coords and coords != st.session_state.get(seen_key):
                st.session_state[seen_key] = coords
                S = 0.4
                if pos_mode == "Earring":
                    rx, ry = landmarks["right_ear"]
                    cur_v  = st.session_state.get("v_off_ear", 0)
                    cur_h  = st.session_state.get("h_off_ear", 0)
                    new_v  = int(cur_v + (int(coords["y"]) - ry - 10 - cur_v) * S)
                    new_h  = int(cur_h + (int(coords["x"]) - rx       - cur_h) * S)
                    st.session_state["_pending_v_off_ear"] = max(-60,  min(60,  new_v))
                    st.session_state["_pending_h_off_ear"] = max(-60,  min(60,  new_h))
                else:
                    nx, ny = landmarks["neck_center"]
                    cur_v  = st.session_state.get("v_off_nec", 0)
                    cur_h  = st.session_state.get("h_off_nec", 0)
                    new_v  = int(cur_v + (int(coords["y"]) - ny - cur_v) * S)
                    new_h  = int(cur_h + (int(coords["x"]) - nx - cur_h) * S)
                    st.session_state["_pending_v_off_nec"] = max(-60,  min(300, new_v))
                    st.session_state["_pending_h_off_nec"] = max(-200, min(200, new_h))
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

    with open(out_path, "rb") as f:
        out_bytes = f.read()
    os.unlink(out_path)

    if not face_found:
        return out_bytes, "⚠️ No face detected — jewellery not applied. Video returned as-is."
    return out_bytes, "OK"


def _video_result_key(video_hash: str) -> str:
    cat     = st.session_state.active_type
    sel_ear = st.session_state.selected_earring
    sel_nec = st.session_state.selected_necklace
    eff_sz  = st.session_state.get("_ear_size_factor", 1.0)
    parts   = [
        video_hash, cat,
        sel_ear["id"] if sel_ear else "",
        sel_nec["id"] if sel_nec else "",
        str(round(eff_sz, 3)),
        str(round(size_factor, 3)),
        str(v_offset_earring), str(h_offset_earring),
        str(v_offset_necklace), str(h_offset_necklace),
        str(round(opacity, 3)),
    ]
    return "vid_" + hashlib.md5("|".join(parts).encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Main — tabs
# ══════════════════════════════════════════════════════════════════════════════

st.title("💍 Jewellery AR Try-On")
st.caption("Powered by MediaPipe · Flux Kontext Pro · Azure OpenAI GPT-4o")

tab1, tab2, tab3 = st.tabs(["📁 Upload Photo", "📷 Camera Capture", "🎬 Video Try-On"])

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

    if st.session_state.source_image is not None:
        show_result(st.session_state.source_image, _download_key="download_camera")
    else:
        st.info("Capture your photo once — then browse and select any jewellery to instantly apply it.")

with tab3:
    st.subheader("🎬 Video Try-On")
    st.caption("Upload a video and the selected jewellery will be overlaid on every frame.")

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
        if ready and not (st.session_state.video_result_bytes
                          and st.session_state.video_result_key == vkey):
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
            st.video(uploaded_video)

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
