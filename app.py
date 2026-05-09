"""
Jewellery AR Try-On — Streamlit App

Stack:
  • Streamlit          — UI + visual jewellery gallery
  • MediaPipe          — ear / neck landmark detection
  • OpenCV + PIL       — local pixel-level overlay (fallback)
  • Flux Kontext Pro   — AI photorealistic overlay (BFL API)
  • Azure OpenAI GPT-4o — jewellery style recommendation
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

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


# ══════════════════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Jewellery AR Try-On", page_icon="💍", layout="wide")

# ── session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("selected_earring",  None),
    ("selected_necklace", None),
    ("page_earring",      0),
    ("page_necklace",     0),
    ("active_type",       "Earring"),
    ("source_image",      None),   # persisted photo — no re-upload needed
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
                    img = Image.open(item["path"]).convert("RGBA")
                    # white preview background so transparent PNG shows nicely
                    bg  = Image.new("RGBA", img.size, (245, 245, 245, 255))
                    bg.paste(img, mask=img)
                    st.image(bg.convert("RGB"), use_container_width=True)
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

    opacity           = st.slider("Opacity",               0.1, 1.0, 1.0, 0.05)
    v_offset_earring  = st.slider("Earring V-Offset (px)", -60,  60,   0,    2)
    h_offset_earring  = st.slider("Earring H-Offset (px)", -60,  60,   0,    2)
    v_offset_necklace = st.slider("Necklace V-Offset (px)",-60, 120,   0,    2)

    # Global size (applies to all)
    size_factor = st.slider("Global Size", 0.3, 3.0, 1.0, 0.05)

    # Per-earring size offset (persists per earring design)
    ear_size_factor = size_factor
    if sel_ear:
        sk = f"earsize_{sel_ear['id']}"
        if sk not in st.session_state:
            st.session_state[sk] = 1.0
        per_ear = st.slider(
            f"Size offset: {sel_ear['name']}",
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

def process_image(source_img: Image.Image) -> tuple[Image.Image | None, str]:
    cat      = st.session_state.active_type
    sel_ear  = st.session_state.selected_earring
    sel_nec  = st.session_state.selected_necklace

    # Validate selection
    if cat == "Earring"  and not sel_ear:
        return None, "Please select an earring from the gallery."
    if cat == "Necklace" and not sel_nec:
        return None, "Please select a necklace from the gallery."
    if cat == "Both"     and not (sel_ear and sel_nec):
        return None, "Please select both an earring and a necklace from the gallery."

    design_name = (sel_ear["name"] if sel_ear else "") or (sel_nec["name"] if sel_nec else "")

    # ── Flux path ──────────────────────────────────────────────────────────────
    if use_flux and _bfl_key():
        ref_img = resolve_overlay_image(sel_ear if sel_ear else sel_nec)
        try:
            with st.spinner("Sending to Flux Kontext Pro… (30–60 s)"):
                result = overlay_with_flux(
                    source_img, ref_img, cat, design_name, _bfl_key()
                )
            return result, "OK"
        except Exception as exc:
            st.warning(f"Flux unavailable ({type(exc).__name__}) — using local overlay.")

    # ── Local MediaPipe + PIL path ─────────────────────────────────────────────
    img_rgb   = np.array(source_img.convert("RGB"))
    landmarks = detect_landmarks(img_rgb)
    if landmarks is None:
        return None, "No face detected. Please use a clear, front-facing photo."

    result = source_img.convert("RGBA")

    if cat in ("Earring", "Both") and sel_ear:
        earring  = resolve_overlay_image(sel_ear)
        # JPG source = real product photo showing both earrings → split in half
        is_pair  = Path(sel_ear["path"]).suffix.lower() in (".jpg", ".jpeg")
        eff_size = st.session_state.get("_ear_size_factor", size_factor)
        result   = overlay_earrings(result, earring, landmarks, eff_size,
                                    v_offset_earring, h_offset_earring, opacity, is_pair=is_pair)

    if cat in ("Necklace", "Both") and sel_nec:
        necklace = resolve_overlay_image(sel_nec)
        result   = overlay_necklace(result, necklace, landmarks, size_factor, v_offset_necklace, opacity)

    return result.convert("RGB"), "OK"


# ══════════════════════════════════════════════════════════════════════════════
#  Shared result renderer
# ══════════════════════════════════════════════════════════════════════════════

def show_result(source_img: Image.Image, _download_key: str = "download_result"):
    with st.spinner("Processing…"):
        result, msg = process_image(source_img)

    if result is None:
        st.error(msg)
        return

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Original")
        st.image(source_img, use_container_width=True)
    with col_out:
        st.subheader("Try-On Result")
        st.image(result, use_container_width=True)

    st.download_button(
        label="⬇️ Download Result",
        data=pil_to_bytes(result),
        file_name="jewellery_tryon.png",
        mime="image/png",
        use_container_width=True,
        key=_download_key,
    )

    # ── GPT-4o auto recommendation ────────────────────────────────────────────
    st.divider()
    st.subheader("✨ AI Style Recommendation")

    if not _azure_ready():
        st.info("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY in .env to enable.")
        return

    sel_ear = st.session_state.selected_earring
    sel_nec = st.session_state.selected_necklace
    desc    = " + ".join(filter(None, [sel_ear["name"] if sel_ear else None,
                                       sel_nec["name"] if sel_nec else None]))

    # Cache key: selected jewellery combo — avoids re-calling on every Streamlit rerun
    rec_key = f"rec_{desc}"
    if rec_key not in st.session_state:
        with st.spinner("GPT-4o is analysing your try-on…"):
            try:
                st.session_state[rec_key] = get_recommendation(
                    result, st.session_state.active_type, desc
                )
            except Exception as exc:
                st.session_state[rec_key] = f"⚠️ Could not get recommendation: {exc}"

    st.success(st.session_state[rec_key])


# ══════════════════════════════════════════════════════════════════════════════
#  Main — tabs
# ══════════════════════════════════════════════════════════════════════════════

st.title("💍 Jewellery AR Try-On")
st.caption("Powered by MediaPipe · Flux Kontext Pro · Azure OpenAI GPT-4o")

tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Camera Capture"])

with tab1:
    uploaded = st.file_uploader(
        "Upload a front-facing photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="upload",
    )
    if uploaded:
        # Store photo — persists across jewellery selections
        st.session_state.source_image = Image.open(uploaded).convert("RGBA")

    if st.session_state.source_image is not None:
        show_result(st.session_state.source_image, _download_key="download_upload")
    else:
        st.info("Upload your photo once — then browse and select any jewellery to instantly apply it.")

with tab2:
    camera_img = st.camera_input("Take a photo", key="camera")
    if camera_img:
        st.session_state.source_image = Image.open(camera_img).convert("RGBA")

    if st.session_state.source_image is not None:
        show_result(st.session_state.source_image, _download_key="download_camera")
    else:
        st.info("Capture your photo once — then browse and select any jewellery to instantly apply it.")
