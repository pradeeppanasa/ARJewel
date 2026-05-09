"""
Flux Kontext Pro overlay via Black Forest Labs API.

Sends the person's photo + jewellery reference image to Flux Kontext Pro,
which returns a photorealistic result with the jewellery naturally blended in.

Docs: https://api.bfl.ml
"""

import base64
import io
import os
import time

import requests
from PIL import Image

_API_BASE   = "https://api.bfl.ml/v1"
_SUBMIT_URL = f"{_API_BASE}/flux-kontext-pro"
_RESULT_URL = f"{_API_BASE}/get_result"

_POLL_INTERVAL = 2    # seconds between polls
_POLL_MAX      = 45   # max attempts (~90 s)


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt(jewellery_type: str, design_name: str) -> str:
    base = "Keep the person's face, skin, hair, and background completely unchanged. Natural studio lighting, photorealistic, high detail."

    prompts = {
        "Earring": (
            f"Add a pair of elegant {design_name} drop earrings to both of the person's ears. "
            "The earrings should appear naturally worn, matching skin-tone shadows. " + base
        ),
        "Necklace": (
            f"Place a beautiful {design_name} necklace around the person's neck and collarbone area. "
            "It should lie naturally on the skin with realistic reflections. " + base
        ),
        "Both": (
            f"Add a matching set: {design_name} drop earrings on both ears and a coordinating "
            f"{design_name} necklace around the neck. Jewellery should look naturally worn. " + base
        ),
    }
    return prompts.get(jewellery_type, f"Add {design_name} jewellery. " + base)


# ── main function ──────────────────────────────────────────────────────────────

def overlay_with_flux(
    person_image: Image.Image,
    jewellery_image: Image.Image | None,
    jewellery_type: str,
    design_name: str,
    api_key: str,
) -> Image.Image:
    """
    Submit to Flux Kontext Pro and poll until the result image is ready.

    Parameters
    ----------
    person_image    : The user's photo (any mode).
    jewellery_image : Optional transparent-BG jewellery PNG used as style reference.
    jewellery_type  : "Earring" | "Necklace" | "Both"
    design_name     : Free-text label used in the prompt (e.g. "Design 1 gold").
    api_key         : BFL_API_KEY from environment / .env.

    Returns the result as an RGB PIL image.
    """
    prompt = _build_prompt(jewellery_type, design_name)
    headers = {"X-Key": api_key, "Content-Type": "application/json"}

    payload: dict = {
        "prompt": prompt,
        "input_image": _to_b64(person_image),
        "output_format": "jpeg",
    }

    # If a jewellery reference image is available, include it so Flux can
    # match the exact design rather than hallucinate a generic piece.
    if jewellery_image is not None:
        payload["reference_image"] = _to_b64(jewellery_image, fmt="PNG")

    # ── submit job ─────────────────────────────────────────────────────────────
    # Honour HTTP_PROXY / HTTPS_PROXY env vars if set (corporate networks)
    proxies = {
        "http":  os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
    }
    proxies = {k: v for k, v in proxies.items() if v}  # drop None values

    resp = requests.post(_SUBMIT_URL, headers=headers, json=payload,
                         timeout=45, proxies=proxies or None)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    # ── poll for result ────────────────────────────────────────────────────────
    for attempt in range(_POLL_MAX):
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            _RESULT_URL,
            params={"id": job_id},
            headers={"X-Key": api_key},
            timeout=15,
            proxies=proxies or None,
        )
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status", "")

        if status == "Ready":
            img_url = data["result"]["sample"]
            img_bytes = requests.get(img_url, timeout=30).content
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if status in ("Error", "Failed", "Content Moderated"):
            raise RuntimeError(f"Flux Kontext Pro job failed — status: {status}")

    raise TimeoutError(
        f"Flux Kontext Pro did not return a result after {_POLL_MAX * _POLL_INTERVAL}s"
    )
