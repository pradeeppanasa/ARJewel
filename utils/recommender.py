"""
Azure OpenAI GPT-4o jewellery recommendation.

Sends the try-on result image to GPT-4o vision and returns
a short, friendly style recommendation.
"""

import base64
import io
import os

from openai import AzureOpenAI
from PIL import Image


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _img_to_b64_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


_SYSTEM_PROMPT = (
    "You are a professional jewellery stylist. "
    "When shown a photo of a person wearing jewellery, give a concise (3-5 sentence) "
    "personalised style recommendation. Comment on how well the piece complements "
    "their features, suggest matching outfits or occasions, and optionally recommend "
    "one alternative style they might also enjoy. Be warm, positive, and specific."
)


def get_recommendation(
    result_image: Image.Image,
    jewellery_type: str,
    design_name: str,
) -> str:
    """
    Returns a GPT-4o style recommendation string for the try-on result.
    Raises on API error — callers should handle and show a fallback message.
    """
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    client = _client()

    user_text = (
        f"I am trying on {jewellery_type.lower()} ({design_name}). "
        "Please give me your personalised style recommendation based on this photo."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": _img_to_b64_url(result_image), "detail": "low"},
                    },
                ],
            },
        ],
        max_tokens=300,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
