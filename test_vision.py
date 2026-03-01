#!/usr/bin/env python3
"""Quick script to compare cloud vision models on sample images before committing to a full index run.

Usage:
    python3 test_vision.py /path/to/image.jpg [/path/to/image2.png ...]

Set API keys as environment variables before running:
    export GEMINI_API_KEY=...        # Gemini
    export OPENAI_API_KEY=...        # GPT-4o-mini
    export OPENROUTER_API_KEY=...    # Qwen2-VL via OpenRouter
    export GROQ_API_KEY=...          # Llama Vision via Groq
    export ANTHROPIC_API_KEY=...     # Claude Haiku
"""

import base64
import sys

import litellm

MODELS = [
    ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    # ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    # ("openrouter/qwen/qwen2-vl-7b-instruct", "OPENROUTER_API_KEY"),
    # ("groq/llama-3.2-11b-vision-preview", "GROQ_API_KEY"),
    ("bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0", "AWS credentials"),
    ("bedrock/eu.anthropic.claude-sonnet-4-6", "AWS credentials"),
    ("bedrock/eu.anthropic.claude-opus-4-6-v1", "AWS credentials"),
]

PROMPT = "Describe the content of this image in detail, including any text visible."


def test_image(img_path: str) -> None:
    print(f"\n{'#'*70}")
    print(f"Image: {img_path}")
    print(f"{'#'*70}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    # Detect MIME type from extension
    ext = img_path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp"}.get(ext, "image/png")

    for model, key_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}  (needs {key_name})")
        print(f"{'='*60}")
        try:
            r = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            print(r.choices[0].message.content)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_vision.py <image> [image2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        test_image(path)
