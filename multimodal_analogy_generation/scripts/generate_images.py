"""
generate_images.py

Generate images from the visual prompts produced by zero_shot_prompt.py or
cot_prompt.py.

Usage:
    python multimodal_analogy_generation/scripts/generate_images.py --model flux-klein  --condition zero_shot
    python multimodal_analogy_generation/scripts/generate_images.py --model stable-core  --condition zero_shot
    python multimodal_analogy_generation/scripts/generate_images.py --model gpt-image  --condition cot

Inputs:
    multimodal_analogy_generation/zero_shot_exploration/{condition}_outputs.jsonl

Outputs:
    multimodal_analogy_generation/zero_shot_exploration/images/{condition}/{model}/{id}.png

Model notes:
    flux-klein  — FLUX.2 [klein] via OpenRouter.
    stable-core — Stability AI Stable Image Core v1.1 on Bedrock.
    gpt-image   — OpenAI GPT Image 1.5, quality=medium.

Authentication (.env):
    AWS_BEARER_TOKEN_BEDROCK   for stable-core
    OPENROUTER_API_KEY         for flux-klein
    OPENAI_API_KEY             for gpt-image
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import boto3
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---------------------------------------------------------

INPUT_DIR = Path("multimodal_analogy_generation/zero_shot_exploration")
OUTPUT_DIR = INPUT_DIR / "images"
BEDROCK_REGION = "us-west-2"

WIDTH = HEIGHT = 1024
MAX_RETRIES = 4

MODELS = {
    "flux-klein": {
        "display": "FLUX.2 Klein via OpenRouter",
        "provider": "openrouter",
        "model_id": "black-forest-labs/flux.2-klein-4b",
    },
    "stable-core": {
        "display": "Stability AI Stable Image Core v1.1 via Bedrock",
        "provider": "bedrock",
        "model_id": "stability.stable-image-core-v1:1",
    },
    "gpt-image": {
        "display": "OpenAI GPT Image 1.5 (quality=medium)",
        "provider": "openai",
        "model_id": "gpt-image-1.5",
    },
}

# --- Retry helper ---------------------------------------------------------


def _retry(fn):
    """Run fn() with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            transient = False
            if isinstance(e, requests.HTTPError):
                status = e.response.status_code if e.response is not None else 0
                transient = status in (408, 429, 500, 502, 503, 504)
            elif isinstance(e, requests.Timeout):
                transient = True
            elif isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code", "")
                transient = code in ("ThrottlingException", "ServiceUnavailableException")
            else:
                msg = str(e).lower()
                transient = "rate" in msg or "timeout" in msg or "throttl" in msg

            if transient and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  transient error ({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise


# --- Backends -------------------------------------------------------------


def gen_flux_klein(prompt: str) -> bytes:
    """FLUX.2 Klein via OpenRouter chat-completions with modalities=['image']."""
    api_key = os.environ["OPENROUTER_API_KEY"]

    def _call():
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODELS["flux-klein"]["model_id"],
                "modalities": ["image"],
                "messages": [{"role": "user", "content": prompt}],
                "image_config": {"aspect_ratio": "1:1"},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()

    data = _retry(_call)
    images = data["choices"][0]["message"].get("images", [])
    if not images:
        raise RuntimeError(f"OpenRouter returned no images: {data}")
    data_url = images[0]["image_url"]["url"]
    return base64.b64decode(data_url.split(",", 1)[1])

def gen_stable_core(client, prompt: str) -> bytes:
    """Stability AI Stable Image Core via Bedrock."""
    body = {
        "prompt": prompt,
        "mode": "text-to-image",
        "aspect_ratio": "1:1",
        "output_format": "png",
    }

    def _call():
        return client.invoke_model(
            modelId=MODELS["stable-core"]["model_id"],
            body=json.dumps(body),
        )

    response = _retry(_call)
    payload = json.loads(response["body"].read())
    if "images" not in payload or not payload["images"]:
        raise RuntimeError(f"Stable Image Core returned no images: {payload}")
    return base64.b64decode(payload["images"][0])


def gen_gpt_image(prompt: str) -> bytes:
    """OpenAI GPT Image 1.5 at medium quality."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _call():
        return client.images.generate(
            model=MODELS["gpt-image"]["model_id"],
            prompt=prompt,
            size=f"{WIDTH}x{HEIGHT}",
            quality="medium",
            n=1,
        )

    result = _retry(_call)
    return base64.b64decode(result.data[0].b64_json)


# --- Main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from visual prompts (zero-shot or CoT).",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Image generation model",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["zero_shot", "cot"],
        help="Which prompts JSONL to read from",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Optional cap on number of new images this run",
    )
    args = parser.parse_args()

    info = MODELS[args.model]
    print(f"Model:     {info['display']}")
    print(f"Condition: {args.condition}")

    input_path = INPUT_DIR / f"{args.condition}_outputs.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    out_dir = OUTPUT_DIR / args.condition / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:    {out_dir}\n")

    bedrock_client = None
    if info["provider"] == "bedrock":
        bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    new_count = 0
    skipped = 0
    failed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            mid = entry["id"]
            prompt = (entry.get("visual_elaboration") or "").strip()

            if not prompt:
                print(f"[{mid}] SKIPPED: empty visual_elaboration")
                continue

            out_path = out_dir / f"{mid}.png"
            if out_path.exists():
                skipped += 1
                continue

            try:
                if args.model == "flux-klein":
                    img = gen_flux_klein(prompt)
                elif args.model == "stable-core":
                    img = gen_stable_core(bedrock_client, prompt)
                elif args.model == "gpt-image":
                    img = gen_gpt_image(prompt)
            except Exception as e:
                failed += 1
                print(f"[{mid}] FAILED: {type(e).__name__}: {e}")
                continue

            out_path.write_bytes(img)
            new_count += 1
            print(f"[{mid}] OK -> {out_path.name}")

            if args.max is not None and new_count >= args.max:
                print(f"\nReached --max limit ({args.max}), stopping.")
                break

    print(f"\nDone. New: {new_count}  Skipped (existed): {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()