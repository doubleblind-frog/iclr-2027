"""
judge.py

Evaluate a generated visual metaphor image across three axes
(metaphor consistency, analogy appropriateness, conceptual integration)
using a two-model VLM judge panel via OpenRouter.

Usage:
    python multimodal_analogy_generation/evaluation/scripts/judge.py multimodal_analogy_generation/iterative_refinement/best/006.png "Going to McDonalds is an adventure."
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Judge models
# ---------------------------------------------------------------------------

JUDGE_MODELS: dict[str, str] = {
    "claude-opus-4-8": "anthropic/claude-opus-4-8",
    "gpt-5.2":         "openai/gpt-5.2",
}

# ---------------------------------------------------------------------------
# Prompt paths
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent.parent / "vlm-judge-prompts"

AXES: dict[str, str] = {
    "metaphor_consistency":    "metaphor-consistency.md",
    "analogy_appropriateness": "analogy-appropriateness.md",
    "conceptual_integration":    "conceptual-integration.md",
}

MAX_TOKENS = 512
TEMPERATURE = 0.0 

def load_prompts(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    """Load all axis prompts from the vlm-judge-prompts directory."""
    prompts: dict[str, str] = {}
    for axis, filename in AXES.items():
        path = prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Judge prompt not found: {path}\n"
                f"Expected prompts directory: {prompts_dir}"
            )
        prompts[axis] = path.read_text(encoding="utf-8").strip()
    return prompts

def encode_image(image_path: str | Path) -> tuple[str, str]:
    """Read a local image file and return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, mime


def image_content_block(image_path: str | Path) -> dict:
    """Build an OpenAI-format image_url content block from a local file."""
    data, mime = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }

def parse_score(text: str) -> Optional[float]:
    """Extract 'Score: X' from model output, return float or None."""
    match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_reasoning(text: str) -> str:
    """Extract 'Reasoning: ...' from model output, falling back to full text."""
    match = re.search(r"Reasoning:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Single judge call
# ---------------------------------------------------------------------------

def call_judge(
    client: OpenAI,
    model_id: str,
    system_prompt: str,
    metaphor_text: str,
    image_path: str | Path,
) -> str:
    """
    Call one judge model with a system prompt, the metaphor text, and the image.
    Returns the raw response string.
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Source Metaphor: {metaphor_text}\n\nTarget Image:",
                    },
                    image_content_block(image_path),
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    choice = response.choices[0]
    raw = choice.message.content
    if raw is None:
        finish_reason = choice.finish_reason
        refusal = getattr(choice.message, "refusal", None)
        raise ValueError(
            f"Null content from {model_id} "
            f"(finish_reason={finish_reason!r}, refusal={refusal!r})"
        )
    return raw


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    image_path: str | Path,
    metaphor_text: str,
    openrouter_api_key: Optional[str] = None,
    prompts_dir: Path = PROMPTS_DIR,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a visual metaphor image across all axes and judge models.

    Returns a nested dict:
        results[axis][model_name] = {
            "score":     float | None,
            "reasoning": str,
            "raw":       str,
        }
    """
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY or pass openrouter_api_key=."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompts = load_prompts(prompts_dir)

    results: dict[str, dict] = {axis: {} for axis in AXES}

    for axis, system_prompt in prompts.items():
        if verbose:
            print(f"\n  [{axis}]")
        for model_name, model_id in JUDGE_MODELS.items():
            if verbose:
                print(f"    {model_name} ... ", end="", flush=True)
            try:
                raw = call_judge(client, model_id, system_prompt, metaphor_text, image_path)
                score = parse_score(raw)
                reasoning = parse_reasoning(raw)
                results[axis][model_name] = {
                    "score":     score,
                    "reasoning": reasoning,
                    "raw":       raw,
                }
                if verbose:
                    score_str = f"{score:.1f}" if score is not None else "N/A"
                    print(f"score={score_str}")
            except Exception as e:
                if verbose:
                    print(f"FAILED ({e})")
                results[axis][model_name] = {
                    "score":     None,
                    "reasoning": None,
                    "raw":       str(e),
                }

    return results

def print_results(metaphor_text: str, results: dict) -> None:
    model_names = list(JUDGE_MODELS.keys())
    col_w = 22
    axis_w = 28

    print(f"\n{'='*70}")
    print(f"Metaphor : {metaphor_text}")
    print(f"{'='*70}")

    header = f"{'Axis':<{axis_w}}" + "".join(f"{m:<{col_w}}" for m in model_names)
    print(f"\n{header}")
    print("-" * len(header))

    for axis, model_scores in results.items():
        scores = [
            f"{model_scores[m]['score']:.1f}" if model_scores[m].get("score") is not None else "N/A"
            for m in model_names
        ]
        print(f"{axis:<{axis_w}}" + "".join(f"{s:<{col_w}}" for s in scores))

    print("-" * len(header))
    avgs = []
    for m in model_names:
        vals = [
            results[ax][m]["score"]
            for ax in results
            if results[ax][m].get("score") is not None
        ]
        avgs.append(f"{sum(vals)/len(vals):.1f}" if vals else "N/A")
    print(f"{'mean':<{axis_w}}" + "".join(f"{a:<{col_w}}" for a in avgs))

    print()
    for axis, model_scores in results.items():
        print(f"\n  [{axis}]")
        for model_name in model_names:
            r = model_scores[model_name].get("reasoning") or "N/A"
            truncated = r[:220] + "…" if len(r) > 220 else r
            print(f"    {model_name}:\n      {truncated}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python judge.py <image_path> <metaphor_text>")
        print('Example: python judge.py output.png "Cars are drivable pesticide spray cans."')
        sys.exit(1)

    _image_path    = sys.argv[1]
    _metaphor_text = " ".join(sys.argv[2:])

    print(f"Image    : {_image_path}")
    print(f"Metaphor : {_metaphor_text}")

    _results = evaluate(_image_path, _metaphor_text)
    print_results(_metaphor_text, _results)
    out_path = Path(_image_path).with_suffix(".judge.json")
    out_path.write_text(json.dumps(_results, indent=2), encoding="utf-8")
    print(f"\nFull results written to: {out_path}")