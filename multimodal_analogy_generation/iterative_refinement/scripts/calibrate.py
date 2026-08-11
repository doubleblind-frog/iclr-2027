"""
calibrate.py

Holdout-set calibration for the visual metaphor refinement pipeline.
Compares three models, three k values and three epsilon values across
25 held-out metaphors, using the VLM judge from judge.py to evaluate
the best image produced by each configuration.

Usage:
python multimodal_analogy_generation/iterative_refinement/scripts/calibrate.py --models openrouter/openai/gpt-5.5 bedrock-mantle:us-east-1/google.gemma-4-31b bedrock/eu.anthropic.claude-sonnet-4-6 --k_values 5 10 15 --epsilon_values 0.05 0.10 0.20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from openai import OpenAI

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from multimodal_analogy_generation.scripts.extract import extract_metaphor_graphs, _configure_lm
from multimodal_analogy_generation.scripts.generate_images import gen_gpt_image
from multimodal_analogy_generation.iterative_refinement.scripts.refine import (
    build_diagnostic,
    REFINE_PROMPT_TEMPLATE,
    _clean_output,
    RoundDiagnostic,
    DOMAIN_MATCH_THRESHOLD,
)
from multimodal_analogy_generation.evaluation.scripts.judge import (
    evaluate as judge_evaluate,
    PROMPTS_DIR as JUDGE_PROMPTS_DIR,
    JUDGE_MODELS,
    AXES as JUDGE_AXES,
)

load_dotenv()

_REAL_OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
_REAL_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("multimodal_analogy_generation/iterative_refinement")
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
HOLDOUT_PATH    = Path("multimodal_analogy_generation/holdout-metaphors.jsonl")
ZERO_SHOT_PATH  = Path("multimodal_analogy_generation/zero_shot_exploration/zero_shot_outputs.jsonl")
GOLD_CSV        = Path("multimodal_analogy_generation/ai_metaphor_domains.csv")

REFINER_REGION      = "eu-central-1"
REFINER_TEMPERATURE = 0.7
REFINER_MAX_TOKENS  = 200
REFINER_MAX_RETRIES = 4

PULLBACK_THRESHOLD = 0.40
DEFAULT_PATIENCE   = 1
EMBEDDER_MODEL     = "all-mpnet-base-v2"

MANTLE_API_KEY_ENV       = "AWS_BEARER_TOKEN_BEDROCK"
MANTLE_REGION            = "us-east-2"
MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/openai/v1"

MANTLE_OSS_REASONING_EFFORT = "low"
_REASONING_MODEL_PREFIXES = ("openai.gpt-oss",)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def model_slug(model_str: str) -> str:
    """Convert a model string to a safe filesystem name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_str)


def _avg(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _is_mantle(model_str: str) -> bool:
    """True for both bedrock-mantle/<id> and bedrock-mantle:<region>/<id>."""
    return model_str.startswith("bedrock-mantle/") or (
        model_str.startswith("bedrock-mantle:") and "/" in model_str
    )


def _mantle_region(model_str: str) -> str:
    """Extract the AWS region from a bedrock-mantle model string.

    bedrock-mantle/<id>            → MANTLE_REGION (default)
    bedrock-mantle:<region>/<id>   → <region>
    """
    if model_str.startswith("bedrock-mantle:"):
        return model_str[len("bedrock-mantle:"):].split("/", 1)[0]
    return MANTLE_REGION


def _mantle_model_id(model_str: str) -> str:
    """Extract the bare model ID from a bedrock-mantle model string."""
    if model_str.startswith("bedrock-mantle:"):
        return model_str.split("/", 1)[1]
    return model_str[len("bedrock-mantle/"):]


def _mantle_base_url(model_str: str) -> str:
    """Build the bedrock-mantle base URL for a model string."""
    region   = _mantle_region(model_str)
    model_id = _mantle_model_id(model_str)
    path = "/v1" if model_id.startswith("openai.gpt-oss") else "/openai/v1"
    return f"https://bedrock-mantle.{region}.api.aws{path}"


def _is_reasoning_model(model_id: str) -> bool:
    """True for GPT-OSS and other reasoning models that accept a reasoning effort param."""
    return any(model_id.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _set_openai_env(api_key: str, api_base: str = "") -> None:
    """Set OPENAI_API_KEY and optionally OPENAI_API_BASE."""
    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    else:
        os.environ.pop("OPENAI_API_BASE", None)


def _restore_openai_env() -> None:
    """Restore OPENAI_API_KEY / OPENAI_API_BASE to their values at import time."""
    _set_openai_env(_REAL_OPENAI_API_KEY, _REAL_OPENAI_API_BASE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_holdout(path: Path) -> list[dict]:
    """Load [{id, metaphor, type}, ...] from holdout JSONL."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_zero_shot_elaborations(path: Path) -> dict[str, str]:
    """Map id -> visual_elaboration from zero_shot_outputs.jsonl."""
    elab: dict[str, str] = {}
    if not path.exists():
        print(f"Warning: zero-shot elaborations not found at {path}")
        return elab
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = str(row.get("id", "")).strip()
            ve  = (row.get("visual_elaboration") or "").strip()
            if eid and ve:
                elab[eid] = ve
    return elab


def load_gold_domains(path: Path) -> dict[str, list[str]]:
    """Map id -> [domain_a, domain_b]."""
    gold: dict[str, list[str]] = {}
    if not path.exists():
        return gold
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols   = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        id_col = cols.get("id")
        a_col  = cols.get("domain 1") or cols.get("domain1")
        b_col  = cols.get("domain 2") or cols.get("domain2")
        if not (a_col and b_col):
            a_col = next((cols[c] for c in cols if "target" in c), None)
            b_col = next((cols[c] for c in cols if "source" in c), None)
        if not (id_col and a_col and b_col):
            print(f"Warning: could not resolve domain columns in {path.name}; "
                  "domain-match logging disabled.")
            return gold
        for row in reader:
            rid = str(row[id_col]).strip()
            gold[rid] = [
                str(row[a_col]).strip().lower(),
                str(row[b_col]).strip().lower(),
            ]
    return gold


def make_embedder(model_name: str):
    """Return a sentence-transformer embedder or None on failure."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model  = SentenceTransformer(model_name, device=device)
        def embed(texts):
            return model.encode(texts, normalize_embeddings=True)
        print(f"Embedder: {model_name} on {device}")
        return embed
    except Exception as e:
        print(f"Warning: embedder unavailable ({e}); domain-match logging disabled.")
        return None


# ---------------------------------------------------------------------------
# Refiners (Bedrock Converse, Bedrock Mantle, OpenRouter)
# ---------------------------------------------------------------------------

class BedrockRefiner:
    """Calls the Bedrock Converse API directly via boto3."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client   = boto3.client("bedrock-runtime", region_name=REFINER_REGION)

    def refine(self, previous_elaboration: str, diagnostic: RoundDiagnostic) -> str:
        prompt = REFINE_PROMPT_TEMPLATE.format(
            previous_elaboration=previous_elaboration,
            feedback=diagnostic.to_feedback_text(),
        )
        for attempt in range(REFINER_MAX_RETRIES):
            try:
                resp = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "temperature": REFINER_TEMPERATURE,
                        "maxTokens":   REFINER_MAX_TOKENS,
                    },
                )
                raw = resp["output"]["message"]["content"][0]["text"]
                return _clean_output(raw)
            except ClientError as e:
                code      = e.response.get("Error", {}).get("Code", "")
                transient = code in ("ThrottlingException", "ServiceUnavailableException")
                if transient and attempt < REFINER_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise


class BedrockMantleRefiner:
    """Calls models on Bedrock via the bedrock-mantle endpoint."""

    def __init__(self, model_str: str):
        self.model_id      = _mantle_model_id(model_str)
        self.is_reasoning  = _is_reasoning_model(self.model_id)
        api_key = os.environ.get(MANTLE_API_KEY_ENV, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {MANTLE_API_KEY_ENV!r} is not set. "
                "Generate a long-term API key at the Amazon Bedrock console "
                "(API keys section) and add it to your .env as "
                "AWS_BEARER_TOKEN_BEDROCK=<key>."
            )
        self.client = OpenAI(
            base_url=_mantle_base_url(model_str),
            api_key=api_key,
        )

    def refine(self, previous_elaboration: str, diagnostic: RoundDiagnostic) -> str:
        prompt = REFINE_PROMPT_TEMPLATE.format(
            previous_elaboration=previous_elaboration,
            feedback=diagnostic.to_feedback_text(),
        )
        for attempt in range(REFINER_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=self.model_id,
                    input=prompt,
                    max_output_tokens=REFINER_MAX_TOKENS,
                )
                if self.is_reasoning:
                    kwargs["reasoning"] = {"effort": MANTLE_OSS_REASONING_EFFORT}
                resp = self.client.responses.create(**kwargs)
                raw = resp.output_text
                if not raw:
                    raise ValueError(
                        f"Empty output from bedrock-mantle "
                        f"(status={getattr(resp, 'status', 'unknown')!r})"
                    )
                return _clean_output(raw)
            except Exception as e:
                if attempt < REFINER_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise


class OpenRouterRefiner:
    """Calls the refiner via OpenRouter's OpenAI-compatible API."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client   = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def refine(self, previous_elaboration: str, diagnostic: RoundDiagnostic) -> str:
        prompt = REFINE_PROMPT_TEMPLATE.format(
            previous_elaboration=previous_elaboration,
            feedback=diagnostic.to_feedback_text(),
        )
        for attempt in range(REFINER_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=REFINER_MAX_TOKENS,
                    temperature=REFINER_TEMPERATURE,
                )
                raw = resp.choices[0].message.content
                if raw is None:
                    raise ValueError(
                        f"Null content from {self.model_id} "
                        f"(finish_reason={resp.choices[0].finish_reason!r})"
                    )
                return _clean_output(raw)
            except Exception as e:
                if attempt < REFINER_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise


def make_refiner(
    model_str: str,
) -> BedrockRefiner | BedrockMantleRefiner | OpenRouterRefiner:
    """Construct the correct refiner from a model string."""
    if _is_mantle(model_str):
        return BedrockMantleRefiner(model_str)
    if model_str.startswith("bedrock/converse/"):
        return BedrockRefiner(model_str[len("bedrock/converse/"):])
    if model_str.startswith("bedrock/"):
        return BedrockRefiner(model_str[len("bedrock/"):])
    if model_str.startswith("openrouter/"):
        return OpenRouterRefiner(model_str[len("openrouter/"):])
    raise ValueError(
        f"Unknown model prefix in '{model_str}'. "
        "Expected one of: 'bedrock-mantle/', 'bedrock-mantle:<region>/', "
        "'bedrock/converse/', 'bedrock/', 'openrouter/'."
    )


# ---------------------------------------------------------------------------
# LM configuration wrapper
# ---------------------------------------------------------------------------

def _configure_lm_for_model(model_str: str) -> None:
    """Wrapper around _configure_lm that handles all model prefix types."""
    if _is_mantle(model_str):
        model_id = _mantle_model_id(model_str)
        api_key  = os.environ.get(MANTLE_API_KEY_ENV, "")
        if not api_key:
            raise EnvironmentError(
                f"{MANTLE_API_KEY_ENV!r} is not set. "
                "Cannot configure LM for bedrock-mantle extraction."
            )
        base_url = _mantle_base_url(model_str)
        _set_openai_env(api_key, base_url)
        _configure_lm(f"openai/{model_id}")
    else:
        _restore_openai_env()
        _configure_lm(model_str)


# ---------------------------------------------------------------------------
# Stopping-criterion simulation
# ---------------------------------------------------------------------------

def simulate_stopping(
    round_scores: list[float],
    k: int,
    epsilon: float,
    patience: int = DEFAULT_PATIENCE,
) -> int:
    """
    Given pullback scores for consecutive rounds (index 0 = round 1),
    return the 0-based index of the last round that would have run under
    the (k, epsilon, patience) stopping criterion.
    """
    cap  = min(k, len(round_scores))
    prev = None
    streak = 0
    for i in range(cap):
        score = round_scores[i]
        if prev is not None:
            if abs(score - prev) < epsilon:
                streak += 1
            else:
                streak = 0
            if streak >= patience:
                return i
        prev = score
    return cap - 1


def best_round_idx(round_scores: list[float], up_to_idx: int) -> int:
    """Index of the highest-pullback round up to (and including) `up_to_idx`."""
    subset = round_scores[: up_to_idx + 1]
    return max(range(len(subset)), key=lambda i: subset[i])


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

class Checkpointer:
    """
    Persists two caches to disk after every write:
        rounds      [slug][mid] = list of round records
        judge_cache [image_path] = judge_evaluate() output dict
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = base_dir / "rounds_data.json"
        self._judge_path  = base_dir / "judge_cache.json"
        self.rounds: dict      = self._load(self._rounds_path)
        self.judge_cache: dict = self._load(self._judge_path)

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_rounds(self):
        self._write(self._rounds_path, self.rounds)

    def save_judge_cache(self):
        self._write(self._judge_path, self.judge_cache)

    def _write(self, path: Path, data: dict):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Phase 1 — Refinement
# ---------------------------------------------------------------------------

def run_refinement_for_model(
    model_str:    str,
    metaphors:    list[dict],
    elaborations: dict[str, str],
    gold:         dict[str, list[str]],
    embed,
    max_k:        int,
    ckpt:         Checkpointer,
) -> None:
    """
    For every metaphor, run up to max_k refinement rounds using model_str
    for both extraction and refinement. Results are checkpointed after each round.
    """
    slug = model_slug(model_str)
    ckpt.rounds.setdefault(slug, {})

    images_root = ckpt.base_dir / "images" / slug
    images_root.mkdir(parents=True, exist_ok=True)

    _configure_lm_for_model(model_str)
    refiner = make_refiner(model_str)

    is_mantle   = _is_mantle(model_str)
    mantle_key  = os.environ.get(MANTLE_API_KEY_ENV, "") if is_mantle else ""
    mantle_base = _mantle_base_url(model_str) if is_mantle else ""

    total = len(metaphors)
    for idx, entry in enumerate(metaphors, 1):
        mid      = str(entry["id"])
        metaphor = entry["metaphor"]

        existing = ckpt.rounds[slug].get(mid, [])
        if len(existing) >= max_k:
            print(f"  [{idx}/{total}] {mid} — already {len(existing)} rounds, skipping.")
            continue

        elaboration = elaborations.get(mid, metaphor)
        if existing:
            elaboration = existing[-1].get("elaboration_next", elaboration)

        gold_domains = gold.get(mid, [])
        print(f"  [{idx}/{total}] {mid}: {metaphor[:55]}...")

        for r in range(len(existing) + 1, max_k + 1):
            img_path = images_root / f"{mid}_k{r:02d}.png"

            # --- image generation ---
            if not img_path.exists():
                try:
                    if is_mantle:
                        _restore_openai_env()
                    img_bytes = gen_gpt_image(elaboration)
                    img_path.write_bytes(img_bytes)
                    if is_mantle:
                        _set_openai_env(mantle_key, mantle_base)
                except Exception as e:
                    if is_mantle:
                        _set_openai_env(mantle_key, mantle_base)
                    print(f"    round {r} image generation FAILED: {e}")
                    break

            # --- domain extraction + pullback ---
            try:
                result = extract_metaphor_graphs(img_path, threshold=PULLBACK_THRESHOLD)
            except Exception as e:
                print(f"    round {r} extraction FAILED: {e}")
                break

            score = float(result.get("pullback_score", 0.0))

            diag = build_diagnostic(
                metaphor=metaphor,
                extraction_result=result,
                gold_domains=gold_domains,
                embed=embed,
            )

            next_elab = elaboration
            if r < max_k:
                try:
                    next_elab = refiner.refine(elaboration, diag)
                except Exception as e:
                    print(f"    round {r} refine FAILED: {e}")

            existing.append({
                "round":            r,
                "pullback_score":   score,
                "image_path":       str(img_path),
                "elaboration":      elaboration,
                "elaboration_next": next_elab,
                "extracted_target": result.get("target_domain", "none"),
                "extracted_source": result.get("source_domain", "none"),
                "coverage":         diag.coverage,
                "concepts_matched": sum(1 for m in diag.concept_matches if m.get("match")),
            })
            ckpt.rounds[slug][mid] = existing
            ckpt.save_rounds()

            traj = " -> ".join(f"{r['pullback_score']:.3f}" for r in existing)
            print(f"    round {r}: pullback={score:.3f}  [{traj}]")
            elaboration = next_elab

    if is_mantle:
        _restore_openai_env()


# ---------------------------------------------------------------------------
# Phase 2 — Judge evaluation with caching
# ---------------------------------------------------------------------------

def get_judge_scores(
    image_path:    str,
    metaphor_text: str,
    ckpt:          Checkpointer,
    verbose:       bool = False,
) -> dict:
    """Return judge_evaluate() output, using the on-disk cache if available."""
    if image_path in ckpt.judge_cache:
        return ckpt.judge_cache[image_path]

    results = judge_evaluate(
        image_path=image_path,
        metaphor_text=metaphor_text,
        prompts_dir=JUDGE_PROMPTS_DIR,
        verbose=verbose,
    )
    ckpt.judge_cache[image_path] = results
    ckpt.save_judge_cache()
    return results


def mean_score_from_judge(judge_results: dict) -> Optional[float]:
    """Grand mean across all axes × judge models."""
    scores = [
        entry["score"]
        for axis_data in judge_results.values()
        for entry in axis_data.values()
        if entry.get("score") is not None
    ]
    return _avg(scores)


# ---------------------------------------------------------------------------
# Phase 2 — Compile results for all (model, k, epsilon) combinations
# ---------------------------------------------------------------------------

def compile_results(
    metaphors:      list[dict],
    models:         list[str],
    k_values:       list[int],
    epsilon_values: list[float],
    patience:       int,
    ckpt:           Checkpointer,
) -> dict:
    """
    For every (model, k, epsilon, metaphor) combination:
      1. Simulate the stopping criterion on stored round data.
      2. Identify the best round by pullback score within that window.
      3. Run the VLM judge on that round's image (cached).
    """
    results: dict = {}
    total_conditions = len(models) * len(k_values) * len(epsilon_values) * len(metaphors)
    done = 0

    for model_str in models:
        slug = model_slug(model_str)
        results[slug] = {}

        for k in k_values:
            results[slug][str(k)] = {}

            for eps in epsilon_values:
                results[slug][str(k)][str(eps)] = {}

                for entry in metaphors:
                    mid      = str(entry["id"])
                    metaphor = entry["metaphor"]

                    rounds = ckpt.rounds.get(slug, {}).get(mid, [])
                    if not rounds:
                        results[slug][str(k)][str(eps)][mid] = {"error": "no_rounds"}
                        done += 1
                        continue

                    scores = [r["pullback_score"] for r in rounds]

                    last_idx = simulate_stopping(scores, k, eps, patience)
                    best_idx = best_round_idx(scores, last_idx)
                    best     = rounds[best_idx]
                    img_path = best["image_path"]

                    try:
                        judge = get_judge_scores(img_path, metaphor, ckpt)
                    except Exception as e:
                        print(f"  Judge FAILED [{slug} k={k} eps={eps} {mid}]: {e}")
                        judge = {}

                    done += 1
                    print(
                        f"  [{done}/{total_conditions}] {slug} k={k} ε={eps} {mid}"
                        f" → round {best['round']} pullback={scores[best_idx]:.3f}"
                        f" judge_mean={_fmt(mean_score_from_judge(judge))}"
                    )

                    results[slug][str(k)][str(eps)][mid] = {
                        "selected_round":    best["round"],
                        "last_round":        rounds[last_idx]["round"],
                        "n_rounds_run":      last_idx + 1,
                        "pullback_score":    scores[best_idx],
                        "image_path":        img_path,
                        "elaboration":       best["elaboration"],
                        "extracted_target":  best["extracted_target"],
                        "extracted_source":  best["extracted_source"],
                        "coverage":          best.get("coverage", 0.0),
                        "concepts_matched":  best.get("concepts_matched", 0),
                        "judge_scores":      judge,
                        "mean_judge_score":  mean_score_from_judge(judge),
                    }

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary_csv(path: Path, results: dict) -> None:
    """One row per (model, k, epsilon): aggregated mean scores."""
    axes        = list(JUDGE_AXES.keys())
    judge_names = list(JUDGE_MODELS.keys())

    fields = (
        ["model_slug", "k", "epsilon",
         "n_metaphors", "mean_pullback", "mean_judge_overall"]
        + [f"judge_{ax}" for ax in axes]
        + [f"judge_model_{m}" for m in judge_names]
    )

    rows = []
    for slug, k_data in results.items():
        for k, eps_data in k_data.items():
            for eps, mid_data in eps_data.items():
                pullbacks: list[float] = []
                overall:   list[float] = []
                ax_scores  = {ax: [] for ax in axes}
                mdl_scores = {m:  [] for m in judge_names}

                for cond in mid_data.values():
                    if "error" in cond:
                        continue
                    pullbacks.append(cond["pullback_score"])
                    if cond["mean_judge_score"] is not None:
                        overall.append(cond["mean_judge_score"])

                    for ax in axes:
                        for jm in judge_names:
                            s = cond["judge_scores"].get(ax, {}).get(jm, {}).get("score")
                            if s is not None:
                                ax_scores[ax].append(s)
                                mdl_scores[jm].append(s)

                rows.append({
                    "model_slug":         slug,
                    "k":                  k,
                    "epsilon":            eps,
                    "n_metaphors":        len([v for v in mid_data.values() if "error" not in v]),
                    "mean_pullback":      _fmt(_avg(pullbacks)),
                    "mean_judge_overall": _fmt(_avg(overall)),
                    **{f"judge_{ax}": _fmt(_avg(ax_scores[ax])) for ax in axes},
                    **{f"judge_model_{m}": _fmt(_avg(mdl_scores[m])) for m in judge_names},
                })

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["model_slug"], r["k"], r["epsilon"])))


def write_per_metaphor_csv(path: Path, results: dict, metaphors: list[dict]) -> None:
    """One row per (model, k, epsilon, metaphor_id): all individual scores."""
    axes        = list(JUDGE_AXES.keys())
    judge_names = list(JUDGE_MODELS.keys())

    base_fields = [
        "model_slug", "k", "epsilon", "metaphor_id", "metaphor_type",
        "selected_round", "n_rounds_run", "pullback_score", "mean_judge_score",
        "extracted_target", "extracted_source",
    ]
    ax_fields   = [f"judge_{ax}" for ax in axes]
    mdl_fields  = [f"judge_model_{m}" for m in judge_names]
    cell_fields = [f"{ax}__{jm}" for ax in axes for jm in judge_names]
    fields = base_fields + ax_fields + mdl_fields + cell_fields

    type_lookup = {str(m["id"]): m.get("type", "") for m in metaphors}

    rows = []
    for slug, k_data in results.items():
        for k, eps_data in k_data.items():
            for eps, mid_data in eps_data.items():
                for mid, cond in mid_data.items():
                    if "error" in cond:
                        continue
                    js = cond.get("judge_scores", {})

                    ax_means  = {}
                    mdl_means = {m: [] for m in judge_names}
                    cells     = {}

                    for ax in axes:
                        ax_vals = []
                        for jm in judge_names:
                            s = js.get(ax, {}).get(jm, {}).get("score")
                            cells[f"{ax}__{jm}"] = _fmt(s) if s is not None else "N/A"
                            if s is not None:
                                ax_vals.append(s)
                                mdl_means[jm].append(s)
                        ax_means[f"judge_{ax}"] = _fmt(_avg(ax_vals))

                    rows.append({
                        "model_slug":       slug,
                        "k":                k,
                        "epsilon":          eps,
                        "metaphor_id":      mid,
                        "metaphor_type":    type_lookup.get(mid, ""),
                        "selected_round":   cond["selected_round"],
                        "n_rounds_run":     cond["n_rounds_run"],
                        "pullback_score":   _fmt(cond["pullback_score"]),
                        "mean_judge_score": _fmt(cond["mean_judge_score"]),
                        "extracted_target": cond["extracted_target"],
                        "extracted_source": cond["extracted_source"],
                        **ax_means,
                        **{f"judge_model_{m}": _fmt(_avg(mdl_means[m])) for m in judge_names},
                        **cells,
                    })

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs=3, required=True, metavar="MODEL",
        help=(
            "Three model strings. Supported prefixes: "
            "'bedrock/<id>' (Converse API, eu-central-1), "
            "'bedrock/converse/<id>' (LiteLLM converse route), "
            "'bedrock-mantle/<id>' (bedrock-mantle, us-east-2; requires AWS_BEARER_TOKEN_BEDROCK), "
            "'bedrock-mantle:<region>/<id>' (bedrock-mantle, explicit region), "
            "'openrouter/<id>' (requires OPENROUTER_API_KEY)."
        ),
    )
    parser.add_argument(
        "--k_values", nargs=3, type=int, required=True, metavar="K",
        help="Three max-round values to compare (e.g. 5 10 15).",
    )
    parser.add_argument(
        "--epsilon_values", nargs=3, type=float, required=True, metavar="EPS",
        help="Three early-stopping epsilon values (e.g. 0.05 0.10 0.20).",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"Consecutive flat rounds required to stop (default: {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--holdout", default=str(HOLDOUT_PATH),
        help=f"Path to holdout JSONL (default: {HOLDOUT_PATH}).",
    )
    parser.add_argument(
        "--skip_refinement", action="store_true",
        help="Skip Phase 1 and go straight to judging (useful if rounds_data.json exists).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N metaphors (useful for smoke-testing).",
    )
    args = parser.parse_args()

    max_k = max(args.k_values)

    if any(_is_mantle(m) for m in args.models):
        mantle_key = os.environ.get(MANTLE_API_KEY_ENV, "")
        if mantle_key:
            print(f"  {MANTLE_API_KEY_ENV}: set ({mantle_key[:6]}...)")
        else:
            print(f"  WARNING: {MANTLE_API_KEY_ENV} is NOT set — mantle calls will fail.")

    print("=" * 65)
    print("Calibration run")
    print(f"  Models:   {args.models}")
    print(f"  k values: {args.k_values}  (running up to k={max_k})")
    print(f"  Epsilons: {args.epsilon_values}")
    print(f"  Patience: {args.patience}")
    print(f"  Holdout:  {args.holdout}")
    print("=" * 65)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpointer(CALIBRATION_DIR)

    metaphors    = load_holdout(Path(args.holdout))
    if args.limit:
        metaphors = metaphors[: args.limit]
    elaborations = load_zero_shot_elaborations(ZERO_SHOT_PATH)
    gold         = load_gold_domains(GOLD_CSV)
    embed        = make_embedder(EMBEDDER_MODEL)

    missing_elab = [m["id"] for m in metaphors if m["id"] not in elaborations]
    if missing_elab:
        print(
            f"\nNote: {len(missing_elab)} metaphors have no zero-shot elaboration "
            f"({missing_elab}).\n      Using raw metaphor text as the initial prompt.\n"
        )

    if not args.skip_refinement:
        for model_str in args.models:
            print(f"\n{'='*65}")
            print(f"Phase 1 — Refinement with: {model_str}")
            print(f"{'='*65}\n")
            run_refinement_for_model(
                model_str=model_str,
                metaphors=metaphors,
                elaborations=elaborations,
                gold=gold,
                embed=embed,
                max_k=max_k,
                ckpt=ckpt,
            )
    else:
        print("\n[Skipping Phase 1 — using existing rounds_data.json]\n")

    print(f"\n{'='*65}")
    print("Phase 2 — Judge evaluation")
    print(f"{'='*65}\n")

    results = compile_results(
        metaphors=metaphors,
        models=args.models,
        k_values=args.k_values,
        epsilon_values=args.epsilon_values,
        patience=args.patience,
        ckpt=ckpt,
    )

    full_output = {
        "config": {
            "models":         args.models,
            "k_values":       args.k_values,
            "epsilon_values": args.epsilon_values,
            "patience":       args.patience,
            "metaphors":      [m["id"] for m in metaphors],
        },
        "results": results,
    }
    results_path = CALIBRATION_DIR / "results.json"
    results_path.write_text(
        json.dumps(full_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = CALIBRATION_DIR / "summary.csv"
    write_summary_csv(summary_path, results)

    per_met_path = CALIBRATION_DIR / "per_metaphor.csv"
    write_per_metaphor_csv(per_met_path, results, metaphors)

    print(f"\n{'='*65}")
    print("Done.")
    print(f"  Full results:     {results_path}")
    print(f"  Summary CSV:      {summary_path}")
    print(f"  Per-metaphor CSV: {per_met_path}")
    print(f"  Rounds data:      {ckpt._rounds_path}")
    print(f"  Judge cache:      {ckpt._judge_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()