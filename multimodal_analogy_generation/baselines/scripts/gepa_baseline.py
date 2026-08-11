"""
gepa_baseline.py

GEPA baseline for visual metaphor generation, competing against the iterative refinement loop (refinement_loop.py / refine.py).

Two variants, selected via --metric:
  pullback  GEPA's reward is the pullback score (scripts/ct_pullback.py),
            computed by generating an image from the candidate elaboration
            and running it through extract_metaphor_graphs(). Textual
            feedback for GEPA's reflection step reuses build_diagnostic()
            from refine.py (same CLIP-grounded diagnostic the project's own
            refiner sees).
  clip      GEPA's reward is a single CLIP image-text cosine
            (refine.py's compute_clip_score) used directly as GEPA's
            optimisation metric rather than as per-round textual feedback.


Run:
    python multimodal_analogy_generation/baselines/scripts/gepa_baseline.py --metric clip --budget light
    python multimodal_analogy_generation/baselines/scripts/gepa_baseline.py --metric pullback --budget light
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import dspy
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from multimodal_analogy_generation.scripts.extract import extract_metaphor_graphs, _configure_lm, DEFAULT_MODEL
from multimodal_analogy_generation.scripts.generate_images import gen_gpt_image
from multimodal_analogy_generation.iterative_refinement.scripts.refine import (
    build_diagnostic,
    DOMAIN_MATCH_THRESHOLD,
    get_clip_scorer,
    compute_clip_score,
)
from multimodal_analogy_generation.iterative_refinement.scripts.refinement_loop import (
    ZERO_SHOT_INPUT,
    HOLDOUT_PATH,
    GOLD_CSV,
    EMBEDDER_MODEL,
    PULLBACK_THRESHOLD,
    load_holdout_ids,
    load_zero_shot,
    load_gold,
    load_subset_ids,
    make_embedder,
    _normalize_id,
)

load_dotenv()

ABLATIONS_DIR = Path("multimodal_analogy_generation/ablations")
BASELINES_DIR = Path("multimodal_analogy_generation/baselines/results")
ABLATION_SUBSET_CSV = ABLATIONS_DIR / "ablation_subset_20.csv"

REFLECTION_MODEL = "anthropic/claude-sonnet-4-6"  # matches refine.py's REFINER_MODEL_ID
REFLECTION_MAX_TOKENS = 1024

DEFAULT_TRAIN_VAL_SPLIT = 0.8
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# DSPy program: the thing GEPA actually optimizes
# ---------------------------------------------------------------------------

class GenerateElaboration(dspy.Signature):
    """Write a single-paragraph, ~55-word prompt for a text-to-image model
    that depicts the given verbal metaphor as a visual metaphor: an
    abstract TARGET concept portrayed through a concrete SOURCE concept,
    such that the relationships among the source elements mirror the
    relationships in the target concept. Both the source and target
    concepts must be unmistakably present and recognisable in a single
    scene, and their structural relationships must be visually depicted,
    not just their surface objects."""

    metaphor: str = dspy.InputField(
        desc="A verbal metaphor, e.g. 'Cars are drivable pesticide spray cans.'"
    )
    elaboration: str = dspy.OutputField(
        desc="~55-word single-paragraph visual description for the image generator."
    )


class ElaborationProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GenerateElaboration)

    def forward(self, metaphor: str):
        return self.predict(metaphor=metaphor)


# ---------------------------------------------------------------------------
# Data loading: train/val pool (excludes holdout + the 20 eval metaphors)
# ---------------------------------------------------------------------------

def build_examples(entries: list[dict], gold: dict[str, list[str]]) -> list[dspy.Example]:
    examples = []
    for e in entries:
        mid = str(e["id"])
        examples.append(
            dspy.Example(
                id=mid,
                metaphor=e["metaphor"],
                gold_domains=gold.get(mid, []),
            ).with_inputs("metaphor")
        )
    return examples


def load_splits(
    train_val_split: float,
    seed: int,
    pool_limit: Optional[int],
    eval_limit: Optional[int],
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    all_entries = load_zero_shot(ZERO_SHOT_INPUT)
    holdout_ids = load_holdout_ids(HOLDOUT_PATH)
    eval_ids = load_subset_ids(ABLATION_SUBSET_CSV)
    gold = load_gold(GOLD_CSV)

    pool = [
        e for e in all_entries
        if str(e["id"]) not in holdout_ids and _normalize_id(e["id"]) not in eval_ids
    ]
    eval_entries = [e for e in all_entries if _normalize_id(e["id"]) in eval_ids]

    rng = random.Random(seed)
    rng.shuffle(pool)
    if pool_limit:
        pool = pool[:pool_limit]
    if eval_limit:
        eval_entries = eval_entries[:eval_limit]

    split_idx = max(1, int(len(pool) * train_val_split))
    train_entries, val_entries = pool[:split_idx], pool[split_idx:] or pool[:1]

    print(
        f"Pool: {len(pool)} metaphors (excluding {len(holdout_ids)} holdout, "
        f"{len(eval_ids)} eval-subset). Train: {len(train_entries)}  "
        f"Val: {len(val_entries)}  Eval (ablation subset): {len(eval_entries)}"
    )

    return (
        build_examples(train_entries, gold),
        build_examples(val_entries, gold),
        build_examples(eval_entries, gold),
    )


# ---------------------------------------------------------------------------
# Rollout cache: image gen + graph extraction are the expensive steps, and
# GEPA repeatedly re-evaluates candidates on overlapping examples, so cache
# every (metaphor, elaboration) pair on disk and reuse across the run
# (and across a crash/resume).
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _rollout_key(metaphor: str, elaboration: str) -> str:
    return hashlib.sha256(f"{metaphor}||{elaboration}".encode("utf-8")).hexdigest()[:24]


def make_generate_and_extract(
    images_dir: Path, cache_path: Path, clip_scorer, compute_pullback: bool
):
    """Return a function (metaphor, elaboration) -> record, backed by an
    on-disk cache keyed on (metaphor, elaboration).

    compute_pullback controls whether the expensive 3-stage VLM extraction
    runs on every rollout. It should only be True for the pullback metric --
    for the clip metric, extract_metaphor_graphs() is never consulted by
    clip_metric(), so running it during optimisation would just add the
    ~3 extra Claude Sonnet vision calls per rollout without affecting the
    score GEPA optimizes against. The cheap CLIP score is always computed
    either way. The final eval pass (run_final_eval) always computes both,
    independently of this flag, since that one only runs 20 times.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_json(cache_path)

    def generate_and_extract(metaphor: str, elaboration: str) -> dict:
        key = _rollout_key(metaphor, elaboration)
        if key in cache:
            return cache[key]

        img_path = images_dir / f"{key}.png"
        if not img_path.exists():
            img_bytes = gen_gpt_image(elaboration)
            img_path.write_bytes(img_bytes)

        extraction_result = (
            extract_metaphor_graphs(img_path, threshold=PULLBACK_THRESHOLD)
            if compute_pullback else None
        )
        clip_score = compute_clip_score(img_path, metaphor, scorer=clip_scorer)

        record = {
            "image_path": str(img_path),
            "pullback_score": (
                float(extraction_result.get("pullback_score", 0.0))
                if extraction_result is not None else None
            ),
            "clip_score": clip_score,
            "extraction_result": extraction_result,
        }
        cache[key] = record
        _save_json(cache_path, cache)
        return record

    return generate_and_extract


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_metrics(generate_and_extract, embed):
    def pullback_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        metaphor = example.metaphor
        elaboration = (pred.elaboration or "").strip()
        if not elaboration:
            return dspy.Prediction(
                score=0.0, feedback="No elaboration text was produced; cannot generate an image."
            )
        try:
            record = generate_and_extract(metaphor, elaboration)
        except Exception as e:
            return dspy.Prediction(
                score=0.0, feedback=f"Image generation/extraction failed: {type(e).__name__}: {e}"
            )
        gold_domains = getattr(example, "gold_domains", []) or []
        diag = build_diagnostic(
            metaphor=metaphor,
            extraction_result=record["extraction_result"],
            gold_domains=gold_domains,
            embed=embed,
            domain_threshold=DOMAIN_MATCH_THRESHOLD,
        )
        return dspy.Prediction(score=diag.pullback_score, feedback=diag.to_feedback_text())

    def clip_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        metaphor = example.metaphor
        elaboration = (pred.elaboration or "").strip()
        if not elaboration:
            return dspy.Prediction(
                score=0.0, feedback="No elaboration text was produced; cannot generate an image."
            )
        try:
            record = generate_and_extract(metaphor, elaboration)
        except Exception as e:
            return dspy.Prediction(
                score=0.0, feedback=f"Image generation failed: {type(e).__name__}: {e}"
            )
        score = record["clip_score"]
        feedback = (
            f'The intended metaphor is: "{metaphor}"\n'
            f"CLIP image-text alignment: {score:.3f} (cosine similarity between "
            "the generated image and the intended metaphor description; higher "
            "means the image more strongly evokes the intended metaphor overall)."
        )
        return dspy.Prediction(score=score, feedback=feedback)

    return {"pullback": pullback_metric, "clip": clip_metric}


# ---------------------------------------------------------------------------
# Final evaluation pass: run the optimized (fixed) program once per
# ablation-subset metaphor, exactly like the other ablations' "best"/"last"
# image, and write a scores.csv judge_stricter.py can consume directly.
# ---------------------------------------------------------------------------

def run_final_eval(
    optimized: dspy.Module,
    eval_examples: list[dspy.Example],
    images_dir: Path,
) -> list[dict]:
    images_dir.mkdir(parents=True, exist_ok=True)
    clip_scorer = get_clip_scorer()
    rows = []
    for ex in eval_examples:
        pred = optimized(metaphor=ex.metaphor)
        elaboration = (pred.elaboration or "").strip()

        img_path = images_dir / f"{ex.id}.png"
        if not img_path.exists():
            try:
                img_bytes = gen_gpt_image(elaboration)
            except Exception as e:
                print(f"  [{ex.id}] generation failed: {type(e).__name__}: {e}")
                continue
            img_path.write_bytes(img_bytes)
        try:
            extraction_result = extract_metaphor_graphs(img_path, threshold=PULLBACK_THRESHOLD)
            clip_score = compute_clip_score(img_path, ex.metaphor, scorer=clip_scorer)
        except Exception as e:
            print(f"  [{ex.id}] scoring failed: {type(e).__name__}: {e}")
            continue
        pullback_score = float(extraction_result.get("pullback_score", 0.0))

        print(f"  [{ex.id}] pullback={pullback_score:.3f}  clip={clip_score:.3f}")
        rows.append({
            "id": ex.id,
            "round": 1,
            "metaphor": ex.metaphor,
            "elaboration": elaboration,
            "pullback_score": pullback_score,
            "clip_score": clip_score,
            "image_path": str(img_path),
        })
    return rows


def write_scores_csv(path: Path, rows: list[dict]) -> None:
    fields = ["id", "round", "metaphor", "elaboration", "pullback_score", "clip_score", "image_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda r: r["id"]):
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", choices=["pullback", "clip"], required=True,
                        help="Which score GEPA optimizes against.")
    parser.add_argument("--budget", choices=["light", "medium", "heavy"], default="light",
                        help="GEPA's auto budget preset (default: light -- start small given "
                             "the cost of image generation + VLM extraction per metric call).")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                        help="Override --budget with an exact metric-call budget "
                             "(useful for dry runs).")
    parser.add_argument("--train-val-split", type=float, default=DEFAULT_TRAIN_VAL_SPLIT,
                        help=f"Fraction of the non-holdout, non-eval pool used for GEPA's "
                             f"train split (default: {DEFAULT_TRAIN_VAL_SPLIT}).")
    parser.add_argument("--pool-limit", type=int, default=None,
                        help="Cap the train/val pool size before splitting (dry runs).")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Cap the number of ablation-subset metaphors in the final "
                             "eval pass (dry runs).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"VLM used by ElaborationProgram and by the graph extractor "
                             f"inside the metric (default: {DEFAULT_MODEL}).")
    parser.add_argument("--output-tag", default=None,
                        help="Output directory name under multimodal_analogy_generation/baselines/results/ "
                             "(default: gepa_<metric>).")
    args = parser.parse_args()

    output_tag = args.output_tag or f"gepa_{args.metric}"
    run_dir = BASELINES_DIR / output_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Metric: {args.metric}  Budget: {args.max_metric_calls or args.budget}  "
          f"Output: {run_dir}")

    trainset, valset, eval_examples = load_splits(
        args.train_val_split, args.seed, args.pool_limit, args.eval_limit
    )

    _configure_lm(args.model)
    embed = make_embedder(EMBEDDER_MODEL)
    clip_scorer = get_clip_scorer()

    generate_and_extract = make_generate_and_extract(
        images_dir=run_dir / "rollout_images",
        cache_path=run_dir / "rollout_cache.json",
        clip_scorer=clip_scorer,
        compute_pullback=(args.metric == "pullback"),
    )
    metrics = make_metrics(generate_and_extract, embed)
    metric_fn = metrics[args.metric]

    reflection_lm = dspy.LM(
        model=REFLECTION_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=REFLECTION_MAX_TOKENS,
    )

    gepa_kwargs = dict(
        metric=metric_fn,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        skip_perfect_score=False,  # pullback score is unbounded above 1.0; never treat any
                                    # minibatch as "already perfect" and stop reflecting on it
        track_stats=True,
        log_dir=str(run_dir / "gepa_run"),
        seed=args.seed,
    )
    if args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.budget

    gepa = dspy.teleprompt.GEPA(**gepa_kwargs)

    print("\nRunning GEPA optimization...")
    optimized = gepa.compile(student=ElaborationProgram(), trainset=trainset, valset=valset)

    optimized.save(str(run_dir / "optimized_program.json"), save_program=False)
    final_instructions = optimized.predict.signature.instructions
    (run_dir / "optimized_instructions.txt").write_text(final_instructions, encoding="utf-8")
    print(f"\nOptimized instructions:\n{final_instructions}\n")

    print(f"Running final eval pass on {len(eval_examples)} ablation-subset metaphors...")
    rows = run_final_eval(optimized, eval_examples, images_dir=run_dir / "images")
    write_scores_csv(run_dir / "scores.csv", rows)

    print(f"\nDone. {len(rows)}/{len(eval_examples)} eval metaphors scored.")
    print(f"  Optimized program : {run_dir / 'optimized_program.json'}")
    print(f"  Instructions      : {run_dir / 'optimized_instructions.txt'}")
    print(f"  Scores            : {run_dir / 'scores.csv'}")
    print(f"  GEPA logs         : {run_dir / 'gepa_run'}")
    print(
        "\nNext: judge_batch.py --scores "
        f"{run_dir / 'scores.csv'} --select last "
        f"--out {run_dir / 'judge_scores.csv'}"
    )


if __name__ == "__main__":
    main()
