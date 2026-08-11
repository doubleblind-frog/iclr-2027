"""
refinement_loop.py

Iterative in-context refinement loop for visual metaphor generation.
Run:
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py --k 10 --epsilon 0.05

ABLATION MODES (combinable via --extractor / --feedback):

One-shot extraction:
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py \\
        --extractor oneshot \\
        --subset-csv multimodal_analogy_generation/ablations/ablation_subset_20.csv \\
        --seed-images-from multimodal_analogy_generation/iterative_refinement/images/k1
No domain guidance:
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py --feedback no_domain_guidance --subset-csv multimodal_analogy_generation/ablations/ablation_subset_20.csv --seed-images-from multimodal_analogy_generation/iterative_refinement/images/k1
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from multimodal_analogy_generation.scripts.extract import extract_metaphor_graphs, _configure_lm, DEFAULT_MODEL
from multimodal_analogy_generation.ablations.scripts.extract_one_shot import extract_metaphor_graphs_oneshot  # ABLATION
from multimodal_analogy_generation.ablations.scripts.refine_no_domain_guidance import build_no_domain_guidance_diagnostic # ABLATION
from multimodal_analogy_generation.scripts.generate_images import gen_gpt_image
from multimodal_analogy_generation.iterative_refinement.scripts.refine import (
    build_diagnostic,
    refine_elaboration,
    make_refiner_client,
    DOMAIN_MATCH_THRESHOLD,
)
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR      = Path("multimodal_analogy_generation/iterative_refinement")
ABLATIONS_DIR   = Path("multimodal_analogy_generation/ablations/results")
ZERO_SHOT_INPUT = Path("multimodal_analogy_generation/zero_shot_exploration/zero_shot_outputs.jsonl")
HOLDOUT_PATH    = Path("multimodal_analogy_generation/holdout-metaphors.jsonl")
GOLD_CSV        = Path("multimodal_analogy_generation/ai_metaphor_domains.csv")

PULLBACK_THRESHOLD = 0.40

DEFAULT_K        = 10
DEFAULT_EPSILON  = 0.05
DEFAULT_PATIENCE = 1

EMBEDDER_MODEL = "all-mpnet-base-v2"

# ABLATION: registry of available extractors. "multistep" is the production
# extractor and is the default everywhere below -- unchanged behaviour if
# --extractor is never passed.
EXTRACTORS: dict[str, Callable] = {
    "multistep": extract_metaphor_graphs,
    "oneshot": extract_metaphor_graphs_oneshot,
}

# Registry of available round-feedback builders. "full" is production
# (CLIP-grounded structural + alignment feedback; see refine.py) and is the
# default everywhere below. Each builder must accept the same (metaphor,
# extraction_result, gold_domains, embed) call signature so run_one_metaphor
# can call any of them identically -- build_diagnostic follows this by
# reading image_path out of extraction_result rather than requiring it as a
# separate argument.
DIAGNOSTIC_BUILDERS: dict[str, Callable] = {
    "full": build_diagnostic,
    "no_domain_guidance": build_no_domain_guidance_diagnostic,  # ABLATION
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

class Checkpointer:
    """
    Persists one JSON record per completed round to a .jsonl file,
    appending immediately after each round finishes.

    On restart, call load() to recover all previously saved records.
    The returned dict maps metaphor_id -> list[round_record], sorted by round.
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, list[dict]]:
        """Return {metaphor_id: [round_records...]} from the checkpoint file."""
        by_id: dict[str, list[dict]] = {}
        if not self.path.exists():
            return by_id
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_id.setdefault(rec["id"], []).append(rec)
        for recs in by_id.values():
            recs.sort(key=lambda r: r["round"])
        return by_id

    def save_round(self, record: dict) -> None:
        """Append one completed round record to the checkpoint file."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_holdout_ids(path: Path) -> set[str]:
    """Return the set of metaphor IDs reserved for calibration (holdout set).
    These are excluded from the main experiment run to prevent data leakage.
    """
    if not path.exists():
        print(f"Warning: holdout file not found at {path}. "
              "No metaphors will be excluded.")
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(str(json.loads(line)["id"]))
    return ids


def load_zero_shot(path: Path) -> list[dict]:
    """Load round-0 elaborations: [{id, metaphor, visual_elaboration}, ...]."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> dict[str, list[str]]:
    """Map id -> [domain_a, domain_b] from the annotation CSV."""
    gold: dict[str, list[str]] = {}
    if not path.exists():
        print(f"Warning: gold CSV not found at {path}; domain-match logging disabled.")
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
            print(f"Warning: could not resolve id + two domain columns in "
                  f"{path.name} (saw {reader.fieldnames}); domain-match logging disabled.")
            return gold
        for row in reader:
            rid = str(row[id_col]).strip()
            gold[rid] = [
                str(row[a_col]).strip().lower(),
                str(row[b_col]).strip().lower(),
            ]
    return gold


# ABLATION: id normalisation + subset loading, mirroring the same
# zero-pad-safe normalisation used elsewhere in this project (see
# build_ablation_subset.py) to avoid re-introducing the zero-padded
# filename vs. unpadded id mismatch from the human-eval pipeline.
def _normalize_id(raw_id) -> str:
    s = Path(str(raw_id).strip()).stem
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        raise ValueError(f"Could not extract a numeric id from: {raw_id!r}")
    return digits.zfill(3)


def load_subset_ids(path: Path) -> set[str]:
    """Load the ablation subset id column (as written by
    build_ablation_subset.py) and return normalised ids."""
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_col = "id" if "id" in (reader.fieldnames or []) else reader.fieldnames[0]
        for row in reader:
            ids.add(_normalize_id(row[id_col]))
    return ids


def seed_round1_images(entries: list[dict], seed_dir: Path, dest_dir: Path) -> None:
    """ABLATION: copy each metaphor's existing round-1 (zero-shot) image from
    the production run into the ablation's own images/k1/ directory, so the
    loop scores round 1 with the ablated extractor WITHOUT regenerating the
    image. Metaphors missing a source image are left alone -- the loop will
    fall back to generating a fresh round-1 image for them, which breaks the
    "same x0" guarantee for that item, so this prints a loud warning."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for e in entries:
        mid = _normalize_id(e["id"])
        src = seed_dir / f"{mid}.png"
        dst = dest_dir / f"{mid}.png"
        if dst.exists():
            continue
        if src.exists():
            shutil.copy2(src, dst)
        else:
            missing.append(mid)
    if missing:
        print(
            f"[ABLATION WARNING] {len(missing)} metaphor(s) have no round-1 "
            f"image at {seed_dir} to seed from and will get a FRESH round-1 "
            f"generation instead, breaking the shared-x0 guarantee for "
            f"those items: {sorted(missing)}"
        )


def make_embedder(model_name: str):
    """Return a callable list[str] -> normalised np.ndarray, or None on failure."""
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
# Per-metaphor loop
# ---------------------------------------------------------------------------

def run_one_metaphor(
    *,
    entry:            dict,
    gold:             dict[str, list[str]],
    embed,
    refiner_client,
    k:                int,
    epsilon:          float,
    patience:         int,
    images_root:      Path,
    elaborations_dir: Path,
    ckpt:             Checkpointer,
    saved_rounds:     list[dict],
    extractor_fn:     Callable = extract_metaphor_graphs,  # ABLATION: injectable
    diagnostic_builder_fn: Callable = build_diagnostic,    # ABLATION: injectable
) -> list[dict]:
    """
    Run up to k rounds for a single metaphor. Returns one record per round."""
    mid         = str(entry["id"])
    metaphor    = entry["metaphor"]

    gold_domains = gold.get(mid, [])

    elab_log_path = elaborations_dir / f"{mid}.jsonl"
    elab_log      = elab_log_path.open("a", encoding="utf-8")

    # Restore state from checkpoint
    records:          list[dict]    = list(saved_rounds)
    n_done:           int           = len(records)
    prev_score:       Optional[float] = records[-1]["pullback_score"] if records else None
    below_eps_streak: int           = 0
    stop_reason:      str           = "max_rounds"

    if records:
        elaboration = records[-1].get("elaboration_next",
                                      records[-1]["elaboration"])
    else:
        elaboration = (entry.get("visual_elaboration") or "").strip()

    if len(records) >= 2:
        for i in range(1, len(records)):
            delta = abs(records[i]["pullback_score"] - records[i-1]["pullback_score"])
            if delta < epsilon:
                below_eps_streak += 1
            else:
                below_eps_streak = 0

        if below_eps_streak >= patience:
            stop_reason = "flattened"
            for rec in records:
                rec["stop_reason"] = stop_reason
            elab_log.close()
            return records

    for r in range(n_done + 1, k + 1):
        round_dir = images_root / f"k{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        img_path = round_dir / f"{mid}.png"

        # 1) Generate image
        if not img_path.exists():
            try:
                img_bytes = gen_gpt_image(elaboration)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"    [{mid}] round {r} generation failed, stopping: {msg}")
                stop_reason = (
                    "generation_failed_round1" if not records
                    else "generation_failed"
                )
                break
            img_path.write_bytes(img_bytes)

        # 2) Extract domains + pullback.
        try:
            result = extractor_fn(img_path, threshold=PULLBACK_THRESHOLD)  # ABLATION: was extract_metaphor_graphs(...)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"    [{mid}] round {r} extraction failed, stopping: {msg}")
            stop_reason = (
                "extraction_failed_round1" if not records
                else "extraction_failed"
            )
            break
        score = float(result.get("pullback_score", 0.0))

        # 3) Build feedback diagnostic.
        diag = diagnostic_builder_fn(  # ABLATION: was build_diagnostic(...)
            metaphor=metaphor,
            extraction_result=result,
            gold_domains=gold_domains,
            embed=embed,
        )

        n_concept_matches = sum(
            1 for m in diag.concept_matches if m.get("match")
        )

        # 4) Refine elaboration for the next round.
        next_elaboration = elaboration
        if r < k:
            try:
                next_elaboration = refine_elaboration(
                    refiner_client, elaboration, diag
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"    [{mid}] round {r} refinement failed: {msg}. "
                      "Keeping previous elaboration for next round.")

        record = {
            "id":               mid,
            "round":            r,
            "pullback_score":   score,
            "metaphor":         metaphor,
            "elaboration":      elaboration,
            "elaboration_next": next_elaboration,
            "gold_domains":     " | ".join(gold_domains),
            "extracted_target": result.get("target_domain", "none"),
            "extracted_source": result.get("source_domain", "none"),
            "concepts_matched": n_concept_matches,
            "n_gold_concepts":  len(diag.concept_matches),
            "coverage":         diag.coverage,
            "image_path":       str(img_path),
            "stop_reason":      "in_progress",
        }

        ckpt.save_round(record)
        records.append(record)

        elab_log.write(json.dumps({
            "round":          r,
            "elaboration":    elaboration,
            "pullback_score": score,
            "feedback":       diag.to_feedback_text(),
        }, ensure_ascii=False) + "\n")
        elab_log.flush()

        traj = " -> ".join(f"{r['pullback_score']:.3f}" for r in records)
        print(f"    round {r}: pullback={score:.3f}  [{traj}]")

        # 5) Stopping check.
        if prev_score is not None:
            if abs(score - prev_score) < epsilon:
                below_eps_streak += 1
            else:
                below_eps_streak = 0
            if below_eps_streak >= patience:
                stop_reason = "flattened"
                break
        prev_score = score

        elaboration = next_elaboration

    elab_log.close()

    for rec in records:
        rec["stop_reason"] = stop_reason
    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_scores_csv(path: Path, all_records: list[dict]) -> None:
    fields = [
        "id", "round", "pullback_score", "extracted_target", "extracted_source",
        "gold_domains", "concepts_matched", "n_gold_concepts",
        "coverage", "stop_reason", "image_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(all_records, key=lambda r: (r["id"], r["round"])):
            w.writerow(rec)


def write_best(best_dir: Path, all_records: list[dict]) -> None:
    """Select argmax-over-rounds pullback per id; copy image + write summary."""
    best_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict[str, list[dict]] = {}
    for rec in all_records:
        by_id.setdefault(rec["id"], []).append(rec)

    summary_path = best_dir / "best_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "best_round", "best_pullback", "n_rounds", "stop_reason"])
        for mid, recs in sorted(by_id.items()):
            best = max(recs, key=lambda r: r["pullback_score"])
            src  = Path(best["image_path"])
            if src.exists():
                shutil.copy2(src, best_dir / f"{mid}.png")
            w.writerow([
                mid, best["round"], f'{best["pullback_score"]:.4f}',
                len(recs), best["stop_reason"],
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Max refinement rounds (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help=(
            f"Single-step pullback change below which a round counts as flat "
            f"(default: {DEFAULT_EPSILON})."
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=(
            f"Consecutive flat rounds required to stop "
            f"(default: {DEFAULT_PATIENCE})."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N metaphors (useful for smoke-testing).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Extractor VLM (default: {DEFAULT_MODEL})",
    )
    # ABLATION: new flags. All default to production behaviour.
    parser.add_argument(
        "--extractor", choices=list(EXTRACTORS.keys()), default="multistep",
        help="Which domain-graph extractor to use (default: multistep, "
             "i.e. unmodified production behaviour).",
    )
    parser.add_argument(
        "--feedback", choices=list(DIAGNOSTIC_BUILDERS.keys()), default="full",
        help="Which round-feedback content to show the refiner "
             "(default: full, i.e. unmodified production behaviour).",
    )
    parser.add_argument(
        "--subset-csv", type=Path, default=None,
        help="If given, restrict the run to metaphor ids listed in this CSV's "
             "'id' column (e.g. an ablation subset file).",
    )
    parser.add_argument(
        "--seed-images-from", type=Path, default=None,
        help="If given, copy each metaphor's round-1 image from this directory "
             "into the run's own images/k1/ before starting, so round 1 is "
             "scored without being regenerated.",
    )
    parser.add_argument(
        "--output-tag", default=None,
        help="If given, write outputs to multimodal_analogy_generation/ablations/results/"
             "<tag>/ instead of the production output directory. Defaults to "
             "--extractor's name when --extractor != 'multistep', otherwise "
             "unset (production directory).",
    )
    args = parser.parse_args()

    # ABLATION: route to a separate output directory whenever this is not a
    # plain production run, so checkpoints/scores/images never collide with
    # the real experiment. Combines every non-default ablation flag into the
    # tag, so e.g. --extractor oneshot --feedback scalar_only together would
    # produce "oneshot_scalar_only" -- this generalises to future combined
    # ablations without further changes here.
    output_tag = args.output_tag
    if output_tag is None:
        tags = []
        if args.extractor != "multistep":
            tags.append(args.extractor)
        if args.feedback != "full":
            tags.append(args.feedback)
        output_tag = "_".join(tags) if tags else None
    experiment_dir = (
        ABLATIONS_DIR / output_tag if output_tag else OUTPUT_DIR
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    images_root      = experiment_dir / "images"
    elaborations_dir = experiment_dir / "elaborations"
    elaborations_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = experiment_dir / "rounds_checkpoint.jsonl"
    ckpt      = Checkpointer(ckpt_path)
    saved     = ckpt.load()
    if saved:
        print(f"Checkpoint found: restored {sum(len(v) for v in saved.values())} "
              f"rounds across {len(saved)} metaphors.")

    all_entries = load_zero_shot(ZERO_SHOT_INPUT)
    holdout_ids = load_holdout_ids(HOLDOUT_PATH)
    entries     = [e for e in all_entries if str(e["id"]) not in holdout_ids]

    print(
        f"Loaded {len(all_entries)} metaphors. "
        f"Excluded {len(holdout_ids)} holdout metaphors. "
        f"Running on {len(entries)} metaphors."
    )

    # ABLATION: restrict to the requested subset, if given.
    if args.subset_csv:
        subset_ids = load_subset_ids(args.subset_csv)
        before = len(entries)
        entries = [e for e in entries if _normalize_id(e["id"]) in subset_ids]
        print(f"  --subset-csv {args.subset_csv}: {before} -> {len(entries)} metaphors "
              f"({len(subset_ids)} ids requested).")
        found_ids = {_normalize_id(e["id"]) for e in entries}
        not_found = subset_ids - found_ids
        if not_found:
            print(f"  [ABLATION WARNING] {len(not_found)} subset id(s) not found "
                  f"in zero-shot input and will be skipped: {sorted(not_found)}")

    if args.limit:
        entries = entries[: args.limit]
        print(f"  (limited to {args.limit} for this run)")

    # ABLATION: seed round-1 images from production before anything else, so
    # the skip-if-exists check in run_one_metaphor takes effect on round 1.
    if args.seed_images_from:
        seed_round1_images(entries, args.seed_images_from, images_root / "k1")

    entries_to_run = []
    skipped        = 0
    for e in entries:
        mid         = str(e["id"])
        mid_saved   = saved.get(mid, [])
        n_done      = len(mid_saved)
        last_reason = mid_saved[-1].get("stop_reason", "in_progress") if mid_saved else "in_progress"
        if n_done >= args.k or last_reason in ("flattened", "generation_failed",
                                                "extraction_failed", "max_rounds"):
            skipped += 1
        else:
            entries_to_run.append(e)

    if skipped:
        print(f"  Skipping {skipped} already-completed metaphors.")
    print(f"  Remaining: {len(entries_to_run)} metaphors to process.")

    gold = load_gold(GOLD_CSV)

    print(
        f"\nk={args.k}  epsilon={args.epsilon}  patience={args.patience}  "
        f"pullback_threshold={PULLBACK_THRESHOLD}  extractor={args.extractor}  "
        f"feedback={args.feedback}  output_dir={experiment_dir}"
    )

    _configure_lm(args.model)
    embed          = make_embedder(EMBEDDER_MODEL)
    refiner_client = make_refiner_client()
    extractor_fn   = EXTRACTORS[args.extractor]  # ABLATION
    diagnostic_builder_fn = DIAGNOSTIC_BUILDERS[args.feedback]  # ABLATION

    all_records: list[dict]            = [r for recs in saved.values() for r in recs]
    failed_ids:  list[tuple[str, str]] = []

    for entry in entries_to_run:
        mid = entry["id"]
        print(f"\n=== {mid}: {entry['metaphor']} ===")

        mid_saved = saved.get(mid, [])
        if mid_saved:
            print(f"    Resuming from round {len(mid_saved) + 1} "
                  f"(rounds 1–{len(mid_saved)} restored from checkpoint).")

        try:
            recs = run_one_metaphor(
                entry=entry,
                gold=gold,
                embed=embed,
                refiner_client=refiner_client,
                k=args.k,
                epsilon=args.epsilon,
                patience=args.patience,
                images_root=images_root,
                elaborations_dir=elaborations_dir,
                ckpt=ckpt,
                saved_rounds=mid_saved,
                extractor_fn=extractor_fn,  # ABLATION
                diagnostic_builder_fn=diagnostic_builder_fn,  # ABLATION
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"    [{mid}] unexpected failure, skipping: {msg}")
            failed_ids.append((mid, msg))
            continue

        if not recs:
            print(f"    [{mid}] no rounds completed, skipping.")
            failed_ids.append((mid, "no_rounds_completed"))
            continue

        all_records = [r for r in all_records if r["id"] != mid]
        all_records.extend(recs)

        traj = " -> ".join(f"{r['pullback_score']:.3f}" for r in recs)
        print(
            f"    rounds: {len(recs)}  pullback: {traj}  "
            f"({recs[-1]['stop_reason']})"
        )

    write_scores_csv(experiment_dir / "scores.csv", all_records)
    write_best(experiment_dir / "best", all_records)

    n_success = len({r["id"] for r in all_records})
    print(
        f"\nDone. {n_success} metaphors with results, "
        f"{len(failed_ids)} skipped."
    )
    print(f"  Per-round scores : {experiment_dir / 'scores.csv'}")
    print(f"  Best per metaphor: {experiment_dir / 'best'}")
    print(f"  Checkpoint       : {ckpt_path}")
    if failed_ids:
        errors_path = experiment_dir / "failed.json"
        errors_path.write_text(json.dumps(failed_ids, indent=2))
        print(f"  Skipped metaphors: {errors_path}")


if __name__ == "__main__":
    main()