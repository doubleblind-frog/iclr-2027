"""
judge_batch.py

Run judge.py's evaluate() over a folder of images and save all scores
to a single CSV.

Usage:
    python multimodal_analogy_generation/evaluation/scripts/judge_batch.py --scores multimodal_analogy_generation/iterative_refinement/scores.csv --select last --lookup multimodal_analogy_generation/zero_shot_exploration/zero_shot_outputs.jsonl --out    multimodal_analogy_generation/evaluation/results/last_judge_scores.csv

Restrict to a fixed set of ids (e.g. the human-eval 200) via --filter-dir,
which keeps only ids that have an {id}.png in the given directory:
    python multimodal_analogy_generation/evaluation/scripts/judge_batch.py \\
        --scores multimodal_analogy_generation/iterative_refinement/scores.csv --select last \\
        --filter-dir multimodal_analogy_generation/evaluation/results/last_filtered \\
        --out multimodal_analogy_generation/evaluation/results/last_judge_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import evaluate, AXES, JUDGE_MODELS, PROMPTS_DIR

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IMAGE_DIR   = Path("best")
LOOKUP_PATH = Path("multimodal_analogy_generation/zero_shot_exploration/zero_shot_outputs.jsonl")
OUT_CSV     = Path("judge_scores.csv")


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def images_from_dir(directory: Path) -> dict[str, Path]:
    """Scan a directory for *.png files; return {id: path}."""
    return {
        p.stem: p
        for p in sorted(directory.glob("*.png"),
                        key=lambda p: (len(p.stem), p.stem))
    }


def images_from_scores(scores_path: Path, select: str) -> dict[str, Path]:
    """
    Derive one image path per metaphor from scores.csv.

      select="best" — round with the highest pullback_score
      select="last" — round with the highest round number (final round run)

    Returns {id: Path(image_path)}.
    """
    if select not in ("best", "last"):
        raise ValueError(f"--select must be 'best' or 'last', got {select!r}")

    rows_by_id: dict[str, list[dict]] = {}
    with scores_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = str(row["id"]).strip()
            rows_by_id.setdefault(mid, []).append(row)

    paths: dict[str, Path] = {}
    for mid, rows in rows_by_id.items():
        if select == "best":
            chosen = max(rows, key=lambda r: float(r.get("pullback_score") or 0))
        else:  # last
            chosen = max(rows, key=lambda r: int(r.get("round") or 0))
        img = Path(chosen["image_path"])
        if img.exists():
            paths[mid] = img
        else:
            print(f"  Warning: image not found for {mid} (round {chosen['round']}): {img}")

    return paths


# ---------------------------------------------------------------------------
# Id filtering (e.g. restrict to the human-eval 200)
# ---------------------------------------------------------------------------

def _normalize_id(raw_id) -> str:
    """Zero-pad-safe id normalisation (3 -> 003), matching the convention used
    across the rest of the ablation tooling."""
    s = Path(str(raw_id).strip()).stem
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        raise ValueError(f"Could not extract a numeric id from: {raw_id!r}")
    return digits.zfill(3)


def load_filter_ids(filter_dir: Path) -> set[str]:
    """Return the normalised ids of all {id}.png images in filter_dir."""
    if not filter_dir.exists():
        raise SystemExit(f"[error] --filter-dir not found: {filter_dir}")
    ids = {_normalize_id(p.stem) for p in filter_dir.glob("*.png")}
    if not ids:
        raise SystemExit(f"[error] no .png files found in --filter-dir: {filter_dir}")
    return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metaphor_lookup(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not path.exists():
        print(f"Warning: lookup file not found at {path}. "
              "Metaphor text will be blank for all images.")
        return lookup
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mid  = str(row.get("id", "")).strip()
            text = (row.get("metaphor") or "").strip()
            if mid:
                lookup[mid] = text
    return lookup


def _avg(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    axes        = list(AXES.keys())
    judge_names = list(JUDGE_MODELS.keys())

    fields = (
        ["id", "metaphor", "mean_judge_score"]
        + [f"judge_{ax}" for ax in axes]
        + [f"judge_model_{m}" for m in judge_names]
        + [f"{ax}__{jm}" for ax in axes for jm in judge_names]
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_row(mid: str, metaphor: str, results: dict) -> dict:
    axes        = list(AXES.keys())
    judge_names = list(JUDGE_MODELS.keys())

    all_scores = []
    ax_means   = {}
    mdl_scores = {m: [] for m in judge_names}
    cells      = {}

    for axis in axes:
        ax_vals = []
        for jm in judge_names:
            s = results.get(axis, {}).get(jm, {}).get("score")
            cells[f"{axis}__{jm}"] = _fmt(s)
            if s is not None:
                ax_vals.append(s)
                mdl_scores[jm].append(s)
                all_scores.append(s)
        ax_means[f"judge_{axis}"] = _fmt(_avg(ax_vals))

    return {
        "id":               mid,
        "metaphor":         metaphor,
        "mean_judge_score": _fmt(_avg(all_scores)),
        **ax_means,
        **{f"judge_model_{m}": _fmt(_avg(mdl_scores[m])) for m in judge_names},
        **cells,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--images", type=Path, default=None,
                     help="Directory of images named {id}.png.")
    src.add_argument("--scores", type=Path, default=None,
                     help="scores.csv from the refinement loop; use with --select.")

    parser.add_argument(
        "--select", choices=["best", "last"], default="best",
        help="Which round to judge when --scores is used. "
             "'best' = highest pullback score (default), "
             "'last' = final round completed.",
    )
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH,
                        help=f"JSONL mapping id -> metaphor text (default: {LOOKUP_PATH})")
    parser.add_argument("--out", type=Path, default=OUT_CSV,
                        help=f"Output CSV path (default: {OUT_CSV})")
    parser.add_argument("--filter-dir", type=Path, default=None,
                        help="Restrict the run to ids that have an image in this "
                             "directory (e.g. last_filtered/ = the human-eval 200).")
    parser.add_argument("--only-id", default=None,
                        help="Restrict the run to a single metaphor id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (smoke-testing).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-image judge output.")
    args = parser.parse_args()

    # Resolve image paths
    if args.scores is not None:
        print(f"Deriving image paths from {args.scores} (select={args.select})")
        image_map = images_from_scores(args.scores, args.select)
    elif args.images is not None:
        print(f"Scanning {args.images} for images")
        image_map = images_from_dir(args.images)
    else:
        print(f"No source specified; scanning default directory {IMAGE_DIR}")
        image_map = images_from_dir(IMAGE_DIR)

    if args.filter_dir is not None:
        keep_ids = load_filter_ids(args.filter_dir)
        before = len(image_map)
        image_map = {mid: p for mid, p in image_map.items() if _normalize_id(mid) in keep_ids}
        print(f"--filter-dir {args.filter_dir}: {before} -> {len(image_map)} images "
              f"({len(keep_ids)} ids requested).")
        if not image_map:
            print("No images left after filtering. Check --filter-dir.")
            return

    if args.only_id is not None:
        target = _normalize_id(args.only_id)
        image_map = {mid: p for mid, p in image_map.items() if _normalize_id(mid) == target}
        if not image_map:
            print(f"No image found for --only-id {args.only_id!r}.")
            return

    cache_path = args.out.with_name(args.out.stem + "_cache.json")

    ids = sorted(image_map, key=lambda s: (len(s), s))
    if args.limit:
        ids = ids[: args.limit]

    if not ids:
        print("No images found. Check --images / --scores arguments.")
        return

    lookup = load_metaphor_lookup(args.lookup)
    cache  = load_cache(cache_path)

    print(f"Images to judge:  {len(ids)}")
    print(f"Metaphor lookup:  {args.lookup} ({len(lookup)} entries)")
    print(f"Output CSV:       {args.out}")
    print(f"Cache:            {cache_path}")
    if cache:
        print(f"Resuming: {len(cache)} images already judged.")

    rows:   list[dict]  = []
    failed: list[str]   = []

    for i, mid in enumerate(ids, 1):
        img_path = image_map[mid]
        metaphor = lookup.get(mid, "")

        if not metaphor:
            print(f"  [{i}/{len(ids)}] {mid}: WARNING — no metaphor text found.")

        if mid in cache:
            print(f"  [{i}/{len(ids)}] {mid}: cached, skipping.")
            rows.append(build_row(mid, metaphor, cache[mid]))
            continue

        print(f"  [{i}/{len(ids)}] {mid}: {metaphor[:60]}")
        try:
            results = evaluate(
                image_path=img_path,
                metaphor_text=metaphor,
                prompts_dir=PROMPTS_DIR,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append(mid)
            continue

        cache[mid] = results
        save_cache(cache_path, cache)

        rows.append(build_row(mid, metaphor, results))

    write_csv(args.out, rows)

    print(f"\nDone. {len(rows)} images judged, {len(failed)} failed.")
    print(f"  Scores CSV : {args.out}")
    print(f"  Cache      : {cache_path}")
    if failed:
        print(f"  Failed IDs : {failed}")


if __name__ == "__main__":
    main()