"""
compare_conditions.py

Compares zero-shot vs. refined visual metaphor scores using
paired Wilcoxon signed-rank tests.

Usage:
    python multimodal_analogy_generation/evaluation/scripts/compare_conditions.py --zero multimodal_analogy_generation/evaluation/results/zero_judge_scores.csv --best multimodal_analogy_generation/evaluation/results/last_judge_scores.csv --filter-dir multimodal_analogy_generation/evaluation/results/last_filtered
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ZERO_CSV   = Path("multimodal_analogy_generation/evaluation/results/zero_judge_scores.csv")
BEST_CSV   = Path("multimodal_analogy_generation/evaluation/results/best_judge_scores.csv")
SCORES_CSV = Path("multimodal_analogy_generation/iterative_refinement/scores.csv")

ALPHA = 0.05

AXES   = ["metaphor_consistency", "analogy_appropriateness", "conceptual_integration"]
JUDGES = ["claude-opus-4-8", "gpt-5.2"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    """Load a judge scores CSV, coercing N/A to NaN and deduplicating by ID."""
    df = pd.read_csv(path, na_values=["N/A", "NA", "n/a", ""])
    df["id"] = (
        pd.to_numeric(df["id"], errors="raise")
        .astype("Int64")      # nullable integer dtype
        .astype(str)
    )
    for col in df.columns:
        if col not in ("id", "metaphor"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["id"].duplicated().any():
        n_dup = df["id"].duplicated().sum()
        print(f"  Warning: {n_dup} duplicate ID(s) in {path.name}. "
              "Keeping row with highest mean_judge_score per ID.")
        df = (
            df.sort_values("mean_judge_score", ascending=False)
              .drop_duplicates(subset="id", keep="first")
              .sort_values("id")
        )
    return df


def load_human_eval_ids(filter_dir: Path) -> set[str]:
    """
    Read the IDs of metaphors included in the human evaluation from the
    filenames in filter_dir (e.g. '001.jpeg' → '001').
    """
    if not filter_dir.exists():
        raise FileNotFoundError(
            f"Filter directory not found: {filter_dir}. "
            "Check the path passed to --filter-dir."
        )
    ids = {
        str(int(p.stem))
        for p in filter_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not ids:
        raise ValueError(
            f"No image files found in {filter_dir}. "
            f"Expected files with extensions {IMAGE_EXTENSIONS}."
        )
    print(f"Human-evaluation filter: {len(ids)} IDs loaded from {filter_dir}")
    return ids


def load_best_rounds(scores_path: Path) -> pd.DataFrame:
    """
    From scores.csv (one row per metaphor × round), return a DataFrame with
    one row per metaphor ID showing the round that achieved the highest
    pullback score.
    """
    df = pd.read_csv(scores_path)
    df["id"] = df["id"].astype(str).str.strip()
    df["pullback_score"] = pd.to_numeric(df["pullback_score"], errors="coerce")
    best = (
        df.loc[df.groupby("id")["pullback_score"].idxmax()]
          .rename(columns={"round": "best_round", "pullback_score": "best_pullback"})
        [["id", "best_round", "best_pullback"]]
        .reset_index(drop=True)
    )
    return best


def summarise_improvement(best_rounds: pd.DataFrame) -> None:
    total    = len(best_rounds)
    round1   = (best_rounds["best_round"] == 1).sum()
    improved = (best_rounds["best_round"] > 1).sum()
    print(f"\nRefinement improvement summary (from scores.csv):")
    print(f"  Total metaphors:                 {total}")
    print(f"  Best score in round 1 (no gain): {round1}  ({100*round1/total:.1f}%)")
    print(f"  Best score after round 1:        {improved}  ({100*improved/total:.1f}%)")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def axis_col(axis: str, judge: str) -> str:
    return f"{axis}__{judge}"


def row_mean(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].mean(axis=1, skipna=True).values


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test; drops NaN pairs before testing."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, np.nan
    if np.all(a - b == 0):
        return np.nan, 1.0
    stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return float(stat), float(p)


def stars(p: float, alpha_corrected: float) -> str:
    if np.isnan(p):
        return "—"
    if p < alpha_corrected / 10:
        return "***"
    if p < alpha_corrected / 2:
        return "**"
    if p < alpha_corrected:
        return "*"
    return "ns"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "    —    "
    if p < 0.001:
        return " < 0.001 "
    return f"  {p:.4f} "


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(
    title: str,
    results: list[dict],
    alpha: float,
    one_tailed: bool,
) -> None:
    tail_note = "(one-tailed, H₁: best > zero)" if one_tailed else "(two-tailed)"
    print(f"\n{'='*80}")
    print(f"  {title}  {tail_note}")
    print(f"  Bonferroni-corrected α = {alpha:.4f}  (family size = {len(results)})")
    print(f"{'='*80}")
    print(f"{'Comparison':<34} {'Med(z)':>7} {'Med(b)':>7} "
          f"{'Mn(z)':>7} {'Mn(b)':>7} {'Δ mean':>8} {'p-value':>10} {'sig':>4}")
    print(f"{'-'*80}")
    for r in results:
        sig = stars(r["p"], alpha)
        delta = r["mean_best"] - r["mean_zero"]
        direction = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(
            f"{r['label']:<34}"
            f" {r['med_zero']:>7.3f} {r['med_best']:>7.3f}"
            f" {r['mean_zero']:>7.3f} {r['mean_best']:>7.3f}"
            f" {direction}{abs(delta):>6.3f}"
            f" {fmt_p(r['p']):>10} {sig:>4}"
        )
    print(f"{'='*80}")
    print("  * p < α   ** p < α/2   *** p < α/10   ns = not significant")
    print("  Direction arrow and Δ mean based on mean (not median) to resolve ties.")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    zero:           pd.DataFrame,
    best:           pd.DataFrame,
    one_tailed:     bool,
    improved_only:  bool,
    best_rounds:    pd.DataFrame | None,
    human_eval_ids: set[str] | None,
) -> None:
    common = sorted(set(zero["id"]) & set(best["id"]))

    if human_eval_ids is not None:
        before = len(common)
        common = [mid for mid in common if mid in human_eval_ids]
        excluded = before - len(common)
        print(f"\nHuman-eval filter applied: {before} → {len(common)} metaphors "
              f"({excluded} excluded as not in human evaluation).")
        missing = human_eval_ids - set(common)
        if missing:
            print(f"  Warning: {len(missing)} human-eval ID(s) not found in "
                  f"both judge score files: {sorted(missing)[:10]}"
                  + (" ..." if len(missing) > 10 else ""))

    if improved_only:
        if best_rounds is None:
            raise ValueError("--improved-only requires --scores.")
        improved_ids = set(
            best_rounds.loc[best_rounds["best_round"] > 1, "id"]
        )
        before = len(common)
        common = [mid for mid in common if mid in improved_ids]
        print(f"\nImproved-only filter: {before} → {len(common)} metaphors "
              f"(removed {before - len(common)} whose best score was in round 1).")

    n = len(common)
    print(f"Metaphors in analysis: {n}")
    if n < 10:
        print("Warning: fewer than 10 matched pairs — results unreliable.")

    z = zero.set_index("id").loc[common]
    b = best.set_index("id").loc[common]

    cell_cols = [axis_col(ax, jm) for ax in AXES for jm in JUDGES]
    non_null_z = z[cell_cols].notna().sum().sum()
    non_null_b = b[cell_cols].notna().sum().sum()
    print(f"Non-null cell values — zero: {non_null_z}/{n * len(cell_cols)}, "
          f"best: {non_null_b}/{n * len(cell_cols)}")

    alternative = "less" if one_tailed else "two-sided"

    # 1. Overall
    zv = row_mean(z, cell_cols)
    bv = row_mean(b, cell_cols)
    _, p = wilcoxon_test(zv, bv, alternative)
    print_table("1. OVERALL (all axes × judges)", [{
        "label":     "Refined vs. zero-shot",
        "med_zero":  float(np.nanmedian(zv)),
        "med_best":  float(np.nanmedian(bv)),
        "mean_zero": float(np.nanmean(zv)),
        "mean_best": float(np.nanmean(bv)),
        "p":         p,
    }], ALPHA, one_tailed)

    # 2. Per judge
    per_judge = []
    for jm in JUDGES:
        cols = [axis_col(ax, jm) for ax in AXES]
        zv = row_mean(z, cols)
        bv = row_mean(b, cols)
        _, p = wilcoxon_test(zv, bv, alternative)
        per_judge.append({
            "label":     jm,
            "med_zero":  float(np.nanmedian(zv)),
            "med_best":  float(np.nanmedian(bv)),
            "mean_zero": float(np.nanmean(zv)),
            "mean_best": float(np.nanmean(bv)),
            "p":         p,
        })
    print_table("2. PER JUDGE (mean across axes)",
                per_judge, ALPHA / len(per_judge), one_tailed)

    # 3. Per axis
    per_axis = []
    for ax in AXES:
        cols = [axis_col(ax, jm) for jm in JUDGES]
        zv = row_mean(z, cols)
        bv = row_mean(b, cols)
        _, p = wilcoxon_test(zv, bv, alternative)
        per_axis.append({
            "label":     ax.replace("_", " "),
            "med_zero":  float(np.nanmedian(zv)),
            "med_best":  float(np.nanmedian(bv)),
            "mean_zero": float(np.nanmean(zv)),
            "mean_best": float(np.nanmean(bv)),
            "p":         p,
        })
    print_table("3. PER AXIS (mean across judges)",
                per_axis, ALPHA / len(per_axis), one_tailed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero", type=Path, default=ZERO_CSV,
                        help=f"Zero-shot judge scores CSV (default: {ZERO_CSV})")
    parser.add_argument("--best", type=Path, default=BEST_CSV,
                        help=f"Refined judge scores CSV (default: {BEST_CSV})")
    parser.add_argument(
        "--scores", type=Path, default=None,
        help="Per-round scores CSV (scores.csv) — required for --improved-only.",
    )
    parser.add_argument(
        "--filter-dir", type=Path, default=None,
        help=(
            "Directory of images whose filenames define the human-evaluation "
            "subset (e.g. multimodal_analogy_generation/evaluation/results/last_filtered). "
            "Only metaphors whose ID matches an image stem (e.g. '001' from "
            "'001.jpeg') are included in the analysis. Applies before all "
            "other filters."
        ),
    )
    parser.add_argument(
        "--one-tailed", action="store_true",
        help="Test H₁: refined > zero-shot (alternative='less' since we pass "
             "zero as a, best as b to wilcoxon). Report two-tailed p in the "
             "thesis unless direction was pre-registered.",
    )
    parser.add_argument(
        "--improved-only", action="store_true",
        help="Restrict to metaphors whose best pullback score was achieved "
             "after round 1 (i.e. refinement actually helped). Requires --scores.",
    )
    args = parser.parse_args()

    print(f"Zero-shot : {args.zero}")
    print(f"Refined   : {args.best}")

    zero = load(args.zero)
    best = load(args.best)
    print(f"Zero-shot rows: {len(zero)}  |  Refined rows: {len(best)}")

    best_rounds = None
    if args.scores:
        best_rounds = load_best_rounds(args.scores)
        summarise_improvement(best_rounds)
    elif args.improved_only:
        parser.error("--improved-only requires --scores.")

    human_eval_ids = None
    if args.filter_dir:
        human_eval_ids = load_human_eval_ids(args.filter_dir)

    analyse(
        zero=zero,
        best=best,
        one_tailed=args.one_tailed,
        improved_only=args.improved_only,
        best_rounds=best_rounds,
        human_eval_ids=human_eval_ids,
    )


if __name__ == "__main__":
    main()