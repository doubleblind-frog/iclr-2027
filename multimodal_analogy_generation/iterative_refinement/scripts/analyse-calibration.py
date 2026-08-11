"""
analyse-calibration.py

Two analyses over calibration/per_metaphor.csv:

  1. Calibration  — find the (model, k, epsilon) that maximises image quality
                    across all 25 metaphors. Use this to set hyperparameters
                    for the main experiment.

  2. Best-image   — for each metaphor, pick the single best (model, k, epsilon)
                    condition, ranked by a composite score.

Usage:
    python multimodal_analogy_generation/iterative_refinement/scripts/analyse-calibration.py --csv multimodal_analogy_generation/iterative_refinement/calibration/per_metaphor.csv --out multimodal_analogy_generation/iterative_refinement/calibration/analysis/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

PULLBACK_WEIGHT = 0.4
JUDGE_WEIGHT    = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "pullback_score", "mean_judge_score",
        "judge_metaphor_consistency", "judge_analogy_appropriateness",
        "judge_conceptual_integration",
        "judge_model_claude-opus-4-8",
        "judge_model_gpt-5.2",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["k"]       = df["k"].astype(int)
    df["epsilon"]  = df["epsilon"].astype(float)
    return df


def composite(df: pd.DataFrame) -> pd.Series:
    """Normalised composite score: weighted pullback + judge."""
    pb  = df["pullback_score"]
    jdg = df["mean_judge_score"]
    pb_norm  = (pb  - pb.min())  / (pb.max()  - pb.min()  + 1e-9)
    jdg_norm = (jdg - jdg.min()) / (jdg.max() - jdg.min() + 1e-9)
    return PULLBACK_WEIGHT * pb_norm + JUDGE_WEIGHT * jdg_norm


# ---------------------------------------------------------------------------
# Analysis 1 — calibration: best (model, k, epsilon)
# ---------------------------------------------------------------------------

def calibration_analysis(df: pd.DataFrame, out: Path) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 1 — CALIBRATION")
    print("=" * 60)

    summary = (
        df.groupby(["model_slug", "k", "epsilon"])
        .agg(
            n                   = ("metaphor_id", "count"),
            mean_pullback       = ("pullback_score",              "mean"),
            std_pullback        = ("pullback_score",              "std"),
            mean_judge          = ("mean_judge_score",            "mean"),
            std_judge           = ("mean_judge_score",            "std"),
            mean_consistency    = ("judge_metaphor_consistency",  "mean"),
            mean_appropriateness= ("judge_analogy_appropriateness","mean"),
            mean_integration    = ("judge_conceptual_integration","mean"),
        )
        .reset_index()
    )
    summary["composite"] = (
        PULLBACK_WEIGHT * (summary["mean_pullback"] / summary["mean_pullback"].max()) +
        JUDGE_WEIGHT    * (summary["mean_judge"]    / summary["mean_judge"].max())
    )
    summary = summary.sort_values("composite", ascending=False)

    print("\nTop 10 (model, k, ε) conditions by composite score:\n")
    print(
        summary.head(10)[[
            "model_slug", "k", "epsilon", "n",
            "mean_pullback", "mean_judge", "composite"
        ]].to_string(index=False, float_format="{:.3f}".format)
    )

    print("\n--- Best condition per model ---")
    for model, grp in summary.groupby("model_slug"):
        best = grp.iloc[0]
        print(
            f"  {model:50s}  k={int(best.k)}  ε={best.epsilon:.2f}"
            f"  pullback={best.mean_pullback:.3f}  judge={best.mean_judge:.3f}"
        )

    print("\n--- Effect of k (mean across all models & epsilons) ---")
    k_effect = (
        df.groupby("k")[["pullback_score", "mean_judge_score"]]
        .mean()
        .rename(columns={"pullback_score": "mean_pullback",
                         "mean_judge_score": "mean_judge"})
    )
    print(k_effect.to_string(float_format="{:.3f}".format))

    print("\n--- Effect of epsilon (mean across all models & k) ---")
    eps_effect = (
        df.groupby("epsilon")[["pullback_score", "mean_judge_score"]]
        .mean()
        .rename(columns={"pullback_score": "mean_pullback",
                         "mean_judge_score": "mean_judge"})
    )
    print(eps_effect.to_string(float_format="{:.3f}".format))

    print("\n--- Model comparison (mean across all k & ε) ---")
    model_effect = (
        df.groupby("model_slug")[["pullback_score", "mean_judge_score"]]
        .mean()
        .rename(columns={"pullback_score": "mean_pullback",
                         "mean_judge_score": "mean_judge"})
        .sort_values("mean_judge", ascending=False)
    )
    print(model_effect.to_string(float_format="{:.3f}".format))

    out_path = out / "calibration_summary.csv"
    summary.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  → Full calibration summary written to {out_path}")

    winner = summary.iloc[0]
    print(
        f"\n★  Recommended config: model={winner.model_slug}"
        f"  k={int(winner.k)}  ε={winner.epsilon:.2f}"
        f"  (composite={winner.composite:.4f})"
    )


# ---------------------------------------------------------------------------
# Analysis 2 — best image per metaphor
# ---------------------------------------------------------------------------

def best_image_analysis(df: pd.DataFrame, out: Path) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 2 — BEST IMAGE PER METAPHOR")
    print("=" * 60)

    df = df.copy()
    df["composite"] = composite(df)

    best_per_metaphor = (
        df.sort_values("composite", ascending=False)
        .groupby("metaphor_id")
        .first()
        .reset_index()
    )[[
        "metaphor_id", "metaphor_type", "model_slug", "k", "epsilon",
        "selected_round", "pullback_score", "mean_judge_score", "composite",
        "judge_metaphor_consistency", "judge_analogy_appropriateness",
        "judge_conceptual_integration", "extracted_target", "extracted_source",
    ]]
    best_per_metaphor = best_per_metaphor.sort_values(
        "mean_judge_score", ascending=False
    )

    print(f"\nBest condition for each of the {len(best_per_metaphor)} metaphors:\n")
    print(
        best_per_metaphor[[
            "metaphor_id", "model_slug", "k", "epsilon",
            "pullback_score", "mean_judge_score"
        ]].to_string(index=False, float_format="{:.3f}".format)
    )

    print("\n--- Win count by model ---")
    print(best_per_metaphor["model_slug"].value_counts().to_string())

    print("\n--- Mean judge score by metaphor type ---")
    print(
        best_per_metaphor.groupby("metaphor_type")["mean_judge_score"]
        .agg(["mean", "std", "count"])
        .to_string(float_format="{:.3f}".format)
    )

    corr_round_judge = df[["selected_round", "mean_judge_score"]].corr().iloc[0, 1]
    corr_round_pull  = df[["selected_round", "pullback_score"]].corr().iloc[0, 1]
    print(
        f"\n--- Correlation: selected_round vs scores ---"
        f"\n  vs mean_judge_score : r = {corr_round_judge:.3f}"
        f"\n  vs pullback_score   : r = {corr_round_pull:.3f}"
    )

    best_path = out / "best_per_metaphor.csv"
    best_per_metaphor.to_csv(best_path, index=False, float_format="%.4f")
    img_lookup_cols = [
        "metaphor_id", "model_slug", "k", "epsilon", "selected_round"
    ]
    merged = best_per_metaphor.merge(
        df[img_lookup_cols + ["image_path"]] if "image_path" in df.columns
        else best_per_metaphor[img_lookup_cols],
        on=img_lookup_cols, how="left"
    )
    if "image_path" in merged.columns:
        img_map = (
            merged.set_index("metaphor_id")["image_path"]
            .dropna()
            .to_dict()
        )
        json_path = out / "best_images.json"
        json_path.write_text(
            json.dumps(img_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  → Best image paths written to {json_path}")

    print(f"  → Best-per-metaphor table written to {best_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv", required=True,
        help="Path to per_metaphor.csv from calibrate.py"
    )
    parser.add_argument(
        "--out", default="multimodal_analogy_generation/iterative_refinement/calibration/analysis",
        help="Output directory for analysis files"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.csv))
    print(f"Loaded {len(df)} rows, {df['metaphor_id'].nunique()} metaphors, "
          f"{df['model_slug'].nunique()} models.")

    calibration_analysis(df, out)
    best_image_analysis(df, out)

    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    main()