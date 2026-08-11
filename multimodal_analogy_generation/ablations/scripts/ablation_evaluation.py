"""
ablation_evaluation.py

Fills in one block of the ablation results tracker table for a single
ablation condition. Computes, per judge axis (metaphor_consistency,
analogy_appropriateness, conceptual_integration).

Run:
python multimodal_analogy_generation/ablations/scripts/ablation_evaluation.py --zero-csv multimodal_analogy_generation/evaluation/results/zero_judge_scores.csv --ablated-csv multimodal_analogy_generation/ablations/results/oneshot/oneshot_last_judge_scores.csv --multistep-csv multimodal_analogy_generation/evaluation/results/last_judge_scores.csv --subset-csv multimodal_analogy_generation/ablations/ablation_subset_20.csv --ablation-name "One-shot extraction"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

AXES = ["metaphor_consistency", "analogy_appropriateness", "conceptual_integration"]
AXIS_LABELS = {
    "metaphor_consistency": "Metaphor consistency",
    "analogy_appropriateness": "Analogy appropriateness",
    "conceptual_integration": "Conceptual integration",
}

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42


def normalize_id(raw_id) -> str:
    s = Path(str(raw_id).strip()).stem
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        raise ValueError(f"Could not extract a numeric id from: {raw_id!r}")
    return digits.zfill(3)


def load_judge_csv(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    df["id"] = df["id"].apply(normalize_id)
    df = df.set_index("id")
    rename_map = {
        "judge_metaphor_consistency": f"{prefix}_metaphor_consistency",
        "judge_analogy_appropriateness": f"{prefix}_analogy_appropriateness",
        "judge_conceptual_integration": f"{prefix}_conceptual_integration",
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {missing}")
    df = df[list(rename_map.keys())].rename(columns=rename_map)
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].unique().tolist()
        print(f"[warn] {path.name}: dropping duplicate id rows for {dupes}")
        df = df[~df.index.duplicated(keep="first")]
    return df


def load_subset_ids(path: Path) -> set:
    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else df.columns[0]
    return {normalize_id(x) for x in df[id_col]}


def bootstrap_ci(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple:
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_wilcoxon(diffs: np.ndarray, alternative: str) -> float:
    if np.all(diffs == 0):
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative=alternative)
        return float(p)
    except ValueError as e:
        print(f"[warn] Wilcoxon failed ({e}); reporting p=nan.")
        return float("nan")


def fmt_p(p: float) -> str:
    if p != p:  # nan
        return "n/a"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-csv", type=Path, required=True)
    parser.add_argument("--ablated-csv", type=Path, required=True)
    parser.add_argument("--multistep-csv", type=Path, required=True,
                         help="Production (multistep) judge scores CSV, e.g. "
                              "multimodal_analogy_generation/evaluation/results/last_judge_scores.csv "
                              "-- used for the direct ablated-vs-multistep test.")
    parser.add_argument("--subset-csv", type=Path, required=True,
                         help="CSV with an 'id' column listing the ablation subset ids.")
    parser.add_argument("--ablation-name", required=True,
                         help="Label for the LaTeX \\multirow cell, e.g. 'One-shot extraction'.")
    parser.add_argument("--multistep-alternative", choices=["two-sided", "greater", "less"],
                         default="two-sided",
                         help="Alternative hypothesis for the ablated-vs-multistep "
                              "Wilcoxon test (default: two-sided, i.e. no directional "
                              "prior). Use 'greater' if you specifically expect the "
                              "ablation to beat multistep, 'less' for the opposite.")
    args = parser.parse_args()

    zero_df = load_judge_csv(args.zero_csv, "zero")
    abl_df = load_judge_csv(args.ablated_csv, "abl")
    multi_df = load_judge_csv(args.multistep_csv, "multi")
    subset_ids = load_subset_ids(args.subset_csv)

    merged = zero_df.join(abl_df, how="inner").join(multi_df, how="inner")
    before = len(merged)
    merged = merged[merged.index.isin(subset_ids)]
    print(f"[info] {before} ids present in all three score files; "
          f"{len(merged)} of those are in the requested subset "
          f"({len(subset_ids)} ids requested).")

    missing = subset_ids - set(merged.index)
    if missing:
        print(f"[warn] {len(missing)} subset id(s) missing from one or more "
              f"judge score files and excluded from the analysis: {sorted(missing)}")

    score_cols = (
        [f"zero_{a}" for a in AXES] + [f"abl_{a}" for a in AXES] + [f"multi_{a}" for a in AXES]
    )
    before = len(merged)
    merged = merged.dropna(subset=score_cols)
    if len(merged) < before:
        print(f"[warn] Dropped {before - len(merged)} metaphor(s) with missing axis scores.")

    n = len(merged)
    print(f"[info] Computing stats on n={n} paired metaphors.\n")

    rng = np.random.default_rng(RANDOM_SEED)

    results = {}
    for axis in AXES:
        ablated_mean = merged[f"abl_{axis}"].mean()

        # vs. zero-shot (same as before)
        diffs_zero = (merged[f"abl_{axis}"] - merged[f"zero_{axis}"]).to_numpy()
        delta_zero = diffs_zero.mean()
        ci_zero = bootstrap_ci(diffs_zero, N_BOOTSTRAP, rng)
        p_zero = paired_wilcoxon(diffs_zero, alternative="greater")

        # vs. multistep (direct head-to-head, replaces % Retained)
        diffs_multi = (merged[f"abl_{axis}"] - merged[f"multi_{axis}"]).to_numpy()
        delta_multi = diffs_multi.mean()
        ci_multi = bootstrap_ci(diffs_multi, N_BOOTSTRAP, rng)
        p_multi = paired_wilcoxon(diffs_multi, alternative=args.multistep_alternative)

        results[axis] = dict(
            ablated=ablated_mean,
            delta_zero=delta_zero, ci_zero=ci_zero, p_zero=p_zero,
            delta_multi=delta_multi, ci_multi=ci_multi, p_multi=p_multi,
        )

        print(
            f"{AXIS_LABELS[axis]:28s}  ablated={ablated_mean:.3f}\n"
            f"    vs zero-shot : delta={delta_zero:+.3f} "
            f"[{ci_zero[0]:+.3f}, {ci_zero[1]:+.3f}]  p={p_zero:.4f}\n"
            f"    vs multistep : delta={delta_multi:+.3f} "
            f"[{ci_multi[0]:+.3f}, {ci_multi[1]:+.3f}]  p={p_multi:.4f} "
            f"({args.multistep_alternative})"
        )

    print("\n--- LaTeX table rows (paste into the tracker) ---")
    print(
        "NOTE: this replaces the single '\\%~Ret.' column with two columns "
        "(delta vs multistep + its p). Update your tabularx column spec and "
        "header accordingly, e.g.:\n"
        "  \\begin{tabularx}{\\textwidth}{@{}p{3.8cm} X c c c c c@{}}\n"
        "  Ablation & Metric & Ablated & $\\Delta$ vs zero (95\\% CI) & $p$ "
        "& $\\Delta$ vs multistep (95\\% CI) & $p$ \\\\\n"
    )
    lines = []
    lines.append(f"\\multirow{{{len(AXES)}}}{{3.8cm}}{{\\textbf{{{args.ablation_name}}}}}")
    for i, axis in enumerate(AXES):
        r = results[axis]
        prefix = " " if i > 0 else ""
        row = (
            f"{prefix} & {AXIS_LABELS[axis]} & {r['ablated']:.3f} & "
            f"{r['delta_zero']:+.3f} [{r['ci_zero'][0]:+.3f}, {r['ci_zero'][1]:+.3f}] & "
            f"{fmt_p(r['p_zero'])} & "
            f"{r['delta_multi']:+.3f} [{r['ci_multi'][0]:+.3f}, {r['ci_multi'][1]:+.3f}] & "
            f"{fmt_p(r['p_multi'])} \\\\"
        )
        if i == 0:
            lines[0] = lines[0] + "\n" + row
        else:
            lines.append(row)
    print("\n".join(lines))


if __name__ == "__main__":
    main()