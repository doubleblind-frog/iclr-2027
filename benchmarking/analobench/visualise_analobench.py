"""
visualise_analobench.py — figures for the AnaloBench evaluation.

Reads the metrics.json and score_distributions.csv produced by
analyse_analobench.py and writes 4 figures plus a combined dashboard.

Usage (from repo root):
    python benchmarking/analobench/visualise_analobench.py --analysis_dir benchmarking/analobench/results/analysis/ --out benchmarking/analobench/results/analysis/figures/

Figures:
    fig1_signal_quality.png         — accuracy vs MRR vs top-2, with rank distribution
    fig2_disagreement_2x2.png       — McNemar 2x2 contingency
    fig3_score_margins.png          — margin when correct vs deficit when wrong
    fig4_confidence_calibration.png — accuracy per confidence bucket
    dashboard.png                   — all figures on one canvas
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_CT       = "#3B82F6"   # blue
COLOR_ZS       = "#F59E0B"   # amber
COLOR_BOTH     = "#10B981"   # green
COLOR_NEITHER  = "#9CA3AF"   # grey
COLOR_GOLD     = "#22C55E"   # green
COLOR_WRONG    = "#EF4444"   # red
COLOR_NEUTRAL  = "#6366F1"   # indigo

plt.rcParams.update({
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.titlesize":   12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

def load_data(analysis_dir: str) -> tuple[dict, pd.DataFrame]:
    """Load metrics.json and score_distributions.csv from the analysis dir."""
    with open(os.path.join(analysis_dir, "metrics.json")) as f:
        metrics = json.load(f)
    df = pd.read_csv(os.path.join(analysis_dir, "score_distributions.csv"))
    return metrics, df


# ---------------------------------------------------------------------------
# Figure 1 — score-signal quality
# ---------------------------------------------------------------------------

def fig_signal_quality(metrics: dict, df: pd.DataFrame, ax_left=None, ax_right=None):
    """Left: accuracy vs MRR vs top-2 bars. Right: rank distribution."""
    standalone = ax_left is None
    if standalone:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    ct = metrics["ct_metrics"]
    zs = metrics["zs_metrics"]

    # Left: comparison bars
    labels  = ["Accuracy\n(top-1)", "Top-2\naccuracy", "MRR"]
    ct_vals = [ct["accuracy"], ct["top2_accuracy"], ct["mean_reciprocal_rank"]]
    zs_vals = [zs["accuracy"], None, None]   # ZS has no per-option scores
    x = np.arange(len(labels))
    w = 0.38

    ax_left.bar(x - w/2, ct_vals, w, label="CT", color=COLOR_CT)
    # Only draw ZS bar where defined
    ax_left.bar(x[0] + w/2, zs_vals[0], w, label="Zero-shot", color=COLOR_ZS)

    for i, v in enumerate(ct_vals):
        label_text = f"{v:.1%}" if i < 2 else f"{v:.3f}"
        ax_left.text(i - w/2, v + 0.02, label_text,
                     ha="center", fontsize=9, fontweight="bold")
    if zs_vals[0] is not None:
        ax_left.text(0 + w/2, zs_vals[0] + 0.02, f"{zs_vals[0]:.1%}",
                     ha="center", fontsize=9, fontweight="bold")

    ax_left.axhline(0.25, ls=":", c=COLOR_NEITHER, lw=1)
    ax_left.text(2.4, 0.27, "random\nbaseline", fontsize=8, color=COLOR_NEITHER)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylim(0, 1.1)
    ax_left.set_ylabel("Score")
    ax_left.set_title("Top-1 accuracy, top-2 accuracy, and MRR")
    ax_left.legend(loc="lower left", framealpha=0.9)

    # Right: rank distribution of correct answer
    rank_dist = ct["rank_of_correct_distribution"]
    ranks = ["rank_1", "rank_2", "rank_3", "rank_4"]
    counts = [rank_dist[r] for r in ranks]
    n = sum(counts)
    cum = np.cumsum(counts) / n

    bars = ax_right.bar(
        ["1st", "2nd", "3rd", "4th"], counts,
        color=[COLOR_GOLD, COLOR_NEUTRAL, COLOR_WRONG, "#7F1D1D"]
    )
    for b, c, cu in zip(bars, counts, cum):
        ax_right.text(
            b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{c}\n({c/n:.0%})",
            ha="center", fontsize=8, fontweight="bold",
        )

    ax_right.set_xlabel("Rank of the gold answer in CT's score list")
    ax_right.set_ylabel("# of tasks")
    ax_right.set_title(f"Rank distribution of the gold answer  (n={n})")
    ax_right.text(
        0.97, 0.95,
        f"Top-2: {(counts[0]+counts[1])/n:.0%}\nMRR: {ct['mean_reciprocal_rank']:.3f}",
        transform=ax_right.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLOR_NEITHER),
        fontsize=9,
    )

    if standalone:
        fig.suptitle("Score-signal quality on AnaloBench (n=200)",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Figure 2 — McNemar
# ---------------------------------------------------------------------------

def fig_disagreement(metrics: dict, ax=None):
    """2x2 contingency of CT correct vs ZS correct."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    cmp = metrics["comparison"]
    matrix = np.array([
        [cmp["both_right"],     cmp["ct_only_right"]],
        [cmp["zs_only_right"],  cmp["both_wrong"]],
    ])

    cell_colors = np.array([
        [COLOR_BOTH, COLOR_CT],
        [COLOR_ZS,   COLOR_NEITHER],
    ])

    for (i, j), v in np.ndenumerate(matrix):
        rect = plt.Rectangle(
            (j, 1 - i), 1, 1,
            facecolor=cell_colors[i, j], edgecolor="white", lw=2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(j + 0.5, 1.7 - i, str(v),
                ha="center", va="center", fontsize=24, fontweight="bold", color="white")
        ax.text(j + 0.5, 1.3 - i, f"({v / cmp['n_compared']:.1%})",
                ha="center", va="center", fontsize=10, color="white")

    # Axis labels
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["ZS correct", "ZS wrong"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["CT wrong", "CT correct"])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    pval = cmp.get("mcnemar_p_value")
    sig_marker = "***" if (pval is not None and pval < 0.001) \
        else "**" if (pval is not None and pval < 0.01) \
        else "*"  if (pval is not None and pval < 0.05) \
        else "n.s."

    ax.set_title(
        f"Per-task agreement between CT and zero-shot  (n={cmp['n_compared']})\n"
        f"McNemar exact two-sided p = {pval}  ({sig_marker})",
        fontweight="bold",
    )

    if standalone:
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Figure 3 — score-margin distributions
# ---------------------------------------------------------------------------

def fig_score_margins(df: pd.DataFrame, ax=None):
    """
    Two boxes: margin when correct, deficit when wrong.

    Margin when correct = top1 - top2, computed only on tasks CT got right.
        Measures how decisively CT picked the right answer.

    Deficit when wrong = top1 - score(gold), computed only on tasks CT got wrong.
        Measures how far below the top the correct answer sat.

    These two quantities are unambiguous regardless of where the gold answer
    ranks.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    correct  = df.loc[df["correct"] == 1, "margin_top1"]
    deficit  = df.loc[df["correct"] == 0, "deficit_when_wrong"]

    positions = [1, 2]
    data = [correct, deficit]
    labels = [
        f"Margin when pullback score\nis correct\n(top1 − top2)",
        f"Deficit of gold\nbehind top-1 when\npullback score is wrong\n(top1 − gold)",
    ]
    colors = [COLOR_GOLD, "#991B1B"]

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Overlay individual points
    for pos, vals, c in zip(positions, data, colors):
        jitter = np.random.RandomState(0).uniform(-0.1, 0.1, len(vals))
        ax.scatter(pos + jitter, vals, color=c, edgecolor="black",
                   s=20, alpha=0.6, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pullback Score")
    ax.set_title("Distribution of top-1 margin and gold-answer deficit")

    if standalone:
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Figure 4 — confidence calibration
# ---------------------------------------------------------------------------

def fig_confidence(metrics: dict, ax=None):
    """Stacked bar showing accuracy per confidence bucket."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    cal = metrics["ct_metrics"]["confidence_calibration"]
    levels = ["HIGH", "MEDIUM", "LOW"]
    n      = [cal[l]["n"]        for l in levels]
    acc    = [cal[l]["accuracy"] for l in levels]
    correct = [int(round(n_i * a_i)) for n_i, a_i in zip(n, acc)]
    wrong   = [n_i - c_i for n_i, c_i in zip(n, correct)]

    x = np.arange(len(levels))
    ax.bar(x, correct, color=COLOR_GOLD, label="Correct", alpha=0.85)
    ax.bar(x, wrong, bottom=correct, color=COLOR_WRONG, label="Wrong", alpha=0.85)

    for i, (c, w_, n_, a_) in enumerate(zip(correct, wrong, n, acc)):
        ax.text(i, n_ + 1, f"{a_:.0%}",
                ha="center", fontsize=10, fontweight="bold")
        ax.text(i, c/2, str(c),
                ha="center", color="white", fontsize=9, fontweight="bold")
        if w_:
            ax.text(i, c + w_/2, str(w_),
                    ha="center", color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n_i})" for l, n_i in zip(levels, n)])
    ax.set_ylabel("# of tasks")
    ax.set_title("CT accuracy by confidence level")
    ax.legend(loc="upper right")

    if standalone:
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------

def make_dashboard(metrics: dict, df: pd.DataFrame, out_path: str):
    """All four figures on a single canvas."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.3,
                          height_ratios=[1, 1.1, 1])

    # Row 1: signal quality (two-panel)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1])
    fig_signal_quality(metrics, df, ax_left=ax1a, ax_right=ax1b)

    # Row 2, left: disagreement 2x2
    ax2 = fig.add_subplot(gs[1, 0])
    fig_disagreement(metrics, ax=ax2)

    # Row 2, right: score margins
    ax3 = fig.add_subplot(gs[1, 1])
    fig_score_margins(df, ax=ax3)

    # Row 3, full width: confidence calibration
    ax4 = fig.add_subplot(gs[2, :])
    fig_confidence(metrics, ax=ax4)

    fig.suptitle(
        "AnaloBench — CT pullback vs zero-shot evaluation",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", required=True,
                   help="Directory containing metrics.json and score_distributions.csv")
    p.add_argument("--out", required=True, help="Output directory for figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics, df = load_data(args.analysis_dir)

    # Individual figures
    figs = [
        ("fig1_signal_quality.png",        lambda: fig_signal_quality(metrics, df)),
        ("fig2_disagreement_2x2.png",      lambda: fig_disagreement(metrics)),
        ("fig3_score_margins.png",         lambda: fig_score_margins(df)),
        ("fig4_confidence_calibration.png", lambda: fig_confidence(metrics)),
    ]
    for name, builder in figs:
        fig = builder()
        path = os.path.join(args.out, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")

    # Combined dashboard
    dash_path = os.path.join(args.out, "dashboard.png")
    make_dashboard(metrics, df, dash_path)
    print(f"  wrote {dash_path}")

    print(f"\nAll figures saved to {args.out}")


if __name__ == "__main__":
    main()