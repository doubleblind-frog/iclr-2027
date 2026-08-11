"""
analyse_analobench.py — evaluation of CT-pullback vs direct zero-shot.

Computes metrics that go beyond accuracy to characterise the quality of the
CT scoring signal itself.

Usage (from repo root):
    python benchmarking/analobench/analyse_analobench.py --ct benchmarking/analobench/results/ct_llm.jsonl --zs benchmarking/analobench/results/direct_zeroshot.jsonl --csv benchmarking/analobench/test_analobench.csv --out benchmarking/analobench/results/analysis/

Outputs (in --out):
    metrics.json                     — all computed metrics in one place
    metrics_summary.md               — readable summary
    disagreements_ct_only.md         — cases where CT correct, ZS wrong
    disagreements_zs_only.md         — cases where ZS correct, CT wrong
    score_distributions.csv          — per-task score data for plotting
"""

import argparse
import csv
import json
import os
from collections import Counter
from statistics import mean, median, stdev

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file produced by analobench_ct.py."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_stories(csv_path: str) -> dict[str, dict]:
    """Load original AnaloBench rows keyed by index for narrative lookup."""
    by_index: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            options, current, buf = {}, None, []
            for line in row["Options"].split("\n"):
                line = line.strip()
                if line[:3] in ("A. ", "B. ", "C. ", "D. "):
                    if current:
                        options[current] = "\n".join(buf).strip()
                    current, buf = line[0], [line[3:]]
                else:
                    buf.append(line)
            if current:
                options[current] = "\n".join(buf).strip()
            by_index[str(row["Index"])] = {
                "sentence": row["Sentence"],
                "story":    row["Story"],
                "options":  options,
                "label":    row["Label"].strip(),
            }
    return by_index


# ---------------------------------------------------------------------------
# Per-task derived quantities
# ---------------------------------------------------------------------------

def per_task_signals(ct_record: dict) -> dict:
    """
    Extract score-signal quantities from one CT result row.

    Returns a dict with:
      label                — gold-truth letter
      predicted            — CT's chosen letter
      correct              — bool
      confidence           — HIGH / MEDIUM / LOW
      scores               — {A: float, B: float, C: float, D: float}
      ranks                — {A: int 1..4, ...}  (1 = highest)
      rank_of_correct      — int 1..4
      rrank_of_correct     — 1 / rank_of_correct  (reciprocal rank)
      score_of_correct     — float
      score_of_predicted   — float
      score_margin_top1    — top1 - top2 (always ≥ 0)
      score_margin_correct — score(correct) - top1   (≤ 0 when wrong)
      correct_in_top2      — bool
      correct_score_zero   — bool, score(correct) == 0
      coverage_of_predicted, coverage_of_correct
      total_sim_of_predicted, total_sim_of_correct
      chain_of_predicted, chain_of_correct
    """
    pull = ct_record.get("pullback_results", {})
    label = ct_record["label"]
    predicted = ct_record["predicted"]

    scores = {l: float(pull.get(l, {}).get("score", 0.0)) for l in "ABCD"}
    sorted_letters = sorted("ABCD", key=lambda l: scores[l], reverse=True)
    ranks = {l: sorted_letters.index(l) + 1 for l in "ABCD"}

    top1 = scores[sorted_letters[0]]
    top2 = scores[sorted_letters[1]]

    return {
        "index":             ct_record["index"],
        "label":             label,
        "predicted":         predicted,
        "correct":           bool(ct_record.get("correct", predicted == label)),
        "confidence":        ct_record.get("confidence"),
        "scores":            scores,
        "ranks":             ranks,
        "rank_of_correct":   ranks[label],
        "rrank_of_correct":  1.0 / ranks[label],
        "score_of_correct":  scores[label],
        "score_of_predicted": scores[predicted],
        "score_margin_top1":    round(top1 - top2, 4),
        "score_margin_correct": round(scores[label] - top1, 4),
        "correct_in_top2":   ranks[label] <= 2,
        "correct_score_zero": scores[label] == 0.0,
        "coverage_of_predicted": pull.get(predicted, {}).get("coverage"),
        "coverage_of_correct":   pull.get(label, {}).get("coverage"),
        "total_sim_of_predicted": pull.get(predicted, {}).get("total_similarity"),
        "total_sim_of_correct":   pull.get(label, {}).get("total_similarity"),
        "chain_of_predicted": pull.get(predicted, {}).get("chain_length"),
        "chain_of_correct":   pull.get(label, {}).get("chain_length"),
    }


# ---------------------------------------------------------------------------
# Metric aggregations
# ---------------------------------------------------------------------------

def aggregate_metrics(signals: list[dict]) -> dict:
    """Compute aggregate metrics across all tasks for one condition."""
    n = len(signals)
    correct = sum(s["correct"] for s in signals)
    in_top2 = sum(s["correct_in_top2"] for s in signals)
    score_zero = sum(s["correct_score_zero"] for s in signals)

    rank_dist = Counter(s["rank_of_correct"] for s in signals)

    correct_signals = [s for s in signals if s["correct"]]
    wrong_signals   = [s for s in signals if not s["correct"]]

    def _agg(vals: list, label: str) -> dict:
        vals = [v for v in vals if v is not None]
        if not vals:
            return {f"{label}_n": 0}
        return {
            f"{label}_n":      len(vals),
            f"{label}_mean":   round(mean(vals), 4),
            f"{label}_median": round(median(vals), 4),
            f"{label}_stdev":  round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            f"{label}_min":    round(min(vals), 4),
            f"{label}_max":    round(max(vals), 4),
        }

    out = {
        "n_tasks":           n,
        "accuracy":          round(correct / n, 4) if n else 0.0,
        "top2_accuracy":     round(in_top2 / n, 4) if n else 0.0,
        "mean_reciprocal_rank": round(mean(s["rrank_of_correct"] for s in signals), 4) if n else 0.0,
        "rank_of_correct_distribution": {
            "rank_1": rank_dist.get(1, 0),
            "rank_2": rank_dist.get(2, 0),
            "rank_3": rank_dist.get(3, 0),
            "rank_4": rank_dist.get(4, 0),
        },
        "frac_correct_score_zero": round(score_zero / n, 4) if n else 0.0,
        "n_correct_score_zero":    score_zero,
    }

    # Score-margin descriptive stats
    out.update(_agg([s["score_margin_top1"] for s in signals], "margin_top1_all"))
    out.update(_agg([s["score_margin_top1"] for s in correct_signals], "margin_top1_when_correct"))
    out.update(_agg([s["score_margin_top1"] for s in wrong_signals], "margin_top1_when_wrong"))

    # How far behind top1 is the correct answer when CT picks wrong?
    deficit = [-s["score_margin_correct"] for s in wrong_signals]
    out.update(_agg(deficit, "deficit_when_wrong"))

    # Confidence calibration
    conf_buckets: dict[str, list] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for s in signals:
        c = s.get("confidence")
        if c in conf_buckets:
            conf_buckets[c].append(s["correct"])
    out["confidence_calibration"] = {
        c: {
            "n":        len(vals),
            "frac":     round(len(vals) / n, 4) if n else 0,
            "accuracy": round(sum(vals) / len(vals), 4) if vals else 0,
        }
        for c, vals in conf_buckets.items()
    }

    # Coverage and chain comparisons
    out.update(_agg([s["coverage_of_correct"]   for s in signals], "coverage_of_correct"))
    out.update(_agg([s["coverage_of_predicted"] for s in signals], "coverage_of_predicted"))
    out.update(_agg([s["chain_of_correct"]   for s in signals], "chain_of_correct"))
    out.update(_agg([s["chain_of_predicted"] for s in signals], "chain_of_predicted"))

    return out


def compare_conditions(ct_signals: list[dict], zs_records: list[dict]) -> dict:
    """Cross-condition comparison: agreement, disagreement, McNemar's test."""
    ct_by_idx = {s["index"]: s for s in ct_signals}
    zs_by_idx = {str(r["index"]): r for r in zs_records}
    common = sorted(set(ct_by_idx) & set(zs_by_idx))

    both_right = ct_only = zs_only = both_wrong = 0
    for idx in common:
        c_ok = ct_by_idx[idx]["correct"]
        z_ok = bool(zs_by_idx[idx].get("correct", zs_by_idx[idx]["predicted"] == zs_by_idx[idx]["label"]))
        if c_ok and z_ok:    both_right += 1
        elif c_ok and not z_ok: ct_only += 1
        elif not c_ok and z_ok: zs_only += 1
        else:                  both_wrong += 1

    n = len(common)
    n_disagree = ct_only + zs_only

    # McNemar's exact test
    mcnemar_p = mcnemar_exact_p(ct_only, zs_only)

    return {
        "n_compared":      n,
        "both_right":      both_right,
        "ct_only_right":   ct_only,
        "zs_only_right":   zs_only,
        "both_wrong":      both_wrong,
        "agreement_rate":  round((both_right + both_wrong) / n, 4) if n else 0,
        "disagreement_rate": round(n_disagree / n, 4) if n else 0,
        "ct_accuracy_within_disagreements": round(ct_only / n_disagree, 4) if n_disagree else None,
        "mcnemar_p_value": round(mcnemar_p, 4) if mcnemar_p is not None else None,
    }


def mcnemar_exact_p(b: int, c: int) -> float | None:
    """
    Exact two-sided McNemar test on disagreement counts (b, c).
    Tests H0: P(CT right & ZS wrong) == P(ZS right & CT wrong).
    Returns the two-sided p-value, or None if degenerate.
    """
    n = b + c
    if n == 0:
        return None
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Qualitative report
# ---------------------------------------------------------------------------

def qualitative_report(
    signals: list[dict],
    zs_records: list[dict],
    stories: dict[str, dict],
    ct_records_by_idx: dict[str, dict],
    side: str,
) -> str:
    """
    Produce markdown listing one disagreement set with full context.
    side: "ct_only" (CT right, ZS wrong) or "zs_only" (ZS right, CT wrong).
    """
    ct_by_idx = {s["index"]: s for s in signals}
    zs_by_idx = {str(r["index"]): r for r in zs_records}

    cases = []
    for idx in sorted(set(ct_by_idx) & set(zs_by_idx)):
        c_ok = ct_by_idx[idx]["correct"]
        z_ok = bool(zs_by_idx[idx].get("correct",
                    zs_by_idx[idx]["predicted"] == zs_by_idx[idx]["label"]))
        if side == "ct_only" and c_ok and not z_ok:
            cases.append(idx)
        elif side == "zs_only" and z_ok and not c_ok:
            cases.append(idx)

    title = ("CT correct, Zero-shot wrong" if side == "ct_only"
             else "Zero-shot correct, CT wrong")
    lines = [f"# Disagreement cases — {title}",
             f"\nTotal: **{len(cases)}** cases\n"]

    for idx in cases:
        ct_sig = ct_by_idx[idx]
        zs_rec = zs_by_idx[idx]
        story  = stories.get(idx, {})
        ct_rec = ct_records_by_idx.get(idx, {})

        lines.append(f"\n---\n\n## Task {idx}")
        if story.get("sentence"):
            lines.append(f"**Theme/proverb:** {story['sentence']}\n")

        lines.append(f"**Gold answer:** {ct_sig['label']}")
        lines.append(f"**CT predicted:** {ct_sig['predicted']} "
                     f"({'✓' if ct_sig['correct'] else '✗'})")
        lines.append(f"**ZS predicted:** {zs_rec['predicted']} "
                     f"({'✓' if zs_rec.get('correct') else '✗'})")
        lines.append(f"**CT confidence:** {ct_sig['confidence']}")

        # Score table
        lines.append("\n**CT pullback scores:**\n")
        lines.append("| Option | Score | Coverage | Matched | Chain | Rank |")
        lines.append("|--------|-------|----------|---------|-------|------|")
        pull = ct_rec.get("pullback_results", {})
        for letter in "ABCD":
            r = pull.get(letter, {})
            mark = ""
            if letter == ct_sig["label"]:     mark += " ★ gold"
            if letter == ct_sig["predicted"]: mark += " ← CT"
            lines.append(
                f"| {letter}{mark} | {r.get('score', 0):.3f} | "
                f"{r.get('coverage', 0):.0%} | {r.get('num_matched', 0)} | "
                f"{r.get('chain_length', 0)} | {ct_sig['ranks'][letter]} |"
            )

        # Reasoning from zero-shot if available
        if zs_rec.get("reasoning"):
            lines.append(f"\n**ZS reasoning:** {zs_rec['reasoning'][:400]}...")

        # Story bodies
        if story.get("story"):
            lines.append(f"\n**Source story:**\n> {story['story'][:600]}...")

        gold = ct_sig["label"]
        pred = ct_sig["predicted"]
        for letter in dict.fromkeys([gold, pred]):  # dedupe, preserve order
            opt = story.get("options", {}).get(letter, "")
            if opt:
                tag = []
                if letter == gold: tag.append("GOLD")
                if letter == pred: tag.append("CT-PREDICTED")
                if letter == zs_rec["predicted"]: tag.append("ZS-PREDICTED")
                lines.append(f"\n**Option {letter} ({', '.join(tag)}):**\n> {opt[:600]}...")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_summary_md(
    ct_metrics: dict,
    zs_metrics: dict,
    comparison: dict,
    out_path: str,
) -> None:
    L = []
    L.append("# AnaloBench evaluation — beyond-accuracy metrics\n")

    L.append("## Headline numbers\n")
    L.append(f"- Tasks compared: **{comparison['n_compared']}**")
    L.append(f"- CT accuracy: **{ct_metrics['accuracy']:.1%}**")
    L.append(f"- ZS accuracy: **{zs_metrics['accuracy']:.1%}**")
    L.append(f"- Agreement rate: **{comparison['agreement_rate']:.1%}**\n")

    L.append("## Score-signal quality (CT only)\n")
    L.append(f"- **Mean Reciprocal Rank** of correct answer: "
             f"**{ct_metrics['mean_reciprocal_rank']:.3f}** "
             "(1.0 = always rank 1, 0.25 = uniform)")
    L.append(f"- **Top-2 accuracy**: **{ct_metrics['top2_accuracy']:.1%}** "
             "— how often the correct answer is in CT's top two scores")
    L.append(f"- **Correct answer ranked 1st**: "
             f"{ct_metrics['rank_of_correct_distribution']['rank_1']}")
    L.append(f"- **Correct answer ranked 2nd**: "
             f"{ct_metrics['rank_of_correct_distribution']['rank_2']}")
    L.append(f"- **Correct answer ranked 3rd**: "
             f"{ct_metrics['rank_of_correct_distribution']['rank_3']}")
    L.append(f"- **Correct answer ranked 4th**: "
             f"{ct_metrics['rank_of_correct_distribution']['rank_4']}")
    L.append(f"- **Correct answer scored 0.0** "
             "(no signal at all): "
             f"{ct_metrics['n_correct_score_zero']} / {ct_metrics['n_tasks']} "
             f"({ct_metrics['frac_correct_score_zero']:.1%})\n")

    L.append("### When CT is correct\n")
    L.append(f"- Mean top1–top2 margin: "
             f"{ct_metrics.get('margin_top1_when_correct_mean', 'n/a')}")
    L.append(f"- Median top1–top2 margin: "
             f"{ct_metrics.get('margin_top1_when_correct_median', 'n/a')}\n")

    L.append("### When CT is wrong\n")
    L.append(f"- Mean top1–top2 margin (CT picked wrong winner this confidently): "
             f"{ct_metrics.get('margin_top1_when_wrong_mean', 'n/a')}")
    L.append(f"- Mean deficit of correct answer behind top1: "
             f"{ct_metrics.get('deficit_when_wrong_mean', 'n/a')}")
    L.append(f"- Median deficit: "
             f"{ct_metrics.get('deficit_when_wrong_median', 'n/a')}\n")

    L.append("## Confidence calibration\n")
    L.append("| Confidence | N tasks | Fraction | Accuracy |")
    L.append("|------------|---------|----------|----------|")
    for level in ("HIGH", "MEDIUM", "LOW"):
        c = ct_metrics["confidence_calibration"].get(level, {})
        L.append(f"| {level} | {c.get('n', 0)} | {c.get('frac', 0):.1%} | "
                 f"{c.get('accuracy', 0):.1%} |")
    L.append("")

    L.append("## CT vs zero-shot comparison\n")
    L.append("| Outcome | Count |")
    L.append("|---------|-------|")
    L.append(f"| Both right | {comparison['both_right']} |")
    L.append(f"| CT right, ZS wrong | **{comparison['ct_only_right']}** |")
    L.append(f"| ZS right, CT wrong | **{comparison['zs_only_right']}** |")
    L.append(f"| Both wrong | {comparison['both_wrong']} |\n")

    L.append(f"- **Agreement rate**: {comparison['agreement_rate']:.1%}")
    L.append(f"- **Disagreement rate**: {comparison['disagreement_rate']:.1%}")
    if comparison["ct_accuracy_within_disagreements"] is not None:
        L.append(f"- **CT's win rate within disagreements**: "
                 f"{comparison['ct_accuracy_within_disagreements']:.1%}")
    if comparison["mcnemar_p_value"] is not None:
        L.append(f"- **McNemar two-sided exact p-value**: "
                 f"{comparison['mcnemar_p_value']:.4f}  "
                 "(low p = methods make systematically different errors)\n")

    L.append("## How to read these numbers\n")
    L.append("- **MRR** measures the score signal directly. If MRR is high "
             "(close to 1) but accuracy is moderate, the algorithm is *almost* "
             "right — useful as an RL signal even when its top-1 is wrong.")
    L.append("- **Top-2 accuracy − top-1 accuracy** quantifies how much "
             "discrimination is lost in the final argmax step.")
    L.append("- **Frac correct scored 0.0** is the no-signal rate. These cases "
             "are unrecoverable for the pullback regardless of decision rule.")
    L.append("- **Margin when wrong** vs **margin when right** characterises "
             "calibration — ideally CT is hesitant when wrong and decisive when right.")
    L.append("- **Disagreement structure** (CT-only vs ZS-only wins) tells you "
             "what kinds of analogies each method handles, even when their "
             "accuracies are comparable.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------
# CSV export for plotting
# ---------------------------------------------------------------------------

def write_score_distributions_csv(signals: list[dict], out_path: str) -> None:
    """Per-task data suitable for plotting score distributions."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "label", "predicted", "correct", "confidence",
            "score_A", "score_B", "score_C", "score_D",
            "rank_of_correct", "rrank_of_correct",
            "score_of_correct", "score_of_predicted",
            "margin_top1", "deficit_when_wrong",
            "coverage_of_correct", "coverage_of_predicted",
            "chain_of_correct", "chain_of_predicted",
        ])
        for s in signals:
            w.writerow([
                s["index"], s["label"], s["predicted"], int(s["correct"]),
                s["confidence"],
                s["scores"]["A"], s["scores"]["B"],
                s["scores"]["C"], s["scores"]["D"],
                s["rank_of_correct"], round(s["rrank_of_correct"], 4),
                s["score_of_correct"], s["score_of_predicted"],
                s["score_margin_top1"],
                -s["score_margin_correct"] if not s["correct"] else 0,
                s.get("coverage_of_correct"), s.get("coverage_of_predicted"),
                s.get("chain_of_correct"), s.get("chain_of_predicted"),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct",   required=True, help="ct_llm.jsonl path")
    p.add_argument("--zs",   required=True, help="direct_zeroshot.jsonl path")
    p.add_argument("--csv",  required=True, help="AnaloBench CSV path")
    p.add_argument("--out",  required=True, help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ct_records = load_jsonl(args.ct)
    zs_records = load_jsonl(args.zs)
    stories    = load_stories(args.csv)

    print(f"Loaded {len(ct_records)} CT records, "
          f"{len(zs_records)} ZS records, "
          f"{len(stories)} story entries")

    # Per-task signals from CT (rich quantities) and ZS (just correctness)
    ct_signals = [per_task_signals(r) for r in ct_records]

    # ZS treated as a thinner record: just label + predicted + correct
    zs_signals = [{
        "index":     str(r["index"]),
        "label":     r["label"],
        "predicted": r["predicted"],
        "correct":   bool(r.get("correct", r["predicted"] == r["label"])),
        # ZS has no per-option scores, so rank-style metrics are not defined
    } for r in zs_records]

    # Aggregate metrics
    ct_metrics = aggregate_metrics(ct_signals)
    # ZS gets a degraded version (only the metrics that don't need scores)
    zs_metrics = {
        "n_tasks":  len(zs_signals),
        "accuracy": round(sum(s["correct"] for s in zs_signals) / len(zs_signals), 4)
                    if zs_signals else 0.0,
    }
    comparison = compare_conditions(ct_signals, zs_records)

    # Write everything
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({
            "ct_metrics":  ct_metrics,
            "zs_metrics":  zs_metrics,
            "comparison":  comparison,
        }, f, indent=2)

    write_summary_md(
        ct_metrics, zs_metrics, comparison,
        os.path.join(args.out, "metrics_summary.md"),
    )

    write_score_distributions_csv(
        ct_signals,
        os.path.join(args.out, "score_distributions.csv"),
    )

    ct_records_by_idx = {str(r["index"]): r for r in ct_records}

    with open(os.path.join(args.out, "disagreements_ct_only.md"), "w") as f:
        f.write(qualitative_report(
            ct_signals, zs_records, stories,
            ct_records_by_idx, side="ct_only",
        ))

    with open(os.path.join(args.out, "disagreements_zs_only.md"), "w") as f:
        f.write(qualitative_report(
            ct_signals, zs_records, stories,
            ct_records_by_idx, side="zs_only",
        ))

    # Console summary
    print("\n" + "=" * 60)
    print(f"CT accuracy:           {ct_metrics['accuracy']:.1%}")
    print(f"CT MRR:                {ct_metrics['mean_reciprocal_rank']:.3f}")
    print(f"CT top-2 accuracy:     {ct_metrics['top2_accuracy']:.1%}")
    print(f"CT correct scored 0:   {ct_metrics['n_correct_score_zero']}/"
          f"{ct_metrics['n_tasks']} ({ct_metrics['frac_correct_score_zero']:.1%})")
    print(f"ZS accuracy:           {zs_metrics['accuracy']:.1%}")
    print()
    print(f"Both right:            {comparison['both_right']}")
    print(f"CT only right:         {comparison['ct_only_right']}")
    print(f"ZS only right:         {comparison['zs_only_right']}")
    print(f"Both wrong:            {comparison['both_wrong']}")
    print(f"McNemar p-value:       {comparison['mcnemar_p_value']}")
    print()
    print(f"Outputs in: {args.out}")


if __name__ == "__main__":
    main()