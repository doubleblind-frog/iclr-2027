"""
analobench_ct_no_chain_bonus.py — "chain bonus removed" ablation of the
AnaloBench CT pipeline's pullback score formula.

ABLATION UNDER TEST: does the chain-length bonus term in the pullback score
formula (score = total_similarity + chain_length * 0.5, from ct_pullback.py)
matter for picking the correct analogy, or does raw edge-similarity
(total_similarity) alone drive the selection?

Run (from repo root):
    python benchmarking/analobench/analobench_ct_no_chain_bonus.py --condition ct_llm_no_chain_bonus
    python benchmarking/analobench/analobench_ct_no_chain_bonus.py --condition ct_llm_no_chain_bonus --limit 20
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ct_pullback import compute_confidence, compute_pullbacks_batched

import analobench_ct as base
from analobench_ct import extract_graphs, extract_graph_from_story


def strip_chain_bonus(pullback_results: dict) -> dict:
    """Return a NEW pullback_results dict with 'score' overridden to drop
    the chain-length bonus term for every option. matched_pairs,
    num_matched, coverage, chain_length, and total_similarity are passed
    through unchanged -- only 'score' changes, and the original (with-bonus)
    value is kept alongside for auditing."""
    out = {}
    for k, v in pullback_results.items():
        v2 = dict(v)
        original_score = v2["score"]
        v2["score"] = v2["total_similarity"]
        v2["chain_bonus_removed"] = True
        v2["original_score_with_bonus"] = original_score
        out[k] = v2
    return out


def run_ct_llm_no_chain_bonus(entry: dict) -> dict:
    """Same as analobench_ct.run_ct_llm(), except the pullback score used
    for BOTH answer selection and confidence is the chain-bonus-stripped
    version. Extraction and matching are otherwise identical to production."""
    log: list[str] = []

    try:
        source_graph, option_graphs, extraction_mode = extract_graphs(entry)
    except Exception as e:
        log.append(f"  [Extraction failed: {e}, falling back to independent]")
        source_graph = extract_graph_from_story(entry["story"])
        option_graphs = {
            l: extract_graph_from_story(entry["options"].get(l, ""))
            for l in "ABCD"
        }
        extraction_mode = "independent"

    log.append(
        f"  [Extraction: {extraction_mode}] "
        f"source={len(source_graph.get('edges', []))} edges, "
        f"options={[len(option_graphs[l].get('edges', [])) for l in 'ABCD']}"
    )

    pullback_results_raw = compute_pullbacks_batched(source_graph, option_graphs)
    pullback_results = strip_chain_bonus(pullback_results_raw)  # ABLATION

    algo_best = max("ABCD", key=lambda l: pullback_results[l]["score"])

    for letter in "ABCD":
        r = pullback_results[letter]
        log.append(
            f"  {letter}: matched={r['num_matched']} coverage={r['coverage']:.0%} "
            f"sim={r['total_similarity']} chain={r['chain_length']} "
            f"score={r['score']} (orig_with_bonus={r['original_score_with_bonus']})"
        )

    confidence, score_gap, rel_gap = compute_confidence(pullback_results)
    top_coverage = pullback_results[algo_best]["coverage"]
    log.append(
        f"  Confidence: {confidence} (gap={score_gap}, rel={rel_gap:.2f}, "
        f"coverage={top_coverage:.0%})  ->  {algo_best}"
    )
    tqdm.write("\n".join(log))

    return {
        "answer": algo_best,
        "algo_answer": algo_best,
        "confidence": confidence,
        "extraction_mode": extraction_mode,
        "pullback_results": {
            k: {**v, "matched_pairs": v["matched_pairs"][:5]}
            for k, v in pullback_results.items()
        },
        "source_graph": source_graph,
        "option_graphs": option_graphs,
    }


# ---------------------------------------------------------------------------
# CLI -- registers the ablated condition into analobench_ct's own
# CONDITION_FNS / run_condition, inheriting resume support and output-file
# naming (output_dir/ct_llm_no_chain_bonus.jsonl) unmodified.
# ---------------------------------------------------------------------------

CONDITION_NAME = "ct_llm_no_chain_bonus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=CONDITION_NAME,
                        choices=[CONDITION_NAME],
                        help="Only this ablated condition is available here.")
    parser.add_argument("--dataset", default=base.DATASET_FILE)
    parser.add_argument("--output_dir", default=base.OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default="openrouter/google/gemma-4-31b-it")
    parser.add_argument("--api_base", default=None)
    args = parser.parse_args()

    base._configure_lm(args.model, args.api_base)
    base.CONDITION_FNS[CONDITION_NAME] = run_ct_llm_no_chain_bonus

    base.run_condition(
        CONDITION_NAME, args.dataset, args.output_dir, args.limit, args.workers,
    )


if __name__ == "__main__":
    main()