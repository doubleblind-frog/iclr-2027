"""
ct_pullback.py — Category Theory pullback algorithm.

The algorithm is domain-agnostic: it operates on graphs whose edges are
[source_node, relation, target_node] triples, regardless of whether those
graphs came from story text or visual panel extraction.

get_encoder()
    Returns the shared ModernBERT-large singleton encoder.

run_pullback(src_edges, opt_edges, src_embs, opt_embs, threshold) -> dict
    Core greedy pullback matching given pre-computed embeddings.
    Returns matched pairs, coverage, chain length, and score.

compute_pullback(source_graph, option_graph, threshold) -> dict
    Convenience wrapper: encodes edges then calls run_pullback.

compute_pullbacks_batched(source_graph, option_graphs, threshold) -> dict
    Encodes the source once and all options in a single batch for efficiency.
    option_graphs is a dict keyed by any labels (e.g. "A"/"B"/"C"/"D" for
    AnaloBench, or 0–7 for I-RAVEN candidates).

compute_confidence(pullback_results) -> (confidence, score_gap, relative_gap)
    Computes HIGH / MEDIUM / LOW confidence from a pullback_results dict.
    Works for any set of labels.
"""

import threading
from typing import Any

from sentence_transformers import SentenceTransformer

_encoder: SentenceTransformer | None = None
_encoder_lock = threading.Lock()


def get_encoder() -> SentenceTransformer:
    """Return the shared ModernBERT-large encoder, loading it on first call."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _encoder


# ---------------------------------------------------------------------------
# Core pullback algorithm
# ---------------------------------------------------------------------------

def run_pullback(
    src_edges: list[list[str]],
    opt_edges: list[list[str]],
    src_embs,
    opt_embs,
    threshold: float = 0.55,
) -> dict:
    """
    Greedy semantic pullback between two edge sets given pre-computed embeddings.

    Finds the largest set of (source_edge, option_edge) pairs such that:
      - each edge is used at most once (bipartite matching)
      - node assignments are globally consistent (the pullback condition):
        if source node X maps to option node X', every edge involving X
        must map to an edge involving X'

    The node consistency constraint is what makes this a pullback rather than
    a simple similarity search — it enforces structural isomorphism, not just
    pairwise edge similarity.

    Args:
      src_edges  — list of [src_node, relation, tgt_node] triples
      opt_edges  — list of [src_node, relation, tgt_node] triples
      src_embs   — (len(src_edges), D) normalised embedding matrix
      opt_embs   — (len(opt_edges), D) normalised embedding matrix
      threshold  — minimum cosine similarity to consider a pair

    Returns a dict with:
      matched_pairs    — list of {"source_edge", "option_edge", "similarity"}
      num_matched      — number of matched edge pairs
      total_similarity — sum of similarities of matched pairs
      coverage         — num_matched / len(src_edges)
      chain_length     — longest matched causal chain in source graph
      score            — total_similarity + chain_length * 0.5
    """
    sim_matrix = src_embs @ opt_embs.T

    node_mapping: dict[str, str] = {}
    used_src: set[int] = set()
    used_opt: set[int] = set()
    matched_pairs = []

    # Sort all candidate pairs by similarity descending, filtered by threshold
    pairs_by_sim = sorted(
        [
            (float(sim_matrix[i, j]), i, j)
            for i in range(len(src_edges))
            for j in range(len(opt_edges))
            if float(sim_matrix[i, j]) >= threshold
        ],
        reverse=True,
    )

    for sim, src_i, opt_j in pairs_by_sim:
        if src_i in used_src or opt_j in used_opt:
            continue

        src_e = src_edges[src_i]
        opt_e = opt_edges[opt_j]

        # Check node consistency: proposed node mapping must not conflict
        # with any previously established mapping
        proposed = {src_e[0]: opt_e[0], src_e[2]: opt_e[2]}
        if any(
            node_mapping.get(s) not in (None, o)
            for s, o in proposed.items()
        ):
            continue

        node_mapping.update(proposed)
        used_src.add(src_i)
        used_opt.add(opt_j)
        matched_pairs.append({
            "source_edge": src_e,
            "option_edge": opt_e,
            "similarity": round(sim, 3),
        })

    # Compute longest matched causal chain in source graph
    adj: dict[str, list[str]] = {}
    for pair in matched_pairs:
        s, _, t = pair["source_edge"]
        adj.setdefault(s, []).append(t)

    def longest_path(node: str, visited: set) -> int:
        visited = visited | {node}
        return max(
            (1 + longest_path(nb, visited)
             for nb in adj.get(node, [])
             if nb not in visited),
            default=0,
        )

    chain = max(
        (longest_path(n, set()) for n in adj), default=0
    ) if adj else 0

    total_sim = sum(p["similarity"] for p in matched_pairs)
    coverage = len(matched_pairs) / len(src_edges) if src_edges else 0.0

    return {
        "matched_pairs":   matched_pairs,
        "num_matched":     len(matched_pairs),
        "total_similarity": round(total_sim, 3),
        "coverage":        round(coverage, 3),
        "chain_length":    chain,
        "score":           round(total_sim + chain * 0.5, 3),
    }


def compute_pullback(
    source_graph: dict,
    option_graph: dict,
    threshold: float = 0.55,
) -> dict:
    """
    Convenience wrapper: encode edges from two graphs, then run pullback.

    Useful when you only have one pair of graphs to compare (e.g. computing
    the reference pullback between row 1 and row 2 in I-RAVEN).
    """
    src_edges = [e for e in source_graph.get("edges", []) if len(e) == 3]
    opt_edges = [e for e in option_graph.get("edges", []) if len(e) == 3]

    empty = {
        "matched_pairs": [], "score": 0.0, "chain_length": 0,
        "num_matched": 0, "total_similarity": 0.0, "coverage": 0.0,
    }
    if not src_edges or not opt_edges:
        return empty

    encoder = get_encoder()
    src_embs = encoder.encode(
        [f"{e[0]} {e[1]} {e[2]}" for e in src_edges],
        normalize_embeddings=True,
    )
    opt_embs = encoder.encode(
        [f"{e[0]} {e[1]} {e[2]}" for e in opt_edges],
        normalize_embeddings=True,
    )
    return run_pullback(src_edges, opt_edges, src_embs, opt_embs, threshold)


def compute_pullbacks_batched(
    source_graph: dict,
    option_graphs: dict[Any, dict],
    threshold: float = 0.55,
) -> dict[Any, dict]:
    """
    Encode the source graph once and all option graphs in a single batch,
    then run pullback for each option.

    option_graphs can use any keys — "A"/"B"/"C"/"D" for AnaloBench,
    0–7 for I-RAVEN candidates, or row indices for cross-row comparison.

    Returns a dict with the same keys as option_graphs, each value being
    the pullback result dict from run_pullback().
    """
    src_edges = [e for e in source_graph.get("edges", []) if len(e) == 3]
    empty = {
        "matched_pairs": [], "score": 0.0, "chain_length": 0,
        "num_matched": 0, "total_similarity": 0.0, "coverage": 0.0,
    }

    if not src_edges:
        return {k: empty for k in option_graphs}

    encoder = get_encoder()
    src_embs = encoder.encode(
        [f"{e[0]} {e[1]} {e[2]}" for e in src_edges],
        normalize_embeddings=True,
    )

    # Collect all option edges in insertion order, remembering offsets
    keys = list(option_graphs.keys())
    opt_edges_per = {
        k: [e for e in option_graphs[k].get("edges", []) if len(e) == 3]
        for k in keys
    }
    all_opt_texts = [
        f"{e[0]} {e[1]} {e[2]}"
        for k in keys
        for e in opt_edges_per[k]
    ]

    if not all_opt_texts:
        return {k: empty for k in keys}

    all_opt_embs = encoder.encode(all_opt_texts, normalize_embeddings=True)

    results: dict[Any, dict] = {}
    offset = 0
    for k in keys:
        opt_edges = opt_edges_per[k]
        n = len(opt_edges)
        if n == 0:
            results[k] = empty
        else:
            results[k] = run_pullback(
                src_edges,
                opt_edges,
                src_embs,
                all_opt_embs[offset: offset + n],
                threshold,
            )
        offset += n

    return results


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    pullback_results: dict[Any, dict],
) -> tuple[str, float, float]:
    """
    Compute confidence level from a set of pullback results.

    Uses relative gap (score_gap / best_score) which is robust across
    different embedding similarity distributions and graph sizes.

    Returns:
      confidence   — "HIGH", "MEDIUM", or "LOW"
      score_gap    — absolute difference between best and second-best score
      relative_gap — score_gap / best_score
    """
    keys = list(pullback_results.keys())
    scores = [pullback_results[k]["score"] for k in keys]
    best = max(scores)
    second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
    gap = round(best - second, 3)
    rel_gap = round(gap / best, 3) if best > 0 else 0.0

    top_key = max(keys, key=lambda k: pullback_results[k]["score"])
    top_matched  = pullback_results[top_key]["num_matched"]
    top_coverage = pullback_results[top_key]["coverage"]

    confidence = (
        "HIGH"
        if rel_gap > 0.25 and top_matched >= 2 and top_coverage >= 0.3
        else "MEDIUM"
        if rel_gap > 0.10 or top_matched >= 3
        else "LOW"
    )
    return confidence, gap, rel_gap