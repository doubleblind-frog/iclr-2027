"""
analobench_ct_no_role_abstraction.py — "no role abstraction" ablation of the
AnaloBench CT pipeline's graph extraction.

ABLATION UNDER TEST: does forcing an abstract, shared role vocabulary across
the five stories (agent/goal/obstacle/method/outcome/...), so that
structurally parallel stories get IDENTICAL node labels, matter for finding
the correct analogy -- or would concrete, per-story node extraction (no
forced cross-story alignment) work just as well once fed into the same
pullback-matching algorithm?

Run (from repo root):
    python benchmarking/analobench/analobench_ct_no_role_abstraction.py --condition ct_llm_no_role
    python benchmarking/analobench/analobench_ct_no_role_abstraction.py --condition ct_llm_no_role --limit 20 --model openrouter/google/gemma-4-31b-it
"""

import argparse
import sys
from pathlib import Path

import dspy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ct_pullback import compute_confidence, compute_pullbacks_batched

import analobench_ct as base
from analobench_ct import RELATION_GUIDANCE, safe_json


# ---------------------------------------------------------------------------
# Ablated instructions -- concrete labels, no cross-story alignment
# ---------------------------------------------------------------------------

NO_ROLE_GRAPH_INSTRUCTIONS = """
Extract a CAUSAL SCHEMA GRAPH of the story.
Output JSON with "nodes" (list of strings) and "edges" (list of [src, rel, tgt]).

NODE RULES:
- 6-9 nodes. Use CONCRETE, story-specific labels naming the actual
  entities, objects, or events described in the story (e.g.
  "old_fisherman", "torn_net", "coastal_storm"), NOT abstract role
  placeholders like "agent" or "obstacle".
- Compound labels are fine.

EDGE RULES -- produce AT LEAST 8 edges, using the story's own entities:
  causal chains, actor intentions and obstacles, tools/methods used,
  outcomes, costs/risks, and any helper or authority actions -- whatever
  is genuinely present in THIS SPECIFIC story.

Use ONLY these relations:
  desires, faces, wants_to_avoid, uses, applies, enables, prevents,
  causes, leads_to, triggers, results_in, reveals, deceives, blocks,
  produces, depends_on, relies_on, supports, opposes
"""


class ExtractGraphNoRoleAbstraction(dspy.Signature):
    """Extract a structured graph from a story using CONCRETE, story-specific
    node labels (not abstract role placeholders). Output JSON with "nodes"
    (list of strings) and "edges" (list of [src, rel, tgt])."""
    story: str = dspy.InputField(desc="The story to represent as a graph")
    graph_instructions: str = dspy.InputField(desc="Node and edge extraction rules")
    relation_guidance: str = dspy.InputField(desc="Preferred relation label vocabulary")
    graph_json: str = dspy.OutputField(
        desc='JSON: {"nodes": [...], "edges": [[src, rel, tgt], ...]}'
    )


class ExtractNodesJointlyNoRoleAbstraction(dspy.Signature):
    """STAGE 1 of two-stage graph construction -- NO-ROLE-ABSTRACTION ABLATION.

    You are given a source story and four option stories (A, B, C, D).

    For each story, extract the key entities and concepts AS THEY NATURALLY
    APPEAR in that story -- concrete, story-specific labels (e.g. "the
    fisherman", "the storm", "the broken net"), not abstract functional
    roles. Assess each story INDEPENDENTLY. Do NOT try to make different
    stories use the same vocabulary, even if they seem to share a similar
    underlying pattern -- extract each story on its own terms and let its
    natural vocabulary stand, whatever it turns out to be.

    NODE RULES:
    - 5-8 nodes per story.
    - Concrete, story-specific labels: name the actual entities, objects,
      or events as described in the story text (e.g. "old_fisherman",
      "torn_net", "storm", "harbor"), not abstract role placeholders like
      "agent" or "obstacle".
    - Compound labels are fine.
    - Do NOT reuse a label across stories unless the exact same concrete
      entity genuinely appears in more than one story (which will be rare).

    Output JSON with keys "source", "A", "B", "C", "D", each a list of
    strings.
    """
    source_story: str = dspy.InputField(desc="The source story")
    option_a: str = dspy.InputField(desc="Option A story")
    option_b: str = dspy.InputField(desc="Option B story")
    option_c: str = dspy.InputField(desc="Option C story")
    option_d: str = dspy.InputField(desc="Option D story")
    scratchpad: str = dspy.OutputField(
        desc="Per-story analysis of the concrete entities present in each "
             "story. Do NOT attempt cross-story alignment here -- assess "
             "each story independently."
    )
    nodes_json: str = dspy.OutputField(
        desc='JSON: {"source": [...], "A": [...], "B": [...], "C": [...], "D": [...]}'
    )


class ExtractRelationsJointlyNoRoleAbstraction(dspy.Signature):
    """STAGE 2 of two-stage graph construction -- NO-ROLE-ABSTRACTION ABLATION.

    You are given five stories and their Stage 1 nodes (concrete,
    story-specific labels -- NOT abstract roles). Find all meaningful
    causal relations between nodes for each story, based ONLY on what is
    genuinely true within that individual story. Assess each story
    INDEPENDENTLY -- do NOT try to make relations consistent, parallel, or
    aligned across stories.

    RELATION RULES:
    - Only add a relation if genuinely supported by the story text.
    - Multiple relations between the same pair allowed.
    - Produce AT LEAST 7 edges per story.
    - Closed vocabulary ONLY:
        causes, results_in, leads_to, triggers, produces,
        enables, prevents, blocks, relies_on,
        desires, wants_to_avoid, faces, pursues,
        reveals, deceives, uses, adjusts_to, provides, receives
    - BANNED: affects, relates_to, involves, is, has, connects_to, is_about

    Work through for every story, using ONLY that story's OWN Stage 1 node
    labels:
      1. main actor desires/pursues -> goal-like node
      2. main actor faces/wants_to_avoid -> obstacle-like node
      3. obstacle causes/prevents -> [effect]
      4. main actor uses -> method/tool-like node
      5. method enables/prevents -> [target]
      6. method leads_to/results_in -> outcome-like node
      7. any helper/authority action and its effect
      8. any cost, risk, or side-effect edges

    Output JSON with keys "source", "A", "B", "C", "D", each a list of
    [source_node, relation, target_node] triples using only that story's
    OWN Stage 1 node labels.
    """
    source_story: str = dspy.InputField(desc="The source story")
    option_a: str = dspy.InputField(desc="Option A story")
    option_b: str = dspy.InputField(desc="Option B story")
    option_c: str = dspy.InputField(desc="Option C story")
    option_d: str = dspy.InputField(desc="Option D story")
    nodes_json: str = dspy.InputField(desc="Node lists from Stage 1 (concrete labels)")
    stage1_scratchpad: str = dspy.InputField(desc="Per-story analysis from Stage 1")
    relations_json: str = dspy.OutputField(
        desc='JSON: {"source": [[n,r,n],...], "A": [...], "B": [...], "C": [...], "D": [...]}'
    )


predict_extract_no_role = dspy.Predict(ExtractGraphNoRoleAbstraction)
predict_nodes_no_role = dspy.Predict(ExtractNodesJointlyNoRoleAbstraction)
predict_relations_no_role = dspy.Predict(ExtractRelationsJointlyNoRoleAbstraction)


def extract_graph_from_story_no_role(story: str) -> dict:
    """No-role-abstraction-consistent single-story fallback extractor, used
    when the staged joint call fails -- mirrors extract_graph_from_story()
    but WITHOUT reintroducing abstract role labels."""
    result = predict_extract_no_role(
        story=story,
        graph_instructions=NO_ROLE_GRAPH_INSTRUCTIONS,
        relation_guidance=RELATION_GUIDANCE,  # relation vocab is unaffected by this ablation
    )
    return safe_json(result.graph_json)


def extract_graphs_no_role_abstraction(entry: dict) -> tuple[dict, dict, str]:
    """Ablated two-stage extraction: concrete, per-story labels, no forced
    cross-story alignment in either stage. Same call structure (Stage 1
    nodes -> Stage 2 relations, same five-story joint context) as
    extract_graphs(), so extraction ARCHITECTURE is held constant; only the
    labelling scheme changes."""
    node_result = predict_nodes_no_role(
        source_story=entry["story"],
        option_a=entry["options"].get("A", ""),
        option_b=entry["options"].get("B", ""),
        option_c=entry["options"].get("C", ""),
        option_d=entry["options"].get("D", ""),
    )
    all_nodes = safe_json(node_result.nodes_json)
    if not isinstance(all_nodes, dict) or "source" not in all_nodes:
        raise ValueError("No-role Stage 1: node extraction returned invalid structure")

    rel_result = predict_relations_no_role(
        source_story=entry["story"],
        option_a=entry["options"].get("A", ""),
        option_b=entry["options"].get("B", ""),
        option_c=entry["options"].get("C", ""),
        option_d=entry["options"].get("D", ""),
        nodes_json=node_result.nodes_json,
        stage1_scratchpad=node_result.scratchpad,
    )
    all_relations = safe_json(rel_result.relations_json)
    if not isinstance(all_relations, dict) or "source" not in all_relations:
        raise ValueError("No-role Stage 2: relation extraction returned invalid structure")

    source_graph = {
        "nodes": all_nodes.get("source", []),
        "edges": all_relations.get("source", []),
    }
    option_graphs = {
        l: {"nodes": all_nodes.get(l, []), "edges": all_relations.get(l, [])}
        for l in "ABCD"
    }
    return source_graph, option_graphs, "staged_no_role_abstraction"


def run_ct_llm_no_role(entry: dict) -> dict:
    """Same as analobench_ct.run_ct_llm(), except graph extraction uses the
    no-role-abstraction path (both primary and fallback). Matching,
    confidence, and logging are IDENTICAL to production -- only the
    labelling scheme fed into them differs."""
    log: list[str] = []

    try:
        source_graph, option_graphs, extraction_mode = extract_graphs_no_role_abstraction(entry)
    except Exception as e:
        log.append(f"  [Extraction failed: {e}, falling back to independent (no-role)]")
        source_graph = extract_graph_from_story_no_role(entry["story"])
        option_graphs = {
            l: extract_graph_from_story_no_role(entry["options"].get(l, ""))
            for l in "ABCD"
        }
        extraction_mode = "independent_no_role_abstraction"

    log.append(
        f"  [Extraction: {extraction_mode}] "
        f"source={len(source_graph.get('edges', []))} edges, "
        f"options={[len(option_graphs[l].get('edges', [])) for l in 'ABCD']}"
    )

    pullback_results = compute_pullbacks_batched(source_graph, option_graphs)
    algo_best = max("ABCD", key=lambda l: pullback_results[l]["score"])

    for letter in "ABCD":
        r = pullback_results[letter]
        log.append(
            f"  {letter}: matched={r['num_matched']} coverage={r['coverage']:.0%} "
            f"sim={r['total_similarity']} chain={r['chain_length']} score={r['score']}"
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
# CONDITION_FNS / run_condition, so resume support, output-file naming
# (output_dir/ct_llm_no_role.jsonl), and confidence-breakdown reporting are
# all inherited unmodified.
# ---------------------------------------------------------------------------

CONDITION_NAME = "ct_llm_no_role"


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
    base.CONDITION_FNS[CONDITION_NAME] = run_ct_llm_no_role

    base.run_condition(
        CONDITION_NAME, args.dataset, args.output_dir, args.limit, args.workers,
    )


if __name__ == "__main__":
    main()