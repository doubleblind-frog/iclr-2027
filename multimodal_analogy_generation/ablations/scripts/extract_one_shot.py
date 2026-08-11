"""
extract_one_shot.py — single-call ablation of the four-stage domain-graph
extraction pipeline in extract.py.

Library module only, no CLI of its own — invoked via refinement_loop.py's
--extractor oneshot flag:
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py --extractor oneshot ...
"""

import json
from pathlib import Path

import dspy

from scripts.ct_pullback import compute_pullback
from multimodal_analogy_generation.scripts.extract import (
    ROLE_VOCAB,
    RELATION_VOCAB,
    PULLBACK_THRESHOLD,
    compress_image_to_data_uri,
    _parse_json,
    _normalise_nodes,
    _role_lookup,
    _role_substituted_edges,
    _EMPTY_PULLBACK,
)


# ---------------------------------------------------------------------------
# Single-call signature
# ---------------------------------------------------------------------------

class OneShotExtractDomainGraphs(dspy.Signature):
    """SINGLE-CALL extraction of both domain graphs from a visual metaphor image.
    You must identify both domains,
    their entities (with roles), and the relations between those entities,
    directly from the image, in a single response.

    A visual metaphor combines two distinct conceptual domains:
      - the TARGET domain (the abstract concept the metaphor is *about*) and
      - the SOURCE domain (the concrete concept the target is being *compared to*).
    For example, an image of a single man wreathed in fog depicts TARGET
    "loneliness" (abstract) via SOURCE "fog" (concrete).

    For each domain, list 4-7 key entities. Each node has:
      1. `name`: a concrete, image-grounded label (snake_case, multi-word
         allowed). Be specific: prefer "construction_worker" over
         "agent_figure"; prefer "rusty_nail" over "metal_object". Names
         should NOT themselves be abstract role labels.
      2. `role`: a single label from this closed vocabulary, describing the
         structural function the entity plays WITHIN ITS OWN DOMAIN:
           {ROLE_VOCAB}
         Pick the closest fit; do not invent new role labels. Roles MAY
         repeat within a domain.
      CRITICAL: assign roles based on what each entity does in its own
      domain. Do NOT choose role labels, or which entities to include, in
      order to make the two domains' structure look parallel. Treat the two
      domains as independent typing exercises, even though you are producing
      both at once -- any genuine structural parallel will be discovered
      later by the pullback computation, not engineered here.

    For each domain, list 3-7 directed relations (head, predicate, tail) between its own
    nodes, using node `name`s (not roles):
      - closed vocabulary ONLY: {RELATION_VOCAB}. Pick the closest match if
        nothing is exact; do not invent new predicates.
      - both head and tail MUST be exact matches to names in that domain's
        node list; multiple edges between the same pair are allowed with
        different predicates
      - do NOT add edges between target nodes and source nodes -- each graph
        is intra-domain only
      - base each domain's relations on what is actually happening in the
        image (source) or genuinely true of the abstract concept (target).
        Do not try to make the two domains' relations mirror each other.

    If source_domain is "none", return empty node and relation lists for
    both domains.
    """

    image: dspy.Image = dspy.InputField(desc="The visual metaphor image")
    scratchpad: str = dspy.OutputField(
        desc="Brief unified reasoning: what is literally depicted, what is "
             "metaphorically suggested, target vs. source, then per-domain "
             "node and relation choices. Keep domain reasoning independent "
             "-- do not justify one domain's choices by reference to the other."
    )
    target_domain: str = dspy.OutputField(
        desc="The TARGET domain -- single noun phrase, lowercase, no articles (or 'none')"
    )
    source_domain: str = dspy.OutputField(
        desc="The SOURCE domain -- single noun phrase, lowercase, no articles (or 'none')"
    )
    nodes_json: str = dspy.OutputField(
        desc='JSON: {"target": [{"name": "...", "role": "..."}, ...], '
             '"source": [{"name": "...", "role": "..."}, ...]}'
    )
    relations_json: str = dspy.OutputField(
        desc='JSON: {"target": [[h,r,t],...], "source": [[h,r,t],...]}'
    )


OneShotExtractDomainGraphs.__doc__ = (
    OneShotExtractDomainGraphs.__doc__
    .replace("{ROLE_VOCAB}", ", ".join(ROLE_VOCAB))
    .replace("{RELATION_VOCAB}", ", ".join(RELATION_VOCAB))
)

predict_oneshot = dspy.Predict(OneShotExtractDomainGraphs)


# ---------------------------------------------------------------------------
# Single-run pipeline (drop-in replacement for extract_metaphor_graphs)
# ---------------------------------------------------------------------------

def extract_metaphor_graphs_oneshot(
    image_path: str | Path,
    threshold: float = PULLBACK_THRESHOLD,
) -> dict:
    """Execute the single-call ablation pipeline on one image and return a
    result dict with the SAME schema as extract.extract_metaphor_graphs(),
    so it can be used as a drop-in replacement (e.g. in refine_loop.py)."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = dspy.Image(url=compress_image_to_data_uri(image_path))

    result: dict = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "extraction_mode": "oneshot",
        "stages": {},
    }

    s = predict_oneshot(image=img)

    target_domain = s.target_domain.strip().lower()
    source_domain = s.source_domain.strip().lower()

    result["stages"]["1_oneshot_extraction"] = {
        "scratchpad": s.scratchpad,
        "target_domain": target_domain,
        "source_domain": source_domain,
        "nodes_json_raw": s.nodes_json,
        "relations_json_raw": s.relations_json,
    }

    if source_domain == "none":
        result["status"] = "no_metaphor_detected"
        result["target_domain"] = target_domain
        result["source_domain"] = "none"
        result["nodes_target"] = []
        result["nodes_source"] = []
        result["edges_target"] = []
        result["edges_source"] = []
        result["pullback"] = dict(_EMPTY_PULLBACK)
        result["pullback_score"] = 0.0
        return result

    nodes_raw = _parse_json(s.nodes_json, default={"target": [], "source": []})
    nodes = {
        "target": _normalise_nodes(nodes_raw.get("target", [])),
        "source": _normalise_nodes(nodes_raw.get("source", [])),
    }

    relations = _parse_json(s.relations_json, default={"target": [], "source": []})

    result["status"] = "ok"
    result["target_domain"] = target_domain
    result["source_domain"] = source_domain
    result["nodes_target"] = nodes["target"]
    result["nodes_source"] = nodes["source"]
    result["edges_target"] = relations.get("target", [])
    result["edges_source"] = relations.get("source", [])

    # Pullback computation -- identical to extract.py's Stage 4
    target_roles = _role_lookup(nodes["target"])
    source_roles = _role_lookup(nodes["source"])
    target_edges_for_pullback = _role_substituted_edges(result["edges_target"], target_roles)
    source_edges_for_pullback = _role_substituted_edges(result["edges_source"], source_roles)

    pullback = compute_pullback(
        {"edges": target_edges_for_pullback},
        {"edges": source_edges_for_pullback},
        threshold=threshold,
    )

    clean_tgt = [e for e in result["edges_target"] if len(e) == 3]
    clean_src = [e for e in result["edges_source"] if len(e) == 3]
    for m in pullback.get("matched_pairs", []):
        for role_e, concrete_e in zip(target_edges_for_pullback, clean_tgt):
            if m["source_edge"] == role_e:
                m["source_edge_concrete"] = concrete_e
                break
        for role_e, concrete_e in zip(source_edges_for_pullback, clean_src):
            if m["option_edge"] == role_e:
                m["option_edge_concrete"] = concrete_e
                break

    result["stages"]["2_pullback"] = {
        "threshold": threshold,
        "score": pullback["score"],
        "num_matched": pullback["num_matched"],
        "coverage": pullback["coverage"],
        "chain_length": pullback["chain_length"],
    }
    result["pullback"] = pullback
    result["pullback_score"] = pullback["score"]
    return result