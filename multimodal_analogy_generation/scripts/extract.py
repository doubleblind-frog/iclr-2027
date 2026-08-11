"""
extract.py — four-stage pipeline that extracts domain graphs from a single
visual metaphor image and computes a categorical pullback score measuring
the structural strength of the metaphor.

Pipeline stages:
    Stage 1  Identify the source and target domains depicted in the image.
    Stage 2  Extract nodes for each domain. Each node is a (name, role) pair
             where `name` is concrete and image-grounded and `role` is drawn
             from a closed semantic-role vocabulary. Roles are assigned
             independently per domain.
    Stage 3  Extract directed relations (edges) for each domain using
             node names only. Roles are intentionally hidden so
             Stage 3 stays grounded in image content rather than reasoning
             about parallel structure across domains.
    Stage 4  Compute the categorical pullback.

Authentication:
    The default model (Claude Sonnet 4.6 via the Anthropic API) expects an
    Anthropic API key in the .env file under the variable name
    ANTHROPIC_API_KEY.

Run:
    python multimodal_analogy_generation/scripts/extract.py --generator human
    python multimodal_analogy_generation/scripts/extract.py --generator human --threshold 0.40
    python multimodal_analogy_generation/scripts/extract.py --generator flux-klein-zero --limit 10
    python multimodal_analogy_generation/scripts/extract.py --generator gpt-image-cot --model openrouter/google/gemma-4-31b-it
"""

import argparse
import csv
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import dspy
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from scripts.ct_pullback import compute_pullback
import base64
import io

load_dotenv()


# ---------------------------------------------------------------------------
# Generators and corresponding file paths
# ---------------------------------------------------------------------------

GENERATORS = {
    "human":            "Human-generated image",
    "flux-klein-zero":  "Zero-shot generation by FLUX.2 Klein",
    "flux-klein-cot":   "Chain-of-thought generation by FLUX.2 Klein",
    "stable-core-zero": "Zero-shot generation by Stability AI Stable Image Core v1.1",
    "stable-core-cot":  "Chain-of-thought generation by Stability AI Stable Image Core v1.1",
    "gpt-image-zero":   "Zero-shot generation by OpenAI GPT Image 1.5 (quality=medium)",
    "gpt-image-cot":    "Chain-of-thought generation by OpenAI GPT Image 1.5 (quality=medium)",
}

GENERATOR_PATHS: dict[str, tuple[str, str]] = {
    "human": (
        "multimodal_analogy_generation/dataset_images",
        "multimodal_analogy_generation/zero_shot_exploration/results/human",
    ),
    "flux-klein-zero": (
        "multimodal_analogy_generation/zero_shot_exploration/images/zero_shot/flux-klein",
        "multimodal_analogy_generation/zero_shot_exploration/results/zero_shot/flux-klein",
    ),
    "flux-klein-cot": (
        "multimodal_analogy_generation/zero_shot_exploration/images/cot/flux-klein",
        "multimodal_analogy_generation/zero_shot_exploration/results/cot/flux-klein",
    ),
    "stable-core-zero": (
        "multimodal_analogy_generation/zero_shot_exploration/images/zero_shot/stable-core",
        "multimodal_analogy_generation/zero_shot_exploration/results/zero_shot/stable-core",
    ),
    "stable-core-cot": (
        "multimodal_analogy_generation/zero_shot_exploration/images/cot/stable-core",
        "multimodal_analogy_generation/zero_shot_exploration/results/cot/stable-core",
    ),
    "gpt-image-zero": (
        "multimodal_analogy_generation/zero_shot_exploration/images/zero_shot/gpt-image",
        "multimodal_analogy_generation/zero_shot_exploration/results/zero_shot/gpt-image",
    ),
    "gpt-image-cot": (
        "multimodal_analogy_generation/zero_shot_exploration/images/cot/gpt-image",
        "multimodal_analogy_generation/zero_shot_exploration/results/cot/gpt-image",
    ),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "anthropic/claude-sonnet-4-6"
PULLBACK_THRESHOLD = 0.40

# Closed vocabulary for relations
RELATION_VOCAB = [
    "causes", "results_in", "leads_to", "triggers", "produces",
    "enables", "prevents", "blocks", "relies_on",
    "desires", "wants_to_avoid", "faces", "pursues",
    "reveals", "deceives", "uses", "adjusts_to", "provides", "receives",
    "envelops", "contains", "is_part_of", "depends_on", "transforms_into",
]

# Closed vocabulary for semantic roles.
ROLE_VOCAB = [
    "agent",
    "patient",
    "instrument",
    "recipient",
    "theme",
    "force",
    "location",
    "medium",
    "goal",
    "source",
    "outcome",
    "attribute",
]


# ---------------------------------------------------------------------------
# Output filename helpers
# ---------------------------------------------------------------------------

def _filenames_for(threshold: float) -> tuple[str, str, str]:
    """Build threshold-tagged output filenames."""
    suffix = f"_{threshold:.2f}"
    return (
        f"results{suffix}.json",
        f"pullback_scores{suffix}.csv",
        f"_errors{suffix}.json",
    )


# ---------------------------------------------------------------------------
# Stage 1 — Domain identification
# ---------------------------------------------------------------------------

class IdentifyDomains(dspy.Signature):
    """STAGE 1 of dual-domain graph extraction from a visual metaphor.

    YOU ARE GIVEN: a single image that depicts a visual metaphor. A visual
    metaphor combines two distinct conceptual domains:
      - the TARGET domain (the abstract concept the metaphor is *about*) and
      - the SOURCE domain (the concrete concept the target is being *compared to*).

    For example, an image of a single man wreathed in fog is a visual metaphor where
    the TARGET is "loneliness" (abstract, what the image is *about*) and the
    SOURCE is "fog" (concrete, what loneliness is being *compared to*).

    YOUR TASK: identify the two domains depicted.

    YOUR OUTPUT WILL BE USED BY: a downstream pipeline that extracts a
    graph of nodes (entities) and edges (relations) for each domain
    independently, then computes a categorical pullback between the two
    graphs to score the structural strength of the metaphor. So the domain
    names you return must be:
      - single canonical noun phrases, lowercase, no articles
      - abstract enough to admit multiple instantiations (prefer
        "loneliness" over "the lonely old man's feelings"; prefer "fog"
        over "the rolling grey morning fog"); however, if the domain clearly
        refers to a specific entity (e.g. "Eiffel Tower" or "Mercedes Benz"), return that as
        the domain name.
      - no longer than 2-3 words (e.g. "social isolation" is fine, but
        "the feeling of being socially isolated in a crowd" is not)
      - distinct from each other (target != source)

    If the image does not clearly depict a visual metaphor (e.g. it is a
    literal photograph with no second domain implied), still return your
    best guess for what the dominant depicted concept is as the target,
    and "none" for the source. Be critical here: if you cannot clearly identify the two
    domains that the metaphor is trying to depict, return "none" for the source
    domain. The downstream pipeline will detect this.
    """
    image: dspy.Image = dspy.InputField(desc="The visual metaphor image")
    scratchpad: str = dspy.OutputField(
        desc="Brief reasoning: what is literally depicted, what is metaphorically suggested, what is target vs source"
    )
    target_domain: str = dspy.OutputField(
        desc="The TARGET domain — single noun phrase, lowercase, no articles (or 'none')"
    )
    source_domain: str = dspy.OutputField(
        desc="The SOURCE domain — single noun phrase, lowercase, no articles (or 'none')"
    )


# ---------------------------------------------------------------------------
# Stage 2 — Node extraction with role typing
# ---------------------------------------------------------------------------

class ExtractNodes(dspy.Signature):
    """STAGE 2 of dual-domain graph extraction from a visual metaphor.

    YOU ARE GIVEN:
      - the same image as Stage 1
      - the target domain and source domain identified in Stage 1

    YOUR TASK: list the key entities (nodes) for each domain. Each node has
    two parts:
      1. A `name`: a concrete, image-grounded label for what this entity
         actually is in the depicted scene or in the abstract domain.
         Names should be specific to the domain and faithful to what is
         visible or genuinely implied by the image. Examples:
           - source domain "construction site": construction_worker,
             hammer, nail, concrete_block, steel_beam
           - target domain "creative work": writer, idea, manuscript,
             struggle, vision
         Names may be multi-word (snake_case). They should NOT be
         abstract role labels like "agent" or "instrument" — that's what
         the `role` field is for.
      2. A `role`: a single label from this closed vocabulary, describing
         the structural function the entity plays in its own domain:
           {ROLE_VOCAB}
         Pick the closest fit. Do not invent new role labels.

    CRITICAL: role labels must describe what the entity does in ITS OWN
    DOMAIN. Do NOT choose role labels to make the two domains' role lists
    look the same. If only the target domain has a clear `agent` and the
    source domain doesn't, that's fine — the asymmetry is informative.
    Forcing parallel role structures across domains defeats the purpose of
    the downstream structural-alignment scoring. Treat the two domains as
    independent role-typing exercises.

    YOUR OUTPUT WILL BE USED BY: Stage 3 will see only the `name` fields
    (not the roles) when extracting relations, so that its reasoning is
    grounded in concrete image content rather than abstract structure.
    Stage 4 will use the `role` fields to compute structural similarity
    between edges across domains.

    NODE RULES:
      - 4–7 nodes per domain.
      - Names: concrete, image-grounded, snake_case, multi-word allowed.
        Be specific. Prefer "construction_worker" over "agent_figure";
        prefer "rusty_nail" over "metal_object".
      - Roles: from the closed vocabulary above, one per node.
      - Names should be distinct within a domain. Roles MAY repeat within
        a domain (two `theme` nodes is fine if both genuinely play that
        structural part).

    If a domain was identified as "none" in Stage 1, return an empty list
    for that domain.
    """
    image: dspy.Image = dspy.InputField(desc="The visual metaphor image")
    target_domain: str = dspy.InputField(desc="Target domain from Stage 1")
    source_domain: str = dspy.InputField(desc="Source domain from Stage 1")
    scratchpad: str = dspy.OutputField(
        desc="Per-domain reasoning. For each domain, list the salient entities "
             "and explain why each role label fits THAT entity in THAT domain. "
             "Do not discuss cross-domain mapping here."
    )
    nodes_json: str = dspy.OutputField(
        desc='JSON: {"target": [{"name": "...", "role": "..."}, ...], '
             '"source": [{"name": "...", "role": "..."}, ...]}'
    )

ExtractNodes.__doc__ = ExtractNodes.__doc__.replace(
    "{ROLE_VOCAB}", ", ".join(ROLE_VOCAB)
)


# ---------------------------------------------------------------------------
# Stage 3 — Relation extraction (concrete names only)
# ---------------------------------------------------------------------------

class ExtractRelations(dspy.Signature):
    """STAGE 3 of dual-domain graph extraction from a visual metaphor.

    YOU ARE GIVEN:
      - the same image as Stage 1 and Stage 2
      - the target and source domain names from Stage 1
      - the per-domain node names from Stage 2 (CONCRETE names only,
        with role labels intentionally hidden)

    YOUR TASK: for each domain, list the directed relations between its
    nodes. A relation is a triple (head, predicate, tail) where:
      - head and tail are concrete node names from the Stage 2 list
        for that domain
      - predicate captures a structural relationship that holds between
        the nodes within that domain

    YOUR OUTPUT WILL BE USED BY: The categorical pullback algorithm,
    which compares the two graphs by finding a structure-preserving
    correspondence (target_nodes ↔ source_nodes) such that target_edges
    align with source_edges.

    Important: when you write a relation, focus on what is actually
    happening in the depicted image (for the source domain) or what
    genuinely holds in the abstract concept (for the target domain). Do
    NOT try to make the two domains' relations mirror each other — the
    pullback step will discover any genuine parallels on its own.

    RELATION RULES:
      - Closed vocabulary ONLY: {RELATION_VOCAB}. Pick the closest match
        if nothing is exact; do not invent new predicates.
      - 3–7 edges per domain. More than 7 risks adding unsupported edges.
      - Both head and tail MUST be exact matches to names in the node
        list for that domain.
      - Multiple edges between the same pair are allowed (different
        predicates).
      - Do NOT add edges between target nodes and source nodes — each
        graph is intra-domain only. Cross-domain mapping happens in the
        pullback step, not here.
      - Assess each domain independently based on the image and your
        reasoning, and define the relations between the nodes that are
        most salient for each independent domain. Do not coordinate
        across domains.

    If a domain has an empty node list, return an empty edge list for it.
    """

    image: dspy.Image = dspy.InputField(desc="The visual metaphor image")
    target_domain: str = dspy.InputField(desc="Target domain from Stage 1")
    source_domain: str = dspy.InputField(desc="Source domain from Stage 1")
    node_names_json: str = dspy.InputField(
        desc='Per-domain node NAMES only (no roles): '
             '{"target": ["...", ...], "source": ["...", ...]}'
    )
    relations_json: str = dspy.OutputField(
        desc='JSON: {"target": [[h,r,t],...], "source": [[h,r,t],...]}'
    )

ExtractRelations.__doc__ = ExtractRelations.__doc__.replace(
    "{RELATION_VOCAB}", ", ".join(RELATION_VOCAB)
)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

predict_domains   = dspy.Predict(IdentifyDomains)
predict_nodes     = dspy.Predict(ExtractNodes)
predict_relations = dspy.Predict(ExtractRelations)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json(raw: str, default):
    if not raw:
        return default
    try:
        return json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return default


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _normalise_nodes(domain_nodes) -> list[dict]:
    out = []
    for n in domain_nodes or []:
        if not isinstance(n, dict):
            continue
        name = str(n.get("name", "")).strip()
        role = str(n.get("role", "")).strip().lower()
        if name:
            out.append({"name": name, "role": role})
    return out


def _role_lookup(nodes: list[dict]) -> dict[str, str]:
    return {n["name"]: n["role"] for n in nodes}


def _role_substituted_edges(
    edges: list[list[str]], role_lookup: dict[str, str]
) -> list[list[str]]:
    result = []
    for e in edges:
        if len(e) != 3:
            continue
        h, r, t = e
        result.append([
            role_lookup.get(h, "theme"),
            r,
            role_lookup.get(t, "theme"),
        ])
    return result


def compress_image_to_data_uri(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    if (
        image_path.stat().st_size <= 3_500_000
        and ext in {"jpeg", "png", "webp", "gif"}
    ):
        data = image_path.read_bytes()
        return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"

    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")

        max_side = 2048
        quality = 90
        best: bytes | None = None
        for _ in range(12):
            buf = im.copy()
            buf.thumbnail((max_side, max_side), Image.LANCZOS)
            out = io.BytesIO()
            buf.save(out, format="JPEG", quality=quality, optimize=True)
            data = out.getvalue()
            if len(data) <= 3_500_000:
                return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            if best is None or len(data) < len(best):
                best = data
            max_side = max(256, int(max_side * 0.75))
            quality = max(40, quality - 10)

    raise RuntimeError(f"Could not compress {image_path.name}")


# ---------------------------------------------------------------------------
# Empty pullback when a metaphor is not detected
# ---------------------------------------------------------------------------

_EMPTY_PULLBACK = {
    "matched_pairs": [], "score": 0.0, "chain_length": 0,
    "num_matched": 0, "total_similarity": 0.0, "coverage": 0.0,
}


# ---------------------------------------------------------------------------
# Single-run pipeline
# ---------------------------------------------------------------------------

def extract_metaphor_graphs(
    image_path: str | Path,
    threshold: float = PULLBACK_THRESHOLD,
) -> dict:
    """Execute the four-stage pipeline on a single image and return a result dict."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = dspy.Image(url=compress_image_to_data_uri(image_path))

    result: dict = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "stages": {},
    }

    # Stage 1
    s1 = predict_domains(image=img)
    result["stages"]["1_domain_identification"] = {
        "scratchpad":    s1.scratchpad,
        "target_domain": s1.target_domain.strip().lower(),
        "source_domain": s1.source_domain.strip().lower(),
    }

    if s1.source_domain.strip().lower() == "none":
        result["status"]        = "no_metaphor_detected"
        result["target_domain"] = s1.target_domain.strip().lower()
        result["source_domain"] = "none"
        result["nodes_target"]  = []
        result["nodes_source"]  = []
        result["edges_target"]  = []
        result["edges_source"]  = []
        result["pullback"]      = dict(_EMPTY_PULLBACK)
        result["pullback_score"] = 0.0
        return result

    # Stage 2
    s2 = predict_nodes(
        image=img,
        target_domain=s1.target_domain,
        source_domain=s1.source_domain,
    )
    nodes_raw = _parse_json(s2.nodes_json, default={"target": [], "source": []})
    nodes = {
        "target": _normalise_nodes(nodes_raw.get("target", [])),
        "source": _normalise_nodes(nodes_raw.get("source", [])),
    }
    result["stages"]["2_node_extraction"] = {
        "scratchpad":    s2.scratchpad,
        "nodes_json_raw": s2.nodes_json,
        "nodes":          nodes,
    }

    # Stage 3
    node_names = {
        "target": [n["name"] for n in nodes["target"]],
        "source": [n["name"] for n in nodes["source"]],
    }
    s3 = predict_relations(
        image=img,
        target_domain=s1.target_domain,
        source_domain=s1.source_domain,
        node_names_json=json.dumps(node_names),
    )
    relations = _parse_json(s3.relations_json, default={"target": [], "source": []})
    result["stages"]["3_relation_extraction"] = {
        "relations_json_raw": s3.relations_json,
        "relations":          relations,
    }

    result["status"]        = "ok"
    result["target_domain"] = s1.target_domain.strip().lower()
    result["source_domain"] = s1.source_domain.strip().lower()
    result["nodes_target"]  = nodes["target"]
    result["nodes_source"]  = nodes["source"]
    result["edges_target"]  = relations.get("target", [])
    result["edges_source"]  = relations.get("source", [])

    # Stage 4
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

    result["stages"]["4_pullback"] = {
        "threshold":    threshold,
        "score":        pullback["score"],
        "num_matched":  pullback["num_matched"],
        "coverage":     pullback["coverage"],
        "chain_length": pullback["chain_length"],
    }
    result["pullback"]       = pullback
    result["pullback_score"] = pullback["score"]
    return result


# ---------------------------------------------------------------------------
# Consolidated results I/O
# ---------------------------------------------------------------------------

def _load_existing_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    try:
        data = json.loads(results_path.read_text())
    except Exception as e:
        print(f"Warning: could not parse {results_path}: {e}. Starting fresh.")
        return []
    if isinstance(data, list):
        return data
    print(f"Warning: unexpected JSON shape in {results_path}. Starting fresh.")
    return []


def _write_results_json(results_path: Path, results: list[dict]) -> None:
    ordered = sorted(results, key=lambda r: r.get("image_name", ""))
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp.write_text(json.dumps(ordered, indent=2))
    tmp.replace(results_path)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def find_images(dataset_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


_print_lock = threading.Lock()


def _process_one(
    image_path: Path, threshold: float
) -> tuple[str, Optional[dict], Optional[str]]:
    try:
        result = extract_metaphor_graphs(image_path, threshold=threshold)
        return image_path.name, result, None
    except Exception as e:
        return image_path.name, None, f"{type(e).__name__}: {e}"


def _write_summary_csv(csv_path: Path, results: list[dict]) -> None:
    rows = sorted(results, key=lambda r: r.get("image_name", ""))
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "pullback_score",
            "target_domain",
            "source_domain",
        ])
        for r in rows:
            image_id = Path(r.get("image_name", "")).stem
            writer.writerow([
                image_id,
                r.get("pullback_score", ""),
                r.get("target_domain", ""),
                r.get("source_domain", ""),
            ])


def run_batch(
    dataset_dir: Path,
    output_dir: Path,
    threshold: float,
    limit: Optional[int] = None,
    workers: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_name, csv_name, errors_name = _filenames_for(threshold)
    results_path = output_dir / results_name
    csv_path     = output_dir / csv_name
    errors_path  = output_dir / errors_name

    images = find_images(dataset_dir)
    if limit:
        images = images[:limit]
    if not images:
        print(f"No images found in {dataset_dir}")
        return

    results: list[dict] = _load_existing_results(results_path)
    processed_names: set[str] = {r.get("image_name", "") for r in results}
    pending = [p for p in images if p.name not in processed_names]

    if processed_names:
        print(
            f"Resuming: {len(processed_names)} images already in "
            f"{results_path.name}, {len(pending)} to do."
        )
    else:
        print(f"Processing {len(pending)} images — {workers} workers")

    if not pending:
        _write_summary_csv(csv_path, results)
        print(f"CSV summary: {csv_path}")
        return

    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_one, img, threshold): img
            for img in pending
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            name, result, err = fut.result()
            if err:
                with _print_lock:
                    tqdm.write(f"  ERROR {name}: {err}")
                errors.append((name, err))
                continue
            if result is None:
                continue
            results.append(result)
            _write_results_json(results_path, results)

    print(f"\nDone. {len(results)} total results, {len(errors)} failed.")
    print(f"Results:     {results_path}")

    _write_summary_csv(csv_path, results)
    print(f"CSV summary: {csv_path}")

    if errors:
        errors_path.write_text(json.dumps(errors, indent=2))
        print(f"Error log:   {errors_path}")


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

def _configure_lm(model: str, api_base: Optional[str] = None) -> None:
    kwargs: dict = {"model": model, "max_tokens": 4096}

    if model.startswith("bedrock/"):
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            raise RuntimeError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. Add it to your .env file."
            )
    elif model.startswith("anthropic/"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    elif model.startswith("openrouter/"):
        kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")
    else:
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

    if api_base:
        kwargs["api_base"] = api_base
    dspy.configure(lm=dspy.LM(**kwargs))
    print(f"Model: {model}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator", required=True, choices=list(GENERATORS.keys()),
        help="Visual metaphor generation mode (selects input and output dirs)",
    )
    parser.add_argument("--threshold", type=float, default=PULLBACK_THRESHOLD,
                        help=f"Cosine-similarity threshold for the pullback "
                             f"(default: {PULLBACK_THRESHOLD})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers for batch mode")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM (default: {DEFAULT_MODEL})")
    parser.add_argument("--api_base", default=None,
                        help="Optional API base URL")
    args = parser.parse_args()

    dataset_dir, output_dir = GENERATOR_PATHS[args.generator]
    print(f"Generator: {args.generator} — {GENERATORS[args.generator]}")
    print(f"  Input:     {dataset_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  Threshold: {args.threshold:.2f}")

    _configure_lm(args.model, args.api_base)

    run_batch(
        Path(dataset_dir),
        Path(output_dir),
        threshold=args.threshold,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()