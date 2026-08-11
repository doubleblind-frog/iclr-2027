"""
refine.py

Refinement stage of the iterative in-context loop for visual metaphor
generation. Per-round feedback to the refiner LM is grounded in a CLIP
image-text alignment score rather than human-annotated gold-standard
domains, so the loop needs no human annotation to run on new metaphors.

Library module only, no CLI of its own — invoked via refinement_loop.py:
    python multimodal_analogy_generation/iterative_refinement/scripts/refinement_loop.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

from multimodal_analogy_generation.zero_shot_exploration.scripts.clip_fidelity import (
    ClipScorer,
    TEXT_TEMPLATE,
    pick_device,
)

REFINER_MODEL_ID = "claude-sonnet-4-6"
REFINER_TEMPERATURE = 0.7
REFINER_MAX_TOKENS = 200
MAX_RETRIES = 4

MAX_WORDS = 55

DOMAIN_MATCH_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# CLIP scorer (single image/text pair, no retrieval)
# ---------------------------------------------------------------------------

_scorer: Optional[ClipScorer] = None


def get_clip_scorer() -> ClipScorer:
    """Lazily instantiate and cache a single ClipScorer for the process.

    Loading the model is the expensive part, so this is built once and
    reused across every round/image in the refinement loop, rather than
    re-instantiated per call.
    """
    global _scorer
    if _scorer is None:
        _scorer = ClipScorer(pick_device())
    return _scorer


def compute_clip_score(
    image_path: str | Path,
    metaphor: str,
    scorer: Optional[ClipScorer] = None,
) -> float:
    """Cosine similarity between one image and one metaphor's text template.

    Same model, preprocessing, and text template as clip_fidelity.py, but
    scores a single image/text pair directly rather than computing
    retrieval metrics over a corpus.
    """
    scorer = scorer or get_clip_scorer()
    image_feat = scorer.encode_images([Path(image_path)])   # [1, D]
    text_feat = scorer.encode_texts([TEXT_TEMPLATE(metaphor)])  # [1, D]
    return float((image_feat * text_feat).sum(dim=-1).item())


# ---------------------------------------------------------------------------
# Structured diagnostic
# ---------------------------------------------------------------------------

@dataclass
class RoundDiagnostic:
    """Everything the refiner is told about one completed round."""
    metaphor: str
    pullback_score: float
    # gold (from human annotation) — kept for logging/evaluation comparability
    # (concept_matches, coverage, etc.); NOT surfaced in to_feedback_text(),
    # which uses clip_score instead so the loop needs no human annotation to run.
    gold_domains: list = field(default_factory=list)
    extracted_domains: list = field(default_factory=list)
    concept_matches: list = field(default_factory=list)
    domain_feedback_available: bool = False
    clip_score: Optional[float] = None
    # structural detail from the pullback result
    num_target_edges: int = 0
    num_matched_edges: int = 0
    coverage: float = 0.0
    matched_relations: list = field(default_factory=list)
    unmatched_target_edges: list = field(default_factory=list)
    extracted_edges_target: list = field(default_factory=list)
    extracted_edges_source: list = field(default_factory=list)

    @property
    def target_domain_match(self) -> Optional[bool]:
        return self.concept_matches[0]["match"] if self.concept_matches else None

    @property
    def source_domain_match(self) -> Optional[bool]:
        return self.concept_matches[1]["match"] if len(self.concept_matches) > 1 else None

    def to_feedback_text(self) -> str:
        """Render the diagnostic as the natural-language block shown to the LM."""
        lines: list[str] = []

        lines.append(f'The intended metaphor is: "{self.metaphor}"')
        ext_str = ", ".join(d for d in self.extracted_domains if d and d != "none") or "(none recovered)"
        lines.append(f"What the previous image actually conveyed: {ext_str}.")

        if self.clip_score is not None:
            lines.append(
                f"CLIP image-text alignment: {self.clip_score:.3f} "
                "(cosine similarity between the image and the intended "
                "metaphor description; higher means the image more strongly "
                "evokes the intended metaphor overall)."
            )

        # Structural signal
        lines.append(
            f"Structural alignment: pullback score {self.pullback_score:.3f}; "
            f"{self.num_matched_edges} of {self.num_target_edges} relations from "
            f"one domain found a structural correspondent in the other "
            f"(coverage {self.coverage:.2f})."
        )
        if self.unmatched_target_edges:
            rels = "; ".join(
                f"{h} {r} {t}" for h, r, t in
                (e for e in self.unmatched_target_edges if len(e) == 3)
            )
            if rels:
                lines.append(
                    "Relationships that were NOT conveyed by the image: "
                    f"{rels}. Make these relationships visible between the "
                    "corresponding elements of the two domains."
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain matching
# ---------------------------------------------------------------------------

def match_domains_bipartite(
    embed,
    gold_domains: list[str],
    extracted_domains: list[str],
    threshold: float = DOMAIN_MATCH_THRESHOLD,
) -> tuple[list[dict], bool]:
    """Order-agnostically assign each gold concept to its best extracted concept.
    """
    golds = [g.strip() for g in gold_domains if g and g.strip()]
    exts = [e.strip() for e in extracted_domains if e and e.strip() and e.strip() != "none"]

    if embed is None or not golds or not exts:
        return ([{"gold": g, "matched_to": None, "cos": None, "match": None}
                 for g in golds], False)

    import numpy as np
    vecs = embed(golds + exts)
    gv = vecs[: len(golds)]
    ev = vecs[len(golds):]
    sim = gv @ ev.T

    n_g, n_e = sim.shape
    assigned_ext: set[int] = set()
    matches: list[dict] = [None] * n_g
    order = sorted(
        ((sim[i, j], i, j) for i in range(n_g) for j in range(n_e)),
        reverse=True,
    )
    assigned_gold: set[int] = set()
    for cos, i, j in order:
        if i in assigned_gold or j in assigned_ext:
            continue
        assigned_gold.add(i)
        assigned_ext.add(j)
        matches[i] = {
            "gold": golds[i],
            "matched_to": exts[j],
            "cos": float(cos),
            "match": bool(cos >= threshold),
        }
    for i in range(n_g):
        if matches[i] is None:
            matches[i] = {"gold": golds[i], "matched_to": None, "cos": None, "match": False}

    return matches, True


def build_diagnostic(
    *,
    metaphor: str,
    extraction_result: dict,
    gold_domains: list[str],
    embed=None,
    domain_threshold: float = DOMAIN_MATCH_THRESHOLD,
    image_path: Optional[str | Path] = None,
    scorer: Optional[ClipScorer] = None,
) -> RoundDiagnostic:
    """Assemble a RoundDiagnostic from one extract.py result dict + gold domains.

    gold_domains is still used to compute concept_matches (for scores.csv
    logging / evaluation), but the CLIP score — not the gold-domain match —
    is what to_feedback_text() shows the refiner. image_path defaults to
    extraction_result["image_path"], set by extract_metaphor_graphs() for
    every round, so callers don't need to pass it explicitly.
    """
    resolved_image_path = image_path or extraction_result.get("image_path")
    if not resolved_image_path:
        raise ValueError(
            "build_diagnostic: no image_path given and extraction_result has "
            "no 'image_path' key to fall back on."
        )
    clip_score = compute_clip_score(resolved_image_path, metaphor, scorer=scorer)

    pullback = extraction_result.get("pullback", {}) or {}
    extracted_target = extraction_result.get("target_domain", "none")
    extracted_source = extraction_result.get("source_domain", "none")
    extracted_domains = [extracted_target, extracted_source]

    concept_matches, domain_available = match_domains_bipartite(
        embed, gold_domains, extracted_domains, domain_threshold
    )

    edges_target = [e for e in extraction_result.get("edges_target", []) if len(e) == 3]
    edges_source = [e for e in extraction_result.get("edges_source", []) if len(e) == 3]

    matched_pairs = pullback.get("matched_pairs", []) or []
    matched_concrete_target = []
    matched_display = []
    for m in matched_pairs:
        ce_t = m.get("source_edge_concrete")
        ce_s = m.get("option_edge_concrete")
        if ce_t:
            matched_concrete_target.append(tuple(ce_t))
        if ce_t and ce_s:
            matched_display.append((ce_t, ce_s))

    unmatched = [e for e in edges_target if tuple(e) not in set(matched_concrete_target)]

    return RoundDiagnostic(
        metaphor=metaphor,
        pullback_score=float(extraction_result.get("pullback_score", 0.0)),
        gold_domains=list(gold_domains),
        extracted_domains=extracted_domains,
        concept_matches=concept_matches,
        domain_feedback_available=domain_available,
        clip_score=clip_score,
        num_target_edges=len(edges_target),
        num_matched_edges=int(pullback.get("num_matched", 0)),
        coverage=float(pullback.get("coverage", 0.0)),
        matched_relations=matched_display,
        unmatched_target_edges=unmatched,
        extracted_edges_target=edges_target,
        extracted_edges_source=edges_source,
    )


# ---------------------------------------------------------------------------
# Refinement prompt
# ---------------------------------------------------------------------------

REFINE_PROMPT_TEMPLATE = (
    "You are refining the prompt given to a text-to-image model so that the "
    "image it produces is a STRONGER visual metaphor for a given idea.\n\n"
    "A visual metaphor depicts an abstract TARGET concept by portraying it "
    "through a concrete SOURCE concept, such that the relationships among the "
    "source elements mirror the relationships in the target idea.\n\n"
    "Here is the previous prompt you wrote and a diagnosis of how the image it "
    "produced fell short:\n\n"
    "PREVIOUS PROMPT:\n{previous_elaboration}\n\n"
    "DIAGNOSIS:\n{feedback}\n\n"
    "Rewrite the prompt to fix the problems above. Specifically:\n"
    "- Make BOTH the intended source and target concepts unmistakably present "
    "and recognisable in a single scene.\n"
    "- Depict the relationships listed as missing, so the structure of the "
    "metaphor is visible, not just its surface objects.\n"
    "- Keep the parts of the previous prompt that already worked; change only "
    "what the diagnosis flags.\n\n"
    "Strict output rules:\n"
    "- Output ONLY the new visual prompt itself.\n"
    "- No preamble, no analysis, no headers, no markdown, no surrounding "
    "quotes, no commentary.\n"
    f"- Keep it under {MAX_WORDS} words (~75 CLIP tokens).\n"
    "- Write it as a single paragraph in natural prose."
)


def _clean_output(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def refine_elaboration(
    client: anthropic.Anthropic,
    previous_elaboration: str,
    diagnostic: RoundDiagnostic,
) -> str:
    """Call the refiner LM and return a new visual-elaboration prompt."""
    user_msg = REFINE_PROMPT_TEMPLATE.format(
        previous_elaboration=previous_elaboration,
        feedback=diagnostic.to_feedback_text(),
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=REFINER_MODEL_ID,
                max_tokens=REFINER_MAX_TOKENS,
                temperature=REFINER_TEMPERATURE,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            return _clean_output(raw)
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            # transient — back off and retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except anthropic.APIStatusError:
            # non-transient (auth, bad request, etc.) — don't retry
            raise


def make_refiner_client() -> anthropic.Anthropic:
    """Reads the API key from the ANTHROPIC_API_KEY environment variable."""
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))