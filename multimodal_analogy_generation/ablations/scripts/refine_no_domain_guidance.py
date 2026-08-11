"""
refine_no_domain_guidance.py - "no domain guidance, no metaphor
text" ablation.

ABLATION UNDER TEST: withhold the gold-standard domain pair / concept-match block from the refiner), 
also withhold the intended verbal metaphor itself, so the refiner sees nothing
but what extraction + the pullback observed about the previous image: what
it was extracted as depicting, the pullback score/coverage, and which
relations were missing.
"""

from dataclasses import dataclass

from multimodal_analogy_generation.iterative_refinement.scripts.refine import (
    RoundDiagnostic,
    build_diagnostic,
    DOMAIN_MATCH_THRESHOLD,
)


@dataclass
class NoDomainGuidanceNoMetaphorDiagnostic(RoundDiagnostic):
    """Identical fields/computation to RoundDiagnostic, only to_feedback_text() is overridden."""

    def to_feedback_text(self) -> str:
        lines: list[str] = []

        ext_str = ", ".join(
            d for d in self.extracted_domains if d and d != "none"
        ) or "(none recovered)"
        lines.append(f"What the previous image actually conveyed: {ext_str}.")

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


def build_no_domain_guidance_no_metaphor_diagnostic(
    *,
    metaphor: str,
    extraction_result: dict,
    gold_domains: list[str],
    embed=None,
    domain_threshold: float = DOMAIN_MATCH_THRESHOLD,
) -> NoDomainGuidanceNoMetaphorDiagnostic:
    full = build_diagnostic(
        metaphor=metaphor,
        extraction_result=extraction_result,
        gold_domains=gold_domains,
        embed=embed,
        domain_threshold=domain_threshold,
    )
    # Re-tag as NoDomainGuidanceNoMetaphorDiagnostic without recomputing
    # anything -- same fields, just a different to_feedback_text().
    return NoDomainGuidanceNoMetaphorDiagnostic(**full.__dict__)
