"""
Uses GPT-5.4 as a DSPy-based LLM judge to evaluate full analogy mappings and compares
the judge's verdict against the deterministic exact-match check.
A mapping is only correct if ALL pairs are right (system accuracy).

Requires the SCAR dataset and the results file produced by benchmark-gpt.py
(WITH_CONCEPTS=True) to already exist at the paths below.

Run (from repo root):
    python scripts/judge_eval.py
"""

import json
import re
import dspy
from dotenv import load_dotenv
import os

load_dotenv()

lm = dspy.LM(
    "openai/gpt-5.4",
    api_key=os.getenv("OPENAI_API_KEY"),
    reasoning_effort="high",
    max_tokens=16190,
)
dspy.configure(lm=lm)

DATASET_FILE = "benchmarking/scar-baseline/scar_dataset.jsonl"
RESULTS_FILE = "benchmarking/scar-baseline/gpt_results_with_concepts.json"


class AnalogicalMappingJudge(dspy.Signature):
    """You are an expert judge evaluating analogical reasoning mappings, grounded in
    Structure Mapping Theory (SMT).

    You are given two systems with their background descriptions, the full list of
    concepts from each system, and a model's predicted 1-to-1 mapping between them.

    The task is a strict 1-to-1 assignment: every concept from System A must be paired
    with exactly one concept from System B and vice versa. There is exactly one correct
    complete mapping, established by human annotators with domain knowledge.

    **Your task**: assess whether the predicted mapping is the correct complete assignment.
    Evaluate the mapping GLOBALLY, not pair by pair in isolation.

    Key principles:

    1. Abstract correspondences are valid. A pair that seems non-obvious in isolation
       may be the only structurally valid option once all other concepts are assigned.
       Do not reject a pair just because it requires abstract reasoning to justify.

    2. To compare two orderings of a cluster: for each concept in System A, ask which
       concept in System B captures its role most precisely — not just adequately.
       The correct ordering will have a tighter justification for every pair; the wrong
       ordering will require looser or more forced reasoning for at least one pair.

    3. Accept the mapping only if you are confident it is the best complete assignment.
    """

    system_a: str = dspy.InputField(desc="Name of System A")
    system_a_background: str = dspy.InputField(desc="Background description of System A")
    system_b: str = dspy.InputField(desc="Name of System B")
    system_b_background: str = dspy.InputField(desc="Background description of System B")
    concepts_a: str = dspy.InputField(desc="Comma-separated list of all concepts from System A")
    concepts_b: str = dspy.InputField(desc="Comma-separated list of all concepts from System B")
    predicted_mappings: str = dspy.InputField(desc="The model's predicted mapping, formatted as (concept_a, concept_b), ...")

    rationale: str = dspy.OutputField(desc="Explanation of which pairs are correct or incorrect and why")
    fulfilled: str = dspy.OutputField(desc="'yes' if every predicted pair is correct, 'no' if any pair is wrong")


judge = dspy.Predict(AnalogicalMappingJudge)


def extract_pairs(text: str) -> list[tuple[str, str]]:
    pattern = r"\(([^()]+),\s*([^()]+)\)"
    matches = re.findall(pattern, text)
    return [(a.strip().lower(), b.strip().lower()) for a, b in matches if a.strip() and b.strip()]


def load_samples(n: int = 10) -> list[dict]:
    with open(DATASET_FILE) as f:
        dataset = {json.loads(l)["id"]: json.loads(l) for l in f}

    with open(RESULTS_FILE) as f:
        results = {e["id"]: e["response"] for e in json.load(f) if "response" in e}

    perfect, imperfect = [], []

    for aid, response in results.items():
        gt = dataset[aid]
        gt_pairs = {(p[0].strip().lower(), p[1].strip().lower()) for p in gt["mappings"]}
        predicted_list = extract_pairs(response)
        predicted_set = set(predicted_list)
        det_correct = gt_pairs == predicted_set

        sample = {
            "id": aid,
            "system_a": gt["system_a"],
            "system_b": gt["system_b"],
            "system_a_background": gt["system_a_background"],
            "system_b_background": gt["system_b_background"],
            "concepts_a": [p[0] for p in gt["mappings"]],
            "concepts_b": [p[1] for p in gt["mappings"]],
            "gt_pairs": sorted(gt_pairs),
            "predicted_pairs": predicted_list,
            "det_correct": det_correct,
        }

        if det_correct and len(perfect) < n // 2:
            perfect.append(sample)
        elif not det_correct and len(imperfect) < n // 2:
            imperfect.append(sample)

        if len(perfect) >= n // 2 and len(imperfect) >= n // 2:
            break

    return perfect + imperfect


def main():
    samples = load_samples(n=16)
    print(f"Loaded {len(samples)} samples ({len(samples)//2} perfect, {len(samples)//2} imperfect by deterministic check)\n")
    print("=" * 100)

    agreements = 0
    for i, sample in enumerate(samples):
        predicted_str = ", ".join(f"({a}, {b})" for a, b in sample["predicted_pairs"])

        result = judge(
            system_a=sample["system_a"],
            system_a_background=sample["system_a_background"],
            system_b=sample["system_b"],
            system_b_background=sample["system_b_background"],
            concepts_a=", ".join(sample["concepts_a"]),
            concepts_b=", ".join(sample["concepts_b"]),
            predicted_mappings=predicted_str,
        )

        judge_verdict = result.fulfilled.strip().lower() == "yes"
        agrees = judge_verdict == sample["det_correct"]
        agreements += agrees

        print(f"\nSample {i+1}/{len(samples)} | ID={sample['id']} | {sample['system_a']} <-> {sample['system_b']}")
        print(f"  GT pairs        : {sample['gt_pairs']}")
        print(f"  Predicted pairs : {sample['predicted_pairs']}")
        print(f"  Deterministic   : {'CORRECT' if sample['det_correct'] else 'WRONG'}")
        print(f"  Judge verdict   : {'CORRECT' if judge_verdict else 'WRONG'} (fulfilled={result.fulfilled.strip()})")
        print(f"  Rationale       : {result.rationale}")
        print(f"  Agreement       : {'✓ AGREE' if agrees else '✗ DISAGREE'}")
        print("-" * 100)

    print(f"\nSummary: Judge agreed with deterministic check on {agreements}/{len(samples)} samples")


if __name__ == "__main__":
    main()
