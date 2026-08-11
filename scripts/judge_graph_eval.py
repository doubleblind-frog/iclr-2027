"""
Uses Claude Opus 4.6 as a DSPy-based LLM judge to evaluate Haiku-generated domain graphs.
Outputs the percentage of valid generated graphs.

Requires benchmarking/scar-baseline/haiku-extended-graphs.jsonl (domain graphs
extracted by Claude Haiku, generated separately) and the SCAR dataset to
already exist.

Run (from repo root):
    python scripts/judge_graph_eval.py
"""

import dspy
import json
import os
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    "anthropic/claude-opus-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=32000,
    temperature=0.0,
)
dspy.configure(lm=lm)


class DomainGraphJudge(dspy.Signature):
    """{PROMPT_TEMPLATE}"""
    full_prompt: str = dspy.InputField()
    domain_name: str = dspy.OutputField()
    fulfilled: str = dspy.OutputField()
    rationale: str = dspy.OutputField()


def load_prompt(path):
    with open(path) as f:
        return f.read()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_scar_backgrounds(scar_path):
    """Build a domain_name → background lookup from the SCAR dataset."""
    backgrounds = {}
    with open(scar_path) as f:
        for line in f:
            entry = json.loads(line)
            backgrounds[entry["system_a"]] = entry["system_a_background"]
            backgrounds[entry["system_b"]] = entry["system_b_background"]
    return backgrounds

def build_prompt(prompt_template, graph, backgrounds):
    background = backgrounds.get(graph["domain_name"], "No background available.")
    return (
        prompt_template
        .replace("[INSERT SYSTEM NAME HERE]", graph["domain_name"])
        .replace("[INSERT DOMAIN GRAPH]", json.dumps(graph, indent=2))
        .replace("[INSERT BACKGROUND HERE]", background)
    )


INPUT_FILE  = "benchmarking/scar-baseline/haiku-extended-graphs.jsonl"
SCAR_FILE   = "benchmarking/scar-baseline/scar_dataset.jsonl"
OUTPUT_FILE = "benchmarking/scar-baseline/haiku-graphs-judged.json"

prompt_template = load_prompt("benchmarking/prompt/judge_domain_graph_prompt.md")
judge = dspy.Predict(DomainGraphJudge)


def main():
    graphs = load_jsonl(INPUT_FILE)
    backgrounds = load_scar_backgrounds(SCAR_FILE)
    print(f"Loaded {len(graphs)} graphs\n" + "=" * 80)

    valid = 0
    results_log = []

    for i, graph in enumerate(graphs):
        try:
            result = judge(full_prompt=build_prompt(prompt_template, graph, backgrounds))
            prediction = result.fulfilled.strip().lower()
            rationale = result.rationale
        except Exception as e:
            prediction = "error"
            rationale = str(e)

        if prediction == "yes":
            valid += 1

        results_log.append({
            "domain_name": graph["domain_name"],
            "prediction": prediction,
            "rationale": rationale,
        })
        print(f"[{i+1}/{len(graphs)}] {graph['domain_name']} → {prediction}")

    pct = valid / len(graphs) * 100 if graphs else 0
    print(f"\n{'='*80}\nValid: {valid}/{len(graphs)} ({pct:.1f}%)")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"valid": valid, "total": len(graphs), "valid_pct": pct, "results": results_log}, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()