"""
Prompts GPT 4.1 Mini to create domain graphs for randomly selected domains from the SCAR dataset,
grounded in a provided background text.

Requires the SCAR dataset at benchmarking/scar-baseline/scar_dataset.jsonl.

Run (from repo root):
    python scripts/generate_analogy_graphs.py
"""

import json
import os
from pathlib import Path
import random

import dspy
from dotenv import load_dotenv

from src.analogy import Analogy

load_dotenv()

lm = dspy.LM(
    model="openai/gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=4096,
)
dspy.configure(lm=lm)

DATASET_FILE = "benchmarking/scar-baseline/scar_dataset.jsonl"
OUTPUT_FILE = "benchmarking/scar-baseline/generated_analogy_graphs.jsonl"


class GenerateAnalogyGraph(dspy.Signature):
    """You are given the name and background description of a domain (e.g. the Solar
    System, the Immune System, a Food Chain). Your task is to use the provided background description to produce a structured
    graph representation of that domain that can be used as an analogy source. You should follow the Structure Mapping Theory 
    (SMT) and consider nodes as entities and edges as the relations between them.

    - **Nodes** are the key concepts or entities within the domain. Only include nodes that can be connected to another node via a meaningful and valid relationship based on the background description.

    - **Edges** represent meaningful relationships between two nodes (e.g. "drives",
      "contains", "produces", "regulates"). These must be abstract, transferable relations.
      Avoid using encyclopedic or domain-specific relations, and instead focus on more general relations 
      that could apply across different domains.

    Start by defining the core concepts of the domain as nodes. Make sure that the concepts that are central to the domain are all included in the graph. 
    Then, go through all of the nodes and consider the relationships between them. 
    For every pair of nodes, determine if there is a meaningful relationship that can be expressed as an edge. If any relation can be expressed between two nodes, add an edge between them with the appropriate relation label. 
    Based on the source and target nodes of an edge, the sentence [source node] [relation] [target node] should be a true statement about the domain according to the provided background description.

    Only include concepts and relationships that are central to the domain's structure.
    """

    domain_name: str = dspy.InputField(desc="The name of the domain")
    background: str = dspy.InputField(desc="Background description of the domain")
    analogy: Analogy = dspy.OutputField(desc="Structured graph representation of the domain")


generate = dspy.Predict(GenerateAnalogyGraph)


def load_existing(output_path: str) -> set[str]:
    """Return set of domain names already written to the output file."""
    seen = set()
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                seen.add(entry["domain_name"])
    return seen


def run(dataset_file: str = DATASET_FILE, output_file: str = OUTPUT_FILE, limit: int | None = None):
    with open(dataset_file) as f:
        entries = [json.loads(l) for l in f]

    random.shuffle(entries)

    already_done = load_existing(output_file)
    processed_count = 0

    print(f"{len(already_done)} domains already done")

    with open(output_file, "a") as out:
        for entry in entries:
            if limit is not None and processed_count >= limit:
                break

            systems = [
                (entry["system_a"], entry["system_a_background"]),
                (entry["system_b"], entry["system_b_background"]),
            ]

            for domain_name, background in systems:
                if domain_name in already_done:
                    continue

                print(f"[{processed_count+1}] {domain_name}")
                try:
                    result = generate(domain_name=domain_name, background=background)
                    record = {
                        "domain_name": domain_name,
                        "analogy": result.analogy.model_dump(),
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()

                    already_done.add(domain_name)
                    processed_count += 1

                    if limit is not None and processed_count >= limit:
                        break

                except Exception as e:
                    print(f"  Error: {e}")
                    record = {"domain_name": domain_name, "error": str(e)}
                    out.write(json.dumps(record) + "\n")
                    out.flush()

            if limit is not None and processed_count >= limit:
                break


if __name__ == "__main__":
    run(limit=50)
