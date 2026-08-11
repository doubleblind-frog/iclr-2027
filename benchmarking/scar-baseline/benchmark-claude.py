"""
Benchmarks the basic SCAR dataset task of concept matching for Claude Opus 4.6.

Requires the SCAR dataset at benchmarking/scar-baseline/scar_dataset.jsonl.
Runs immediately on import (no __main__ guard) and resumes from an existing
output file if one is already present.

Run (from repo root):
    python benchmarking/scar-baseline/benchmark-claude.py
"""

import anthropic
import json
import time
from pathlib import Path
import random
from dotenv import load_dotenv
import os

load_dotenv()

WITH_CONCEPTS = True

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

output_path = "benchmarking/scar-baseline/claude_results_without_concepts.json"

if WITH_CONCEPTS:
    output_path = "benchmarking/scar-baseline/claude_results_with_concepts.json"

def run_analogy(analogy: dict) -> dict:
    # Extract domains
    system_a = analogy["system_a"]
    system_b = analogy["system_b"]

    # Extract entities
    concepts_a = [pair[0] for pair in analogy["mappings"]]
    concepts_b = [pair[1] for pair in analogy["mappings"]]

    # Shuffle independently
    random.shuffle(concepts_a)
    random.shuffle(concepts_b)

    print(f"System A: {system_a}")
    print(f"System B: {system_b}")
    print(f"Concepts in System A: {concepts_a}")
    print(f"Concepts in System B: {concepts_b}")

    if WITH_CONCEPTS:
        prompt = f"""

        /* Task prompt */  
        For two given systems, you are required to create an analogy by extracting concepts from the backgrounds of systems and matching the concepts in each system with one another in a one-to-one mapping. 
        /* Data */ 
        System A: {system_a}
        System B: {system_b}

        Concepts in System A: {", ".join(concepts_a)}
        Concepts in System B: {", ".join(concepts_b)}
        /* Question */  
        Question: Please establish the mappings between concepts. The format should be a list: (Concept1_SystemA, Concept1_SystemB), (Concept2_SystemA, Concept2_SystemB), ..."""

    else:
        prompt = f"""

        /* Task prompt */  
        For two given systems, you are required to create an analogy by extracting concepts that exist within the systems and are relevant for the analogy and matching the concepts in each system with one another in a one-to-one mapping.
        /* Data */ 
        System A: {system_a}
        System B: {system_b}

        /* Question */  
        Question: Please extract concepts from the systems and establish the mappings between concepts. The format should be a list: (Concept1_SystemA, Concept1_SystemB), (Concept2_SystemA, Concept2_SystemB), ..."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "id": analogy["id"],
        "response": next(block.text for block in response.content if block.type == "text"),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

def run_benchmark(delay: float = 0.5):
    # read the analogies from the scar_dataset.jsonl
    with open("benchmarking/scar-baseline/scar_dataset.jsonl") as f:
        analogies = [json.loads(line) for line in f]
    results = []

    # analogies = analogies[:2]

    if Path(output_path).exists(): 
        with open(output_path) as f: 
            results = json.load(f)


    completed_ids = {r["id"] for r in results}

    for i, analogy in enumerate(analogies):
        if analogy["id"] in completed_ids:
            continue

        print(f"Running {i+1}/{len(analogies)}: {analogy['id']}")
        
        try:
            result = run_analogy(analogy)
            results.append(result)
        except anthropic.RateLimitError:
            print("Rate limited — waiting 60s")
            time.sleep(60)
            result = run_analogy(analogy)
            results.append(result)
        except Exception as e:
            print(f"Error on {analogy['id']}: {e}")
            results.append({"id": analogy["id"], "error": str(e)})
        
        # Save after every analogy in case of interruption
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        time.sleep(delay)
    
    return results

run_benchmark()
