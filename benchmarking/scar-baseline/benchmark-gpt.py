"""
Benchmarks the basic SCAR dataset task of concept matching for GPT 5.4.

Requires the SCAR dataset at benchmarking/scar-baseline/scar_dataset.jsonl.
Runs immediately on import (no __main__ guard) and resumes from an existing
output file if one is already present.

Run (from repo root):
    python benchmarking/scar-baseline/benchmark-gpt.py
"""

import openai
import json
import time
from pathlib import Path
import random
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

WITH_CONCEPTS = False

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

output_path = "benchmarking/scar-baseline/gpt_results_without_concepts.json"

if WITH_CONCEPTS:
    output_path = "benchmarking/scar-baseline/gpt_results_with_concepts.json"

save_lock = threading.Lock()

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

    response = client.chat.completions.create(
        model="gpt-5.4",
        reasoning_effort="low",
        max_completion_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "id": analogy["id"],
        "response": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

def run_benchmark(max_workers: int = 5):
    # read the analogies from the scar_dataset.jsonl
    with open("benchmarking/scar-baseline/scar_dataset.jsonl") as f:
        analogies = [json.loads(line) for line in f]

    # analogies = analogies[:2]

    if Path(output_path).exists():
        with open(output_path) as f:
            results = json.load(f)
    else:
        results = []

    completed_ids = {r["id"] for r in results}
    remaining = [a for a in analogies if a["id"] not in completed_ids]

    print(f"{len(remaining)} analogies remaining out of {len(analogies)}")

    def process(analogy, index):
        print(f"Running {index}: {analogy['id']}")
        try:
            return run_analogy(analogy)
        except openai.RateLimitError:
            print("Rate limited — waiting 60s")
            time.sleep(60)
            return run_analogy(analogy)
        except Exception as e:
            print(f"Error on {analogy['id']}: {e}")
            return {"id": analogy["id"], "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process, analogy, i + 1): analogy
            for i, analogy in enumerate(remaining)
        }

        for future in as_completed(futures):
            result = future.result()
            with save_lock:
                results.append(result)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

    return results

run_benchmark()
