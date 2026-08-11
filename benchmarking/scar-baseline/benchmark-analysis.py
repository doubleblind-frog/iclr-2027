"""
Analyze the results of the benchmarking, computing local and global accuracies.

Requires a results file already produced by benchmark-claude.py or
benchmark-gpt.py at the path below (toggle WITH_CONCEPTS to match the run
you want to analyse).

Run (from repo root):
    python benchmarking/scar-baseline/benchmark-analysis.py
"""

import json
import re

WITH_CONCEPTS = True

RESULTS_FILE = (
    "benchmarking/scar-baseline/claude_results_with_concepts.json"
    if WITH_CONCEPTS
    else "benchmarking/scar-baseline/claude_results_without_concepts.json"
)
OUTPUT_FILE = (
    "benchmarking/scar-baseline/local_accuracy_with_concepts.json"
    if WITH_CONCEPTS
    else "benchmarking/scar-baseline/local_accuracy_without_concepts.json"
)
DATASET_FILE = "benchmarking/scar-baseline/scar_dataset.jsonl"

def load_jsonl(file_path: str) -> dict[int, dict]:
    """Load JSONL and index by id."""
    records = {}
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            records[record["id"]] = record
    return records


def load_results(file_path: str) -> dict[int, str]:
    """Load results JSON and index by id."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # add error handling by printing the id of the entries without responses, and then not including them in the return

    for entry in data:
        if "response" not in entry:
            print(f"[WARN] id {entry['id']} has no response, skipping.")
            data.remove(entry)
            continue

    return {entry["id"]: entry["response"] for entry in data}


def extract_concept_pairs(response: str) -> list[tuple[str, str]]:
    """Extract (entity1, entity2) pairs from a response string."""
    pattern = r'\(([^()]+),\s*([^()]+)\)'
    matches = re.findall(pattern, response)
    return [(e1.strip().lower(), e2.strip().lower()) for e1, e2 in matches
            if e1.strip() and e2.strip()]


def normalise(text: str) -> str:
    """Lowercase and strip for fuzzy comparison."""
    return text.strip().lower()


def compute_local_accuracy(
    extracted: list[tuple[str, str]],
    ground_truth: list[list[str]]
) -> float:
    """
    For each ground-truth mapping, check whether it appears in the extracted
    pairs. Returns the fraction of ground-truth mappings that were found.
    """
    if not ground_truth:
        return 0.0

    extracted_set = {(normalise(a), normalise(b)) for a, b in extracted}
    correct = sum(
        1 for gt in ground_truth
        if (normalise(gt[0]), normalise(gt[1])) in extracted_set
    )
    return correct / len(ground_truth)


if __name__ == "__main__":
    dataset   = load_jsonl(DATASET_FILE)
    results   = load_results(RESULTS_FILE)

    local_accuracies = []
    total_correct    = 0
    total_mappings   = 0

    for entry_id, response in results.items():
        if entry_id not in dataset:
            print(f"[WARN] id {entry_id} not found in dataset, skipping.")
            continue

        ground_truth = dataset[entry_id]["mappings"]   # list of [e1, e2]
        extracted    = extract_concept_pairs(response)

        local_acc = compute_local_accuracy(extracted, ground_truth)
        local_accuracies.append({"id": entry_id, "accuracy": local_acc})

        # Accumulate globals
        extracted_set = {(normalise(a), normalise(b)) for a, b in extracted}
        total_correct += sum(
            1 for gt in ground_truth
            if (normalise(gt[0]), normalise(gt[1])) in extracted_set
        )
        total_mappings += len(ground_truth)

    global_accuracy = total_correct / total_mappings if total_mappings else 0.0

    # ── write local accuracies ────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(local_accuracies, f, indent=2)

    print(f"Mode            : {'with' if WITH_CONCEPTS else 'without'} concepts")
    print(f"Global accuracy : {global_accuracy:.4f}  "
          f"({total_correct}/{total_mappings} mappings correct)")
    print(f"Local accuracies written to: {OUTPUT_FILE}")


    def pick_tricky_analogies():
        # Open the output file and print all analogies that don't have 100% local accuracy
        with open(OUTPUT_FILE, "r") as f:
            local_accuracies = json.load(f)

        tricky_ids = [entry["id"] for entry in local_accuracies if entry["accuracy"] < 1.0]
        print(f"Tricky analogies (accuracy < 1.0): {tricky_ids}")

pick_tricky_analogies()    