"""
Category Theory based Pipeline for AnaloBench with separate node and edge extraction.

Requires the AnaloBench T1 dataset CSV (not included; see benchmarking/README
or the AnaloBench repo at https://github.com/JHU-CLSP/AnaloBench) placed at
the path given by --dataset (default: benchmarking/analobench/test_analobench.csv).

Run (from repo root):
    python benchmarking/analobench/analobench_ct.py --condition ct_llm
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

import dspy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ct_pullback import (
    compute_confidence,
    compute_pullbacks_batched,
    get_encoder,
)

load_dotenv()

_encoder: Optional[SentenceTransformer] = None
_encoder_lock = threading.Lock()


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = SentenceTransformer("answerdotai/ModernBERT-large")
    return _encoder


DATASET_FILE = "benchmarking/analobench/test_analobench.csv"
OUTPUT_DIR   = "benchmarking/analobench/results"

RELATION_GUIDANCE = """
Use relation labels from this vocabulary wherever they accurately describe
the story. Ordered from most to least structurally important:

  CAUSAL:       causes, results_in, enables, prevents, leads_to, triggers
  INSTRUMENTAL: uses, produces, applies, blocks
  INTENTIONAL:  desires, wants_to_avoid, pursues, opposes, attempts_to_overcome
  STRUCTURAL:   relies_on, supports, depends_on, adjusts_to, replaces
  EPISTEMIC:    reveals, examines, represents, reflects_on, values, deceives
  RELATIONAL:   faces, receives, provides, responds_with

BANNED: affects, relates_to, involves, is_part_of, has, connects_to, is_about
"""

GRAPH_INSTRUCTIONS = """
Extract a CAUSAL SCHEMA GRAPH of the story following Structure Mapping Theory.
Output JSON with "nodes" (list of strings) and "edges" (list of [src, rel, tgt]).

NODE RULES:
- 6-9 nodes. Use ABSTRACT ROLE labels, not story-specific names.
  Use: agent, goal, obstacle, method, tool, outcome, helper, victim,
  constraint, alternative, deception, revelation, true_state, false_belief,
  forced_action, resistance, cost, risk, resource, authority.
- Compound labels are fine: obstacle_internal, method_primary.
- NO proper names, place names, or specific object names.

EDGE RULES — produce AT LEAST 8 edges:
  1. agent desires/pursues → goal
  2. agent faces/wants_to_avoid → obstacle
  3. obstacle causes/prevents → [effect]
  4. agent uses/applies → method
  5. method enables/prevents/blocks → [target]
  6. method leads_to/results_in → outcome
  7. helper/authority action and its effect
  8. any cost, risk, or side-effect edges
  9. outcome depends_on/relies_on → condition (if applicable)

Use ONLY these relations:
  desires, faces, wants_to_avoid, uses, applies, enables, prevents,
  causes, leads_to, triggers, results_in, reveals, deceives, blocks,
  produces, depends_on, relies_on, supports, opposes
"""


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

class DirectZeroShot(dspy.Signature):
    """You are given a source story and four candidate stories (A, B, C, D).
    Identify which candidate is the BEST structural analogy to the source.
    A structural analogy shares the same underlying causal mechanism and
    pattern of goals/obstacles/outcomes — even if surface content differs.
    Do NOT choose based on topic or surface similarity.
    Answer with a single letter: A, B, C, or D.
    """
    source_story: str = dspy.InputField(desc="The source story")
    option_a: str = dspy.InputField(desc="Candidate A")
    option_b: str = dspy.InputField(desc="Candidate B")
    option_c: str = dspy.InputField(desc="Candidate C")
    option_d: str = dspy.InputField(desc="Candidate D")
    reasoning: str = dspy.OutputField(desc="Brief reasoning")
    answer: Literal["A", "B", "C", "D"] = dspy.OutputField(desc="Single letter: A, B, C, or D")

class ExtractGraph(dspy.Signature):
    """Extract a structured graph from a story following Structure Mapping Theory.
    Output JSON with "nodes" (list of strings) and "edges" (list of [src, rel, tgt]).
    """
    story: str = dspy.InputField(desc="The story to represent as a graph")
    graph_instructions: str = dspy.InputField(desc="Node and edge extraction rules")
    relation_guidance: str = dspy.InputField(desc="Preferred relation label vocabulary")
    graph_json: str = dspy.OutputField(
        desc='JSON: {"nodes": [...], "edges": [[src, rel, tgt], ...]}'
    )
    
# ---------------------------------------------------------------------------
# Staged extraction: first nodes, then edges
# ---------------------------------------------------------------------------

class ExtractNodesJointly(dspy.Signature):
    """STAGE 1 of two-stage graph construction.

    You are given a source story and four option stories (A, B, C, D).
    The graphs built here will be matched by a graph-similarity algorithm
    to find structural analogies — so SHARED NODE VOCABULARY across
    structurally parallel stories is essential.

    For each story identify:
      (a) Agent — who is the main actor?
      (b) Goal — what do they want?
      (c) Obstacle — what blocks or threatens the goal?
      (d) Method — what action or tool do they use?
      (e) Outcome — what final state results?
      (f) Secondary actors — helpers, authorities, antagonists?

    Then decide which stories share the same abstract causal pattern and
    assign IDENTICAL node labels to roles that play the same function.
    Write an explicit mapping table before the JSON output.

    NODE RULES:
    - 5–8 nodes per story.
    - Abstract functional labels: agent, goal, obstacle, method, tool,
      outcome, helper, victim, constraint, alternative, cost, risk,
      deception_tool, revelation, true_state, false_belief, authority,
      forced_action, resistance.
    - Compound labels allowed: obstacle_internal, method_primary.
    - NO proper names, place names, or specific object names.
    - Analogous stories MUST share identical labels for parallel roles.

    Output JSON with keys "source", "A", "B", "C", "D", each a list of strings.
    """
    source_story: str = dspy.InputField(desc="The source story")
    option_a: str = dspy.InputField(desc="Option A story")
    option_b: str = dspy.InputField(desc="Option B story")
    option_c: str = dspy.InputField(desc="Option C story")
    option_d: str = dspy.InputField(desc="Option D story")
    scratchpad: str = dspy.OutputField(
        desc="Analysis of each story (a)–(f) and alignment mapping table"
    )
    nodes_json: str = dspy.OutputField(
        desc='JSON: {"source": [...], "A": [...], "B": [...], "C": [...], "D": [...]}'
    )


class ExtractRelationsJointly(dspy.Signature):
    """STAGE 2 of two-stage graph construction.

    You are given five stories, their Stage 1 nodes, and the Stage 1 scratchpad.
    Find all meaningful causal relations between nodes for each story.

    Relations are embedded as "[src] [rel] [tgt]" and compared by cosine
    similarity. Structurally parallel stories should produce identical edge
    strings — use the Stage 1 alignment table to ensure this.

    RELATION RULES:
    - Only add a relation if genuinely supported by the story text.
    - Multiple relations between the same pair allowed.
    - Produce AT LEAST 7 edges per story.
    - Closed vocabulary ONLY:
        causes, results_in, leads_to, triggers, produces,
        enables, prevents, blocks, relies_on,
        desires, wants_to_avoid, faces, pursues,
        reveals, deceives, uses, adjusts_to, provides, receives
    - BANNED: affects, relates_to, involves, is, has, connects_to, is_about

    Work through for every story:
      1. agent desires/pursues → goal
      2. agent faces/wants_to_avoid → obstacle
      3. obstacle causes/prevents → [effect]
      4. agent uses → method/tool
      5. method enables/prevents → [target]
      6. method leads_to/results_in → outcome
      7. any helper/authority action and effect
      8. any cost, risk, or side-effect edges

    Output JSON with keys "source", "A", "B", "C", "D", each a list of
    [source_node, relation, target_node] triples using only Stage 1 node labels.
    """
    source_story: str = dspy.InputField(desc="The source story")
    option_a: str = dspy.InputField(desc="Option A story")
    option_b: str = dspy.InputField(desc="Option B story")
    option_c: str = dspy.InputField(desc="Option C story")
    option_d: str = dspy.InputField(desc="Option D story")
    nodes_json: str = dspy.InputField(desc="Node lists from Stage 1")
    stage1_scratchpad: str = dspy.InputField(desc="Alignment analysis from Stage 1")
    relations_json: str = dspy.OutputField(
        desc='JSON: {"source": [[n,r,n],...], "A": [...], "B": [...], "C": [...], "D": [...]}'
    )


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

predict_direct_zs     = dspy.Predict(DirectZeroShot)
predict_extract       = dspy.Predict(ExtractGraph)
predict_nodes         = dspy.Predict(ExtractNodesJointly)
predict_relations     = dspy.Predict(ExtractRelationsJointly)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_json(text: str) -> dict | list:
    if not text:
        return {}
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        pass
    return {}


def extract_graph_from_story(story: str) -> dict:
    result = predict_extract(
        story=story,
        graph_instructions=GRAPH_INSTRUCTIONS,
        relation_guidance=RELATION_GUIDANCE,
    )
    return safe_json(result.graph_json)

# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def extract_graphs(entry: dict) -> tuple[dict, dict, str]:
    """
    Extract source and option graphs using two LLM calls (ExtractNodesJointly → ExtractRelationsJointly)

    Returns (source_graph, option_graphs, extraction_mode).
    Falls back to independent per-story extraction on failure.
    """
    # Stage 1 — nodes with scratchpad alignment
    node_result = predict_nodes(
        source_story=entry["story"],
        option_a=entry["options"].get("A", ""),
        option_b=entry["options"].get("B", ""),
        option_c=entry["options"].get("C", ""),
        option_d=entry["options"].get("D", ""),
    )
    all_nodes = safe_json(node_result.nodes_json)
    if not isinstance(all_nodes, dict) or "source" not in all_nodes:
        raise ValueError("Staged Stage 1: node extraction returned invalid structure")

    # Stage 2 — relations conditioned on Stage 1 + scratchpad
    rel_result = predict_relations(
        source_story=entry["story"],
        option_a=entry["options"].get("A", ""),
        option_b=entry["options"].get("B", ""),
        option_c=entry["options"].get("C", ""),
        option_d=entry["options"].get("D", ""),
        nodes_json=node_result.nodes_json,
        stage1_scratchpad=node_result.scratchpad,
    )
    all_relations = safe_json(rel_result.relations_json)
    if not isinstance(all_relations, dict) or "source" not in all_relations:
        raise ValueError("Staged Stage 2: relation extraction returned invalid structure")

    source_graph = {
        "nodes": all_nodes.get("source", []),
        "edges": all_relations.get("source", []),
    }
    option_graphs = {
        l: {"nodes": all_nodes.get(l, []), "edges": all_relations.get(l, [])}
        for l in "ABCD"
    }
    return source_graph, option_graphs, "staged"

# ---------------------------------------------------------------------------
# Condition implementations
# ---------------------------------------------------------------------------

def run_direct_zeroshot(entry: dict) -> dict:
    result = predict_direct_zs(
        source_story=entry["story"],
        option_a=entry["options"].get("A", ""),
        option_b=entry["options"].get("B", ""),
        option_c=entry["options"].get("C", ""),
        option_d=entry["options"].get("D", ""),
    )
    return {"answer": result.answer, "reasoning": result.reasoning}


def run_ct_llm(entry: dict) -> dict:
    """
    CT pipeline — algo-only (no LLM selection call):
      1. Graph extraction (joint or staged, controlled by --extraction flag)
      2. Batched semantic pullback with ModernBERT-large embeddings
      3. Pullback score winner = final answer
    """
    log: list[str] = []

    # Graph extraction
    try:
        source_graph, option_graphs, extraction_mode = extract_graphs(entry)
    except Exception as e:
        log.append(f"  [Extraction failed: {e}, falling back to independent]")
        source_graph = extract_graph_from_story(entry["story"])
        option_graphs = {
            l: extract_graph_from_story(entry["options"].get(l, ""))
            for l in "ABCD"
        }
        extraction_mode = "independent"

    log.append(
        f"  [Extraction: {extraction_mode}] "
        f"source={len(source_graph.get('edges', []))} edges, "
        f"options={[len(option_graphs[l].get('edges', [])) for l in 'ABCD']}"
    )

    # Batched semantic pullback
    pullback_results = compute_pullbacks_batched(source_graph, option_graphs)
    algo_best = max("ABCD", key=lambda l: pullback_results[l]["score"])

    for letter in "ABCD":
        r = pullback_results[letter]
        log.append(
            f"  {letter}: matched={r['num_matched']} coverage={r['coverage']:.0%} "
            f"sim={r['total_similarity']} chain={r['chain_length']} score={r['score']}"
        )

    # Confidence (relative gap — robust across embedding distributions)
    confidence, score_gap, rel_gap = compute_confidence(pullback_results)
    top_coverage = pullback_results[algo_best]["coverage"]
    log.append(
        f"  Confidence: {confidence} (gap={score_gap}, rel={rel_gap:.2f}, "
        f"coverage={top_coverage:.0%})  →  {algo_best}"
    )
    tqdm.write("\n".join(log))
 
    return {
        "answer": algo_best,
        "algo_answer": algo_best,
        "confidence": confidence,
        "extraction_mode": extraction_mode,
        "pullback_results": {
            k: {**v, "matched_pairs": v["matched_pairs"][:5]}
            for k, v in pullback_results.items()
        },
        "source_graph": source_graph,
        "option_graphs": option_graphs,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_analobench(csv_file: str) -> list[dict]:
    entries = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            options, current_key, buffer = {}, None, []
            for line in row["Options"].split("\n"):
                line = line.strip()
                if line.startswith(("A. ", "B. ", "C. ", "D. ")):
                    if current_key:
                        options[current_key] = "\n".join(buffer).strip()
                    current_key, buffer = line[0], [line[3:]]
                else:
                    buffer.append(line)
            if current_key:
                options[current_key] = "\n".join(buffer).strip()
            entries.append({
                "index": row["Index"],
                "sentence": row["Sentence"],
                "story": row["Story"],
                "options": options,
                "label": row["Label"].strip(),
            })
    return entries


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CONDITION_FNS = {
    "direct_zeroshot": run_direct_zeroshot,
    "ct_llm":          run_ct_llm,
}


def run_condition(
    condition: str,
    dataset_file: str = DATASET_FILE,
    output_dir: str = OUTPUT_DIR,
    limit: Optional[int] = None,
    workers: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{condition}.jsonl")

    # Resume support
    completed: dict[str, dict] = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "index" in rec and rec.get("predicted") not in (None, "?"):
                        completed[str(rec["index"])] = rec
                except Exception:
                    pass
        if completed:
            tqdm.write(f"  [Resume] {len(completed)} already completed, skipping.")

    entries = load_analobench(dataset_file)
    if limit:
        entries = entries[:limit]
    remaining = [e for e in entries if str(e["index"]) not in completed]
    tqdm.write(
        f"  [Resume] {len(completed)} done, {len(remaining)} remaining "
        f"of {len(entries)} total."
    )

    fn = CONDITION_FNS[condition]

    # Seed counters from completed records
    correct = sum(1 for r in completed.values() if r.get("correct", False))
    confidence_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    confidence_correct: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for rec in completed.values():
        c = rec.get("confidence")
        if c in confidence_counts:
            confidence_counts[c] += 1
            if rec.get("correct", False):
                confidence_correct[c] += 1

    lock = threading.Lock()
    skipped = 0

    def process(entry: dict) -> tuple:
        for attempt in range(3):
            try:
                return entry, fn(entry)
            except Exception as e:
                if attempt == 2:
                    tqdm.write(f"  ERROR [{entry['index']}]: {e}")
                    return entry, {"answer": "?", "error": str(e)}
                wait = 2 ** attempt
                tqdm.write(f"  RETRY [{entry['index']}] in {wait}s: {e}")
                time.sleep(wait)

    with open(output_file, "a") as out:
        pbar = tqdm(total=len(remaining), desc=condition, unit="item")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed({pool.submit(process, e): e for e in remaining}):
                entry, pred = future.result()
                is_correct = pred["answer"] == entry["label"]

                with lock:
                    failed = pred["answer"] == "?"
                    if not failed:
                        correct += is_correct
                    c = pred.get("confidence")
                    if c in confidence_counts:
                        confidence_counts[c] += 1
                        if is_correct:
                            confidence_correct[c] += 1
                    out.write(json.dumps({
                        "index":     entry["index"],
                        "sentence":  entry["sentence"],
                        "label":     entry["label"],
                        "predicted": pred["answer"],
                        "correct":   is_correct,
                        **{k: v for k, v in pred.items() if k != "answer"},
                    }) + "\n")
                    out.flush()
                    if failed:
                        skipped += 1
                    scored = pbar.n + 1 - skipped
                    tqdm.write(
                        f"[{entry['index']}] {'✓' if is_correct else ('?' if failed else '✗')} "
                        f"predicted={pred['answer']}  label={entry['label']}"
                    )
                    pbar.update(1)
                    pbar.set_postfix(
                        acc=f"{correct/scored:.1%}" if scored else "n/a",
                        correct=f"{correct}/{scored}",
                    )
        pbar.close()

    n = len(entries)
    scored = n - skipped
    acc = correct / scored if scored else 0

    print(f"\n{'='*50}")
    print(f"Condition : {condition}")
    print(f"Accuracy  : {correct}/{scored} = {acc:.1%}"
          + (f"  ({skipped} failed)" if skipped else ""))

    if any(confidence_counts.values()):
        total = sum(confidence_counts.values())
        print(f"\nPullback confidence breakdown:")
        for level in ["HIGH", "MEDIUM", "LOW"]:
            cnt = confidence_counts[level]
            a = confidence_correct[level] / cnt if cnt else 0
            print(f"  {level:<8} {cnt:>4} tasks ({cnt/total*100:4.1f}%)  "
                  f"accuracy: {confidence_correct[level]}/{cnt} = {a:.1%}")
        hm = confidence_counts["HIGH"] + confidence_counts["MEDIUM"]
        print(f"  HIGH+MEDIUM: {hm}/{total} = {hm/total:.1%} had meaningful signal")

    print(f"\nOutput    : {output_file}")
    return correct, n, acc


def run_all(dataset_file=DATASET_FILE, output_dir=OUTPUT_DIR, limit=None, workers=1):
    summary = {}
    for condition in CONDITION_FNS:
        print(f"\n{'='*50}\nRunning: {condition}\n{'='*50}")
        _, _, acc = run_condition(condition, dataset_file, output_dir, limit, workers)
        summary[condition] = acc
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for cond, acc in summary.items():
        print(f"  {cond:<20} {acc:.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _configure_lm(model: str, api_base: Optional[str] = None) -> None:
    api_key = (
        os.getenv("OPENROUTER_API_KEY") if model.startswith("openrouter/")
        else os.getenv("OPENAI_API_KEY")
    )
    kwargs: dict = {"model": model, "api_key": api_key, "max_tokens": 8192}
    if api_base:
        kwargs["api_base"] = api_base
    dspy.configure(lm=dspy.LM(**kwargs))
    print(f"Model     : {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=["direct_zeroshot", "ct_llm", "all"],
        default="ct_llm",
    )
    parser.add_argument("--dataset", default=DATASET_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (careful with rate limits)")
    parser.add_argument(
        "--model",
        default="openrouter/google/gemma-4-31b-it",
        help="LiteLLM model string, e.g. openai/gpt-4.1-mini",
    )
    parser.add_argument("--api_base", default=None)
    args = parser.parse_args()

    _configure_lm(args.model, args.api_base)

    if args.condition == "all":
        run_all(args.dataset, args.output_dir, args.limit, args.workers)
    else:
        run_condition(
            args.condition, args.dataset, args.output_dir,
            args.limit, args.workers,
        )