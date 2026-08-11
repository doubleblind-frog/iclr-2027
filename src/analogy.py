"""
analogy.py — Pydantic schema for a domain graph (nodes/edges).

Library module only, no CLI of its own — imported by
scripts/generate_analogy_graphs.py and scripts/judge_graph_eval.py as the
structured-output type for the SCAR domain-graph generation pipeline.
"""

from pydantic import BaseModel

# do not include the color for now

class Node(BaseModel):
    name: str


class Edge(BaseModel):
    name: str
    source: Node
    target: Node

class Analogy(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
