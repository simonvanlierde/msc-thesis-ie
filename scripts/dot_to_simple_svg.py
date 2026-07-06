"""Render Snakemake's small DOT DAG to a simple standalone SVG.

This is a fallback for development machines without the Graphviz ``dot`` binary.
The canonical command remains ``snakemake --dag | dot -Tsvg``.
"""

# ruff: noqa: T201 # Allow print() for SVG output; this script is not a library.

from __future__ import annotations

import html
import re
import sys

NODE_RE = re.compile(r'^\s*(\d+)\[label = "([^"]+)"')
EDGE_RE = re.compile(r"^\s*(\d+) -> (\d+)")


def main() -> None:
    """Read DOT from stdin and write a compact left-to-right SVG to stdout."""
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for line in sys.stdin:
        node_match = NODE_RE.match(line)
        if node_match:
            nodes[node_match.group(1)] = node_match.group(2)
            continue
        edge_match = EDGE_RE.match(line)
        if edge_match:
            edges.append((edge_match.group(1), edge_match.group(2)))

    if not nodes:
        msg = "No DOT nodes found on stdin."
        raise SystemExit(msg)

    indegree = dict.fromkeys(nodes, 0)
    children = {node: [] for node in nodes}
    for source, target in edges:
        children.setdefault(source, []).append(target)
        indegree[target] = indegree.get(target, 0) + 1

    ranks: dict[str, int] = {}
    queue = [node for node in nodes if indegree.get(node, 0) == 0]
    for node in queue:
        ranks[node] = 0
    while queue:
        source = queue.pop(0)
        for target in children.get(source, []):
            ranks[target] = max(ranks.get(target, 0), ranks[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    by_rank: dict[int, list[str]] = {}
    for node in nodes:
        by_rank.setdefault(ranks.get(node, 0), []).append(node)

    width = 260 * (max(by_rank, default=0) + 1)
    height = max(140, 95 * max(len(rank_nodes) for rank_nodes in by_rank.values()))
    positions: dict[str, tuple[int, int]] = {}
    for rank, rank_nodes in by_rank.items():
        step = height // (len(rank_nodes) + 1)
        for index, node in enumerate(sorted(rank_nodes, key=lambda key: nodes[key])):
            positions[node] = (130 + rank * 260, step * (index + 1))

    print(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    print(
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3"'
        ' orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#555"/></marker></defs>',
    )
    print('<rect width="100%" height="100%" fill="white"/>')
    for source, target in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        print(
            f'<line x1="{x1 + 90}" y1="{y1}" x2="{x2 - 90}" y2="{y2}"'
            ' stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>',
        )
    for node, label in nodes.items():
        x, y = positions[node]
        print(
            f'<rect x="{x - 90}" y="{y - 24}" width="180" height="48"'
            ' rx="6" fill="#f7fafc" stroke="#2b6cb0" stroke-width="1.5"/>',
        )
        label_lines = label.split(r"\n")
        start_y = y - 5 if len(label_lines) > 1 else y + 5
        print(
            f'<text x="{x}" y="{start_y}" text-anchor="middle" font-family="Arial,'
            ' sans-serif" font-size="13" fill="#1a202c">',
        )
        for index, label_line in enumerate(label_lines):
            dy = 0 if index == 0 else 16
            print(f'<tspan x="{x}" dy="{dy}">{html.escape(label_line)}</tspan>')
        print("</text>")
    print("</svg>")


if __name__ == "__main__":
    main()
