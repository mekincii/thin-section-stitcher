from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.overlap_graph import (
    build_overlap_graph,
    graph_summary,
    prepare_overlap_edges,
    save_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and analyze the microscope-image overlap graph."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing the original microscope images.",
    )

    parser.add_argument(
        "--consistency",
        type=Path,
        default=Path("outputs/match_consistency.csv"),
    )

    parser.add_argument(
        "--edges-output",
        type=Path,
        default=Path("outputs/overlap_edges.csv"),
    )

    parser.add_argument(
        "--graph-output",
        type=Path,
        default=Path("outputs/overlap_graph.graphml"),
    )

    return parser.parse_args()


def print_summary(
    label: str,
    graph: nx.Graph,
) -> None:
    summary = graph_summary(graph)

    print()
    print(label)
    print("-" * len(label))

    print(f"Nodes: {summary['nodes']}")
    print(f"Edges: {summary['edges']}")
    print(f"Connected components: {summary['components']}")
    print(
        f"Largest component: "
        f"{summary['largest_component']} images"
    )

    print(
        f"Degree: min={summary['min_degree']}, "
        f"mean={summary['mean_degree']:.2f}, "
        f"max={summary['max_degree']}"
    )

    isolates = summary["isolates"]

    if isolates:
        print(
            "Isolated images: "
            + ", ".join(isolates)
        )
    else:
        print("Isolated images: none")


def main() -> None:
    args = parse_args()

    image_paths = discover_images(
        args.dataset.expanduser().resolve()
    )

    image_names = [
        path.name
        for path in image_paths
    ]

    consistency = pd.read_csv(
        args.consistency
    )

    edges = prepare_overlap_edges(
        consistency
    )

    print("=" * 72)
    print("OVERLAP GRAPH ANALYSIS")
    print("=" * 72)

    print(f"Images: {len(image_names)}")
    print(f"Candidate overlaps: {len(edges)}")

    print()
    print("Confidence distribution:")
    print(
        edges["confidence"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Cross-scale consistency warnings: "
        f"{edges['consistency_warning'].sum()}"
    )

    high_graph = build_overlap_graph(
        image_names,
        edges,
        confidence_levels={"high"},
        exclude_warnings=True,
    )

    trusted_graph = build_overlap_graph(
        image_names,
        edges,
        confidence_levels={"high", "medium"},
        exclude_warnings=True,
    )

    print_summary(
        "HIGH-CONFIDENCE GRAPH",
        high_graph,
    )

    print_summary(
        "HIGH + MEDIUM CONFIDENCE GRAPH",
        trusted_graph,
    )

    components = sorted(
        nx.connected_components(trusted_graph),
        key=len,
        reverse=True,
    )

    print()
    print("Component sizes:")
    print(
        ", ".join(
            str(len(component))
            for component in components
        )
    )

    degree_table = pd.DataFrame(
        [
            {
                "image": node,
                "degree": trusted_graph.degree(node),
            }
            for node in trusted_graph.nodes
        ]
    ).sort_values(
        ["degree", "image"],
        ascending=[True, True],
    )

    print()
    print("Lowest-degree images:")
    print(
        degree_table
        .head(20)
        .to_string(index=False)
    )

    args.edges_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges.to_csv(
        args.edges_output,
        index=False,
    )

    save_graph(
        trusted_graph,
        args.graph_output,
    )

    print()
    print(
        f"Edge table saved to: "
        f"{args.edges_output.resolve()}"
    )

    print(
        f"Graph saved to: "
        f"{args.graph_output.resolve()}"
    )


if __name__ == "__main__":
    main()