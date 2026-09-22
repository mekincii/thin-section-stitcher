from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.layout import (
    build_layout_graph,
    choose_root,
    maximum_confidence_tree,
    propagate_global_transforms,
    transform_point,
    transform_rotation_deg,
)
from thin_section_stitcher.overlap_graph import (
    build_overlap_graph,
    prepare_overlap_edges,
)

VERIFICATION_SCALE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the global thin-section image layout."
    )

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--consistency",
        type=Path,
        default=Path("outputs/match_consistency.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/initial_global_layout.csv"),
    )

    return parser.parse_args()


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

    graph = build_overlap_graph(
        image_names,
        edges,
        confidence_levels={"high"},
        exclude_warnings=True,
    )

    if not nx.is_connected(graph):
        raise RuntimeError(
            "High-confidence overlap graph is not connected."
        )

    layout_graph = build_layout_graph(graph)

    tree = maximum_confidence_tree(
        layout_graph
    )

    root = choose_root(tree)

    transforms = propagate_global_transforms(
        tree,
        root=root,
        image_scale=VERIFICATION_SCALE,
    )

    first_image = image_paths[0]

    import cv2

    sample = cv2.imread(str(first_image))

    if sample is None:
        raise RuntimeError(
            f"Could not read image: {first_image}"
        )

    height, width = sample.shape[:2]

    records = []

    depths = nx.single_source_shortest_path_length(
        tree,
        root,
    )

    for image_name in image_names:
        transform = transforms[image_name]

        center_x, center_y = transform_point(
            transform,
            width / 2.0,
            height / 2.0,
        )

        records.append(
            {
                "image": image_name,
                "center_x": center_x,
                "center_y": center_y,
                "rotation_deg": transform_rotation_deg(
                    transform
                ),
                "tree_depth": depths[image_name],
            }
        )

    dataframe = pd.DataFrame(records)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output,
        index=False,
    )

    print("=" * 72)
    print("INITIAL GLOBAL LAYOUT")
    print("=" * 72)

    print(f"Images positioned: {len(dataframe)}")
    print(f"Spanning-tree edges: {tree.number_of_edges()}")
    print(f"Root image: {root}")
    print(f"Maximum tree depth: {max(depths.values())}")

    print()
    print("Global center-coordinate range:")
    print(
        f"x: {dataframe['center_x'].min():.1f} "
        f"-> {dataframe['center_x'].max():.1f}"
    )
    print(
        f"y: {dataframe['center_y'].min():.1f} "
        f"-> {dataframe['center_y'].max():.1f}"
    )

    print()
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()