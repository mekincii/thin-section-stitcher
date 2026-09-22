from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import networkx as nx
import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.layout import (
    build_layout_graph,
    choose_root,
    maximum_confidence_tree,
    optimize_global_poses,
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
        description="Globally optimize microscope image positions."
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
        default=Path("outputs/optimized_global_layout.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = args.dataset.expanduser().resolve()

    image_paths = discover_images(dataset)

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
            "High-confidence graph must be connected."
        )

    layout_graph = build_layout_graph(
        graph
    )

    tree = maximum_confidence_tree(
        layout_graph
    )

    root = choose_root(tree)

    initial_transforms = (
        propagate_global_transforms(
            tree,
            root=root,
            image_scale=VERIFICATION_SCALE,
        )
    )

    sample = cv2.imread(
        str(image_paths[0])
    )

    if sample is None:
        raise RuntimeError(
            f"Could not read image: {image_paths[0]}"
        )

    height, width = sample.shape[:2]

    optimized = optimize_global_poses(
        graph,
        initial_transforms,
        root=root,
        image_width=width,
        image_height=height,
        image_scale=VERIFICATION_SCALE,
    )

    records = []

    for image_name in image_names:
        transform = optimized[image_name]

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
                "rotation_deg": (
                    transform_rotation_deg(
                        transform
                    )
                ),
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output,
        index=False,
    )

    print("=" * 72)
    print("OPTIMIZED GLOBAL LAYOUT")
    print("=" * 72)

    print(f"Images positioned: {len(dataframe)}")
    print(f"Constraints used: {graph.number_of_edges()}")
    print(f"Fixed root: {root}")

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
    print(
        f"Saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()