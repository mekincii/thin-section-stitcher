from __future__ import annotations

import argparse
from math import atan2, degrees, hypot
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.layout import (
    build_layout_graph,
    choose_root,
    maximum_confidence_tree,
    propagate_global_transforms,
    relative_transform,
)
from thin_section_stitcher.overlap_graph import (
    build_overlap_graph,
    prepare_overlap_edges,
)

VERIFICATION_SCALE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure loop-closure error in the initial global layout."
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
        default=Path("outputs/layout_residuals.csv"),
    )

    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help=(
            "Optional saved global layout. "
            "If omitted, analyze the spanning-tree initialization."
        ),
    )

    return parser.parse_args()


def transforms_from_layout(
    layout: pd.DataFrame,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    """
    Reconstruct rigid image-to-global transforms from saved
    image-center coordinates and rotations.
    """
    transforms: dict[str, np.ndarray] = {}

    center_local = np.array(
        [
            image_width / 2.0,
            image_height / 2.0,
        ],
        dtype=float,
    )

    for row in layout.to_dict(orient="records"):
        angle = np.radians(
            float(row["rotation_deg"])
        )

        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=float,
        )

        center_global = np.array(
            [
                float(row["center_x"]),
                float(row["center_y"]),
            ],
            dtype=float,
        )

        translation = (
            center_global
            - rotation @ center_local
        )

        transform = np.eye(
            3,
            dtype=float,
        )

        transform[:2, :2] = rotation
        transform[:2, 2] = translation

        transforms[str(row["image"])] = (
            transform
        )

    return transforms


def rotation_deg(transform: np.ndarray) -> float:
    return degrees(
        atan2(
            float(transform[1, 0]),
            float(transform[0, 0]),
        )
    )


def transform_scale(transform: np.ndarray) -> float:
    return hypot(
        float(transform[0, 0]),
        float(transform[1, 0]),
    )


def wrapped_angle_error(
    angle_a: float,
    angle_b: float,
) -> float:
    difference = (
        angle_a - angle_b + 180.0
    ) % 360.0 - 180.0

    return abs(difference)


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

    sample = cv2.imread(
        str(image_paths[0])
    )

    if sample is None:
        raise RuntimeError(
            f"Could not read image: {image_paths[0]}"
        )

    height, width = sample.shape[:2]

    if args.layout is None:
        transforms = propagate_global_transforms(
            tree,
            root=root,
            image_scale=VERIFICATION_SCALE,
        )

        layout_label = "spanning-tree initialization"

    else:
        saved_layout = pd.read_csv(
            args.layout
        )

        transforms = transforms_from_layout(
            saved_layout,
            image_width=width,
            image_height=height,
        )

        missing = (
            set(image_names)
            - set(transforms)
        )

        if missing:
            raise RuntimeError(
                f"Layout is missing images: "
                f"{sorted(missing)}"
            )

        layout_label = str(args.layout)

    center_a = np.array(
        [
            width / 2.0,
            height / 2.0,
            1.0,
        ],
        dtype=float,
    )

    tree_edges = {
        frozenset((node_a, node_b))
        for node_a, node_b in tree.edges
    }

    records = []

    for node_a, node_b, data in graph.edges(
        data=True
    ):
        edge_key = frozenset(
            (node_a, node_b)
        )

        if edge_key in tree_edges:
            continue

        image_a = str(data["image_a"])
        image_b = str(data["image_b"])

        # Global optimization deliberately assumes rigid transforms:
        # translation + rotation, with scale fixed to 1.
        measured = relative_transform(
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            rotation_deg=float(
                data["rotation_deg"]
            ),
            scale=1.0,
            image_scale=VERIFICATION_SCALE,
        )

        predicted = (
            np.linalg.inv(
                transforms[image_b]
            )
            @ transforms[image_a]
        )

        measured_center = (
            measured @ center_a
        )

        predicted_center = (
            predicted @ center_a
        )

        center_error = hypot(
            float(
                measured_center[0]
                - predicted_center[0]
            ),
            float(
                measured_center[1]
                - predicted_center[1]
            ),
        )

        translation_error = hypot(
            float(
                measured[0, 2]
                - predicted[0, 2]
            ),
            float(
                measured[1, 2]
                - predicted[1, 2]
            ),
        )

        measured_rotation = rotation_deg(
            measured
        )

        predicted_rotation = rotation_deg(
            predicted
        )

        rotation_error = wrapped_angle_error(
            measured_rotation,
            predicted_rotation,
        )

        scale_error = abs(
            transform_scale(measured)
            - transform_scale(predicted)
        )

        records.append(
            {
                "image_a": image_a,
                "image_b": image_b,
                "tree_path_length": (
                    nx.shortest_path_length(
                        tree,
                        image_a,
                        image_b,
                    )
                ),
                "inliers": int(
                    data["inliers"]
                ),
                "inlier_ratio": float(
                    data["inlier_ratio"]
                ),
                "center_error_px": center_error,
                "translation_error_px": (
                    translation_error
                ),
                "rotation_error_deg": (
                    rotation_error
                ),
                "scale_error": scale_error,
            }
        )

    residuals = pd.DataFrame(
        records
    )

    residuals = residuals.sort_values(
        "center_error_px",
        ascending=False,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals.to_csv(
        args.output,
        index=False,
    )

    print("=" * 72)
    print("GLOBAL LAYOUT LOOP-CLOSURE ANALYSIS")
    print("=" * 72)

    print(f"Layout analyzed: {layout_label}")
    print(f"Root image: {root}")
    print(f"Tree edges: {tree.number_of_edges()}")
    print(
        f"Non-tree edges tested: "
        f"{len(residuals)}"
    )

    print()
    print("Center-position closure error (pixels):")
    print(
        residuals["center_error_px"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print("Rotation closure error (degrees):")
    print(
        residuals["rotation_error_deg"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print("Worst loop closures:")
    print(
        residuals[
            [
                "image_a",
                "image_b",
                "tree_path_length",
                "inliers",
                "inlier_ratio",
                "center_error_px",
                "rotation_error_deg",
                "scale_error",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()