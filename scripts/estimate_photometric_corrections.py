from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from thin_section_stitcher.dataset import (
    discover_images,
)
from thin_section_stitcher.overlap_graph import (
    prepare_overlap_edges,
)
from thin_section_stitcher.photometric import (
    estimate_photometric_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate conservative overlap-derived photometric "
            "standardization gains for microscope images."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--layout",
        type=Path,
        default=Path(
            "outputs/optimized_global_layout.csv"
        ),
    )

    parser.add_argument(
        "--consistency",
        type=Path,
        default=Path(
            "outputs/match_consistency.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/photometric_gains.csv"
        ),
    )

    parser.add_argument(
        "--measurements-output",
        type=Path,
        default=Path(
            "outputs/photometric_measurements.csv"
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--root",
        type=str,
        default="60.jpg",
    )

    parser.add_argument(
        "--include-medium",
        action="store_true",
    )

    parser.add_argument(
        "--validated-bridges",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def print_gain_summary(
    gains: pd.DataFrame,
) -> None:
    print()
    print("GAIN SUMMARY")
    print("-" * 72)

    for column in [
        "gain_blue",
        "gain_green",
        "gain_red",
    ]:
        values = gains[
            column
        ]

        print(
            f"{column:12s} "
            f"min={values.min():.4f}  "
            f"median={values.median():.4f}  "
            f"max={values.max():.4f}"
        )

    working = gains.copy()

    working[
        "max_log_deviation"
    ] = np.max(
        np.abs(
            np.log(
                working[
                    [
                        "gain_blue",
                        "gain_green",
                        "gain_red",
                    ]
                ]
            )
        ),
        axis=1,
    )

    print()
    print(
        "Images requiring the largest correction:"
    )

    print(
        working.sort_values(
            "max_log_deviation",
            ascending=False,
        )[
            [
                "image",
                "gain_blue",
                "gain_green",
                "gain_red",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


def main() -> None:
    args = parse_args()

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    image_paths = discover_images(
        dataset
    )

    layout = pd.read_csv(
        args.layout
    )

    consistency = pd.read_csv(
        args.consistency
    )

    edges = prepare_overlap_edges(
        consistency
    )

    confidence_levels = {"high"}

    if args.include_medium:
        confidence_levels.add("medium")

    trusted_edges = edges[
        edges["confidence"].isin(confidence_levels)
        & ~edges["consistency_warning"]
        ].copy()

    if args.validated_bridges is not None:
        bridges = pd.read_csv(
            args.validated_bridges
        )

        bridge_edges = pd.DataFrame(
            {
                "image_a": bridges["image_a"],
                "image_b": bridges["image_b"],
                "inlier_ratio_verification": (
                    bridges["inlier_ratio"]
                ),
            }
        )

        existing_pairs = {
            frozenset((row.image_a, row.image_b))
            for row in trusted_edges.itertuples()
        }

        bridge_edges = bridge_edges[
            ~bridge_edges.apply(
                lambda row: frozenset(
                    (row["image_a"], row["image_b"])
                ) in existing_pairs,
                axis=1,
            )
        ]

        trusted_edges = pd.concat(
            [
                trusted_edges,
                bridge_edges,
            ],
            ignore_index=True,
            sort=False,
        )

    print("=" * 72)
    print("PHOTOMETRIC STANDARDIZATION ESTIMATION")
    print("=" * 72)

    print(
        f"Images: {len(image_paths)}"
    )

    print(
        f"Trusted overlap constraints: "
        f"{len(trusted_edges)}"
    )

    print(
        f"Estimation scale: "
        f"{args.scale:.3f}"
    )

    photometric_graph = nx.Graph()

    photometric_graph.add_nodes_from(
        path.name
        for path in image_paths
    )

    photometric_graph.add_edges_from(
        (
            str(row["image_a"]),
            str(row["image_b"]),
        )
        for row in trusted_edges.to_dict(
            orient="records"
        )
    )

    if not nx.is_connected(
            photometric_graph
    ):
        components = sorted(
            nx.connected_components(
                photometric_graph
            ),
            key=len,
            reverse=True,
        )

        raise RuntimeError(
            "Photometric overlap graph is not connected. "
            f"Component sizes: "
            f"{[len(component) for component in components]}"
        )

    result = estimate_photometric_gains(
        image_paths,
        layout,
        trusted_edges,
        scale=args.scale,
        root=args.root,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.measurements_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.gains.to_csv(
        args.output,
        index=False,
    )

    result.measurements.to_csv(
        args.measurements_output,
        index=False,
    )

    print()
    print("=" * 72)
    print("ESTIMATION COMPLETE")
    print("=" * 72)

    print(
        f"Successful overlap measurements: "
        f"{result.successful_pairs}"
    )

    print(
        f"Rejected/insufficient overlaps: "
        f"{result.failed_pairs}"
    )

    print_gain_summary(
        result.gains
    )

    print()
    print(
        f"Gains: "
        f"{args.output.resolve()}"
    )

    print(
        f"Pair measurements: "
        f"{args.measurements_output.resolve()}"
    )


if __name__ == "__main__":
    main()