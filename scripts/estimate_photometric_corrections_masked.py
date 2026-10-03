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
from thin_section_stitcher.photometric_content_aware import (
    estimate_content_aware_photometric_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental content-aware photometric "
            "standardization."
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
        "--validated-bridges",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--include-medium",
        action="store_true",
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
        "--output",
        type=Path,
        default=Path(
            "outputs/photometric_masked_gains.csv"
        ),
    )

    parser.add_argument(
        "--measurements-output",
        type=Path,
        default=Path(
            "outputs/photometric_masked_measurements.csv"
        ),
    )

    parser.add_argument(
        "--mask-summary-output",
        type=Path,
        default=Path(
            "outputs/photometric_mask_summary.csv"
        ),
    )

    return parser.parse_args()


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

    confidence_levels = {
        "high"
    }

    if args.include_medium:
        confidence_levels.add(
            "medium"
        )

    trusted_edges = edges[
        edges["confidence"].isin(
            confidence_levels
        )
        & ~edges[
            "consistency_warning"
        ]
    ].copy()

    if args.validated_bridges is not None:
        bridges = pd.read_csv(
            args.validated_bridges
        )

        existing_pairs = {
            frozenset(
                (
                    str(row.image_a),
                    str(row.image_b),
                )
            )
            for row in trusted_edges.itertuples()
        }

        bridge_records = []

        for row in bridges.itertuples(
            index=False
        ):
            pair = frozenset(
                (
                    str(row.image_a),
                    str(row.image_b),
                )
            )

            if pair in existing_pairs:
                continue

            bridge_records.append(
                {
                    "image_a": str(
                        row.image_a
                    ),
                    "image_b": str(
                        row.image_b
                    ),
                    "inlier_ratio_verification": (
                        float(
                            row.inlier_ratio
                        )
                    ),
                }
            )

        if bridge_records:
            trusted_edges = pd.concat(
                [
                    trusted_edges,
                    pd.DataFrame(
                        bridge_records
                    ),
                ],
                ignore_index=True,
                sort=False,
            )

    graph = nx.Graph()

    graph.add_nodes_from(
        path.name
        for path in image_paths
    )

    graph.add_edges_from(
        (
            str(row["image_a"]),
            str(row["image_b"]),
        )
        for row in trusted_edges.to_dict(
            orient="records"
        )
    )

    if not nx.is_connected(
        graph
    ):
        components = sorted(
            nx.connected_components(
                graph
            ),
            key=len,
            reverse=True,
        )

        raise RuntimeError(
            "Masked photometric graph is not connected. "
            f"Component sizes: "
            f"{[len(component) for component in components]}"
        )

    print("=" * 72)
    print("CONTENT-AWARE PHOTOMETRIC ESTIMATION")
    print("=" * 72)

    print(
        f"Images: {len(image_paths)}"
    )

    print(
        "Trusted overlap constraints: "
        f"{len(trusted_edges)}"
    )

    print(
        f"Estimation scale: "
        f"{args.scale:.3f}"
    )

    result = (
        estimate_content_aware_photometric_gains(
            image_paths,
            layout,
            trusted_edges,
            scale=args.scale,
            root=args.root,
        )
    )

    args.output.parent.mkdir(
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

    result.mask_summary.to_csv(
        args.mask_summary_output,
        index=False,
    )

    print()
    print("=" * 72)
    print("ESTIMATION COMPLETE")
    print("=" * 72)

    print(
        "Successful overlap measurements: "
        f"{result.successful_pairs}"
    )

    print(
        "Rejected/insufficient overlaps: "
        f"{result.failed_pairs}"
    )

    retained = (
        result.mask_summary[
            "retained_fraction"
        ]
    )

    print()
    print("MASK RETENTION")
    print(
        f"min={retained.min():.1%}  "
        f"median={retained.median():.1%}  "
        f"max={retained.max():.1%}"
    )

    print()
    print("GAIN SUMMARY")

    for column in [
        "gain_blue",
        "gain_green",
        "gain_red",
    ]:
        values = result.gains[
            column
        ]

        print(
            f"{column:12s} "
            f"min={values.min():.4f}  "
            f"median={values.median():.4f}  "
            f"max={values.max():.4f}"
        )

    working = result.gains.copy()

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
        "Largest corrections:"
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

    print()
    print(
        f"Gains: "
        f"{args.output.resolve()}"
    )

    print(
        f"Measurements: "
        f"{args.measurements_output.resolve()}"
    )

    print(
        f"Mask summary: "
        f"{args.mask_summary_output.resolve()}"
    )


if __name__ == "__main__":
    main()