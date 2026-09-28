from __future__ import annotations

import argparse
from pathlib import Path

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

    trusted_edges = edges[
        (
            edges["confidence"]
            == "high"
        )
        & (
            ~edges[
                "consistency_warning"
            ]
        )
    ].copy()

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