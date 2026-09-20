from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thin_section_stitcher.matching import (
    compare_cross_scale_geometry,
)

DISCOVERY_SCALE = 0.5
VERIFICATION_SCALE = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare discovery and verification geometry "
            "for candidate image overlaps."
        )
    )

    parser.add_argument(
        "--discovery",
        type=Path,
        default=Path("outputs/all_pair_discovery.csv"),
    )

    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("outputs/candidate_verification.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/match_consistency.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    discovery = pd.read_csv(args.discovery)
    verification = pd.read_csv(args.verification)

    discovery = discovery[
        discovery["candidate"] == True
    ].copy()

    merged = discovery.merge(
        verification,
        on=["image_a", "image_b"],
        suffixes=("_discovery", "_verification"),
        validate="one_to_one",
    )

    consistency_records = []

    for row in merged.itertuples(index=False):
        consistency = compare_cross_scale_geometry(
            dx_a=row.dx_discovery,
            dy_a=row.dy_discovery,
            rotation_a=row.rotation_deg_discovery,
            scale_a=row.estimated_scale_discovery,
            image_scale_a=DISCOVERY_SCALE,
            dx_b=row.dx_verification,
            dy_b=row.dy_verification,
            rotation_b=row.rotation_deg_verification,
            scale_b=row.estimated_scale_verification,
            image_scale_b=VERIFICATION_SCALE,
        )

        consistency_records.append(
            {
                "translation_error_px": (
                    consistency.translation_error_px
                ),
                "rotation_error_deg": (
                    consistency.rotation_error_deg
                ),
                "scale_error": consistency.scale_error,
            }
        )

    consistency_df = pd.DataFrame(
        consistency_records
    )

    merged = pd.concat(
        [
            merged.reset_index(drop=True),
            consistency_df,
        ],
        axis=1,
    )

    merged = merged.sort_values(
        [
            "translation_error_px",
            "rotation_error_deg",
            "ransac_inliers_verification",
        ],
        ascending=[True, True, False],
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        args.output,
        index=False,
    )

    print()
    print("=" * 72)
    print("CROSS-SCALE GEOMETRY CONSISTENCY")
    print("=" * 72)
    print(f"Candidates compared: {len(merged)}")

    print()
    print("Translation error in original-image pixels:")
    print(
        merged["translation_error_px"]
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
    print("Rotation error (degrees):")
    print(
        merged["rotation_error_deg"]
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
    print("Scale error:")
    print(
        merged["scale_error"]
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
    print("Most geometrically consistent pairs:")
    print(
        merged[
            [
                "image_a",
                "image_b",
                "ransac_inliers_verification",
                "inlier_ratio_verification",
                "translation_error_px",
                "rotation_error_deg",
                "scale_error",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Least geometrically consistent pairs:")
    print(
        merged[
            [
                "image_a",
                "image_b",
                "ransac_inliers_verification",
                "inlier_ratio_verification",
                "translation_error_px",
                "rotation_error_deg",
                "scale_error",
            ]
        ]
        .tail(20)
        .to_string(index=False)
    )

    print()
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()