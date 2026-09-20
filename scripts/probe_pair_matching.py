from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.features import extract_sift_from_path
from thin_section_stitcher.matching import (
    match_descriptors,
    verify_geometry,
)

DEFAULT_PAIRS = (
    "1:2",
    "20:21",
    "100:101",
    "1:100",
    "20:150",
    "50:170",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe SIFT matching behaviour on selected image pairs."
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing microscope images.",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Pairs written as IMAGE_A:IMAGE_B.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--nfeatures",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/pair_matching_probe.csv"
        ),
    )

    return parser.parse_args()


def parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":")
        return int(left), int(right)
    except ValueError as exc:
        raise ValueError(
            f"Invalid pair '{value}'. Expected format such as 20:21."
        ) from exc


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser().resolve()

    image_paths = discover_images(dataset_path)

    numeric_images = {
        int(path.stem): path
        for path in image_paths
        if path.stem.isdigit()
    }

    pairs = [
        parse_pair(value)
        for value in args.pairs
    ]

    required_ids = {
        image_id
        for pair in pairs
        for image_id in pair
    }

    missing = sorted(
        required_ids - numeric_images.keys()
    )

    if missing:
        raise RuntimeError(
            "Requested images not found: "
            + ", ".join(map(str, missing))
        )

    print(f"Dataset: {dataset_path}")
    print(
        f"Configuration: scale={args.scale}, "
        f"nfeatures={args.nfeatures}, "
        f"ratio={args.ratio}"
    )

    print()
    print("Extracting features...")

    feature_cache = {}

    for image_id in sorted(required_ids):
        keypoints, descriptors, image_size = (
            extract_sift_from_path(
                numeric_images[image_id],
                scale=args.scale,
                nfeatures=args.nfeatures,
            )
        )

        feature_cache[image_id] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_size": image_size,
        }

        print(
            f"{image_id:>4}: "
            f"{len(keypoints):>5} keypoints"
        )

    records = []

    print()
    print("Matching pairs...")

    for image_a, image_b in pairs:
        start = perf_counter()

        result = match_descriptors(
            feature_cache[image_a]["descriptors"],
            feature_cache[image_b]["descriptors"],
            ratio=args.ratio,
        )

        geometry = verify_geometry(
            feature_cache[image_a]["keypoints"],
            feature_cache[image_b]["keypoints"],
            result.mutual_matches,
        )

        elapsed = perf_counter() - start

        records.append(
            {
                "image_a": image_a,
                "image_b": image_b,
                "scale": args.scale,
                "nfeatures": args.nfeatures,
                "forward_good": result.forward_count,
                "backward_good": result.backward_count,
                "mutual_matches": result.mutual_count,
                "median_distance": result.median_distance,
                "ransac_inliers": geometry.inliers,
                "inlier_ratio": geometry.inlier_ratio,
                "dx": geometry.dx,
                "dy": geometry.dy,
                "rotation_deg": geometry.rotation_deg,
                "estimated_scale": geometry.scale,
                "match_time_s": elapsed,
            }
        )

    dataframe = pd.DataFrame(records)

    print()
    print("=" * 95)
    print("PAIR MATCHING PROBE")
    print("=" * 95)
    print(
        dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print("=" * 95)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Results saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()