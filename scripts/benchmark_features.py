from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.features import extract_sift_from_path

DEFAULT_SCALES = (1.0, 0.75, 0.5)
DEFAULT_NFEATURES = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SIFT feature extraction at multiple "
            "image scales and feature-count limits."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing microscope images.",
    )

    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=DEFAULT_SCALES,
        help="Image scales to benchmark.",
    )

    parser.add_argument(
        "--nfeatures",
        nargs="+",
        type=int,
        default=DEFAULT_NFEATURES,
        help=(
            "Maximum SIFT features to retain. "
            "Use 0 for unlimited."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feature_benchmark.csv"),
        help="Detailed benchmark CSV destination.",
    )

    return parser.parse_args()


def benchmark_configuration(
    image_paths: list[Path],
    scale: float,
    nfeatures: int,
) -> list[dict]:
    records = []

    feature_label = (
        "unlimited"
        if nfeatures == 0
        else str(nfeatures)
    )

    print()
    print(
        f"Benchmarking scale={scale:.2f}, "
        f"nfeatures={feature_label}"
    )

    for image_path in tqdm(
        image_paths,
        desc=f"SIFT {scale:.2f} / {feature_label}",
    ):
        start = perf_counter()

        keypoints, descriptors, image_size = extract_sift_from_path(
            image_path,
            scale=scale,
            nfeatures=nfeatures,
        )

        elapsed = perf_counter() - start

        width, height = image_size

        descriptor_count = (
            0
            if descriptors is None
            else len(descriptors)
        )

        records.append(
            {
                "filename": image_path.name,
                "scale": scale,
                "nfeatures": nfeatures,
                "width": width,
                "height": height,
                "keypoints": len(keypoints),
                "descriptors": descriptor_count,
                "elapsed_seconds": elapsed,
            }
        )

    return records


def print_summary(dataframe: pd.DataFrame) -> None:
    print()
    print("=" * 100)
    print("SIFT FEATURE EXTRACTION BENCHMARK")
    print("=" * 100)

    summary = (
        dataframe.groupby(
            ["scale", "nfeatures"]
        )
        .agg(
            width=("width", "first"),
            height=("height", "first"),
            total_time_s=("elapsed_seconds", "sum"),
            mean_time_s=("elapsed_seconds", "mean"),
            total_keypoints=("keypoints", "sum"),
            mean_keypoints=("keypoints", "mean"),
            min_keypoints=("keypoints", "min"),
            max_keypoints=("keypoints", "max"),
            total_descriptors=("descriptors", "sum"),
        )
        .reset_index()
    )

    summary["descriptor_memory_mib"] = (
        summary["total_descriptors"]
        * 128
        * 4
        / (1024**2)
    )

    summary["total_time_s"] = (
        summary["total_time_s"].round(2)
    )
    summary["mean_time_s"] = (
        summary["mean_time_s"].round(3)
    )
    summary["mean_keypoints"] = (
        summary["mean_keypoints"].round(1)
    )
    summary["descriptor_memory_mib"] = (
        summary["descriptor_memory_mib"].round(1)
    )

    print(summary.to_string(index=False))

    print("=" * 100)


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser().resolve()

    image_paths = discover_images(dataset_path)

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in: {dataset_path}"
        )

    print(f"Dataset: {dataset_path}")
    print(f"Images: {len(image_paths)}")

    all_records = []

    total_start = perf_counter()

    for scale in args.scales:
        for nfeatures in args.nfeatures:
            all_records.extend(
                benchmark_configuration(
                    image_paths,
                    scale,
                    nfeatures,
                )
            )

    total_elapsed = perf_counter() - total_start

    dataframe = pd.DataFrame(all_records)

    print_summary(dataframe)

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
        f"Complete benchmark runtime: "
        f"{total_elapsed:.2f} seconds"
    )
    print(
        f"Detailed results saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()