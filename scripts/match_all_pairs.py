from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.features import extract_sift_from_path
from thin_section_stitcher.matching import (
    DiscoveryCriteria,
    is_plausible_overlap,
    match_descriptors,
    verify_geometry,
)

DISCOVERY_SCALE = 0.5
DISCOVERY_FEATURES = 4000

VERIFICATION_SCALE = 0.75
VERIFICATION_FEATURES = 8000

LOWE_RATIO = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover possible overlaps among all microscope "
            "image pairs and verify promising candidates."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing microscope images.",
    )

    parser.add_argument(
        "--discovery-output",
        type=Path,
        default=Path("outputs/all_pair_discovery.csv"),
    )

    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path("outputs/candidate_verification.csv"),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save discovery progress every N newly processed pairs.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing discovery checkpoint and start again.",
    )

    return parser.parse_args()


def extract_feature_cache(
    image_paths: list[Path],
    scale: float,
    nfeatures: int,
    label: str,
) -> dict[str, dict]:
    cache = {}

    for image_path in tqdm(
        image_paths,
        desc=label,
    ):
        keypoints, descriptors, image_size = extract_sift_from_path(
            image_path,
            scale=scale,
            nfeatures=nfeatures,
        )

        cache[image_path.name] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_size": image_size,
        }

    return cache


def evaluate_pair(
    image_a: str,
    image_b: str,
    feature_cache: dict[str, dict],
    scale: float,
    nfeatures: int,
) -> dict:
    start = perf_counter()

    features_a = feature_cache[image_a]
    features_b = feature_cache[image_b]

    matches = match_descriptors(
        features_a["descriptors"],
        features_b["descriptors"],
        ratio=LOWE_RATIO,
    )

    geometry = verify_geometry(
        features_a["keypoints"],
        features_b["keypoints"],
        matches.mutual_matches,
    )

    elapsed = perf_counter() - start

    return {
        "image_a": image_a,
        "image_b": image_b,
        "scale": scale,
        "nfeatures": nfeatures,
        "ratio_threshold": LOWE_RATIO,
        "forward_good": matches.forward_count,
        "backward_good": matches.backward_count,
        "mutual_matches": matches.mutual_count,
        "median_distance": matches.median_distance,
        "ransac_inliers": geometry.inliers,
        "inlier_ratio": geometry.inlier_ratio,
        "dx": geometry.dx,
        "dy": geometry.dy,
        "rotation_deg": geometry.rotation_deg,
        "estimated_scale": geometry.scale,
        "match_time_s": elapsed,
        "_match_result": matches,
        "_geometry_result": geometry,
    }


def save_discovery_records(
    records: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        path,
        index=False,
    )


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    image_paths = discover_images(dataset_path)

    if len(image_paths) < 2:
        raise RuntimeError(
            "At least two images are required."
        )

    path_by_name = {
        path.name: path
        for path in image_paths
    }

    total_pairs = (
        len(image_paths)
        * (len(image_paths) - 1)
        // 2
    )

    criteria = DiscoveryCriteria()

    print(f"Dataset: {dataset_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Possible pairs: {total_pairs}")

    print()
    print("DISCOVERY")
    print(
        f"scale={DISCOVERY_SCALE}, "
        f"features={DISCOVERY_FEATURES}"
    )

    discovery_cache = extract_feature_cache(
        image_paths,
        scale=DISCOVERY_SCALE,
        nfeatures=DISCOVERY_FEATURES,
        label="Extracting discovery features",
    )

    records: list[dict] = []
    processed_pairs: set[tuple[str, str]] = set()

    if (
        args.discovery_output.exists()
        and not args.restart
    ):
        existing = pd.read_csv(
            args.discovery_output
        )

        records = existing.to_dict(
            orient="records"
        )

        processed_pairs = {
            (row["image_a"], row["image_b"])
            for row in records
        }

        print()
        print(
            f"Resuming from "
            f"{len(processed_pairs)} processed pairs."
        )

    new_pairs_processed = 0

    pair_iterator = combinations(
        image_paths,
        2,
    )

    print()
    print("Comparing all pairs...")

    for path_a, path_b in tqdm(
        pair_iterator,
        total=total_pairs,
        desc="Discovery matching",
    ):
        pair_key = (
            path_a.name,
            path_b.name,
        )

        if pair_key in processed_pairs:
            continue

        result = evaluate_pair(
            path_a.name,
            path_b.name,
            discovery_cache,
            scale=DISCOVERY_SCALE,
            nfeatures=DISCOVERY_FEATURES,
        )

        candidate = is_plausible_overlap(
            result["_match_result"],
            result["_geometry_result"],
            criteria,
        )

        result["candidate"] = candidate

        result.pop("_match_result")
        result.pop("_geometry_result")

        records.append(result)

        new_pairs_processed += 1

        if (
            new_pairs_processed
            % args.checkpoint_every
            == 0
        ):
            save_discovery_records(
                records,
                args.discovery_output,
            )

    save_discovery_records(
        records,
        args.discovery_output,
    )

    discovery_df = pd.DataFrame(records)

    discovery_df["candidate"] = (
        (
            discovery_df["mutual_matches"]
            >= criteria.min_mutual_matches
        )
        & (
            discovery_df["ransac_inliers"]
            >= criteria.min_inliers
        )
        & (
            discovery_df["inlier_ratio"]
            >= criteria.min_inlier_ratio
        )
        & discovery_df["estimated_scale"].between(
            criteria.min_scale,
            criteria.max_scale,
        )
    )

    discovery_df.to_csv(
        args.discovery_output,
        index=False,
    )

    candidate_df = discovery_df[
        discovery_df["candidate"] == True
    ].copy()

    candidate_df = candidate_df.sort_values(
        [
            "ransac_inliers",
            "inlier_ratio",
            "mutual_matches",
        ],
        ascending=False,
    )

    print()
    print("=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"Pairs examined: {len(discovery_df)}")
    print(f"Candidates: {len(candidate_df)}")

    if candidate_df.empty:
        print("No candidate overlaps found.")
        return

    print()
    print("Strongest discovery candidates:")
    print(
        candidate_df[
            [
                "image_a",
                "image_b",
                "mutual_matches",
                "ransac_inliers",
                "inlier_ratio",
                "rotation_deg",
                "estimated_scale",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    candidate_names = sorted(
        set(candidate_df["image_a"])
        | set(candidate_df["image_b"])
    )

    verification_paths = [
        path_by_name[name]
        for name in candidate_names
    ]

    print()
    print("VERIFICATION")
    print(
        f"scale={VERIFICATION_SCALE}, "
        f"features={VERIFICATION_FEATURES}"
    )
    print(
        f"Images involved in candidates: "
        f"{len(verification_paths)}"
    )

    verification_cache = extract_feature_cache(
        verification_paths,
        scale=VERIFICATION_SCALE,
        nfeatures=VERIFICATION_FEATURES,
        label="Extracting verification features",
    )

    verification_records = []

    for row in tqdm(
        candidate_df.itertuples(index=False),
        total=len(candidate_df),
        desc="Verifying candidates",
    ):
        result = evaluate_pair(
            row.image_a,
            row.image_b,
            verification_cache,
            scale=VERIFICATION_SCALE,
            nfeatures=VERIFICATION_FEATURES,
        )

        result.pop("_match_result")
        result.pop("_geometry_result")

        verification_records.append(
            result
        )

    verification_df = pd.DataFrame(
        verification_records
    )

    verification_df = verification_df.sort_values(
        [
            "ransac_inliers",
            "inlier_ratio",
            "mutual_matches",
        ],
        ascending=False,
    )

    args.verification_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    verification_df.to_csv(
        args.verification_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(
        verification_df[
            [
                "image_a",
                "image_b",
                "mutual_matches",
                "ransac_inliers",
                "inlier_ratio",
                "rotation_deg",
                "estimated_scale",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print(
        f"Discovery results: "
        f"{args.discovery_output.resolve()}"
    )
    print(
        f"Verification results: "
        f"{args.verification_output.resolve()}"
    )


if __name__ == "__main__":
    main()