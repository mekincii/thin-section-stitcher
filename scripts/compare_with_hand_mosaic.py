from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from thin_section_stitcher.comparison import (
    comparison_images,
    estimate_mosaic_registration,
    load_color_image,
    specimen_mask,
    warp_hand_to_automatic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register an independently hand-stitched thin-section "
            "mosaic to the automatic reconstruction and compare "
            "similarity and affine global registration models."
        )
    )

    parser.add_argument(
        "--auto",
        type=Path,
        default=Path(
            "outputs/raw_mosaic_preview.png"
        ),
        help=(
            "Automatic mosaic used as the reference coordinate system."
        ),
    )

    parser.add_argument(
        "--hand",
        type=Path,
        required=True,
        help=(
            "Independently hand-stitched mosaic."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/hand_mosaic_comparison"
        ),
    )

    parser.add_argument(
        "--working-max-dim",
        type=int,
        default=2500,
        help=(
            "Maximum image dimension used during SIFT registration."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "similarity",
            "affine",
            "both",
        ],
        default="both",
        help=(
            "Global registration model to evaluate."
        ),
    )

    return parser.parse_args()


def save_image(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(path),
        image,
    ):
        raise RuntimeError(
            f"Could not save: {path}"
        )


def run_model(
    model: str,
    automatic: np.ndarray,
    hand: np.ndarray,
    automatic_mask: np.ndarray,
    hand_mask: np.ndarray,
    output_root: Path,
    working_max_dim: int,
) -> dict:
    print()
    print("-" * 72)
    print(
        f"MODEL: {model.upper()}"
    )
    print("-" * 72)

    registration = (
        estimate_mosaic_registration(
            automatic,
            hand,
            max_dimension=(
                working_max_dim
            ),
            model=model,
        )
    )

    (
        registered_hand,
        registered_hand_mask,
    ) = warp_hand_to_automatic(
        hand,
        hand_mask,
        registration.transform_hand_to_auto,
        automatic.shape,
    )

    (
        overlay,
        difference,
        edge_overlay,
        overlap_mask,
    ) = comparison_images(
        automatic,
        registered_hand,
        automatic_mask,
        registered_hand_mask,
    )

    overlap_pixels = int(
        np.count_nonzero(
            overlap_mask
        )
    )

    automatic_pixels = int(
        np.count_nonzero(
            automatic_mask
        )
    )

    registered_hand_pixels = int(
        np.count_nonzero(
            registered_hand_mask
        )
    )

    automatic_overlap_fraction = (
        overlap_pixels
        / automatic_pixels
        if automatic_pixels
        else 0.0
    )

    hand_overlap_fraction = (
        overlap_pixels
        / registered_hand_pixels
        if registered_hand_pixels
        else 0.0
    )

    output_dir = (
        output_root
        / model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        output_dir
        / "registered_hand.png",
        registered_hand,
    )

    save_image(
        output_dir
        / "registered_hand_mask.png",
        registered_hand_mask,
    )

    save_image(
        output_dir
        / "overlay_50_50.png",
        overlay,
    )

    save_image(
        output_dir
        / "difference_structure.png",
        difference,
    )

    save_image(
        output_dir
        / "edges_overlay.png",
        edge_overlay,
    )

    save_image(
        output_dir
        / "overlap_mask.png",
        overlap_mask,
    )

    match_preview = cv2.drawMatches(
        registration.hand_working,
        registration.keypoints_hand,
        registration.auto_working,
        registration.keypoints_auto,
        registration.inlier_matches[
            :150
        ],
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )

    save_image(
        output_dir
        / "inlier_matches.png",
        match_preview,
    )

    np.savetxt(
        output_dir
        / "transform_hand_to_auto.csv",
        registration.transform_hand_to_auto,
        delimiter=",",
        fmt="%.12f",
    )

    print(
        f"Mutual SIFT matches: "
        f"{registration.mutual_matches}"
    )

    print(
        f"RANSAC inliers: "
        f"{registration.inliers}"
    )

    print(
        f"Inlier ratio: "
        f"{registration.inlier_ratio:.3f}"
    )

    print(
        f"Median inlier residual: "
        f"{registration.median_inlier_error_px:.2f} "
        f"automatic-mosaic px"
    )

    print(
        f"Rotation: "
        f"{registration.rotation_deg:.3f}°"
    )

    print(
        f"Scale X: "
        f"{registration.scale_x:.5f}"
    )

    print(
        f"Scale Y: "
        f"{registration.scale_y:.5f}"
    )

    print(
        f"Equivalent scale: "
        f"{registration.equivalent_scale:.5f}"
    )

    print(
        f"Scale anisotropy: "
        f"{registration.anisotropy_ratio:.5f}"
    )

    print(
        f"Basis angle: "
        f"{registration.basis_angle_deg:.3f}°"
    )

    print(
        f"Shear departure from 90°: "
        f"{registration.shear_from_orthogonal_deg:.3f}°"
    )

    print(
        f"Automatic specimen overlap: "
        f"{automatic_overlap_fraction:.1%}"
    )

    print(
        f"Registered hand specimen overlap: "
        f"{hand_overlap_fraction:.1%}"
    )

    print(
        f"Outputs: "
        f"{output_dir.resolve()}"
    )

    return {
        "model": model,
        "mutual_matches": (
            registration.mutual_matches
        ),
        "inliers": (
            registration.inliers
        ),
        "inlier_ratio": (
            registration.inlier_ratio
        ),
        "median_error_px": (
            registration.median_inlier_error_px
        ),
        "rotation_deg": (
            registration.rotation_deg
        ),
        "scale_x": (
            registration.scale_x
        ),
        "scale_y": (
            registration.scale_y
        ),
        "equivalent_scale": (
            registration.equivalent_scale
        ),
        "anisotropy_ratio": (
            registration.anisotropy_ratio
        ),
        "basis_angle_deg": (
            registration.basis_angle_deg
        ),
        "shear_deg": (
            registration.shear_from_orthogonal_deg
        ),
        "automatic_overlap": (
            automatic_overlap_fraction
        ),
        "hand_overlap": (
            hand_overlap_fraction
        ),
    }


def print_model_comparison(
    results: list[dict],
) -> None:
    print()
    print("=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)

    for result in results:
        print(
            f"{result['model']:10s} "
            f"inliers={result['inliers']:4d}  "
            f"ratio={result['inlier_ratio']:.3f}  "
            f"median={result['median_error_px']:.2f}px  "
            f"anisotropy={result['anisotropy_ratio']:.5f}  "
            f"shear={result['shear_deg']:.3f}°"
        )


def main() -> None:
    args = parse_args()

    automatic_path = (
        args.auto
        .expanduser()
        .resolve()
    )

    hand_path = (
        args.hand
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    automatic = load_color_image(
        automatic_path
    )

    hand = load_color_image(
        hand_path
    )

    automatic_mask = specimen_mask(
        automatic
    )

    hand_mask = specimen_mask(
        hand
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        output_dir
        / "automatic_mask.png",
        automatic_mask,
    )

    save_image(
        output_dir
        / "hand_mask_original.png",
        hand_mask,
    )

    print("=" * 72)
    print("HAND VS AUTOMATIC MOSAIC COMPARISON")
    print("=" * 72)

    print(
        f"Automatic mosaic: "
        f"{automatic.shape[1]} x "
        f"{automatic.shape[0]}"
    )

    print(
        f"Hand mosaic: "
        f"{hand.shape[1]} x "
        f"{hand.shape[0]}"
    )

    print(
        f"Working maximum dimension: "
        f"{args.working_max_dim}"
    )

    models = (
        [
            "similarity",
            "affine",
        ]
        if args.model == "both"
        else [
            args.model
        ]
    )

    results = []

    for model in models:
        result = run_model(
            model=model,
            automatic=automatic,
            hand=hand,
            automatic_mask=(
                automatic_mask
            ),
            hand_mask=(
                hand_mask
            ),
            output_root=output_dir,
            working_max_dim=(
                args.working_max_dim
            ),
        )

        results.append(
            result
        )

    comparison_table = pd.DataFrame(
        results
    )

    comparison_path = (
        output_dir
        / "model_comparison.csv"
    )

    comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    print_model_comparison(
        results
    )

    print()
    print(
        f"Comparison table: "
        f"{comparison_path}"
    )

    print(
        f"All outputs: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()