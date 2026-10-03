from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from thin_section_stitcher.photometric import (
    layout_transforms,
    load_scaled_images,
    measure_pair_photometry,
    solve_channel_gains,
)
from thin_section_stitcher.photometric_masking import (
    build_content_mask,
)


@dataclass(slots=True)
class ContentAwarePhotometricResult:
    gains: pd.DataFrame
    measurements: pd.DataFrame
    mask_summary: pd.DataFrame
    successful_pairs: int
    failed_pairs: int


def estimate_content_aware_photometric_gains(
    image_paths: list[Path],
    layout: pd.DataFrame,
    trusted_edges: pd.DataFrame,
    scale: float = 0.25,
    root: str = "60.jpg",
    max_samples_per_pair: int = 100_000,
) -> ContentAwarePhotometricResult:
    """
    Estimate per-image B/G/R gains using only content-aware
    masked pixels in geometrically trusted overlaps.

    The normal photometric measurement machinery is reused after
    rejected source pixels are set to zero. The generic pipeline
    therefore remains unchanged.
    """
    (
        scaled_images,
        original_width,
        original_height,
    ) = load_scaled_images(
        image_paths,
        scale=scale,
    )

    transforms = layout_transforms(
        layout,
        image_width=original_width,
        image_height=original_height,
    )

    masked_images: dict[str, np.ndarray] = {}
    mask_coverages: dict[str, float] = {}

    for image_name, image in scaled_images.items():
        mask = build_content_mask(
            image
        )

        masked = image.copy()
        masked[mask == 0] = 0

        masked_images[image_name] = masked

        mask_coverages[image_name] = float(
            np.count_nonzero(mask)
            / mask.size
        )

    image_names = [
        path.name
        for path in image_paths
    ]

    measurements = []
    failed_pairs = 0

    for edge in tqdm(
        trusted_edges.to_dict(
            orient="records"
        ),
        desc="Measuring masked overlap illumination",
    ):
        image_a = str(
            edge["image_a"]
        )

        image_b = str(
            edge["image_b"]
        )

        if (
            image_a not in masked_images
            or image_b not in masked_images
            or image_a not in transforms
            or image_b not in transforms
        ):
            failed_pairs += 1
            continue

        measurement = measure_pair_photometry(
            masked_images[image_a],
            masked_images[image_b],
            transforms[image_a],
            transforms[image_b],
            original_width=original_width,
            original_height=original_height,
            render_scale=scale,
            image_a_name=image_a,
            image_b_name=image_b,
            geometric_quality=float(
                edge[
                    "inlier_ratio_verification"
                ]
            ),
            max_samples=max_samples_per_pair,
        )

        if measurement is None:
            failed_pairs += 1
            continue

        measurements.append(
            measurement
        )

    if not measurements:
        raise RuntimeError(
            "No usable masked photometric measurements."
        )

    gains_blue = solve_channel_gains(
        image_names,
        measurements,
        "log_ratio_blue",
        root=root,
    )

    gains_green = solve_channel_gains(
        image_names,
        measurements,
        "log_ratio_green",
        root=root,
    )

    gains_red = solve_channel_gains(
        image_names,
        measurements,
        "log_ratio_red",
        root=root,
    )

    gains = pd.DataFrame(
        [
            {
                "image": image_name,
                "gain_blue": gains_blue[
                    image_name
                ],
                "gain_green": gains_green[
                    image_name
                ],
                "gain_red": gains_red[
                    image_name
                ],
            }
            for image_name in image_names
        ]
    )

    measurements_frame = pd.DataFrame(
        [
            {
                "image_a": measurement.image_a,
                "image_b": measurement.image_b,
                "log_ratio_blue": (
                    measurement.log_ratio_blue
                ),
                "log_ratio_green": (
                    measurement.log_ratio_green
                ),
                "log_ratio_red": (
                    measurement.log_ratio_red
                ),
                "valid_pixels": (
                    measurement.valid_pixels
                ),
                "sampled_pixels": (
                    measurement.sampled_pixels
                ),
                "geometric_quality": (
                    measurement.geometric_quality
                ),
                "centroid_ax": (
                    measurement.centroid_ax
                ),
                "centroid_ay": (
                    measurement.centroid_ay
                ),
                "centroid_bx": (
                    measurement.centroid_bx
                ),
                "centroid_by": (
                    measurement.centroid_by
                ),
                "mask_coverage_a": (
                    mask_coverages[
                        measurement.image_a
                    ]
                ),
                "mask_coverage_b": (
                    mask_coverages[
                        measurement.image_b
                    ]
                ),
            }
            for measurement in measurements
        ]
    )

    mask_summary = pd.DataFrame(
        [
            {
                "image": image_name,
                "retained_fraction": (
                    mask_coverages[
                        image_name
                    ]
                ),
            }
            for image_name in image_names
        ]
    )

    return ContentAwarePhotometricResult(
        gains=gains,
        measurements=measurements_frame,
        mask_summary=mask_summary,
        successful_pairs=len(
            measurements
        ),
        failed_pairs=failed_pairs,
    )