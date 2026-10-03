from __future__ import annotations

import cv2
import numpy as np


def build_content_mask(
    image: np.ndarray,
    min_gray: int = 25,
    max_gray: int = 225,
    min_local_std: float = 5.0,
    texture_window: int = 9,
) -> np.ndarray:
    """
    Build a conservative mask for photometric estimation.

    Retains moderately exposed, textured specimen regions while
    suppressing blank glass/background, clipped regions, and strongly
    saturated blue annotation material.

    Returns a uint8 mask containing 0 or 255.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a three-channel BGR image."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray_float = gray.astype(
        np.float32
    )

    mean = cv2.boxFilter(
        gray_float,
        ddepth=-1,
        ksize=(
            texture_window,
            texture_window,
        ),
        normalize=True,
    )

    mean_squared = cv2.boxFilter(
        gray_float * gray_float,
        ddepth=-1,
        ksize=(
            texture_window,
            texture_window,
        ),
        normalize=True,
    )

    variance = np.maximum(
        mean_squared - mean * mean,
        0.0,
    )

    local_std = np.sqrt(
        variance
    )

    exposure_mask = (
        (gray >= min_gray)
        & (gray <= max_gray)
    )

    texture_mask = (
        local_std >= min_local_std
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Suppress the highly saturated blue annotation/tape visible
    # around portions of these microscope acquisitions. Moderate
    # blue illumination casts are deliberately retained.
    strong_blue_annotation = (
        (hue >= 90)
        & (hue <= 135)
        & (saturation >= 120)
        & (value >= 70)
    )

    valid = (
        exposure_mask
        & texture_mask
        & ~strong_blue_annotation
    )

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    mask[valid] = 255

    return mask