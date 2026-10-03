from __future__ import annotations

import cv2
import numpy as np


def _downsample_size(
    width: int,
    height: int,
    factor: int,
) -> tuple[int, int]:
    if factor < 1:
        raise ValueError(
            "downsample_factor must be >= 1."
        )

    return (
        max(1, round(width / factor)),
        max(1, round(height / factor)),
    )


def estimate_flatfield(
    image: np.ndarray,
    mask: np.ndarray,
    downsample_factor: int = 8,
    blur_sigma_small: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a smooth luminance-only illumination field.

    The same field will later be applied to all colour channels,
    preserving the source image's colour balance.
    """
    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "Expected a three-channel BGR image."
        )

    if mask.shape != image.shape[:2]:
        raise ValueError(
            "Mask dimensions must match image dimensions."
        )

    if blur_sigma_small <= 0.0:
        raise ValueError(
            "blur_sigma_small must be positive."
        )

    height, width = image.shape[:2]

    (
        small_width,
        small_height,
    ) = _downsample_size(
        width,
        height,
        downsample_factor,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    ) / 255.0

    mask_float = (
        mask.astype(np.float32)
        / 255.0
    )

    gray_small = cv2.resize(
        gray,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    mask_small = cv2.resize(
        mask_float,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    support_small = cv2.GaussianBlur(
        mask_small,
        (0, 0),
        sigmaX=blur_sigma_small,
        sigmaY=blur_sigma_small,
        borderType=cv2.BORDER_REFLECT,
    )

    weighted = (
        gray_small
        * mask_small
    )

    numerator = cv2.GaussianBlur(
        weighted,
        (0, 0),
        sigmaX=blur_sigma_small,
        sigmaY=blur_sigma_small,
        borderType=cv2.BORDER_REFLECT,
    )

    epsilon = 1e-6

    field_small = (
        numerator
        / np.maximum(
            support_small,
            epsilon,
        )
    )

    # Use only reasonably supported locations to establish the
    # reference illumination level.
    reference_values = field_small[
        support_small >= 0.25
    ]

    if reference_values.size == 0:
        raise RuntimeError(
            "No supported pixels available "
            "for flat-field estimation."
        )

    reference = float(
        np.median(
            reference_values
        )
    )

    if reference <= epsilon:
        raise RuntimeError(
            "Flat-field reference intensity "
            "is too small."
        )

    field_small /= reference

    field = cv2.resize(
        field_small,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(
        np.float32
    )

    support = cv2.resize(
        support_small,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(
        np.float32
    )

    # Preserve the previous three-channel interface, but each
    # channel deliberately receives exactly the same field.
    field_bgr = np.repeat(
        field[
            :,
            :,
            None,
        ],
        3,
        axis=2,
    )

    return (
        field_bgr,
        support,
    )


def apply_flatfield(
    image: np.ndarray,
    field: np.ndarray,
    support: np.ndarray,
    blur_sigma_small: float = 30.0,
    min_support: float = 0.05,
    full_support: float = 0.50,
    min_correction: float = 0.95,
    max_correction: float = 1.06,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Apply a conservative luminance flat-field correction.

    Correction strength fades smoothly toward identity where the
    illumination estimate has weak mask support.
    """
    if field.shape != image.shape:
        raise ValueError(
            "Field dimensions must match image dimensions."
        )

    if support.shape != image.shape[:2]:
        raise ValueError(
            "Support dimensions must match image dimensions."
        )

    if (
        min_correction <= 0.0
        or max_correction <= 0.0
        or min_correction > max_correction
    ):
        raise ValueError(
            "Invalid correction bounds."
        )

    if not (
        0.0 <= min_support
        < full_support
        <= 1.0
    ):
        raise ValueError(
            "Require 0 <= min_support "
            "< full_support <= 1."
        )

    epsilon = 1e-6

    raw_correction = (
        1.0
        / np.maximum(
            field[:, :, 0],
            epsilon,
        )
    )

    raw_correction = np.clip(
        raw_correction,
        min_correction,
        max_correction,
    )

    confidence = (
        (support - min_support)
        / (
            full_support
            - min_support
        )
    )

    confidence = np.clip(
        confidence,
        0.0,
        1.0,
    )

    # Smoothstep: removes any visible transition at the support
    # thresholds.
    confidence = (
        confidence
        * confidence
        * (
            3.0
            - 2.0 * confidence
        )
    )

    blended_correction = (
        1.0
        + confidence
        * (
            raw_correction
            - 1.0
        )
    ).astype(
        np.float32
    )

    correction = np.repeat(
        blended_correction[
            :,
            :,
            None,
        ],
        3,
        axis=2,
    )

    corrected = (
        image.astype(np.float32)
        * correction
    )

    corrected = np.clip(
        corrected,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return (
        corrected,
        correction,
    )


def correct_image_flatfield(
    image: np.ndarray,
    mask: np.ndarray,
    downsample_factor: int = 8,
    blur_sigma_small: float = 20.0,
    min_support: float = 0.05,
    full_support: float = 0.50,
    min_correction: float = 0.90,
    max_correction: float = 1.12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    (
        field,
        support,
    ) = estimate_flatfield(
        image,
        mask,
        downsample_factor=(
            downsample_factor
        ),
        blur_sigma_small=(
            blur_sigma_small
        ),
    )

    (
        corrected,
        correction,
    ) = apply_flatfield(
        image,
        field,
        support,
        min_correction=(
            min_correction
        ),
        max_correction=(
            max_correction
        ),
        min_support=(
            min_support
        ),
        full_support=(
            full_support
        ),
    )

    return (
        corrected,
        field,
        correction,
        support,
    )