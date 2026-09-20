from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_grayscale_image(path: Path) -> np.ndarray:
    """
    Load an image as an 8-bit grayscale array.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def resize_image(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Resize an image while preserving aspect ratio.

    INTER_AREA is well suited for image downsampling.
    """
    if not 0 < scale <= 1.0:
        raise ValueError("Scale must be greater than 0 and at most 1.0.")

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def create_sift(
    nfeatures: int = 0,
) -> cv2.SIFT:
    """
    Create a SIFT detector.

    nfeatures=0 allows OpenCV to retain all detected features.
    """
    return cv2.SIFT.create(
        nfeatures=nfeatures,
    )


def extract_sift_features(
    image: np.ndarray,
    sift: cv2.SIFT,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Detect SIFT keypoints and compute descriptors.
    """
    keypoints, descriptors = sift.detectAndCompute(
        image,
        None,
    )

    return list(keypoints), descriptors


def extract_sift_from_path(
    path: Path,
    scale: float = 1.0,
    nfeatures: int = 0,
) -> tuple[
    list[cv2.KeyPoint],
    np.ndarray | None,
    tuple[int, int],
]:
    """
    Load an image, optionally downsample it, and extract SIFT features.

    Returns:
        keypoints
        descriptors
        processed image size as (width, height)
    """
    image = load_grayscale_image(path)
    image = resize_image(image, scale)

    sift = create_sift(nfeatures=nfeatures)

    keypoints, descriptors = extract_sift_features(
        image,
        sift,
    )

    height, width = image.shape

    return (
        keypoints,
        descriptors,
        (width, height),
    )