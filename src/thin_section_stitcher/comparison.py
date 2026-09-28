from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path

import cv2
import numpy as np

from thin_section_stitcher.features import create_sift
from thin_section_stitcher.matching import match_descriptors


@dataclass(slots=True)
class MosaicRegistration:
    """
    Result of registering an independently stitched mosaic to the
    automatic reconstruction.
    """

    model: str
    transform_hand_to_auto: np.ndarray

    auto_working: np.ndarray
    hand_working: np.ndarray
    auto_mask_working: np.ndarray
    hand_mask_working: np.ndarray

    keypoints_auto: list[cv2.KeyPoint]
    keypoints_hand: list[cv2.KeyPoint]
    inlier_matches: list[cv2.DMatch]

    mutual_matches: int
    inliers: int
    inlier_ratio: float

    rotation_deg: float

    scale_x: float
    scale_y: float
    equivalent_scale: float
    anisotropy_ratio: float

    basis_angle_deg: float
    shear_from_orthogonal_deg: float

    median_inlier_error_px: float


def load_color_image(
    path: Path,
) -> np.ndarray:
    """
    Load a colour image.
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


def corner_background_color(
    image: np.ndarray,
    fraction: float = 0.05,
) -> np.ndarray:
    """
    Estimate the background colour from the four image corners.
    """
    height, width = image.shape[:2]

    patch_height = max(
        8,
        int(height * fraction),
    )

    patch_width = max(
        8,
        int(width * fraction),
    )

    patches = [
        image[
            :patch_height,
            :patch_width,
        ],
        image[
            :patch_height,
            width - patch_width:,
        ],
        image[
            height - patch_height:,
            :patch_width,
        ],
        image[
            height - patch_height:,
            width - patch_width:,
        ],
    ]

    pixels = np.concatenate(
        [
            patch.reshape(-1, 3)
            for patch in patches
        ],
        axis=0,
    )

    return np.median(
        pixels,
        axis=0,
    ).astype(np.float32)


def specimen_mask(
    image: np.ndarray,
    threshold: float = 24.0,
) -> np.ndarray:
    """
    Estimate the specimen region using colour distance from the
    corner-derived background.

    Only the largest foreground component is retained. This helps
    remove annotations, labels, and isolated background artefacts.
    """
    background = corner_background_color(
        image
    )

    difference = (
        image.astype(np.float32)
        - background
    )

    distance = np.linalg.norm(
        difference,
        axis=2,
    )

    mask = (
        distance > threshold
    ).astype(np.uint8) * 255

    minimum_dimension = min(
        image.shape[:2]
    )

    kernel_size = max(
        3,
        minimum_dimension // 300,
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count <= 1:
        raise RuntimeError(
            "Could not identify a specimen region."
        )

    largest_label = 1 + int(
        np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA,
            ]
        )
    )

    largest = np.zeros_like(
        mask
    )

    largest[
        labels == largest_label
    ] = 255

    largest = cv2.dilate(
        largest,
        kernel,
        iterations=1,
    )

    return largest


def resize_for_registration(
    image: np.ndarray,
    mask: np.ndarray,
    max_dimension: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Resize an image for mosaic-level registration.

    Returns:
        resized image
        resized specimen mask
        full-resolution -> working-resolution scale matrix
    """
    if max_dimension <= 0:
        raise ValueError(
            "max_dimension must be positive."
        )

    height, width = image.shape[:2]

    scale = (
        max_dimension
        / max(height, width)
    )

    target_width = max(
        1,
        round(width * scale),
    )

    target_height = max(
        1,
        round(height * scale),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized_image = cv2.resize(
        image,
        (
            target_width,
            target_height,
        ),
        interpolation=interpolation,
    )

    resized_mask = cv2.resize(
        mask,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    scale_x = (
        target_width / width
    )

    scale_y = (
        target_height / height
    )

    scale_matrix = np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        resized_image,
        resized_mask,
        scale_matrix,
    )


def registration_gray(
    image: np.ndarray,
) -> np.ndarray:
    """
    Produce a locally contrast-normalized grayscale image for
    mosaic-level SIFT extraction.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        gray
    )


def affine_characteristics(
    transform: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Describe the linear part of a 2D affine transform.

    scale_x and scale_y are the lengths of the transformed basis
    vectors. Their ratio gives a simple measure of anisotropy.

    basis_angle_deg measures the angle between the transformed x/y
    basis vectors. A pure similarity transform has an angle of 90°.
    Departure from 90° therefore provides an intuitive shear measure.
    """
    linear = transform[:2, :2]

    first_axis = linear[:, 0]
    second_axis = linear[:, 1]

    scale_x = float(
        np.linalg.norm(
            first_axis
        )
    )

    scale_y = float(
        np.linalg.norm(
            second_axis
        )
    )

    if scale_x <= 0.0 or scale_y <= 0.0:
        raise RuntimeError(
            "Degenerate affine transformation."
        )

    determinant = float(
        np.linalg.det(
            linear
        )
    )

    equivalent_scale = float(
        np.sqrt(
            abs(determinant)
        )
    )

    anisotropy_ratio = (
        max(
            scale_x,
            scale_y,
        )
        / min(
            scale_x,
            scale_y,
        )
    )

    cosine_angle = float(
        np.dot(
            first_axis,
            second_axis,
        )
        / (
            scale_x
            * scale_y
        )
    )

    cosine_angle = float(
        np.clip(
            cosine_angle,
            -1.0,
            1.0,
        )
    )

    basis_angle_deg = float(
        np.degrees(
            np.arccos(
                cosine_angle
            )
        )
    )

    shear_from_orthogonal_deg = (
        90.0
        - basis_angle_deg
    )

    rotation_deg = degrees(
        atan2(
            float(first_axis[1]),
            float(first_axis[0]),
        )
    )

    return (
        rotation_deg,
        scale_x,
        scale_y,
        equivalent_scale,
        anisotropy_ratio,
        basis_angle_deg,
        shear_from_orthogonal_deg,
    )


def estimate_mosaic_registration(
    automatic: np.ndarray,
    hand: np.ndarray,
    max_dimension: int = 2500,
    nfeatures: int = 12000,
    ratio: float = 0.78,
    model: str = "similarity",
) -> MosaicRegistration:
    """
    Register the independently hand-stitched mosaic to the automatic
    mosaic.

    Supported models:

    similarity:
        translation + rotation + uniform scale

    affine:
        translation + rotation + independent axis scaling + shear
    """
    if model not in {
        "similarity",
        "affine",
    }:
        raise ValueError(
            f"Unsupported registration model: {model}"
        )

    auto_mask = specimen_mask(
        automatic
    )

    hand_mask = specimen_mask(
        hand
    )

    (
        auto_working,
        auto_mask_working,
        auto_scale_matrix,
    ) = resize_for_registration(
        automatic,
        auto_mask,
        max_dimension=max_dimension,
    )

    (
        hand_working,
        hand_mask_working,
        hand_scale_matrix,
    ) = resize_for_registration(
        hand,
        hand_mask,
        max_dimension=max_dimension,
    )

    auto_gray = registration_gray(
        auto_working
    )

    hand_gray = registration_gray(
        hand_working
    )

    sift = create_sift(
        nfeatures=nfeatures
    )

    (
        keypoints_auto_raw,
        descriptors_auto,
    ) = sift.detectAndCompute(
        auto_gray,
        auto_mask_working,
    )

    (
        keypoints_hand_raw,
        descriptors_hand,
    ) = sift.detectAndCompute(
        hand_gray,
        hand_mask_working,
    )

    keypoints_auto = list(
        keypoints_auto_raw
    )

    keypoints_hand = list(
        keypoints_hand_raw
    )

    matches = match_descriptors(
        descriptors_hand,
        descriptors_auto,
        ratio=ratio,
    )

    if matches.mutual_count < 12:
        raise RuntimeError(
            "Too few mutual mosaic matches: "
            f"{matches.mutual_count}"
        )

    points_hand = np.float32(
        [
            keypoints_hand[
                match.queryIdx
            ].pt
            for match
            in matches.mutual_matches
        ]
    )

    points_auto = np.float32(
        [
            keypoints_auto[
                match.trainIdx
            ].pt
            for match
            in matches.mutual_matches
        ]
    )

    if model == "similarity":
        (
            transform_working,
            inlier_mask,
        ) = cv2.estimateAffinePartial2D(
            points_hand,
            points_auto,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=10000,
            confidence=0.999,
            refineIters=50,
        )

    else:
        (
            transform_working,
            inlier_mask,
        ) = cv2.estimateAffine2D(
            points_hand,
            points_auto,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=10000,
            confidence=0.999,
            refineIters=50,
        )

    if (
        transform_working is None
        or inlier_mask is None
    ):
        raise RuntimeError(
            "Could not estimate "
            f"{model} hand-to-automatic registration."
        )

    inlier_flags = (
        inlier_mask.ravel() > 0
    )

    inliers = int(
        inlier_flags.sum()
    )

    inlier_ratio = (
        inliers
        / matches.mutual_count
    )

    working_homogeneous = np.eye(
        3,
        dtype=float,
    )

    working_homogeneous[
        :2,
        :,
    ] = transform_working

    # Convert the working-resolution transformation back into the
    # original coordinate systems of the two mosaic images.
    transform_full = (
        np.linalg.inv(
            auto_scale_matrix
        )
        @ working_homogeneous
        @ hand_scale_matrix
    )

    (
        rotation_deg,
        scale_x,
        scale_y,
        equivalent_scale,
        anisotropy_ratio,
        basis_angle_deg,
        shear_from_orthogonal_deg,
    ) = affine_characteristics(
        transform_full
    )

    predicted_working = cv2.transform(
        points_hand.reshape(
            -1,
            1,
            2,
        ),
        transform_working,
    ).reshape(
        -1,
        2,
    )

    errors_working = np.linalg.norm(
        predicted_working
        - points_auto,
        axis=1,
    )

    auto_scale = (
        auto_working.shape[1]
        / automatic.shape[1]
    )

    errors_auto_pixels = (
        errors_working[
            inlier_flags
        ]
        / auto_scale
    )

    median_error = float(
        np.median(
            errors_auto_pixels
        )
    )

    inlier_matches = [
        match
        for match, is_inlier
        in zip(
            matches.mutual_matches,
            inlier_flags,
            strict=True,
        )
        if is_inlier
    ]

    return MosaicRegistration(
        model=model,
        transform_hand_to_auto=(
            transform_full
        ),
        auto_working=(
            auto_working
        ),
        hand_working=(
            hand_working
        ),
        auto_mask_working=(
            auto_mask_working
        ),
        hand_mask_working=(
            hand_mask_working
        ),
        keypoints_auto=(
            keypoints_auto
        ),
        keypoints_hand=(
            keypoints_hand
        ),
        inlier_matches=(
            inlier_matches
        ),
        mutual_matches=(
            matches.mutual_count
        ),
        inliers=inliers,
        inlier_ratio=(
            inlier_ratio
        ),
        rotation_deg=(
            rotation_deg
        ),
        scale_x=(
            scale_x
        ),
        scale_y=(
            scale_y
        ),
        equivalent_scale=(
            equivalent_scale
        ),
        anisotropy_ratio=(
            anisotropy_ratio
        ),
        basis_angle_deg=(
            basis_angle_deg
        ),
        shear_from_orthogonal_deg=(
            shear_from_orthogonal_deg
        ),
        median_inlier_error_px=(
            median_error
        ),
    )


def warp_hand_to_automatic(
    hand: np.ndarray,
    hand_mask: np.ndarray,
    transform: np.ndarray,
    automatic_shape: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Warp the hand mosaic and specimen mask into automatic-mosaic
    coordinates.
    """
    height, width = (
        automatic_shape[:2]
    )

    affine = transform[:2]

    registered = cv2.warpAffine(
        hand,
        affine,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    registered_mask = cv2.warpAffine(
        hand_mask,
        affine,
        (
            width,
            height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        registered,
        registered_mask,
    )


def comparison_images(
    automatic: np.ndarray,
    registered_hand: np.ndarray,
    automatic_mask: np.ndarray,
    registered_hand_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Produce diagnostic comparison images.

    Returns:
        50/50 RGB overlay
        contrast-normalized grayscale difference
        red/cyan edge overlay
        common-overlap mask
    """
    overlap = (
        (automatic_mask > 0)
        & (registered_hand_mask > 0)
    )

    overlay = automatic.copy()

    blended = cv2.addWeighted(
        automatic,
        0.5,
        registered_hand,
        0.5,
        0.0,
    )

    overlay[
        overlap
    ] = blended[
        overlap
    ]

    gray_auto = cv2.cvtColor(
        automatic,
        cv2.COLOR_BGR2GRAY,
    )

    gray_hand = cv2.cvtColor(
        registered_hand,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    normalized_auto = clahe.apply(
        gray_auto
    )

    normalized_hand = clahe.apply(
        gray_hand
    )

    difference = cv2.absdiff(
        normalized_auto,
        normalized_hand,
    )

    difference[
        ~overlap
    ] = 0

    edges_auto = cv2.Canny(
        normalized_auto,
        60,
        160,
    )

    edges_hand = cv2.Canny(
        normalized_hand,
        60,
        160,
    )

    edges_auto[
        ~overlap
    ] = 0

    edges_hand[
        ~overlap
    ] = 0

    edge_overlay = np.zeros_like(
        automatic
    )

    # Automatic reconstruction = red.
    edge_overlay[:, :, 2] = (
        edges_auto
    )

    # Hand reconstruction = cyan.
    edge_overlay[:, :, 0] = (
        edges_hand
    )

    edge_overlay[:, :, 1] = (
        edges_hand
    )

    overlap_image = np.zeros(
        automatic.shape[:2],
        dtype=np.uint8,
    )

    overlap_image[
        overlap
    ] = 255

    return (
        overlay,
        difference,
        edge_overlay,
        overlap_image,
    )