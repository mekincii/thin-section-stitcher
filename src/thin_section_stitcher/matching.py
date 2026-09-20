from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, degrees, hypot

import cv2
import numpy as np


@dataclass(slots=True)
class DescriptorMatchResult:
    """
    Result of bidirectional descriptor matching.
    """

    forward_good: list[cv2.DMatch]
    backward_good: list[cv2.DMatch]
    mutual_matches: list[cv2.DMatch]

    @property
    def forward_count(self) -> int:
        return len(self.forward_good)

    @property
    def backward_count(self) -> int:
        return len(self.backward_good)

    @property
    def mutual_count(self) -> int:
        return len(self.mutual_matches)

    @property
    def median_distance(self) -> float | None:
        if not self.mutual_matches:
            return None

        distances = [
            match.distance
            for match in self.mutual_matches
        ]

        return float(np.median(distances))


@dataclass(slots=True)
class GeometricVerificationResult:
    """
    Geometric consistency of descriptor matches.
    """

    success: bool
    inliers: int
    inlier_ratio: float
    dx: float | None
    dy: float | None
    rotation_deg: float | None
    scale: float | None
    transform: np.ndarray | None


@dataclass(slots=True, frozen=True)
class DiscoveryCriteria:
    """
    Permissive criteria for deciding whether a pair deserves
    higher-quality verification.
    """

    min_mutual_matches: int = 12
    min_inliers: int = 8
    min_inlier_ratio: float = 0.50
    min_scale: float = 0.95
    max_scale: float = 1.05


@dataclass(slots=True, frozen=True)
class CrossScaleConsistency:
    """
    Agreement between geometric transforms estimated at two image scales.
    """

    translation_error_px: float
    rotation_error_deg: float
    scale_error: float


def create_flann_matcher(
    trees: int = 5,
    checks: int = 128,
) -> cv2.FlannBasedMatcher:
    """
    Create a FLANN KD-tree matcher suitable for SIFT descriptors.

    A relatively high number of search checks is used because matching
    reliability is more important here than maximum speed.
    """
    index_params = {
        "algorithm": 1,  # FLANN_INDEX_KDTREE
        "trees": trees,
    }

    search_params = {
        "checks": checks,
    }

    return cv2.FlannBasedMatcher(
        index_params,
        search_params,
    )


def ratio_filter(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Apply Lowe's ratio test to k-nearest-neighbour matches.
    """
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be between 0 and 1.")

    good_matches = []

    for neighbours in knn_matches:
        if len(neighbours) < 2:
            continue

        best, second_best = neighbours[:2]

        if best.distance < ratio * second_best.distance:
            good_matches.append(best)

    return good_matches


def match_descriptors(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    ratio: float = 0.75,
    trees: int = 5,
    checks: int = 128,
) -> DescriptorMatchResult:
    """
    Match two SIFT descriptor sets in both directions.

    Matches must pass Lowe's ratio test in both A->B and B->A directions.
    Only mutually consistent matches are retained.
    """
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        return DescriptorMatchResult([], [], [])

    descriptors_a = np.asarray(
        descriptors_a,
        dtype=np.float32,
    )

    descriptors_b = np.asarray(
        descriptors_b,
        dtype=np.float32,
    )

    matcher = create_flann_matcher(
        trees=trees,
        checks=checks,
    )

    forward_knn = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    backward_knn = matcher.knnMatch(
        descriptors_b,
        descriptors_a,
        k=2,
    )

    forward_good = ratio_filter(
        forward_knn,
        ratio=ratio,
    )

    backward_good = ratio_filter(
        backward_knn,
        ratio=ratio,
    )

    backward_pairs = {
        (match.trainIdx, match.queryIdx)
        for match in backward_good
    }

    mutual_matches = [
        match
        for match in forward_good
        if (match.queryIdx, match.trainIdx)
        in backward_pairs
    ]

    return DescriptorMatchResult(
        forward_good=forward_good,
        backward_good=backward_good,
        mutual_matches=mutual_matches,
    )


def verify_geometry(
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_threshold: float = 3.0,
) -> GeometricVerificationResult:
    """
    Verify whether descriptor matches agree on one physically plausible
    2D transformation.

    Uses a partial affine transform:
    translation + rotation + uniform scale.
    """
    if len(matches) < 4:
        return GeometricVerificationResult(
            success=False,
            inliers=0,
            inlier_ratio=0.0,
            dx=None,
            dy=None,
            rotation_deg=None,
            scale=None,
            transform=None,
        )

    points_a = np.float32(
        [
            keypoints_a[match.queryIdx].pt
            for match in matches
        ]
    )

    points_b = np.float32(
        [
            keypoints_b[match.trainIdx].pt
            for match in matches
        ]
    )

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        points_a,
        points_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )

    if transform is None or inlier_mask is None:
        return GeometricVerificationResult(
            success=False,
            inliers=0,
            inlier_ratio=0.0,
            dx=None,
            dy=None,
            rotation_deg=None,
            scale=None,
            transform=None,
        )

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(matches)

    a = float(transform[0, 0])
    b = float(transform[1, 0])

    scale = hypot(a, b)
    rotation_deg = degrees(atan2(b, a))

    dx = float(transform[0, 2])
    dy = float(transform[1, 2])

    return GeometricVerificationResult(
        success=True,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        dx=dx,
        dy=dy,
        rotation_deg=rotation_deg,
        scale=scale,
        transform=transform,
    )


def is_plausible_overlap(
    matches: DescriptorMatchResult,
    geometry: GeometricVerificationResult,
    criteria: DiscoveryCriteria | None = None,
) -> bool:
    """
    Decide whether a pair should advance to the verification stage.
    """
    if criteria is None:
        criteria = DiscoveryCriteria()

    if (
        not geometry.success
        or geometry.scale is None
        or geometry.rotation_deg is None
    ):
        return False

    return (
            matches.mutual_count >= criteria.min_mutual_matches
            and geometry.inliers >= criteria.min_inliers
            and geometry.inlier_ratio >= criteria.min_inlier_ratio
            and criteria.min_scale
            <= geometry.scale
            <= criteria.max_scale
    )


def compare_cross_scale_geometry(
    dx_a: float,
    dy_a: float,
    rotation_a: float,
    scale_a: float,
    image_scale_a: float,
    dx_b: float,
    dy_b: float,
    rotation_b: float,
    scale_b: float,
    image_scale_b: float,
) -> CrossScaleConsistency:
    """
    Compare two transforms after converting translations back to
    original-image coordinates.
    """
    original_dx_a = dx_a / image_scale_a
    original_dy_a = dy_a / image_scale_a

    original_dx_b = dx_b / image_scale_b
    original_dy_b = dy_b / image_scale_b

    translation_error = hypot(
        original_dx_a - original_dx_b,
        original_dy_a - original_dy_b,
    )

    rotation_difference = (
        rotation_a - rotation_b + 180.0
    ) % 360.0 - 180.0

    rotation_error = abs(rotation_difference)
    scale_error = abs(scale_a - scale_b)

    return CrossScaleConsistency(
        translation_error_px=translation_error,
        rotation_error_deg=rotation_error,
        scale_error=scale_error,
    )