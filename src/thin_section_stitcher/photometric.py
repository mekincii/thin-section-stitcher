from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, floor, log
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

from thin_section_stitcher.mosaic import (
    pose_from_center,
    scaled_warp_matrix,
    transformed_corners,
)


@dataclass(slots=True)
class PairPhotometricMeasurement:
    image_a: str
    image_b: str

    log_ratio_blue: float
    log_ratio_green: float
    log_ratio_red: float

    centroid_ax: float
    centroid_ay: float
    centroid_bx: float
    centroid_by: float

    valid_pixels: int
    sampled_pixels: int
    geometric_quality: float


@dataclass(slots=True)
class PhotometricEstimationResult:
    gains: pd.DataFrame
    measurements: pd.DataFrame
    successful_pairs: int
    failed_pairs: int


@dataclass(slots=True)
class PhotometricCalibration:
    gains_by_image: dict[str, np.ndarray]
    field_coefficients_bgr: np.ndarray
    max_field_correction: float


def load_photometric_calibration(
    gains_path: Path,
    field_path: Path,
    max_field_correction: float = 1.5,
) -> PhotometricCalibration:
    """
    Load the final per-image gains and shared illumination field.

    Channel order is OpenCV BGR.
    """
    if max_field_correction < 1.0:
        raise ValueError(
            "max_field_correction must be >= 1."
        )

    gains_frame = pd.read_csv(
        gains_path
    )

    required_columns = {
        "image",
        "gain_blue",
        "gain_green",
        "gain_red",
    }

    missing = (
        required_columns
        - set(gains_frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Photometric gains are missing columns: "
            f"{sorted(missing)}"
        )

    gains_by_image = {
        str(row["image"]): np.array(
            [
                float(row["gain_blue"]),
                float(row["gain_green"]),
                float(row["gain_red"]),
            ],
            dtype=np.float32,
        )
        for row in gains_frame.to_dict(
            orient="records"
        )
    }

    field_document = json.loads(
        field_path.read_text(
            encoding="utf-8"
        )
    )

    channels = field_document[
        "channels"
    ]

    field_coefficients_bgr = np.array(
        [
            channels["blue"],
            channels["green"],
            channels["red"],
        ],
        dtype=np.float32,
    )

    if field_coefficients_bgr.shape != (
        3,
        5,
    ):
        raise RuntimeError(
            "Expected a 3 x 5 photometric field."
        )

    return PhotometricCalibration(
        gains_by_image=gains_by_image,
        field_coefficients_bgr=(
            field_coefficients_bgr
        ),
        max_field_correction=(
            max_field_correction
        ),
    )


def apply_photometric_calibration(
    image_name: str,
    image: np.ndarray,
    calibration: PhotometricCalibration,
) -> np.ndarray:
    """
    Standardize one microscope image.

    Model:

        observed =
            standardized
            * exp(field(x, y))
            / correction_gain

    therefore:

        standardized =
            observed
            * correction_gain
            * exp(-field(x, y))

    The spatial field correction is conservatively clipped.
    """
    if image_name not in (
        calibration.gains_by_image
    ):
        raise RuntimeError(
            "No photometric gain for image: "
            f"{image_name}"
        )

    height, width = image.shape[:2]

    x = np.linspace(
        -1.0,
        1.0,
        width,
        dtype=np.float32,
    )[None, :]

    y = np.linspace(
        -1.0,
        1.0,
        height,
        dtype=np.float32,
    )[:, None]

    x_squared = x * x
    y_squared = y * y
    xy = y * x

    corrected = image.astype(
        np.float32
    )

    gains = (
        calibration.gains_by_image[
            image_name
        ]
    )

    maximum = (
        calibration.max_field_correction
    )

    minimum = 1.0 / maximum

    for channel in range(3):
        coefficients = (
            calibration
            .field_coefficients_bgr[
                channel
            ]
        )

        log_field = (
            coefficients[0] * x
            + coefficients[1] * y
            + coefficients[2] * x_squared
            + coefficients[3] * xy
            + coefficients[4] * y_squared
        )

        field_correction = np.exp(
            -log_field
        )

        field_correction = np.clip(
            field_correction,
            minimum,
            maximum,
        )

        corrected[
            :,
            :,
            channel,
        ] *= (
            gains[channel]
            * field_correction
        )

    return np.clip(
        corrected,
        0,
        255,
    ).astype(np.uint8)


def apply_bgr_gains(
    image: np.ndarray,
    gain_blue: float,
    gain_green: float,
    gain_red: float,
) -> np.ndarray:
    """
    Apply conservative per-channel multiplicative gains.
    """
    gains = np.array(
        [
            gain_blue,
            gain_green,
            gain_red,
        ],
        dtype=np.float32,
    )

    corrected = (
        image.astype(np.float32)
        * gains.reshape(1, 1, 3)
    )

    return np.clip(
        corrected,
        0,
        255,
    ).astype(np.uint8)


def load_scaled_images(
    image_paths: list[Path],
    scale: float,
) -> tuple[
    dict[str, np.ndarray],
    int,
    int,
]:
    """
    Load and cache all microscope images at the photometric
    estimation scale.
    """
    if not 0.0 < scale <= 1.0:
        raise ValueError(
            "scale must be in the interval (0, 1]."
        )

    if not image_paths:
        raise ValueError(
            "No microscope images were supplied."
        )

    first = cv2.imread(
        str(image_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first is None:
        raise RuntimeError(
            f"Could not read image: {image_paths[0]}"
        )

    image_height, image_width = (
        first.shape[:2]
    )

    scaled_images: dict[
        str,
        np.ndarray,
    ] = {}

    for path in tqdm(
        image_paths,
        desc="Loading photometric images",
    ):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        if (
            image.shape[1] != image_width
            or image.shape[0] != image_height
        ):
            raise RuntimeError(
                f"Inconsistent dimensions: {path}"
            )

        if scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        scaled_images[
            path.name
        ] = image

    return (
        scaled_images,
        image_width,
        image_height,
    )


def layout_transforms(
    layout: pd.DataFrame,
    image_width: int,
    image_height: int,
) -> dict[str, np.ndarray]:
    """
    Reconstruct every optimized image-to-global transform.
    """
    transforms: dict[
        str,
        np.ndarray,
    ] = {}

    for row in layout.to_dict(
        orient="records"
    ):
        image_name = str(
            row["image"]
        )

        transforms[
            image_name
        ] = pose_from_center(
            center_x=float(
                row["center_x"]
            ),
            center_y=float(
                row["center_y"]
            ),
            rotation_deg=float(
                row["rotation_deg"]
            ),
            image_width=image_width,
            image_height=image_height,
        )

    return transforms


def measure_pair_photometry(
    image_a: np.ndarray,
    image_b: np.ndarray,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    original_width: int,
    original_height: int,
    render_scale: float,
    image_a_name: str,
    image_b_name: str,
    geometric_quality: float,
    max_samples: int = 100_000,
) -> PairPhotometricMeasurement | None:
    """
    Compare corresponding pixels from one geometrically verified
    overlap.

    The median logarithmic B/G/R intensity ratio is estimated after
    suppressing image borders, very dark pixels, saturated pixels,
    and high-frequency registration noise.
    """
    corners_a = transformed_corners(
        transform_a,
        image_width=original_width,
        image_height=original_height,
    )

    corners_b = transformed_corners(
        transform_b,
        image_width=original_width,
        image_height=original_height,
    )

    min_x = max(
        float(
            corners_a[:, 0].min()
        ),
        float(
            corners_b[:, 0].min()
        ),
    )

    min_y = max(
        float(
            corners_a[:, 1].min()
        ),
        float(
            corners_b[:, 1].min()
        ),
    )

    max_x = min(
        float(
            corners_a[:, 0].max()
        ),
        float(
            corners_b[:, 0].max()
        ),
    )

    max_y = min(
        float(
            corners_a[:, 1].max()
        ),
        float(
            corners_b[:, 1].max()
        ),
    )

    if (
        min_x >= max_x
        or min_y >= max_y
    ):
        return None

    origin_x = floor(
        min_x * render_scale
    )

    origin_y = floor(
        min_y * render_scale
    )

    end_x = ceil(
        max_x * render_scale
    )

    end_y = ceil(
        max_y * render_scale
    )

    roi_width = (
        end_x - origin_x
    )

    roi_height = (
        end_y - origin_y
    )

    if (
        roi_width <= 0
        or roi_height <= 0
    ):
        return None

    matrix_a = scaled_warp_matrix(
        transform_a,
        render_scale=render_scale,
        canvas_origin_x=origin_x,
        canvas_origin_y=origin_y,
    )

    matrix_b = scaled_warp_matrix(
        transform_b,
        render_scale=render_scale,
        canvas_origin_x=origin_x,
        canvas_origin_y=origin_y,
    )

    warped_a = cv2.warpAffine(
        image_a,
        matrix_a,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_b = cv2.warpAffine(
        image_b,
        matrix_b,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    source_mask_a = np.full(
        image_a.shape[:2],
        255,
        dtype=np.uint8,
    )

    source_mask_b = np.full(
        image_b.shape[:2],
        255,
        dtype=np.uint8,
    )

    mask_a = cv2.warpAffine(
        source_mask_a,
        matrix_a,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask_b = cv2.warpAffine(
        source_mask_b,
        matrix_b,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid = (
        (mask_a > 0)
        & (mask_b > 0)
    )

    # Remove warped image boundaries where interpolation is the least stable.
    valid_uint8 = (
        valid.astype(np.uint8)
        * 255
    )

    erosion_kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    valid_uint8 = cv2.erode(
        valid_uint8,
        erosion_kernel,
        iterations=1,
    )

    valid = (
        valid_uint8 > 0
    )

    # We are estimating illumination, not geological texture.
    # A mild blur makes the statistic less sensitive to sub-pixel
    # registration differences.
    blurred_a = cv2.GaussianBlur(
        warped_a,
        (5, 5),
        1.2,
    )

    blurred_b = cv2.GaussianBlur(
        warped_b,
        (5, 5),
        1.2,
    )

    gray_a = cv2.cvtColor(
        blurred_a,
        cv2.COLOR_BGR2GRAY,
    )

    gray_b = cv2.cvtColor(
        blurred_b,
        cv2.COLOR_BGR2GRAY,
    )

    # Exclude nearly black areas, saturated regions, and borders.
    valid &= (
        (gray_a >= 20)
        & (gray_a <= 235)
        & (gray_b >= 20)
        & (gray_b <= 235)
    )

    valid_indices = np.flatnonzero(
        valid
    )

    valid_pixels = len(valid_indices)

    if valid_pixels < 2_000:
        return None

    if (
        len(valid_indices)
        > max_samples
    ):
        sample_positions = np.linspace(
            0,
            len(valid_indices) - 1,
            max_samples,
            dtype=np.int64,
        )

        valid_indices = (
            valid_indices[
                sample_positions
            ]
        )

    sample_y, sample_x = np.unravel_index(
        valid_indices,
        valid.shape,
    )

    roi_points = np.column_stack(
        [
            sample_x,
            sample_y,
        ]
    ).astype(
        np.float32
    ).reshape(
        -1,
        1,
        2,
    )

    inverse_a = cv2.invertAffineTransform(
        matrix_a
    )

    inverse_b = cv2.invertAffineTransform(
        matrix_b
    )

    source_points_a = cv2.transform(
        roi_points,
        inverse_a,
    ).reshape(
        -1,
        2,
    )

    source_points_b = cv2.transform(
        roi_points,
        inverse_b,
    ).reshape(
        -1,
        2,
    )

    scaled_height_a, scaled_width_a = (
        image_a.shape[:2]
    )

    scaled_height_b, scaled_width_b = (
        image_b.shape[:2]
    )

    normalized_ax = (
        2.0
        * source_points_a[:, 0]
        / max(
            scaled_width_a - 1,
            1,
        )
        - 1.0
    )

    normalized_ay = (
        2.0
        * source_points_a[:, 1]
        / max(
            scaled_height_a - 1,
            1,
        )
        - 1.0
    )

    normalized_bx = (
        2.0
        * source_points_b[:, 0]
        / max(
            scaled_width_b - 1,
            1,
        )
        - 1.0
    )

    normalized_by = (
        2.0
        * source_points_b[:, 1]
        / max(
            scaled_height_b - 1,
            1,
        )
        - 1.0
    )

    centroid_ax = float(
        np.median(
            normalized_ax
        )
    )

    centroid_ay = float(
        np.median(
            normalized_ay
        )
    )

    centroid_bx = float(
        np.median(
            normalized_bx
        )
    )

    centroid_by = float(
        np.median(
            normalized_by
        )
    )

    flat_a = blurred_a.reshape(
        -1,
        3,
    ).astype(np.float32)

    flat_b = blurred_b.reshape(
        -1,
        3,
    ).astype(np.float32)

    samples_a = flat_a[
        valid_indices
    ]

    samples_b = flat_b[
        valid_indices
    ]

    # Additional per-channel protection against zeros and clipping.
    channel_valid = (
        (samples_a > 5)
        & (samples_a < 250)
        & (samples_b > 5)
        & (samples_b < 250)
    )

    log_ratios = []

    for channel in range(3):
        channel_mask = (
            channel_valid[
                :,
                channel
            ]
        )

        if (
            int(
                channel_mask.sum()
            )
            < 1_000
        ):
            return None

        values_a = samples_a[
            channel_mask,
            channel,
        ]

        values_b = samples_b[
            channel_mask,
            channel,
        ]

        ratio = (
            np.log(
                values_b + 1.0
            )
            - np.log(
                values_a + 1.0
            )
        )

        log_ratios.append(
            float(
                np.median(
                    ratio
                )
            )
        )

    return PairPhotometricMeasurement(
        image_a=image_a_name,
        image_b=image_b_name,
        log_ratio_blue=log_ratios[0],
        log_ratio_green=log_ratios[1],
        log_ratio_red=log_ratios[2],
        centroid_ax=centroid_ax,
        centroid_ay=centroid_ay,
        centroid_bx=centroid_bx,
        centroid_by=centroid_by,
        valid_pixels=valid_pixels,
        sampled_pixels=len(valid_indices),
        geometric_quality=float(
            geometric_quality
        ),
    )


def solve_channel_gains(
    image_names: list[str],
    measurements: list[
        PairPhotometricMeasurement
    ],
    channel_attribute: str,
    root: str,
) -> dict[str, float]:
    """
    Solve one globally consistent set of multiplicative gains.

    If:
        corrected_A = gain_A * observed_A
        corrected_B = gain_B * observed_B

    then an overlap measurement provides:

        log(gain_A) - log(gain_B)
        ≈ log(observed_B) - log(observed_A)

    A robust least-squares solution combines all trusted overlaps.
    """
    if root not in image_names:
        raise ValueError(
            f"Photometric root not found: {root}"
        )

    free_images = [
        name
        for name in image_names
        if name != root
    ]

    index = {
        name: idx
        for idx, name
        in enumerate(
            free_images
        )
    }

    def log_gain(
        parameters: np.ndarray,
        image_name: str,
    ) -> float:
        if image_name == root:
            return 0.0

        return float(
            parameters[
                index[
                    image_name
                ]
            ]
        )

    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        residuals = []

        for measurement in measurements:
            observed_difference = float(
                getattr(
                    measurement,
                    channel_attribute,
                )
            )

            predicted_difference = (
                log_gain(
                    parameters,
                    measurement.image_a,
                )
                - log_gain(
                    parameters,
                    measurement.image_b,
                )
            )

            weight = np.sqrt(
                max(
                    measurement.geometric_quality,
                    0.1,
                )
            )

            residuals.append(
                weight
                * (
                    predicted_difference
                    - observed_difference
                )
            )

        return np.asarray(
            residuals,
            dtype=float,
        )

    max_log_gain = log(
        2.0
    )

    result = least_squares(
        residual_function,
        np.zeros(
            len(free_images),
            dtype=float,
        ),
        bounds=(
            -max_log_gain,
            max_log_gain,
        ),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=500,
    )

    if not result.success:
        raise RuntimeError(
            "Photometric optimization failed: "
            f"{result.message}"
        )

    solved_logs = {
        root: 0.0,
    }

    for name in free_images:
        solved_logs[
            name
        ] = float(
            result.x[
                index[name]
            ]
        )

    # Remove arbitrary dependence on the chosen root.
    # The geometric mean gain across all images becomes 1.
    median_log_gain = float(
        np.median(
            list(
                solved_logs.values()
            )
        )
    )

    gains = {
        name: float(
            np.exp(
                value
                - median_log_gain
            )
        )
        for name, value
        in solved_logs.items()
    }

    return gains


def estimate_photometric_gains(
    image_paths: list[Path],
    layout: pd.DataFrame,
    trusted_edges: pd.DataFrame,
    scale: float = 0.25,
    root: str = "60.jpg",
    max_samples_per_pair: int = 100_000,
) -> PhotometricEstimationResult:
    """
    Estimate globally consistent B/G/R gains from trusted geometric
    overlaps.
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

    image_names = [
        path.name
        for path in image_paths
    ]

    measurements: list[
        PairPhotometricMeasurement
    ] = []

    failed_pairs = 0

    edge_records = (
        trusted_edges.to_dict(
            orient="records"
        )
    )

    for edge in tqdm(
        edge_records,
        desc="Measuring overlap illumination",
    ):
        image_a = str(
            edge["image_a"]
        )

        image_b = str(
            edge["image_b"]
        )

        if (
            image_a not in scaled_images
            or image_b not in scaled_images
            or image_a not in transforms
            or image_b not in transforms
        ):
            failed_pairs += 1
            continue

        measurement = (
            measure_pair_photometry(
                scaled_images[
                    image_a
                ],
                scaled_images[
                    image_b
                ],
                transforms[
                    image_a
                ],
                transforms[
                    image_b
                ],
                original_width=(
                    original_width
                ),
                original_height=(
                    original_height
                ),
                render_scale=scale,
                image_a_name=image_a,
                image_b_name=image_b,
                geometric_quality=float(
                    edge[
                        "inlier_ratio_verification"
                    ]
                ),
                max_samples=(
                    max_samples_per_pair
                ),
            )
        )

        if measurement is None:
            failed_pairs += 1
            continue

        measurements.append(
            measurement
        )

    if not measurements:
        raise RuntimeError(
            "No usable photometric overlap measurements."
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
                "gain_blue": (
                    gains_blue[
                        image_name
                    ]
                ),
                "gain_green": (
                    gains_green[
                        image_name
                    ]
                ),
                "gain_red": (
                    gains_red[
                        image_name
                    ]
                ),
            }
            for image_name
            in image_names
        ]
    )

    measurements_frame = (
        pd.DataFrame(
            [
                {
                    "image_a": (
                        measurement.image_a
                    ),
                    "image_b": (
                        measurement.image_b
                    ),
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
                }
                for measurement
                in measurements
            ]
        )
    )

    return PhotometricEstimationResult(
        gains=gains,
        measurements=(
            measurements_frame
        ),
        successful_pairs=len(
            measurements
        ),
        failed_pairs=(
            failed_pairs
        ),
    )