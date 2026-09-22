from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(slots=True, frozen=True)
class MosaicBounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(slots=True)
class MosaicRenderResult:
    mosaic: np.ndarray
    coverage_preview: np.ndarray
    canvas_width: int
    canvas_height: int
    covered_pixels: int
    max_overlap: int


def pose_from_center(
    center_x: float,
    center_y: float,
    rotation_deg: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Reconstruct an image-to-global rigid transform from the saved
    global image-center coordinate and image rotation.
    """
    angle = radians(rotation_deg)

    rotation = np.array(
        [
            [cos(angle), -sin(angle)],
            [sin(angle), cos(angle)],
        ],
        dtype=float,
    )

    local_center = np.array(
        [
            image_width / 2.0,
            image_height / 2.0,
        ],
        dtype=float,
    )

    global_center = np.array(
        [
            center_x,
            center_y,
        ],
        dtype=float,
    )

    translation = (
        global_center
        - rotation @ local_center
    )

    transform = np.eye(
        3,
        dtype=float,
    )

    transform[:2, :2] = rotation
    transform[:2, 2] = translation

    return transform


def transformed_corners(
    transform: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Return the four image corners in global coordinates.
    """
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(image_width), 0.0, 1.0],
            [
                float(image_width),
                float(image_height),
                1.0,
            ],
            [0.0, float(image_height), 1.0],
        ],
        dtype=float,
    )

    transformed = (
        transform @ corners.T
    ).T

    return transformed[:, :2]


def compute_mosaic_bounds(
    layout: pd.DataFrame,
    image_width: int,
    image_height: int,
    padding_px: float = 0.0,
) -> MosaicBounds:
    """
    Compute the bounding rectangle required to contain every
    transformed microscope image.
    """
    all_corners = []

    for row in layout.to_dict(
        orient="records"
    ):
        transform = pose_from_center(
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            rotation_deg=float(
                row["rotation_deg"]
            ),
            image_width=image_width,
            image_height=image_height,
        )

        corners = transformed_corners(
            transform,
            image_width=image_width,
            image_height=image_height,
        )

        all_corners.append(corners)

    corners_array = np.vstack(
        all_corners
    )

    return MosaicBounds(
        min_x=float(
            corners_array[:, 0].min()
            - padding_px
        ),
        min_y=float(
            corners_array[:, 1].min()
            - padding_px
        ),
        max_x=float(
            corners_array[:, 0].max()
            + padding_px
        ),
        max_y=float(
            corners_array[:, 1].max()
            + padding_px
        ),
    )


def scaled_warp_matrix(
    transform: np.ndarray,
    render_scale: float,
    canvas_origin_x: int,
    canvas_origin_y: int,
) -> np.ndarray:
    """
    Convert a full-resolution image-to-global transform into a
    transform for a resized image and scaled mosaic canvas.
    """
    matrix = np.array(
        [
            [
                transform[0, 0],
                transform[0, 1],
                (
                    render_scale
                    * transform[0, 2]
                    - canvas_origin_x
                ),
            ],
            [
                transform[1, 0],
                transform[1, 1],
                (
                    render_scale
                    * transform[1, 2]
                    - canvas_origin_y
                ),
            ],
        ],
        dtype=np.float64,
    )

    return matrix


def render_average_mosaic(
    image_paths: list[Path],
    layout: pd.DataFrame,
    render_scale: float = 0.25,
    padding_px: float = 64.0,
) -> MosaicRenderResult:
    """
    Render a diagnostic mosaic by averaging all pixels contributing
    to the same global location.
    """
    if not 0.0 < render_scale <= 1.0:
        raise ValueError(
            "render_scale must be in the interval (0, 1]."
        )

    if not image_paths:
        raise ValueError(
            "No microscope images were supplied."
        )

    first_image = cv2.imread(
        str(image_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_image is None:
        raise RuntimeError(
            f"Could not read image: {image_paths[0]}"
        )

    image_height, image_width = (
        first_image.shape[:2]
    )

    layout_by_image = {
        str(row["image"]): row
        for row in layout.to_dict(
            orient="records"
        )
    }

    missing = [
        path.name
        for path in image_paths
        if path.name not in layout_by_image
    ]

    if missing:
        raise RuntimeError(
            "Layout is missing images: "
            f"{missing}"
        )

    bounds = compute_mosaic_bounds(
        layout,
        image_width=image_width,
        image_height=image_height,
        padding_px=padding_px,
    )

    canvas_origin_x = floor(
        bounds.min_x * render_scale
    )

    canvas_origin_y = floor(
        bounds.min_y * render_scale
    )

    canvas_max_x = ceil(
        bounds.max_x * render_scale
    )

    canvas_max_y = ceil(
        bounds.max_y * render_scale
    )

    canvas_width = (
        canvas_max_x
        - canvas_origin_x
    )

    canvas_height = (
        canvas_max_y
        - canvas_origin_y
    )

    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError(
            "Computed mosaic canvas has invalid dimensions."
        )

    pixel_count = (
        canvas_width
        * canvas_height
    )

    core_memory_mib = (
        pixel_count
        * (
            3 * np.dtype(np.float32).itemsize
            + np.dtype(np.float32).itemsize
        )
        / (1024**2)
    )

    print(
        f"Source image size: "
        f"{image_width} x {image_height}"
    )

    print(
        f"Render scale: {render_scale:.3f}"
    )

    print(
        f"Canvas size: "
        f"{canvas_width} x {canvas_height}"
    )

    print(
        f"Core accumulator memory: "
        f"~{core_memory_mib:.1f} MiB"
    )

    accumulator = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.float32,
    )

    weights = np.zeros(
        (
            canvas_height,
            canvas_width,
        ),
        dtype=np.float32,
    )

    for image_path in tqdm(
        image_paths,
        desc="Rendering images",
    ):
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if (
            image.shape[1] != image_width
            or image.shape[0] != image_height
        ):
            raise RuntimeError(
                f"Inconsistent image dimensions: "
                f"{image_path}"
            )

        if render_scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=render_scale,
                fy=render_scale,
                interpolation=cv2.INTER_AREA,
            )

        row = layout_by_image[
            image_path.name
        ]

        transform = pose_from_center(
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

        warp_matrix = scaled_warp_matrix(
            transform,
            render_scale=render_scale,
            canvas_origin_x=canvas_origin_x,
            canvas_origin_y=canvas_origin_y,
        )

        scaled_height, scaled_width = (
            image.shape[:2]
        )

        local_corners = np.array(
            [
                [0.0, 0.0],
                [float(scaled_width), 0.0],
                [
                    float(scaled_width),
                    float(scaled_height),
                ],
                [
                    0.0,
                    float(scaled_height),
                ],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        global_corners = cv2.transform(
            local_corners,
            warp_matrix,
        ).reshape(-1, 2)

        roi_x0 = max(
            0,
            floor(
                float(
                    global_corners[:, 0].min()
                )
            ),
        )

        roi_y0 = max(
            0,
            floor(
                float(
                    global_corners[:, 1].min()
                )
            ),
        )

        roi_x1 = min(
            canvas_width,
            ceil(
                float(
                    global_corners[:, 0].max()
                )
            ),
        )

        roi_y1 = min(
            canvas_height,
            ceil(
                float(
                    global_corners[:, 1].max()
                )
            ),
        )

        if (
            roi_x0 >= roi_x1
            or roi_y0 >= roi_y1
        ):
            continue

        roi_matrix = warp_matrix.copy()

        roi_matrix[0, 2] -= roi_x0
        roi_matrix[1, 2] -= roi_y0

        roi_width = roi_x1 - roi_x0
        roi_height = roi_y1 - roi_y0

        warped = cv2.warpAffine(
            image,
            roi_matrix,
            (
                roi_width,
                roi_height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        source_mask = np.full(
            (
                scaled_height,
                scaled_width,
            ),
            255,
            dtype=np.uint8,
        )

        warped_mask = cv2.warpAffine(
            source_mask,
            roi_matrix,
            (
                roi_width,
                roi_height,
            ),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        valid = warped_mask > 0

        accumulator_roi = accumulator[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]

        weights_roi = weights[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]

        accumulator_roi[valid] += (
            warped[valid]
        )

        weights_roi[valid] += 1.0

    valid_canvas = weights > 0

    for channel in range(3):
        np.divide(
            accumulator[:, :, channel],
            weights,
            out=accumulator[:, :, channel],
            where=valid_canvas,
        )

    mosaic = np.clip(
        accumulator,
        0,
        255,
    ).astype(np.uint8)

    mosaic[~valid_canvas] = 0

    max_overlap = int(
        weights.max()
    )

    coverage_preview = np.zeros(
        weights.shape,
        dtype=np.uint8,
    )

    if max_overlap > 0:
        coverage_preview[valid_canvas] = (
            np.clip(
                (
                    weights[valid_canvas]
                    / max_overlap
                    * 255.0
                ),
                0,
                255,
            )
            .astype(np.uint8)
        )

    return MosaicRenderResult(
        mosaic=mosaic,
        coverage_preview=coverage_preview,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        covered_pixels=int(
            valid_canvas.sum()
        ),
        max_overlap=max_overlap,
    )