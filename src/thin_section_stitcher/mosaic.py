from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
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


@dataclass(slots=True, frozen=True)
class TiledImagePlacement:
    path: Path
    warp_matrix: np.ndarray
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(slots=True, frozen=True)
class TiledMosaicRenderResult:
    output_path: Path
    canvas_width: int
    canvas_height: int
    tile_size: int
    tile_count: int
    covered_pixels: int
    bigtiff: bool


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


def source_feather_weights(
    image_width: int,
    image_height: int,
    feather_fraction: float = 0.15,
    minimum_weight: float = 0.05,
) -> np.ndarray:
    """
    Create a smooth confidence map for one microscope frame.

    Pixels near the source-image border receive less weight, while
    the central region receives full weight.

    The minimum is deliberately non-zero. Therefore regions covered
    by only one source image retain their original intensity after
    normalization.
    """
    if not 0.0 < feather_fraction <= 0.5:
        raise ValueError(
            "feather_fraction must be in (0, 0.5]."
        )

    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError(
            "minimum_weight must be in (0, 1]."
        )

    x = np.arange(
        image_width,
        dtype=np.float32,
    )

    y = np.arange(
        image_height,
        dtype=np.float32,
    )

    distance_x = np.minimum(
        x + 1.0,
        image_width - x,
    )

    distance_y = np.minimum(
        y + 1.0,
        image_height - y,
    )

    distance = np.minimum(
        distance_y[:, None],
        distance_x[None, :],
    )

    feather_width = max(
        1.0,
        min(
            image_width,
            image_height,
        )
        * feather_fraction,
    )

    weights = np.clip(
        distance / feather_width,
        minimum_weight,
        1.0,
    )

    return weights.astype(
        np.float32
    )


def render_average_mosaic(
    image_paths: list[Path],
    layout: pd.DataFrame,
    render_scale: float = 0.25,
    padding_px: float = 64.0,
    image_preprocessor: (
        Callable[
            [str, np.ndarray],
            np.ndarray,
        ]
        | None
    ) = None,
    blend_mode: str = "average",
    feather_fraction: float = 0.15,
    minimum_feather_weight: float = 0.05,
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

    if blend_mode not in {
        "average",
        "feather",
    }:
        raise ValueError(
            "blend_mode must be 'average' or 'feather'."
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

    coverage_count = np.zeros(
        (
            canvas_height,
            canvas_width,
        ),
        dtype=np.uint16,
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

        if image_preprocessor is not None:
            original_shape = image.shape

            image = image_preprocessor(
                image_path.name,
                image,
            )

            if image.shape != original_shape:
                raise RuntimeError(
                    "Image preprocessor changed "
                    f"dimensions for {image_path.name}"
                )

        row = layout_by_image[
            image_path.name
        ]

        scaled_height, scaled_width = (
            image.shape[:2]
        )

        if blend_mode == "feather":
            source_weights = (
                source_feather_weights(
                    image_width=scaled_width,
                    image_height=scaled_height,
                    feather_fraction=(
                        feather_fraction
                    ),
                    minimum_weight=(
                        minimum_feather_weight
                    ),
                )
            )
        else:
            source_weights = np.ones(
                (
                    scaled_height,
                    scaled_width,
                ),
                dtype=np.float32,
            )

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

        warped_weights = cv2.warpAffine(
            source_weights,
            roi_matrix,
            (
                roi_width,
                roi_height,
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        warped_weights[
            warped_mask == 0
        ] = 0.0

        valid = warped_mask > 0

        accumulator_roi = accumulator[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]

        weights_roi = weights[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]

        local_weights = (
            warped_weights[
                valid
            ]
        )

        accumulator_roi[
            valid
        ] += (
                warped[
                    valid
                ].astype(
                    np.float32
                )
                * local_weights[
                    :,
                    None,
                ]
        )

        weights_roi[
            valid
        ] += local_weights

        coverage_count[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ][valid] += 1

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
        coverage_count.max()
    )

    coverage_preview = np.zeros(
        coverage_count.shape,
        dtype=np.uint8,
    )

    if max_overlap > 0:
        coverage_preview[
            valid_canvas
        ] = (
            np.clip(
                (
                        coverage_count[
                            valid_canvas
                        ]
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


def prepare_full_resolution_placements(
    image_paths: list[Path],
    layout: pd.DataFrame,
    image_width: int,
    image_height: int,
    padding_px: float,
) -> tuple[
    list[TiledImagePlacement],
    int,
    int,
]:
    """
    Precompute full-resolution image placement and canvas-space
    bounding boxes.
    """
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
        bounds.min_x
    )

    canvas_origin_y = floor(
        bounds.min_y
    )

    canvas_max_x = ceil(
        bounds.max_x
    )

    canvas_max_y = ceil(
        bounds.max_y
    )

    canvas_width = (
        canvas_max_x
        - canvas_origin_x
    )

    canvas_height = (
        canvas_max_y
        - canvas_origin_y
    )

    placements: list[
        TiledImagePlacement
    ] = []

    for path in image_paths:
        row = layout_by_image[
            path.name
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
            render_scale=1.0,
            canvas_origin_x=(
                canvas_origin_x
            ),
            canvas_origin_y=(
                canvas_origin_y
            ),
        )

        source_corners = np.array(
            [
                [0.0, 0.0],
                [
                    float(image_width),
                    0.0,
                ],
                [
                    float(image_width),
                    float(image_height),
                ],
                [
                    0.0,
                    float(image_height),
                ],
            ],
            dtype=np.float32,
        ).reshape(
            -1,
            1,
            2,
        )

        canvas_corners = cv2.transform(
            source_corners,
            warp_matrix,
        ).reshape(
            -1,
            2,
        )

        x0 = max(
            0,
            floor(
                float(
                    canvas_corners[
                        :,
                        0,
                    ].min()
                )
            ),
        )

        y0 = max(
            0,
            floor(
                float(
                    canvas_corners[
                        :,
                        1,
                    ].min()
                )
            ),
        )

        x1 = min(
            canvas_width,
            ceil(
                float(
                    canvas_corners[
                        :,
                        0,
                    ].max()
                )
            ),
        )

        y1 = min(
            canvas_height,
            ceil(
                float(
                    canvas_corners[
                        :,
                        1,
                    ].max()
                )
            ),
        )

        placements.append(
            TiledImagePlacement(
                path=path,
                warp_matrix=(
                    warp_matrix
                ),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )

    return (
        placements,
        canvas_width,
        canvas_height,
    )


def boxes_intersect(
    ax0: int,
    ay0: int,
    ax1: int,
    ay1: int,
    bx0: int,
    by0: int,
    bx1: int,
    by1: int,
) -> bool:
    return (
        ax0 < bx1
        and ax1 > bx0
        and ay0 < by1
        and ay1 > by0
    )


def render_tiled_mosaic_to_tiff(
    image_paths: list[Path],
    layout: pd.DataFrame,
    output_path: Path,
    padding_px: float = 64.0,
    tile_size: int = 4096,
    image_preprocessor: (
        Callable[
            [str, np.ndarray],
            np.ndarray,
        ]
        | None
    ) = None,
    blend_mode: str = "feather",
    feather_fraction: float = 0.15,
    minimum_feather_weight: float = 0.05,
) -> TiledMosaicRenderResult:
    """
    Render the full-resolution mosaic tile by tile directly into
    an uncompressed TIFF.

    Only one floating-point output tile is kept in memory at once.
    """
    if not image_paths:
        raise ValueError(
            "No microscope images were supplied."
        )

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be positive."
        )

    if blend_mode not in {
        "average",
        "feather",
    }:
        raise ValueError(
            "blend_mode must be "
            "'average' or 'feather'."
        )

    first_image = cv2.imread(
        str(image_paths[0]),
        cv2.IMREAD_COLOR,
    )

    if first_image is None:
        raise RuntimeError(
            f"Could not read image: "
            f"{image_paths[0]}"
        )

    image_height, image_width = (
        first_image.shape[:2]
    )

    (
        placements,
        canvas_width,
        canvas_height,
    ) = prepare_full_resolution_placements(
        image_paths,
        layout,
        image_width=image_width,
        image_height=image_height,
        padding_px=padding_px,
    )

    estimated_bytes = (
        canvas_width
        * canvas_height
        * 3
    )

    # Classic TIFF has a ~4 GiB structural limit.
    # Our current mosaic should remain below it, but switch
    # automatically if a future dataset exceeds that range.
    bigtiff = (
        estimated_bytes
        >= 3_800_000_000
    )

    output_path = (
        output_path
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Source image size: "
        f"{image_width} x "
        f"{image_height}"
    )

    print(
        f"Full-resolution canvas: "
        f"{canvas_width} x "
        f"{canvas_height}"
    )

    print(
        "Estimated RGB size: "
        f"{estimated_bytes / (1024**3):.2f} GiB"
    )

    print(
        f"Tile size: "
        f"{tile_size} x {tile_size}"
    )

    print(
        f"BigTIFF: {bigtiff}"
    )

    output = tifffile.memmap(
        output_path,
        shape=(
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=bigtiff,
        metadata=None,
    )

    if blend_mode == "feather":
        source_weights = (
            source_feather_weights(
                image_width=image_width,
                image_height=image_height,
                feather_fraction=(
                    feather_fraction
                ),
                minimum_weight=(
                    minimum_feather_weight
                ),
            )
        )
    else:
        source_weights = np.ones(
            (
                image_height,
                image_width,
            ),
            dtype=np.float32,
        )

    source_mask = np.full(
        (
            image_height,
            image_width,
        ),
        255,
        dtype=np.uint8,
    )

    tile_origins = [
        (
            tile_x,
            tile_y,
        )
        for tile_y in range(
            0,
            canvas_height,
            tile_size,
        )
        for tile_x in range(
            0,
            canvas_width,
            tile_size,
        )
    ]

    covered_pixels = 0

    for tile_x0, tile_y0 in tqdm(
        tile_origins,
        desc="Rendering full-resolution tiles",
    ):
        tile_x1 = min(
            canvas_width,
            tile_x0 + tile_size,
        )

        tile_y1 = min(
            canvas_height,
            tile_y0 + tile_size,
        )

        tile_width = (
            tile_x1 - tile_x0
        )

        tile_height = (
            tile_y1 - tile_y0
        )

        accumulator = np.zeros(
            (
                tile_height,
                tile_width,
                3,
            ),
            dtype=np.float32,
        )

        weights = np.zeros(
            (
                tile_height,
                tile_width,
            ),
            dtype=np.float32,
        )

        candidates = [
            placement
            for placement in placements
            if boxes_intersect(
                tile_x0,
                tile_y0,
                tile_x1,
                tile_y1,
                placement.x0,
                placement.y0,
                placement.x1,
                placement.y1,
            )
        ]

        for placement in candidates:
            image = cv2.imread(
                str(
                    placement.path
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise RuntimeError(
                    f"Could not read image: "
                    f"{placement.path}"
                )

            if (
                image.shape[1]
                != image_width
                or image.shape[0]
                != image_height
            ):
                raise RuntimeError(
                    "Inconsistent image "
                    "dimensions: "
                    f"{placement.path}"
                )

            if (
                image_preprocessor
                is not None
            ):
                original_shape = (
                    image.shape
                )

                image = (
                    image_preprocessor(
                        placement.path.name,
                        image,
                    )
                )

                if (
                    image.shape
                    != original_shape
                ):
                    raise RuntimeError(
                        "Image preprocessor "
                        "changed dimensions for "
                        f"{placement.path.name}"
                    )

            roi_x0 = max(
                tile_x0,
                placement.x0,
            )

            roi_y0 = max(
                tile_y0,
                placement.y0,
            )

            roi_x1 = min(
                tile_x1,
                placement.x1,
            )

            roi_y1 = min(
                tile_y1,
                placement.y1,
            )

            if (
                roi_x0 >= roi_x1
                or roi_y0 >= roi_y1
            ):
                continue

            roi_width = (
                roi_x1 - roi_x0
            )

            roi_height = (
                roi_y1 - roi_y0
            )

            roi_matrix = (
                placement
                .warp_matrix
                .copy()
            )

            roi_matrix[
                0,
                2,
            ] -= roi_x0

            roi_matrix[
                1,
                2,
            ] -= roi_y0

            warped = cv2.warpAffine(
                image,
                roi_matrix,
                (
                    roi_width,
                    roi_height,
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=(
                    cv2.BORDER_CONSTANT
                ),
                borderValue=0,
            )

            warped_mask = cv2.warpAffine(
                source_mask,
                roi_matrix,
                (
                    roi_width,
                    roi_height,
                ),
                flags=cv2.INTER_NEAREST,
                borderMode=(
                    cv2.BORDER_CONSTANT
                ),
                borderValue=0,
            )

            warped_weights = (
                cv2.warpAffine(
                    source_weights,
                    roi_matrix,
                    (
                        roi_width,
                        roi_height,
                    ),
                    flags=cv2.INTER_LINEAR,
                    borderMode=(
                        cv2.BORDER_CONSTANT
                    ),
                    borderValue=0.0,
                )
            )

            warped_weights[
                warped_mask == 0
            ] = 0.0

            valid = (
                warped_mask > 0
            )

            if not np.any(
                valid
            ):
                continue

            local_x0 = (
                roi_x0 - tile_x0
            )

            local_y0 = (
                roi_y0 - tile_y0
            )

            local_x1 = (
                local_x0
                + roi_width
            )

            local_y1 = (
                local_y0
                + roi_height
            )

            accumulator_roi = (
                accumulator[
                    local_y0:local_y1,
                    local_x0:local_x1,
                ]
            )

            weights_roi = (
                weights[
                    local_y0:local_y1,
                    local_x0:local_x1,
                ]
            )

            local_weights = (
                warped_weights[
                    valid
                ]
            )

            accumulator_roi[
                valid
            ] += (
                warped[
                    valid
                ].astype(
                    np.float32
                )
                * local_weights[
                    :,
                    None,
                ]
            )

            weights_roi[
                valid
            ] += (
                local_weights
            )

        valid_tile = (
            weights > 0
        )

        covered_pixels += int(
            valid_tile.sum()
        )

        for channel in range(3):
            np.divide(
                accumulator[
                    :,
                    :,
                    channel,
                ],
                weights,
                out=accumulator[
                    :,
                    :,
                    channel,
                ],
                where=valid_tile,
            )

        tile_bgr = np.clip(
            accumulator,
            0,
            255,
        ).astype(
            np.uint8
        )

        tile_bgr[
            ~valid_tile
        ] = 0

        # tifffile expects RGB, while OpenCV works in BGR.
        tile_rgb = tile_bgr[
            :,
            :,
            ::-1
        ]

        output[
            tile_y0:tile_y1,
            tile_x0:tile_x1,
        ] = tile_rgb

        # Free the large tile arrays immediately.
        del accumulator
        del weights
        del tile_bgr
        del tile_rgb

    output.flush()

    del output

    return TiledMosaicRenderResult(
        output_path=output_path,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tile_size=tile_size,
        tile_count=len(
            tile_origins
        ),
        covered_pixels=(
            covered_pixels
        ),
        bigtiff=bigtiff,
    )