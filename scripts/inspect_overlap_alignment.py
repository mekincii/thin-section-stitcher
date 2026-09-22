from __future__ import annotations

import argparse
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from thin_section_stitcher.dataset import discover_images
from thin_section_stitcher.mosaic import (
    pose_from_center,
    transformed_corners,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect local alignment quality for selected "
            "overlapping microscope-image pairs."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--layout",
        type=Path,
        default=Path(
            "outputs/optimized_global_layout.csv"
        ),
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help=(
            "Image-number pairs such as "
            "116:120 63:155 12:13."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/overlap_diagnostics"
        ),
    )

    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--preview-max-height",
        type=int,
        default=900,
    )

    return parser.parse_args()


def parse_pair(
    value: str,
) -> tuple[str, str]:
    try:
        left, right = value.split(
            ":",
            maxsplit=1,
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid pair: {value}"
        ) from error

    return f"{int(left)}.jpg", f"{int(right)}.jpg"


def warp_image_to_overlap(
    image: np.ndarray,
    transform: np.ndarray,
    roi_x0: int,
    roi_y0: int,
    roi_width: int,
    roi_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    warp_matrix = transform[:2].copy()

    warp_matrix[0, 2] -= roi_x0
    warp_matrix[1, 2] -= roi_y0

    warped = cv2.warpAffine(
        image,
        warp_matrix,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    source_mask = np.full(
        image.shape[:2],
        255,
        dtype=np.uint8,
    )

    mask = cv2.warpAffine(
        source_mask,
        warp_matrix,
        (
            roi_width,
            roi_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped, mask


def label_panel(
    image: np.ndarray,
    text: str,
) -> np.ndarray:
    result = image.copy()

    cv2.rectangle(
        result,
        (0, 0),
        (
            min(result.shape[1], 420),
            42,
        ),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        result,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def resize_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_width / width,
        max_height / height,
    )

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def inspect_pair(
    image_a_path: Path,
    image_b_path: Path,
    layout_by_image: dict[str, dict],
    output_dir: Path,
    preview_max_width: int,
    preview_max_height: int,
) -> dict:
    image_a = cv2.imread(
        str(image_a_path),
        cv2.IMREAD_COLOR,
    )

    image_b = cv2.imread(
        str(image_b_path),
        cv2.IMREAD_COLOR,
    )

    if image_a is None:
        raise RuntimeError(
            f"Could not read image: {image_a_path}"
        )

    if image_b is None:
        raise RuntimeError(
            f"Could not read image: {image_b_path}"
        )

    if image_a.shape != image_b.shape:
        raise RuntimeError(
            "Diagnostic pair has inconsistent image sizes."
        )

    image_height, image_width = (
        image_a.shape[:2]
    )

    row_a = layout_by_image[
        image_a_path.name
    ]

    row_b = layout_by_image[
        image_b_path.name
    ]

    transform_a = pose_from_center(
        center_x=float(
            row_a["center_x"]
        ),
        center_y=float(
            row_a["center_y"]
        ),
        rotation_deg=float(
            row_a["rotation_deg"]
        ),
        image_width=image_width,
        image_height=image_height,
    )

    transform_b = pose_from_center(
        center_x=float(
            row_b["center_x"]
        ),
        center_y=float(
            row_b["center_y"]
        ),
        rotation_deg=float(
            row_b["rotation_deg"]
        ),
        image_width=image_width,
        image_height=image_height,
    )

    corners_a = transformed_corners(
        transform_a,
        image_width=image_width,
        image_height=image_height,
    )

    corners_b = transformed_corners(
        transform_b,
        image_width=image_width,
        image_height=image_height,
    )

    roi_x0 = floor(
        max(
            corners_a[:, 0].min(),
            corners_b[:, 0].min(),
        )
    )

    roi_y0 = floor(
        max(
            corners_a[:, 1].min(),
            corners_b[:, 1].min(),
        )
    )

    roi_x1 = ceil(
        min(
            corners_a[:, 0].max(),
            corners_b[:, 0].max(),
        )
    )

    roi_y1 = ceil(
        min(
            corners_a[:, 1].max(),
            corners_b[:, 1].max(),
        )
    )

    if (
        roi_x0 >= roi_x1
        or roi_y0 >= roi_y1
    ):
        raise RuntimeError(
            f"No global bounding-box overlap for "
            f"{image_a_path.name} and "
            f"{image_b_path.name}."
        )

    roi_width = roi_x1 - roi_x0
    roi_height = roi_y1 - roi_y0

    warped_a, mask_a = (
        warp_image_to_overlap(
            image_a,
            transform_a,
            roi_x0,
            roi_y0,
            roi_width,
            roi_height,
        )
    )

    warped_b, mask_b = (
        warp_image_to_overlap(
            image_b,
            transform_b,
            roi_x0,
            roi_y0,
            roi_width,
            roi_height,
        )
    )

    valid = (
        (mask_a > 0)
        & (mask_b > 0)
    )

    valid_y, valid_x = np.where(
        valid
    )

    if len(valid_x) == 0:
        raise RuntimeError(
            f"No actual overlap for "
            f"{image_a_path.name} and "
            f"{image_b_path.name}."
        )

    crop_x0 = int(valid_x.min())
    crop_x1 = int(valid_x.max()) + 1

    crop_y0 = int(valid_y.min())
    crop_y1 = int(valid_y.max()) + 1

    warped_a = warped_a[
        crop_y0:crop_y1,
        crop_x0:crop_x1,
    ]

    warped_b = warped_b[
        crop_y0:crop_y1,
        crop_x0:crop_x1,
    ]

    valid = valid[
        crop_y0:crop_y1,
        crop_x0:crop_x1,
    ]

    panel_a = warped_a.copy()
    panel_b = warped_b.copy()

    panel_a[~valid] = 0
    panel_b[~valid] = 0

    blend = cv2.addWeighted(
        panel_a,
        0.5,
        panel_b,
        0.5,
        0.0,
    )

    blend[~valid] = 0

    gray_a = cv2.cvtColor(
        panel_a,
        cv2.COLOR_BGR2GRAY,
    )

    gray_b = cv2.cvtColor(
        panel_b,
        cv2.COLOR_BGR2GRAY,
    )

    edges_a = cv2.Canny(
        gray_a,
        60,
        160,
    )

    edges_b = cv2.Canny(
        gray_b,
        60,
        160,
    )

    edges_a[~valid] = 0
    edges_b[~valid] = 0

    edge_overlay = np.zeros_like(
        panel_a
    )

    # A = red, B = cyan.
    # Coincident edges therefore approach white.
    edge_overlay[:, :, 2] = edges_a
    edge_overlay[:, :, 0] = edges_b
    edge_overlay[:, :, 1] = edges_b

    preview_a = resize_preview(
        panel_a,
        preview_max_width,
        preview_max_height,
    )

    preview_b = resize_preview(
        panel_b,
        preview_max_width,
        preview_max_height,
    )

    preview_blend = resize_preview(
        blend,
        preview_max_width,
        preview_max_height,
    )

    preview_edges = resize_preview(
        edge_overlay,
        preview_max_width,
        preview_max_height,
    )

    preview_a = label_panel(
        preview_a,
        image_a_path.name,
    )

    preview_b = label_panel(
        preview_b,
        image_b_path.name,
    )

    preview_blend = label_panel(
        preview_blend,
        "50/50 overlay",
    )

    preview_edges = label_panel(
        preview_edges,
        "Edges: A red / B cyan",
    )

    top = np.hstack(
        [
            preview_a,
            preview_b,
        ]
    )

    bottom = np.hstack(
        [
            preview_blend,
            preview_edges,
        ]
    )

    contact_sheet = np.vstack(
        [
            top,
            bottom,
        ]
    )

    pair_name = (
        f"{image_a_path.stem}_"
        f"{image_b_path.stem}"
    )

    pair_dir = (
        output_dir
        / pair_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_path = (
        pair_dir
        / "diagnostic.png"
    )

    blend_path = (
        pair_dir
        / "blend_full.png"
    )

    edges_path = (
        pair_dir
        / "edges_full.png"
    )

    if not cv2.imwrite(
        str(contact_path),
        contact_sheet,
    ):
        raise RuntimeError(
            f"Could not save: {contact_path}"
        )

    if not cv2.imwrite(
        str(blend_path),
        blend,
    ):
        raise RuntimeError(
            f"Could not save: {blend_path}"
        )

    if not cv2.imwrite(
        str(edges_path),
        edge_overlay,
    ):
        raise RuntimeError(
            f"Could not save: {edges_path}"
        )

    return {
        "image_a": image_a_path.name,
        "image_b": image_b_path.name,
        "overlap_width": int(
            panel_a.shape[1]
        ),
        "overlap_height": int(
            panel_a.shape[0]
        ),
        "overlap_pixels": int(
            valid.sum()
        ),
        "diagnostic": str(
            contact_path
        ),
    }


def main() -> None:
    args = parse_args()

    dataset = (
        args.dataset
        .expanduser()
        .resolve()
    )

    image_paths = discover_images(
        dataset
    )

    path_by_name = {
        path.name: path
        for path in image_paths
    }

    layout = pd.read_csv(
        args.layout
    )

    layout_by_image = {
        str(row["image"]): row
        for row in layout.to_dict(
            orient="records"
        )
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print("=" * 72)
    print("LOCAL OVERLAP ALIGNMENT DIAGNOSTICS")
    print("=" * 72)

    for pair_text in args.pairs:
        image_a, image_b = (
            parse_pair(pair_text)
        )

        if image_a not in path_by_name:
            raise RuntimeError(
                f"Image not found: {image_a}"
            )

        if image_b not in path_by_name:
            raise RuntimeError(
                f"Image not found: {image_b}"
            )

        print(
            f"Inspecting "
            f"{image_a} <-> {image_b} ..."
        )

        record = inspect_pair(
            path_by_name[image_a],
            path_by_name[image_b],
            layout_by_image,
            args.output_dir,
            args.preview_max_width,
            args.preview_max_height,
        )

        records.append(
            record
        )

    summary = pd.DataFrame(
        records
    )

    summary_path = (
        args.output_dir
        / "summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 72)

    print(
        summary[
            [
                "image_a",
                "image_b",
                "overlap_width",
                "overlap_height",
                "overlap_pixels",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Saved to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()