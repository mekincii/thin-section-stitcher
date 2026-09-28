from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEFAULT_TARGETS = {
    "01_center": (0.50, 0.50),
    "02_upper_left": (0.32, 0.30),
    "03_upper_right": (0.65, 0.30),
    "04_right_middle": (0.77, 0.50),
    "05_lower_center": (0.53, 0.73),
    "06_left_middle": (0.29, 0.50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract lossless native-resolution QA crops "
            "directly from the final mosaic TIFF."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/thin_section_fullres.tif"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/fullres_qa"
        ),
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=1536,
    )

    return parser.parse_args()


def extract_crop(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    half = crop_size // 2

    x0 = max(
        0,
        center_x - half,
    )

    y0 = max(
        0,
        center_y - half,
    )

    x1 = min(
        width,
        x0 + crop_size,
    )

    y1 = min(
        height,
        y0 + crop_size,
    )

    # Preserve requested size near outer canvas boundaries.
    x0 = max(
        0,
        x1 - crop_size,
    )

    y0 = max(
        0,
        y1 - crop_size,
    )

    return image[
        y0:y1,
        x0:x1,
    ]


def coverage_fraction(
    crop: np.ndarray,
) -> float:
    """
    Estimate how much of the crop contains rendered mosaic data.

    The final canvas background is pure black.
    """
    occupied = np.any(
        crop > 0,
        axis=2,
    )

    return float(
        occupied.mean()
    )


def main() -> None:
    args = parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if args.crop_size <= 0:
        raise ValueError(
            "crop-size must be positive."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("FULL-RESOLUTION TIFF QA CROPS")
    print("=" * 72)

    print(
        f"Input: {input_path}"
    )

    image = tifffile.memmap(
        input_path,
        mode="r",
    )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise RuntimeError(
            "Expected an RGB TIFF."
        )

    height, width = image.shape[:2]

    print(
        f"TIFF dimensions: "
        f"{width} x {height}"
    )

    print(
        f"Crop size: "
        f"{args.crop_size} x "
        f"{args.crop_size}"
    )

    print()

    for name, (
        fraction_x,
        fraction_y,
    ) in DEFAULT_TARGETS.items():
        center_x = round(
            width * fraction_x
        )

        center_y = round(
            height * fraction_y
        )

        crop_rgb = extract_crop(
            image,
            center_x=center_x,
            center_y=center_y,
            crop_size=args.crop_size,
        )

        fraction = coverage_fraction(
            crop_rgb
        )

        # TIFF is RGB. OpenCV expects BGR when writing.
        crop_bgr = crop_rgb[
            :,
            :,
            ::-1,
        ]

        output_path = (
            output_dir
            / f"{name}.png"
        )

        if not cv2.imwrite(
            str(output_path),
            crop_bgr,
        ):
            raise RuntimeError(
                f"Could not save: "
                f"{output_path}"
            )

        print(
            f"{name:18s} "
            f"center=({center_x:5d}, "
            f"{center_y:5d})  "
            f"coverage={fraction:6.1%}"
        )

    print()
    print("=" * 72)
    print("QA CROPS COMPLETE")
    print("=" * 72)

    print(
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()