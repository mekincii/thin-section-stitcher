from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from thin_section_stitcher.dataset import (
    discover_images,
)
from thin_section_stitcher.photometric_flatfield import (
    correct_image_flatfield,
)
from thin_section_stitcher.photometric_masking import (
    build_content_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect conservative per-image flat-field correction."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image numbers, for example 1 39 55.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/flatfield_diagnostics"
        ),
    )

    return parser.parse_args()


def normalize_visualization(
    array: np.ndarray,
) -> np.ndarray:
    """
    Robustly map a float array to uint8 for diagnostic display.
    """
    low = float(
        np.percentile(
            array,
            2,
        )
    )

    high = float(
        np.percentile(
            array,
            98,
        )
    )

    if high <= low:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = (
        (array - low)
        / (high - low)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).astype(
        np.uint8
    )


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

    by_name = {
        path.name: path
        for path in image_paths
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("FLAT-FIELD DIAGNOSTIC")
    print("=" * 72)

    for value in args.images:
        name = f"{int(value)}.jpg"

        if name not in by_name:
            raise RuntimeError(
                f"Image not found: {name}"
            )

        image = cv2.imread(
            str(by_name[name]),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {name}"
            )

        mask = build_content_mask(
            image
        )

        (
            corrected,
            field,
            correction,
            support,
        ) = correct_image_flatfield(
            image,
            mask,
            downsample_factor=8,
            blur_sigma_small=30.0,
            min_support=0.05,
            full_support=0.50,
            min_correction=0.95,
            max_correction=1.06,
        )

        stem = Path(name).stem

        field_mean = field.mean(
            axis=2
        )

        correction_mean = correction.mean(
            axis=2
        )

        field_preview = (
            normalize_visualization(
                field_mean
            )
        )

        correction_preview = (
            normalize_visualization(
                correction_mean
            )
        )

        support_preview = np.clip(
            support * 255.0,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_original.png"
            ),
            image,
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_mask.png"
            ),
            mask,
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_field.png"
            ),
            field_preview,
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_correction.png"
            ),
            correction_preview,
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_support.png"
            ),
            support_preview,
        )

        cv2.imwrite(
            str(
                args.output_dir
                / f"{stem}_corrected.png"
            ),
            corrected,
        )

        valid = support >= 0.05

        if np.any(valid):
            valid_correction = correction[
                valid,
                :,
            ]

            lower_fraction = float(
                np.mean(
                    valid_correction <= 0.9501
                )
            )

            upper_fraction = float(
                np.mean(
                    valid_correction >= 1.0599
                )
            )

            correction_min = float(
                valid_correction.min()
            )

            correction_median = float(
                np.median(
                    valid_correction
                )
            )

            correction_max = float(
                valid_correction.max()
            )

        else:
            correction_min = 1.0
            correction_median = 1.0
            correction_max = 1.0

        print(
            f"{name:10s} "
            f"correction="
            f"{correction_min:.3f} "
            f"/ {correction_median:.3f} "
            f"/ {correction_max:.3f}  "
            f"at_min={lower_fraction:.1%}  "
            f"at_max={upper_fraction:.1%}"
        )

    print()
    print(
        f"Saved to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()