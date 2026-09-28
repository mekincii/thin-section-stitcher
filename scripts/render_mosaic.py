from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from thin_section_stitcher.dataset import (
    discover_images,
)
from thin_section_stitcher.mosaic import (
    render_average_mosaic,
)
from thin_section_stitcher.photometric import (
    apply_photometric_calibration,
    load_photometric_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a raw diagnostic thin-section mosaic "
            "from an optimized global layout."
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
        "--output",
        type=Path,
        default=Path(
            "outputs/raw_mosaic_preview.png"
        ),
    )

    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=Path(
            "outputs/raw_mosaic_coverage.png"
        ),
    )

    parser.add_argument(
        "--render-scale",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=64.0,
    )

    parser.add_argument(
        "--tiff-output",
        type=Path,
        default=Path(
            "outputs/raw_mosaic_preview.tif"
        ),
    )

    parser.add_argument(
        "--photometric-gains",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--photometric-field",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-field-correction",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--blend-mode",
        choices=[
            "average",
            "feather",
        ],
        default="average",
    )

    parser.add_argument(
        "--feather-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--minimum-feather-weight",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


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

    layout = pd.read_csv(
        args.layout
    )

    print("=" * 72)
    print("RAW MOSAIC RENDER")
    print("=" * 72)

    print(
        f"Images: {len(image_paths)}"
    )

    print(
        f"Layout: {args.layout}"
    )

    has_gains = (
        args.photometric_gains
        is not None
    )

    has_field = (
        args.photometric_field
        is not None
    )

    if has_gains != has_field:
        raise RuntimeError(
            "--photometric-gains and "
            "--photometric-field must "
            "be supplied together."
        )

    image_preprocessor = None

    if has_gains and has_field:
        calibration = (
            load_photometric_calibration(
                args.photometric_gains,
                args.photometric_field,
                max_field_correction=(
                    args.max_field_correction
                ),
            )
        )

        def image_preprocessor(
            image_name: str,
            image,
        ):
            return (
                apply_photometric_calibration(
                    image_name,
                    image,
                    calibration,
                )
            )

        print(
            "Photometric standardization: ON"
        )

        print(
            "Maximum spatial correction: "
            f"{args.max_field_correction:.2f}x"
        )

    else:
        print(
            "Photometric standardization: OFF"
        )

    print(
        f"Blend mode: {args.blend_mode}"
    )

    if args.blend_mode == "feather":
        print(
            "Feather fraction: "
            f"{args.feather_fraction:.2f}"
        )

        print(
            "Minimum feather weight: "
            f"{args.minimum_feather_weight:.2f}"
        )

    result = render_average_mosaic(
        image_paths,
        layout,
        render_scale=args.render_scale,
        padding_px=args.padding,
        image_preprocessor=(
            image_preprocessor
        ),
        blend_mode=args.blend_mode,
        feather_fraction=(
            args.feather_fraction
        ),
        minimum_feather_weight=(
            args.minimum_feather_weight
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.coverage_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(args.output),
        result.mosaic,
    ):
        raise RuntimeError(
            f"Could not save mosaic: "
            f"{args.output}"
        )

    if not cv2.imwrite(
        str(args.coverage_output),
        result.coverage_preview,
    ):
        raise RuntimeError(
            f"Could not save coverage image: "
            f"{args.coverage_output}"
        )

    canvas_pixels = (
        result.canvas_width
        * result.canvas_height
    )

    coverage_fraction = (
        result.covered_pixels
        / canvas_pixels
        if canvas_pixels
        else 0.0
    )

    args.tiff_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
            str(args.tiff_output),
            result.mosaic,
    ):
        raise RuntimeError(
            f"Could not save TIFF: "
            f"{args.tiff_output}"
        )

    print()
    print("=" * 72)
    print("RENDER COMPLETE")
    print("=" * 72)

    print(
        f"Canvas: "
        f"{result.canvas_width} "
        f"x {result.canvas_height}"
    )

    print(
        f"Covered canvas area: "
        f"{coverage_fraction:.1%}"
    )

    print(
        f"Maximum image overlap: "
        f"{result.max_overlap}"
    )

    print()
    print(
        f"Mosaic: "
        f"{args.output.resolve()}"
    )

    print(
        f"Coverage: "
        f"{args.coverage_output.resolve()}"
    )

    print(
        f"TIFF: "
        f"{args.tiff_output.resolve()}"
    )


if __name__ == "__main__":
    main()