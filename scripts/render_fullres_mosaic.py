from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thin_section_stitcher.dataset import (
    discover_images,
)
from thin_section_stitcher.mosaic import (
    render_tiled_mosaic_to_tiff,
)
from thin_section_stitcher.photometric import (
    apply_photometric_calibration,
    load_photometric_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the final full-resolution "
            "standardized and feather-blended "
            "thin-section mosaic."
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
            "outputs/"
            "optimized_global_layout.csv"
        ),
    )

    parser.add_argument(
        "--photometric-gains",
        type=Path,
        default=Path(
            "outputs/"
            "final_photometric_gains.csv"
        ),
    )

    parser.add_argument(
        "--photometric-field",
        type=Path,
        default=Path(
            "outputs/"
            "photometric_field.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/"
            "thin_section_fullres.tif"
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=64.0,
    )

    parser.add_argument(
        "--max-field-correction",
        type=float,
        default=1.5,
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

    print("=" * 72)
    print("FINAL FULL-RESOLUTION MOSAIC")
    print("=" * 72)

    print(
        f"Images: {len(image_paths)}"
    )

    print(
        "Photometric standardization: ON"
    )

    print(
        "Maximum spatial correction: "
        f"{args.max_field_correction:.2f}x"
    )

    print(
        "Blend mode: feather"
    )

    print(
        "Feather fraction: "
        f"{args.feather_fraction:.2f}"
    )

    print(
        "Minimum feather weight: "
        f"{args.minimum_feather_weight:.2f}"
    )

    result = (
        render_tiled_mosaic_to_tiff(
            image_paths,
            layout,
            output_path=args.output,
            padding_px=args.padding,
            tile_size=args.tile_size,
            image_preprocessor=(
                image_preprocessor
            ),
            blend_mode="feather",
            feather_fraction=(
                args.feather_fraction
            ),
            minimum_feather_weight=(
                args.minimum_feather_weight
            ),
        )
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

    size_gib = (
        result.output_path
        .stat()
        .st_size
        / (1024**3)
    )

    print()
    print("=" * 72)
    print("FULL-RESOLUTION RENDER COMPLETE")
    print("=" * 72)

    print(
        f"Canvas: "
        f"{result.canvas_width} x "
        f"{result.canvas_height}"
    )

    print(
        f"Tiles rendered: "
        f"{result.tile_count}"
    )

    print(
        f"Covered canvas area: "
        f"{coverage_fraction:.1%}"
    )

    print(
        f"BigTIFF: "
        f"{result.bigtiff}"
    )

    print(
        f"File size: "
        f"{size_gib:.2f} GiB"
    )

    print(
        f"TIFF: "
        f"{result.output_path}"
    )


if __name__ == "__main__":
    main()