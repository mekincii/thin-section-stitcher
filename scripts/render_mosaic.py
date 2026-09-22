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

    result = render_average_mosaic(
        image_paths,
        layout,
        render_scale=args.render_scale,
        padding_px=args.padding,
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


if __name__ == "__main__":
    main()