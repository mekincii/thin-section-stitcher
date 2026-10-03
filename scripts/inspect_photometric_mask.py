from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from thin_section_stitcher.dataset import (
    discover_images,
)
from thin_section_stitcher.photometric_masking import (
    build_content_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect content-aware photometric masks."
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
        "--scale",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/photometric_mask_diagnostics"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = (
        args.dataset.expanduser().resolve()
    )

    paths = discover_images(
        dataset
    )

    by_name = {
        path.name: path
        for path in paths
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHOTOMETRIC MASK DIAGNOSTIC")
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
                f"Could not read {name}"
            )

        if args.scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_AREA,
            )

        mask = build_content_mask(
            image
        )

        masked = image.copy()
        masked[mask == 0] = 0

        coverage = (
            (mask > 0).sum()
            / mask.size
        )

        stem = Path(name).stem

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
                / f"{stem}_masked.png"
            ),
            masked,
        )

        print(
            f"{name:10s} "
            f"retained={coverage:.1%}"
        )

    print()
    print(
        f"Saved to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()