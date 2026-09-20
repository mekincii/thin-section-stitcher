from __future__ import annotations

import argparse
from pathlib import Path

from thin_section_stitcher.dataset import (
    inspect_dataset,
    print_dataset_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a folder containing petrographic "
            "thin-section microscope images."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the microscope image directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_report.csv"),
        help="CSV report destination.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser().resolve()

    dataframe = inspect_dataset(
        dataset_path,
        recursive=args.recursive,
    )

    print_dataset_summary(
        dataframe,
        dataset_path,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        args.output,
        index=False,
    )

    print()
    print(f"Report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()