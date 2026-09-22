from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the initial global microscope-image layout."
    )

    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("outputs/initial_global_layout.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    layout = pd.read_csv(args.layout)

    _fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        layout["center_x"],
        layout["center_y"],
        s=20,
    )

    for row in layout.itertuples(index=False):
        image_number = Path(row.image).stem

        ax.text(
            row.center_x,
            row.center_y,
            image_number,
            fontsize=6,
        )

    ax.set_title("Global thin-section image layout")
    ax.set_xlabel("Global x (original-image pixels)")
    ax.set_ylabel("Global y (original-image pixels)")
    ax.set_aspect("equal")

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()