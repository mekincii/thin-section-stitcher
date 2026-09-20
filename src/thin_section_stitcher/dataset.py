from __future__ import annotations

import re
from pathlib import Path

import cv2
import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def natural_sort_key(path: Path) -> list[int | str]:
    """
    Sort filenames naturally.

    Example:
        1.jpg, 2.jpg, 10.jpg

    instead of:
        1.jpg, 10.jpg, 2.jpg
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def extract_sequence_number(path: Path) -> int | None:
    """
    Extract a sequence number from filenames such as '17.jpg'.

    Returns None for non-numeric filenames.
    """
    if path.stem.isdigit():
        return int(path.stem)

    return None


def discover_images(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    """
    Find supported image files in a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()

    images = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images, key=natural_sort_key)


def inspect_image(path: Path) -> dict:
    """
    Read one image and collect basic metadata.
    """
    file_size_bytes = path.stat().st_size

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        return {
            "filename": path.name,
            "sequence_number": extract_sequence_number(path),
            "path": str(path),
            "extension": path.suffix.lower(),
            "readable": False,
            "width": None,
            "height": None,
            "channels": None,
            "dtype": None,
            "file_size_bytes": file_size_bytes,
        }

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    return {
        "filename": path.name,
        "sequence_number": extract_sequence_number(path),
        "path": str(path),
        "extension": path.suffix.lower(),
        "readable": True,
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "file_size_bytes": file_size_bytes,
    }


def inspect_dataset(
    directory: Path,
    recursive: bool = False,
) -> pd.DataFrame:
    """
    Inspect every supported image in a dataset directory.
    """
    image_paths = discover_images(directory, recursive=recursive)

    records = [
        inspect_image(path)
        for path in image_paths
    ]

    return pd.DataFrame(records)


def print_dataset_summary(
    dataframe: pd.DataFrame,
    directory: Path,
) -> None:
    """
    Print a concise summary of dataset characteristics.
    """
    print()
    print("=" * 60)
    print("THIN SECTION DATASET INSPECTION")
    print("=" * 60)

    print(f"Dataset: {directory}")
    print(f"Images found: {len(dataframe)}")

    if dataframe.empty:
        print("No supported images found.")
        return

    readable = dataframe["readable"].sum()
    unreadable = len(dataframe) - readable

    print(f"Readable: {readable}")
    print(f"Unreadable: {unreadable}")

    total_size_mb = dataframe["file_size_bytes"].sum() / (1024**2)
    print(f"Total size: {total_size_mb:.2f} MB")

    print()
    print("Extensions:")
    print(dataframe["extension"].value_counts().to_string())

    sequence_numbers = (
        dataframe["sequence_number"]
        .dropna()
        .astype(int)
        .sort_values()
        .tolist()
    )

    if sequence_numbers:
        sequence_min = min(sequence_numbers)
        sequence_max = max(sequence_numbers)

        expected_numbers = set(range(sequence_min, sequence_max + 1))
        actual_numbers = set(sequence_numbers)
        missing_numbers = sorted(expected_numbers - actual_numbers)

        print()
        print("Filename sequence:")
        print(f"Range: {sequence_min} -> {sequence_max}")
        print(f"Numeric filenames: {len(sequence_numbers)}")

        if missing_numbers:
            print(
                "Missing numbers: "
                + ", ".join(map(str, missing_numbers))
            )
            print("Sequence contiguous: NO")
        else:
            print("Missing numbers: none")
            print("Sequence contiguous: YES")

    readable_df = dataframe[dataframe["readable"]]

    if readable_df.empty:
        return

    dimensions = readable_df[["width", "height"]].value_counts()

    print()
    print("Image dimensions:")
    print(dimensions.to_string())

    print()
    print("Channels:")
    print(readable_df["channels"].value_counts().to_string())

    print()
    print("Data types:")
    print(readable_df["dtype"].value_counts().to_string())

    consistent_dimensions = len(dimensions) == 1

    print()
    print(
        "Consistent dimensions: "
        f"{'YES' if consistent_dimensions else 'NO'}"
    )

    print("=" * 60)