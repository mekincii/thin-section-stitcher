from __future__ import annotations

import argparse
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

FIELD_COLUMNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a shared quadratic microscope "
            "illumination field improves overlap photometric "
            "consistency beyond per-image exposure terms."
        )
    )

    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path(
            "outputs/photometric_measurements.csv"
        ),
    )

    parser.add_argument(
        "--root",
        type=str,
        default="60.jpg",
    )

    return parser.parse_args()


def field_basis(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Quadratic illumination basis without a constant term.

    Coordinates are normalized so:
        center = (0, 0)
        image edges ~= +/-1
    """
    return np.column_stack(
        [
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
    )


def image_names_from_measurements(
    measurements: pd.DataFrame,
) -> list[str]:
    return sorted(
        set(
            measurements[
                "image_a"
            ].astype(str)
        )
        | set(
            measurements[
                "image_b"
            ].astype(str)
        )
    )


def build_design_matrix(
    measurements: pd.DataFrame,
    image_names: list[str],
    root: str,
    include_field: bool,
) -> tuple[
    np.ndarray,
    dict[str, int],
]:
    """
    Model observed log intensity ratio:

        log(B / A)
        =
        exposure_B - exposure_A
        + field(position_B)
        - field(position_A)
    """
    free_images = [
        image_name
        for image_name in image_names
        if image_name != root
    ]

    image_index = {
        image_name: index
        for index, image_name
        in enumerate(
            free_images
        )
    }

    parameter_count = len(
        free_images
    )

    if include_field:
        parameter_count += (
            FIELD_COLUMNS
        )

    design = np.zeros(
        (
            len(measurements),
            parameter_count,
        ),
        dtype=float,
    )

    for row_index, row in enumerate(
        measurements.to_dict(
            orient="records"
        )
    ):
        image_a = str(
            row["image_a"]
        )

        image_b = str(
            row["image_b"]
        )

        if image_a != root:
            design[
                row_index,
                image_index[image_a],
            ] -= 1.0

        if image_b != root:
            design[
                row_index,
                image_index[image_b],
            ] += 1.0

    if include_field:
        basis_a = field_basis(
            measurements[
                "centroid_ax"
            ].to_numpy(
                dtype=float
            ),
            measurements[
                "centroid_ay"
            ].to_numpy(
                dtype=float
            ),
        )

        basis_b = field_basis(
            measurements[
                "centroid_bx"
            ].to_numpy(
                dtype=float
            ),
            measurements[
                "centroid_by"
            ].to_numpy(
                dtype=float
            ),
        )

        design[
            :,
            len(free_images):,
        ] = (
            basis_b
            - basis_a
        )

    return (
        design,
        image_index,
    )


def measurement_weights(
    measurements: pd.DataFrame,
) -> np.ndarray:
    geometric = np.clip(
        measurements[
            "geometric_quality"
        ].to_numpy(
            dtype=float
        ),
        0.1,
        1.0,
    )

    pixel_support = np.clip(
        measurements[
            "valid_pixels"
        ].to_numpy(
            dtype=float
        )
        / 50_000.0,
        0.25,
        1.0,
    )

    return np.sqrt(
        geometric
        * pixel_support
    )


def fit_model(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Robustly fit one channel.
    """
    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        return (
            weights
            * (
                design
                @ parameters
                - target
            )
        )

    result = least_squares(
        residual_function,
        np.zeros(
            design.shape[1],
            dtype=float,
        ),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=1000,
    )

    if not result.success:
        raise RuntimeError(
            "Photometric diagnostic fit failed: "
            f"{result.message}"
        )

    return result.x


def residual_statistics(
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    absolute = np.abs(
        target
        - predicted
    )

    return {
        "median": float(
            np.median(
                absolute
            )
        ),
        "p90": float(
            np.percentile(
                absolute,
                90,
            )
        ),
        "p95": float(
            np.percentile(
                absolute,
                95,
            )
        ),
    }


def field_range(
    coefficients: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """
    Evaluate the fitted illumination field over the normalized
    microscope frame.

    Returned values are multiplicative illumination factors
    relative to the image center.
    """
    axis = np.linspace(
        -1.0,
        1.0,
        81,
    )

    grid_x, grid_y = np.meshgrid(
        axis,
        axis,
    )

    basis = field_basis(
        grid_x.ravel(),
        grid_y.ravel(),
    )

    log_field = (
        basis
        @ coefficients
    )

    return (
        exp(
            float(
                log_field.min()
            )
        ),
        exp(
            float(
                log_field.max()
            )
        ),
    )


def analyze_channel(
    measurements: pd.DataFrame,
    image_names: list[str],
    root: str,
    target_column: str,
) -> dict:
    target = measurements[
        target_column
    ].to_numpy(
        dtype=float
    )

    weights = measurement_weights(
        measurements
    )

    gain_design, _ = (
        build_design_matrix(
            measurements,
            image_names,
            root=root,
            include_field=False,
        )
    )

    gain_parameters = fit_model(
        gain_design,
        target,
        weights,
    )

    gain_prediction = (
        gain_design
        @ gain_parameters
    )

    gain_stats = (
        residual_statistics(
            target,
            gain_prediction,
        )
    )

    field_design, image_index = (
        build_design_matrix(
            measurements,
            image_names,
            root=root,
            include_field=True,
        )
    )

    field_parameters = fit_model(
        field_design,
        target,
        weights,
    )

    field_prediction = (
        field_design
        @ field_parameters
    )

    field_stats = (
        residual_statistics(
            target,
            field_prediction,
        )
    )

    field_coefficients = (
        field_parameters[
            len(image_index):
        ]
    )

    (
        minimum_field,
        maximum_field,
    ) = field_range(
        field_coefficients
    )

    improvement = (
        1.0
        - (
            field_stats[
                "median"
            ]
            / gain_stats[
                "median"
            ]
        )
        if gain_stats[
            "median"
        ] > 0
        else 0.0
    )

    return {
        "gain_median": (
            gain_stats["median"]
        ),
        "gain_p90": (
            gain_stats["p90"]
        ),
        "gain_p95": (
            gain_stats["p95"]
        ),
        "field_median": (
            field_stats["median"]
        ),
        "field_p90": (
            field_stats["p90"]
        ),
        "field_p95": (
            field_stats["p95"]
        ),
        "median_improvement": (
            improvement
        ),
        "field_min": minimum_field,
        "field_max": maximum_field,
        "coefficients": (
            field_coefficients
        ),
    }


def main() -> None:
    args = parse_args()

    measurements = pd.read_csv(
        args.measurements
    )

    required_columns = {
        "image_a",
        "image_b",
        "centroid_ax",
        "centroid_ay",
        "centroid_bx",
        "centroid_by",
        "valid_pixels",
        "geometric_quality",
    }

    missing = (
        required_columns
        - set(
            measurements.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Measurements are missing "
            f"required columns: {sorted(missing)}"
        )

    image_names = (
        image_names_from_measurements(
            measurements
        )
    )

    if args.root not in image_names:
        raise RuntimeError(
            f"Root image not found: "
            f"{args.root}"
        )

    channels = {
        "blue": "log_ratio_blue",
        "green": "log_ratio_green",
        "red": "log_ratio_red",
    }

    print("=" * 72)
    print("PHOTOMETRIC FIELD DIAGNOSTIC")
    print("=" * 72)

    print(
        f"Measurements: "
        f"{len(measurements)}"
    )

    print(
        f"Images: "
        f"{len(image_names)}"
    )

    print(
        f"Reference image: "
        f"{args.root}"
    )

    for channel_name, column in (
        channels.items()
    ):
        result = analyze_channel(
            measurements,
            image_names,
            root=args.root,
            target_column=column,
        )

        print()
        print(
            channel_name.upper()
        )

        print(
            "  gain-only:"
            f" median={result['gain_median']:.4f}"
            f" p90={result['gain_p90']:.4f}"
            f" p95={result['gain_p95']:.4f}"
        )

        print(
            "  gain+field:"
            f" median={result['field_median']:.4f}"
            f" p90={result['field_p90']:.4f}"
            f" p95={result['field_p95']:.4f}"
        )

        print(
            "  median residual improvement: "
            f"{result['median_improvement']:.1%}"
        )

        print(
            "  fitted illumination range: "
            f"{result['field_min']:.3f}"
            " -> "
            f"{result['field_max']:.3f}"
        )

        coefficients = result[
            "coefficients"
        ]

        print(
            "  field coefficients "
            "[x, y, x², xy, y²]: "
            + " ".join(
                f"{value:+.5f}"
                for value
                in coefficients
            )
        )


if __name__ == "__main__":
    main()