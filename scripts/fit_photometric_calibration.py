from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

CHANNELS = {
    "blue": "log_ratio_blue",
    "green": "log_ratio_green",
    "red": "log_ratio_red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the final overlap-derived photometric calibration "
            "for thin-section mosaic rendering."
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
        "--gains-output",
        type=Path,
        default=Path(
            "outputs/final_photometric_gains.csv"
        ),
    )

    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path(
            "outputs/photometric_field.json"
        ),
    )

    parser.add_argument(
        "--exposure-regularization",
        type=float,
        default=0.10,
        help=(
            "Weak regularization on per-image log exposure terms."
        ),
    )

    return parser.parse_args()


def field_basis(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
    )


def measurement_weights(
    measurements: pd.DataFrame,
) -> np.ndarray:
    geometric = np.clip(
        measurements[
            "geometric_quality"
        ].to_numpy(dtype=float),
        0.1,
        1.0,
    )

    support = np.clip(
        measurements[
            "valid_pixels"
        ].to_numpy(dtype=float)
        / 50_000.0,
        0.25,
        1.0,
    )

    return np.sqrt(
        geometric * support
    )


def image_names(
    measurements: pd.DataFrame,
) -> list[str]:
    names = (
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

    return sorted(
        names,
        key=lambda name: int(
            Path(name).stem
        ),
    )


def fit_channel(
    measurements: pd.DataFrame,
    names: list[str],
    target_column: str,
    regularization: float,
) -> tuple[
    dict[str, float],
    np.ndarray,
    dict[str, float],
]:
    """
    Fit:

        observed log(B/A)
        =
        exposure_B - exposure_A
        + field(B position) - field(A position)

    The returned per-image values are correction gains, i.e.
    exp(-exposure).
    """
    index = {
        name: i
        for i, name in enumerate(names)
    }

    image_count = len(names)

    target = measurements[
        target_column
    ].to_numpy(dtype=float)

    weights = measurement_weights(
        measurements
    )

    basis_a = field_basis(
        measurements[
            "centroid_ax"
        ].to_numpy(dtype=float),
        measurements[
            "centroid_ay"
        ].to_numpy(dtype=float),
    )

    basis_b = field_basis(
        measurements[
            "centroid_bx"
        ].to_numpy(dtype=float),
        measurements[
            "centroid_by"
        ].to_numpy(dtype=float),
    )

    field_difference = (
        basis_b - basis_a
    )

    records = measurements.to_dict(
        orient="records"
    )

    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        exposures = parameters[
            :image_count
        ]

        field = parameters[
            image_count:
        ]

        prediction = np.empty(
            len(records),
            dtype=float,
        )

        for row_index, row in enumerate(
            records
        ):
            image_a = str(
                row["image_a"]
            )

            image_b = str(
                row["image_b"]
            )

            prediction[
                row_index
            ] = (
                exposures[
                    index[image_b]
                ]
                - exposures[
                    index[image_a]
                ]
                + field_difference[
                    row_index
                ]
                @ field
            )

        data_residuals = (
            weights
            * (
                prediction
                - target
            )
        )

        # Weakly prefer small frame-to-frame exposure corrections.
        regularization_residuals = (
            np.sqrt(
                regularization
            )
            * exposures
        )

        # Fix the otherwise arbitrary global exposure level.
        gauge_residual = np.array(
            [
                np.mean(
                    exposures
                )
                * 10.0
            ],
            dtype=float,
        )

        return np.concatenate(
            [
                data_residuals,
                regularization_residuals,
                gauge_residual,
            ]
        )

    result = least_squares(
        residual_function,
        np.zeros(
            image_count + 5,
            dtype=float,
        ),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=1500,
    )

    if not result.success:
        raise RuntimeError(
            "Final photometric calibration failed: "
            f"{result.message}"
        )

    exposures = result.x[
        :image_count
    ].copy()

    field = result.x[
        image_count:
    ].copy()

    # Remove the arbitrary global brightness level.
    exposures -= np.median(
        exposures
    )

    correction_gains = {
        name: float(
            np.exp(
                -exposures[
                    index[name]
                ]
            )
        )
        for name in names
    }

    prediction = np.empty(
        len(records),
        dtype=float,
    )

    for row_index, row in enumerate(
        records
    ):
        image_a = str(
            row["image_a"]
        )

        image_b = str(
            row["image_b"]
        )

        prediction[
            row_index
        ] = (
            exposures[
                index[image_b]
            ]
            - exposures[
                index[image_a]
            ]
            + field_difference[
                row_index
            ]
            @ field
        )

    absolute_error = np.abs(
        prediction
        - target
    )

    statistics = {
        "median_residual": float(
            np.median(
                absolute_error
            )
        ),
        "p90_residual": float(
            np.percentile(
                absolute_error,
                90,
            )
        ),
        "p95_residual": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),
    }

    return (
        correction_gains,
        field,
        statistics,
    )


def field_range(
    coefficients: np.ndarray,
) -> tuple[float, float]:
    axis = np.linspace(
        -1.0,
        1.0,
        101,
    )

    x, y = np.meshgrid(
        axis,
        axis,
    )

    values = (
        field_basis(
            x.ravel(),
            y.ravel(),
        )
        @ coefficients
    )

    factors = np.exp(
        values
    )

    return (
        float(
            factors.min()
        ),
        float(
            factors.max()
        ),
    )


def main() -> None:
    args = parse_args()

    measurements = pd.read_csv(
        args.measurements
    )

    names = image_names(
        measurements
    )

    gains_by_channel: dict[
        str,
        dict[str, float],
    ] = {}

    fields: dict[
        str,
        list[float],
    ] = {}

    print("=" * 72)
    print("FINAL PHOTOMETRIC CALIBRATION")
    print("=" * 72)

    print(
        f"Measurements: "
        f"{len(measurements)}"
    )

    print(
        f"Images: "
        f"{len(names)}"
    )

    print(
        "Exposure regularization: "
        f"{args.exposure_regularization:.3f}"
    )

    for channel_name, target_column in (
        CHANNELS.items()
    ):
        (
            correction_gains,
            coefficients,
            statistics,
        ) = fit_channel(
            measurements,
            names,
            target_column,
            regularization=(
                args.exposure_regularization
            ),
        )

        gains_by_channel[
            channel_name
        ] = correction_gains

        fields[
            channel_name
        ] = [
            float(value)
            for value in coefficients
        ]

        field_min, field_max = (
            field_range(
                coefficients
            )
        )

        gain_values = np.array(
            list(
                correction_gains.values()
            ),
            dtype=float,
        )

        print()
        print(
            channel_name.upper()
        )

        print(
            "  correction gains: "
            f"{gain_values.min():.3f}"
            " -> "
            f"{gain_values.max():.3f}"
        )

        print(
            "  field illumination: "
            f"{field_min:.3f}"
            " -> "
            f"{field_max:.3f}"
        )

        print(
            "  residuals: "
            f"median="
            f"{statistics['median_residual']:.4f} "
            f"p90="
            f"{statistics['p90_residual']:.4f} "
            f"p95="
            f"{statistics['p95_residual']:.4f}"
        )

        print(
            "  coefficients: "
            + " ".join(
                f"{value:+.5f}"
                for value
                in coefficients
            )
        )

    gain_frame = pd.DataFrame(
        [
            {
                "image": name,
                "gain_blue": (
                    gains_by_channel[
                        "blue"
                    ][name]
                ),
                "gain_green": (
                    gains_by_channel[
                        "green"
                    ][name]
                ),
                "gain_red": (
                    gains_by_channel[
                        "red"
                    ][name]
                ),
            }
            for name in names
        ]
    )

    args.gains_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gain_frame.to_csv(
        args.gains_output,
        index=False,
    )

    args.field_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_document = {
        "coordinate_system": (
            "normalized source image coordinates; "
            "x,y in approximately [-1,1], "
            "center=(0,0)"
        ),
        "basis": [
            "x",
            "y",
            "x^2",
            "x*y",
            "y^2",
        ],
        "model": (
            "observed = standardized * "
            "exp(field(x,y)) / correction_gain"
        ),
        "channels": fields,
    }

    args.field_output.write_text(
        json.dumps(
            field_document,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CALIBRATION SAVED")
    print("=" * 72)

    print(
        f"Gains: "
        f"{args.gains_output.resolve()}"
    )

    print(
        f"Field: "
        f"{args.field_output.resolve()}"
    )


if __name__ == "__main__":
    main()