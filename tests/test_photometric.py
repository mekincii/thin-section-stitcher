import numpy as np

from thin_section_stitcher.photometric import (
    PhotometricCalibration,
    apply_photometric_calibration,
)


def test_identity_photometric_calibration_preserves_image():
    calibration = PhotometricCalibration(
        gains_by_image={
            "1.jpg": np.ones(
                3,
                dtype=np.float32,
            )
        },
        field_coefficients_bgr=np.zeros(
            (
                3,
                5,
            ),
            dtype=np.float32,
        ),
        max_field_correction=1.5,
    )

    image = np.array(
        [
            [
                [10, 20, 30],
                [40, 50, 60],
            ],
            [
                [70, 80, 90],
                [100, 110, 120],
            ],
        ],
        dtype=np.uint8,
    )

    corrected = (
        apply_photometric_calibration(
            "1.jpg",
            image,
            calibration,
        )
    )

    assert np.array_equal(
        corrected,
        image,
    )


def test_photometric_gains_follow_bgr_channel_order():
    calibration = PhotometricCalibration(
        gains_by_image={
            "1.jpg": np.array(
                [
                    1.2,
                    0.8,
                    1.0,
                ],
                dtype=np.float32,
            )
        },
        field_coefficients_bgr=np.zeros(
            (
                3,
                5,
            ),
            dtype=np.float32,
        ),
        max_field_correction=1.5,
    )

    image = np.full(
        (
            2,
            2,
            3,
        ),
        100,
        dtype=np.uint8,
    )

    corrected = (
        apply_photometric_calibration(
            "1.jpg",
            image,
            calibration,
        )
    )

    expected_pixel = np.array(
        [
            120,
            80,
            100,
        ],
        dtype=np.uint8,
    )

    assert np.all(
        corrected
        == expected_pixel
    )