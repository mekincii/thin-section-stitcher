import numpy as np

from thin_section_stitcher.mosaic import (
    pose_from_center,
    scaled_warp_matrix,
    source_feather_weights,
)


def test_pose_from_center_maps_image_center_to_global_center():
    transform = pose_from_center(
        center_x=100.0,
        center_y=200.0,
        rotation_deg=0.0,
        image_width=20,
        image_height=10,
    )

    local_center = np.array(
        [
            10.0,
            5.0,
            1.0,
        ]
    )

    mapped = (
        transform
        @ local_center
    )

    assert np.allclose(
        mapped[:2],
        [100.0, 200.0],
    )


def test_scaled_warp_matrix_scales_translation():
    transform = np.eye(
        3,
        dtype=float,
    )

    transform[0, 2] = 40.0
    transform[1, 2] = 80.0

    matrix = scaled_warp_matrix(
        transform,
        render_scale=0.25,
        canvas_origin_x=2,
        canvas_origin_y=3,
    )

    expected = np.array(
        [
            [1.0, 0.0, 8.0],
            [0.0, 1.0, 17.0],
        ]
    )

    assert np.allclose(
        matrix,
        expected,
    )


def test_feather_weights_reduce_source_edges():
    weights = source_feather_weights(
        image_width=100,
        image_height=80,
        feather_fraction=0.15,
        minimum_weight=0.05,
    )

    assert weights.shape == (
        80,
        100,
    )

    assert np.all(
        weights > 0.0
    )

    assert np.all(
        weights <= 1.0
    )

    center = weights[
        40,
        50,
    ]

    corner = weights[
        0,
        0,
    ]

    upper_edge = weights[
        0,
        50,
    ]

    assert np.isclose(
        center,
        1.0,
    )

    assert corner < center
    assert upper_edge < center