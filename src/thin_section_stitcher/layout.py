from __future__ import annotations

from math import atan2, cos, degrees, radians, sin

import networkx as nx
import numpy as np
from scipy.optimize import least_squares


def relative_transform(
    dx: float,
    dy: float,
    rotation_deg: float,
    scale: float,
    image_scale: float,
) -> np.ndarray:
    """
    Build a homogeneous transform mapping coordinates in image A
    into coordinates in image B.

    Translation is converted back to original-image pixels.
    """
    angle = radians(rotation_deg)

    a = scale * cos(angle)
    b = scale * sin(angle)

    return np.array(
        [
            [a, -b, dx / image_scale],
            [b, a, dy / image_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def edge_weight(
    inliers: int,
    inlier_ratio: float,
) -> float:
    """
    Confidence weight used for the initial spanning tree.
    """
    return float(inliers) * float(inlier_ratio)


def build_layout_graph(
    overlap_graph: nx.Graph,
) -> nx.Graph:
    """
    Copy the overlap graph and add layout confidence weights.
    """
    graph = overlap_graph.copy()

    for _, _, data in graph.edges(data=True):
        data["weight"] = edge_weight(
            int(data["inliers"]),
            float(data["inlier_ratio"]),
        )

    return graph


def maximum_confidence_tree(
    graph: nx.Graph,
) -> nx.Graph:
    """
    Select a high-confidence backbone connecting every image.
    """
    if not nx.is_connected(graph):
        raise ValueError(
            "Overlap graph must be connected before layout initialization."
        )

    return nx.maximum_spanning_tree(
        graph,
        weight="weight",
    )


def choose_root(
    tree: nx.Graph,
) -> str:
    """
    Choose a central node to reduce transform propagation depth.
    """
    centers = nx.center(tree)

    return str(
        max(
            centers,
            key=lambda node: tree.degree(node),
        )
    )


def propagate_global_transforms(
    tree: nx.Graph,
    root: str,
    image_scale: float,
) -> dict[str, np.ndarray]:
    """
    Propagate pairwise transforms through the spanning tree.

    Each resulting matrix maps original image coordinates into
    one shared global coordinate system.
    """
    transforms = {
        root: np.eye(3, dtype=float)
    }

    for parent, child in nx.bfs_edges(tree, root):
        data = tree.edges[parent, child]

        transform_ab = relative_transform(
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            rotation_deg=float(data["rotation_deg"]),
            scale=float(data["scale"]),
            image_scale=image_scale,
        )

        image_a = str(data.get("image_a", parent))
        image_b = str(data.get("image_b", child))

        if parent == image_a and child == image_b:
            step = np.linalg.inv(transform_ab)
        elif parent == image_b and child == image_a:
            step = transform_ab
        else:
            raise RuntimeError(
                f"Edge orientation missing for {parent} <-> {child}"
            )

        transforms[child] = (
            transforms[parent] @ step
        )

    return transforms


def transform_point(
    transform: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    """
    Transform one 2D point into global coordinates.
    """
    point = transform @ np.array(
        [x, y, 1.0],
        dtype=float,
    )

    return float(point[0]), float(point[1])


def transform_rotation_deg(
    transform: np.ndarray,
) -> float:
    """
    Extract global image rotation from a transform.
    """
    return degrees(
        atan2(
            transform[1, 0],
            transform[0, 0],
        )
    )


def pose_matrix(
    x: float,
    y: float,
    rotation_deg: float,
) -> np.ndarray:
    """
    Construct a rigid 2D image-to-global transform.
    """
    angle = radians(rotation_deg)

    return np.array(
        [
            [cos(angle), -sin(angle), x],
            [sin(angle), cos(angle), y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def optimize_global_poses(
    graph: nx.Graph,
    initial_transforms: dict[str, np.ndarray],
    root: str,
    image_width: int,
    image_height: int,
    image_scale: float,
) -> dict[str, np.ndarray]:
    """
    Refine all image poses simultaneously using every graph edge.

    The root pose is fixed to remove global translation/rotation ambiguity.
    Each image has three free parameters:
    global x, global y, global rotation.
    """
    nodes = [
        str(node)
        for node in graph.nodes
        if str(node) != root
    ]

    index = {
        node: offset
        for offset, node in enumerate(nodes)
    }

    initial_parameters = np.zeros(
        len(nodes) * 3,
        dtype=float,
    )

    for node in nodes:
        transform = initial_transforms[node]

        offset = index[node] * 3

        initial_parameters[offset] = float(
            transform[0, 2]
        )
        initial_parameters[offset + 1] = float(
            transform[1, 2]
        )
        initial_parameters[offset + 2] = (
            transform_rotation_deg(transform)
        )

    root_transform = initial_transforms[root]

    center = np.array(
        [
            image_width / 2.0,
            image_height / 2.0,
            1.0,
        ],
        dtype=float,
    )

    x_axis = np.array(
        [
            image_width,
            image_height / 2.0,
            1.0,
        ],
        dtype=float,
    )

    y_axis = np.array(
        [
            image_width / 2.0,
            image_height,
            1.0,
        ],
        dtype=float,
    )

    reference_points = (
        center,
        x_axis,
        y_axis,
    )

    def unpack(
        parameters: np.ndarray,
    ) -> dict[str, np.ndarray]:
        transforms = {
            root: root_transform,
        }

        for node in nodes:
            offset = index[node] * 3

            transforms[node] = pose_matrix(
                x=float(parameters[offset]),
                y=float(parameters[offset + 1]),
                rotation_deg=float(
                    parameters[offset + 2]
                ),
            )

        return transforms

    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        transforms = unpack(parameters)

        residuals: list[float] = []

        for node_a, node_b, data in graph.edges(
            data=True
        ):
            image_a = str(data["image_a"])
            image_b = str(data["image_b"])

            measured = relative_transform(
                dx=float(data["dx"]),
                dy=float(data["dy"]),
                rotation_deg=float(
                    data["rotation_deg"]
                ),
                scale=1.0,
                image_scale=image_scale,
            )

            predicted = (
                np.linalg.inv(
                    transforms[image_b]
                )
                @ transforms[image_a]
            )

            weight = np.sqrt(
                max(
                    float(data["inlier_ratio"]),
                    0.01,
                )
            )

            for point in reference_points:
                measured_point = measured @ point
                predicted_point = predicted @ point

                residuals.append(
                    weight
                    * float(
                        predicted_point[0]
                        - measured_point[0]
                    )
                )

                residuals.append(
                    weight
                    * float(
                        predicted_point[1]
                        - measured_point[1]
                    )
                )

        return np.asarray(
            residuals,
            dtype=float,
        )

    result = least_squares(
        residual_function,
        initial_parameters,
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=300,
        verbose=1,
    )

    if not result.success:
        raise RuntimeError(
            f"Global optimization failed: "
            f"{result.message}"
        )

    return unpack(result.x)