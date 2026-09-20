from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def classify_confidence(
    inliers: int,
    inlier_ratio: float,
) -> str:
    """
    Classify an overlap using the current provisional thresholds.
    """
    if inliers >= 50 and inlier_ratio >= 0.70:
        return "high"

    if inliers >= 20 and inlier_ratio >= 0.50:
        return "medium"

    return "weak"


def is_consistency_warning(
    translation_error_px: float,
    rotation_error_deg: float,
    scale_error: float,
) -> bool:
    """
    Flag unusually inconsistent cross-scale geometry.
    """
    return (
        translation_error_px > 10.0
        or rotation_error_deg > 0.30
        or scale_error > 0.005
    )


def prepare_overlap_edges(
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add confidence and consistency information to candidate overlaps.
    """
    dataframe = consistency.copy()

    dataframe["confidence"] = dataframe.apply(
        lambda row: classify_confidence(
            int(row["ransac_inliers_verification"]),
            float(row["inlier_ratio_verification"]),
        ),
        axis=1,
    )

    dataframe["consistency_warning"] = dataframe.apply(
        lambda row: is_consistency_warning(
            float(row["translation_error_px"]),
            float(row["rotation_error_deg"]),
            float(row["scale_error"]),
        ),
        axis=1,
    )

    dataframe["trusted"] = (
        dataframe["confidence"].isin(["high", "medium"])
        & ~dataframe["consistency_warning"]
    )

    return dataframe


def build_overlap_graph(
    image_names: list[str],
    edges: pd.DataFrame,
    confidence_levels: set[str] | None = None,
    exclude_warnings: bool = True,
) -> nx.Graph:
    """
    Build an undirected overlap graph.

    Every microscope image is included as a node, even if it has no
    accepted overlap.
    """
    graph = nx.Graph()

    graph.add_nodes_from(image_names)

    selected = edges.copy()

    if confidence_levels is not None:
        selected = selected[
            selected["confidence"].isin(confidence_levels)
        ]

    if exclude_warnings:
        selected = selected[
            ~selected["consistency_warning"]
        ]

    for row in selected.to_dict(orient="records"):
        graph.add_edge(
            str(row["image_a"]),
            str(row["image_b"]),
            confidence=str(row["confidence"]),
            inliers=int(row["ransac_inliers_verification"]),
            inlier_ratio=float(row["inlier_ratio_verification"]),
            dx=float(row["dx_verification"]),
            dy=float(row["dy_verification"]),
            rotation_deg=float(row["rotation_deg_verification"]),
            scale=float(row["estimated_scale_verification"]),
            translation_error_px=float(row["translation_error_px"]),
            rotation_error_deg=float(row["rotation_error_deg"]),
            scale_error=float(row["scale_error"]),
        )

    return graph


def graph_summary(
    graph: nx.Graph,
) -> dict:
    """
    Return basic topology statistics.
    """
    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    isolates = sorted(nx.isolates(graph))

    degrees: dict[str, int] = {
        str(node): int(degree)
        for node, degree in graph.degree
    }

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component": (
            len(components[0])
            if components
            else 0
        ),
        "isolates": isolates,
        "min_degree": (
            min(degrees.values())
            if degrees
            else 0
        ),
        "max_degree": (
            max(degrees.values())
            if degrees
            else 0
        ),
        "mean_degree": (
            sum(degrees.values()) / len(degrees)
            if degrees
            else 0.0
        ),
    }


def save_graph(
    graph: nx.Graph,
    path: Path,
) -> None:
    """
    Save the graph in GraphML format.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        path,
    )