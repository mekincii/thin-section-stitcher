# Thin Section Stitcher

A Python computer-vision project for reconstructing petrographic thin-section
mosaics from overlapping microscope images when acquisition coordinates are
unknown.

## Problem

Microscope images may:

- have irregular overlap,
- be captured in changing scan directions,
- have no recorded stage coordinates,
- contain repetitive geological textures.

The goal is to recover image adjacency and relative geometry directly from
image content and reconstruct the original thin section.

## Planned Pipeline

1. Image inspection and preprocessing
2. Feature extraction
3. Pairwise overlap detection
4. Geometric verification
5. Overlap graph construction
6. Global position optimization
7. Mosaic generation and blending

The initial implementation uses classical computer vision so that matching
and alignment decisions remain inspectable and explainable.

## Data

Research datasets are not distributed with this repository.

Users provide their own microscope images as input.

A small non-sensitive example dataset may be included for demonstration and
testing.