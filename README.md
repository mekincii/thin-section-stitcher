# Thin Section Stitcher

A Python computer-vision pipeline for reconstructing petrographic thin-section
mosaics from overlapping microscope images when acquisition coordinates are
unknown.

The project recovers image adjacency and relative geometry directly from image
content, builds a globally consistent image layout, standardizes illumination,
and renders a full-resolution stitched mosaic.

## Motivation

Microscope image collections are not always acquired with usable stage
coordinates. Images may:

- have irregular and unknown overlap,
- be captured in changing scan directions,
- contain large image rotations,
- have variable illumination,
- contain repetitive geological textures,
- and arrive only as an unordered image collection.

The goal of this project is to reconstruct the thin section without assuming
that filename order corresponds to physical position.

The implementation intentionally uses inspectable classical computer-vision
methods so that overlap discovery, geometric verification, global
optimization, photometric correction, and rendering can all be diagnosed
independently.

## Pipeline

The implemented workflow is:

1. Dataset inspection
2. SIFT feature extraction
3. Coarse all-pairs overlap discovery
4. Higher-resolution geometric verification
5. Overlap graph construction
6. Global rigid image-layout initialization
7. Global pose optimization
8. Loop-closure and local alignment diagnostics
9. Raw mosaic rendering
10. Independent hand-mosaic comparison
11. Overlap-derived photometric standardization
12. Feather-weighted blending
13. Tiled full-resolution TIFF rendering
14. Native-resolution output QA

## Geometric reconstruction

Pairwise image overlap is detected using SIFT features and bidirectional
descriptor matching.

Candidate matches are geometrically verified with a partial affine model
containing:

- translation,
- rotation,
- uniform scale.

Rotation is retained as geometric information rather than used as a rejection
criterion. This is important because valid image pairs in the tested dataset
include relative rotations exceeding 60 degrees.

Verified image relationships are represented as an overlap graph.

The final layout uses a rigid pose model:

- global x position,
- global y position,
- image rotation.

Per-image scale is not optimized because the microscope optical configuration
is assumed to remain constant during acquisition.

## Photometric standardization

Microscope images may record the same specimen region with different
illumination and color response.

Photometric standardization is estimated directly from geometrically trusted
image overlaps.

The final model contains:

- a shared low-frequency quadratic illumination field,
- separate B, G, and R field coefficients,
- regularized per-image multiplicative color/exposure gains,
- conservative clipping of spatial illumination correction.

The shared illumination field substantially reduced overlap photometric
residuals in the validation dataset.

The final spatial correction is capped at 1.5x to prevent aggressive
extrapolation near image boundaries.

## Blending

Overlapping images are combined using feather weighting.

The final configuration uses:

- feather fraction: `0.15`
- minimum source-frame weight: `0.05`

Pixels near source-image borders receive lower blending confidence, while
central pixels receive full weight.

This changes only the relative weighting inside overlaps. Regions covered by a
single source image retain their calibrated source intensity.

## Full-resolution rendering

The final mosaic is rendered tile-by-tile rather than using one
full-resolution floating-point accumulator.

This keeps peak memory usage manageable even for very large mosaics.

For the development dataset:

- source images: 173
- source image size: 2560 x 1920 px
- final mosaic size: 24,695 x 22,204 px
- final mosaic size: approximately 548 megapixels
- rendered canvas coverage: approximately 56.6%
- final TIFF size: approximately 1.53 GiB

The primary image deliverable is an RGB uint8 TIFF.

## Validation

Several independent checks were used during development.

### Overlap graph

The final high-confidence overlap graph contains:

- 173 images
- 572 high-confidence edges
- one connected component
- no isolated images

### Global layout consistency

After rigid pose optimization, high-confidence geometric constraints showed
low residual disagreement across the graph.

Representative full-resolution overlaps were also inspected directly,
including:

- strong ordinary overlaps,
- large-rotation overlaps,
- the largest loop-closure residual cases.

No global full-resolution rematching step was required.

### Independent hand mosaic

An independently hand-stitched reconstruction was registered to the automatic
mosaic as an external comparison.

Similarity registration produced:

- 1,731 RANSAC inliers
- median residual: approximately 5.47 automatic-mosaic pixels

A full affine diagnostic produced:

- 1,784 RANSAC inliers
- median residual: approximately 4.00 pixels
- scale anisotropy: approximately 1.0024
- shear departure from orthogonality: approximately 0.213 degrees

The small affine deformation indicates that the two reconstructions agree
strongly at specimen scale without requiring substantial global stretching or
shearing.

The hand reconstruction is treated as an independent reference, not as ground
truth.

### Native full-resolution QA

Six 1536 x 1536 crops were extracted directly from the final TIFF, bypassing
viewer-side resampling and contrast adjustment.

No obvious:

- duplicated geological features,
- stitching discontinuities,
- tile-boundary artifacts,
- major exposure jumps,
- or large photometric seams

were observed.

## Installation

Python 3.11 or newer is required.

Create and activate a virtual environment, then install the project in editable
mode:

```bash
pip install -e ".[dev]"
```

Main dependencies include:

- NumPy
- OpenCV
- pandas
- SciPy
- NetworkX
- Matplotlib
- tifffile
- tqdm

Development tools:

- pytest
- Ruff

## Final rendering

After geometric reconstruction and photometric calibration have been
generated, the final full-resolution mosaic can be rendered with:

```powershell
python scripts\render_fullres_mosaic.py `
  "C:\path\to\microscope-images"
```

Default inputs are read from the local `outputs/` directory, including:

- `optimized_global_layout.csv`
- `final_photometric_gains.csv`
- `photometric_field.json`

The default final output is:

```text
outputs/thin_section_fullres.tif
```

Generated outputs are intentionally excluded from version control.

## Preview rendering

A lower-resolution diagnostic or standardized preview can be produced with
`scripts/render_mosaic.py`.

The rendering system supports:

- ordinary averaging,
- photometric calibration,
- feather blending,
- TIFF output,
- coverage visualization.

Raw diagnostic mosaics should be retained alongside standardized products so
that image-processing provenance remains inspectable.

## Testing

Run:

```bash
ruff check .
pytest
```

The unit tests cover core rendering behavior including:

- rigid pose reconstruction,
- scaled coordinate transforms,
- feather-weight behavior,
- identity photometric calibration,
- BGR photometric gain ordering.

## Repository structure

```text
docs/
    decisions.md

scripts/
    inspect_dataset.py
    benchmark_features.py
    probe_pair_matching.py
    match_all_pairs.py
    analyze_match_consistency.py
    analyze_overlap_graph.py
    initialize_global_layout.py
    optimize_global_layout.py
    analyze_layout_residuals.py
    render_mosaic.py
    inspect_overlap_alignment.py
    compare_with_hand_mosaic.py
    estimate_photometric_corrections.py
    analyze_photometric_field.py
    fit_photometric_calibration.py
    render_fullres_mosaic.py
    extract_fullres_qa_crops.py

src/thin_section_stitcher/
    dataset.py
    features.py
    matching.py
    overlap_graph.py
    layout.py
    mosaic.py
    comparison.py
    photometric.py

tests/
    test_mosaic.py
    test_photometric.py
```

## Design decisions

Important technical decisions and their rationale are recorded in
`docs/decisions.md`.

These include decisions concerning:

- coarse-to-fine matching,
- geometric confidence thresholds,
- large image rotations,
- rigid global pose optimization,
- full-resolution rematching,
- independent hand-mosaic validation,
- photometric standardization,
- final rendering.

## Data and privacy

Research microscope imagery is not distributed with this repository.

The repository contains code, documentation, and tests only. Private research
datasets, generated mosaics, pairwise measurement files, calibration outputs,
and other research artifacts remain local.

A deliberately non-sensitive demonstration dataset may be added separately in
the future.

## Status

The `v0.1.0` release represents a complete research prototype capable of
reconstructing, validating, photometrically standardizing, and rendering a
large coordinate-free petrographic thin-section mosaic.
