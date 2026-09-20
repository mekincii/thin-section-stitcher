# Technical Decisions

## TD-001 — Working image scale and SIFT feature cap

**Status:** Provisional

For initial overlap discovery, microscope images are processed at a scale of
0.5, reducing the native 2560 × 1920 images to 1280 × 960.

SIFT feature extraction is limited to approximately 4,000 strongest features
per image.

### Evidence

On the 173-image research dataset:

| Scale / Features | Mean keypoints | Extraction time | Descriptor memory |
|------------------|---------------:|----------------:|------------------:|
| 1.0 / unlimited  |         76,661 |         88.25 s |                 — |
| 0.75 / unlimited |         56,595 |         71.49 s |                 — |
| 0.5 / unlimited  |         27,696 |         32.73 s |        2339.5 MiB |
| 0.5 / 8,000      |          8,000 |         21.40 s |         675.8 MiB |
| 0.5 / 4,000      |          4,000 |         19.44 s |         337.9 MiB |
| 0.5 / 2,000      |          2,000 |         21.78 s |         169.0 MiB |

Half-resolution therefore retains a large number of geological image features
while substantially reducing computation and descriptor storage.

The 4,000-feature cap is a provisional balance between feature richness and
matching cost. It must be validated during pairwise overlap detection.

### Full-resolution data

Original images remain unchanged and will be used for final alignment and
mosaic rendering. Downsampling is only used during overlap discovery.

## TD-002 — Coarse-to-fine overlap matching

**Status:** Provisional

Overlap reconstruction uses multiple quality tiers rather than applying the
most expensive configuration to every possible image pair.

### Discovery tier

- Scale: 0.5
- SIFT features: 4,000
- Purpose: screen all possible image pairs for plausible overlap

### Verification tier

- Scale: 0.75
- SIFT features: 8,000
- Purpose: re-evaluate candidate overlaps with richer image information

### Final alignment

Accepted overlaps will later be refined using the original full-resolution
images.

This approach prioritizes reconstruction reliability while avoiding expensive
high-resolution descriptor matching between clearly unrelated image pairs.

## TD-003 — Overlap confidence

**Status:** Provisional

Cross-scale verification showed highly stable geometric estimates across
candidate overlaps. Median translation disagreement between 0.5-scale and
0.75-scale estimates was approximately 0.52 pixels when converted to the
original image coordinate system.

Cross-scale consistency is therefore used as a supporting confidence signal,
but not as the sole overlap acceptance criterion.

Initial overlap confidence is primarily based on RANSAC evidence:

- High confidence: at least 50 inliers and an inlier ratio of at least 0.70
- Medium confidence: at least 20 inliers and an inlier ratio of at least 0.50
- Weak confidence: below those levels

These thresholds remain provisional and will be evaluated using the topology
of the overlap graph.

## TD-004 — Rotation is not an overlap rejection criterion

**Status:** Accepted

Initial discovery used an absolute rotation limit of 20 degrees. Graph
analysis revealed that this incorrectly excluded genuine overlaps.

High-quality examples include verified overlaps with rotations of approximately
-41, -49, and -63 degrees. These pairs retain hundreds to thousands of RANSAC
inliers, high inlier ratios, and estimated image scales close to 1.0.

Rotation therefore describes acquisition geometry rather than match quality
and is retained as transform metadata but is not used as a discovery rejection
criterion.

Overlap screening continues to use descriptor evidence, RANSAC support,
inlier ratio, and physically plausible image scale.

## TD-005 — High-confidence overlap graph is globally connected

**Status:** Accepted

After removing rotation as an overlap rejection criterion, the verified
high-confidence overlap graph contains all 173 microscope images in a single
connected component.

The high-confidence graph contains 572 usable edges, with every image having
at least two high-confidence neighbours.

Medium-confidence overlaps are therefore not required to establish global
connectivity. Global layout initialization will use high-confidence overlaps
as the primary geometric backbone, while medium-confidence edges may later be
used for validation or additional constraints.