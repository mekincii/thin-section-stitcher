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
|---|---:|---:|---:|
| 1.0 / unlimited | 76,661 | 88.25 s | — |
| 0.75 / unlimited | 56,595 | 71.49 s | — |
| 0.5 / unlimited | 27,696 | 32.73 s | 2339.5 MiB |
| 0.5 / 8,000 | 8,000 | 21.40 s | 675.8 MiB |
| 0.5 / 4,000 | 4,000 | 19.44 s | 337.9 MiB |
| 0.5 / 2,000 | 2,000 | 21.78 s | 169.0 MiB |

Half-resolution therefore retains a large number of geological image features
while substantially reducing computation and descriptor storage.

The 4,000-feature cap is a provisional balance between feature richness and
matching cost. It must be validated during pairwise overlap detection.

### Full-resolution data

Original images remain unchanged and will be used for final alignment and
mosaic rendering. Downsampling is only used during overlap discovery.