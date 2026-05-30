# A4 CO3D Summary

This summary consolidates local A4 state-convergence results on two available CO3D sequences.

## Setup

- Sequence 1: `co3d_apple_110_13051_23361`
- Sequence 2: `co3d_bottle_618_100690_201667`
- Frames per sequence: 24
- Image size: 224
- Device: CPU
- Compared update types: `cut3r`, `ttt3r`

## Apple

- `cut3r`: mean delta norm `157.63`, last delta norm `155.86`, mean delta cosine `0.0518`
- `ttt3r`: mean delta norm `95.89`, last delta norm `84.15`, mean delta cosine `0.4283`

Interpretation:

- `ttt3r` reduces mean state update magnitude by about `39.2%`
- `ttt3r` reduces final state update magnitude by about `46.0%`
- cosine alignment is much higher under `ttt3r`, indicating more stable state evolution

## Bottle

- `cut3r`: mean delta norm `251.77`, last delta norm `247.54`, mean delta cosine `0.2186`
- `ttt3r`: mean delta norm `126.87`, last delta norm `117.92`, mean delta cosine `0.4917`

Interpretation:

- `ttt3r` reduces mean state update magnitude by about `49.6%`
- `ttt3r` reduces final state update magnitude by about `52.4%`
- cosine alignment again increases substantially, showing more consistent update directions

## Overall Takeaway

Across both local CO3D sequences, `ttt3r` shows the same qualitative behavior:

- lower state delta norm
- lower final-step update magnitude
- higher consecutive-delta cosine alignment

This supports the A4 narrative that gated recurrent updates produce a more stable and faster-converging state trajectory than the plain recurrent baseline.

## Files

- `analysis_results/a4_co3d_apple/summary.csv`
- `analysis_results/a4_co3d_apple/delta_norm_curve.png`
- `analysis_results/a4_co3d_apple/delta_cosine_curve.png`
- `analysis_results/a4_co3d_bottle/summary.csv`
- `analysis_results/a4_co3d_bottle/delta_norm_curve.png`
- `analysis_results/a4_co3d_bottle/delta_cosine_curve.png`
