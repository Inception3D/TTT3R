# A2 Proxy Summary

This is a local proxy version of A2 using available CO3D windows on the current machine.

## Setup

- Sequences: `co3d_apple_110_13051_23361`, `co3d_bottle_618_100690_201667`
- Windows: 12 total windows, 12 frames per window
- Baseline: `cut3r`
- Method: `ttt3r`
- Device: CPU
- Image size: 224

## Main Result

- Mean convergence improvement: `+38.10%`
- Pearson correlation between baseline cosine variance and convergence improvement: `0.8896`
- Spearman correlation: `0.6154`

## Interpretation

- Windows with larger state-dynamics variability tend to benefit more from gated updates.
- This qualitatively matches the intended A2 story: higher temporal variability creates more room for adaptive control.

## Limitation

This is **not** the final formal A2 in the paper sense.

- It uses local CO3D windows rather than ScanNet/TUM relpose scenes
- It uses convergence-improvement proxy instead of relpose ATE improvement
- The current local model does not expose the final exported `momentum_inv_t1` path for direct formal reproduction

## Output Files

- `a2_proxy_points.csv`
- `summary.csv`
- `variance_vs_improvement.png`
