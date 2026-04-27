# Research Split Sweep Summary

Evaluated across 5 random noun-heldout splits (`split_seed` 0..4).

| Method | Mean Acc | Std | Min | Max |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.487 | 0.008 | 0.478 | 0.500 |
| embedding-only bridge | 0.589 | 0.010 | 0.572 | 0.600 |
| neural baseline (MLP ensemble) | 0.584 | 0.011 | 0.572 | 0.600 |
| ONA direct propagation | 0.589 | 0.010 | 0.572 | 0.600 |
| ONA revision (with conflict context) | 0.922 | 0.010 | 0.906 | 0.933 |
| ONA multi-hop + revision | 1.000 | 0.000 | 1.000 | 1.000 |

