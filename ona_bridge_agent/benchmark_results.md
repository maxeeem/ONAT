# ONA Bridge Benchmark Results

Source JSON: `benchmark_results.json`

## Overall Accuracy

| Method | Accuracy |
|---|---:|
| exact lexical baseline | 0.500 |
| embedding-only bridge | 0.583 |
| neural baseline (MLP) | 0.542 |
| ONA direct rule propagation | 0.583 |
| ONA conflicting evidence / revision | 0.917 |
| ONA multi-hop causal chain | 1.000 |

## Accuracy by Scenario

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.500 | 0.500 |
| embedding-only bridge | 0.000 | 1.000 | 0.500 | 1.000 |
| neural baseline (MLP) | 0.000 | 1.000 | 0.250 | 1.000 |
| ONA direct rule propagation | 0.000 | 1.000 | 0.500 | 1.000 |
| ONA conflicting evidence / revision | 1.000 | 1.000 | 0.500 | 1.000 |
| ONA multi-hop causal chain | 1.000 | 1.000 | 1.000 | 1.000 |

