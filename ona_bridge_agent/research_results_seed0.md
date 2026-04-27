# Research Evaluation Results

Source JSON: `research_results_seed0.json`

Test examples: 180 (held-out noun pairs)

## Overall Accuracy with 95% Bootstrap CI

| Method | Accuracy | 95% CI |
|---|---:|---|
| exact lexical baseline | 0.483 | [0.411, 0.561] |
| embedding-only bridge | 0.600 | [0.528, 0.672] |
| neural baseline (MLP ensemble) | 0.600 | [0.528, 0.672] |
| ONA direct propagation | 0.600 | [0.528, 0.672] |
| ONA revision (with conflict context) | 0.933 | [0.894, 0.967] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

## Scenario Accuracy (Held-Out Nouns)

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.600 | 0.300 |
| embedding-only bridge | 0.000 | 1.000 | 0.600 | 1.000 |
| neural baseline (MLP ensemble) | 0.000 | 1.000 | 0.600 | 1.000 |
| ONA direct propagation | 0.000 | 1.000 | 0.600 | 1.000 |
| ONA revision (with conflict context) | 1.000 | 1.000 | 0.600 | 1.000 |
| ONA multi-hop + revision | 1.000 | 1.000 | 1.000 | 1.000 |

## McNemar Tests vs ONA Multi-Hop

| Compared Method | b | c | p-value |
|---|---:|---:|---:|
| exact lexical baseline | 93 | 0 | 0.000000 |
| embedding-only bridge | 72 | 0 | 0.000000 |
| neural baseline (MLP ensemble) | 72 | 0 | 0.000000 |
| ONA direct propagation | 72 | 0 | 0.000000 |
| ONA revision (with conflict context) | 12 | 0 | 0.000488 |

