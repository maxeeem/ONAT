# Research Evaluation Results

Source JSON: `research_results_seed3.json`

Test examples: 180 (held-out noun pairs)

## Overall Accuracy with 95% Bootstrap CI

| Method | Accuracy | 95% CI |
|---|---:|---|
| exact lexical baseline | 0.489 | [0.417, 0.561] |
| embedding-only bridge | 0.594 | [0.522, 0.667] |
| neural baseline (MLP ensemble) | 0.572 | [0.494, 0.639] |
| ONA direct propagation | 0.594 | [0.522, 0.667] |
| ONA revision (with conflict context) | 0.928 | [0.894, 0.961] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

## Scenario Accuracy (Held-Out Nouns)

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.567 | 0.367 |
| embedding-only bridge | 0.000 | 1.000 | 0.567 | 1.000 |
| neural baseline (MLP ensemble) | 0.000 | 1.000 | 0.433 | 1.000 |
| ONA direct propagation | 0.000 | 1.000 | 0.567 | 1.000 |
| ONA revision (with conflict context) | 1.000 | 1.000 | 0.567 | 1.000 |
| ONA multi-hop + revision | 1.000 | 1.000 | 1.000 | 1.000 |

## McNemar Tests vs ONA Multi-Hop

| Compared Method | b | c | p-value |
|---|---:|---:|---:|
| exact lexical baseline | 92 | 0 | 0.000000 |
| embedding-only bridge | 73 | 0 | 0.000000 |
| neural baseline (MLP ensemble) | 77 | 0 | 0.000000 |
| ONA direct propagation | 73 | 0 | 0.000000 |
| ONA revision (with conflict context) | 13 | 0 | 0.000244 |

