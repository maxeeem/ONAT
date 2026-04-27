# Research Evaluation Results

Source JSON: `research_results_seed4.json`

Test examples: 180 (held-out noun pairs)

## Overall Accuracy with 95% Bootstrap CI

| Method | Accuracy | 95% CI |
|---|---:|---|
| exact lexical baseline | 0.483 | [0.406, 0.556] |
| embedding-only bridge | 0.572 | [0.500, 0.644] |
| neural baseline (MLP ensemble) | 0.594 | [0.522, 0.667] |
| ONA direct propagation | 0.572 | [0.500, 0.644] |
| ONA revision (with conflict context) | 0.906 | [0.861, 0.944] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

## Scenario Accuracy (Held-Out Nouns)

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.433 | 0.467 |
| embedding-only bridge | 0.000 | 1.000 | 0.433 | 1.000 |
| neural baseline (MLP ensemble) | 0.000 | 1.000 | 0.567 | 1.000 |
| ONA direct propagation | 0.000 | 1.000 | 0.433 | 1.000 |
| ONA revision (with conflict context) | 1.000 | 1.000 | 0.433 | 1.000 |
| ONA multi-hop + revision | 1.000 | 1.000 | 1.000 | 1.000 |

## McNemar Tests vs ONA Multi-Hop

| Compared Method | b | c | p-value |
|---|---:|---:|---:|
| exact lexical baseline | 93 | 0 | 0.000000 |
| embedding-only bridge | 77 | 0 | 0.000000 |
| neural baseline (MLP ensemble) | 73 | 0 | 0.000000 |
| ONA direct propagation | 77 | 0 | 0.000000 |
| ONA revision (with conflict context) | 17 | 0 | 0.000015 |

