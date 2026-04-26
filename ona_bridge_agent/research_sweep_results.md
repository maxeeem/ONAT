# Research Sweep Results

Source JSON: `research_sweep_results.json`

Split seeds: 0, 1, 2, 3, 4

| Method | Mean Acc | Std | Min | Max |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.487 | 0.008 | 0.478 | 0.500 |
| embedding-only bridge | 0.589 | 0.010 | 0.572 | 0.600 |
| neural baseline (MLP ensemble) | 0.584 | 0.011 | 0.572 | 0.600 |
| ONA direct propagation | 0.589 | 0.010 | 0.572 | 0.600 |
| ONA revision (with conflict context) | 0.922 | 0.010 | 0.906 | 0.933 |
| ONA multi-hop + revision | 1.000 | 0.000 | 1.000 | 1.000 |

## Per-Split Detailed Tables

### split_seed=0

# Research Evaluation Results

Source JSON: `seed_0`

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


### split_seed=1

# Research Evaluation Results

Source JSON: `seed_1`

Test examples: 180 (held-out noun pairs)

## Overall Accuracy with 95% Bootstrap CI

| Method | Accuracy | 95% CI |
|---|---:|---|
| exact lexical baseline | 0.500 | [0.428, 0.572] |
| embedding-only bridge | 0.594 | [0.522, 0.667] |
| neural baseline (MLP ensemble) | 0.572 | [0.494, 0.644] |
| ONA direct propagation | 0.594 | [0.522, 0.667] |
| ONA revision (with conflict context) | 0.928 | [0.883, 0.961] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

## Scenario Accuracy (Held-Out Nouns)

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.567 | 0.433 |
| embedding-only bridge | 0.000 | 1.000 | 0.567 | 1.000 |
| neural baseline (MLP ensemble) | 0.000 | 1.000 | 0.433 | 1.000 |
| ONA direct propagation | 0.000 | 1.000 | 0.567 | 1.000 |
| ONA revision (with conflict context) | 1.000 | 1.000 | 0.567 | 1.000 |
| ONA multi-hop + revision | 1.000 | 1.000 | 1.000 | 1.000 |

## McNemar Tests vs ONA Multi-Hop

| Compared Method | b | c | p-value |
|---|---:|---:|---:|
| exact lexical baseline | 90 | 0 | 0.000000 |
| embedding-only bridge | 73 | 0 | 0.000000 |
| neural baseline (MLP ensemble) | 77 | 0 | 0.000000 |
| ONA direct propagation | 73 | 0 | 0.000000 |
| ONA revision (with conflict context) | 13 | 0 | 0.000244 |


### split_seed=2

# Research Evaluation Results

Source JSON: `seed_2`

Test examples: 180 (held-out noun pairs)

## Overall Accuracy with 95% Bootstrap CI

| Method | Accuracy | 95% CI |
|---|---:|---|
| exact lexical baseline | 0.478 | [0.406, 0.561] |
| embedding-only bridge | 0.583 | [0.506, 0.656] |
| neural baseline (MLP ensemble) | 0.583 | [0.511, 0.650] |
| ONA direct propagation | 0.583 | [0.506, 0.656] |
| ONA revision (with conflict context) | 0.917 | [0.878, 0.956] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

## Scenario Accuracy (Held-Out Nouns)

| Method | conflicting_evidence | lexical_core | multihop_chain | synonym_generalization |
|---|---:|---:|---:|---:|
| exact lexical baseline | 0.000 | 1.000 | 0.500 | 0.367 |
| embedding-only bridge | 0.000 | 1.000 | 0.500 | 1.000 |
| neural baseline (MLP ensemble) | 0.000 | 1.000 | 0.500 | 1.000 |
| ONA direct propagation | 0.000 | 1.000 | 0.500 | 1.000 |
| ONA revision (with conflict context) | 1.000 | 1.000 | 0.500 | 1.000 |
| ONA multi-hop + revision | 1.000 | 1.000 | 1.000 | 1.000 |

## McNemar Tests vs ONA Multi-Hop

| Compared Method | b | c | p-value |
|---|---:|---:|---:|
| exact lexical baseline | 94 | 0 | 0.000000 |
| embedding-only bridge | 75 | 0 | 0.000000 |
| neural baseline (MLP ensemble) | 75 | 0 | 0.000000 |
| ONA direct propagation | 75 | 0 | 0.000000 |
| ONA revision (with conflict context) | 15 | 0 | 0.000061 |


### split_seed=3

# Research Evaluation Results

Source JSON: `seed_3`

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


### split_seed=4

# Research Evaluation Results

Source JSON: `seed_4`

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


