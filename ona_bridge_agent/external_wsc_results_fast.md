# External WSC Evaluation (Offline, No-Mock)

Source JSON: `external_wsc_results_fast.json`

## Full WSC273

Examples: 273

| Method | Accuracy | 95% CI |
|---|---:|---|
| sentence_transformer_replacement | 0.498 | [0.440, 0.557] |
| nearest_mention | 0.498 | [0.436, 0.557] |
| gpt2_sentence_score | 0.524 | [0.462, 0.579] |
| gpt2-medium_sentence_score | 0.549 | [0.487, 0.608] |

McNemar vs `gpt2-medium_sentence_score`:

| Method | b | c | p-value |
|---|---:|---:|---:|
| sentence_transformer_replacement | 81 | 67 | 0.285217 |
| nearest_mention | 53 | 39 | 0.174984 |
| gpt2_sentence_score | 24 | 17 | 0.348889 |

## WSC Causal `because ... was ...` Paired Subset

Examples: 24 across 12 minimal-pair groups (leave-one-group-out training for centroid/ONA mapping)

| Method | Accuracy | 95% CI |
|---|---:|---|
| descriptor_centroid_lopo | 0.542 | [0.333, 0.750] |
| ona_direct_lopo | 0.542 | [0.333, 0.750] |
| ona_multihop_lopo | 0.542 | [0.333, 0.750] |

McNemar vs `ona_multihop_lopo`:

| Method | b | c | p-value |
|---|---:|---:|---:|
| descriptor_centroid_lopo | 0 | 0 | 1.000000 |
| ona_direct_lopo | 0 | 0 | 1.000000 |

