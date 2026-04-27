# External WSC Evaluation (Offline, No-Mock)

Source JSON: `external_wsc_results_strong.json`

## Full WSC273

Examples: 273

| Method | Accuracy | 95% CI |
|---|---:|---|
| sentence_transformer_replacement | 0.498 | [0.440, 0.557] |
| nearest_mention | 0.498 | [0.436, 0.557] |
| gpt2_sentence_score | 0.524 | [0.462, 0.579] |
| gpt2-medium_sentence_score | 0.549 | [0.487, 0.608] |
| gpt2-large_sentence_score | 0.590 | [0.531, 0.648] |
| bert-base-uncased_mlm_option_score | 0.553 | [0.495, 0.612] |
| roberta-large_mlm_option_score | 0.689 | [0.634, 0.744] |

McNemar vs `roberta-large_mlm_option_score`:

| Method | b | c | p-value |
|---|---:|---:|---:|
| sentence_transformer_replacement | 108 | 56 | 0.000060 |
| nearest_mention | 92 | 40 | 0.000007 |
| gpt2_sentence_score | 65 | 20 | 0.000001 |
| gpt2-medium_sentence_score | 65 | 27 | 0.000093 |
| gpt2-large_sentence_score | 53 | 26 | 0.003183 |
| bert-base-uncased_mlm_option_score | 59 | 22 | 0.000048 |

## Full WSC273 Learned Bridge + ONA (Stratified CV)

Examples: 273 with 5-fold CV (seed 13, 22 learned features)

| Method | Accuracy | 95% CI |
|---|---:|---|
| learned_bridge_linear_kfold | 0.685 | [0.634, 0.736] |
| learned_bridge_gated_kfold | 0.700 | [0.645, 0.751] |
| learned_ona_direct_kfold | 0.700 | [0.645, 0.751] |
| learned_ona_multihop_kfold | 0.700 | [0.645, 0.751] |
| learned_ona_revision_kfold | 0.667 | [0.612, 0.722] |

| Method | Brier | Log Loss | ECE (10-bin) |
|---|---:|---:|---:|
| learned_bridge_linear_kfold | 0.206 | 0.617 | 0.080 |
| learned_bridge_gated_kfold | 0.228 | 0.872 | 0.164 |
| learned_ona_direct_kfold | 0.243 | 1.608 | 0.187 |
| learned_ona_multihop_kfold | 0.228 | 1.209 | 0.166 |
| learned_ona_revision_kfold | 0.262 | 1.116 | 0.226 |

McNemar vs `learned_bridge_gated_kfold`:

| Method | b | c | p-value |
|---|---:|---:|---:|
| learned_bridge_linear_kfold | 38 | 34 | 0.723948 |
| learned_ona_direct_kfold | 0 | 0 | 1.000000 |
| learned_ona_multihop_kfold | 0 | 0 | 1.000000 |
| learned_ona_revision_kfold | 21 | 12 | 0.162756 |

## Cross-Section Comparison vs Best Full-WSC Neural Baseline

Full-WSC anchor method: `roberta-large_mlm_option_score`

| CV Method | CV Acc | Anchor Acc | Delta | McNemar p-value |
|---|---:|---:|---:|---:|
| learned_bridge_linear_kfold | 0.685 | 0.689 | -0.004 | 1.000000 |
| learned_bridge_gated_kfold | 0.700 | 0.689 | +0.011 | 0.581055 |
| learned_ona_direct_kfold | 0.700 | 0.689 | +0.011 | 0.581055 |
| learned_ona_multihop_kfold | 0.700 | 0.689 | +0.011 | 0.581055 |
| learned_ona_revision_kfold | 0.667 | 0.689 | -0.022 | 0.417692 |

## WSC Causal `because ... was ...` Paired Subset

Examples: 26 across 13 minimal-pair groups (leave-one-group-out training for centroid/ONA mapping; includes paired groups with opposite labels)

| Method | Accuracy | 95% CI |
|---|---:|---|
| descriptor_centroid_lopo | 0.538 | [0.346, 0.731] |
| ona_direct_lopo | 0.538 | [0.346, 0.731] |
| ona_multihop_lopo | 0.538 | [0.346, 0.731] |
| learned_bridge_lopo | 0.615 | [0.423, 0.808] |
| learned_ona_direct_lopo | 0.654 | [0.462, 0.808] |
| learned_ona_multihop_lopo | 0.654 | [0.462, 0.808] |

McNemar vs `learned_ona_direct_lopo`:

| Method | b | c | p-value |
|---|---:|---:|---:|
| descriptor_centroid_lopo | 8 | 5 | 0.581055 |
| ona_direct_lopo | 7 | 4 | 0.548828 |
| ona_multihop_lopo | 7 | 4 | 0.548828 |
| learned_bridge_lopo | 1 | 0 | 1.000000 |
| learned_ona_multihop_lopo | 0 | 0 | 1.000000 |

