# OpenNARS Bridge Agent (Prototype)

Proof-of-concept neuro-symbolic bridge:

neural-ish semantic mapping -> uncertain Narsese truth values -> ONA inference/revision

This repository currently validates the bridge and inference loop on a constructed fit/pronoun-style disambiguation benchmark. It does **not** yet demonstrate end-to-end improvement for a trained neural model.

## Current Measured Results

Repro command:

```bash
cd ona_bridge_agent
python3 -m ona_bridge_agent.benchmark --ona-cmd ../OpenNARS-for-Applications/NAR --output-json benchmark_results.json --output-md benchmark_results.md
```

Current benchmark size: 48 examples across 4 scenario groups.

| Ablation | Overall Accuracy |
|---|---:|
| 1. exact lexical baseline | 0.500 |
| 2. embedding-only bridge | 0.583 |
| 3. ONA direct rule propagation | 0.583 |
| 4. ONA conflicting evidence / revision | 0.917 |
| 5. ONA multi-hop causal chain | 1.000 |
| 6. neural baseline (MLP) | 0.542 |

Per-scenario results and failure cases are in `ona_bridge_agent/benchmark_results.json` and `ona_bridge_agent/benchmark_results.md`.

## Research-Grade Evaluation (Held-Out + Stats)

Repro command:

```bash
cd ona_bridge_agent
python3 -m ona_bridge_agent.research_eval \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --n-pairs 60 \
  --train-frac 0.5 \
  --split-seed 0 \
  --mlp-seeds 12 \
  --bootstrap-samples 2000 \
  --output-json research_results.json \
  --output-md research_results.md
```

This run uses 360 generated examples total and evaluates on 180 held-out noun-pair test examples.

| Method | Test Accuracy | 95% Bootstrap CI |
|---|---:|---|
| exact lexical baseline | 0.483 | [0.411, 0.556] |
| embedding-only bridge | 0.600 | [0.528, 0.667] |
| neural baseline (MLP ensemble) | 0.600 | [0.528, 0.667] |
| ONA direct propagation | 0.600 | [0.528, 0.667] |
| ONA revision (with conflict context) | 0.933 | [0.894, 0.967] |
| ONA multi-hop + revision | 1.000 | [1.000, 1.000] |

McNemar exact tests against `ONA multi-hop + revision` in this run are significant for every baseline and ablation (`p <= 0.000488`).

Multi-split robustness (5 noun-heldout splits, seeds 0..4):
- exact lexical baseline: mean `0.487` (std `0.008`)
- embedding-only bridge: mean `0.589` (std `0.010`)
- neural baseline (MLP ensemble): mean `0.584` (std `0.011`)
- ONA direct propagation: mean `0.589` (std `0.010`)
- ONA revision (with conflict context): mean `0.922` (std `0.010`)
- ONA multi-hop + revision: mean `1.000` (std `0.000`)

Artifacts:
- `ona_bridge_agent/research_results.json`
- `ona_bridge_agent/research_results.md`
- `ona_bridge_agent/research_sweep_results.json`
- `ona_bridge_agent/research_sweep_results.md`

## External Benchmark (WSC273, Offline)

Repro command:

```bash
cd ona_bridge_agent
python3 -m ona_bridge_agent.external_wsc_eval \
  --lm-models gpt2,gpt2-medium \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --output-json external_wsc_results.json \
  --output-md external_wsc_results.md
```

This uses the locally cached WSC273 Arrow file and local HuggingFace model snapshots only (no mocked tie-breakers, no network dependence at run time).

Full WSC273 (273 examples):
- `sentence_transformer_replacement`: `0.498` (95% CI `[0.440, 0.557]`)
- `gpt2_sentence_score`: `0.524` (95% CI `[0.462, 0.579]`)
- `gpt2-medium_sentence_score`: `0.549` (95% CI `[0.487, 0.608]`)

ONA-executable causal paired subset (24 examples, 12 minimal-pair groups, leave-one-group-out):
- `descriptor_centroid_lopo`: `0.542`
- `ona_direct_lopo`: `0.542`
- `ona_multihop_lopo`: `0.542`
- `learned_bridge_lopo`: `0.583`
- `learned_ona_direct_lopo`: `0.625`
- `learned_ona_multihop_lopo`: `0.625`

Current takeaway: external integration is real and reproducible, and learned bridge behavior improves the executable subset, but the system still does not beat stronger neural baselines on full WSC273.

## Caveats

- The benchmark is synthetic and intentionally controlled.
- The bridge still performs most semantic grounding work; ONA performs propagation, revision, and multi-hop inference on top.
- The neural baseline is an offline MLP bag-of-words model trained on non-conflict examples; it is included as a comparable non-symbolic baseline, not as a state-of-the-art LM baseline.
