# ONA Bridge Agent

This package focuses on a clear bridge loop:

```text
Controlled sentence -> uncertain concept memberships -> Narsese truth values -> ONA inference/revision
```

The claim is limited: this is a working prototype on a constructed benchmark for disambiguation-like fit reasoning.

## Requirements

- Python 3.10+
- ONA built locally from `OpenNARS-for-Applications` (`./NAR`)
- `torch` for the offline neural baseline in `benchmark.py`

## Main benchmark (recommended)

Run the explicit six-ablation benchmark with real ONA:

```bash
cd ona_bridge_agent
python3 -m ona_bridge_agent.benchmark \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --output-json benchmark_results.json \
  --output-md benchmark_results.md
```

Ablations:
1. exact lexical baseline
2. embedding-only bridge
3. ONA direct rule propagation
4. ONA with conflicting evidence / revision
5. ONA with multi-hop causal chain
6. neural baseline (offline MLP)

The benchmark currently has 48 examples across:
- `lexical_core`
- `synonym_generalization`
- `conflicting_evidence`
- `multihop_chain`

## Research evaluation (held-out + significance)

For a stricter protocol (noun-heldout split, multi-seed neural baseline, bootstrap CI, McNemar tests):

```bash
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

To run multiple split seeds and aggregate:

```bash
python3 -m ona_bridge_agent.research_sweep \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --n-pairs 60 \
  --split-seeds 0,1,2,3,4 \
  --output-json research_sweep_results.json \
  --output-md research_sweep_results.md
```

Current observed range across split seeds (0..4):
- ONA multi-hop + revision: `1.000` (stable)
- ONA revision only: `0.906` to `0.933`
- embedding/direct/neural baselines: `0.572` to `0.600`

## External WSC273 evaluation

Run:

```bash
python3 -m ona_bridge_agent.external_wsc_eval \
  --lm-models gpt2,gpt2-medium \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --output-json external_wsc_results.json \
  --output-md external_wsc_results.md
```

What it does:
- evaluates full cached WSC273 (273 examples) with neural baselines
- extracts a strict executable `because ... was ...` minimal-pair subset and runs ONA on it

Current full-WSC results:
- `sentence_transformer_replacement`: `0.498`
- `gpt2_sentence_score`: `0.524`
- `gpt2-medium_sentence_score`: `0.549`

Current ONA subset results (24 examples, LOPO):
- `descriptor_centroid_lopo`: `0.542`
- `ona_direct_lopo`: `0.542`
- `ona_multihop_lopo`: `0.542`
- `learned_bridge_lopo`: `0.583`
- `learned_ona_direct_lopo`: `0.625`
- `learned_ona_multihop_lopo`: `0.625`

Interpretation: the external benchmark path is now real, but the current symbolic bridge is not yet competitive with stronger neural baselines on full WSC.

## Legacy experiment runner

You can still run the older small suite:

```bash
python3 -m ona_bridge_agent --ona-cmd ../OpenNARS-for-Applications/NAR --include-heldout --output-json results.json
```

## What is real here

- ONA subprocess execution in file mode (`./NAR shell < file.nal`).
- Explicit mapping from soft memberships to NARS truth values.
- Executable ablations showing where direct propagation, revision, and multi-hop are each required.

## Current limitations

- Synthetic benchmark only; no broad Winograd claim.
- The bridge still performs most semantic grounding.
- No end-to-end training of a neural model through ONA.

## Files

```text
ona_bridge_agent/
  bridge.py       # bridge extraction + rule-mode generation
  benchmark.py    # six-ablation benchmark runner
  research_eval.py
  research_sweep.py
  external_wsc_eval.py
  experiments.py  # legacy small runner
  dataset.py      # toy + benchmark datasets
  ona.py          # ONA subprocess runner + parser
  types.py        # dataclasses
```
