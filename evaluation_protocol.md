# Evaluation Protocol (No-Mock, Reproducible)

Last updated: April 26, 2026

## Scope

This repository has two evaluation tracks:

1. controlled synthetic fit/disambiguation benchmark (`benchmark.py`, `research_eval.py`)
2. external offline WSC273 benchmark (`external_wsc_eval.py`)

All reported scores must come from executable code paths with no oracle fallback and no manual answer injection.

## Environment

- Python interpreter: `python` (Conda environment with `torch`, `sentence_transformers`, `transformers`, `datasets`)
- ONA binary: `../OpenNARS-for-Applications/NAR`
- WSC cache path default:
  `/Users/maxeeem/.cache/huggingface/datasets/winograd_wsc/wsc273/0.0.0/0651311f3b6dda14889d9a063030a02458395ee50ab9f41cca4cd5a89c0c3dce/winograd_wsc-test.arrow`

## Synthetic Benchmark Protocol

### Main six-ablation benchmark

```bash
cd ona_bridge_agent
python -m ona_bridge_agent.benchmark \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --output-json benchmark_results.json \
  --output-md benchmark_results.md
```

### Held-out research run

```bash
python -m ona_bridge_agent.research_eval \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --n-pairs 60 \
  --train-frac 0.5 \
  --split-seed 0 \
  --mlp-seeds 12 \
  --bootstrap-samples 2000 \
  --output-json research_results.json \
  --output-md research_results.md
```

### Multi-split sweep

```bash
python -m ona_bridge_agent.research_sweep \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --n-pairs 60 \
  --split-seeds 0,1,2,3,4 \
  --output-json research_sweep_results.json \
  --output-md research_sweep_results.md
```

## External WSC273 Protocol

### Canonical run

```bash
python -m ona_bridge_agent.external_wsc_eval \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --cycles 30 \
  --cv-folds 5 \
  --cv-seed 13 \
  --output-json external_wsc_results.json \
  --output-md external_wsc_results.md
```

Default external models:

- causal LM baselines: `gpt2`, `gpt2-medium`
- MLM baselines: `bert-base-uncased`, `roberta-large`
- SentenceTransformer: `all-MiniLM-L6-v2` local snapshot

### Optional stronger run

```bash
python -m ona_bridge_agent.external_wsc_eval \
  --causal-lm-models gpt2,gpt2-medium,gpt2-large \
  --mlm-models bert-base-uncased,roberta-large \
  --ona-cmd ../OpenNARS-for-Applications/NAR \
  --cycles 30 \
  --cv-folds 5 \
  --cv-seed 13 \
  --output-json external_wsc_results_strong.json \
  --output-md external_wsc_results_strong.md
```

## Statistics and Reporting Rules

- Accuracy reported for every method.
- 95% bootstrap CI on example-level correctness (`samples=2000`, seed fixed inside script).
- Paired McNemar exact tests:
  - full WSC section: each method vs full-section anchor
  - learned CV section: each method vs CV anchor
  - cross-section: each CV method vs best full-WSC neural anchor
- Raw per-example predictions must remain in JSON artifacts.

## Guardrails

- No manual label overrides.
- No hidden tie-breakers outside committed code.
- Any change to split logic, seed defaults, or model list must update this file and regenerate artifacts.
