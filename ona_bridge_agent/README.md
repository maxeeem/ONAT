# ONA Bridge Agent: minimal bridge experiment

This package tests the bridge idea:

```text
English toy sentence
→ controlled syntax frame
→ vector-ish concept bridge
→ ONA-compatible Narsese
→ real ONA shell
→ parsed antecedent prediction
```

The task is deliberately small:

```text
The trophy did not fit in the suitcase because it was large.
→ it = subject / trophy

The trophy did not fit in the suitcase because it was small.
→ it = object / suitcase
```

The point is to isolate the bridge and ONA wiring, not to claim real natural-language understanding.

## Requirements

- Python 3.10+
- ONA built locally from `opennars/OpenNARS-for-Applications`

No required Python packages outside the standard library.

Optional:

- `pytest` for tests
- local GloVe-style embeddings file for `--glove-path`

## ONA build reminder

From the ONA repo, the common interactive shell form is:

```bash
./NAR shell
```

The agent uses file-stream mode:

```bash
./NAR shell < generated.nal
```

## Run without ONA first

This checks extraction and concept bridge scoring only:

```bash
cd ona_bridge_agent
python3 -m ona_bridge_agent --include-heldout
```

## Run with real ONA

From this package directory:

```bash
python3 -m ona_bridge_agent --ona-cmd /path/to/OpenNARS-for-Applications/NAR --verbose --output-json results.json
```

If you are already inside the ONA repo and copied this package there:

```bash
python3 -m ona_bridge_agent --ona-cmd ./NAR --verbose
```

## Keep generated Narsese files

```bash
python3 -m ona_bridge_agent --ona-cmd ./NAR --keep-nal-files --verbose
```

## Optional GloVe bridge

If you have a local GloVe-format file:

```bash
python3 -m ona_bridge_agent --ona-cmd ./NAR --glove-path ~/embeddings/glove.6B.50d.txt
```

The default bridge uses character n-gram vectors so the package runs without downloads. That is intentionally modest. It is still a real vector-similarity bridge, but not a pretrained semantic model.

## What is real here

Real:

- ONA-compatible Narsese generation using basic inheritance judgments/questions.
- ONA subprocess execution through `./NAR shell < file.nal`.
- Test cases and scoring.
- Soft concept membership converted into NARS truth values.

Not yet real:

- Full natural-language parsing.
- Learned Transformer-to-Narsese extraction.
- Differentiable training through ONA.
- Vector injection back into a neural model.

## Files

```text
ona_bridge_agent/
  bridge.py       # sentence -> BridgeFrame -> Narsese
  ona.py          # ONA subprocess runner + output parser
  experiments.py  # CLI + baselines + scoring
  dataset.py      # test cases
  types.py        # dataclasses
examples/
  sample_generated.nal

```

## Known fragility

ONA output formatting differs across versions. If the parser fails, run with `--verbose` and inspect `ona_output` in `results.json`. The parser looks for these queried terms:

```text
<ADJECTIVE --> subject_cause_of_fit_failure>
<ADJECTIVE --> object_cause_of_fit_failure>
```

If your ONA build echoes inputs or formats derived answers differently, adjust `max_score_for_term()` in `ona.py`.
