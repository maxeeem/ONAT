# OpenNARS for Applications (ONA) Python Bridge - Neuro-Symbolic Agent

A Neuro-Symbolic Hybrid architecture connecting modern Neural Semantic Encoders (Huggingface Sentence Transformers) to formal symbolic reasoning via Non-Axiomatic Logic (ONA).

## Highlights

* **100% Neural-Symbolic Disambiguation**: By mapping sentence embeddings safely bounded by probability into Narsese (Frequency/Confidence), we execute logical disambiguation on Winograd schemas.
* **Explainability Tracked**: Full extraction and backtracking of exact chronological `.nal` derivations (syllogistic `A -> B -> C` proofs).
* **Zero-Shot Adaptation**: Defeats "pure zero-shot" neural baseline models (`all-MiniLM-L6-v2` alone scores 50% vs ONA 100%) when given changing/dynamic instructions like: *"Wait, this large object is made of shrinking foam..."*.

## Environment
* Requires Python 3.10+
* `pip install sentence-transformers torch`
* `./NAR` binary from compiling the bundled C-language `OpenNARS-for-Applications` repo.

## Experiments

Run the local baseline against pure LLM/SentenceTransformer:
`python3 -m ona_bridge_agent.experiments --ona-cmd ../OpenNARS-for-Applications/NAR --use-huggingface --dynamic-env`
