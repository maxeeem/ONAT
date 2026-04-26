from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .bridge import CalibratedConceptMapper, FitReasoningBridge
from .dataset import build_research_examples
from .ona import ONAFileRunner, predict_from_ona_output
from .types import Example

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


@dataclass
class MLPModel:
    vocab: dict[str, int]
    model: object
    torch: object

    def _encode(self, sentence: str) -> list[float]:
        vec = [0.0] * len(self.vocab)
        for tok in _TOKEN_RE.findall(sentence.lower()):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def predict(self, sentence: str) -> str:
        t = self.torch
        x = t.tensor([self._encode(sentence)], dtype=t.float32)
        with t.no_grad():
            logits = self.model(x)
            pred = int(t.argmax(logits, dim=1).item())
        return "subject" if pred == 0 else "object"


def split_by_pair(examples: list[Example], train_frac: float, split_seed: int = 0) -> tuple[list[Example], list[Example]]:
    if not (0.1 <= train_frac <= 0.9):
        raise ValueError("train_frac must be between 0.1 and 0.9")

    pair_ids = sorted({int(ex.subject.rsplit("_", 1)[1]) for ex in examples})
    rng = random.Random(split_seed)
    rng.shuffle(pair_ids)
    cutoff = max(1, int(len(pair_ids) * train_frac))
    train_ids = set(pair_ids[:cutoff])
    train_rows = [ex for ex in examples if int(ex.subject.rsplit("_", 1)[1]) in train_ids]
    test_rows = [ex for ex in examples if int(ex.subject.rsplit("_", 1)[1]) not in train_ids]
    return train_rows, test_rows


def exact_lexical_baseline(ex: Example) -> str:
    if ex.adjective == "large":
        return "subject"
    if ex.adjective == "small":
        return "object"
    return "subject"


def embedding_only_bridge(ex: Example, mapper: CalibratedConceptMapper) -> str:
    memberships = mapper.memberships(ex.adjective)
    subject_score = memberships.get("large_like", 0.0)
    object_score = memberships.get("small_like", 0.0)
    return "subject" if subject_score >= object_score else "object"


def train_mlp(train_rows: list[Example], seed: int, hidden_dim: int = 64, epochs: int = 250) -> MLPModel:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for the MLP baseline") from exc

    random.seed(seed)
    torch.manual_seed(seed)

    vocab: dict[str, int] = {}
    for ex in train_rows:
        for tok in _TOKEN_RE.findall(ex.sentence.lower()):
            if tok not in vocab:
                vocab[tok] = len(vocab)

    def encode(sentence: str) -> list[float]:
        vec = [0.0] * len(vocab)
        for tok in _TOKEN_RE.findall(sentence.lower()):
            idx = vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    model = torch.nn.Sequential(
        torch.nn.Linear(len(vocab), hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 2),
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_train = torch.tensor([encode(ex.sentence) for ex in train_rows], dtype=torch.float32)
    y_train = torch.tensor([0 if ex.expected == "subject" else 1 for ex in train_rows], dtype=torch.long)

    for _ in range(epochs):
        optim.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optim.step()

    return MLPModel(vocab=vocab, model=model, torch=torch)


def _with_context(lines: list[str], context_rules: list[str], enabled: bool) -> list[str]:
    if not enabled or not context_rules:
        return lines
    if len(lines) < 3:
        return lines + context_rules
    return lines[:-3] + context_rules + lines[-3:]


def run_ona(
    runner: ONAFileRunner,
    bridge: FitReasoningBridge,
    rows: list[Example],
    use_context: bool,
    cycles: int,
    timeout_sec: int,
) -> list[str]:
    preds: list[str] = []
    for ex in rows:
        frame = bridge.extract(ex.sentence, known_adjective=ex.adjective)
        lines = bridge.to_narsese(frame, cycles=cycles)
        lines = _with_context(lines, ex.context_rules, enabled=use_context)
        output, _ = runner.run(lines, timeout_sec=timeout_sec, keep_file=False)
        pred, _scores, _explanations = predict_from_ona_output(output, ex.adjective)
        preds.append(pred or "subject")
    return preds


def accuracy(preds: list[str], rows: list[Example]) -> float:
    if not rows:
        return 0.0
    return sum(1 for p, ex in zip(preds, rows) if p == ex.expected) / len(rows)


def scenario_accuracy(preds: list[str], rows: list[Example]) -> dict[str, float]:
    pred_groups: dict[str, list[str]] = defaultdict(list)
    row_groups: dict[str, list[Example]] = defaultdict(list)
    for pred, ex in zip(preds, rows):
        pred_groups[ex.scenario].append(pred)
        row_groups[ex.scenario].append(ex)
    return {s: accuracy(pred_groups[s], row_groups[s]) for s in sorted(row_groups)}


def bootstrap_ci(preds: list[str], rows: list[Example], samples: int = 1000, seed: int = 123) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return 0.0, 0.0
    values = []
    for _ in range(samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        correct = sum(1 for i in idxs if preds[i] == rows[i].expected)
        values.append(correct / n)
    values.sort()
    lo = values[int(0.025 * (samples - 1))]
    hi = values[int(0.975 * (samples - 1))]
    return lo, hi


def mcnemar_exact(a_preds: list[str], b_preds: list[str], rows: list[Example]) -> dict[str, float]:
    b = 0  # a wrong, b right
    c = 0  # a right, b wrong
    for pa, pb, ex in zip(a_preds, b_preds, rows):
        a_ok = pa == ex.expected
        b_ok = pb == ex.expected
        if (not a_ok) and b_ok:
            b += 1
        elif a_ok and (not b_ok):
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2.0 * tail)
    return {"b": float(b), "c": float(c), "p_value": p}


def collect_failures(preds: list[str], rows: list[Example], limit: int = 10) -> list[dict[str, str]]:
    bad = []
    for pred, ex in zip(preds, rows):
        if pred != ex.expected:
            bad.append(
                {
                    "scenario": ex.scenario,
                    "sentence": ex.sentence,
                    "adjective": ex.adjective,
                    "expected": ex.expected,
                    "predicted": pred,
                }
            )
        if len(bad) >= limit:
            break
    return bad


def majority_vote(predictions_by_seed: list[list[str]]) -> list[str]:
    if not predictions_by_seed:
        return []
    n = len(predictions_by_seed[0])
    voted = []
    for i in range(n):
        subj_votes = sum(1 for run in predictions_by_seed if run[i] == "subject")
        obj_votes = len(predictions_by_seed) - subj_votes
        voted.append("subject" if subj_votes >= obj_votes else "object")
    return voted


def build_markdown(summary: dict, out_json: str) -> str:
    lines = []
    lines.append("# Research Evaluation Results")
    lines.append("")
    lines.append(f"Source JSON: `{out_json}`")
    lines.append("")
    lines.append(f"Test examples: {summary['n_test']} (held-out noun pairs)")
    lines.append("")
    lines.append("## Overall Accuracy with 95% Bootstrap CI")
    lines.append("")
    lines.append("| Method | Accuracy | 95% CI |")
    lines.append("|---|---:|---|")
    for name, data in summary["methods"].items():
        lo, hi = data["bootstrap_ci_95"]
        lines.append(f"| {name} | {data['overall_accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    lines.append("## Scenario Accuracy (Held-Out Nouns)")
    lines.append("")
    scenarios = summary["scenarios"]
    lines.append("| Method | " + " | ".join(scenarios) + " |")
    lines.append("|---|" + "|".join(["---:" for _ in scenarios]) + "|")
    for name, data in summary["methods"].items():
        vals = [f"{data['scenario_accuracy'].get(s, 0.0):.3f}" for s in scenarios]
        lines.append("| " + name + " | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## McNemar Tests vs ONA Multi-Hop")
    lines.append("")
    lines.append("| Compared Method | b | c | p-value |")
    lines.append("|---|---:|---:|---:|")
    for name, stat in summary["mcnemar_vs_ona_multihop"].items():
        lines.append(f"| {name} | {int(stat['b'])} | {int(stat['c'])} | {stat['p_value']:.6f} |")
    lines.append("")
    return "\n".join(lines)


def run_eval(args: argparse.Namespace) -> dict:
    all_rows = build_research_examples(n_pairs=args.n_pairs)
    train_rows, test_rows = split_by_pair(all_rows, train_frac=args.train_frac, split_seed=args.split_seed)

    mapper = CalibratedConceptMapper()
    runner = ONAFileRunner(args.ona_cmd)
    bridge_direct = FitReasoningBridge(embedder=mapper, concept_threshold=args.concept_threshold, rule_mode="direct")
    bridge_multihop = FitReasoningBridge(embedder=mapper, concept_threshold=args.concept_threshold, rule_mode="multihop")

    method_preds: dict[str, list[str]] = {}
    method_preds["exact lexical baseline"] = [exact_lexical_baseline(ex) for ex in test_rows]
    method_preds["embedding-only bridge"] = [embedding_only_bridge(ex, mapper) for ex in test_rows]

    mlp_train = [ex for ex in train_rows if ex.scenario in {"lexical_core", "synonym_generalization"}]
    mlp_runs: list[list[str]] = []
    mlp_seed_accs: list[float] = []
    for seed in range(args.mlp_seeds):
        model = train_mlp(mlp_train, seed=seed)
        preds = [model.predict(ex.sentence) for ex in test_rows]
        mlp_runs.append(preds)
        mlp_seed_accs.append(accuracy(preds, test_rows))
    method_preds["neural baseline (MLP ensemble)"] = majority_vote(mlp_runs)

    method_preds["ONA direct propagation"] = run_ona(
        runner=runner,
        bridge=bridge_direct,
        rows=test_rows,
        use_context=False,
        cycles=args.cycles,
        timeout_sec=args.timeout_sec,
    )
    method_preds["ONA revision (with conflict context)"] = run_ona(
        runner=runner,
        bridge=bridge_direct,
        rows=test_rows,
        use_context=True,
        cycles=args.cycles,
        timeout_sec=args.timeout_sec,
    )
    method_preds["ONA multi-hop + revision"] = run_ona(
        runner=runner,
        bridge=bridge_multihop,
        rows=test_rows,
        use_context=True,
        cycles=args.cycles,
        timeout_sec=args.timeout_sec,
    )

    scenarios = sorted({ex.scenario for ex in test_rows})
    summary = {
        "n_total": len(all_rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_frac": args.train_frac,
        "split_seed": args.split_seed,
        "n_pairs": args.n_pairs,
        "scenarios": scenarios,
        "mlp_seed_accuracies": mlp_seed_accs,
        "methods": {},
    }

    for method_name, preds in method_preds.items():
        ci = bootstrap_ci(preds, test_rows, samples=args.bootstrap_samples, seed=42)
        summary["methods"][method_name] = {
            "overall_accuracy": accuracy(preds, test_rows),
            "bootstrap_ci_95": [ci[0], ci[1]],
            "scenario_accuracy": scenario_accuracy(preds, test_rows),
            "failure_cases": collect_failures(preds, test_rows),
        }

    anchor = "ONA multi-hop + revision"
    mcnemar = {}
    for name, preds in method_preds.items():
        if name == anchor:
            continue
        mcnemar[name] = mcnemar_exact(preds, method_preds[anchor], test_rows)
    summary["mcnemar_vs_ona_multihop"] = mcnemar

    detailed_rows = []
    for i, ex in enumerate(test_rows):
        row = {
            "idx": i,
            "sentence": ex.sentence,
            "scenario": ex.scenario,
            "adjective": ex.adjective,
            "expected": ex.expected,
        }
        for method_name, preds in method_preds.items():
            row[f"pred::{method_name}"] = preds[i]
        detailed_rows.append(row)

    return {
        "summary": summary,
        "config": {
            "ona_cmd": args.ona_cmd,
            "cycles": args.cycles,
            "timeout_sec": args.timeout_sec,
            "concept_threshold": args.concept_threshold,
            "mlp_seeds": args.mlp_seeds,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "test_rows": detailed_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-grade evaluation with held-out splits and significance tests.")
    parser.add_argument("--ona-cmd", required=True, help='ONA command without "shell", e.g. "./NAR" or "/path/to/NAR".')
    parser.add_argument("--n-pairs", type=int, default=60, help="Number of unique noun pairs to generate.")
    parser.add_argument("--train-frac", type=float, default=0.5, help="Fraction of noun pairs reserved for training baselines.")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed used to shuffle noun-pair train/test split.")
    parser.add_argument("--mlp-seeds", type=int, default=10, help="Number of random seeds for MLP baseline.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--concept-threshold", type=float, default=0.20)
    parser.add_argument("--output-json", default="research_results.json")
    parser.add_argument("--output-md", default="research_results.md")
    args = parser.parse_args()

    results = run_eval(args)
    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.write_text(build_markdown(results["summary"], str(out_json)) + "\n", encoding="utf-8")

    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
