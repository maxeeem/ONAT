from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .bridge import CalibratedConceptMapper, FitReasoningBridge
from .dataset import build_benchmark_examples
from .ona import ONAFileRunner, predict_from_ona_output
from .types import Example

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


@dataclass
class NeuralMLPModel:
    vocab: dict[str, int]
    model: object
    torch: object

    def encode(self, sentence: str) -> list[float]:
        vec = [0.0] * len(self.vocab)
        for tok in _TOKEN_RE.findall(sentence.lower()):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def predict(self, sentence: str) -> str:
        t = self.torch
        x = t.tensor([self.encode(sentence)], dtype=t.float32)
        with t.no_grad():
            logits = self.model(x)
            pred = int(t.argmax(logits, dim=1).item())
        return "subject" if pred == 0 else "object"


def exact_lexical_baseline(ex: Example) -> str:
    if ex.adjective == "large":
        return "subject"
    if ex.adjective == "small":
        return "object"
    return "subject"  # deterministic fallback instead of abstaining


def embedding_only_bridge(ex: Example, mapper: CalibratedConceptMapper) -> str:
    memberships = mapper.memberships(ex.adjective)
    subject_score = memberships.get("large_like", 0.0)
    object_score = memberships.get("small_like", 0.0)
    return "subject" if subject_score >= object_score else "object"


def _insert_context_rules(lines: list[str], context_rules: list[str], include_context: bool) -> list[str]:
    if not include_context or not context_rules:
        return lines
    if len(lines) < 3:
        return lines + context_rules
    return lines[:-3] + context_rules + lines[-3:]


def run_ona_method(
    runner: ONAFileRunner,
    bridge: FitReasoningBridge,
    examples: list[Example],
    include_context: bool,
    cycles: int,
    timeout_sec: int,
) -> list[str]:
    preds: list[str] = []
    for ex in examples:
        frame = bridge.extract(ex.sentence, known_adjective=ex.adjective)
        lines = bridge.to_narsese(frame, cycles=cycles)
        lines = _insert_context_rules(lines, ex.context_rules, include_context=include_context)
        output, _ = runner.run(lines, timeout_sec=timeout_sec, keep_file=False)
        pred, _scores, _explanations = predict_from_ona_output(output, ex.adjective)
        preds.append(pred or "subject")
    return preds


def train_neural_mlp_baseline(examples: list[Example]) -> NeuralMLPModel:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for the neural baseline.") from exc

    random.seed(7)
    torch.manual_seed(7)

    # Train only on non-conflict scenarios so the baseline cannot use hidden context rules.
    train_rows = [ex for ex in examples if ex.scenario in {"lexical_core", "synonym_generalization"}]

    vocab: dict[str, int] = {}
    for ex in train_rows:
        for tok in _TOKEN_RE.findall(ex.sentence.lower()):
            if tok not in vocab:
                vocab[tok] = len(vocab)

    model = torch.nn.Sequential(
        torch.nn.Linear(len(vocab), 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.CrossEntropyLoss()

    def encode(sentence: str) -> list[float]:
        vec = [0.0] * len(vocab)
        for tok in _TOKEN_RE.findall(sentence.lower()):
            idx = vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    x_train = torch.tensor([encode(ex.sentence) for ex in train_rows], dtype=torch.float32)
    y_train = torch.tensor([0 if ex.expected == "subject" else 1 for ex in train_rows], dtype=torch.long)

    for _epoch in range(200):
        optim.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optim.step()

    return NeuralMLPModel(vocab=vocab, model=model, torch=torch)


def accuracy(preds: list[str], rows: list[Example]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for pred, ex in zip(preds, rows) if pred == ex.expected)
    return correct / len(rows)


def per_scenario_accuracy(preds: list[str], rows: list[Example]) -> dict[str, float]:
    grouped_preds: dict[str, list[str]] = defaultdict(list)
    grouped_rows: dict[str, list[Example]] = defaultdict(list)
    for pred, ex in zip(preds, rows):
        grouped_preds[ex.scenario].append(pred)
        grouped_rows[ex.scenario].append(ex)
    return {scenario: accuracy(grouped_preds[scenario], grouped_rows[scenario]) for scenario in sorted(grouped_rows)}


def failure_cases(preds: list[str], rows: list[Example], max_rows: int = 5) -> list[dict[str, str]]:
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
            if len(bad) >= max_rows:
                break
    return bad


def build_markdown_report(summary: dict, output_json_path: str | None) -> str:
    lines = []
    lines.append("# ONA Bridge Benchmark Results")
    lines.append("")
    if output_json_path:
        lines.append(f"Source JSON: `{output_json_path}`")
        lines.append("")

    lines.append("## Overall Accuracy")
    lines.append("")
    lines.append("| Method | Accuracy |")
    lines.append("|---|---:|")
    for method_name, method_data in summary["methods"].items():
        lines.append(f"| {method_name} | {method_data['overall_accuracy']:.3f} |")
    lines.append("")

    lines.append("## Accuracy by Scenario")
    lines.append("")
    scenarios = summary["scenarios"]
    header = "| Method | " + " | ".join(scenarios) + " |"
    divider = "|---|" + "|".join(["---:" for _ in scenarios]) + "|"
    lines.append(header)
    lines.append(divider)
    for method_name, method_data in summary["methods"].items():
        values = [f"{method_data['scenario_accuracy'].get(s, 0.0):.3f}" for s in scenarios]
        lines.append("| " + method_name + " | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> dict:
    examples = build_benchmark_examples()
    mapper = CalibratedConceptMapper()

    neural_model = train_neural_mlp_baseline(examples)
    neural_preds = [neural_model.predict(ex.sentence) for ex in examples]

    direct_bridge = FitReasoningBridge(embedder=mapper, concept_threshold=args.concept_threshold, rule_mode="direct")
    multihop_bridge = FitReasoningBridge(embedder=mapper, concept_threshold=args.concept_threshold, rule_mode="multihop")
    runner = ONAFileRunner(args.ona_cmd)

    method_preds: dict[str, list[str]] = {
        "exact lexical baseline": [exact_lexical_baseline(ex) for ex in examples],
        "embedding-only bridge": [embedding_only_bridge(ex, mapper) for ex in examples],
        "neural baseline (MLP)": neural_preds,
        "ONA direct rule propagation": run_ona_method(
            runner=runner,
            bridge=direct_bridge,
            examples=examples,
            include_context=False,
            cycles=args.cycles,
            timeout_sec=args.timeout_sec,
        ),
        "ONA conflicting evidence / revision": run_ona_method(
            runner=runner,
            bridge=direct_bridge,
            examples=examples,
            include_context=True,
            cycles=args.cycles,
            timeout_sec=args.timeout_sec,
        ),
        "ONA multi-hop causal chain": run_ona_method(
            runner=runner,
            bridge=multihop_bridge,
            examples=examples,
            include_context=True,
            cycles=args.cycles,
            timeout_sec=args.timeout_sec,
        ),
    }

    scenarios = sorted({ex.scenario for ex in examples})
    summary = {
        "n_examples": len(examples),
        "scenarios": scenarios,
        "methods": {},
    }
    for method_name, preds in method_preds.items():
        summary["methods"][method_name] = {
            "overall_accuracy": accuracy(preds, examples),
            "scenario_accuracy": per_scenario_accuracy(preds, examples),
            "failure_cases": failure_cases(preds, examples, max_rows=8),
        }

    return {
        "summary": summary,
        "examples": [
            {
                "sentence": ex.sentence,
                "subject": ex.subject,
                "object": ex.object,
                "adjective": ex.adjective,
                "expected": ex.expected,
                "scenario": ex.scenario,
                "context_rules": ex.context_rules,
            }
            for ex in examples
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run explicit six-way ablations for the ONA bridge benchmark.")
    parser.add_argument("--ona-cmd", required=True, help='ONA command without "shell", e.g. "./NAR" or "/path/to/NAR".')
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--concept-threshold", type=float, default=0.20)
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument("--output-md", default="benchmark_results.md")
    args = parser.parse_args()

    results = run_benchmark(args)

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = build_markdown_report(results["summary"], str(out_json))
    out_md = Path(args.output_md)
    out_md.write_text(md + "\n", encoding="utf-8")

    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
