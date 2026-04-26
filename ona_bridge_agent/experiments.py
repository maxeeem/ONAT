from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .bridge import CalibratedConceptMapper, FitReasoningBridge, OptionalGloveConceptEmbedder, SentenceTransformerConceptEmbedder
from .dataset import EXAMPLES, HELDOUT_EXAMPLES
from .ona import ONAFileRunner, predict_from_ona_output
from .types import Example


def exact_lexical_baseline(ex: Example) -> str | None:
    if ex.adjective == "large":
        return "subject"
    if ex.adjective == "small":
        return "object"
    return None


def embedding_bridge_only(bridge: FitReasoningBridge, ex: Example) -> tuple[str | None, dict[str, float]]:
    memberships = bridge.embedder.memberships(ex.adjective)
    large = memberships.get("large_like", 0.0)
    small = memberships.get("small_like", 0.0)
    if large == 0 and small == 0:
        return None, memberships
    return ("subject" if large >= small else "object"), memberships


def run_suite(args: argparse.Namespace) -> int:
    if args.use_huggingface:
        embedder = SentenceTransformerConceptEmbedder(args.hf_model)
        embedder_name = f"hf_sentence_transformer_{args.hf_model}"
    elif args.glove_path:
        embedder = OptionalGloveConceptEmbedder(args.glove_path)
        embedder_name = "glove"
    else:
        embedder = CalibratedConceptMapper()
        embedder_name = "calibrated"

    bridge = FitReasoningBridge(embedder=embedder, concept_threshold=args.concept_threshold)
    data = EXAMPLES + (HELDOUT_EXAMPLES if args.include_heldout else [])

    runner = ONAFileRunner(args.ona_cmd) if args.ona_cmd else None
    rows = []

    for ex in data:
        frame = bridge.extract(ex.sentence)
        narsese = bridge.to_narsese(frame, cycles=args.cycles)

        exact_pred = exact_lexical_baseline(ex)
        embed_pred, memberships = embedding_bridge_only(bridge, ex)
        ona_pred = None
        ona_scores = None
        ona_output = None
        nal_file = None

        if runner is not None:
            ona_output, nal_file = runner.run(
                narsese,
                timeout_sec=args.timeout_sec,
                keep_file=args.keep_nal_files,
            )
            ona_pred, ona_scores, ona_explanations = predict_from_ona_output(ona_output, frame.adjective)

        row = {
            "sentence": ex.sentence,
            "adjective": ex.adjective,
            "expected": ex.expected,
            "exact_pred": exact_pred,
            "embedding_pred": embed_pred,
            "ona_pred": ona_pred,
            "embedding_memberships": memberships,
            "ona_scores": ona_scores,
            "ona_explanations": ona_explanations if runner is not None else None,
            "narsese": narsese if args.verbose else None,
            "ona_output": ona_output if args.verbose else None,
            "nal_file": nal_file,
        }
        rows.append(row)

        status_exact = "PASS" if exact_pred == ex.expected else "FAIL"
        status_embed = "PASS" if embed_pred == ex.expected else "FAIL"
        status_ona = "SKIP" if runner is None else ("PASS" if ona_pred == ex.expected else "FAIL")
        print(f"{ex.sentence}")
        print(f"  expected={ex.expected}")
        print(f"  exact={exact_pred} [{status_exact}]")
        print(f"  embedding={embed_pred} memberships={memberships} [{status_embed}]")
        if runner is not None:
            print(f"  ona={ona_pred} scores={ona_scores} [{status_ona}]")
            if ona_explanations and ona_pred:
                print("  ona_explanation (logical derivation):")
                for step in ona_explanations.get(ona_pred, []):
                    print(f"    - {step}")
        print()

    def acc(key: str):
        vals = [r for r in rows if r[key] is not None]
        if not vals:
            return None
        return sum(1 for r in vals if r[key] == r["expected"]) / len(vals)

    summary = {
        "embedder": embedder_name,
        "n": len(rows),
        "exact_accuracy": acc("exact_pred"),
        "embedding_accuracy": acc("embedding_pred"),
        "ona_accuracy": acc("ona_pred"),
    }
    print("SUMMARY")
    print(json.dumps(summary, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    # Return failure only if ONA was requested and did not match all examples.
    if runner is not None:
        return 0 if summary["ona_accuracy"] == 1.0 else 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Toy ONA bridge experiment for fit/coreference reasoning.")
    parser.add_argument("--ona-cmd", default=None, help='ONA command without "shell", e.g. "./NAR" or "/path/to/NAR".')
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--concept-threshold", type=float, default=0.20)
    parser.add_argument("--glove-path", default=None, help="Optional local GloVe-style embeddings file.")
    parser.add_argument("--use-huggingface", action="store_true", help="Use sentence-transformers pretrained model.")
    parser.add_argument("--hf-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name to use.")
    parser.add_argument("--include-heldout", action="store_true", help="Include extra adjectives not in the primary suite.")
    parser.add_argument("--verbose", action="store_true", help="Include Narsese and raw ONA output in JSON/stdout.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--keep-nal-files", action="store_true")
    args = parser.parse_args()
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
