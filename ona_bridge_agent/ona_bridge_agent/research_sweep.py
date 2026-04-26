from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .research_eval import build_markdown, run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-split research evaluation and aggregate summary.")
    parser.add_argument("--ona-cmd", required=True)
    parser.add_argument("--n-pairs", type=int, default=60)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--mlp-seeds", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--concept-threshold", type=float, default=0.20)
    parser.add_argument("--split-seeds", default="0,1,2,3,4", help="Comma-separated split seeds.")
    parser.add_argument("--output-json", default="research_sweep_results.json")
    parser.add_argument("--output-md", default="research_sweep_results.md")
    args = parser.parse_args()

    split_seeds = [int(x.strip()) for x in args.split_seeds.split(",") if x.strip()]
    if not split_seeds:
        raise ValueError("No split seeds provided")

    per_split = {}
    for split_seed in split_seeds:
        run_args = argparse.Namespace(
            ona_cmd=args.ona_cmd,
            n_pairs=args.n_pairs,
            train_frac=args.train_frac,
            split_seed=split_seed,
            mlp_seeds=args.mlp_seeds,
            bootstrap_samples=args.bootstrap_samples,
            cycles=args.cycles,
            timeout_sec=args.timeout_sec,
            concept_threshold=args.concept_threshold,
        )
        result = run_eval(run_args)
        per_split[f"seed_{split_seed}"] = result

    methods = list(next(iter(per_split.values()))["summary"]["methods"].keys())
    aggregate = {"n_splits": len(split_seeds), "split_seeds": split_seeds, "methods": {}}
    for method in methods:
        vals = [per_split[f"seed_{s}"]["summary"]["methods"][method]["overall_accuracy"] for s in split_seeds]
        aggregate["methods"][method] = {
            "mean_accuracy": sum(vals) / len(vals),
            "std_accuracy": statistics.pstdev(vals),
            "min_accuracy": min(vals),
            "max_accuracy": max(vals),
            "per_split": vals,
        }

    out = {"aggregate": aggregate, "per_split": per_split}
    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Research Sweep Results")
    lines.append("")
    lines.append(f"Source JSON: `{out_json}`")
    lines.append("")
    lines.append(f"Split seeds: {', '.join(str(s) for s in split_seeds)}")
    lines.append("")
    lines.append("| Method | Mean Acc | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for method, row in aggregate["methods"].items():
        lines.append(
            f"| {method} | {row['mean_accuracy']:.3f} | {row['std_accuracy']:.3f} | {row['min_accuracy']:.3f} | {row['max_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-Split Detailed Tables")
    lines.append("")
    for split_seed in split_seeds:
        lines.append(f"### split_seed={split_seed}")
        lines.append("")
        summary = per_split[f"seed_{split_seed}"]["summary"]
        lines.append(build_markdown(summary, f"seed_{split_seed}"))
        lines.append("")

    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(aggregate, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
