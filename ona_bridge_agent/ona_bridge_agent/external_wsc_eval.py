from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from .ona import ONAFileRunner, max_score_for_term

_DEFAULT_WSC_ARROW = Path(
    "/Users/maxeeem/.cache/huggingface/datasets/winograd_wsc/wsc273/0.0.0/"
    "0651311f3b6dda14889d9a063030a02458395ee50ab9f41cca4cd5a89c0c3dce/"
    "winograd_wsc-test.arrow"
)
_DEFAULT_ST_MODEL = Path(
    "/Users/maxeeem/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)

_BECAUSE_WAS_RE = re.compile(r"\bbecause\s+([^,.;]+?)\s+was\s+([^,.]+)", re.IGNORECASE)
_ATOM_RE = re.compile(r"[^a-zA-Z0-9_]")


@dataclass(frozen=True)
class WSCExample:
    idx: int
    text: str
    pronoun: str
    pronoun_loc: int
    options: tuple[str, str]
    label: int
    source: str


@dataclass(frozen=True)
class CausalSubsetExample:
    idx: int
    group_key: str
    text: str
    descriptor: str
    options: tuple[str, str]
    label: int


def _sanitize_atom(text: str) -> str:
    atom = _ATOM_RE.sub("_", text.strip().lower())
    atom = re.sub(r"_+", "_", atom).strip("_")
    if not atom:
        atom = "descriptor"
    if atom[0].isdigit():
        atom = "n_" + atom
    return atom


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _bootstrap_ci_binary(correct: list[int], samples: int = 2000, seed: int = 123) -> tuple[float, float]:
    import random

    rng = random.Random(seed)
    n = len(correct)
    if n == 0:
        return (0.0, 0.0)
    vals = []
    for _ in range(samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(correct[i] for i in idxs) / n)
    vals.sort()
    lo = vals[int(0.025 * (samples - 1))]
    hi = vals[int(0.975 * (samples - 1))]
    return (lo, hi)


def _mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> dict[str, float]:
    b = 0  # a wrong, b right
    c = 0  # a right, b wrong
    for ac, bc in zip(a_correct, b_correct):
        if ac == 0 and bc == 1:
            b += 1
        elif ac == 1 and bc == 0:
            c += 1
    n = b + c
    if n == 0:
        return {"b": 0.0, "c": 0.0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p_value = min(1.0, 2.0 * tail)
    return {"b": float(b), "c": float(c), "p_value": p_value}


def _replace_pronoun(ex: WSCExample, option: str) -> str:
    start = ex.pronoun_loc
    end = start + len(ex.pronoun)
    return ex.text[:start] + option + ex.text[end:]


def load_wsc_examples(arrow_path: Path) -> list[WSCExample]:
    ds = Dataset.from_file(str(arrow_path))
    out: list[WSCExample] = []
    for i, row in enumerate(ds):
        out.append(
            WSCExample(
                idx=i,
                text=row["text"],
                pronoun=row["pronoun"],
                pronoun_loc=int(row["pronoun_loc"]),
                options=(row["options"][0], row["options"][1]),
                label=int(row["label"]),
                source=row["source"],
            )
        )
    return out


def build_causal_subset(examples: list[WSCExample]) -> list[CausalSubsetExample]:
    groups: dict[str, list[CausalSubsetExample]] = {}
    for ex in examples:
        m = _BECAUSE_WAS_RE.search(ex.text)
        if not m:
            continue
        pron_phrase = m.group(1).strip().lower()
        descriptor = m.group(2).strip().lower()
        prefix = ex.text[: m.start()].strip().lower()
        opts = tuple(o.lower().strip() for o in ex.options)
        key = f"{prefix}||{opts[0]}||{opts[1]}||{pron_phrase}"
        row = CausalSubsetExample(
            idx=ex.idx,
            group_key=key,
            text=ex.text,
            descriptor=descriptor,
            options=ex.options,
            label=ex.label,
        )
        groups.setdefault(key, []).append(row)

    subset: list[CausalSubsetExample] = []
    for key, rows in groups.items():
        if len(rows) != 2:
            continue
        labels = {r.label for r in rows}
        descriptors = {r.descriptor for r in rows}
        # Keep strict minimal pairs: opposite labels and distinct descriptors.
        if labels != {0, 1}:
            continue
        if len(descriptors) != 2:
            continue
        subset.extend(sorted(rows, key=lambda r: r.idx))
    return sorted(subset, key=lambda r: r.idx)


def _resolve_lm_snapshot(model_name: str) -> Path:
    model_dir = Path(f"/Users/maxeeem/.cache/huggingface/hub/models--{model_name}/snapshots")
    if not model_dir.exists():
        raise FileNotFoundError(f"Local snapshot directory not found for model {model_name}: {model_dir}")
    snaps = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    if not snaps:
        raise FileNotFoundError(f"No local snapshots found for model {model_name}: {model_dir}")
    return snaps[-1]


def eval_full_wsc(
    examples: list[WSCExample],
    st_model_path: Path,
    lm_snapshots: dict[str, Path],
) -> dict:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    st = SentenceTransformer(str(st_model_path))
    lm_runtime: dict[str, tuple[object, object]] = {}
    for name, snap in lm_snapshots.items():
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True)
        model.eval()
        lm_runtime[name] = (tok, model)

    method_preds: dict[str, list[int]] = {
        "sentence_transformer_replacement": [],
        "nearest_mention": [],
    }
    for name in lm_runtime:
        method_preds[f"{name}_sentence_score"] = []
    rows = []

    for ex in examples:
        text0 = _replace_pronoun(ex, ex.options[0])
        text1 = _replace_pronoun(ex, ex.options[1])

        # ST context similarity baseline.
        vecs = st.encode([ex.text, text0, text1], normalize_embeddings=False)
        sim0 = _cosine(vecs[0], vecs[1])
        sim1 = _cosine(vecs[0], vecs[2])
        pred_st = 0 if sim0 >= sim1 else 1

        # Nearest-mention baseline (distance from pronoun position).
        left = ex.text[: ex.pronoun_loc].lower()
        pos0 = left.rfind(ex.options[0].lower())
        pos1 = left.rfind(ex.options[1].lower())
        d0 = (ex.pronoun_loc - pos0) if pos0 >= 0 else 10**9
        d1 = (ex.pronoun_loc - pos1) if pos1 >= 0 else 10**9
        pred_nearest = 0 if d0 <= d1 else 1

        row = {
            "idx": ex.idx,
            "text": ex.text,
            "options": [ex.options[0], ex.options[1]],
            "label": ex.label,
            "pred_sentence_transformer_replacement": pred_st,
            "pred_nearest_mention": pred_nearest,
            "sim0": sim0,
            "sim1": sim1,
        }
        method_preds["sentence_transformer_replacement"].append(pred_st)
        method_preds["nearest_mention"].append(pred_nearest)

        for lm_name, (tok, model) in lm_runtime.items():
            with torch.no_grad():
                t0 = tok(text0, return_tensors="pt")
                t1 = tok(text1, return_tensors="pt")
                l0 = model(**t0, labels=t0["input_ids"]).loss.item()
                l1 = model(**t1, labels=t1["input_ids"]).loss.item()
            score0 = -l0
            score1 = -l1
            pred = 0 if score0 >= score1 else 1
            key = f"{lm_name}_sentence_score"
            method_preds[key].append(pred)
            row[f"pred_{key}"] = pred
            row[f"{lm_name}_score0"] = score0
            row[f"{lm_name}_score1"] = score1

        rows.append(row)

    summary_methods = {}
    for name, preds in method_preds.items():
        correct = [1 if p == ex.label else 0 for p, ex in zip(preds, examples)]
        ci = _bootstrap_ci_binary(correct)
        summary_methods[name] = {
            "accuracy": sum(correct) / len(correct),
            "bootstrap_ci_95": [ci[0], ci[1]],
        }

    anchor_name = max(summary_methods.items(), key=lambda kv: kv[1]["accuracy"])[0]
    anchor_preds = method_preds[anchor_name]
    anchor_correct = [1 if p == ex.label else 0 for p, ex in zip(anchor_preds, examples)]

    mcnemar_vs_anchor = {}
    for name, preds in method_preds.items():
        if name == anchor_name:
            continue
        corr = [1 if p == ex.label else 0 for p, ex in zip(preds, examples)]
        mcnemar_vs_anchor[name] = _mcnemar_exact(corr, anchor_correct)

    return {
        "n_examples": len(examples),
        "anchor_method": anchor_name,
        "methods": summary_methods,
        "mcnemar_vs_anchor": mcnemar_vs_anchor,
        "rows": rows,
    }


def _ona_predict(output: str, atom: str) -> tuple[int, dict[str, float]]:
    term0 = f"<{atom} --> option0_coref>"
    term1 = f"<{atom} --> option1_coref>"
    score0 = max_score_for_term(output, term0)
    score1 = max_score_for_term(output, term1)
    pred = 0 if score0 >= score1 else 1
    return pred, {"option0": score0, "option1": score1}


def _descriptor_centroids(train_rows: list[CausalSubsetExample], model) -> tuple[list[float], list[float]]:
    by_label = {0: [], 1: []}
    for r in train_rows:
        by_label[r.label].append(r.descriptor)
    if not by_label[0] or not by_label[1]:
        raise ValueError("Need both labels in training descriptors")
    v0 = model.encode(by_label[0], normalize_embeddings=False)
    v1 = model.encode(by_label[1], normalize_embeddings=False)
    c0 = (v0.sum(axis=0) / len(v0)).tolist()
    c1 = (v1.sum(axis=0) / len(v1)).tolist()
    return c0, c1


def eval_ona_subset(subset_rows: list[CausalSubsetExample], st_model_path: Path, ona_cmd: str, cycles: int) -> dict:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch

    model = SentenceTransformer(str(st_model_path))
    runner = ONAFileRunner(ona_cmd)

    groups: dict[str, list[CausalSubsetExample]] = {}
    for r in subset_rows:
        groups.setdefault(r.group_key, []).append(r)
    ordered_group_keys = sorted(groups.keys())

    # Precompute feature vectors for learned bridge.
    row_features: dict[int, list[float]] = {}
    row_sims: dict[int, tuple[float, float]] = {}
    for ex in subset_rows:
        # Approximate pronoun replacement at first matching pronoun token.
        low = ex.text.lower()
        pron_match = re.search(r"\b(it|he|she|they)\b", low)
        if pron_match is None:
            start = 0
            end = 0
        else:
            start = pron_match.start()
            end = pron_match.end()
        text0 = ex.text[:start] + ex.options[0] + ex.text[end:]
        text1 = ex.text[:start] + ex.options[1] + ex.text[end:]
        vec = model.encode(
            [ex.descriptor, ex.options[0], ex.options[1], ex.text, text0, text1],
            normalize_embeddings=False,
        )
        s_desc0 = _cosine(vec[0], vec[1])
        s_desc1 = _cosine(vec[0], vec[2])
        s_ctx0 = _cosine(vec[3], vec[4])
        s_ctx1 = _cosine(vec[3], vec[5])
        s_opt = _cosine(vec[4], vec[5])
        feats = [
            s_desc0,
            s_desc1,
            s_ctx0,
            s_ctx1,
            s_opt,
            s_desc0 - s_desc1,
            s_ctx0 - s_ctx1,
        ]
        row_features[ex.idx] = feats
        row_sims[ex.idx] = (s_desc0, s_desc1)

    pred_centroid: dict[int, int] = {}
    pred_ona_direct: dict[int, int] = {}
    pred_ona_multihop: dict[int, int] = {}
    pred_learned_bridge: dict[int, int] = {}
    pred_learned_ona_direct: dict[int, int] = {}
    pred_learned_ona_multihop: dict[int, int] = {}
    details = []

    for group_key in ordered_group_keys:
        test_rows = groups[group_key]
        train_rows = [r for k, rows in groups.items() if k != group_key for r in rows]
        c0, c1 = _descriptor_centroids(train_rows, model)

        # Train learned bridge on training rows for this LOPO split.
        x_train = np.array([row_features[r.idx] for r in train_rows], dtype=np.float32)
        y_train = np.array([r.label for r in train_rows], dtype=np.int64)
        x_test = np.array([row_features[r.idx] for r in test_rows], dtype=np.float32)
        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd

        tx = torch.tensor(x_train, dtype=torch.float32)
        ty = torch.tensor(y_train, dtype=torch.long)
        ttest = torch.tensor(x_test, dtype=torch.float32)
        clf = torch.nn.Linear(tx.shape[1], 2)
        opt = torch.optim.Adam(clf.parameters(), lr=0.05, weight_decay=1e-3)
        for _epoch in range(400):
            opt.zero_grad()
            logits = clf(tx)
            loss = torch.nn.functional.cross_entropy(logits, ty)
            loss.backward()
            opt.step()
        with torch.no_grad():
            test_logits = clf(ttest)
            test_probs = torch.nn.functional.softmax(test_logits, dim=1).cpu().numpy().tolist()

        for ex, probs in zip(test_rows, test_probs):
            dvec = model.encode([ex.descriptor], normalize_embeddings=False)[0]
            sim0 = _cosine(dvec, c0)
            sim1 = _cosine(dvec, c1)
            centroid_pred = 0 if sim0 >= sim1 else 1

            atom = _sanitize_atom(ex.descriptor)
            f0 = max(0.0, sim0)
            f1 = max(0.0, sim1)
            c0v = min(0.95, max(0.4, f0 * 0.9))
            c1v = min(0.95, max(0.4, f1 * 0.9))

            lines_direct = [
                f"<{atom} --> descriptor>. %1.00;0.90%",
                f"<{atom} --> option0_like>. %{f0:.2f};{c0v:.2f}%",
                f"<{atom} --> option1_like>. %{f1:.2f};{c1v:.2f}%",
                "<option0_like --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_coref>. %1.00;0.90%",
                f"<{atom} --> option0_coref>?",
                f"<{atom} --> option1_coref>?",
                str(cycles),
            ]
            out_direct, _ = runner.run(lines_direct, timeout_sec=10, keep_file=False)
            pred_direct, scores_direct = _ona_predict(out_direct, atom)

            lines_multihop = [
                f"<{atom} --> descriptor>. %1.00;0.90%",
                f"<{atom} --> option0_like>. %{f0:.2f};{c0v:.2f}%",
                f"<{atom} --> option1_like>. %{f1:.2f};{c1v:.2f}%",
                "<option0_like --> option0_intermediate>. %1.00;0.90%",
                "<option0_intermediate --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_intermediate>. %1.00;0.90%",
                "<option1_intermediate --> option1_coref>. %1.00;0.90%",
                f"<{atom} --> option0_coref>?",
                f"<{atom} --> option1_coref>?",
                str(cycles),
            ]
            out_multihop, _ = runner.run(lines_multihop, timeout_sec=10, keep_file=False)
            pred_multihop, scores_multihop = _ona_predict(out_multihop, atom)

            # Learned bridge prediction and learned ONA conversions.
            lp0 = float(probs[0])
            lp1 = float(probs[1])
            learned_pred = 0 if lp0 >= lp1 else 1
            lc0 = min(0.95, max(0.4, lp0 * 0.9 + 0.05))
            lc1 = min(0.95, max(0.4, lp1 * 0.9 + 0.05))
            lines_learned_direct = [
                f"<{atom} --> descriptor>. %1.00;0.90%",
                f"<{atom} --> option0_like>. %{lp0:.2f};{lc0:.2f}%",
                f"<{atom} --> option1_like>. %{lp1:.2f};{lc1:.2f}%",
                "<option0_like --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_coref>. %1.00;0.90%",
                f"<{atom} --> option0_coref>?",
                f"<{atom} --> option1_coref>?",
                str(cycles),
            ]
            out_ld, _ = runner.run(lines_learned_direct, timeout_sec=10, keep_file=False)
            pred_ld, scores_ld = _ona_predict(out_ld, atom)

            lines_learned_multihop = [
                f"<{atom} --> descriptor>. %1.00;0.90%",
                f"<{atom} --> option0_like>. %{lp0:.2f};{lc0:.2f}%",
                f"<{atom} --> option1_like>. %{lp1:.2f};{lc1:.2f}%",
                "<option0_like --> option0_intermediate>. %1.00;0.90%",
                "<option0_intermediate --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_intermediate>. %1.00;0.90%",
                "<option1_intermediate --> option1_coref>. %1.00;0.90%",
                f"<{atom} --> option0_coref>?",
                f"<{atom} --> option1_coref>?",
                str(cycles),
            ]
            out_lm, _ = runner.run(lines_learned_multihop, timeout_sec=10, keep_file=False)
            pred_lm, scores_lm = _ona_predict(out_lm, atom)

            pred_centroid[ex.idx] = centroid_pred
            pred_ona_direct[ex.idx] = pred_direct
            pred_ona_multihop[ex.idx] = pred_multihop
            pred_learned_bridge[ex.idx] = learned_pred
            pred_learned_ona_direct[ex.idx] = pred_ld
            pred_learned_ona_multihop[ex.idx] = pred_lm
            details.append(
                {
                    "idx": ex.idx,
                    "group_key": ex.group_key,
                    "descriptor": ex.descriptor,
                    "label": ex.label,
                    "sim0": sim0,
                    "sim1": sim1,
                    "learned_prob0": lp0,
                    "learned_prob1": lp1,
                    "pred_descriptor_centroid": centroid_pred,
                    "pred_ona_direct": pred_direct,
                    "pred_ona_multihop": pred_multihop,
                    "pred_learned_bridge": learned_pred,
                    "pred_learned_ona_direct": pred_ld,
                    "pred_learned_ona_multihop": pred_lm,
                    "ona_scores_direct": scores_direct,
                    "ona_scores_multihop": scores_multihop,
                    "ona_scores_learned_direct": scores_ld,
                    "ona_scores_learned_multihop": scores_lm,
                }
            )

    ordered = sorted(subset_rows, key=lambda r: r.idx)
    method_preds = {
        "descriptor_centroid_lopo": [pred_centroid[r.idx] for r in ordered],
        "ona_direct_lopo": [pred_ona_direct[r.idx] for r in ordered],
        "ona_multihop_lopo": [pred_ona_multihop[r.idx] for r in ordered],
        "learned_bridge_lopo": [pred_learned_bridge[r.idx] for r in ordered],
        "learned_ona_direct_lopo": [pred_learned_ona_direct[r.idx] for r in ordered],
        "learned_ona_multihop_lopo": [pred_learned_ona_multihop[r.idx] for r in ordered],
    }

    summary_methods = {}
    for name, preds in method_preds.items():
        correct = [1 if p == r.label else 0 for p, r in zip(preds, ordered)]
        ci = _bootstrap_ci_binary(correct)
        summary_methods[name] = {
            "accuracy": sum(correct) / len(correct),
            "bootstrap_ci_95": [ci[0], ci[1]],
        }

    best_name = max(
        summary_methods.items(),
        key=lambda kv: kv[1]["accuracy"],
    )[0]
    best_correct = [1 if p == r.label else 0 for p, r in zip(method_preds[best_name], ordered)]
    mcnemar_vs_multihop = {}
    for name, preds in method_preds.items():
        if name == best_name:
            continue
        corr = [1 if p == r.label else 0 for p, r in zip(preds, ordered)]
        mcnemar_vs_multihop[name] = _mcnemar_exact(corr, best_correct)

    return {
        "n_examples": len(ordered),
        "n_groups": len(ordered_group_keys),
        "anchor_method": best_name,
        "methods": summary_methods,
        "mcnemar_vs_ona_multihop": mcnemar_vs_multihop,
        "rows": sorted(details, key=lambda d: d["idx"]),
    }


def to_markdown(results: dict, output_json_path: str) -> str:
    lines = []
    lines.append("# External WSC Evaluation (Offline, No-Mock)")
    lines.append("")
    lines.append(f"Source JSON: `{output_json_path}`")
    lines.append("")

    full = results["full_wsc273"]
    lines.append("## Full WSC273")
    lines.append("")
    lines.append(f"Examples: {full['n_examples']}")
    lines.append("")
    lines.append("| Method | Accuracy | 95% CI |")
    lines.append("|---|---:|---|")
    for name, d in full["methods"].items():
        lo, hi = d["bootstrap_ci_95"]
        lines.append(f"| {name} | {d['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    anchor = full.get("anchor_method", "")
    lines.append(f"McNemar vs `{anchor}`:")
    lines.append("")
    lines.append("| Method | b | c | p-value |")
    lines.append("|---|---:|---:|---:|")
    for name, d in full["mcnemar_vs_anchor"].items():
        lines.append(f"| {name} | {int(d['b'])} | {int(d['c'])} | {d['p_value']:.6f} |")
    lines.append("")

    subset = results["causal_subset_lopo"]
    lines.append("## WSC Causal `because ... was ...` Paired Subset")
    lines.append("")
    lines.append(
        f"Examples: {subset['n_examples']} across {subset['n_groups']} minimal-pair groups "
        "(leave-one-group-out training for centroid/ONA mapping)"
    )
    lines.append("")
    lines.append("| Method | Accuracy | 95% CI |")
    lines.append("|---|---:|---|")
    for name, d in subset["methods"].items():
        lo, hi = d["bootstrap_ci_95"]
        lines.append(f"| {name} | {d['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    subset_anchor = subset.get("anchor_method", "ona_multihop_lopo")
    lines.append(f"McNemar vs `{subset_anchor}`:")
    lines.append("")
    lines.append("| Method | b | c | p-value |")
    lines.append("|---|---:|---:|---:|")
    for name, d in subset["mcnemar_vs_ona_multihop"].items():
        lines.append(f"| {name} | {int(d['b'])} | {int(d['c'])} | {d['p_value']:.6f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline external WSC273 evaluation with executable ONA subset.")
    parser.add_argument("--wsc-arrow-path", default=str(_DEFAULT_WSC_ARROW))
    parser.add_argument("--st-model-path", default=str(_DEFAULT_ST_MODEL))
    parser.add_argument(
        "--lm-models",
        default="gpt2,gpt2-medium",
        help="Comma-separated local HuggingFace causal LM model names to score full WSC273.",
    )
    parser.add_argument("--ona-cmd", required=True, help='ONA command without "shell", e.g. "./NAR".')
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--output-json", default="external_wsc_results.json")
    parser.add_argument("--output-md", default="external_wsc_results.md")
    args = parser.parse_args()

    wsc_path = Path(args.wsc_arrow_path)
    if not wsc_path.exists():
        raise FileNotFoundError(f"WSC Arrow file not found: {wsc_path}")

    lm_names = [m.strip() for m in args.lm_models.split(",") if m.strip()]
    lm_paths = {name: _resolve_lm_snapshot(name) for name in lm_names}

    all_examples = load_wsc_examples(wsc_path)
    full = eval_full_wsc(all_examples, Path(args.st_model_path), lm_paths)

    subset_rows = build_causal_subset(all_examples)
    subset = eval_ona_subset(subset_rows, Path(args.st_model_path), args.ona_cmd, cycles=args.cycles)

    out = {
        "config": {
            "wsc_arrow_path": str(wsc_path),
            "st_model_path": args.st_model_path,
            "lm_model_paths": {k: str(v) for k, v in lm_paths.items()},
            "ona_cmd": args.ona_cmd,
            "cycles": args.cycles,
        },
        "full_wsc273": full,
        "causal_subset_lopo": subset,
    }

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.write_text(to_markdown(out, str(out_json)) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "full_wsc273_n": full["n_examples"],
            "full_wsc273_methods": full["methods"],
            "causal_subset_n": subset["n_examples"],
            "causal_subset_groups": subset["n_groups"],
            "causal_subset_methods": subset["methods"],
        },
        indent=2,
    ))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
