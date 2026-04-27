from __future__ import annotations

import argparse
import gc
import json
import math
import random
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


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


def _prob1_from_score_pair(scores: dict[str, float]) -> float:
    s0 = float(scores.get("option0", 0.0))
    s1 = float(scores.get("option1", 0.0))
    total = s0 + s1
    if total <= 1e-12:
        return 0.5
    return s1 / total


def _calibration_metrics_binary(prob1: list[float], labels: list[int], bins: int = 10) -> dict[str, float]:
    eps = 1e-8
    n = len(labels)
    if n == 0:
        return {"brier": 0.0, "log_loss": 0.0, "ece": 0.0}
    brier = sum((p - y) ** 2 for p, y in zip(prob1, labels)) / n
    log_loss = -sum(
        y * math.log(max(eps, min(1.0 - eps, p))) + (1 - y) * math.log(max(eps, min(1.0 - eps, 1.0 - p)))
        for p, y in zip(prob1, labels)
    ) / n

    conf = [max(p, 1.0 - p) for p in prob1]
    pred = [1 if p >= 0.5 else 0 for p in prob1]
    correct = [1 if pr == y else 0 for pr, y in zip(pred, labels)]
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idxs = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idxs:
            continue
        acc = sum(correct[i] for i in idxs) / len(idxs)
        avg_conf = sum(conf[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(acc - avg_conf)
    return {"brier": brier, "log_loss": log_loss, "ece": ece}


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
        # Keep paired groups with opposite labels.
        if labels != {0, 1}:
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


def _stratified_kfold_indices(labels: list[int], n_folds: int, seed: int) -> list[list[int]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    by_label: dict[int, list[int]] = {}
    for idx, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(idx)
    rng = random.Random(seed)
    for idxs in by_label.values():
        rng.shuffle(idxs)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for idxs in by_label.values():
        for j, idx in enumerate(idxs):
            folds[j % n_folds].append(idx)
    out = []
    for fold in folds:
        if fold:
            out.append(sorted(fold))
    return out


def _train_linear_softmax_probs(x_train, y_train, x_test, seed: int) -> list[list[float]]:
    import torch

    torch.manual_seed(seed)
    tx = torch.tensor(x_train, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.long)
    ttest = torch.tensor(x_test, dtype=torch.float32)
    clf = torch.nn.Linear(tx.shape[1], 2)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05, weight_decay=1e-3)
    for _epoch in range(450):
        opt.zero_grad()
        logits = clf(tx)
        loss = torch.nn.functional.cross_entropy(logits, ty)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(clf(ttest), dim=1).cpu().numpy().tolist()
    return probs


def _nearest_mention_probs(ex: WSCExample) -> tuple[float, float]:
    left = ex.text[: ex.pronoun_loc].lower()
    pos0 = left.rfind(ex.options[0].lower())
    pos1 = left.rfind(ex.options[1].lower())
    d0 = (ex.pronoun_loc - pos0) if pos0 >= 0 else 10**9
    d1 = (ex.pronoun_loc - pos1) if pos1 >= 0 else 10**9
    s0 = 0.0 if d0 >= 10**8 else 1.0 / (1.0 + float(d0))
    s1 = 0.0 if d1 >= 10**8 else 1.0 / (1.0 + float(d1))
    denom = s0 + s1
    if denom <= 1e-12:
        return 0.5, 0.5
    return s0 / denom, s1 / denom


def _confidence_from_probs(p0: float, p1: float, low: float = 0.45, high: float = 0.95) -> float:
    margin = abs(p0 - p1)
    return min(high, max(low, low + (high - low) * margin))


def _ona_lines_from_probs(
    atom: str,
    p0: float,
    p1: float,
    cycles: int,
    mode: str,
    secondary: tuple[float, float] | None = None,
) -> list[str]:
    c_primary = _confidence_from_probs(p0, p1, low=0.50, high=0.95)
    lines = [
        f"<{atom} --> descriptor>. %1.00;0.90%",
        f"<{atom} --> option0_like>. %{p0:.3f};{c_primary:.3f}%",
        f"<{atom} --> option1_like>. %{p1:.3f};{c_primary:.3f}%",
    ]
    if secondary is not None:
        s0, s1 = secondary
        c_secondary = _confidence_from_probs(s0, s1, low=0.35, high=0.75)
        lines.extend(
            [
                f"<{atom} --> option0_like>. %{s0:.3f};{c_secondary:.3f}%",
                f"<{atom} --> option1_like>. %{s1:.3f};{c_secondary:.3f}%",
            ]
        )
    if mode == "multihop":
        lines.extend(
            [
                "<option0_like --> option0_intermediate>. %1.00;0.90%",
                "<option0_intermediate --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_intermediate>. %1.00;0.90%",
                "<option1_intermediate --> option1_coref>. %1.00;0.90%",
            ]
        )
    elif mode == "direct":
        lines.extend(
            [
                "<option0_like --> option0_coref>. %1.00;0.90%",
                "<option1_like --> option1_coref>. %1.00;0.90%",
            ]
        )
    else:
        raise ValueError(f"Unknown ONA mode: {mode}")
    lines.extend([f"<{atom} --> option0_coref>?", f"<{atom} --> option1_coref>?", str(cycles)])
    return lines


def _tune_gated_mixture_params(
    train_indices: list[int],
    labels: list[int],
    roberta_prob1: list[float],
    gpt2m_prob1: list[float],
    bert_prob1: list[float],
    inner_seed: int,
) -> dict[str, float]:
    # threshold on RoBERTa confidence; fallback is convex mix of (gpt2-medium, bert, roberta)
    threshold_grid = [x / 100.0 for x in range(5, 45, 2)]
    weight_grid = [x / 10.0 for x in range(0, 11)]

    y_train = [labels[i] for i in train_indices]
    inner_folds_local = _stratified_kfold_indices(y_train, n_folds=4, seed=inner_seed)

    best_acc = -1.0
    best_params = {"threshold": 0.35, "w_gpt2m": 0.7, "w_bert": 0.0}
    for thr in threshold_grid:
        for w_g in weight_grid:
            for w_b in weight_grid:
                if w_g + w_b > 1.0:
                    continue
                w_r = 1.0 - w_g - w_b
                vals = []
                for local_val_idxs in inner_folds_local:
                    val_global = [train_indices[i] for i in sorted(set(local_val_idxs))]
                    preds = []
                    for idx in val_global:
                        p_r = roberta_prob1[idx]
                        conf = abs(p_r - 0.5)
                        if conf >= thr:
                            p = p_r
                        else:
                            p = w_g * gpt2m_prob1[idx] + w_b * bert_prob1[idx] + w_r * p_r
                        preds.append(1 if p >= 0.5 else 0)
                    acc = sum(1 for p, idx in zip(preds, val_global) if p == labels[idx]) / len(val_global)
                    vals.append(acc)
                mean_acc = sum(vals) / len(vals)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = {"threshold": thr, "w_gpt2m": w_g, "w_bert": w_b}
    return best_params


def _score_causal_lm_sentence(tokenizer, model, text: str) -> float:
    import torch

    with torch.no_grad():
        toks = tokenizer(text, return_tensors="pt")
        loss = model(**toks, labels=toks["input_ids"]).loss.item()
    return -loss


def _score_mlm_option(tokenizer, model, text: str, option_start: int, option_end: int) -> float:
    import torch

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token id for MLM scoring")

    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=256,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"]

    target_idxs = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if max(s, option_start) < min(e, option_end):
            target_idxs.append(i)
    if not target_idxs:
        return float("-inf")

    total = 0.0
    with torch.no_grad():
        for idx in target_idxs:
            masked = input_ids.clone()
            orig = int(masked[idx].item())
            masked[idx] = mask_id
            out = model(input_ids=masked.unsqueeze(0), attention_mask=attention_mask)
            logp = torch.log_softmax(out.logits[0, idx], dim=-1)[orig].item()
            total += logp
    return total / len(target_idxs)


def eval_full_wsc(
    examples: list[WSCExample],
    st_model_path: Path,
    causal_lm_snapshots: dict[str, Path],
    mlm_snapshots: dict[str, Path],
) -> dict:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    st = SentenceTransformer(str(st_model_path))

    method_preds: dict[str, list[int]] = {
        "sentence_transformer_replacement": [],
        "nearest_mention": [],
    }
    rows: list[dict] = []
    prepared: list[tuple[WSCExample, str, str]] = []

    for ex in examples:
        text0 = _replace_pronoun(ex, ex.options[0])
        text1 = _replace_pronoun(ex, ex.options[1])
        prepared.append((ex, text0, text1))

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

        rows.append(row)

    for lm_name, snap in causal_lm_snapshots.items():
        key = f"{lm_name}_sentence_score"
        method_preds[key] = []
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(snap), local_files_only=True)
        model.eval()
        for i, (_ex, text0, text1) in enumerate(prepared):
            score0 = _score_causal_lm_sentence(tok, model, text0)
            score1 = _score_causal_lm_sentence(tok, model, text1)
            pred = 0 if score0 >= score1 else 1
            method_preds[key].append(pred)
            rows[i][f"pred_{key}"] = pred
            rows[i][f"{lm_name}_score0"] = score0
            rows[i][f"{lm_name}_score1"] = score1
        del model
        del tok
        gc.collect()

    for mlm_name, snap in mlm_snapshots.items():
        key = f"{mlm_name}_mlm_option_score"
        method_preds[key] = []
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(str(snap), local_files_only=True)
        model.eval()
        for i, (ex, text0, text1) in enumerate(prepared):
            s = ex.pronoun_loc
            e0 = s + len(ex.options[0])
            e1 = s + len(ex.options[1])
            score0 = _score_mlm_option(tok, model, text0, s, e0)
            score1 = _score_mlm_option(tok, model, text1, s, e1)
            pred = 0 if score0 >= score1 else 1
            method_preds[key].append(pred)
            rows[i][f"pred_{key}"] = pred
            rows[i][f"{mlm_name}_score0"] = score0
            rows[i][f"{mlm_name}_score1"] = score1
        del model
        del tok
        gc.collect()

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


def eval_full_wsc_learned_cv(
    examples: list[WSCExample],
    st_model_path: Path,
    ona_cmd: str,
    cycles: int,
    cv_folds: int,
    cv_seed: int,
    aux_score_rows: list[dict] | None = None,
) -> dict:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(str(st_model_path))
    runner = ONAFileRunner(ona_cmd)
    aux_by_idx = {int(r["idx"]): r for r in (aux_score_rows or []) if "idx" in r}
    feature_score_pairs: list[tuple[str, str]] = []
    if aux_score_rows:
        sample = aux_score_rows[0]
        for k in sorted(sample.keys()):
            if not k.endswith("_score0"):
                continue
            k1 = k[:-1] + "1"
            if k1 in sample:
                feature_score_pairs.append((k, k1))

    features: list[list[float]] = []
    labels: list[int] = []
    nearest_probs: list[tuple[float, float]] = []
    rows: list[dict] = []
    score_prob1_by_model: dict[str, list[float]] = {}
    for k0, k1 in feature_score_pairs:
        model_name = k0[:-7]  # strip "_score0"
        score_prob1_by_model[model_name] = [0.5] * len(examples)

    for i, ex in enumerate(examples):
        text0 = _replace_pronoun(ex, ex.options[0])
        text1 = _replace_pronoun(ex, ex.options[1])
        vec = model.encode(
            [ex.text, text0, text1, ex.options[0], ex.options[1]],
            normalize_embeddings=False,
        )
        sim_ctx0 = _cosine(vec[0], vec[1])
        sim_ctx1 = _cosine(vec[0], vec[2])
        sim_opt0 = _cosine(vec[0], vec[3])
        sim_opt1 = _cosine(vec[0], vec[4])
        sim_repl = _cosine(vec[1], vec[2])
        near0, near1 = _nearest_mention_probs(ex)
        feats = [
            sim_ctx0,
            sim_ctx1,
            sim_ctx0 - sim_ctx1,
            sim_opt0,
            sim_opt1,
            sim_opt0 - sim_opt1,
            sim_repl,
        ]
        aux_row = aux_by_idx.get(ex.idx, {})
        for k0, k1 in feature_score_pairs:
            s0 = float(aux_row.get(k0, 0.0))
            s1 = float(aux_row.get(k1, 0.0))
            feats.extend([s0, s1, s0 - s1])
            model_name = k0[:-7]
            score_prob1_by_model[model_name][i] = _sigmoid(s1 - s0)
        features.append(feats)
        labels.append(ex.label)
        nearest_probs.append((near0, near1))
        rows.append(
            {
                "idx": ex.idx,
                "text": ex.text,
                "options": [ex.options[0], ex.options[1]],
                "label": ex.label,
                "sim_ctx0": sim_ctx0,
                "sim_ctx1": sim_ctx1,
                "sim_opt0": sim_opt0,
                "sim_opt1": sim_opt1,
                "sim_replacement": sim_repl,
                "nearest_prob0": near0,
                "nearest_prob1": near1,
            }
        )
        for model_name, probs in score_prob1_by_model.items():
            rows[-1][f"{model_name}_prob1"] = probs[i]

    folds = _stratified_kfold_indices(labels, n_folds=cv_folds, seed=cv_seed)

    pred_bridge_linear: dict[int, int] = {}
    pred_bridge_gated: dict[int, int] = {}
    pred_ona_direct: dict[int, int] = {}
    pred_ona_multihop: dict[int, int] = {}
    pred_ona_revision: dict[int, int] = {}
    prob1_bridge_linear: dict[int, float] = {}
    prob1_bridge_gated: dict[int, float] = {}
    prob1_ona_direct: dict[int, float] = {}
    prob1_ona_multihop: dict[int, float] = {}
    prob1_ona_revision: dict[int, float] = {}
    gated_params_by_fold: list[dict[str, float] | None] = []

    gated_available = all(
        name in score_prob1_by_model for name in ["roberta-large", "gpt2-medium", "bert-base-uncased"]
    )

    for fold_id, test_idxs in enumerate(folds):
        test_set = set(test_idxs)
        train_idxs = [i for i in range(len(examples)) if i not in test_set]
        x_train = np.array([features[i] for i in train_idxs], dtype=np.float32)
        y_train = np.array([labels[i] for i in train_idxs], dtype=np.int64)
        x_test = np.array([features[i] for i in test_idxs], dtype=np.float32)
        mu = x_train.mean(axis=0, keepdims=True)
        sd = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd
        probs = _train_linear_softmax_probs(x_train, y_train, x_test, seed=cv_seed + fold_id)

        if gated_available:
            params = _tune_gated_mixture_params(
                train_indices=train_idxs,
                labels=labels,
                roberta_prob1=score_prob1_by_model["roberta-large"],
                gpt2m_prob1=score_prob1_by_model["gpt2-medium"],
                bert_prob1=score_prob1_by_model["bert-base-uncased"],
                inner_seed=cv_seed + 1000 + fold_id,
            )
            gated_params_by_fold.append(params)
        else:
            params = None
            gated_params_by_fold.append(None)

        for local_i, row_idx in enumerate(test_idxs):
            p0_linear = float(probs[local_i][0])
            p1_linear = float(probs[local_i][1])
            bridge_linear_pred = 0 if p0_linear >= p1_linear else 1

            if params is not None:
                p_roberta = score_prob1_by_model["roberta-large"][row_idx]
                conf = abs(p_roberta - 0.5)
                if conf >= params["threshold"]:
                    p1_gated = p_roberta
                else:
                    w_g = params["w_gpt2m"]
                    w_b = params["w_bert"]
                    w_r = 1.0 - w_g - w_b
                    p1_gated = (
                        w_g * score_prob1_by_model["gpt2-medium"][row_idx]
                        + w_b * score_prob1_by_model["bert-base-uncased"][row_idx]
                        + w_r * p_roberta
                    )
                p0_gated = 1.0 - p1_gated
            else:
                p0_gated = p0_linear
                p1_gated = p1_linear
            bridge_gated_pred = 0 if p0_gated >= p1_gated else 1

            atom = f"wsc_{row_idx}"

            out_direct, _ = runner.run(
                _ona_lines_from_probs(atom, p0_gated, p1_gated, cycles=cycles, mode="direct"),
                timeout_sec=10,
                keep_file=False,
            )
            ona_direct_pred, scores_direct = _ona_predict(out_direct, atom)

            out_multihop, _ = runner.run(
                _ona_lines_from_probs(atom, p0_gated, p1_gated, cycles=cycles, mode="multihop"),
                timeout_sec=10,
                keep_file=False,
            )
            ona_multihop_pred, scores_multihop = _ona_predict(out_multihop, atom)

            out_revision, _ = runner.run(
                _ona_lines_from_probs(
                    atom,
                    p0_gated,
                    p1_gated,
                    cycles=cycles,
                    mode="direct",
                    secondary=nearest_probs[row_idx],
                ),
                timeout_sec=10,
                keep_file=False,
            )
            ona_revision_pred, scores_revision = _ona_predict(out_revision, atom)

            pred_bridge_linear[row_idx] = bridge_linear_pred
            pred_bridge_gated[row_idx] = bridge_gated_pred
            pred_ona_direct[row_idx] = ona_direct_pred
            pred_ona_multihop[row_idx] = ona_multihop_pred
            pred_ona_revision[row_idx] = ona_revision_pred
            prob1_bridge_linear[row_idx] = p1_linear
            prob1_bridge_gated[row_idx] = p1_gated
            prob1_ona_direct[row_idx] = _prob1_from_score_pair(scores_direct)
            prob1_ona_multihop[row_idx] = _prob1_from_score_pair(scores_multihop)
            prob1_ona_revision[row_idx] = _prob1_from_score_pair(scores_revision)

            rows[row_idx]["cv_fold"] = fold_id
            rows[row_idx]["pred_learned_bridge_linear_kfold"] = bridge_linear_pred
            rows[row_idx]["pred_learned_bridge_gated_kfold"] = bridge_gated_pred
            rows[row_idx]["pred_learned_ona_direct_kfold"] = ona_direct_pred
            rows[row_idx]["pred_learned_ona_multihop_kfold"] = ona_multihop_pred
            rows[row_idx]["pred_learned_ona_revision_kfold"] = ona_revision_pred
            rows[row_idx]["learned_linear_prob0"] = p0_linear
            rows[row_idx]["learned_linear_prob1"] = p1_linear
            rows[row_idx]["learned_gated_prob0"] = p0_gated
            rows[row_idx]["learned_gated_prob1"] = p1_gated
            rows[row_idx]["gated_params"] = params
            rows[row_idx]["ona_scores_direct"] = scores_direct
            rows[row_idx]["ona_scores_multihop"] = scores_multihop
            rows[row_idx]["ona_scores_revision"] = scores_revision

    ordered = list(range(len(examples)))
    method_preds = {
        "learned_bridge_linear_kfold": [pred_bridge_linear[i] for i in ordered],
        "learned_bridge_gated_kfold": [pred_bridge_gated[i] for i in ordered],
        "learned_ona_direct_kfold": [pred_ona_direct[i] for i in ordered],
        "learned_ona_multihop_kfold": [pred_ona_multihop[i] for i in ordered],
        "learned_ona_revision_kfold": [pred_ona_revision[i] for i in ordered],
    }
    method_prob1 = {
        "learned_bridge_linear_kfold": [prob1_bridge_linear[i] for i in ordered],
        "learned_bridge_gated_kfold": [prob1_bridge_gated[i] for i in ordered],
        "learned_ona_direct_kfold": [prob1_ona_direct[i] for i in ordered],
        "learned_ona_multihop_kfold": [prob1_ona_multihop[i] for i in ordered],
        "learned_ona_revision_kfold": [prob1_ona_revision[i] for i in ordered],
    }
    labels_ordered = [examples[i].label for i in ordered]

    summary_methods = {}
    for name, preds in method_preds.items():
        correct = [1 if p == ex.label else 0 for p, ex in zip(preds, examples)]
        ci = _bootstrap_ci_binary(correct)
        cal = _calibration_metrics_binary(method_prob1[name], labels_ordered)
        summary_methods[name] = {
            "accuracy": sum(correct) / len(correct),
            "bootstrap_ci_95": [ci[0], ci[1]],
            "calibration": {
                "brier": cal["brier"],
                "log_loss": cal["log_loss"],
                "ece_10bin": cal["ece"],
            },
        }

    best_name = max(summary_methods.items(), key=lambda kv: kv[1]["accuracy"])[0]
    best_correct = [1 if p == ex.label else 0 for p, ex in zip(method_preds[best_name], examples)]
    mcnemar_vs_anchor = {}
    for name, preds in method_preds.items():
        if name == best_name:
            continue
        corr = [1 if p == ex.label else 0 for p, ex in zip(preds, examples)]
        mcnemar_vs_anchor[name] = _mcnemar_exact(corr, best_correct)

    return {
        "n_examples": len(examples),
        "n_folds": len(folds),
        "cv_seed": cv_seed,
        "feature_count": len(features[0]) if features else 0,
        "gated_mixture_available": gated_available,
        "gated_params_by_fold": gated_params_by_fold,
        "anchor_method": best_name,
        "methods": summary_methods,
        "mcnemar_vs_anchor": mcnemar_vs_anchor,
        "rows": rows,
    }


def compare_cv_to_full_anchor(full: dict, full_cv: dict) -> dict:
    full_anchor = full["anchor_method"]
    full_anchor_key = f"pred_{full_anchor}"
    full_by_idx = {int(r["idx"]): r for r in full["rows"]}
    cv_by_idx = {int(r["idx"]): r for r in full_cv["rows"]}
    common_idxs = sorted(set(full_by_idx.keys()) & set(cv_by_idx.keys()))

    out = {
        "full_anchor_method": full_anchor,
        "comparisons": {},
    }
    full_anchor_correct = [1 if full_by_idx[i][full_anchor_key] == full_by_idx[i]["label"] else 0 for i in common_idxs]
    full_anchor_acc = sum(full_anchor_correct) / len(full_anchor_correct)

    for cv_method in full_cv["methods"].keys():
        cv_key = f"pred_{cv_method}"
        cv_correct = [1 if cv_by_idx[i][cv_key] == cv_by_idx[i]["label"] else 0 for i in common_idxs]
        cv_acc = sum(cv_correct) / len(cv_correct)
        out["comparisons"][cv_method] = {
            "cv_accuracy": cv_acc,
            "full_anchor_accuracy": full_anchor_acc,
            "delta_accuracy": cv_acc - full_anchor_acc,
            "mcnemar_vs_full_anchor": _mcnemar_exact(cv_correct, full_anchor_correct),
        }
    return out


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

    cv = results["full_wsc273_learned_cv"]
    lines.append("## Full WSC273 Learned Bridge + ONA (Stratified CV)")
    lines.append("")
    lines.append(
        f"Examples: {cv['n_examples']} with {cv['n_folds']}-fold CV (seed {cv['cv_seed']}, "
        f"{cv['feature_count']} learned features)"
    )
    lines.append("")
    lines.append("| Method | Accuracy | 95% CI |")
    lines.append("|---|---:|---|")
    for name, d in cv["methods"].items():
        lo, hi = d["bootstrap_ci_95"]
        lines.append(f"| {name} | {d['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")
    lines.append("| Method | Brier | Log Loss | ECE (10-bin) |")
    lines.append("|---|---:|---:|---:|")
    for name, d in cv["methods"].items():
        cal = d.get("calibration", {})
        lines.append(
            f"| {name} | {cal.get('brier', 0.0):.3f} | {cal.get('log_loss', 0.0):.3f} | "
            f"{cal.get('ece_10bin', 0.0):.3f} |"
        )
    lines.append("")
    cv_anchor = cv.get("anchor_method", "")
    lines.append(f"McNemar vs `{cv_anchor}`:")
    lines.append("")
    lines.append("| Method | b | c | p-value |")
    lines.append("|---|---:|---:|---:|")
    for name, d in cv["mcnemar_vs_anchor"].items():
        lines.append(f"| {name} | {int(d['b'])} | {int(d['c'])} | {d['p_value']:.6f} |")
    lines.append("")

    cross = results.get("cross_section_full_vs_cv", {})
    if cross:
        lines.append("## Cross-Section Comparison vs Best Full-WSC Neural Baseline")
        lines.append("")
        lines.append(f"Full-WSC anchor method: `{cross['full_anchor_method']}`")
        lines.append("")
        lines.append("| CV Method | CV Acc | Anchor Acc | Delta | McNemar p-value |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, d in cross["comparisons"].items():
            stat = d["mcnemar_vs_full_anchor"]
            lines.append(
                f"| {name} | {d['cv_accuracy']:.3f} | {d['full_anchor_accuracy']:.3f} | "
                f"{d['delta_accuracy']:+.3f} | {stat['p_value']:.6f} |"
            )
        lines.append("")

    subset = results["causal_subset_lopo"]
    lines.append("## WSC Causal `because ... was ...` Paired Subset")
    lines.append("")
    lines.append(
        f"Examples: {subset['n_examples']} across {subset['n_groups']} minimal-pair groups "
        "(leave-one-group-out training for centroid/ONA mapping; includes paired groups with opposite labels)"
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
        "--causal-lm-models",
        default="gpt2,gpt2-medium",
        help="Comma-separated local HuggingFace causal LM model names to score full WSC273.",
    )
    parser.add_argument(
        "--mlm-models",
        default="bert-base-uncased,roberta-large",
        help="Comma-separated local HuggingFace masked LM model names for option token scoring.",
    )
    parser.add_argument("--ona-cmd", required=True, help='ONA command without "shell", e.g. "./NAR".')
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified folds for learned full-WSC bridge.")
    parser.add_argument("--cv-seed", type=int, default=13, help="Seed for CV fold assignment and classifier init.")
    parser.add_argument("--output-json", default="external_wsc_results.json")
    parser.add_argument("--output-md", default="external_wsc_results.md")
    args = parser.parse_args()

    wsc_path = Path(args.wsc_arrow_path)
    if not wsc_path.exists():
        raise FileNotFoundError(f"WSC Arrow file not found: {wsc_path}")

    causal_names = [m.strip() for m in args.causal_lm_models.split(",") if m.strip()]
    mlm_names = [m.strip() for m in args.mlm_models.split(",") if m.strip()]
    causal_paths = {name: _resolve_lm_snapshot(name) for name in causal_names}
    mlm_paths = {name: _resolve_lm_snapshot(name) for name in mlm_names}

    all_examples = load_wsc_examples(wsc_path)
    full = eval_full_wsc(all_examples, Path(args.st_model_path), causal_paths, mlm_paths)
    full_cv = eval_full_wsc_learned_cv(
        all_examples,
        Path(args.st_model_path),
        args.ona_cmd,
        cycles=args.cycles,
        cv_folds=args.cv_folds,
        cv_seed=args.cv_seed,
        aux_score_rows=full["rows"],
    )
    cross_section = compare_cv_to_full_anchor(full, full_cv)

    subset_rows = build_causal_subset(all_examples)
    subset = eval_ona_subset(subset_rows, Path(args.st_model_path), args.ona_cmd, cycles=args.cycles)

    out = {
        "config": {
            "wsc_arrow_path": str(wsc_path),
            "st_model_path": args.st_model_path,
            "causal_lm_model_paths": {k: str(v) for k, v in causal_paths.items()},
            "mlm_model_paths": {k: str(v) for k, v in mlm_paths.items()},
            "ona_cmd": args.ona_cmd,
            "cycles": args.cycles,
            "cv_folds": args.cv_folds,
            "cv_seed": args.cv_seed,
        },
        "full_wsc273": full,
        "full_wsc273_learned_cv": full_cv,
        "cross_section_full_vs_cv": cross_section,
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
            "full_wsc273_learned_cv_methods": full_cv["methods"],
            "cross_section_full_vs_cv": cross_section,
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
