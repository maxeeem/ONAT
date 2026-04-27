from pathlib import Path

import pytest

from ona_bridge_agent.external_wsc_eval import (
    _DEFAULT_WSC_ARROW,
    _calibration_metrics_binary,
    _nearest_mention_probs,
    _prob1_from_score_pair,
    _sanitize_atom,
    _tune_gated_mixture_params,
    _stratified_kfold_indices,
    WSCExample,
    build_causal_subset,
    load_wsc_examples,
)


def test_sanitize_atom():
    assert _sanitize_atom("too hard to pronounce") == "too_hard_to_pronounce"
    assert _sanitize_atom("123 test") == "n_123_test"
    assert _sanitize_atom("...") == "descriptor"


@pytest.mark.skipif(not Path(_DEFAULT_WSC_ARROW).exists(), reason="Local cached WSC273 arrow not available")
def test_build_causal_subset_has_minimal_pairs():
    rows = load_wsc_examples(Path(_DEFAULT_WSC_ARROW))
    subset = build_causal_subset(rows)
    assert len(subset) >= 20
    keys = {r.group_key for r in subset}
    assert len(subset) == 2 * len(keys)


def test_stratified_kfold_indices_cover_all_rows():
    labels = [0, 1] * 7 + [0, 1]
    folds = _stratified_kfold_indices(labels, n_folds=4, seed=9)
    flat = sorted(i for fold in folds for i in fold)
    assert flat == list(range(len(labels)))
    assert len(folds) == 4
    for fold in folds:
        fold_labels = {labels[i] for i in fold}
        assert fold_labels == {0, 1}


def test_nearest_mention_probs_normalize():
    ex = WSCExample(
        idx=0,
        text="The trophy does not fit in the suitcase because it is too large.",
        pronoun="it",
        pronoun_loc=46,
        options=("trophy", "suitcase"),
        label=0,
        source="unit",
    )
    p0, p1 = _nearest_mention_probs(ex)
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert pytest.approx(1.0, abs=1e-8) == p0 + p1


def test_calibration_helpers_are_bounded():
    p1 = _prob1_from_score_pair({"option0": 0.3, "option1": 0.7})
    assert 0.0 <= p1 <= 1.0
    cal = _calibration_metrics_binary([0.9, 0.2, 0.7, 0.1], [1, 0, 1, 0])
    assert 0.0 <= cal["brier"] <= 1.0
    assert cal["log_loss"] >= 0.0
    assert 0.0 <= cal["ece"] <= 1.0


def test_tune_gated_mixture_params_ranges():
    train_indices = list(range(12))
    labels = [0, 1] * 6
    roberta = [0.7, 0.3] * 6
    gpt2m = [0.6, 0.4] * 6
    bert = [0.55, 0.45] * 6
    params = _tune_gated_mixture_params(
        train_indices=train_indices,
        labels=labels,
        roberta_prob1=roberta,
        gpt2m_prob1=gpt2m,
        bert_prob1=bert,
        inner_seed=7,
    )
    assert 0.0 <= params["threshold"] <= 1.0
    assert 0.0 <= params["w_gpt2m"] <= 1.0
    assert 0.0 <= params["w_bert"] <= 1.0
    assert params["w_gpt2m"] + params["w_bert"] <= 1.0 + 1e-8
