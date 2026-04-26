from pathlib import Path

import pytest

from ona_bridge_agent.external_wsc_eval import (
    _DEFAULT_WSC_ARROW,
    _nearest_mention_probs,
    _sanitize_atom,
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
