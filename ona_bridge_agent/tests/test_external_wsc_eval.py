from pathlib import Path

import pytest

from ona_bridge_agent.external_wsc_eval import (
    _DEFAULT_WSC_ARROW,
    _sanitize_atom,
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
