from ona_bridge_agent.dataset import build_research_examples
from ona_bridge_agent.research_eval import mcnemar_exact, split_by_pair


def test_build_research_examples_size():
    rows = build_research_examples(n_pairs=5)
    assert len(rows) == 30  # 6 examples per noun pair
    scenarios = {r.scenario for r in rows}
    assert scenarios == {"lexical_core", "synonym_generalization", "conflicting_evidence", "multihop_chain"}


def test_split_by_pair_disjoint():
    rows = build_research_examples(n_pairs=10)
    train, test = split_by_pair(rows, train_frac=0.5, split_seed=7)
    train_ids = {int(r.subject.rsplit("_", 1)[1]) for r in train}
    test_ids = {int(r.subject.rsplit("_", 1)[1]) for r in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == 10


def test_mcnemar_exact_nontrivial():
    class R:
        def __init__(self, expected):
            self.expected = expected

    rows = [R("subject"), R("object"), R("subject"), R("object")]
    a_preds = ["subject", "subject", "subject", "object"]  # one miss at index 1
    b_preds = ["subject", "object", "object", "object"]  # one miss at index 2
    stat = mcnemar_exact(a_preds, b_preds, rows)
    assert stat["b"] == 1
    assert stat["c"] == 1
    assert 0.0 <= stat["p_value"] <= 1.0
