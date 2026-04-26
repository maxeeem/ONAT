from ona_bridge_agent.bridge import FitReasoningBridge
from ona_bridge_agent.dataset import EXAMPLES


def test_bridge_extracts_all_examples():
    bridge = FitReasoningBridge()
    for ex in EXAMPLES:
        frame = bridge.extract(ex.sentence)
        assert frame.subject == ex.subject
        assert frame.object == ex.object
        assert frame.adjective == ex.adjective
        lines = bridge.to_narsese(frame)
        assert any(f"<{ex.adjective} -->" in line for line in lines)
        assert any("subject_cause_of_fit_failure" in line for line in lines)
        assert any("object_cause_of_fit_failure" in line for line in lines)


def test_embedding_bridge_memberships_are_available():
    bridge = FitReasoningBridge()
    for adj in ["large", "small", "huge", "tiny", "oversized", "narrow"]:
        memberships = bridge.embedder.memberships(adj)
        assert "large_like" in memberships
        assert "small_like" in memberships
        assert max(memberships.values()) > 0
