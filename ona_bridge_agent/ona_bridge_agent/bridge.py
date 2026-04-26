from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from .types import BridgeFrame, Claim

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _sanitize_atom(text: str) -> str:
    """Keep generated atoms boring so ONA accepts them."""
    atom = re.sub(r"[^a-zA-Z0-9_]", "_", text.strip().lower())
    atom = re.sub(r"_+", "_", atom).strip("_")
    if not atom:
        raise ValueError(f"Cannot sanitize empty atom from {text!r}")
    if atom[0].isdigit():
        atom = "n_" + atom
    return atom


class ToyFitSyntaxExtractor:
    """
    Extracts the deliberately tiny grammar used by the experiment:

        The trophy did not fit in the suitcase because it was large.
        The trophy fit in the suitcase because it was large.

    This is intentionally not the research claim. It is just the controlled
    sentence-to-frame frontend so we can isolate the bridge/ONA behavior.
    """

    def extract(self, sentence: str) -> tuple[str, str, str, bool]:
        tokens = [_sanitize_atom(t) for t in _TOKEN_RE.findall(sentence)]
        if len(tokens) < 10:
            raise ValueError(f"Sentence too short for toy grammar: {sentence}")
        if tokens[0] != "the":
            raise ValueError(f"Expected sentence to start with 'The': {sentence}")
        subject = tokens[1]
        negated = "not" in tokens
        try:
            in_idx = tokens.index("in")
            object_ = tokens[in_idx + 2] if tokens[in_idx + 1] == "the" else tokens[in_idx + 1]
            adjective = tokens[-1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Could not parse toy fit sentence: {sentence}") from exc
        return subject, object_, adjective, negated


class CalibratedConceptMapper:
    """
    Hand-calibrated mapping from adjectives to concept memberships.

    This replaces fuzzy embeddings with explicit, auditable mappings
    for better control over uncertainty in the neuro-symbolic loop.
    """

    def __init__(self):
        # Calibrated mappings: adjective -> {concept: (frequency, confidence)}
        self.mappings = {
            "large": {"large_like": (0.9, 0.8)},
            "huge": {"large_like": (0.95, 0.85)},
            "oversized": {"large_like": (0.7, 0.7), "small_like": (0.3, 0.6)},  # Conflicting
            "enormous": {"large_like": (0.92, 0.82)},
            "big": {"large_like": (0.85, 0.75)},
            "small": {"small_like": (0.9, 0.8)},
            "tiny": {"small_like": (0.95, 0.85)},
            "narrow": {"small_like": (0.88, 0.78)},
            "cramped": {"small_like": (0.82, 0.72)},
            "little": {"small_like": (0.87, 0.77)},
            # Terms that require multi-hop rules to reach fit-failure causes.
            "bulky": {"bulky_like": (0.93, 0.83)},
            "massive": {"bulky_like": (0.90, 0.80), "large_like": (0.10, 0.40)},
            "compact": {"compact_like": (0.90, 0.80)},
            "slender": {"compact_like": (0.88, 0.78)},
        }
        self.concepts = ("large_like", "small_like", "bulky_like", "compact_like")

    def truth_for(self, adjective: str, concept: str) -> tuple[float, float]:
        adj_lower = adjective.lower()
        if adj_lower in self.mappings and concept in self.mappings[adj_lower]:
            return self.mappings[adj_lower][concept]
        return 0.0, 0.0

    def memberships(self, adjective: str) -> dict[str, float]:
        adj_lower = adjective.lower()
        out = {concept: 0.0 for concept in self.concepts}
        if adj_lower in self.mappings:
            for concept, (freq, _conf) in self.mappings[adj_lower].items():
                out[concept] = freq
        return out


class SentenceTransformerConceptEmbedder:
    """
    Real pretrained-vector backend using SentenceTransformers.

    Provides genuine neural embeddings where cosine similarity
    maps directly to NARS frequency, with confidence scaling
    proportional to frequency to reflect prediction certainty.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", prototypes: dict[str, list[str]] | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Please run `pip install sentence-transformers`."
            )

        self.model = SentenceTransformer(model_name)
        self.prototypes = prototypes or {
            "large_like": ["large", "huge", "oversized", "enormous", "big", "gigantic", "massive"],
            "small_like": ["small", "tiny", "narrow", "cramped", "little", "miniature", "tight"],
        }
        
        # Precompute prototype vectors (average of words in each category)
        self.prototype_vectors = {}
        for concept, words in self.prototypes.items():
            vecs = self.model.encode(words)
            self.prototype_vectors[concept] = sum(vecs) / len(vecs)

    def cosine(self, a, b) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def memberships(self, adjective: str) -> dict[str, float]:
        v = self.model.encode([adjective])[0]
        memberships = {}
        for concept, pv in self.prototype_vectors.items():
            sim = self.cosine(v, pv)
            # Map cosine similarity [-1, 1] to frequency [0, 1] using a rectifier/scaling
            freq = max(0.0, sim) 
            memberships[concept] = freq
            
        # SentenceTransformers often cluster antonyms. Only keep the strongest match to avoid contradictory spam, 
        # unless it's genuinely ambiguous.
        if memberships:
            max_concept = max(memberships, key=memberships.get)
            max_val = memberships[max_concept]
            return {concept: val for concept, val in memberships.items() if val >= max_val - 0.15}
        return memberships


class OptionalGloveConceptEmbedder:
    """
    Optional real pretrained-vector backend.

    Use this when you have a local GloVe-style text file:
        word val1 val2 ... valN

    Example:
        --glove-path ~/embeddings/glove.6B.50d.txt

    No internet/download is attempted here.
    """

    def __init__(self, glove_path: str | Path, prototypes: dict[str, list[str]] | None = None):
        self.prototypes = prototypes or {
            "large_like": ["large", "huge", "oversized", "enormous", "big"],
            "small_like": ["small", "tiny", "narrow", "cramped", "little"],
        }
        needed = {w for words in self.prototypes.values() for w in words}
        self.vectors: dict[str, list[float]] = {}
        path = Path(glove_path).expanduser()
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                word = parts[0].lower()
                if word in needed:
                    self.vectors[word] = [float(x) for x in parts[1:]]
        self.prototype_vectors = {}
        for concept, words in self.prototypes.items():
            vecs = [self.vectors[w] for w in words if w in self.vectors]
            if vecs:
                self.prototype_vectors[concept] = [sum(col) / len(vecs) for col in zip(*vecs)]

    def _vector_for_oov(self, word: str) -> list[float] | None:
        # Keep OOV behavior explicit; don't silently invent semantic knowledge.
        return self.vectors.get(word.lower())

    def cosine(self, a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def memberships(self, adjective: str) -> dict[str, float]:
        v = self._vector_for_oov(adjective)
        if v is None:
            return {concept: 0.0 for concept in self.prototype_vectors}
        return {concept: self.cosine(v, pv) for concept, pv in self.prototype_vectors.items()}


class FitReasoningBridge:
    """
    Bridge from toy English sentence -> uncertain ONA-compatible Narsese claims.

    The core idea:
      syntax extractor gives the controlled frame
      concept embedder gives soft memberships like huge -> large_like
      NARS truth values receive those memberships as frequency/confidence inputs
    """

    def __init__(
        self,
        embedder=None,
        concept_threshold: float = 0.20,
        rule_mode: Literal["direct", "multihop"] = "multihop",
    ):
        self.syntax = ToyFitSyntaxExtractor()
        self.embedder = embedder or CalibratedConceptMapper()
        self.concept_threshold = concept_threshold
        self.rule_mode = rule_mode

    def _background_rules(self) -> list[Claim]:
        if self.rule_mode == "direct":
            return [
                Claim("<large_like --> subject_cause_of_fit_failure>", 1.00, 0.90, "background_rule"),
                Claim("<small_like --> object_cause_of_fit_failure>", 1.00, 0.90, "background_rule"),
            ]

        return [
            Claim("<large_like --> object_too_big>", 1.00, 0.90, "background_rule"),
            Claim("<object_too_big --> subject_cause_of_fit_failure>", 1.00, 0.90, "background_rule"),
            Claim("<small_like --> container_too_small>", 1.00, 0.90, "background_rule"),
            Claim("<container_too_small --> object_cause_of_fit_failure>", 1.00, 0.90, "background_rule"),
            Claim("<bulky_like --> large_like>", 1.00, 0.88, "background_rule"),
            Claim("<compact_like --> small_like>", 1.00, 0.88, "background_rule"),
        ]

    def extract(self, sentence: str, known_adjective: str | None = None) -> BridgeFrame:
        subject, object_, adjective, negated = self.syntax.extract(sentence)
        if known_adjective:
            adjective = known_adjective
            
        frame = BridgeFrame(
            sentence=sentence,
            subject=subject,
            object=object_,
            relation="fit_in",
            adjective=adjective,
            negated=negated,
        )

        frame.claims.extend(
            [
                Claim(f"<{subject} --> subject>", 1.00, 0.90, "syntax"),
                Claim(f"<{object_} --> object>", 1.00, 0.90, "syntax"),
                Claim("<fit_in --> relation>", 1.00, 0.85, "syntax"),
                Claim(f"<{adjective} --> observed_property>", 1.00, 0.90, "syntax"),
            ]
        )
        frame.claims.append(
            Claim("<fit_failure --> current_event>" if negated else "<fit_success --> current_event>", 1.00, 0.85, "syntax")
        )

        for concept, sim in self.embedder.memberships(adjective).items():
            if sim >= self.concept_threshold:
                # For calibrated mapper, use auditable per-concept truth values.
                if hasattr(self.embedder, "truth_for"):
                    freq, conf = self.embedder.truth_for(adjective, concept)
                    frame.claims.append(Claim(f"<{adjective} --> {concept}>", freq, conf, "calibrated_concept"))
                elif hasattr(self.embedder, "model"):
                    # Map cosine similarity (which is `sim` here) to ONA parameters
                    freq = min(1.0, max(0.0, sim))
                    # Confidence drops if frequency drops, but stays bounded
                    conf = min(0.9, max(0.4, sim * 0.9))
                    frame.claims.append(Claim(f"<{adjective} --> {concept}>", freq, conf, "neural_embedding"))
                else:
                    # Fallback for old embedders
                    frame.claims.append(Claim(f"<{adjective} --> {concept}>", sim, 0.65, "concept_embedding"))

        # Add conflicting evidence for certain adjectives to test ONA revision
        if adjective == "oversized":
            # Add conflicting small_like with moderate confidence
            frame.claims.append(Claim(f"<{adjective} --> small_like>", 0.49, 0.50, "conflicting_evidence"))

        frame.claims.extend(self._background_rules())
        return frame

    def to_narsese(self, frame: BridgeFrame, cycles: int = 40) -> list[str]:
        lines = [claim.to_narsese() for claim in frame.claims]
        adj = frame.adjective
        # Queries are ONA-compatible basic inheritance questions.
        lines.extend(
            [
                f"<{adj} --> subject_cause_of_fit_failure>?",
                f"<{adj} --> object_cause_of_fit_failure>?",
                str(cycles),
            ]
        )
        return lines
