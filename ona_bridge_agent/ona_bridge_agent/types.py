from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Antecedent = Literal["subject", "object"]


@dataclass(frozen=True)
class Example:
    sentence: str
    subject: str
    object: str
    adjective: str
    expected: Antecedent
    negated: bool = True
    context_rules: list[str] = field(default_factory=list)
    scenario: str = "core"


@dataclass(frozen=True)
class Claim:
    """A bridge-produced Narsese claim with NARS truth values."""

    term: str
    frequency: float
    confidence: float
    source: str

    def to_narsese(self) -> str:
        f = min(max(self.frequency, 0.0), 1.0)
        c = min(max(self.confidence, 0.0), 1.0)
        return f"{self.term}. %{f:.2f};{c:.2f}%"


@dataclass
class BridgeFrame:
    sentence: str
    subject: str
    object: str
    relation: str
    adjective: str
    negated: bool
    claims: list[Claim] = field(default_factory=list)
