from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass
class Candidate:
    candidate_id: str
    code: str
    raw_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SearchBackend(Protocol):
    """Minimal contract needed by the human-attention controller."""
    def propose_and_evaluate(self) -> Candidate: ...
    def reject(self, candidate: Candidate) -> None: ...
    def promote(self, candidate: Candidate) -> None: ...
    def apply_human_feedback(self, candidate: Candidate, outcome: str, targeted_prompt: str | None = None) -> None: ...
