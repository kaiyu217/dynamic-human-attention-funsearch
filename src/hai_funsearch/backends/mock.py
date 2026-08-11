from __future__ import annotations
import numpy as np
from .base import Candidate


class MockSearchBackend:
    """API-free debugging backend."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.incumbent = 0.0
        self.feedback = 0.0

    def propose_and_evaluate(self) -> Candidate:
        self.t += 1
        score = float(self.rng.normal(0.45 + self.feedback, 0.18))
        score = float(np.clip(score, 0, 1))
        return Candidate(
            candidate_id=f"cand-{self.t:04d}",
            code=f"def heuristic_{self.t}(x): return {score:.6f}",
            raw_score=score,
            metrics={"score": score},
            diagnostics={"margin": score - self.incumbent},
        )

    def reject(self, candidate: Candidate) -> None:
        self.feedback *= 0.95

    def promote(self, candidate: Candidate) -> None:
        self.incumbent = max(self.incumbent, candidate.raw_score)
        self.feedback = min(0.25, self.feedback + 0.03 * candidate.raw_score)

    def apply_human_feedback(self, candidate: Candidate, outcome: str, targeted_prompt: str | None = None) -> None:
        if outcome == "approve":
            self.promote(candidate)
            self.feedback = min(0.30, self.feedback + 0.03)
        elif outcome == "revise":
            self.feedback = min(0.35, self.feedback + 0.08)
        else:
            self.feedback = max(-0.05, self.feedback - 0.01)
