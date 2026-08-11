from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .backends.base import SearchBackend, Candidate
from .state import Action, ReviewOutcome, State


@dataclass
class ControllerConfig:
    horizon: int
    attention_budget: int


class SimulatedOracleReviewer:
    """Reproducible reviewer for policy research; not a behavioral human model."""
    def review(self, candidate: Candidate, promise: float, uncertainty: float) -> tuple[ReviewOutcome, str | None]:
        # Transparent deterministic rules suitable for unit tests / pilots.
        if promise >= 0.72 and uncertainty <= 0.65:
            return ReviewOutcome.APPROVE, None
        if promise <= 0.30 and uncertainty <= 0.60:
            return ReviewOutcome.REJECT, None
        return ReviewOutcome.REVISE, "Focus the next mutation on the candidate's weakest evaluator diagnostic."


class AttentionController:
    def __init__(self, backend: SearchBackend, policy, reviewer=None, config: ControllerConfig | None = None):
        self.backend = backend
        self.policy = policy
        self.reviewer = reviewer or SimulatedOracleReviewer()
        self.config = config or ControllerConfig(horizon=20, attention_budget=4)

    @staticmethod
    def candidate_features(candidate: Candidate) -> tuple[float, float]:
        # Placeholder mapping. Replace with calibrated models learned from repeated runs.
        promise = float(np.clip(candidate.raw_score, 0, 1))
        margin = abs(float(candidate.diagnostics.get("margin", 0.0)))
        uncertainty = float(np.clip(1.0 - min(1.0, margin * 2.0), 0, 1))
        return promise, uncertainty

    def run(self):
        budget = self.config.attention_budget
        records = []
        for t in range(1, self.config.horizon + 1):
            cand = self.backend.propose_and_evaluate()
            p, u = self.candidate_features(cand)
            state = State(p, u, budget, self.config.horizon - t + 1)
            action = self.policy.action(state)
            outcome = None
            prompt = None
            if action == Action.REVIEW and budget > 0:
                budget -= 1
                outcome, prompt = self.reviewer.review(cand, p, u)
                self.backend.apply_human_feedback(cand, outcome.value, prompt)
            elif action == Action.PROMOTE:
                self.backend.promote(cand)
            else:
                self.backend.reject(cand)
            records.append({
                "t": t,
                "candidate_id": cand.candidate_id,
                "raw_score": cand.raw_score,
                "promise": p,
                "uncertainty": u,
                "budget_after": budget,
                "action": action.value,
                "review_outcome": outcome.value if outcome else None,
                "revision_prompt": prompt,
            })
        return records
