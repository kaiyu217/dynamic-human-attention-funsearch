from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math

from .state import Action, ReviewOutcome, State


@dataclass
class StructuralModel:
    """Transparent structural model used for theory demos and unit tests.

    Empirical work should replace these functions with estimates from repeated runs.
    """

    review_value_scale: float = 0.65
    review_cost: float = 0.04
    revision_prob_scale: float = 0.65
    feedback_strength: float = 0.18
    transition_spread: float = 0.08

    def auto_reward(self, action: Action, p: float, u: float) -> float:
        if action == Action.REJECT:
            return 0.0
        if action == Action.PROMOTE:
            # Positive only for sufficiently promising candidates; uncertainty penalizes auto-promotion.
            return (2.0 * p - 1.0) - 0.20 * u * (1.0 - p)
        raise ValueError("auto_reward only supports reject/promote")

    def review_increment(self, p: float, u: float) -> float:
        # VOI is largest for intermediate promise and high uncertainty.
        return self.review_value_scale * u * 4.0 * p * (1.0 - p)

    def expected_review_reward(self, p: float, u: float) -> float:
        base = max(
            self.auto_reward(Action.REJECT, p, u),
            self.auto_reward(Action.PROMOTE, p, u),
        )
        return base + self.review_increment(p, u) - self.review_cost

    def immediate_reward(self, action: Action, p: float, u: float) -> float:
        if action == Action.REVIEW:
            return self.expected_review_reward(p, u)
        return self.auto_reward(action, p, u)

    def review_outcome_probs(self, p: float, u: float) -> dict[ReviewOutcome, float]:
        revise = min(0.80, self.revision_prob_scale * u * 4.0 * p * (1.0 - p))
        rest = 1.0 - revise
        approve = rest * p
        reject = rest * (1.0 - p)
        return {
            ReviewOutcome.APPROVE: approve,
            ReviewOutcome.REJECT: reject,
            ReviewOutcome.REVISE: revise,
        }

    def _two_point(self, mean_p: float, mean_u: float) -> list[tuple[float, float, float]]:
        """Two-point approximation to stochastic next-candidate features."""
        d = self.transition_spread
        lo_p = min(1.0, max(0.0, mean_p - d))
        hi_p = min(1.0, max(0.0, mean_p + d))
        lo_u = min(1.0, max(0.0, mean_u - d / 2))
        hi_u = min(1.0, max(0.0, mean_u + d / 2))
        return [(0.5, lo_p, hi_u), (0.5, hi_p, lo_u)]

    def next_feature_distribution(
        self,
        action: Action,
        p: float,
        u: float,
        outcome: ReviewOutcome | None = None,
    ) -> list[tuple[float, float, float]]:
        """Return (probability, next_promise, next_uncertainty)."""
        if action == Action.REJECT:
            mean_p = 0.46 + 0.05 * p
            mean_u = 0.56
        elif action == Action.PROMOTE:
            mean_p = 0.44 + 0.24 * p
            mean_u = max(0.18, 0.55 - 0.12 * p)
        elif action == Action.REVIEW:
            if outcome is None:
                raise ValueError("Review transition requires an outcome")
            if outcome == ReviewOutcome.APPROVE:
                mean_p = 0.48 + 0.28 * p + 0.05 * self.feedback_strength
                mean_u = max(0.12, 0.48 - 0.15 * u)
            elif outcome == ReviewOutcome.REJECT:
                # Negative feedback redirects search away from a bad lineage.
                mean_p = 0.47 + 0.08 * (1.0 - p)
                mean_u = max(0.15, 0.50 - 0.10 * u)
            else:  # revise
                mean_p = 0.50 + 0.22 * p + self.feedback_strength * u * (1.0 - p)
                mean_u = max(0.10, 0.44 - 0.18 * u)
        else:
            raise ValueError(action)
        return self._two_point(min(0.96, mean_p), min(0.95, mean_u))

    def transition_distribution(self, state: State, action: Action) -> list[tuple[float, State]]:
        if state.remaining <= 1:
            return []
        next_budget = state.budget - (1 if action == Action.REVIEW else 0)
        next_remaining = state.remaining - 1
        out: list[tuple[float, State]] = []
        if action != Action.REVIEW:
            for prob, p2, u2 in self.next_feature_distribution(action, state.promise, state.uncertainty):
                out.append((prob, State(p2, u2, next_budget, next_remaining)))
            return out

        for outcome, po in self.review_outcome_probs(state.promise, state.uncertainty).items():
            for pt, p2, u2 in self.next_feature_distribution(
                action, state.promise, state.uncertainty, outcome=outcome
            ):
                out.append((po * pt, State(p2, u2, next_budget, next_remaining)))
        total = sum(prob for prob, _ in out)
        return [(prob / total, s) for prob, s in out]
