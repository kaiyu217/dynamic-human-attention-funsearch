from dataclasses import dataclass
from ..state import Action, State


@dataclass
class AIOnly:
    promote_threshold: float = 0.70
    def action(self, state: State) -> Action:
        return Action.PROMOTE if state.promise >= self.promote_threshold else Action.REJECT


@dataclass
class PeriodicReview:
    period: int = 5
    horizon: int = 20
    promote_threshold: float = 0.70
    def action(self, state: State) -> Action:
        t = self.horizon - state.remaining + 1
        if state.budget > 0 and t % self.period == 0:
            return Action.REVIEW
        return Action.PROMOTE if state.promise >= self.promote_threshold else Action.REJECT


@dataclass
class ScoreBandReview:
    low: float = 0.40
    high: float = 0.70
    def action(self, state: State) -> Action:
        if state.budget > 0 and self.low <= state.promise < self.high:
            return Action.REVIEW
        return Action.PROMOTE if state.promise >= self.high else Action.REJECT


@dataclass
class UncertaintyReview:
    uncertainty_threshold: float = 0.70
    promote_threshold: float = 0.70
    def action(self, state: State) -> Action:
        if state.budget > 0 and state.uncertainty >= self.uncertainty_threshold:
            return Action.REVIEW
        return Action.PROMOTE if state.promise >= self.promote_threshold else Action.REJECT
