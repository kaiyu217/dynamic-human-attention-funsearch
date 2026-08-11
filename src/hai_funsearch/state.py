from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    REJECT = "reject"
    PROMOTE = "promote"
    REVIEW = "review"


class ReviewOutcome(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"


@dataclass(frozen=True)
class State:
    promise: float
    uncertainty: float
    budget: int
    remaining: int

    def clipped(self) -> "State":
        return State(
            promise=min(1.0, max(0.0, self.promise)),
            uncertainty=min(1.0, max(0.0, self.uncertainty)),
            budget=max(0, int(self.budget)),
            remaining=max(0, int(self.remaining)),
        )
