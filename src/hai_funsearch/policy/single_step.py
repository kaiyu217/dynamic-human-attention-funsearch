from dataclasses import dataclass
from ..model import StructuralModel
from ..state import Action


@dataclass
class SingleStepDecision:
    action: Action
    q_reject: float
    q_promote: float
    q_review: float


def decide(model: StructuralModel, p: float, u: float, review_shadow_price: float = 0.0) -> SingleStepDecision:
    q_r = model.immediate_reward(Action.REJECT, p, u)
    q_p = model.immediate_reward(Action.PROMOTE, p, u)
    q_h = model.immediate_reward(Action.REVIEW, p, u) - review_shadow_price
    values = {Action.REJECT: q_r, Action.PROMOTE: q_p, Action.REVIEW: q_h}
    action = max(values, key=values.get)
    return SingleStepDecision(action, q_r, q_p, q_h)
