import numpy as np
from hai_funsearch.model import StructuralModel
from hai_funsearch.policy.dp import FiniteHorizonDP
from hai_funsearch.state import Action, State


def test_no_review_when_budget_zero():
    dp = FiniteHorizonDP(StructuralModel(), np.linspace(0, 1, 11), np.linspace(0, 1, 6))
    for p in [0.1, 0.5, 0.9]:
        assert dp.action(State(p, 0.9, 0, 5)) != Action.REVIEW


def test_review_consumes_budget_in_transition():
    m = StructuralModel()
    s = State(0.5, 0.8, 2, 4)
    nxt = m.transition_distribution(s, Action.REVIEW)
    assert nxt
    assert all(s2.budget == 1 and s2.remaining == 3 for _, s2 in nxt)
    assert abs(sum(prob for prob, _ in nxt) - 1.0) < 1e-9
