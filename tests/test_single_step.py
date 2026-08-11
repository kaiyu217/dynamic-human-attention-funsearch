from hai_funsearch.model import StructuralModel
from hai_funsearch.policy.single_step import decide
from hai_funsearch.state import Action


def test_threshold_pattern_high_uncertainty():
    m = StructuralModel()
    lam = 0.18
    assert decide(m, 0.05, 0.9, lam).action == Action.REJECT
    assert decide(m, 0.50, 0.9, lam).action == Action.REVIEW
    assert decide(m, 0.95, 0.9, lam).action == Action.PROMOTE
