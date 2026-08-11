from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd

from ..model import StructuralModel
from ..state import Action, State


class FiniteHorizonDP:
    def __init__(
        self,
        model: StructuralModel,
        p_grid: np.ndarray | None = None,
        u_grid: np.ndarray | None = None,
    ):
        self.model = model
        self.p_grid = np.asarray(p_grid if p_grid is not None else np.linspace(0, 1, 21), dtype=float)
        self.u_grid = np.asarray(u_grid if u_grid is not None else np.linspace(0, 1, 11), dtype=float)

    def snap(self, x: float, grid: np.ndarray) -> float:
        return float(grid[np.argmin(np.abs(grid - x))])

    def canonical(self, state: State) -> State:
        return State(
            self.snap(state.promise, self.p_grid),
            self.snap(state.uncertainty, self.u_grid),
            state.budget,
            state.remaining,
        )

    @lru_cache(maxsize=None)
    def value(self, p: float, u: float, b: int, n: int) -> float:
        if n <= 0:
            return 0.0
        state = State(p, u, b, n)
        return max(self.q_value(state, a) for a in self.feasible_actions(state))

    def feasible_actions(self, state: State) -> list[Action]:
        actions = [Action.REJECT, Action.PROMOTE]
        if state.budget > 0:
            actions.append(Action.REVIEW)
        return actions

    def q_value(self, state: State, action: Action) -> float:
        imm = self.model.immediate_reward(action, state.promise, state.uncertainty)
        if state.remaining <= 1:
            return imm
        future = 0.0
        for prob, s2 in self.model.transition_distribution(state, action):
            s2 = self.canonical(s2)
            future += prob * self.value(s2.promise, s2.uncertainty, s2.budget, s2.remaining)
        return imm + future

    def action(self, state: State) -> Action:
        state = self.canonical(state)
        values = {a: self.q_value(state, a) for a in self.feasible_actions(state)}
        return max(values, key=values.get)

    def policy_table(self, max_budget: int, horizon: int) -> pd.DataFrame:
        rows = []
        for n in range(1, horizon + 1):
            for b in range(0, max_budget + 1):
                for u in self.u_grid:
                    for p in self.p_grid:
                        s = State(float(p), float(u), b, n)
                        a = self.action(s)
                        rows.append({
                            "promise": p,
                            "uncertainty": u,
                            "budget": b,
                            "remaining": n,
                            "action": a.value,
                            "value": self.value(float(p), float(u), b, n),
                        })
        return pd.DataFrame(rows)

    def threshold_table(self, max_budget: int, horizon: int) -> pd.DataFrame:
        policy = self.policy_table(max_budget=max_budget, horizon=horizon)
        rows = []
        for (u, b, n), g in policy.groupby(["uncertainty", "budget", "remaining"], sort=True):
            g = g.sort_values("promise")
            review_ps = g.loc[g.action == Action.REVIEW.value, "promise"].to_numpy()
            promote_ps = g.loc[g.action == Action.PROMOTE.value, "promise"].to_numpy()
            rows.append({
                "uncertainty": u,
                "budget": int(b),
                "remaining": int(n),
                "review_lower": float(review_ps.min()) if len(review_ps) else np.nan,
                "review_upper": float(review_ps.max()) if len(review_ps) else np.nan,
                "promote_lower": float(promote_ps.min()) if len(promote_ps) else np.nan,
            })
        return pd.DataFrame(rows)
