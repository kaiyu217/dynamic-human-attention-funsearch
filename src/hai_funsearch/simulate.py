from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model import StructuralModel
from .state import Action, State


@dataclass
class EpisodeResult:
    total_reward: float
    reviews_used: int
    final_budget: int
    trajectory: pd.DataFrame


def _sample_transition(model: StructuralModel, state: State, action: Action, rng: np.random.Generator) -> State | None:
    dist = model.transition_distribution(state, action)
    if not dist:
        return None
    probs = np.array([x[0] for x in dist], dtype=float)
    probs /= probs.sum()
    idx = rng.choice(len(dist), p=probs)
    return dist[idx][1]


def run_episode(model: StructuralModel, policy, initial: State, seed: int = 0) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    state = initial
    total = 0.0
    rows = []
    initial_budget = initial.budget
    while state.remaining > 0:
        action = policy.action(state)
        if action == Action.REVIEW and state.budget <= 0:
            action = Action.REJECT
        reward = model.immediate_reward(action, state.promise, state.uncertainty)
        total += reward
        rows.append({
            "promise": state.promise,
            "uncertainty": state.uncertainty,
            "budget": state.budget,
            "remaining": state.remaining,
            "action": action.value,
            "reward": reward,
        })
        nxt = _sample_transition(model, state, action, rng)
        if nxt is None:
            state = State(state.promise, state.uncertainty, state.budget - (action == Action.REVIEW), 0)
        else:
            state = nxt
    return EpisodeResult(
        total_reward=total,
        reviews_used=initial_budget - state.budget,
        final_budget=state.budget,
        trajectory=pd.DataFrame(rows),
    )
