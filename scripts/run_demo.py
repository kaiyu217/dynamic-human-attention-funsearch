from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hai_funsearch.model import StructuralModel
from hai_funsearch.state import State
from hai_funsearch.policy.dp import FiniteHorizonDP
from hai_funsearch.policy.baselines import AIOnly, PeriodicReview, ScoreBandReview, UncertaintyReview
from hai_funsearch.simulate import run_episode


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    model = StructuralModel(**cfg["model"])
    p_grid = np.linspace(0, 1, cfg["p_grid_points"])
    u_grid = np.linspace(0, 1, cfg["u_grid_points"])
    dp = FiniteHorizonDP(model, p_grid=p_grid, u_grid=u_grid)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    policy = dp.policy_table(cfg["attention_budget"], cfg["horizon"])
    policy.to_csv(out / "policy_table.csv", index=False)
    thresholds = dp.threshold_table(cfg["attention_budget"], cfg["horizon"])
    thresholds.to_csv(out / "thresholds.csv", index=False)

    # Policy map at the start of the run.
    start = policy[(policy.budget == cfg["attention_budget"]) & (policy.remaining == cfg["horizon"])].copy()
    code = {"reject": 0, "review": 1, "promote": 2}
    mat = start.pivot(index="uncertainty", columns="promise", values="action").replace(code).astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat.values, origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    ax.set_xlabel("Automated promise p")
    ax.set_ylabel("Uncertainty u")
    ax.set_title("Finite-horizon policy at start of run")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Reject", "Review", "Promote"])
    fig.tight_layout()
    fig.savefig(out / "policy_map.png", dpi=180)
    plt.close(fig)

    horizon = cfg["horizon"]
    budget = cfg["attention_budget"]
    policies = {
        "dp": dp,
        "ai_only": AIOnly(),
        "periodic": PeriodicReview(period=5, horizon=horizon),
        "score_band": ScoreBandReview(),
        "uncertainty": UncertaintyReview(),
    }
    rows = []
    for name, pol in policies.items():
        for seed in range(cfg["simulation_seeds"]):
            rng = np.random.default_rng(seed)
            init = State(float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.25, 0.85)), budget, horizon)
            ep = run_episode(model, pol, init, seed=seed)
            rows.append({"policy": name, "seed": seed, "total_reward": ep.total_reward, "reviews_used": ep.reviews_used})
    sim = pd.DataFrame(rows)
    sim.to_csv(out / "simulation_runs.csv", index=False)
    summary = sim.groupby("policy").agg(
        mean_reward=("total_reward", "mean"),
        sd_reward=("total_reward", "std"),
        mean_reviews=("reviews_used", "mean"),
    ).reset_index().sort_values("mean_reward", ascending=False)
    summary.to_csv(out / "simulation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
