from __future__ import annotations
import numpy as np
import pandas as pd


def calibrate_promise_by_bins(
    df: pd.DataFrame,
    raw_score_col: str = "promise_raw",
    improved_col: str = "improved",
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dependency-light empirical calibration.

    Returns a copy of df with `promise` and a calibration table.
    """
    out = df.copy()
    out["_bin"] = pd.qcut(out[raw_score_col], q=min(bins, out[raw_score_col].nunique()), duplicates="drop")
    cal = (
        out.groupby("_bin", observed=True)
        .agg(raw_mean=(raw_score_col, "mean"), promise=(improved_col, "mean"), count=(improved_col, "size"))
        .reset_index(drop=True)
        .sort_values("raw_mean")
    )
    out["promise"] = np.interp(out[raw_score_col], cal.raw_mean, cal.promise)
    out.drop(columns=["_bin"], inplace=True)
    return out, cal


def uncertainty_from_replicates(scores: np.ndarray) -> float:
    """Normalize the standard error of repeated evaluator scores to [0,1]."""
    scores = np.asarray(scores, dtype=float)
    if scores.size <= 1:
        return 1.0
    se = float(np.std(scores, ddof=1) / np.sqrt(scores.size))
    scale = max(1e-8, float(np.mean(np.abs(scores))) + float(np.std(scores)))
    return float(np.clip(se / scale, 0.0, 1.0))


def estimate_review_outcomes(df: pd.DataFrame, p_bins: int = 5, u_bins: int = 5, alpha: float = 1.0) -> pd.DataFrame:
    d = df[df.action == "review"].copy()
    if d.empty:
        return pd.DataFrame(columns=["p_bin", "u_bin", "outcome", "probability", "count"])
    d["p_bin"] = pd.cut(d.promise, np.linspace(0, 1, p_bins + 1), include_lowest=True)
    d["u_bin"] = pd.cut(d.uncertainty, np.linspace(0, 1, u_bins + 1), include_lowest=True)
    outcomes = ["approve", "reject", "revise"]
    rows = []
    for (pb, ub), g in d.groupby(["p_bin", "u_bin"], observed=True):
        counts = g.review_outcome.value_counts().to_dict()
        denom = len(g) + alpha * len(outcomes)
        for o in outcomes:
            rows.append({"p_bin": str(pb), "u_bin": str(ub), "outcome": o,
                         "probability": (counts.get(o, 0) + alpha) / denom,
                         "count": len(g)})
    return pd.DataFrame(rows)


def estimate_transition_table(df: pd.DataFrame, p_bins: int = 5, u_bins: int = 5) -> pd.DataFrame:
    d = df.dropna(subset=["next_promise", "next_uncertainty"]).copy()
    if d.empty:
        return pd.DataFrame()
    edges_p = np.linspace(0, 1, p_bins + 1)
    edges_u = np.linspace(0, 1, u_bins + 1)
    d["p_bin"] = pd.cut(d.promise, edges_p, include_lowest=True)
    d["u_bin"] = pd.cut(d.uncertainty, edges_u, include_lowest=True)
    d["p_next_bin"] = pd.cut(d.next_promise, edges_p, include_lowest=True)
    d["u_next_bin"] = pd.cut(d.next_uncertainty, edges_u, include_lowest=True)
    keys = ["action", "review_outcome", "p_bin", "u_bin", "p_next_bin", "u_next_bin"]
    tab = d.groupby(keys, observed=True, dropna=False).size().rename("count").reset_index()
    denom_keys = ["action", "review_outcome", "p_bin", "u_bin"]
    tab["probability"] = tab["count"] / tab.groupby(denom_keys, dropna=False)["count"].transform("sum")
    return tab
