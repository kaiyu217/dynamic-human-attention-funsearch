from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hai_funsearch.model import StructuralModel
from hai_funsearch.policy.dp import FiniteHorizonDP
from hai_funsearch.backends.mock import MockSearchBackend
from hai_funsearch.controller import AttentionController, ControllerConfig


def main():
    horizon, budget = 20, 4
    dp = FiniteHorizonDP(StructuralModel(), np.linspace(0, 1, 21), np.linspace(0, 1, 11))
    ctl = AttentionController(MockSearchBackend(seed=7), dp, config=ControllerConfig(horizon, budget))
    records = ctl.run()
    df = pd.DataFrame(records)
    out = ROOT / "outputs" / "mock_controller_trace.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
