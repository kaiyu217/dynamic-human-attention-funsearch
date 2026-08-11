"""CodeEvolve integration helpers.

Recommended insertion point: after sandboxed evaluation produces fitness/metrics and before the
candidate is selected into the island population / MAP-Elites archive. Keep CodeEvolve as a git
submodule or external dependency rather than copying its source into this repository.
"""

import json
from pathlib import Path
import pandas as pd


class CodeEvolveResultAdapter:
    @staticmethod
    def read_jsonl(path: str | Path) -> pd.DataFrame:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
