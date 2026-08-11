"""Adapter notes for google-deepmind/funsearch.

The official repository exposes the evolutionary algorithm and program database but intentionally
omits the production LLM and sandbox. The clean insertion point for this project is after a generated
program has been evaluated and before it is permanently admitted / used to influence future prompts.

For a real integration, map FunSearch's evaluated-program record into `Candidate`, compute
promise/uncertainty from evaluator metrics, query the attention policy, then either:
  - reject: do not admit/use the candidate;
  - promote: admit normally;
  - review: call the reviewer, then approve/reject or add targeted feedback to the next prompt.

This module intentionally does not vendor or modify DeepMind's code.
"""

import json
from pathlib import Path
import pandas as pd


class FunSearchLogAdapter:
    @staticmethod
    def read_jsonl(path: str | Path) -> pd.DataFrame:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
