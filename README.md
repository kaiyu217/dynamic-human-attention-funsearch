# Dynamic Human Attention for FunSearch-Style Heuristic Discovery

This repository implements a research pipeline for **dynamically allocating a fixed human-attention budget across multiple iterations of AI-driven heuristic discovery**.

At each iteration an evolutionary coding system (FunSearch, CodeEvolve, AlphaEvolve, or another compatible backend) proposes and automatically evaluates a candidate heuristic. A decision policy then chooses one of three actions:

- **REJECT**: discard a clearly weak candidate and continue autonomously;
- **PROMOTE**: accept a clearly strong candidate into the search population/archive without human review;
- **REVIEW**: spend one unit of scarce human attention. Review can lead to **approve**, **reject**, or **revise**; targeted revision feedback can affect future candidate generation.

The research contribution is the **human-attention policy**, not a particular search engine.

## Core state and decision model

At iteration `t`, the minimal state is

$$
S_t=(p_t,u_t,b_t,n_t),
$$

where:

- `p_t` = calibrated automated **promise** of the current candidate;
- `u_t` = automated **uncertainty** about the candidate/evaluation;
- `b_t` = remaining human-review budget;
- `n_t` = remaining search iterations.

The finite-horizon Bellman equation is

$$
V_n(p,u,b)=\max\{Q_R,Q_P,Q_H\},
$$

with `R=reject`, `P=promote`, and `H=human review`. Review is infeasible when `b=0`.

The implementation includes:

1. a one-step value-of-information policy;
2. a finite-horizon dynamic program;
3. threshold extraction;
4. AI-only, periodic-review, score-review, and uncertainty-review baselines;
5. a simulated reviewer and search environment for fully reproducible tests;
6. estimators for promise, uncertainty, review outcomes, transition kernels, and rewards from repeated search logs;
7. backend interfaces/adapters for FunSearch-style systems;
8. experiment scripts and plots.

## Why threshold policies are expected

Under standard single-crossing / monotonicity assumptions:

- very low promise -> **reject**;
- very high promise -> **promote**;
- intermediate promise with sufficiently high uncertainty -> **review**.

The review region expands when uncertainty raises the value of information. With a fixed budget, the finite-horizon DP endogenously prices attention through its opportunity cost.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python scripts/run_demo.py
pytest -q
```

The demo writes:

- `outputs/policy_table.csv`
- `outputs/thresholds.csv`
- `outputs/simulation_summary.csv`
- `outputs/policy_map.png`

## Real evolutionary-search backends

### Original FunSearch

The official FunSearch repository contains the evolutionary algorithm and a single-threaded pipeline, but does **not** provide the LLM, sandbox, or distributed execution infrastructure. Use `FunSearchLogAdapter` to ingest candidate/evaluator traces, or wire your own generator into the `SearchBackend` interface.

### CodeEvolve (recommended open/reproducible backend)

CodeEvolve is a fully open-source AlphaEvolve-style evolutionary coding framework with sandboxed evaluation, checkpoints, multiple OpenAI-compatible LLM backends, mock models, and candidate databases. The human-attention controller can be inserted between candidate evaluation and population/archive update.

### AlphaEvolve on Google Cloud

The public Google Cloud client library exposes the seed program, evaluator, controller loop, candidate metrics, and candidate insights. The same human-attention controller can wrap candidate handling, but a provisioned Gemini Enterprise / AlphaEvolve service is required.

## Suggested empirical study

Use repeated independent runs and fixed random seeds. Compare:

- AI-only search;
- periodic review every `k` iterations;
- simple promise-score review;
- uncertainty-only review;
- one-step/myopic value-of-information policy;
- finite-horizon dynamic policy.

Primary outcomes:

- final best held-out heuristic score;
- probability of discovering an improvement over the seed/baseline;
- human reviews used;
- improvement per review;
- regret versus an oracle policy;
- time-to-best candidate;
- search diversity / population quality (optional).

See `docs/formulation.md` and `docs/estimation.md` for the research formulation and estimation plan.
