# Estimation and Evaluation from Repeated Search Runs

Each repeated FunSearch/CodeEvolve/AlphaEvolve run should log one row per evaluated candidate.

Recommended columns:

`run_id, seed, t, candidate_id, parent_id, raw_score, heldout_score, incumbent_before, promise_raw, promise, uncertainty, action, review_outcome, revision_prompt, reward, next_promise, next_uncertainty, final_best_score`

## Promise `p_t`

Interpret promise as

$$
p_t=P(\Delta_t>\epsilon\mid z_t),
$$

where `Delta_t` is robust held-out improvement over the incumbent and `z_t` contains automatically available candidate features.

Practical estimators:

1. Start with evaluator score margin over incumbent.
2. Add robustness features: fraction of benchmark instances improved, validity rate, constraint violations, runtime.
3. Calibrate the raw score using out-of-run or cross-run empirical bins / isotonic or logistic calibration.
4. Use strict train/calibration/test run splits to avoid learning the policy on evaluation runs.

`src/hai_funsearch/estimation/empirical.py` includes a dependency-light bin calibrator.

## Uncertainty `u_t`

Possible sources, all available without a human study:

- bootstrap standard error across benchmark instances;
- variance across repeated evaluator seeds;
- score instability under small perturbations;
- disagreement across evaluator variants;
- distance from previously observed candidate regions.

Normalize the chosen uncertainty statistic to `[0,1]`.

## Human-review outcome probabilities

Estimate

$$
\pi_o(p,u)=P(o\mid p,u,H)
$$

from reviewed candidates. For a first normative simulation with no human subjects, define a reproducible **oracle reviewer** using a hidden validation set:

- approve: robustly improves and passes diagnostics;
- reject: fails robust improvement / validity requirements;
- revise: has positive promise but a localized diagnostic failure that can be expressed as targeted feedback.

This supports policy development but must be described as a **simulated expert / oracle**, not empirical human behavior.

## Transition kernels

Estimate

$$
F_{a,o}(p',u'\mid p,u)
$$

by discretizing `(p,u)` and estimating next-state frequencies conditional on action and review outcome.

Causal identification needs action coverage. Recommended initial design:

- collect autonomous runs for `R/P` transitions;
- use a randomized or epsilon-greedy pilot policy for a subset of runs so review occurs across the state space;
- pair runs with common random seeds when possible;
- use separate runs to estimate revision spillovers.

The provided empirical transition estimator uses binned counts with Laplace smoothing.

## Rewards

Preferred empirical reward:

$$
r_t=M_{t+1}-M_t,
$$

or, for longer-term credit,

$$
r_t=E[M_T\mid a_t]-E[M_T\mid \text{counterfactual baseline}].
$$

For the first paper, incremental incumbent improvement is easier to identify and interpret. Final-score comparisons across policies remain the primary experimental endpoint.

## Evaluation design

Use at least 30 independent seeds per policy when computationally feasible. Report mean, median, confidence intervals, and paired differences under common seeds.

Baselines:

1. AI-only search.
2. Periodic review every `k` iterations.
3. Promise-score review (review candidates in a fixed score band or top score threshold until budget is exhausted).
4. Uncertainty-only review.
5. One-step VOI policy.
6. Finite-horizon DP.

Outcomes:

- final best held-out score;
- probability of beating the seed / baseline heuristic;
- reviews used;
- gain per review;
- time-to-best;
- policy regret vs. an oracle using realized future outcomes;
- optionally, archive diversity and robustness.
