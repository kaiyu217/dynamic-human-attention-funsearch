# Research Formulation

## 1. Research question

**How should a fixed human-attention budget be allocated over a finite AI heuristic-discovery run when each current candidate has an automated promise score and uncertainty level, and human review may alter the future search trajectory?**

This is a dynamic extension of static risk-based human oversight. Attention spent at iteration `t` cannot be used later, and review feedback may change future candidate quality.

## 2. Minimal state

$$
S_t=(p_t,u_t,b_t,n_t)
$$

- `p_t in [0,1]`: calibrated probability / normalized promise that candidate `t` yields a meaningful robust improvement.
- `u_t in [0,1]`: epistemic or evaluation uncertainty.
- `b_t in {0,...,B}`: review units remaining.
- `n_t in {1,...,T}`: iterations remaining, including the current iteration.

The minimal model intentionally excludes full population history. This requires a **state-aggregation / Markov sufficiency assumption**: conditional on `(p,u)` and the chosen action/outcome, the distribution of the next candidate is adequately summarized by the transition kernel below. If repeated-run diagnostics reject this assumption, add the incumbent best score or population-quality summary as a fifth state variable.

## 3. Actions

$$
a_t \in \{R,P,H\}
$$

- `R`: automatically reject current candidate;
- `P`: automatically promote current candidate to the evolutionary population/archive;
- `H`: allocate one unit of human attention.

`H` is feasible only if `b_t>0`.

## 4. Human-review outcomes

Conditional on review:

$$
o_t \in \{A,J,V\}
$$

- `A`: approve;
- `J`: reject;
- `V`: targeted revision request.

with

$$
\pi_o(p,u)=P(o_t=o\mid p_t=p,u_t=u,H).
$$

A revision produces a targeted prompt / diagnostic message that is fed back into the search engine.

## 5. Rewards

Let `G_t` be the incremental value of the search after handling candidate `t`. A practical reward is the change in robust held-out incumbent score:

$$
r_t = M_{t+1}-M_t,
$$

where `M_t` is the best validated score before iteration `t`.

For the simplified threshold model, define ex-ante rewards

$$
r_R(p,u), \quad r_P(p,u), \quad r_H(p,u),
$$

with a review-value term that is largest when the candidate is both promising and uncertain. The demo uses

$$
r_H=\max\{r_R,r_P\}+\kappa\,u\,4p(1-p)-c_H.
$$

This is a transparent structural approximation; in the empirical model these reward functions are estimated from repeated runs.

## 6. Transitions

Budget and horizon evolve deterministically:

$$
b_{t+1}=b_t-\mathbf{1}\{a_t=H\},\qquad n_{t+1}=n_t-1.
$$

Candidate features evolve stochastically:

$$
(p_{t+1},u_{t+1})\sim F_{a,o}(\cdot\mid p_t,u_t),
$$

where `o` is relevant only after review. Separate transition kernels allow human approval/rejection/revision to influence future generations.

## 7. Single-step model

For one candidate, introduce `lambda >= 0` as the shadow price of one unit of human attention:

$$
Q_R=r_R,\quad Q_P=r_P,\quad Q_H=r_H-\lambda.
$$

Choose the action with maximum value. The one-step model isolates the **value of information (VOI)** from dynamic opportunity cost.

If `r_P-r_R` is increasing in `p` and the incremental review value

$$
VOI(p,u)=r_H-\max\{r_R,r_P\}
$$

is single-peaked in `p` and nondecreasing in `u`, then there are lower and upper promise thresholds:

$$
p_L(u,\lambda)\le p_U(u,\lambda),
$$

such that weak candidates are rejected, strong candidates are promoted, and sufficiently uncertain intermediate candidates are reviewed.

## 8. Finite-horizon DP

Let `V_n(p,u,b)` denote optimal expected future reward with `n` iterations and `b` review units remaining.

$$
Q_R=r_R+\mathbb E_{F_R}[V_{n-1}],
$$

$$
Q_P=r_P+\mathbb E_{F_P}[V_{n-1}],
$$

and, for `b>0`,

$$
Q_H=r_H+\sum_o \pi_o(p,u)\,\mathbb E_{F_{H,o}}[V_{n-1}(p',u',b-1)].
$$

Then

$$
V_n(p,u,b)=\max\{Q_R,Q_P,Q_H\}.
$$

The endogenous shadow value of one unit of attention is approximately

$$
\lambda_{b,n}(p,u)=V_n(p,u,b)-V_n(p,u,b-1).
$$

This is the dynamic opportunity cost absent from the original static allocation model.

## 9. Assumptions sufficient for interpretable threshold structure

1. **Calibrated promise:** larger `p` implies stochastically larger realized candidate value.
2. **Single crossing:** `r_P-r_R` increases in `p`.
3. **Uncertainty raises review value:** review's incremental value is nondecreasing in `u`.
4. **Intermediate-candidate VOI:** review advantage is single-peaked / quasi-concave in `p`.
5. **Monotone transitions:** higher-promise states lead to stochastically higher-promise future states under each action/outcome.
6. **Unit attention cost:** each review consumes one identical budget unit.
7. **State aggregation sufficiency:** `(p,u,b,n)` is approximately Markov for the initial model.

Under these assumptions, a reject-review-promote threshold policy is expected. Comparative statics in remaining horizon are more delicate: with exogenous future arrivals, scarce attention is usually saved when many iterations remain; with strong revision spillovers, early review can instead become more valuable. This interaction is an empirical/theoretical target rather than an assumption.
