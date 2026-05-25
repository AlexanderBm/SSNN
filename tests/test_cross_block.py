"""Tests for cross-block epistasis: stats, closed-form solver, two-stage trainer."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from ssnn.cross_block_stats import (
    compute_cross_block_scalars,
    triage_cross_pairs,
)
from ssnn.cross_block_risk import (
    compute_cross_block_loss,
    solve_cross_block_coefficients,
)
from ssnn.cross_block_optimizer import train_two_stage, predict_two_stage
from ssnn.utils import nn_prediction_r2


def _make_block_genotypes(B, p_b, n, rng):
    return [rng.standard_normal((n, p_b)) for _ in range(B)]


def test_cross_block_scalars_monte_carlo():
    """Monte Carlo check: q_ab matches the planted analytical expectation.

    Genotypes are standard Gaussian (Sigma = I) within each block.  We
    plant a single cross-block interaction y += gamma * x_i^A * x_j^B
    using ridge-like weights beta_hat_a = e_i and beta_hat_b = e_j.
    Then

        E[q_ab] = gamma * Var(x_i^A) * Var(x_j^B) = gamma

    when blocks A, B are independent standard normal.  Linear additive
    components have zero expectation in this cross moment because the
    third Gaussian moment vanishes.
    """
    rng = np.random.default_rng(0)
    B = 2
    p_b = 4
    n = 50000
    gamma = 0.7

    X_blocks = _make_block_genotypes(B, p_b, n, rng)
    i_a, j_b = 1, 2
    additive = rng.standard_normal(p_b) * 0.2
    y = (
        X_blocks[0] @ additive
        + X_blocks[1] @ additive
        + gamma * X_blocks[0][:, i_a] * X_blocks[1][:, j_b]
        + rng.standard_normal(n) * 0.3
    )
    y = y - y.mean()

    beta_hats = [np.zeros(p_b), np.zeros(p_b)]
    beta_hats[0][i_a] = 1.0
    beta_hats[1][j_b] = 1.0

    q = compute_cross_block_scalars(X_blocks, y, beta_hats)

    # Standard error: q_ab is mean of n products with variance ~ E[(f_a f_b y)^2].
    f_a = X_blocks[0] @ beta_hats[0]
    f_b = X_blocks[1] @ beta_hats[1]
    sample_var = np.var(f_a * f_b * y)
    sem = np.sqrt(sample_var / n)

    assert abs(q[0, 1] - gamma) < 3.0 * sem + 0.02


def test_triage_detects_planted_pairs():
    """B=5 setup: only the (0,1) and (2,3) planted pairs should fire.

    Interactions are planted along block-specific direction vectors w_b
    and the triage step is provided beta_hat = w_b directly so the
    projection captures the planted signal cleanly.  This isolates the
    statistical detection question from the orthogonal question of
    estimating beta_hat from data.
    """
    rng = np.random.default_rng(7)
    B = 5
    p_b = 6
    n = 80000
    gamma = 0.8

    X_blocks = _make_block_genotypes(B, p_b, n, rng)

    planted = [(0, 1), (2, 3)]
    w_dirs = [rng.standard_normal(p_b) / np.sqrt(p_b) for _ in range(B)]

    additive = [rng.standard_normal(p_b) * 0.15 for _ in range(B)]
    y = sum(X_blocks[b] @ additive[b] for b in range(B))
    for (a, b) in planted:
        y = y + gamma * (X_blocks[a] @ w_dirs[a]) * (X_blocks[b] @ w_dirs[b])
    y = y + rng.standard_normal(n) * 0.5
    y = y - y.mean()

    beta_hats = w_dirs

    q = compute_cross_block_scalars(X_blocks, y, beta_hats)

    f_blocks = [X_blocks[b] @ beta_hats[b] for b in range(B)]
    per_block_Ef2 = [float(np.mean(f ** 2)) for f in f_blocks]
    E_y2 = float(np.mean(y ** 2))

    sig = triage_cross_pairs(q, per_block_Ef2, E_y2, n, threshold=4.0)

    for pair in planted:
        assert pair in sig, f"missed planted pair {pair} (q matrix: {q})"

    extras = [pair for pair in sig if pair not in planted]
    assert extras == [], f"false positives: {extras}"


def test_closed_form_matches_gradient_descent():
    """Closed-form c_ab* must match scipy.optimize.minimize to 4 decimals."""
    rng = np.random.default_rng(123)
    B = 4
    per_block_Ef2 = list(rng.uniform(0.3, 0.8, size=B))
    E_y2 = 1.7
    per_block_Eyf = list(rng.uniform(0.05, 0.3, size=B))

    q = np.zeros((B, B))
    for a in range(B):
        for b in range(a + 1, B):
            q[a, b] = rng.uniform(-0.2, 0.2)
            q[b, a] = q[a, b]

    significant_pairs = [(0, 1), (1, 2), (0, 3)]

    c_star = solve_cross_block_coefficients(per_block_Ef2, q, significant_pairs)

    def to_dict(vec):
        return {pair: float(vec[i]) for i, pair in enumerate(significant_pairs)}

    def loss_vec(vec):
        return compute_cross_block_loss(
            to_dict(vec), per_block_Ef2, q, E_y2, per_block_Eyf, significant_pairs,
        )

    x0 = np.zeros(len(significant_pairs))
    res = minimize(loss_vec, x0, method="BFGS", options={"gtol": 1e-10})

    for i, pair in enumerate(significant_pairs):
        assert abs(c_star[pair] - res.x[i]) < 1e-4, (
            f"pair {pair}: closed-form {c_star[pair]} vs numeric {res.x[i]}"
        )


def test_two_stage_never_hurts():
    """Population-level: with no cross-block signal, c_ab = 0 minimizes loss.

    The decoupled Stage-2 quadratic L_2(c) = const - 2 c_ab q_ab +
    c_ab^2 E[f_a^2] E[f_b^2] is convex with minimum at c_ab* =
    q_ab / (E[f_a^2] E[f_b^2]).  When the population q_ab is 0 (no
    cross-block interaction), the optimum is c_ab* = 0 and L_2 reduces
    exactly to the Stage-1 loss.

    Operationally we check three things:
    (i) the closed-form solver returns 0 for all pairs when q_ab = 0;
    (ii) computed loss with empty c_dict equals computed loss with
        explicit zeros;
    (iii) on a small synthetic dataset, even if triage falsely fires,
        the Two-Stage test R^2 stays close to the additive R^2 (within
        an empirical tolerance reflecting finite-sample noise in c_ab).
    """
    B = 4

    q_zero = np.zeros((B, B))
    Ef2 = [0.3, 0.5, 0.7, 0.4]
    pairs = [(0, 1), (1, 2), (0, 3)]

    c_zero = solve_cross_block_coefficients(Ef2, q_zero, pairs)
    for pair in pairs:
        assert c_zero[pair] == 0.0

    Eyf = [0.1, 0.15, 0.2, 0.12]
    E_y2 = 1.5
    L_empty = compute_cross_block_loss({}, Ef2, q_zero, E_y2, Eyf, [])
    L_zeros = compute_cross_block_loss(c_zero, Ef2, q_zero, E_y2, Eyf, pairs)
    assert abs(L_empty - L_zeros) < 1e-12

    rng = np.random.default_rng(202)
    B = 4
    p_b = 6
    n_train = 8000
    n_test = 4000
    n_ref = 2000

    Sigma_blocks = [np.eye(p_b) for _ in range(B)]
    beta_star_blocks = [rng.standard_normal(p_b) * 0.25 for _ in range(B)]

    X_train_blocks = _make_block_genotypes(B, p_b, n_train, rng)
    X_test_blocks = _make_block_genotypes(B, p_b, n_test, rng)
    X_ref_blocks = _make_block_genotypes(B, p_b, n_ref, rng)

    def phen(X_blocks, noise_rng):
        y = sum(X_blocks[b] @ beta_star_blocks[b] for b in range(B))
        y = y + noise_rng.standard_normal(len(X_blocks[0])) * 1.0
        return y - y.mean()

    y_train = phen(X_train_blocks, rng)
    y_test = phen(X_test_blocks, rng)

    Sigma_beta_blocks = []
    beta_hats = []
    for b in range(B):
        Sigma_beta_blocks.append(X_train_blocks[b].T @ y_train / n_train)
        beta_hats.append(np.linalg.solve(
            X_train_blocks[b].T @ X_train_blocks[b] / n_train + 1e-2 * np.eye(p_b),
            Sigma_beta_blocks[b],
        ))

    E_y2 = float(np.mean(y_train ** 2))

    per_block_results, c_dict, _ = train_two_stage(
        Sigma_blocks=Sigma_blocks,
        Sigma_beta_blocks=Sigma_beta_blocks,
        E_y2=E_y2,
        Gamma_blocks=None,
        X_ref_blocks=X_ref_blocks,
        X_train_blocks=X_train_blocks,
        y_train=y_train,
        beta_hats=beta_hats,
        n_train=n_train,
        m=3,
        activation="relu",
        lr=0.05,
        max_iters=200,
        rng=np.random.default_rng(5),
    )

    pred_additive = predict_two_stage(X_test_blocks, per_block_results, {}, "relu")
    pred_two = predict_two_stage(X_test_blocks, per_block_results, c_dict, "relu")

    ss_tot = np.var(y_test)
    r2_add = 1.0 - np.mean((y_test - pred_additive) ** 2) / ss_tot
    r2_two = 1.0 - np.mean((y_test - pred_two) ** 2) / ss_tot

    assert r2_two >= r2_add - 0.10, (
        f"Two-stage R^2 ({r2_two:.4f}) much worse than additive R^2 ({r2_add:.4f}); "
        f"c_dict has {len(c_dict)} entries"
    )
