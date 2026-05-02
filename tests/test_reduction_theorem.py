"""
Test: Proposition 1 (Reduction to Linear PRS).

If sigma(z) = z (identity activation), the population risk L(a, W)
reduces to:

    L = const + ||Wa - beta*||_Sigma^2

where ||v||_Sigma^2 = v^T Sigma v.

Therefore minimizing over (a, W) yields Wa = beta*, recovering the
standard linear PRS weight estimation from summary statistics.

This test verifies:
1. The loss formula matches ||Wa - beta*||_Sigma^2 + const.
2. The optimizer recovers Wa = beta* (the optimal linear weights).
3. The recovered weights produce the same prediction R^2 as the
   direct linear solution Sigma^{-1} Sigma_beta.
"""

import numpy as np
import pytest

from ssnn.population_risk import compute_loss
from ssnn.optimizer import train
from ssnn.utils import (
    generate_ld_matrix,
    generate_gwas_summary_stats,
    linear_prs_weights,
    prediction_r2,
)


@pytest.fixture
def linear_problem():
    """Problem setup for the reduction theorem test."""
    rng = np.random.default_rng(99)
    p = 8
    Sigma = generate_ld_matrix(p, n_blocks=2, decay=0.5)
    beta_star = rng.standard_normal(p) * 0.3
    sigma_eps = 1.0

    stats = generate_gwas_summary_stats(
        Sigma, beta_star, n=1_000_000, sigma_eps=sigma_eps,
        rng=rng, return_individual_data=True,
    )
    return stats, rng


def test_loss_equals_sigma_norm(linear_problem):
    """Verify L = E[y^2] - beta*^T Sigma beta* + ||Wa - beta*||_Sigma^2."""
    stats, rng = linear_problem
    Sigma = stats["Sigma"]
    beta_star = stats["beta_star"]
    Sigma_beta = stats["Sigma_beta"]
    E_y2 = stats["E_y2"]
    p = len(beta_star)

    m = 4
    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    L = compute_loss(a, W, Sigma, Sigma_beta, E_y2, "identity")

    # Under identity activation:
    # L = E[y^2] - 2 (Wa)^T Sigma beta* + (Wa)^T Sigma (Wa)
    #   = E[y^2] - beta*^T Sigma beta* + (Wa - beta*)^T Sigma (Wa - beta*)
    Wa = W.T @ a
    residual = Wa - beta_star
    sigma_norm_sq = residual @ Sigma @ residual
    irreducible = E_y2 - beta_star @ Sigma @ beta_star

    expected = irreducible + sigma_norm_sq

    assert L == pytest.approx(expected, abs=1e-10)


def test_optimizer_recovers_linear_weights(linear_problem):
    """The identity-activation NN optimizer should recover Wa = beta*."""
    stats, rng = linear_problem
    Sigma = stats["Sigma"]
    Sigma_beta = stats["Sigma_beta"]
    E_y2 = stats["E_y2"]
    beta_star = stats["beta_star"]

    # Use m = p so that W is square and Wa = beta* is exactly achievable
    p = len(beta_star)
    result = train(
        Sigma, Sigma_beta, E_y2,
        m=p,
        activation="identity",
        lr=0.005,
        max_iters=10000,
        tol=1e-12,
        init_scale=0.01,
        rng=np.random.default_rng(42),
    )

    Wa = result.W.T @ result.a
    np.testing.assert_allclose(Wa, beta_star, atol=1e-2)


def test_nn_r2_matches_linear_r2(linear_problem):
    """The optimized identity-NN should give the same R^2 as the direct
    linear solution beta* = Sigma^{-1} Sigma_beta."""
    stats, rng = linear_problem
    Sigma = stats["Sigma"]
    Sigma_beta = stats["Sigma_beta"]
    E_y2 = stats["E_y2"]
    X = stats["X"]
    y = stats["y"]

    beta_linear = linear_prs_weights(Sigma, Sigma_beta)
    r2_linear = prediction_r2(X, y, beta_linear)

    p = Sigma.shape[0]
    result = train(
        Sigma, Sigma_beta, E_y2,
        m=p,
        activation="identity",
        lr=0.005,
        max_iters=10000,
        tol=1e-12,
        init_scale=0.01,
        rng=np.random.default_rng(42),
    )

    Wa = result.W.T @ result.a
    r2_nn = prediction_r2(X, y, Wa)

    assert r2_nn == pytest.approx(r2_linear, abs=1e-2)
