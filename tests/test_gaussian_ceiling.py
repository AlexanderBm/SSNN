"""
Test: Proposition 2 (Gaussian Ceiling).

Under x ~ N(0, Sigma) and y = beta*^T x + eps with eps independent of x,
the Bayes-optimal predictor is E[y|x] = beta*^T x, which is linear.
Therefore the global minimum of L(a, W) over all (a, W) achieves exactly
the same prediction error as the optimal linear model.

This test trains a ReLU network on summary stats and verifies:
1. The NN's final loss equals the irreducible error sigma_eps^2.
2. The NN's prediction R^2 on held-out data matches the linear R^2.
3. The NN cannot beat the linear model (R^2_NN <= R^2_linear + tolerance).
"""

import numpy as np
import pytest

from ssnn.optimizer import train
from ssnn.utils import (
    generate_ld_matrix,
    generate_gwas_summary_stats,
    linear_prs_weights,
    prediction_r2,
    nn_prediction_r2,
)


@pytest.fixture
def ceiling_problem():
    """Moderate-size problem for the Gaussian ceiling test.

    Uses p=6, m=6 so the optimizer converges in reasonable time.
    """
    rng = np.random.default_rng(77)
    p = 6

    Sigma = generate_ld_matrix(p, n_blocks=2, decay=0.4)
    beta_star = rng.standard_normal(p) * 0.5
    sigma_eps = 1.0

    stats = generate_gwas_summary_stats(
        Sigma, beta_star, n=500_000, sigma_eps=sigma_eps,
        rng=rng, return_individual_data=True,
    )

    # Separate held-out data for evaluation
    rng_test = np.random.default_rng(999)
    X_test = rng_test.multivariate_normal(np.zeros(p), Sigma, size=100_000)
    y_test = X_test @ beta_star + rng_test.normal(0, sigma_eps, size=100_000)

    return stats, X_test, y_test


def test_nn_loss_reaches_irreducible_error(ceiling_problem):
    """The ReLU NN's optimized loss should approach sigma_eps^2."""
    stats, _, _ = ceiling_problem
    sigma_eps = stats["sigma_eps"]

    result = train(
        stats["Sigma"], stats["Sigma_beta"], stats["E_y2"],
        m=6,
        activation="relu",
        lr=0.005,
        max_iters=5000,
        tol=1e-10,
        init_scale=0.01,
        rng=np.random.default_rng(123),
    )

    # Irreducible error = sigma_eps^2 (variance of epsilon)
    irreducible = sigma_eps ** 2

    # The final loss should be close to the irreducible error
    # (allowing some tolerance for optimizer convergence)
    assert result.loss_history[-1] == pytest.approx(irreducible, rel=0.05)


def test_nn_r2_matches_linear_r2(ceiling_problem):
    """The NN prediction R^2 should match the linear R^2 on held-out data."""
    stats, X_test, y_test = ceiling_problem

    # Linear baseline
    beta_linear = linear_prs_weights(stats["Sigma"], stats["Sigma_beta"])
    r2_linear = prediction_r2(X_test, y_test, beta_linear)

    # Train ReLU network
    result = train(
        stats["Sigma"], stats["Sigma_beta"], stats["E_y2"],
        m=6,
        activation="relu",
        lr=0.005,
        max_iters=5000,
        tol=1e-10,
        init_scale=0.01,
        rng=np.random.default_rng(123),
    )

    r2_nn = nn_prediction_r2(X_test, y_test, result.a, result.W, "relu")

    # The NN R^2 should be close to (and not exceed) the linear R^2
    assert r2_nn == pytest.approx(r2_linear, abs=0.02)


def test_nn_cannot_beat_linear(ceiling_problem):
    """Run the NN from multiple initializations -- none should beat linear."""
    stats, X_test, y_test = ceiling_problem

    beta_linear = linear_prs_weights(stats["Sigma"], stats["Sigma_beta"])
    r2_linear = prediction_r2(X_test, y_test, beta_linear)

    for seed in [1, 2, 3]:
        result = train(
            stats["Sigma"], stats["Sigma_beta"], stats["E_y2"],
            m=6,
            activation="relu",
            lr=0.005,
            max_iters=3000,
            tol=1e-10,
            init_scale=0.01,
            rng=np.random.default_rng(seed),
        )

        r2_nn = nn_prediction_r2(X_test, y_test, result.a, result.W, "relu")

        # NN should not meaningfully exceed linear R^2
        assert r2_nn <= r2_linear + 0.02, (
            f"Seed {seed}: NN R^2 = {r2_nn:.4f} > linear R^2 = {r2_linear:.4f}"
        )
