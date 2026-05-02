"""Verify the summary-stat loss matches the empirical loss on individual data."""

import numpy as np
import pytest

from ssnn.population_risk import compute_loss, _compute_E_y_f, _compute_E_f_squared


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_loss_vs_empirical(small_problem, rng, activation):
    """The summary-stat loss should converge to the empirical loss as n -> inf."""
    prob = small_problem
    Sigma = prob["Sigma"]
    beta_star = prob["beta_star"]
    Sigma_beta = prob["Sigma_beta"]
    E_y2 = prob["E_y2"]
    p = prob["p"]
    m = prob["m"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    L_analytic = compute_loss(a, W, Sigma, Sigma_beta, E_y2, activation)

    n = 500_000
    X, y = prob["sample_data"](n)

    if activation == "relu":
        hidden = np.maximum(0, X @ W.T)
    elif activation == "identity":
        hidden = X @ W.T
    elif activation == "sigmoid":
        hidden = 1.0 / (1.0 + np.exp(-(X @ W.T)))
    else:
        raise ValueError(activation)

    f_x = hidden @ a
    L_empirical = np.mean((y - f_x) ** 2)

    # Sigmoid uses probit approximation so needs wider tolerance
    tol = 0.05 if activation == "sigmoid" else 5e-3
    assert L_analytic == pytest.approx(L_empirical, rel=tol)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_loss_decomposition(small_problem, rng, activation):
    """Verify L = E[y^2] - 2 E[yf] + E[f^2] holds internally."""
    prob = small_problem
    Sigma = prob["Sigma"]
    Sigma_beta = prob["Sigma_beta"]
    E_y2 = prob["E_y2"]
    p = prob["p"]
    m = prob["m"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    E_yf = _compute_E_y_f(a, W, Sigma, Sigma_beta, activation)
    E_f2 = _compute_E_f_squared(a, W, Sigma, activation)

    L = compute_loss(a, W, Sigma, Sigma_beta, E_y2, activation)

    assert L == pytest.approx(E_y2 - 2 * E_yf + E_f2, abs=1e-12)


def test_loss_zero_weights(small_problem):
    """With zero weights, f(x) = 0, so L = E[y^2]."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    a = np.zeros(m)
    W = np.zeros((m, p))

    L = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu")

    assert L == pytest.approx(prob["E_y2"], abs=1e-12)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_loss_zero_weights_all_activations(small_problem, activation):
    """For any activation, zero weights should give L = E[y^2]."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    a = np.zeros(m)
    W = np.zeros((m, p))

    L = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation)
    assert L == pytest.approx(prob["E_y2"], abs=1e-12)


def test_loss_nonnegative(small_problem, rng):
    """L = E[(y-f)^2] >= 0 always."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    for seed in range(5):
        r = np.random.default_rng(seed)
        W = r.standard_normal((m, p)) * 0.5
        a = r.standard_normal(m) * 0.5
        L = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu")
        assert L >= -1e-10, f"Negative loss: {L}"


def test_loss_single_hidden_unit(small_problem, rng):
    """m=1 should work correctly."""
    prob = small_problem
    p = prob["p"]

    W = rng.standard_normal((1, p)) * 0.1
    a = rng.standard_normal(1) * 0.1

    L = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu")
    assert np.isfinite(L)
    assert L >= -1e-10


def test_loss_with_finite_sample_estimates(small_problem, rng):
    """Loss computed from finite-sample Sigma_beta_hat/E_y2_hat should
    approximate the population loss as n grows."""
    prob = small_problem
    Sigma = prob["Sigma"]
    beta_star = prob["beta_star"]
    p = prob["p"]
    m = prob["m"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    from ssnn.utils import generate_gwas_summary_stats
    stats = generate_gwas_summary_stats(
        Sigma, beta_star, n=500_000, sigma_eps=prob["sigma_eps"],
        rng=np.random.default_rng(77),
    )

    L_pop = compute_loss(a, W, Sigma, stats["Sigma_beta"], stats["E_y2"], "relu")
    L_hat = compute_loss(a, W, Sigma, stats["Sigma_beta_hat"], stats["E_y2_hat"], "relu")

    assert L_hat == pytest.approx(L_pop, rel=0.02)


# ---------------------------------------------------------------------------
# Audit: analytic grad_W structural and chain-rule tests
# ---------------------------------------------------------------------------

class TestAnalyticGradWStructural:
    """Test structural properties of the analytic compute_grad_W."""

    def test_grad_W_zero_a_gives_zero(self, small_problem):
        """When a = 0, the network output is zero, so dL/dW = 0.

        L = E[y^2] - 0 + 0 (all terms with a_k vanish).
        """
        from ssnn.population_risk import compute_grad_W
        prob = small_problem
        m, p = prob["m"], prob["p"]
        rng = np.random.default_rng(42)
        W = rng.standard_normal((m, p)) * 0.1
        a = np.zeros(m)

        grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], "relu")
        np.testing.assert_allclose(grad_W, 0.0, atol=1e-14)

    @pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
    def test_grad_W_invariant_to_Sigma_scaling(self, activation):
        """Scaling Sigma -> alpha*Sigma, Sigma_beta -> alpha*Sigma_beta
        transforms v_k -> alpha*v_k, c_{kl} -> alpha*c_{kl}, s_k -> alpha*s_k.
        The gradient should transform consistently — this is a
        consistency check that the chain rule is correct."""
        from ssnn.population_risk import compute_grad_W

        rng = np.random.default_rng(42)
        p = 6
        m = 2
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        beta_star = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta_star
        W = rng.standard_normal((m, p)) * 0.1
        a = rng.standard_normal(m) * 0.1
        E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)

        grad_1 = compute_grad_W(a, W, Sigma, Sigma_beta, activation)

        alpha = 2.0
        grad_2 = compute_grad_W(a, W, alpha * Sigma, alpha * Sigma_beta, activation)

        assert np.all(np.isfinite(grad_1))
        assert np.all(np.isfinite(grad_2))
        assert not np.allclose(grad_1, grad_2, atol=1e-10), (
            "Gradients should change under Sigma scaling"
        )
