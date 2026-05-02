"""Verify Stein's lemma and Gaussian integral machinery against Monte Carlo."""

import numpy as np
import pytest

from ssnn.gaussian_integrals import (
    projection_variance,
    pairwise_covariance,
    stein_cross_moment,
    activation_cross_moment,
)


@pytest.fixture
def setup(rng):
    p = 8
    A = rng.standard_normal((p, p))
    Sigma = A @ A.T / p + np.eye(p) * 0.1

    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star

    w_k = rng.standard_normal(p) * 0.1
    w_l = rng.standard_normal(p) * 0.1

    return Sigma, beta_star, Sigma_beta, w_k, w_l


def test_projection_variance(setup, rng):
    Sigma, _, _, w_k, _ = setup
    v = projection_variance(Sigma, w_k)

    X = rng.multivariate_normal(np.zeros(len(w_k)), Sigma, size=500_000)
    mc = np.var(X @ w_k)
    assert v == pytest.approx(mc, rel=5e-3)


def test_pairwise_covariance_symmetry(setup):
    Sigma, _, _, w_k, w_l = setup
    C = pairwise_covariance(Sigma, w_k, w_l)

    assert C.shape == (2, 2)
    assert C[0, 1] == pytest.approx(C[1, 0], abs=1e-14)
    assert C[0, 0] > 0
    assert C[1, 1] > 0


def test_pairwise_covariance_vs_mc(setup, rng):
    Sigma, _, _, w_k, w_l = setup
    C = pairwise_covariance(Sigma, w_k, w_l)

    X = rng.multivariate_normal(np.zeros(len(w_k)), Sigma, size=500_000)
    z_k = X @ w_k
    z_l = X @ w_l
    mc_cov = np.cov(z_k, z_l)

    np.testing.assert_allclose(C, mc_cov, atol=5e-3)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_stein_cross_moment_vs_mc(setup, rng, activation):
    """Compare E[y sigma(w_k^T x)] from Stein's lemma to Monte Carlo."""
    Sigma, beta_star, Sigma_beta, w_k, _ = setup
    p = len(beta_star)

    analytic = stein_cross_moment(Sigma, w_k, Sigma_beta, activation)

    n = 1_000_000
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    y = X @ beta_star + rng.normal(0, 1, size=n)
    z_k = X @ w_k

    if activation == "relu":
        sigma_vals = np.maximum(0, z_k)
    elif activation == "sigmoid":
        sigma_vals = 1.0 / (1.0 + np.exp(-z_k))
    else:
        sigma_vals = z_k

    mc = np.mean(y * sigma_vals)

    # Sigmoid uses probit approximation, so wider tolerance
    tol = 5e-2 if activation == "sigmoid" else 5e-3
    assert analytic == pytest.approx(mc, abs=tol)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_activation_cross_moment_vs_mc(setup, rng, activation):
    """Compare E[sigma(z_k) sigma(z_l)] from closed form to Monte Carlo."""
    Sigma, _, _, w_k, w_l = setup
    p = len(w_k)

    analytic = activation_cross_moment(Sigma, w_k, w_l, activation)

    n = 1_000_000
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    z_k = X @ w_k
    z_l = X @ w_l

    if activation == "relu":
        mc = np.mean(np.maximum(0, z_k) * np.maximum(0, z_l))
    elif activation == "sigmoid":
        mc = np.mean(
            (1.0 / (1.0 + np.exp(-z_k))) * (1.0 / (1.0 + np.exp(-z_l)))
        )
    else:
        mc = np.mean(z_k * z_l)

    tol = 5e-3 if activation != "sigmoid" else 5e-3
    assert analytic == pytest.approx(mc, abs=tol)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_projection_variance_zero_weight(setup):
    """Zero weight vector should give zero projection variance."""
    Sigma, _, _, _, _ = setup
    p = Sigma.shape[0]
    assert projection_variance(Sigma, np.zeros(p)) == pytest.approx(0.0, abs=1e-15)


def test_pairwise_covariance_same_vector(setup):
    """C(w, w) diagonal entries should equal, off-diag should equal diag."""
    Sigma, _, _, w_k, _ = setup
    C = pairwise_covariance(Sigma, w_k, w_k)
    assert C[0, 0] == pytest.approx(C[1, 1], abs=1e-14)
    assert C[0, 1] == pytest.approx(C[0, 0], abs=1e-14)


def test_stein_cross_moment_zero_weight(setup):
    """Zero weight should give zero cross moment for any activation."""
    Sigma, _, Sigma_beta, _, _ = setup
    p = Sigma.shape[0]
    w_zero = np.zeros(p)
    for act in ["relu", "identity"]:
        val = stein_cross_moment(Sigma, w_zero, Sigma_beta, act)
        assert val == pytest.approx(0.0, abs=1e-14)
