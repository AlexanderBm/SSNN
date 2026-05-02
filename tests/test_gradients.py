"""Verify analytic gradients against finite-difference approximations."""

import numpy as np
import pytest

from ssnn.population_risk import (
    compute_loss,
    compute_grad_a,
    compute_grad_W,
)


def _numerical_grad_a(a, W, Sigma, Sigma_beta, E_y2, activation, eps=1e-6):
    """Finite-difference gradient w.r.t. a."""
    grad = np.zeros_like(a)
    for k in range(len(a)):
        a_plus = a.copy()
        a_plus[k] += eps
        a_minus = a.copy()
        a_minus[k] -= eps
        L_plus = compute_loss(a_plus, W, Sigma, Sigma_beta, E_y2, activation)
        L_minus = compute_loss(a_minus, W, Sigma, Sigma_beta, E_y2, activation)
        grad[k] = (L_plus - L_minus) / (2 * eps)
    return grad


def _numerical_grad_W(a, W, Sigma, Sigma_beta, E_y2, activation, eps=1e-6):
    """Finite-difference gradient w.r.t. W."""
    grad = np.zeros_like(W)
    for k in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[k, j] += eps
            W_minus = W.copy()
            W_minus[k, j] -= eps
            L_plus = compute_loss(a, W_plus, Sigma, Sigma_beta, E_y2, activation)
            L_minus = compute_loss(a, W_minus, Sigma, Sigma_beta, E_y2, activation)
            grad[k, j] = (L_plus - L_minus) / (2 * eps)
    return grad


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_grad_a_vs_finite_diff(small_problem, rng, activation):
    prob = small_problem
    m, p = prob["m"], prob["p"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    analytic = compute_grad_a(a, W, prob["Sigma"], prob["Sigma_beta"], activation)
    numerical = _numerical_grad_a(
        a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation,
    )

    np.testing.assert_allclose(analytic, numerical, atol=1e-5)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_grad_W_vs_finite_diff(small_problem, rng, activation):
    """The W gradient (computed via numerical diff of closed-form loss)
    should match a direct finite-diff of the loss."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], activation)
    numerical = _numerical_grad_W(
        a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation,
    )

    np.testing.assert_allclose(grad_W, numerical, atol=1e-4)


@pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
def test_gradient_descent_direction(small_problem, rng, activation):
    """A small step in the negative gradient direction should decrease the loss."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    L0 = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation)

    grad_a = compute_grad_a(a, W, prob["Sigma"], prob["Sigma_beta"], activation)
    grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], activation)

    lr = 1e-4
    a_new = a - lr * grad_a
    W_new = W - lr * grad_W

    L1 = compute_loss(a_new, W_new, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation)

    assert L1 < L0


# ---------------------------------------------------------------------------
# Gradient edge cases
# ---------------------------------------------------------------------------

def test_grad_a_zero_weights(small_problem):
    """At a=0, W=0, grad_a should point in a direction that decreases loss.
    Specifically, dL/da_k = -2 E[y sigma(w_k^T x)] = 0 when W=0."""
    prob = small_problem
    m, p = prob["m"], prob["p"]
    a = np.zeros(m)
    W = np.zeros((m, p))

    grad = compute_grad_a(a, W, prob["Sigma"], prob["Sigma_beta"], "relu")
    np.testing.assert_allclose(grad, 0.0, atol=1e-14)


def test_grad_W_single_unit(small_problem, rng):
    """Gradient computation should work with m=1."""
    prob = small_problem
    p = prob["p"]

    W = rng.standard_normal((1, p)) * 0.1
    a = rng.standard_normal(1) * 0.1

    grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], "relu")
    numerical = _numerical_grad_W(
        a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu",
    )
    np.testing.assert_allclose(grad_W, numerical, atol=1e-4)


def test_grad_a_not_trivially_zero(small_problem, rng):
    """With nonzero random weights, gradient should generally be nonzero."""
    prob = small_problem
    m, p = prob["m"], prob["p"]

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    grad = compute_grad_a(a, W, prob["Sigma"], prob["Sigma_beta"], "relu")
    assert np.linalg.norm(grad) > 1e-8, "Gradient is suspiciously close to zero"


# ---------------------------------------------------------------------------
# Audit: analytic W-gradient chain rule tests
# ---------------------------------------------------------------------------

class TestAnalyticGradWAudit:
    """Additional tests for the analytic compute_grad_W implementation.

    These verify the chain rule through v_k, c_{kl}, s_k for all
    activations, and test edge cases not covered by the original tests.
    """

    @pytest.mark.parametrize("activation", ["relu", "identity", "sigmoid"])
    def test_grad_W_multiple_seeds(self, small_problem, activation):
        """Validate grad_W vs FD across multiple random initializations."""
        prob = small_problem
        m, p = prob["m"], prob["p"]

        for seed in [10, 20, 30]:
            r = np.random.default_rng(seed)
            W = r.standard_normal((m, p)) * 0.1
            a = r.standard_normal(m) * 0.1

            grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], activation)
            numerical = _numerical_grad_W(
                a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], activation,
            )
            np.testing.assert_allclose(grad_W, numerical, atol=1e-4, err_msg=(
                f"seed={seed}, activation={activation}"
            ))

    def test_grad_W_single_unit_all_activations(self, small_problem):
        """m=1 gradient should match FD for all activations."""
        prob = small_problem
        p = prob["p"]
        r = np.random.default_rng(77)
        W = r.standard_normal((1, p)) * 0.1
        a = r.standard_normal(1) * 0.1

        for act in ["relu", "identity", "sigmoid"]:
            grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], act)
            numerical = _numerical_grad_W(
                a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], act,
            )
            np.testing.assert_allclose(grad_W, numerical, atol=1e-4, err_msg=f"activation={act}")

    def test_grad_W_larger_problem(self):
        """Test with a larger problem (more hidden units, more SNPs)."""
        rng = np.random.default_rng(55)
        p = 15
        m = 5

        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        beta_star = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta_star
        E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)

        W = rng.standard_normal((m, p)) * 0.05
        a = rng.standard_normal(m) * 0.05

        for act in ["relu", "sigmoid"]:
            grad_W = compute_grad_W(a, W, Sigma, Sigma_beta, act)
            numerical = _numerical_grad_W(a, W, Sigma, Sigma_beta, E_y2, act, eps=1e-6)
            np.testing.assert_allclose(grad_W, numerical, atol=1e-3, err_msg=f"activation={act}")

    def test_grad_W_descent_reduces_loss_per_unit(self, small_problem):
        """Verify that updating each w_k independently along its gradient
        direction decreases the loss."""
        prob = small_problem
        m, p = prob["m"], prob["p"]
        rng = np.random.default_rng(42)
        W = rng.standard_normal((m, p)) * 0.1
        a = rng.standard_normal(m) * 0.1

        L0 = compute_loss(a, W, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu")
        grad_W = compute_grad_W(a, W, prob["Sigma"], prob["Sigma_beta"], "relu")

        lr = 1e-5
        W_new = W.copy()
        W_new -= lr * grad_W
        L1 = compute_loss(a, W_new, prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], "relu")
        assert L1 < L0
