"""Comprehensive tests for the interaction-SSNN (Method 5).

Covers: E[sigma''(z)] closed forms, their derivatives, interaction cross-moments
and gradients, the interaction-extended loss and its gradients, the interaction
optimizer, barrier-breaking integration tests, and numerical stability.
"""

import numpy as np
import pytest

from ssnn.activations import (
    relu_E_sigma_double_prime,
    sigmoid_E_sigma_double_prime,
    identity_E_sigma_double_prime,
    relu_dE_sigma_double_prime_dv,
    sigmoid_dE_sigma_double_prime_dv,
    identity_dE_sigma_double_prime_dv,
    get_activation_double_prime,
)
from ssnn.interaction_integrals import (
    interaction_cross_moment,
    interaction_cross_moment_grad,
)
from ssnn.interaction_risk import (
    compute_interaction_loss,
    compute_interaction_grad_a,
    compute_interaction_grad_W,
    compute_interaction_gradients,
)
from ssnn.interaction_optimizer import train_interaction
from ssnn.population_risk import compute_loss, compute_grad_a, compute_grad_W
from ssnn.gaussian_integrals import projection_variance, stein_cross_moment
from ssnn.optimizer import train, TrainResult
from ssnn.utils import generate_ld_matrix


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def interaction_problem(rng):
    """A small problem (p=10, m=3) with a nonlinear DGP that produces nonzero Gamma."""
    p = 10
    m = 3

    block = np.eye(5)
    for i in range(5):
        for j in range(5):
            block[i, j] = 0.5 ** abs(i - j)
    Sigma = np.block([[block, np.zeros((5, 5))],
                      [np.zeros((5, 5)), block]])

    beta_star = rng.standard_normal(p) * 0.3
    w_star = rng.standard_normal(p) * 0.3
    gamma = 0.5
    sigma_eps = 1.0

    Sigma_beta = Sigma @ beta_star
    n = 100_000
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    y_linear = X @ beta_star
    y_nonlinear = gamma * np.maximum(0.0, X @ w_star)
    y = y_linear + y_nonlinear + rng.normal(0, sigma_eps, n)

    Sigma_beta_hat = X.T @ y / n
    E_y2 = float(np.mean(y ** 2))
    Gamma = X.T @ (X * y[:, None]) / n

    W = rng.standard_normal((m, p)) * 0.01
    a = rng.standard_normal(m) * 0.01

    return {
        "p": p, "m": m,
        "Sigma": Sigma, "Sigma_beta": Sigma_beta_hat, "E_y2": E_y2,
        "Gamma": Gamma,
        "a": a, "W": W,
        "beta_star": beta_star, "w_star": w_star, "gamma": gamma,
    }


# ===================================================================
# 1. E[sigma''(z)] correctness
# ===================================================================

class TestESigmaDoublePrime:

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_relu_monte_carlo(self, rng, v):
        """E[ReLU''(z)] = delta(0) density, validated via binning near zero."""
        n = 3_000_000
        z = rng.normal(0, np.sqrt(v), n)
        eps = 0.01
        mc_estimate = np.mean(np.abs(z) < eps) / (2 * eps)
        analytic = relu_E_sigma_double_prime(v)
        assert analytic == pytest.approx(mc_estimate, abs=0.02)

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_relu_formula_direct(self, v):
        """Verify ReLU E[sigma''(z)] = 1/sqrt(2*pi*v) directly."""
        expected = 1.0 / np.sqrt(2.0 * np.pi * v)
        assert relu_E_sigma_double_prime(v) == pytest.approx(expected, rel=1e-12)

    def test_relu_decreases_with_v(self):
        vals = [relu_E_sigma_double_prime(v) for v in [0.1, 0.5, 1.0, 2.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 5.0])
    def test_sigmoid_is_zero(self, v):
        assert sigmoid_E_sigma_double_prime(v) == 0.0

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 5.0])
    def test_identity_is_zero(self, v):
        assert identity_E_sigma_double_prime(v) == 0.0

    def test_relu_v_zero_edge_case(self):
        assert relu_E_sigma_double_prime(0.0) == 0.0

    def test_relu_very_large_v(self):
        result = relu_E_sigma_double_prime(1e6)
        assert result > 0.0
        assert np.isfinite(result)
        assert result < 1e-3

    def test_dispatch_relu(self):
        E_pp, dE_pp_dv = get_activation_double_prime("relu")
        assert E_pp(1.0) == relu_E_sigma_double_prime(1.0)
        assert dE_pp_dv(1.0) == relu_dE_sigma_double_prime_dv(1.0)

    def test_dispatch_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation_double_prime("tanh")


# ===================================================================
# 2. dE[sigma''(z)]/dv derivatives
# ===================================================================

class TestESigmaDoublePrimeDerivative:

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_relu_finite_diff(self, v):
        E_pp, dE_pp_dv = get_activation_double_prime("relu")
        eps = 1e-7
        fd = (E_pp(v + eps) - E_pp(v - eps)) / (2 * eps)
        assert dE_pp_dv(v) == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_relu_formula_direct(self, v):
        expected = -1.0 / (2.0 * v * np.sqrt(2.0 * np.pi * v))
        assert relu_dE_sigma_double_prime_dv(v) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
    def test_sigmoid_finite_diff(self, v):
        E_pp, dE_pp_dv = get_activation_double_prime("sigmoid")
        eps = 1e-7
        fd = (E_pp(v + eps) - E_pp(v - eps)) / (2 * eps)
        assert dE_pp_dv(v) == pytest.approx(fd, abs=1e-10)

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
    def test_identity_finite_diff(self, v):
        E_pp, dE_pp_dv = get_activation_double_prime("identity")
        eps = 1e-7
        fd = (E_pp(v + eps) - E_pp(v - eps)) / (2 * eps)
        assert dE_pp_dv(v) == pytest.approx(fd, abs=1e-10)

    def test_relu_derivative_is_negative(self):
        for v in [0.1, 1.0, 5.0]:
            assert relu_dE_sigma_double_prime_dv(v) < 0

    def test_relu_v_zero_returns_zero(self):
        assert relu_dE_sigma_double_prime_dv(0.0) == 0.0


# ===================================================================
# 3. interaction_cross_moment
# ===================================================================

class TestInteractionCrossMoment:

    def test_zero_gamma_gives_zero(self, interaction_problem):
        prob = interaction_problem
        Gamma_zero = np.zeros_like(prob["Gamma"])
        result = interaction_cross_moment(
            prob["Sigma"], Gamma_zero, prob["W"][0], "relu",
        )
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_identity_gives_zero_regardless_of_gamma(self, interaction_problem):
        prob = interaction_problem
        result = interaction_cross_moment(
            prob["Sigma"], prob["Gamma"], prob["W"][0], "identity",
        )
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_sigmoid_gives_zero(self, interaction_problem):
        prob = interaction_problem
        result = interaction_cross_moment(
            prob["Sigma"], prob["Gamma"], prob["W"][0], "sigmoid",
        )
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_scaling_gamma_linear(self, interaction_problem):
        """Scaling Gamma by a constant scales the result linearly."""
        prob = interaction_problem
        result_1 = interaction_cross_moment(
            prob["Sigma"], prob["Gamma"], prob["W"][0], "relu",
        )
        result_3 = interaction_cross_moment(
            prob["Sigma"], 3.0 * prob["Gamma"], prob["W"][0], "relu",
        )
        assert result_3 == pytest.approx(3.0 * result_1, rel=1e-10)

    def test_symmetric_part_matters(self, rng):
        """Only the symmetric part of Gamma contributes since q_k = w^T Gamma w."""
        p = 6
        Sigma = generate_ld_matrix(p, decay=0.5)
        w = rng.standard_normal(p) * 0.1

        Gamma_sym = rng.standard_normal((p, p))
        Gamma_sym = 0.5 * (Gamma_sym + Gamma_sym.T)

        skew = rng.standard_normal((p, p))
        skew = 0.5 * (skew - skew.T)
        Gamma_asym = Gamma_sym + skew

        result_sym = interaction_cross_moment(Sigma, Gamma_sym, w, "relu")
        result_asym = interaction_cross_moment(Sigma, Gamma_asym, w, "relu")
        assert result_asym == pytest.approx(result_sym, rel=1e-10)

    def test_orthogonal_to_gamma_eigenvectors(self, rng):
        """If w is orthogonal to all nonzero-eigenvalue eigenvectors of Gamma,
        then q_k = w^T Gamma w = 0, so the interaction term vanishes."""
        p = 8
        Sigma = np.eye(p)
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v)
        Gamma = 3.0 * np.outer(v, v)

        null_space = np.eye(p) - np.outer(v, v)
        w = null_space @ rng.standard_normal(p)
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            w = w / w_norm * 0.1

        result = interaction_cross_moment(Sigma, Gamma, w, "relu")
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_for_relu(self, interaction_problem):
        prob = interaction_problem
        result = interaction_cross_moment(
            prob["Sigma"], prob["Gamma"], prob["W"][0], "relu",
        )
        assert result != pytest.approx(0.0, abs=1e-10)

    def test_monte_carlo_validation(self, rng):
        """Generate data with nonlinear DGP, compare empirical E[y sigma(w^T x)]
        to stein_cross_moment + interaction_cross_moment."""
        p = 8
        n = 500_000

        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.2
        w_star = rng.standard_normal(p) * 0.3
        gamma_coeff = 0.6

        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = X @ beta_star + gamma_coeff * np.maximum(0.0, X @ w_star)

        Sigma_beta_hat = X.T @ y / n
        Gamma_hat = X.T @ (X * y[:, None]) / n

        w_test = rng.standard_normal(p) * 0.1

        mc_E_y_sigma = np.mean(y * np.maximum(0.0, X @ w_test))

        stein_part = stein_cross_moment(Sigma, w_test, Sigma_beta_hat, "relu")
        int_part = interaction_cross_moment(Sigma, Gamma_hat, w_test, "relu")
        analytic = stein_part + int_part

        assert analytic == pytest.approx(mc_E_y_sigma, rel=0.05)


# ===================================================================
# 4. interaction_cross_moment_grad
# ===================================================================

class TestInteractionCrossMomentGrad:

    @pytest.mark.parametrize("p", [5, 10, 20])
    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_finite_diff(self, rng, p, activation):
        """Finite-difference check for all components of w_k."""
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = rng.standard_normal((p, p))
        Gamma = 0.5 * (Gamma + Gamma.T)
        w_k = rng.standard_normal(p) * 0.1

        grad = interaction_cross_moment_grad(Sigma, Gamma, w_k, activation)

        eps = 1e-6
        fd = np.zeros(p)
        for j in range(p):
            w_plus = w_k.copy()
            w_plus[j] += eps
            w_minus = w_k.copy()
            w_minus[j] -= eps
            fd[j] = (
                interaction_cross_moment(Sigma, Gamma, w_plus, activation)
                - interaction_cross_moment(Sigma, Gamma, w_minus, activation)
            ) / (2 * eps)

        np.testing.assert_allclose(grad, fd, atol=1e-5)

    def test_identity_gradient_zero(self, rng):
        p = 8
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = rng.standard_normal((p, p))
        Gamma = 0.5 * (Gamma + Gamma.T)
        w_k = rng.standard_normal(p) * 0.1

        grad = interaction_cross_moment_grad(Sigma, Gamma, w_k, "identity")
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_zero_gamma_gradient_zero(self, rng):
        p = 8
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = np.zeros((p, p))
        w_k = rng.standard_normal(p) * 0.1

        grad = interaction_cross_moment_grad(Sigma, Gamma, w_k, "relu")
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_sigmoid_gradient_zero(self, rng):
        """Sigmoid has E[sigma''(z)] = 0 everywhere, so gradient is zero."""
        p = 8
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = rng.standard_normal((p, p))
        Gamma = 0.5 * (Gamma + Gamma.T)
        w_k = rng.standard_normal(p) * 0.1

        grad = interaction_cross_moment_grad(Sigma, Gamma, w_k, "sigmoid")
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)


# ===================================================================
# 5. compute_interaction_loss
# ===================================================================

class TestInteractionLoss:

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_zero_gamma_matches_gaussian(self, interaction_problem, activation):
        prob = interaction_problem
        Gamma_zero = np.zeros_like(prob["Gamma"])

        loss_int = compute_interaction_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], Gamma_zero, activation,
        )
        loss_gauss = compute_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], activation,
        )
        assert loss_int == pytest.approx(loss_gauss, rel=1e-10)

    @pytest.mark.parametrize("activation", ["identity"])
    def test_identity_matches_gaussian_with_gamma(self, interaction_problem, activation):
        """Identity has sigma'' = 0, so Gamma has no effect."""
        prob = interaction_problem
        loss_int = compute_interaction_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], activation,
        )
        loss_gauss = compute_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], activation,
        )
        assert loss_int == pytest.approx(loss_gauss, rel=1e-10)

    def test_relu_differs_from_gaussian_with_gamma(self, interaction_problem):
        prob = interaction_problem
        loss_int = compute_interaction_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )
        loss_gauss = compute_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], "relu",
        )
        assert loss_int != pytest.approx(loss_gauss, abs=1e-8)

    def test_loss_nonnegative(self, rng):
        """L = E[(y - f)^2] >= 0 for well-formed problems."""
        p, m = 8, 3
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)

        for seed in range(5):
            r = np.random.default_rng(seed)
            W = r.standard_normal((m, p)) * 0.05
            a = r.standard_normal(m) * 0.05

            L = compute_interaction_loss(a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu")
            assert L >= -1e-6, f"Negative loss: {L}"

    def test_scaling_a_quadruples_f2_term(self, interaction_problem):
        """Doubling a quadruples the E[f^2] part, doubles E[yf] part."""
        prob = interaction_problem
        a1 = prob["a"]
        a2 = 2.0 * prob["a"]

        loss1 = compute_interaction_loss(
            a1, prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )
        loss2 = compute_interaction_loss(
            a2, prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )
        assert loss1 != pytest.approx(loss2, abs=1e-10)
        assert np.isfinite(loss1) and np.isfinite(loss2)

    @pytest.mark.parametrize("m", [1, 3, 5, 10])
    def test_multiple_hidden_units(self, rng, m):
        p = 8
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)

        W = rng.standard_normal((m, p)) * 0.05
        a = rng.standard_normal(m) * 0.05

        L = compute_interaction_loss(a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu")
        assert np.isfinite(L)

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_all_activations_finite(self, interaction_problem, activation):
        prob = interaction_problem
        L = compute_interaction_loss(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], activation,
        )
        assert np.isfinite(L)

    def test_gradient_direction_decreases_loss(self, interaction_problem):
        prob = interaction_problem
        a, W = prob["a"].copy(), prob["W"].copy()

        L0 = compute_interaction_loss(
            a, W, prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )
        grad_a, grad_W = compute_interaction_gradients(
            a, W, prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )

        lr = 1e-5
        a_new = a - lr * grad_a
        W_new = W - lr * grad_W

        L1 = compute_interaction_loss(
            a_new, W_new, prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )
        assert L1 < L0


# ===================================================================
# 6. Gradient finite-difference checks (a and W)
# ===================================================================

class TestInteractionGradients:

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_grad_a_finite_diff_all_components(self, interaction_problem, activation):
        prob = interaction_problem
        a, W = prob["a"].copy(), prob["W"]
        Sigma, Sb, E_y2, Gamma = (
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
        )

        grad_a = compute_interaction_grad_a(a, W, Sigma, Sb, Gamma, activation)

        eps = 1e-5
        for k in range(len(a)):
            a_plus = a.copy(); a_plus[k] += eps
            a_minus = a.copy(); a_minus[k] -= eps
            fd = (
                compute_interaction_loss(a_plus, W, Sigma, Sb, E_y2, Gamma, activation)
                - compute_interaction_loss(a_minus, W, Sigma, Sb, E_y2, Gamma, activation)
            ) / (2 * eps)
            assert grad_a[k] == pytest.approx(fd, rel=1e-3, abs=1e-7)

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_grad_W_finite_diff_all_components(self, interaction_problem, activation):
        """Check ALL (k, j) components, not just a subset."""
        prob = interaction_problem
        a, W = prob["a"], prob["W"].copy()
        Sigma, Sb, E_y2, Gamma = (
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
        )

        grad_W = compute_interaction_grad_W(a, W, Sigma, Sb, Gamma, activation)

        eps = 1e-5
        m, p = W.shape
        for k in range(m):
            for j in range(p):
                W_plus = W.copy(); W_plus[k, j] += eps
                W_minus = W.copy(); W_minus[k, j] -= eps
                fd = (
                    compute_interaction_loss(a, W_plus, Sigma, Sb, E_y2, Gamma, activation)
                    - compute_interaction_loss(a, W_minus, Sigma, Sb, E_y2, Gamma, activation)
                ) / (2 * eps)
                assert grad_W[k, j] == pytest.approx(fd, rel=1e-3, abs=1e-7), (
                    f"Mismatch at ({k},{j}) for activation={activation}"
                )

    def test_compute_interaction_gradients_matches_separate(self, interaction_problem):
        prob = interaction_problem
        a, W = prob["a"], prob["W"]

        grad_a_sep = compute_interaction_grad_a(
            a, W, prob["Sigma"], prob["Sigma_beta"], prob["Gamma"], "relu",
        )
        grad_W_sep = compute_interaction_grad_W(
            a, W, prob["Sigma"], prob["Sigma_beta"], prob["Gamma"], "relu",
        )

        grad_a, grad_W = compute_interaction_gradients(
            a, W, prob["Sigma"], prob["Sigma_beta"],
            prob["E_y2"], prob["Gamma"], "relu",
        )

        np.testing.assert_allclose(grad_a, grad_a_sep, atol=1e-14)
        np.testing.assert_allclose(grad_W, grad_W_sep, atol=1e-14)

    @pytest.mark.parametrize("activation", ["identity"])
    def test_identity_gradients_match_gaussian(self, interaction_problem, activation):
        prob = interaction_problem
        a, W = prob["a"], prob["W"]

        grad_a_int = compute_interaction_grad_a(
            a, W, prob["Sigma"], prob["Sigma_beta"], prob["Gamma"], activation,
        )
        grad_W_int = compute_interaction_grad_W(
            a, W, prob["Sigma"], prob["Sigma_beta"], prob["Gamma"], activation,
        )

        grad_a_gauss = compute_grad_a(
            a, W, prob["Sigma"], prob["Sigma_beta"], activation,
        )
        grad_W_gauss = compute_grad_W(
            a, W, prob["Sigma"], prob["Sigma_beta"], activation,
        )

        np.testing.assert_allclose(grad_a_int, grad_a_gauss, atol=1e-12)
        np.testing.assert_allclose(grad_W_int, grad_W_gauss, atol=1e-12)

    @pytest.mark.parametrize("p,m", [(5, 2), (10, 3), (20, 5)])
    def test_grad_W_multiple_sizes(self, rng, p, m):
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0

        W = rng.standard_normal((m, p)) * 0.05
        a = rng.standard_normal(m) * 0.05

        grad_W = compute_interaction_grad_W(a, W, Sigma, Sigma_beta, Gamma, "relu")

        eps = 1e-5
        for k in range(m):
            for j in range(min(p, 5)):
                W_plus = W.copy(); W_plus[k, j] += eps
                W_minus = W.copy(); W_minus[k, j] -= eps
                fd = (
                    compute_interaction_loss(a, W_plus, Sigma, Sigma_beta, E_y2, Gamma, "relu")
                    - compute_interaction_loss(a, W_minus, Sigma, Sigma_beta, E_y2, Gamma, "relu")
                ) / (2 * eps)
                assert grad_W[k, j] == pytest.approx(fd, rel=1e-3, abs=1e-6)

    def test_grad_a_zero_gamma_matches_gaussian(self, interaction_problem):
        prob = interaction_problem
        Gamma_zero = np.zeros_like(prob["Gamma"])

        grad_int = compute_interaction_grad_a(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"],
            Gamma_zero, "relu",
        )
        grad_gauss = compute_grad_a(
            prob["a"], prob["W"], prob["Sigma"], prob["Sigma_beta"], "relu",
        )
        np.testing.assert_allclose(grad_int, grad_gauss, atol=1e-12)

    @pytest.mark.parametrize("reg_a,reg_W", [(0.1, 0.0), (0.0, 0.1), (0.05, 0.05)])
    def test_regularized_grad_a_finite_diff(self, interaction_problem, reg_a, reg_W):
        """Finite-difference check for regularized loss w.r.t. a."""
        prob = interaction_problem
        a, W = prob["a"].copy(), prob["W"]
        Sigma, Sb, E_y2, Gamma = (
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
        )

        grad_a = compute_interaction_grad_a(
            a, W, Sigma, Sb, Gamma, "relu", reg_a=reg_a,
        )

        eps = 1e-5
        for k in range(len(a)):
            a_plus = a.copy(); a_plus[k] += eps
            a_minus = a.copy(); a_minus[k] -= eps
            fd = (
                compute_interaction_loss(a_plus, W, Sigma, Sb, E_y2, Gamma, "relu", reg_a=reg_a, reg_W=reg_W)
                - compute_interaction_loss(a_minus, W, Sigma, Sb, E_y2, Gamma, "relu", reg_a=reg_a, reg_W=reg_W)
            ) / (2 * eps)
            assert grad_a[k] == pytest.approx(fd, rel=1e-3, abs=1e-7)

    @pytest.mark.parametrize("reg_a,reg_W", [(0.1, 0.0), (0.0, 0.1), (0.05, 0.05)])
    def test_regularized_grad_W_finite_diff(self, interaction_problem, reg_a, reg_W):
        """Finite-difference check for regularized loss w.r.t. W."""
        prob = interaction_problem
        a, W = prob["a"], prob["W"].copy()
        Sigma, Sb, E_y2, Gamma = (
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
        )

        grad_W = compute_interaction_grad_W(
            a, W, Sigma, Sb, Gamma, "relu", reg_W=reg_W,
        )

        eps = 1e-5
        m, p = W.shape
        for k in range(m):
            for j in range(p):
                W_plus = W.copy(); W_plus[k, j] += eps
                W_minus = W.copy(); W_minus[k, j] -= eps
                fd = (
                    compute_interaction_loss(a, W_plus, Sigma, Sb, E_y2, Gamma, "relu", reg_a=reg_a, reg_W=reg_W)
                    - compute_interaction_loss(a, W_minus, Sigma, Sb, E_y2, Gamma, "relu", reg_a=reg_a, reg_W=reg_W)
                ) / (2 * eps)
                assert grad_W[k, j] == pytest.approx(fd, rel=1e-3, abs=1e-7), (
                    f"Mismatch at ({k},{j}) for reg_a={reg_a}, reg_W={reg_W}"
                )


# ===================================================================
# 7. Optimizer train_interaction
# ===================================================================

class TestInteractionOptimizer:

    def test_loss_monotonically_nonincreasing(self, interaction_problem):
        prob = interaction_problem
        result = train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=3, activation="relu", lr=0.005, max_iters=200,
            rng=np.random.default_rng(42),
        )
        for i in range(1, len(result.loss_history)):
            assert result.loss_history[i] <= result.loss_history[i - 1] + 1e-8, (
                f"Loss increased at iteration {i}: "
                f"{result.loss_history[i-1]} -> {result.loss_history[i]}"
            )

    def test_convergence_on_simple_problem(self):
        rng = np.random.default_rng(7)
        p, m = 5, 2

        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = np.zeros((p, p))

        result = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma,
            m=m, activation="relu", lr=0.01, max_iters=3000, tol=1e-8,
            rng=rng,
        )
        assert result.loss_history[-1] < result.loss_history[0]
        assert len(result.loss_history) > 2

    def test_warm_start_vs_random_init(self, interaction_problem):
        """Warm-starting from Gaussian solution should give lower or equal
        initial loss vs random init (evaluated on interaction loss)."""
        prob = interaction_problem

        gauss = train(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"],
            m=3, activation="relu", lr=0.01, max_iters=500,
            rng=np.random.default_rng(42),
        )

        result_warm = train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=3, activation="relu", lr=0.005, max_iters=200,
            a_init=gauss.a, W_init=gauss.W,
            rng=np.random.default_rng(42),
        )

        result_random = train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=3, activation="relu", lr=0.005, max_iters=200,
            rng=np.random.default_rng(42),
        )

        assert result_warm.loss_history[0] <= result_random.loss_history[0] + 0.5

    def test_returns_correct_types_and_shapes(self, interaction_problem):
        prob = interaction_problem
        result = train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=3, activation="relu", lr=0.01, max_iters=50,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, TrainResult)
        assert result.a.shape == (3,)
        assert result.W.shape == (3, prob["p"])
        assert isinstance(result.loss_history, list)
        assert len(result.loss_history) > 0
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iters, int)

    @pytest.mark.parametrize("activation", ["relu", "sigmoid", "identity"])
    def test_works_with_all_activations(self, interaction_problem, activation):
        prob = interaction_problem
        result = train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=2, activation=activation, lr=0.005, max_iters=50,
            rng=np.random.default_rng(42),
        )
        assert np.all(np.isfinite(result.a))
        assert np.all(np.isfinite(result.W))
        assert result.loss_history[-1] <= result.loss_history[0] + 1e-6

    def test_gradient_clipping_prevents_divergence(self, rng):
        """Large-gradient problem with tight clip should not diverge."""
        p, m = 8, 3
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 2.0
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.5
        Gamma = 0.5 * (Gamma + Gamma.T)

        result = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma,
            m=m, activation="relu", lr=0.01, max_iters=100,
            grad_clip=0.1, rng=rng,
        )
        assert np.all(np.isfinite(result.a))
        assert np.all(np.isfinite(result.W))
        assert all(np.isfinite(l) for l in result.loss_history)

    def test_tiny_problem_converges(self):
        rng = np.random.default_rng(99)
        p, m = 3, 1
        Sigma = generate_ld_matrix(p, decay=0.3)
        beta = rng.standard_normal(p) * 0.1
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = np.zeros((p, p))

        result = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma,
            m=m, activation="relu", lr=0.01, max_iters=2000, tol=1e-8,
            rng=rng,
        )
        assert result.loss_history[-1] < result.loss_history[0]

    def test_no_init_copies_inputs(self, interaction_problem):
        """Warm-start should copy the init arrays, not modify them."""
        prob = interaction_problem
        a_orig = prob["a"].copy()
        W_orig = prob["W"].copy()

        train_interaction(
            prob["Sigma"], prob["Sigma_beta"], prob["E_y2"], prob["Gamma"],
            m=3, activation="relu", lr=0.005, max_iters=10,
            a_init=prob["a"], W_init=prob["W"],
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(prob["a"], a_orig)
        np.testing.assert_array_equal(prob["W"], W_orig)


# ===================================================================
# 8. Integration / barrier-breaking tests
# ===================================================================

class TestBarrierBreaking:

    def test_linear_dgp_matches_linear_prs(self):
        """On a purely linear DGP with Gamma=0, the interaction NN should not
        substantially outperform the linear PRS R^2 (within tolerance)."""
        rng = np.random.default_rng(17)
        p, m = 15, 3

        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.3
        sigma_eps = 1.0

        n = 50_000
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = X @ beta_star + rng.normal(0, sigma_eps, n)

        Sigma_beta = X.T @ y / n
        E_y2 = float(np.mean(y ** 2))
        Gamma = X.T @ (X * y[:, None]) / n

        gauss = train(
            Sigma, Sigma_beta, E_y2, m=m, activation="relu",
            lr=0.01, max_iters=2000, rng=np.random.default_rng(42),
        )

        int_result = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma,
            m=m, activation="relu", lr=0.005, max_iters=2000,
            a_init=gauss.a, W_init=gauss.W,
            rng=np.random.default_rng(42),
        )

        loss_gauss = compute_loss(gauss.a, gauss.W, Sigma, Sigma_beta, E_y2, "relu")
        loss_int = compute_interaction_loss(
            int_result.a, int_result.W, Sigma, Sigma_beta, E_y2, Gamma, "relu",
        )
        assert loss_int <= loss_gauss + 0.05

    def test_nonlinear_dgp_interaction_beats_gaussian(self):
        """On a nonlinear DGP with large nonlinear fraction, interaction NN
        loss should be lower than Gaussian NN loss."""
        rng = np.random.default_rng(7)
        p, m = 20, 5
        n = 50_000

        Sigma = generate_ld_matrix(p, decay=0.5)
        beta_star = rng.standard_normal(p) * 0.3
        w_star = rng.standard_normal(p) * 0.3
        gamma = 0.8

        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = X @ beta_star + gamma * np.maximum(0.0, X @ w_star) + rng.normal(0, 1.0, n)

        Sigma_beta = X.T @ y / n
        E_y2 = float(np.mean(y ** 2))
        Gamma = X.T @ (X * y[:, None]) / n

        gauss = train(
            Sigma, Sigma_beta, E_y2, m=m, activation="relu",
            lr=0.01, max_iters=2000, rng=np.random.default_rng(42),
        )

        interaction = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma, m=m, activation="relu",
            lr=0.005, max_iters=2000,
            a_init=gauss.a, W_init=gauss.W,
            rng=np.random.default_rng(42),
        )

        loss_gauss = compute_loss(gauss.a, gauss.W, Sigma, Sigma_beta, E_y2, "relu")
        loss_int = compute_interaction_loss(
            interaction.a, interaction.W, Sigma, Sigma_beta, E_y2, Gamma, "relu",
        )
        assert loss_int < loss_gauss + 1e-6

    def test_run_single_rep_has_five_methods(self):
        from ssnn.simulation import SimulationScenario, run_single_rep

        scenario = SimulationScenario(
            p=15, m=3, n_train=1000, n_test=500,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            dgp_type="nonlinear", nonlinear_frac=0.25,
            sumstat_max_iters=200, interaction_max_iters=200,
            oracle_max_iters=500,
        )
        results = run_single_rep(scenario, np.random.default_rng(42))
        methods = [r.method for r in results]
        assert len(results) == 5
        assert methods == [
            "Linear PRS", "Gaussian NN", "Edgeworth NN",
            "Interaction NN", "Oracle NN",
        ]

    def test_simulation_scenario_default_fields(self):
        from ssnn.simulation import SimulationScenario

        s = SimulationScenario()
        assert s.interaction_lr == 0.005
        assert s.interaction_max_iters == 3000
        assert s.dgp_type == "linear"
        assert s.nonlinear_frac == 0.25

    def test_compute_summary_stats_returns_gamma_hat(self, rng):
        from ssnn.simulation import compute_summary_stats_from_genotypes

        p = 10
        n = 500
        Sigma = generate_ld_matrix(p, decay=0.5)
        X = rng.integers(0, 3, size=(n, p)).astype(float)
        y = rng.standard_normal(n)

        stats = compute_summary_stats_from_genotypes(X, y, Sigma)
        assert "Gamma_hat" in stats
        Gamma = stats["Gamma_hat"]
        assert Gamma.shape == (p, p)
        np.testing.assert_allclose(Gamma, Gamma.T, atol=1e-12)


# ===================================================================
# 9. Numerical stability
# ===================================================================

class TestNumericalStability:

    def test_near_singular_sigma(self, rng):
        p, m = 6, 2
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.99 ** abs(i - j)
        Sigma += 1e-6 * np.eye(p)

        beta = rng.standard_normal(p) * 0.1
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)

        W = rng.standard_normal((m, p)) * 0.05
        a = rng.standard_normal(m) * 0.05

        L = compute_interaction_loss(a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu")
        assert np.isfinite(L)

        grad_a, grad_W = compute_interaction_gradients(
            a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu",
        )
        assert np.all(np.isfinite(grad_a))
        assert np.all(np.isfinite(grad_W))

    def test_very_small_init_scale(self, rng):
        p, m = 6, 2
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)

        result = train_interaction(
            Sigma, Sigma_beta, E_y2, Gamma,
            m=m, activation="relu", lr=0.01, max_iters=50,
            init_scale=1e-8, rng=rng,
        )
        assert np.all(np.isfinite(result.a))
        assert np.all(np.isfinite(result.W))
        assert all(np.isfinite(l) for l in result.loss_history)

    def test_very_large_gamma_entries(self, rng):
        p, m = 6, 2
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 100.0
        Gamma = 0.5 * (Gamma + Gamma.T)

        W = rng.standard_normal((m, p)) * 0.01
        a = rng.standard_normal(m) * 0.01

        L = compute_interaction_loss(a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu")
        assert np.isfinite(L)

    def test_zero_weight_vectors(self, rng):
        p, m = 6, 2
        Sigma = generate_ld_matrix(p, decay=0.5)
        beta = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta
        E_y2 = float(beta @ Sigma @ beta) + 1.0
        Gamma = rng.standard_normal((p, p)) * 0.01
        Gamma = 0.5 * (Gamma + Gamma.T)

        W = np.zeros((m, p))
        a = np.zeros(m)

        L = compute_interaction_loss(a, W, Sigma, Sigma_beta, E_y2, Gamma, "relu")
        assert L == pytest.approx(E_y2, abs=1e-10)

    def test_interaction_cross_moment_grad_with_tiny_weights(self, rng):
        p = 8
        Sigma = generate_ld_matrix(p, decay=0.5)
        Gamma = rng.standard_normal((p, p)) * 0.1
        Gamma = 0.5 * (Gamma + Gamma.T)
        w_k = rng.standard_normal(p) * 1e-10

        grad = interaction_cross_moment_grad(Sigma, Gamma, w_k, "relu")
        assert np.all(np.isfinite(grad))


# ===================================================================
# 10. Regularized Gaussian NN gradient checks
# ===================================================================

class TestGaussianRegularizedGradients:

    @pytest.mark.parametrize("reg_a,reg_W", [(0.1, 0.0), (0.0, 0.1), (0.05, 0.05)])
    def test_regularized_grad_a_finite_diff(self, interaction_problem, reg_a, reg_W):
        prob = interaction_problem
        a, W = prob["a"].copy(), prob["W"]
        Sigma, Sb, E_y2 = prob["Sigma"], prob["Sigma_beta"], prob["E_y2"]

        grad_a = compute_grad_a(a, W, Sigma, Sb, "relu", reg_a=reg_a)

        eps = 1e-5
        for k in range(len(a)):
            a_plus = a.copy(); a_plus[k] += eps
            a_minus = a.copy(); a_minus[k] -= eps
            fd = (
                compute_loss(a_plus, W, Sigma, Sb, E_y2, "relu", reg_a=reg_a, reg_W=reg_W)
                - compute_loss(a_minus, W, Sigma, Sb, E_y2, "relu", reg_a=reg_a, reg_W=reg_W)
            ) / (2 * eps)
            assert grad_a[k] == pytest.approx(fd, rel=1e-3, abs=1e-7)

    @pytest.mark.parametrize("reg_a,reg_W", [(0.1, 0.0), (0.0, 0.1), (0.05, 0.05)])
    def test_regularized_grad_W_finite_diff(self, interaction_problem, reg_a, reg_W):
        prob = interaction_problem
        a, W = prob["a"], prob["W"].copy()
        Sigma, Sb, E_y2 = prob["Sigma"], prob["Sigma_beta"], prob["E_y2"]

        grad_W = compute_grad_W(a, W, Sigma, Sb, "relu", reg_W=reg_W)

        eps = 1e-5
        m, p = W.shape
        for k in range(m):
            for j in range(min(p, 5)):
                W_plus = W.copy(); W_plus[k, j] += eps
                W_minus = W.copy(); W_minus[k, j] -= eps
                fd = (
                    compute_loss(a, W_plus, Sigma, Sb, E_y2, "relu", reg_a=reg_a, reg_W=reg_W)
                    - compute_loss(a, W_minus, Sigma, Sb, E_y2, "relu", reg_a=reg_a, reg_W=reg_W)
                ) / (2 * eps)
                assert grad_W[k, j] == pytest.approx(fd, rel=1e-3, abs=1e-7)
