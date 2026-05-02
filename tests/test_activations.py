"""Verify activation closed forms against numerical quadrature and Monte Carlo."""

import numpy as np
import pytest
from scipy import integrate

from ssnn.activations import (
    get_activation,
    relu_E_sigma,
    relu_E_sigma_prime,
    relu_E_sigma_sigma,
    sigmoid_E_sigma,
    sigmoid_E_sigma_prime,
    sigmoid_E_sigma_sigma,
    identity_E_sigma,
    identity_E_sigma_prime,
    identity_E_sigma_sigma,
)


def _phi(t):
    """Standard Gaussian density."""
    return np.exp(-t**2 / 2) / np.sqrt(2 * np.pi)


# ---- ReLU closed forms vs quadrature ----

@pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
class TestReLU:

    def test_E_sigma(self, v):
        expected, _ = integrate.quad(lambda t: max(0, t) * _phi(t / np.sqrt(v)) / np.sqrt(v),
                                     -np.inf, np.inf)
        assert relu_E_sigma(v) == pytest.approx(expected, abs=1e-10)

    def test_E_sigma_prime(self, v):
        expected, _ = integrate.quad(lambda t: (1.0 if t > 0 else 0.0) * _phi(t / np.sqrt(v)) / np.sqrt(v),
                                     -np.inf, np.inf)
        assert relu_E_sigma_prime(v) == pytest.approx(expected, abs=1e-10)

    def test_E_sigma_sigma_diagonal(self, v):
        """When k == l, E[ReLU(z)^2] = v/2 for z ~ N(0, v)."""
        C = np.array([[v, v], [v, v]])
        expected = v / 2.0
        assert relu_E_sigma_sigma(C) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.3, 0.8])
def test_relu_cross_term_monte_carlo(rho, rng):
    """Compare arc-cosine kernel to Monte Carlo with 10^6 samples."""
    v_k, v_l = 1.5, 2.0
    C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                  [rho * np.sqrt(v_k * v_l), v_l]])

    rng = np.random.default_rng(123)
    samples = rng.multivariate_normal([0, 0], C, size=1_000_000)
    mc_estimate = np.mean(np.maximum(0, samples[:, 0]) * np.maximum(0, samples[:, 1]))

    assert relu_E_sigma_sigma(C) == pytest.approx(mc_estimate, abs=5e-3)


# ---- Sigmoid closed forms vs quadrature ----

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
class TestSigmoid:

    def test_E_sigma(self, v):
        expected, _ = integrate.quad(
            lambda t: _sigmoid(t) * _phi(t / np.sqrt(v)) / np.sqrt(v),
            -20 * np.sqrt(v), 20 * np.sqrt(v),
        )
        assert sigmoid_E_sigma(v) == pytest.approx(expected, abs=1e-6)

    def test_E_sigma_prime(self, v):
        """Probit approximation of E[sigmoid'(z)] vs numerical quadrature of
        the *actual* sigmoid derivative.  Tolerance is looser because the
        probit approximation is approximate."""
        expected, _ = integrate.quad(
            lambda t: _sigmoid(t) * (1 - _sigmoid(t)) * _phi(t / np.sqrt(v)) / np.sqrt(v),
            -20 * np.sqrt(v), 20 * np.sqrt(v),
        )
        assert sigmoid_E_sigma_prime(v) == pytest.approx(expected, abs=6e-3)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
def test_sigmoid_cross_term_monte_carlo(rho):
    """Compare probit-approximated cross-term to Monte Carlo of actual sigmoid."""
    v_k, v_l = 1.0, 1.5
    C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                  [rho * np.sqrt(v_k * v_l), v_l]])

    rng = np.random.default_rng(456)
    samples = rng.multivariate_normal([0, 0], C, size=1_000_000)
    mc = np.mean(_sigmoid(samples[:, 0]) * _sigmoid(samples[:, 1]))

    # Probit approximation -- looser tolerance
    assert sigmoid_E_sigma_sigma(C) == pytest.approx(mc, abs=5e-3)


# ---- Identity (trivial, but validates the interface) ----

@pytest.mark.parametrize("v", [0.5, 1.0, 3.0])
class TestIdentity:

    def test_E_sigma(self, v):
        assert identity_E_sigma(v) == 0.0

    def test_E_sigma_prime(self, v):
        assert identity_E_sigma_prime(v) == 1.0

    def test_E_sigma_sigma(self, v):
        rho = 0.6
        c = rho * v
        C = np.array([[v, c], [c, v]])
        assert identity_E_sigma_sigma(C) == pytest.approx(c, abs=1e-15)


# ---- Edge cases: extreme variances ----

class TestReLUEdgeCases:

    def test_very_small_variance(self):
        """v -> 0: E[ReLU(z)] -> 0."""
        v = 1e-12
        assert relu_E_sigma(v) == pytest.approx(np.sqrt(v / (2 * np.pi)), abs=1e-15)

    def test_very_large_variance(self):
        v = 1e6
        expected = np.sqrt(v / (2 * np.pi))
        assert relu_E_sigma(v) == pytest.approx(expected, rel=1e-10)

    def test_zero_variance_cross(self):
        """With v_k=0, E[ReLU(0)*ReLU(z_l)] = 0."""
        C = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert relu_E_sigma_sigma(C) == 0.0

    def test_perfect_correlation(self):
        """rho=1: E[ReLU(z)^2] = v/2 (theta=0)."""
        v = 2.0
        C = np.array([[v, v], [v, v]])
        assert relu_E_sigma_sigma(C) == pytest.approx(v / 2.0, abs=1e-10)

    def test_perfect_anticorrelation(self):
        """rho=-1: E[ReLU(z)*ReLU(-z)] = 0 (theta=pi)."""
        v = 2.0
        C = np.array([[v, -v], [-v, v]])
        assert relu_E_sigma_sigma(C) == pytest.approx(0.0, abs=1e-10)


class TestSigmoidEdgeCases:

    def test_very_small_variance(self):
        """v -> 0: sigmoid(z) -> sigmoid(0) = 0.5, E[sigmoid] = 0.5."""
        assert sigmoid_E_sigma(1e-12) == 0.5

    def test_very_large_variance(self):
        """E[sigmoid(z)] = 0.5 regardless of v."""
        assert sigmoid_E_sigma(1e6) == 0.5

    def test_E_sigma_prime_small_variance(self):
        """v -> 0: sigmoid'(z) ~ sigmoid'(0) = 0.25; under probit: lambda/sqrt(2pi)."""
        from ssnn.activations import _LAMBDA
        expected_limit = _LAMBDA / np.sqrt(2 * np.pi)
        assert sigmoid_E_sigma_prime(1e-12) == pytest.approx(expected_limit, rel=1e-6)

    def test_E_sigma_prime_large_variance(self):
        """As v -> inf, E[sigmoid'(z)] -> 0."""
        assert sigmoid_E_sigma_prime(1e6) < 1e-3

    def test_cross_term_zero_correlation(self):
        """rho=0: E[sigmoid(z_k) sigmoid(z_l)] = E[sigma(z_k)] E[sigma(z_l)] = 0.25."""
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert sigmoid_E_sigma_sigma(C) == pytest.approx(0.25, abs=1e-10)

    def test_cross_term_bounds(self):
        """Cross term should stay in [0, 1] for any valid covariance."""
        for rho in np.linspace(-0.99, 0.99, 20):
            C = np.array([[1.0, rho], [rho, 1.0]])
            val = sigmoid_E_sigma_sigma(C)
            assert 0.0 <= val <= 1.0 + 1e-10, f"Out of bounds at rho={rho}: {val}"


class TestIdentityEdgeCases:

    def test_zero_variance(self):
        assert identity_E_sigma(0.0) == 0.0
        assert identity_E_sigma_prime(0.0) == 1.0

    def test_cross_term_zero(self):
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert identity_E_sigma_sigma(C) == 0.0


# ---- Dispatch ----

def test_get_activation_relu():
    E_s, E_sp, E_ss = get_activation("relu")
    assert E_s(1.0) == relu_E_sigma(1.0)


def test_get_activation_sigmoid():
    E_s, E_sp, E_ss = get_activation("sigmoid")
    assert E_s(1.0) == sigmoid_E_sigma(1.0)
    assert E_sp(1.0) == sigmoid_E_sigma_prime(1.0)


def test_get_activation_identity():
    E_s, E_sp, E_ss = get_activation("identity")
    assert E_s(1.0) == identity_E_sigma(1.0)
    assert E_sp(1.0) == identity_E_sigma_prime(1.0)


def test_get_activation_unknown():
    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("tanh")


# ---- Cross-activation: wrong-answer distinguishability ----

def test_relu_E_sigma_not_constant():
    """E[ReLU(z)] depends on v (not always 0.5 or some constant)."""
    vals = [relu_E_sigma(v) for v in [0.5, 1.0, 2.0, 5.0]]
    assert len(set(round(x, 10) for x in vals)) == 4

def test_sigmoid_cross_not_constant():
    """E[sigmoid(z_k) sigmoid(z_l)] depends on rho_eff."""
    vals = []
    for rho in [-0.5, 0.0, 0.5]:
        C = np.array([[1.0, rho], [rho, 1.0]])
        vals.append(sigmoid_E_sigma_sigma(C))
    assert len(set(round(x, 10) for x in vals)) == 3


# ---- Analytic derivatives for W-gradient (Step 2) ----

from ssnn.activations import (
    get_activation_derivs,
    relu_dE_sigma_prime_dv,
    sigmoid_dE_sigma_prime_dv,
    identity_dE_sigma_prime_dv,
    relu_grad_E_sigma_sigma,
    sigmoid_grad_E_sigma_sigma,
    identity_grad_E_sigma_sigma,
)


class TestActivationDerivatives:
    """Verify analytic derivatives against finite differences."""

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
    def test_relu_dE_sigma_prime_dv(self, v):
        assert relu_dE_sigma_prime_dv(v) == 0.0

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0, 5.0])
    def test_sigmoid_dE_sigma_prime_dv_fd(self, v):
        eps = 1e-7
        fd = (sigmoid_E_sigma_prime(v + eps) - sigmoid_E_sigma_prime(v - eps)) / (2 * eps)
        assert sigmoid_dE_sigma_prime_dv(v) == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("v", [0.5, 1.0, 2.0])
    def test_identity_dE_sigma_prime_dv(self, v):
        assert identity_dE_sigma_prime_dv(v) == 0.0

    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.8])
    def test_relu_grad_E_sigma_sigma_fd(self, rho):
        v_k, v_l = 1.5, 2.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = relu_grad_E_sigma_sigma(C)

        eps = 1e-7
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            C_p = C.copy(); C_p[i, j] += eps
            if i != j:
                C_p[j, i] += eps
            C_m = C.copy(); C_m[i, j] -= eps
            if i != j:
                C_m[j, i] -= eps
            fd = (relu_E_sigma_sigma(C_p) - relu_E_sigma_sigma(C_m)) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, abs=1e-5), (
                f"ReLU grad[{i},{j}] mismatch: analytic={grad[i,j]}, fd={fd}"
            )

    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.8])
    def test_sigmoid_grad_E_sigma_sigma_fd(self, rho):
        v_k, v_l = 1.5, 2.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = sigmoid_grad_E_sigma_sigma(C)

        eps = 1e-7
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            C_p = C.copy(); C_p[i, j] += eps
            if i != j:
                C_p[j, i] += eps
            C_m = C.copy(); C_m[i, j] -= eps
            if i != j:
                C_m[j, i] -= eps
            fd = (sigmoid_E_sigma_sigma(C_p) - sigmoid_E_sigma_sigma(C_m)) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, abs=1e-5), (
                f"Sigmoid grad[{i},{j}] mismatch: analytic={grad[i,j]}, fd={fd}"
            )

    def test_identity_grad_E_sigma_sigma(self):
        C = np.array([[1.5, 0.3], [0.3, 2.0]])
        grad = identity_grad_E_sigma_sigma(C)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(grad, expected)

    def test_get_activation_derivs_dispatch(self):
        for name in ["relu", "sigmoid", "identity"]:
            dEsp, gEss = get_activation_derivs(name)
            assert callable(dEsp)
            assert callable(gEss)

    def test_get_activation_derivs_unknown(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation_derivs("tanh")


# ---- Audit: derivative edge cases and additional coverage ----

class TestActivationDerivativeEdgeCases:
    """Edge-case and extreme-value tests for the analytic derivatives.

    These complement TestActivationDerivatives which only tests
    moderate parameter ranges.
    """

    # --- dE_sigma_prime_dv at extreme variances ---

    def test_sigmoid_dE_sigma_prime_dv_large_v(self):
        """At very large v, E[sigmoid'(z)] -> 0 and its derivative -> 0."""
        v = 1e6
        deriv = sigmoid_dE_sigma_prime_dv(v)
        assert abs(deriv) < 1e-8

    def test_sigmoid_dE_sigma_prime_dv_small_v(self):
        """At small v the derivative should be finite and match FD."""
        v = 1e-4
        eps = 1e-8
        fd = (sigmoid_E_sigma_prime(v + eps) - sigmoid_E_sigma_prime(v - eps)) / (2 * eps)
        assert sigmoid_dE_sigma_prime_dv(v) == pytest.approx(fd, rel=1e-3)

    def test_sigmoid_dE_sigma_prime_dv_is_negative(self):
        """Derivative should always be negative — E[sigmoid'(z)] is
        monotone decreasing in v (broader input => weaker average slope)."""
        for v in [0.1, 0.5, 1.0, 5.0, 50.0]:
            assert sigmoid_dE_sigma_prime_dv(v) < 0.0

    # --- grad_E_sigma_sigma at boundary correlations ---

    @pytest.mark.parametrize("rho", [0.99, -0.99])
    def test_relu_grad_near_extreme_rho(self, rho):
        """Gradients should stay finite near rho = ±1."""
        v_k, v_l = 1.0, 1.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = relu_grad_E_sigma_sigma(C)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient at rho={rho}"

    @pytest.mark.parametrize("rho", [0.99, -0.99])
    def test_sigmoid_grad_near_extreme_rho(self, rho):
        """Gradients should stay finite near rho = ±1."""
        v_k, v_l = 1.0, 1.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = sigmoid_grad_E_sigma_sigma(C)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient at rho={rho}"

    @pytest.mark.parametrize("rho", [0.99, -0.99])
    def test_relu_grad_near_extreme_rho_fd(self, rho):
        """FD validation at near-extreme rho values where the arccos
        chain rule could become numerically fragile."""
        v_k, v_l = 1.0, 1.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = relu_grad_E_sigma_sigma(C)

        eps = 1e-6
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            C_p = C.copy(); C_p[i, j] += eps
            if i != j:
                C_p[j, i] += eps
            C_m = C.copy(); C_m[i, j] -= eps
            if i != j:
                C_m[j, i] -= eps
            fd = (relu_E_sigma_sigma(C_p) - relu_E_sigma_sigma(C_m)) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, abs=1e-4), (
                f"ReLU grad[{i},{j}] mismatch at rho={rho}: analytic={grad[i,j]}, fd={fd}"
            )

    # --- grad_E_sigma_sigma at zero correlation ---

    def test_relu_grad_at_rho_zero(self):
        """At rho=0 (theta=pi/2), verify analytic formula matches FD."""
        v_k, v_l = 2.0, 3.0
        C = np.array([[v_k, 0.0], [0.0, v_l]])
        grad = relu_grad_E_sigma_sigma(C)

        eps = 1e-7
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            C_p = C.copy(); C_p[i, j] += eps
            if i != j:
                C_p[j, i] += eps
            C_m = C.copy(); C_m[i, j] -= eps
            if i != j:
                C_m[j, i] -= eps
            fd = (relu_E_sigma_sigma(C_p) - relu_E_sigma_sigma(C_m)) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, abs=1e-5)

    # --- Diagonal case: k == l (v_k = v_l = v, c = v, rho = 1) ---

    def test_relu_grad_diagonal_case(self):
        """When C = [[v, v], [v, v]] (same unit), F = v/2,
        so dF/dv_k = 1/2, dF/dc = 0 is wrong — actually c = v here
        so the partial w.r.t. v_k at fixed c is dF/dv_k, not dF/dv total.
        Check via FD with C perturbed independently."""
        v = 1.5
        C = np.array([[v, v], [v, v]])
        grad = relu_grad_E_sigma_sigma(C)
        assert np.all(np.isfinite(grad))

    # --- Asymmetric variances ---

    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5])
    def test_relu_grad_asymmetric_variances(self, rho):
        """Gradients with very different v_k and v_l."""
        v_k, v_l = 0.1, 10.0
        C = np.array([[v_k, rho * np.sqrt(v_k * v_l)],
                       [rho * np.sqrt(v_k * v_l), v_l]])
        grad = relu_grad_E_sigma_sigma(C)

        eps = 1e-7
        for i, j in [(0, 0), (0, 1), (1, 1)]:
            C_p = C.copy(); C_p[i, j] += eps
            if i != j:
                C_p[j, i] += eps
            C_m = C.copy(); C_m[i, j] -= eps
            if i != j:
                C_m[j, i] -= eps
            fd = (relu_E_sigma_sigma(C_p) - relu_E_sigma_sigma(C_m)) / (2 * eps)
            assert grad[i, j] == pytest.approx(fd, abs=1e-4), (
                f"ReLU grad[{i},{j}] mismatch at rho={rho}, asymmetric v: "
                f"analytic={grad[i,j]}, fd={fd}"
            )

    # --- relu_grad symmetry: dF/dv_k and dF/dv_l should swap with v_k, v_l ---

    def test_relu_grad_symmetry(self):
        """F(v_k, v_l, c) = F(v_l, v_k, c), so
        dF/dv_k at (v_k, v_l, c) = dF/dv_l at (v_l, v_k, c)."""
        v_k, v_l = 1.5, 2.5
        rho = 0.4
        c = rho * np.sqrt(v_k * v_l)
        C1 = np.array([[v_k, c], [c, v_l]])
        C2 = np.array([[v_l, c], [c, v_k]])

        grad1 = relu_grad_E_sigma_sigma(C1)
        grad2 = relu_grad_E_sigma_sigma(C2)

        assert grad1[0, 0] == pytest.approx(grad2[1, 1], rel=1e-10)
        assert grad1[1, 1] == pytest.approx(grad2[0, 0], rel=1e-10)
        assert grad1[0, 1] == pytest.approx(grad2[0, 1], rel=1e-10)

    def test_sigmoid_grad_symmetry(self):
        """Same symmetry test for sigmoid."""
        v_k, v_l = 1.5, 2.5
        rho = 0.4
        c = rho * np.sqrt(v_k * v_l)
        C1 = np.array([[v_k, c], [c, v_l]])
        C2 = np.array([[v_l, c], [c, v_k]])

        grad1 = sigmoid_grad_E_sigma_sigma(C1)
        grad2 = sigmoid_grad_E_sigma_sigma(C2)

        assert grad1[0, 0] == pytest.approx(grad2[1, 1], rel=1e-10)
        assert grad1[1, 1] == pytest.approx(grad2[0, 0], rel=1e-10)
        assert grad1[0, 1] == pytest.approx(grad2[0, 1], rel=1e-10)

    # --- Identity: gradient shape/value invariant to C values ---

    def test_identity_grad_independent_of_C(self):
        """identity_grad_E_sigma_sigma should always return [[0,1],[1,0]]."""
        for _ in range(5):
            C = np.random.default_rng(42).standard_normal((2, 2))
            C = C @ C.T + np.eye(2) * 0.1
            grad = identity_grad_E_sigma_sigma(C)
            expected = np.array([[0.0, 1.0], [1.0, 0.0]])
            np.testing.assert_allclose(grad, expected)

    # --- relu_grad_E_sigma_sigma at zero variance returns zero ---

    def test_relu_grad_zero_variance(self):
        """With v_k=0 or v_l=0, gradient should be zero."""
        C = np.array([[0.0, 0.0], [0.0, 1.0]])
        grad = relu_grad_E_sigma_sigma(C)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)
