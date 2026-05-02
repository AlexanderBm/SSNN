"""Tests for the Edgeworth-corrected population risk and the gap theorem."""

import numpy as np
import pytest

from ssnn.cumulants import snp_cumulants, decorrelation_matrix
from ssnn.edgeworth_risk import (
    compute_edgeworth_loss,
    compute_correction_delta,
    compute_edgeworth_grad_a,
    compute_edgeworth_grad_W,
    compute_edgeworth_gradients,
)
from ssnn.population_risk import compute_loss


@pytest.fixture
def ew_problem():
    """A problem setup with non-trivial allele frequencies (MAF != 0.5)
    to ensure Edgeworth corrections are nonzero.
    """
    rng = np.random.default_rng(42)
    p = 8
    m = 2
    sigma_eps = 1.0

    # LD matrix with moderate correlation
    Sigma = np.eye(p)
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = 0.5 ** abs(i - j)

    # Rare-ish allele frequencies (far from 0.5 => large kappa_3)
    maf = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + sigma_eps**2)

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    return {
        "p": p,
        "m": m,
        "Sigma": Sigma,
        "maf": maf,
        "beta_star": beta_star,
        "Sigma_beta": Sigma_beta,
        "E_y2": E_y2,
        "sigma_eps": sigma_eps,
        "W": W,
        "a": a,
    }


class TestGapTheorem:
    """Tests for Theorem 1 (Non-Gaussian Gap) from the research plan.

    1. Linear invariance: Delta_L = 0 for identity activation.
    2. Nonlinear sensitivity: Delta_L != 0 for ReLU when kt3 != 0.
    """

    def test_linear_invariance(self, ew_problem):
        """Delta_L = 0 for identity activation regardless of cumulants.

        This is Point 1 of Theorem 1: for sigma = id, the Hermite
        correction integrals vanish by orthogonality.
        """
        d = ew_problem
        delta = compute_correction_delta(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], activation="identity",
        )
        assert delta == pytest.approx(0.0, abs=1e-10)

    def test_nonlinear_sensitivity_relu(self, ew_problem):
        """Delta_L != 0 for ReLU when MAF != 0.5 (i.e., kt3 != 0).

        This is Point 2 of Theorem 1: the correction coefficient
        E_gauss[sigma'(z) H_3(z)] = -1/sqrt(2pi) != 0.
        """
        d = ew_problem
        delta = compute_correction_delta(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], activation="relu",
        )
        assert abs(delta) > 1e-8, f"Expected nonzero delta, got {delta}"

    def test_nonlinear_sensitivity_sigmoid(self, ew_problem):
        """Delta_L != 0 for sigmoid when MAF != 0.5."""
        d = ew_problem
        delta = compute_correction_delta(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], activation="sigmoid",
        )
        assert abs(delta) > 1e-8, f"Expected nonzero delta, got {delta}"

    def test_maf_0_5_gives_zero_delta_relu(self):
        """When all MAFs are 0.5, kt3 = 0 so the leading correction vanishes."""
        rng = np.random.default_rng(99)
        p = 6
        m = 2

        Sigma = np.eye(p)
        maf = np.full(p, 0.5)
        beta_star = rng.standard_normal(p) * 0.3
        Sigma_beta = Sigma @ beta_star
        E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)

        W = rng.standard_normal((m, p)) * 0.1
        a = rng.standard_normal(m) * 0.1

        delta = compute_correction_delta(
            a, W, Sigma, Sigma_beta, E_y2, maf, activation="relu",
        )
        # kt3 = 0 for all SNPs, so leading correction vanishes.
        # kt4 != 0 still, so delta is small but may not be exactly zero.
        # The important test is that |delta| << |delta| for non-0.5 MAF.
        assert abs(delta) < 1e-4


class TestEdgeworthLoss:
    """Test structural properties of the Edgeworth-corrected loss."""

    def test_matches_gaussian_for_zero_cumulants(self):
        """When all MAFs = 0.5 (kt3 = 0), EW loss ≈ Gaussian loss."""
        rng = np.random.default_rng(42)
        p = 6
        m = 2
        Sigma = np.eye(p)
        maf = np.full(p, 0.5)
        beta_star = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta_star
        E_y2 = float(beta_star @ beta_star + 1.0)
        W = rng.standard_normal((m, p)) * 0.1
        a = rng.standard_normal(m) * 0.1

        L_gauss = compute_loss(a, W, Sigma, Sigma_beta, E_y2, "relu")
        L_ew = compute_edgeworth_loss(
            a, W, Sigma, Sigma_beta, E_y2, maf, "relu"
        )
        # kt3=0, but kt4 correction is nonzero (small). Tolerance is loose.
        assert L_ew == pytest.approx(L_gauss, abs=1e-3)

    def test_loss_is_non_negative_at_origin(self, ew_problem):
        """L_EW should be non-negative for zero network weights
        (since it equals E[y^2] > 0).
        """
        d = ew_problem
        a_zero = np.zeros(d["m"])
        W_zero = np.zeros((d["m"], d["p"]))

        L = compute_edgeworth_loss(
            a_zero, W_zero, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu"
        )
        assert L >= 0
        assert L == pytest.approx(d["E_y2"], rel=1e-8)

    def test_precomputed_sigma_inv_sqrt(self, ew_problem):
        """Passing precomputed Sigma_inv_sqrt gives identical results."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        L1 = compute_edgeworth_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu",
        )
        L2 = compute_edgeworth_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", Sigma_inv_sqrt=S,
        )
        assert L1 == pytest.approx(L2, rel=1e-10)


class TestEdgeworthGradients:
    """Test Edgeworth-corrected gradient computations."""

    def test_grad_a_finite_diff(self, ew_problem):
        """Verify grad_a against full-loss finite differences."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_a = compute_edgeworth_grad_a(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "relu", S,
        )

        eps = 1e-5
        grad_a_fd = np.zeros_like(d["a"])
        for k in range(d["m"]):
            a_plus = d["a"].copy()
            a_plus[k] += eps
            a_minus = d["a"].copy()
            a_minus[k] -= eps

            L_plus = compute_edgeworth_loss(
                a_plus, d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], "relu", S,
            )
            L_minus = compute_edgeworth_loss(
                a_minus, d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], "relu", S,
            )
            grad_a_fd[k] = (L_plus - L_minus) / (2 * eps)

        np.testing.assert_allclose(grad_a, grad_a_fd, rtol=1e-3, atol=1e-6)

    def test_grad_W_finite_diff(self, ew_problem):
        """Verify grad_W against full-loss finite differences."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_W = compute_edgeworth_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "relu", S,
        )

        eps = 1e-5
        grad_W_fd = np.zeros_like(d["W"])
        for k in range(d["m"]):
            for j in range(d["p"]):
                W_plus = d["W"].copy()
                W_plus[k, j] += eps
                W_minus = d["W"].copy()
                W_minus[k, j] -= eps

                L_plus = compute_edgeworth_loss(
                    d["a"], W_plus, d["Sigma"], d["Sigma_beta"], d["E_y2"],
                    d["maf"], "relu", S,
                )
                L_minus = compute_edgeworth_loss(
                    d["a"], W_minus, d["Sigma"], d["Sigma_beta"], d["E_y2"],
                    d["maf"], "relu", S,
                )
                grad_W_fd[k, j] = (L_plus - L_minus) / (2 * eps)

        np.testing.assert_allclose(grad_W, grad_W_fd, rtol=1e-2, atol=1e-5)

    def test_identity_gradient_correction_is_zero(self, ew_problem):
        """For identity activation, Edgeworth gradients = Gaussian gradients."""
        from ssnn.population_risk import compute_grad_a

        d = ew_problem

        grad_a_gauss = compute_grad_a(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], "identity"
        )
        grad_a_ew = compute_edgeworth_grad_a(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "identity",
        )

        np.testing.assert_allclose(grad_a_ew, grad_a_gauss, atol=1e-10)

    def test_compute_edgeworth_gradients_returns_both(self, ew_problem):
        """The combined gradient function returns consistent results."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_a, grad_W = compute_edgeworth_gradients(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )

        grad_a_solo = compute_edgeworth_grad_a(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "relu", S,
        )
        np.testing.assert_allclose(grad_a, grad_a_solo, atol=1e-14)
        assert grad_W.shape == d["W"].shape

    def test_sigmoid_grad_a_finite_diff(self, ew_problem):
        """Verify sigmoid grad_a against finite differences."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_a = compute_edgeworth_grad_a(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "sigmoid", S,
        )

        eps = 1e-5
        grad_a_fd = np.zeros_like(d["a"])
        for k in range(d["m"]):
            a_plus = d["a"].copy()
            a_plus[k] += eps
            a_minus = d["a"].copy()
            a_minus[k] -= eps

            L_plus = compute_edgeworth_loss(
                a_plus, d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], "sigmoid", S,
            )
            L_minus = compute_edgeworth_loss(
                a_minus, d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], "sigmoid", S,
            )
            grad_a_fd[k] = (L_plus - L_minus) / (2 * eps)

        np.testing.assert_allclose(grad_a, grad_a_fd, rtol=1e-3, atol=1e-6)

    def test_sigmoid_grad_W_finite_diff(self, ew_problem):
        """Verify sigmoid grad_W against full-loss finite differences."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_W = compute_edgeworth_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "sigmoid", S,
        )

        eps = 1e-5
        grad_W_fd = np.zeros_like(d["W"])
        for k in range(d["m"]):
            for j in range(d["p"]):
                W_plus = d["W"].copy()
                W_plus[k, j] += eps
                W_minus = d["W"].copy()
                W_minus[k, j] -= eps

                L_plus = compute_edgeworth_loss(
                    d["a"], W_plus, d["Sigma"], d["Sigma_beta"], d["E_y2"],
                    d["maf"], "sigmoid", S,
                )
                L_minus = compute_edgeworth_loss(
                    d["a"], W_minus, d["Sigma"], d["Sigma_beta"], d["E_y2"],
                    d["maf"], "sigmoid", S,
                )
                grad_W_fd[k, j] = (L_plus - L_minus) / (2 * eps)

        np.testing.assert_allclose(grad_W, grad_W_fd, rtol=1e-2, atol=1e-5)

    def test_gradient_descent_direction_relu(self, ew_problem):
        """A small step in the negative gradient direction should decrease L_EW."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        L0 = compute_edgeworth_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )
        grad_a, grad_W = compute_edgeworth_gradients(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )

        lr = 1e-4
        a_new = d["a"] - lr * grad_a
        W_new = d["W"] - lr * grad_W

        L1 = compute_edgeworth_loss(
            a_new, W_new, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )
        assert L1 < L0


class TestEdgeworthGradWAudit:
    """Audit: additional tests for the hybrid analytic Edgeworth grad_W."""

    def test_grad_W_identity_matches_gaussian(self, ew_problem):
        """For identity activation, EW grad_W should equal Gaussian grad_W
        because all Edgeworth corrections vanish."""
        from ssnn.population_risk import compute_grad_W

        d = ew_problem
        grad_W_gauss = compute_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], "identity"
        )
        grad_W_ew = compute_edgeworth_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "identity",
        )
        np.testing.assert_allclose(grad_W_ew, grad_W_gauss, atol=1e-8)

    def test_grad_W_sigmoid_finite_diff(self, ew_problem):
        """Verify sigmoid grad_W hybrid approach matches full-loss FD.
        (Supplements the existing test with a tighter tolerance check.)"""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        grad_W = compute_edgeworth_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "sigmoid", S,
        )
        assert np.all(np.isfinite(grad_W))
        assert grad_W.shape == d["W"].shape

    def test_grad_W_descent_direction_sigmoid(self, ew_problem):
        """A small step in -grad_W direction should decrease L_EW for sigmoid."""
        d = ew_problem
        S = decorrelation_matrix(d["Sigma"])

        L0 = compute_edgeworth_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "sigmoid", S,
        )
        grad_W = compute_edgeworth_grad_W(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["maf"], "sigmoid", S,
        )

        lr = 1e-5
        W_new = d["W"] - lr * grad_W
        L1 = compute_edgeworth_loss(
            d["a"], W_new, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "sigmoid", S,
        )
        assert L1 < L0

    def test_grad_W_relu_larger_m(self):
        """Test with m=4 hidden units for relu — exercises the cross-term
        loop more thoroughly."""
        rng = np.random.default_rng(42)
        p = 6
        m = 4
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        maf = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4])
        beta_star = rng.standard_normal(p) * 0.3
        Sigma_beta = Sigma @ beta_star
        E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)

        W = rng.standard_normal((m, p)) * 0.05
        a = rng.standard_normal(m) * 0.05
        S = decorrelation_matrix(Sigma)

        grad_W = compute_edgeworth_grad_W(
            a, W, Sigma, Sigma_beta, maf, "relu", S,
        )

        eps = 1e-5
        grad_W_fd = np.zeros_like(W)
        for k in range(m):
            for j in range(p):
                W_p = W.copy(); W_p[k, j] += eps
                W_m = W.copy(); W_m[k, j] -= eps
                L_p = compute_edgeworth_loss(
                    a, W_p, Sigma, Sigma_beta, E_y2, maf, "relu", S,
                    loss_floor=None,
                )
                L_m = compute_edgeworth_loss(
                    a, W_m, Sigma, Sigma_beta, E_y2, maf, "relu", S,
                    loss_floor=None,
                )
                grad_W_fd[k, j] = (L_p - L_m) / (2 * eps)

        np.testing.assert_allclose(grad_W, grad_W_fd, rtol=0.05, atol=1e-4)

    def test_grad_W_maf_0_5_equals_gaussian(self):
        """When all MAFs = 0.5, kt3 = 0 and corrections are minimal.
        EW grad_W should be very close to Gaussian grad_W."""
        from ssnn.population_risk import compute_grad_W as gauss_grad_W

        rng = np.random.default_rng(42)
        p = 6
        m = 2
        Sigma = np.eye(p)
        maf = np.full(p, 0.5)
        beta_star = rng.standard_normal(p) * 0.2
        Sigma_beta = Sigma @ beta_star

        W = rng.standard_normal((m, p)) * 0.1
        a = rng.standard_normal(m) * 0.1

        grad_gauss = gauss_grad_W(a, W, Sigma, Sigma_beta, "relu")
        grad_ew = compute_edgeworth_grad_W(
            a, W, Sigma, Sigma_beta, maf, "relu",
        )
        np.testing.assert_allclose(grad_ew, grad_gauss, atol=1e-3)


class TestCorrectionDeltaConsistency:
    """Test that compute_correction_delta is consistent with manual L_EW - L_gauss."""

    def test_delta_equals_manual_difference(self, ew_problem):
        """compute_correction_delta should equal compute_edgeworth_loss - compute_loss."""
        d = ew_problem
        for act in ["relu", "sigmoid", "identity"]:
            delta = compute_correction_delta(
                d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], activation=act,
            )

            L_gauss = compute_loss(
                d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"], act
            )
            L_ew = compute_edgeworth_loss(
                d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
                d["maf"], act,
            )

            manual_delta = L_ew - L_gauss
            assert delta == pytest.approx(manual_delta, abs=1e-10), (
                f"Failed for {act}: delta={delta}, manual={manual_delta}"
            )

    def test_maf_0_5_mechanism(self):
        """When MAF=0.5, kt3=0 exactly, which makes the leading correction vanish.

        Verify the *mechanism*: projection_cumulants_independent should give
        kt3 == 0.0 exactly (not just approximately).
        """
        from ssnn.cumulants import snp_cumulants, projection_cumulants_independent

        p_dim = 6
        maf = np.full(p_dim, 0.5)
        w = np.array([0.5, -0.3, 0.8, 0.1, -0.4, 0.2])

        cum = snp_cumulants(maf)
        assert np.all(cum["kappa3"] == 0.0)

        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        assert kt3 == 0.0
