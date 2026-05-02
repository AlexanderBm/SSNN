"""Tests for Edgeworth-corrected activation expectations."""

import numpy as np
import pytest
from scipy.stats import norm

from ssnn.edgeworth_integrals import (
    hermite_3,
    hermite_4,
    hermite_6,
    edgeworth_E_sigma,
    edgeworth_E_sigma_prime,
    edgeworth_E_sigma_sigma,
    _relu_E_sigma_H3,
    _relu_E_sigma_H4,
    _relu_E_sigma_H6,
    _relu_E_sigma_prime_H3,
    _relu_E_sigma_prime_H4,
    _relu_E_sigma_prime_H6,
    _sigmoid_E_sigma_H3,
    _sigmoid_E_sigma_H4,
    _sigmoid_E_sigma_H6,
    _sigmoid_E_sigma_prime_H3,
    _sigmoid_E_sigma_prime_H4,
    _sigmoid_E_sigma_prime_H6,
    _gauss_hermite_expect,
    _sigmoid_approx,
    _sigmoid_approx_prime,
    _LAMBDA,
)


class TestHermitePolynomials:
    """Verify Hermite polynomial values at known points."""

    def test_H3_at_zero(self):
        assert hermite_3(0.0) == pytest.approx(0.0, abs=1e-14)

    def test_H4_at_zero(self):
        assert hermite_4(0.0) == pytest.approx(3.0, abs=1e-14)

    def test_H6_at_zero(self):
        assert hermite_6(0.0) == pytest.approx(-15.0, abs=1e-14)

    def test_H3_orthogonality(self):
        """E[H_3(z) H_3(z)] = 3! = 6 for z ~ N(0,1)."""
        n = 5_000_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        emp = np.mean(hermite_3(z)**2)
        assert emp == pytest.approx(6.0, rel=0.05)

    def test_H4_orthogonality(self):
        """E[H_4(z)^2] = 4! = 24."""
        n = 5_000_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        emp = np.mean(hermite_4(z)**2)
        assert emp == pytest.approx(24.0, rel=0.1)

    def test_H3_H4_orthogonality(self):
        """E[H_3(z) H_4(z)] = 0 (orthogonality of different orders)."""
        n = 5_000_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        emp = np.mean(hermite_3(z) * hermite_4(z))
        assert emp == pytest.approx(0.0, abs=0.15)

    def test_hermite_recurrence_relation(self):
        """H_{n+1}(x) = x H_n(x) - n H_{n-1}(x) for probabilist's Hermite.

        Test H_4(x) = x H_3(x) - 3 H_2(x) where H_2(x) = x^2 - 1
        and  H_6(x) can be verified via H_6 = x H_5 - 5 H_4 where
        H_5(x) = x^5 - 10x^3 + 15x.
        """
        x_vals = np.linspace(-3, 3, 50)

        H2 = x_vals**2 - 1.0
        H3 = hermite_3(x_vals)
        H4 = hermite_4(x_vals)
        H5 = x_vals**5 - 10*x_vals**3 + 15*x_vals
        H6 = hermite_6(x_vals)

        np.testing.assert_allclose(H4, x_vals * H3 - 3.0 * H2, atol=1e-10)
        np.testing.assert_allclose(H6, x_vals * H5 - 5.0 * H4, atol=1e-10)

    def test_H6_orthogonality(self):
        """E[H_6(z)^2] = 6! = 720.

        H_6^2 has very high variance (involves z^12), requiring many samples.
        """
        n = 20_000_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        emp = np.mean(hermite_6(z)**2)
        assert emp == pytest.approx(720.0, rel=0.15)

    def test_hermite_mean_zero(self):
        """E[H_k(z)] = 0 for k >= 1, z ~ N(0,1)."""
        n = 5_000_000
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        assert np.mean(hermite_3(z)) == pytest.approx(0.0, abs=0.05)
        assert np.mean(hermite_4(z)) == pytest.approx(0.0, abs=0.1)
        assert np.mean(hermite_6(z)) == pytest.approx(0.0, abs=0.5)


class TestReLUCorrectionIntegrals:
    """Test ReLU correction integrals against Monte Carlo estimates."""

    @pytest.fixture
    def mc_samples(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(2_000_000)

    def test_E_sigma_H3(self, mc_samples):
        z = mc_samples
        relu_z = np.maximum(0, z)
        emp = np.mean(relu_z * hermite_3(z))
        assert _relu_E_sigma_H3() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_H4(self, mc_samples):
        z = mc_samples
        relu_z = np.maximum(0, z)
        emp = np.mean(relu_z * hermite_4(z))
        assert _relu_E_sigma_H4() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_H6(self, mc_samples):
        z = mc_samples
        relu_z = np.maximum(0, z)
        emp = np.mean(relu_z * hermite_6(z))
        assert _relu_E_sigma_H6() == pytest.approx(emp, abs=0.15)

    def test_E_sigma_prime_H3(self, mc_samples):
        z = mc_samples
        indicator = (z > 0).astype(float)
        emp = np.mean(indicator * hermite_3(z))
        assert _relu_E_sigma_prime_H3() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_prime_H4(self, mc_samples):
        z = mc_samples
        indicator = (z > 0).astype(float)
        emp = np.mean(indicator * hermite_4(z))
        assert _relu_E_sigma_prime_H4() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_prime_H6(self, mc_samples):
        z = mc_samples
        indicator = (z > 0).astype(float)
        emp = np.mean(indicator * hermite_6(z))
        assert _relu_E_sigma_prime_H6() == pytest.approx(emp, abs=0.05)

    def test_relu_exact_values(self):
        """Verify exact closed-form values for ReLU correction integrals."""
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        assert _relu_E_sigma_H3() == 0.0
        assert _relu_E_sigma_H4() == pytest.approx(-inv_sqrt_2pi, rel=1e-12)
        assert _relu_E_sigma_H6() == pytest.approx(3.0 * inv_sqrt_2pi, rel=1e-12)
        assert _relu_E_sigma_prime_H3() == pytest.approx(-inv_sqrt_2pi, rel=1e-12)
        assert _relu_E_sigma_prime_H4() == 0.0
        assert _relu_E_sigma_prime_H6() == 0.0


class TestSigmoidCorrectionIntegrals:
    """Test sigmoid correction integrals against MC of the probit approximation."""

    @pytest.fixture
    def mc_samples(self):
        rng = np.random.default_rng(55)
        return rng.standard_normal(2_000_000)

    def test_E_sigma_H3(self, mc_samples):
        z = mc_samples
        sig_z = norm.cdf(_LAMBDA * z)
        emp = np.mean(sig_z * hermite_3(z))
        assert _sigmoid_E_sigma_H3() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_H4(self, mc_samples):
        z = mc_samples
        sig_z = norm.cdf(_LAMBDA * z)
        emp = np.mean(sig_z * hermite_4(z))
        assert _sigmoid_E_sigma_H4() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_H6(self, mc_samples):
        z = mc_samples
        sig_z = norm.cdf(_LAMBDA * z)
        emp = np.mean(sig_z * hermite_6(z))
        assert _sigmoid_E_sigma_H6() == pytest.approx(emp, abs=0.15)

    def test_E_sigma_prime_H3(self, mc_samples):
        z = mc_samples
        sig_prime_z = _LAMBDA * norm.pdf(_LAMBDA * z)
        emp = np.mean(sig_prime_z * hermite_3(z))
        assert _sigmoid_E_sigma_prime_H3() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_prime_H4(self, mc_samples):
        z = mc_samples
        sig_prime_z = _LAMBDA * norm.pdf(_LAMBDA * z)
        emp = np.mean(sig_prime_z * hermite_4(z))
        assert _sigmoid_E_sigma_prime_H4() == pytest.approx(emp, abs=0.01)

    def test_E_sigma_prime_H6(self, mc_samples):
        z = mc_samples
        sig_prime_z = _LAMBDA * norm.pdf(_LAMBDA * z)
        emp = np.mean(sig_prime_z * hermite_6(z))
        assert _sigmoid_E_sigma_prime_H6() == pytest.approx(emp, abs=0.05)

    def test_sigmoid_correction_integrals_structure(self):
        """E[sigmoid(z) H_3(z)] != 0 (sigmoid is not symmetric about 1/2).

        E[sigmoid'(z) H_3(z)] ~ 0 because the probit-approximated sigmoid'
        is lambda*phi(lambda*z), which is symmetric (even function), while
        H_3 is antisymmetric (odd function), making the integral vanish.
        """
        assert abs(_sigmoid_E_sigma_H3()) > 1e-6
        assert abs(_sigmoid_E_sigma_prime_H3()) < 1e-10


class TestEdgeworthCorrectedExpectations:
    """Test the Edgeworth-corrected E[sigma] and E[sigma'] functions."""

    def test_zero_cumulants_recovers_gaussian(self):
        """With kt3 = kt4 = 0, Edgeworth result = Gaussian result."""
        from ssnn.activations import get_activation

        for act in ["relu", "sigmoid", "identity"]:
            E_sigma_gauss, _, _ = get_activation(act)
            v = 1.5

            ew = edgeworth_E_sigma(v, 0.0, 0.0, act)
            gauss = E_sigma_gauss(v)
            assert ew == pytest.approx(gauss, abs=1e-10), f"Failed for {act}"

    def test_zero_cumulants_sigma_prime_recovers_gaussian(self):
        from ssnn.activations import get_activation

        for act in ["relu", "sigmoid", "identity"]:
            _, E_sigma_prime_gauss, _ = get_activation(act)
            v = 2.0

            ew = edgeworth_E_sigma_prime(v, 0.0, 0.0, act)
            gauss = E_sigma_prime_gauss(v)
            assert ew == pytest.approx(gauss, abs=1e-10), f"Failed for {act}"

    def test_identity_correction_is_zero(self):
        """For identity activation, corrections vanish for any cumulants."""
        from ssnn.activations import get_activation
        E_sigma_gauss, _, _ = get_activation("identity")

        v = 1.0
        ew = edgeworth_E_sigma(v, 0.5, 0.3, "identity")
        gauss = E_sigma_gauss(v)
        assert ew == pytest.approx(gauss, abs=1e-14)

    def test_identity_sigma_prime_correction_is_zero(self):
        """For identity sigma', all Hermite correction integrals vanish."""
        from ssnn.activations import get_activation
        _, E_sp_gauss, _ = get_activation("identity")

        v = 2.0
        ew = edgeworth_E_sigma_prime(v, 1.0, 0.5, "identity")
        gauss = E_sp_gauss(v)
        assert ew == pytest.approx(gauss, abs=1e-14)

    def test_relu_correction_nonzero_for_kt3(self):
        """For ReLU with nonzero kt3, correction should be nonzero."""
        from ssnn.activations import get_activation
        E_sigma_prime_gauss, = [get_activation("relu")[1]]

        v = 1.0
        kt3 = 0.5
        ew = edgeworth_E_sigma_prime(v, kt3, 0.0, "relu")
        gauss = E_sigma_prime_gauss(v)

        # The correction coefficient for sigma' H_3 is -1/sqrt(2pi) != 0
        assert ew != pytest.approx(gauss, abs=1e-10)

    def test_edgeworth_E_sigma_sigma_zero_cumulants(self):
        """With zero cumulants, cross-term matches Gaussian."""
        from ssnn.activations import get_activation

        for act in ["relu", "sigmoid", "identity"]:
            _, _, E_ss_gauss = get_activation(act)

            C = np.array([[1.0, 0.3], [0.3, 1.5]])
            ew = edgeworth_E_sigma_sigma(C, 0.0, 0.0, 0.0, 0.0, act)
            gauss = E_ss_gauss(C)
            assert ew == pytest.approx(gauss, abs=1e-8), f"Failed for {act}"

    def test_relu_E_sigma_prime_H3_is_minus_one_over_sqrt2pi(self):
        """This is the key number from the research plan: -1/sqrt(2pi)."""
        expected = -1.0 / np.sqrt(2.0 * np.pi)
        assert _relu_E_sigma_prime_H3() == pytest.approx(expected, rel=1e-12)

    def test_full_expansion_three_terms(self):
        """Verify all three Edgeworth terms (kt3/6, kt4/24, kt3^2/72)
        contribute correctly for ReLU E[sigma(z)].

        Since E[ReLU(z) H_3(z)] = 0, the kt3 linear term vanishes for
        E[sigma]. But the kt4 and kt3^2 terms are nonzero.
        """
        from ssnn.activations import get_activation
        E_sigma_gauss, _, _ = get_activation("relu")

        v = 1.0
        kt3 = 0.3
        kt4 = 0.1

        gauss = E_sigma_gauss(v)
        full = edgeworth_E_sigma(v, kt3, kt4, "relu")

        expected_correction = (
            (kt3 / 6.0) * _relu_E_sigma_H3()
            + (kt4 / 24.0) * _relu_E_sigma_H4()
            + (kt3**2 / 72.0) * _relu_E_sigma_H6()
        )

        assert full - gauss == pytest.approx(expected_correction, rel=1e-10)

        assert _relu_E_sigma_H3() == 0.0
        assert _relu_E_sigma_H4() != 0.0
        assert _relu_E_sigma_H6() != 0.0

    def test_full_expansion_sigma_prime(self):
        """Verify all three terms for E[sigma'(z)] with ReLU.

        E[1(z>0) H_3(z)] = -1/sqrt(2pi) != 0, so the kt3 term is nonzero.
        E[1(z>0) H_4(z)] = 0, E[1(z>0) H_6(z)] = 0 for ReLU.
        """
        from ssnn.activations import get_activation
        _, E_sp_gauss, _ = get_activation("relu")

        v = 1.0
        kt3 = 0.4
        kt4 = 0.2

        gauss = E_sp_gauss(v)
        full = edgeworth_E_sigma_prime(v, kt3, kt4, "relu")

        expected_correction = (
            (kt3 / 6.0) * _relu_E_sigma_prime_H3()
            + (kt4 / 24.0) * _relu_E_sigma_prime_H4()
            + (kt3**2 / 72.0) * _relu_E_sigma_prime_H6()
        )

        assert full - gauss == pytest.approx(expected_correction, rel=1e-10)

        assert _relu_E_sigma_prime_H3() != 0.0
        assert _relu_E_sigma_prime_H4() == 0.0
        assert _relu_E_sigma_prime_H6() == 0.0

    def test_sigmoid_correction_nonzero(self):
        """Sigmoid with nonzero kt3 should produce a different result from Gaussian."""
        from ssnn.activations import get_activation
        _, E_sp_gauss, _ = get_activation("sigmoid")

        v = 1.0
        ew = edgeworth_E_sigma_prime(v, 0.5, 0.0, "sigmoid")
        gauss = E_sp_gauss(v)
        assert ew != pytest.approx(gauss, abs=1e-10)

    def test_edgeworth_E_sigma_sigma_relu_mc(self):
        """Test edgeworth_E_sigma_sigma for ReLU against Monte Carlo using
        actual Binomial genotypes, verifying the cross-term correction
        independently."""
        rng = np.random.default_rng(333)
        p_dim = 15
        n_samples = 1_000_000
        mafs = np.full(p_dim, 0.1)

        X = np.zeros((n_samples, p_dim))
        for j in range(p_dim):
            X[:, j] = rng.binomial(2, mafs[j], size=n_samples) - 2 * mafs[j]

        w_k = rng.standard_normal(p_dim) * 0.3
        w_l = rng.standard_normal(p_dim) * 0.3

        z_k = X @ w_k
        z_l = X @ w_l

        z_k_std = z_k / np.std(z_k)
        z_l_std = z_l / np.std(z_l)

        true_E_ss = np.mean(np.maximum(0, z_k_std) * np.maximum(0, z_l_std))

        from ssnn.cumulants import snp_cumulants, projection_cumulants_independent
        cum = snp_cumulants(mafs)
        kt3_k, kt4_k = projection_cumulants_independent(
            w_k, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        kt3_l, kt4_l = projection_cumulants_independent(
            w_l, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        rho = np.corrcoef(z_k, z_l)[0, 1]
        C = np.array([[1.0, rho], [rho, 1.0]])

        ew_E_ss = edgeworth_E_sigma_sigma(C, kt3_k, kt4_k, kt3_l, kt4_l, "relu")

        from ssnn.activations import get_activation
        _, _, gauss_E_ss = get_activation("relu")
        gauss = gauss_E_ss(C)

        gauss_err = abs(true_E_ss - gauss)
        ew_err = abs(true_E_ss - ew_E_ss)
        assert ew_err < gauss_err + 0.01, (
            f"Edgeworth cross-term should not be much worse: "
            f"gauss_err={gauss_err:.5f}, ew_err={ew_err:.5f}"
        )


class TestEdgeworthMonteCarlo:
    """Verify Edgeworth corrections against Monte Carlo with non-Gaussian data."""

    def test_relu_correction_direction(self):
        """Edgeworth-corrected E[ReLU(z)] should be closer to the
        true non-Gaussian expectation than the Gaussian estimate.

        The Edgeworth expansion is valid for sums of many independent
        variables (CLT regime), so we use a projection z = w^T x
        across multiple rare-variant SNPs.
        """
        rng = np.random.default_rng(123)
        n_samples = 500_000

        # 20 independent SNPs with rare allele frequency
        p_dim = 20
        maf_val = 0.1
        mafs = np.full(p_dim, maf_val)
        w = rng.standard_normal(p_dim) * 0.5

        # Simulate centered genotypes
        X = np.zeros((n_samples, p_dim))
        for j in range(p_dim):
            X[:, j] = rng.binomial(2, mafs[j], size=n_samples) - 2 * mafs[j]

        z_raw = X @ w
        var_z = np.var(z_raw)
        z = z_raw / np.sqrt(var_z)

        true_E_relu = np.mean(np.maximum(0, z))

        gauss_E = 1.0 / np.sqrt(2 * np.pi)

        from ssnn.cumulants import snp_cumulants, projection_cumulants_independent
        cum = snp_cumulants(mafs)
        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        ew_E = edgeworth_E_sigma(1.0, kt3, kt4, "relu")

        gauss_err = abs(true_E_relu - gauss_E)
        ew_err = abs(true_E_relu - ew_E)
        assert ew_err < gauss_err, (
            f"Edgeworth should improve: gauss_err={gauss_err:.6f}, ew_err={ew_err:.6f}"
        )

    def test_sigmoid_correction_direction(self):
        """Edgeworth-corrected E[sigmoid'(z)] should be closer to
        the true non-Gaussian expectation than the Gaussian estimate."""
        rng = np.random.default_rng(456)
        n_samples = 500_000

        p_dim = 20
        mafs = np.full(p_dim, 0.1)
        w = rng.standard_normal(p_dim) * 0.5

        X = np.zeros((n_samples, p_dim))
        for j in range(p_dim):
            X[:, j] = rng.binomial(2, mafs[j], size=n_samples) - 2 * mafs[j]

        z_raw = X @ w
        var_z = np.var(z_raw)
        z = z_raw / np.sqrt(var_z)

        true_E_sp = np.mean(_LAMBDA * norm.pdf(_LAMBDA * z))

        from ssnn.activations import get_activation
        _, gauss_E_sp, _ = get_activation("sigmoid")
        gauss = gauss_E_sp(1.0)

        from ssnn.cumulants import snp_cumulants, projection_cumulants_independent
        cum = snp_cumulants(mafs)
        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        ew = edgeworth_E_sigma_prime(1.0, kt3, kt4, "sigmoid")

        gauss_err = abs(true_E_sp - gauss)
        ew_err = abs(true_E_sp - ew)
        assert ew_err < gauss_err + 0.005, (
            f"Edgeworth should not be worse: gauss_err={gauss_err:.6f}, ew_err={ew_err:.6f}"
        )
