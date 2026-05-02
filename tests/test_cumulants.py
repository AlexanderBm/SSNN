"""Tests for genotype cumulant computations."""

import numpy as np
import pytest

from ssnn.cumulants import (
    snp_cumulants,
    projection_cumulants_independent,
    projection_cumulants_ld,
    decorrelation_matrix,
)


class TestSNPCumulants:
    """Test per-SNP cumulant formulas against the Binomial(2, p) distribution."""

    def test_maf_0_5_symmetric(self):
        """At MAF = 0.5, kappa_3 = 0 (symmetric distribution)."""
        cum = snp_cumulants(np.array([0.5]))
        assert cum["kappa2"][0] == pytest.approx(0.5, abs=1e-14)
        assert cum["kappa3"][0] == pytest.approx(0.0, abs=1e-14)

    def test_maf_0_1_rare_variant(self):
        """Check cumulants for a rare variant (MAF = 0.1)."""
        p = 0.1
        q = 0.9
        cum = snp_cumulants(np.array([p]))
        assert cum["kappa2"][0] == pytest.approx(2 * p * q, rel=1e-12)
        assert cum["kappa3"][0] == pytest.approx(2 * p * q * (1 - 2*p), rel=1e-12)
        assert cum["kappa4"][0] == pytest.approx(2 * p * q * (1 - 6*p*q), rel=1e-12)

    def test_kappa3_positive_for_rare_alleles(self):
        """For MAF < 0.5, kappa_3 > 0 (right-skewed)."""
        cum = snp_cumulants(np.array([0.05, 0.1, 0.2, 0.3, 0.4]))
        assert np.all(cum["kappa3"][:4] > 0)  # All MAF < 0.5

    def test_kappa3_negative_for_common_alleles(self):
        """For MAF > 0.5, kappa_3 < 0 (left-skewed).

        Strictly, MAF is in (0, 0.5] by convention, but the formula
        is valid for any p in (0, 1).
        """
        cum = snp_cumulants(np.array([0.6, 0.7, 0.8, 0.9]))
        assert np.all(cum["kappa3"] < 0)

    def test_monte_carlo_moments(self):
        """Verify cumulants against empirical moments of Binomial(2, p)."""
        rng = np.random.default_rng(42)
        maf = np.array([0.05, 0.2, 0.35])
        n_samples = 500_000

        for j, p_val in enumerate(maf):
            x = rng.binomial(2, p_val, size=n_samples).astype(float)
            x_centered = x - 2 * p_val

            emp_var = np.var(x_centered, ddof=0)
            emp_skew_num = np.mean(x_centered**3)

            cum = snp_cumulants(np.array([p_val]))
            assert cum["kappa2"][0] == pytest.approx(emp_var, rel=0.02)
            assert cum["kappa3"][0] == pytest.approx(emp_skew_num, abs=0.005)

    def test_monte_carlo_kappa4(self):
        """Verify kappa4 against empirical fourth cumulant of Binomial(2, p).

        kappa4 = mu4 - 3*sigma^4 where mu4 is the centered fourth moment.
        """
        rng = np.random.default_rng(77)
        n_samples = 1_000_000

        for p_val in [0.05, 0.15, 0.3, 0.45]:
            x = rng.binomial(2, p_val, size=n_samples).astype(float)
            x_centered = x - 2 * p_val

            emp_mu4 = np.mean(x_centered**4)
            emp_var = np.var(x_centered, ddof=0)
            emp_kappa4 = emp_mu4 - 3.0 * emp_var**2

            cum = snp_cumulants(np.array([p_val]))
            assert cum["kappa4"][0] == pytest.approx(emp_kappa4, abs=0.01)

    def test_maf_0_5_kappa4_nonzero(self):
        """At MAF=0.5, kappa3=0 but kappa4 is nonzero (discrete distribution)."""
        cum = snp_cumulants(np.array([0.5]))
        assert cum["kappa3"][0] == pytest.approx(0.0, abs=1e-14)
        expected_k4 = 2 * 0.5 * 0.5 * (1.0 - 6.0 * 0.5 * 0.5)
        assert cum["kappa4"][0] == pytest.approx(expected_k4, abs=1e-14)
        assert cum["kappa4"][0] != 0.0

    def test_extreme_rare_maf(self):
        """Very rare variant (MAF ~ 0.01): large skewness, kappa3 ~ 2p."""
        p_val = 0.01
        cum = snp_cumulants(np.array([p_val]))
        assert cum["kappa3"][0] > 0
        assert cum["kappa3"][0] == pytest.approx(2 * p_val * (1 - p_val) * (1 - 2*p_val), rel=1e-12)
        assert cum["kappa3"][0] / cum["kappa2"][0]**1.5 > 1.0

    def test_near_boundary_maf(self):
        """MAFs very close to 0 and 1 should not produce NaN or Inf."""
        for p_val in [0.001, 0.999]:
            cum = snp_cumulants(np.array([p_val]))
            for key in ["kappa2", "kappa3", "kappa4"]:
                assert np.isfinite(cum[key][0]), f"Non-finite {key} at MAF={p_val}"

    def test_vectorized_multiple_snps(self):
        """Vectorized computation matches per-SNP computation."""
        mafs = np.array([0.05, 0.1, 0.2, 0.35, 0.5])
        cum_vec = snp_cumulants(mafs)

        for j, p_val in enumerate(mafs):
            cum_single = snp_cumulants(np.array([p_val]))
            assert cum_vec["kappa2"][j] == pytest.approx(cum_single["kappa2"][0], abs=1e-14)
            assert cum_vec["kappa3"][j] == pytest.approx(cum_single["kappa3"][0], abs=1e-14)
            assert cum_vec["kappa4"][j] == pytest.approx(cum_single["kappa4"][0], abs=1e-14)


class TestProjectionCumulantsIndependent:
    """Test projection cumulants for independent SNPs."""

    def test_uniform_weights_equal_maf(self):
        """When all MAFs are equal and weights are uniform, kappa3
        should equal the single-SNP standardized skewness."""
        p_val = 0.2
        p_dim = 10
        maf = np.full(p_dim, p_val)
        w = np.ones(p_dim)
        cum = snp_cumulants(maf)

        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        # For uniform weights and equal MAF: kt3 = kappa3_j / kappa2_j^{3/2} / sqrt(p)
        single_std_skew = cum["kappa3"][0] / cum["kappa2"][0]**1.5
        assert kt3 == pytest.approx(single_std_skew / np.sqrt(p_dim), rel=1e-10)

    def test_zero_weights_give_zero_cumulants(self):
        """Zero weight vector should give zero cumulants."""
        w = np.zeros(5)
        kappa2 = np.ones(5) * 0.5
        kappa3 = np.ones(5) * 0.1
        kappa4 = np.ones(5) * 0.05

        kt3, kt4 = projection_cumulants_independent(w, kappa2, kappa3, kappa4)
        assert kt3 == 0.0
        assert kt4 == 0.0

    def test_maf_0_5_gives_zero_kt3(self):
        """When all MAFs are 0.5, kappa_3 = 0 so kt3 = 0."""
        p_dim = 10
        maf = np.full(p_dim, 0.5)
        w = np.random.default_rng(42).standard_normal(p_dim)
        cum = snp_cumulants(maf)

        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        assert kt3 == pytest.approx(0.0, abs=1e-14)

    def test_single_snp_recovers_marginal(self):
        """With p=1 and w=1, projection cumulants = marginal cumulants."""
        maf = np.array([0.15])
        w = np.array([1.0])
        cum = snp_cumulants(maf)

        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        expected_kt3 = cum["kappa3"][0] / cum["kappa2"][0]**1.5
        expected_kt4 = cum["kappa4"][0] / cum["kappa2"][0]**2
        assert kt3 == pytest.approx(expected_kt3, rel=1e-10)
        assert kt4 == pytest.approx(expected_kt4, rel=1e-10)

    def test_clt_scaling(self):
        """As p grows (with equal weights), kt3 should shrink ~ 1/sqrt(p)."""
        maf_val = 0.1
        kt3_values = []
        for p_dim in [10, 100, 1000]:
            maf = np.full(p_dim, maf_val)
            w = np.ones(p_dim)
            cum = snp_cumulants(maf)
            kt3, _ = projection_cumulants_independent(
                w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
            )
            kt3_values.append(kt3)

        # kt3 ~ 1/sqrt(p), so ratios should be sqrt(10)
        ratio1 = kt3_values[0] / kt3_values[1]
        ratio2 = kt3_values[1] / kt3_values[2]
        assert ratio1 == pytest.approx(np.sqrt(10), rel=0.01)
        assert ratio2 == pytest.approx(np.sqrt(10), rel=0.01)

    def test_kt4_clt_scaling(self):
        """kt4 should scale as 1/p for equal weights and equal MAFs."""
        maf_val = 0.2
        kt4_values = []
        for p_dim in [10, 100, 1000]:
            maf = np.full(p_dim, maf_val)
            w = np.ones(p_dim)
            cum = snp_cumulants(maf)
            _, kt4 = projection_cumulants_independent(
                w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
            )
            kt4_values.append(kt4)

        ratio1 = kt4_values[0] / kt4_values[1]
        ratio2 = kt4_values[1] / kt4_values[2]
        assert ratio1 == pytest.approx(10.0, rel=0.01)
        assert ratio2 == pytest.approx(10.0, rel=0.01)

    def test_mc_projection_cumulants(self):
        """Monte Carlo validation: sample z = w^T x for independent Binomial
        SNPs and compare empirical skewness to the formula."""
        rng = np.random.default_rng(88)
        p_dim = 10
        mafs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.1])
        w = rng.standard_normal(p_dim) * 0.5
        n_samples = 1_000_000

        X = np.zeros((n_samples, p_dim))
        for j in range(p_dim):
            X[:, j] = rng.binomial(2, mafs[j], size=n_samples) - 2 * mafs[j]

        z = X @ w
        var_z = np.var(z)
        emp_kt3 = np.mean(z**3) / var_z**1.5
        emp_kt4 = (np.mean(z**4) / var_z**2) - 3.0

        cum = snp_cumulants(mafs)
        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        assert kt3 == pytest.approx(emp_kt3, abs=0.05)
        assert kt4 == pytest.approx(emp_kt4, abs=0.1)


class TestProjectionCumulantsLD:
    """Test projection cumulants with LD correction."""

    def test_identity_sigma_matches_independent(self):
        """When Sigma = I, LD-corrected cumulants equal the independent ones."""
        p = 5
        Sigma = np.eye(p)
        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15])
        w = np.array([0.5, -0.3, 0.8, 0.1, -0.4])

        kt3_ld, kt4_ld = projection_cumulants_ld(w, maf, Sigma)

        cum = snp_cumulants(maf)
        kt3_ind, kt4_ind = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        assert kt3_ld == pytest.approx(kt3_ind, rel=1e-8)
        assert kt4_ld == pytest.approx(kt4_ind, rel=1e-8)

    def test_decorrelation_matrix_inverts_sigma(self):
        """Sigma^{-1/2} @ Sigma @ Sigma^{-1/2} = I."""
        p = 5
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        S = decorrelation_matrix(Sigma)
        product = S @ Sigma @ S
        np.testing.assert_allclose(product, np.eye(p), atol=1e-10)

    def test_precomputed_sigma_inv_sqrt(self):
        """Passing precomputed Sigma_inv_sqrt gives same result."""
        p = 5
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15])
        w = np.array([0.5, -0.3, 0.8, 0.1, -0.4])

        S = decorrelation_matrix(Sigma)
        kt3_a, kt4_a = projection_cumulants_ld(w, maf, Sigma)
        kt3_b, kt4_b = projection_cumulants_ld(w, maf, Sigma, Sigma_inv_sqrt=S)

        assert kt3_a == pytest.approx(kt3_b, rel=1e-12)
        assert kt4_a == pytest.approx(kt4_b, rel=1e-12)

    def test_decorrelation_identity_is_identity(self):
        """Sigma^{-1/2} of I should be I."""
        p = 5
        S = decorrelation_matrix(np.eye(p))
        np.testing.assert_allclose(S, np.eye(p), atol=1e-10)

    def test_decorrelation_symmetric(self):
        """Sigma^{-1/2} should be symmetric."""
        p = 6
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.7 ** abs(i - j)

        S = decorrelation_matrix(Sigma)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_single_snp_ld(self):
        """With p=1, the LD-corrected cumulants should match independent."""
        Sigma = np.array([[0.8]])
        maf = np.array([0.2])
        w = np.array([1.5])

        kt3_ld, kt4_ld = projection_cumulants_ld(w, maf, Sigma)

        cum = snp_cumulants(maf)
        kt3_ind, kt4_ind = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        assert kt3_ld == pytest.approx(kt3_ind, rel=1e-6)
        assert kt4_ld == pytest.approx(kt4_ind, rel=1e-6)

    def test_high_correlation_ld(self):
        """With near-perfect LD (decay=0.95), computation should still be stable."""
        p = 5
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.95 ** abs(i - j)

        maf = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        w = np.array([0.5, -0.3, 0.8, 0.1, -0.4])

        kt3, kt4 = projection_cumulants_ld(w, maf, Sigma)
        assert np.isfinite(kt3)
        assert np.isfinite(kt4)


class TestProjectionCumulantGradients:
    """Test analytic cumulant gradients against finite differences."""

    def test_independent_gradients_vs_fd(self):
        """Gradient of kt3 and kt4 w.r.t. w for independent SNPs."""
        from ssnn.cumulants import projection_cumulant_gradients_independent

        rng = np.random.default_rng(42)
        p = 6
        maf = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4])
        cum = snp_cumulants(maf)
        w = rng.standard_normal(p) * 0.3

        g3, g4 = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        eps = 1e-7
        g3_fd = np.zeros(p)
        g4_fd = np.zeros(p)
        for j in range(p):
            w_p = w.copy(); w_p[j] += eps
            w_m = w.copy(); w_m[j] -= eps

            kt3_p, kt4_p = projection_cumulants_independent(
                w_p, cum["kappa2"], cum["kappa3"], cum["kappa4"]
            )
            kt3_m, kt4_m = projection_cumulants_independent(
                w_m, cum["kappa2"], cum["kappa3"], cum["kappa4"]
            )
            g3_fd[j] = (kt3_p - kt3_m) / (2 * eps)
            g4_fd[j] = (kt4_p - kt4_m) / (2 * eps)

        np.testing.assert_allclose(g3, g3_fd, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(g4, g4_fd, rtol=1e-4, atol=1e-8)

    def test_ld_gradients_vs_fd(self):
        """Gradient of kt3 and kt4 w.r.t. w for LD-correlated SNPs."""
        from ssnn.cumulants import projection_cumulant_gradients_ld

        rng = np.random.default_rng(42)
        p = 6
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)

        maf = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4])
        w = rng.standard_normal(p) * 0.3

        g3, g4 = projection_cumulant_gradients_ld(w, maf, Sigma)

        eps = 1e-7
        g3_fd = np.zeros(p)
        g4_fd = np.zeros(p)
        for j in range(p):
            w_p = w.copy(); w_p[j] += eps
            w_m = w.copy(); w_m[j] -= eps

            kt3_p, _ = projection_cumulants_ld(w_p, maf, Sigma)
            kt3_m, _ = projection_cumulants_ld(w_m, maf, Sigma)
            _, kt4_p = projection_cumulants_ld(w_p, maf, Sigma)
            _, kt4_m = projection_cumulants_ld(w_m, maf, Sigma)
            g3_fd[j] = (kt3_p - kt3_m) / (2 * eps)
            g4_fd[j] = (kt4_p - kt4_m) / (2 * eps)

        np.testing.assert_allclose(g3, g3_fd, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(g4, g4_fd, rtol=1e-4, atol=1e-8)

    def test_zero_weight_returns_zero_gradient(self):
        """At w = 0, the cumulant gradient should be zero (degenerate)."""
        from ssnn.cumulants import projection_cumulant_gradients_independent

        p = 4
        maf = np.array([0.1, 0.2, 0.3, 0.4])
        cum = snp_cumulants(maf)
        w = np.zeros(p)

        g3, g4 = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        np.testing.assert_allclose(g3, 0.0, atol=1e-15)
        np.testing.assert_allclose(g4, 0.0, atol=1e-15)


class TestProjectionCumulantGradientsAudit:
    """Additional tests for cumulant gradients from the audit."""

    def test_independent_gradients_maf_0_5(self):
        """When all MAFs = 0.5, kappa3 = 0, so kt3 = 0 always.
        grad_kt3 should be zero because numerator and its derivative are both 0."""
        from ssnn.cumulants import projection_cumulant_gradients_independent

        p = 5
        maf = np.full(p, 0.5)
        cum = snp_cumulants(maf)
        w = np.array([0.5, -0.3, 0.8, 0.1, -0.4])

        g3, g4 = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        np.testing.assert_allclose(g3, 0.0, atol=1e-14)
        assert np.any(np.abs(g4) > 1e-10), "kt4 gradient should be nonzero"

    def test_independent_gradients_single_snp(self):
        """With p=1, verify the gradient formula reduces to the
        quotient-rule derivative of N_r/V^{r/2} for a single term."""
        from ssnn.cumulants import projection_cumulant_gradients_independent

        maf = np.array([0.15])
        cum = snp_cumulants(maf)
        w = np.array([2.0])

        g3, g4 = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        eps = 1e-7
        w_p = w.copy(); w_p[0] += eps
        w_m = w.copy(); w_m[0] -= eps
        kt3_p, kt4_p = projection_cumulants_independent(
            w_p, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        kt3_m, kt4_m = projection_cumulants_independent(
            w_m, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        g3_fd = (kt3_p - kt3_m) / (2 * eps)
        g4_fd = (kt4_p - kt4_m) / (2 * eps)

        assert g3[0] == pytest.approx(g3_fd, rel=1e-4, abs=1e-8)
        assert g4[0] == pytest.approx(g4_fd, rel=1e-4, abs=1e-8)

    def test_independent_gradient_scaling_with_w(self):
        """Scaling w by alpha: kt3(alpha*w) = kt3(w) (scale-invariant),
        so d/d(alpha) kt3(alpha*w) = 0 at alpha=1, meaning
        w^T grad_kt3 = 0 (Euler's theorem for degree-0 homogeneous func)."""
        from ssnn.cumulants import projection_cumulant_gradients_independent

        rng = np.random.default_rng(99)
        p = 8
        maf = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        cum = snp_cumulants(maf)
        w = rng.standard_normal(p) * 0.5

        g3, g4 = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        assert w @ g3 == pytest.approx(0.0, abs=1e-10), (
            "kt3 is degree-0 homogeneous in w, so w . grad(kt3) = 0"
        )
        assert w @ g4 == pytest.approx(0.0, abs=1e-10), (
            "kt4 is degree-0 homogeneous in w, so w . grad(kt4) = 0"
        )

    def test_ld_gradients_identity_sigma_matches_independent(self):
        """When Sigma = I, LD gradients should match independent gradients."""
        from ssnn.cumulants import (
            projection_cumulant_gradients_independent,
            projection_cumulant_gradients_ld,
        )

        rng = np.random.default_rng(42)
        p = 5
        Sigma = np.eye(p)
        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15])
        w = rng.standard_normal(p) * 0.3
        cum = snp_cumulants(maf)

        g3_ind, g4_ind = projection_cumulant_gradients_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )
        g3_ld, g4_ld = projection_cumulant_gradients_ld(w, maf, Sigma)

        np.testing.assert_allclose(g3_ld, g3_ind, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(g4_ld, g4_ind, rtol=1e-6, atol=1e-10)

    def test_ld_gradients_high_correlation(self):
        """Gradients under strong LD should still be finite and match FD."""
        from ssnn.cumulants import projection_cumulant_gradients_ld

        rng = np.random.default_rng(42)
        p = 4
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.9 ** abs(i - j)

        maf = np.array([0.1, 0.2, 0.3, 0.4])
        w = rng.standard_normal(p) * 0.3

        g3, g4 = projection_cumulant_gradients_ld(w, maf, Sigma)
        assert np.all(np.isfinite(g3))
        assert np.all(np.isfinite(g4))

        eps = 1e-7
        g3_fd = np.zeros(p)
        g4_fd = np.zeros(p)
        for j in range(p):
            w_p = w.copy(); w_p[j] += eps
            w_m = w.copy(); w_m[j] -= eps
            kt3_p, _ = projection_cumulants_ld(w_p, maf, Sigma)
            kt3_m, _ = projection_cumulants_ld(w_m, maf, Sigma)
            _, kt4_p = projection_cumulants_ld(w_p, maf, Sigma)
            _, kt4_m = projection_cumulants_ld(w_m, maf, Sigma)
            g3_fd[j] = (kt3_p - kt3_m) / (2 * eps)
            g4_fd[j] = (kt4_p - kt4_m) / (2 * eps)

        np.testing.assert_allclose(g3, g3_fd, rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(g4, g4_fd, rtol=1e-3, atol=1e-8)
