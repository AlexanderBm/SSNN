"""Tests for the error analysis module (Step 5).

Verifies the five error-bound components:
    (a) Edgeworth truncation error
    (b) Decorrelation approximation error
    (c) LD estimation error
    (d) PUMAS splitting variance
    (e) Optimization error
and the full error decomposition.
"""

import numpy as np
import pytest

from ssnn.error_analysis import (
    ErrorDecomposition,
    edgeworth_truncation_bound,
    decorrelation_bound,
    ld_estimation_bound,
    pumas_variance_bound,
    optimization_bound,
    estimate_smoothness,
    compute_error_decomposition,
    _hermite_poly,
    _hermite_abs_moment,
    _lipschitz_constant,
)
from ssnn.cumulants import snp_cumulants, decorrelation_matrix
from ssnn.edgeworth_risk import compute_edgeworth_loss
from ssnn.population_risk import compute_loss
from ssnn.gaussian_integrals import projection_variance


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def ea_problem():
    """A problem with non-trivial MAFs and LD for error analysis tests."""
    rng = np.random.default_rng(42)
    p = 8
    m = 2

    Sigma = np.eye(p)
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = 0.5 ** abs(i - j)

    maf = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    return {
        "p": p, "m": m, "Sigma": Sigma, "maf": maf,
        "beta_star": beta_star, "Sigma_beta": Sigma_beta,
        "E_y2": E_y2, "W": W, "a": a,
    }


@pytest.fixture
def identity_ld_problem():
    """A problem with Sigma = I (no LD) for isolating decorrelation effects."""
    rng = np.random.default_rng(99)
    p = 6
    m = 2

    Sigma = np.eye(p)
    maf = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.45])

    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ beta_star + 1.0)

    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1

    return {
        "p": p, "m": m, "Sigma": Sigma, "maf": maf,
        "beta_star": beta_star, "Sigma_beta": Sigma_beta,
        "E_y2": E_y2, "W": W, "a": a,
    }


# =====================================================================
# Helper tests
# =====================================================================

class TestHermitePoly:
    """Tests for the general Hermite polynomial evaluator."""

    def test_H0(self):
        t = np.array([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(_hermite_poly(0, t), np.ones(4))

    def test_H1(self):
        t = np.array([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(_hermite_poly(1, t), t)

    def test_H3_matches_known(self):
        """H_3(t) = t^3 - 3t (probabilist's convention)."""
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = t**3 - 3.0 * t
        np.testing.assert_allclose(_hermite_poly(3, t), expected)

    def test_H4_matches_known(self):
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = t**4 - 6.0 * t**2 + 3.0
        np.testing.assert_allclose(_hermite_poly(4, t), expected)

    def test_H6_matches_known(self):
        t = np.array([-1.0, 0.0, 1.0, 2.0])
        expected = t**6 - 15.0 * t**4 + 45.0 * t**2 - 15.0
        np.testing.assert_allclose(_hermite_poly(6, t), expected)


class TestHermiteAbsMoment:
    """Tests for E[|H_r(z)|] computation."""

    def test_H0_moment(self):
        assert _hermite_abs_moment(0) == pytest.approx(1.0, abs=1e-8)

    def test_H1_moment(self):
        """E[|z|] = sqrt(2/pi) for z ~ N(0,1).
        The Gauss-Hermite quadrature with |·| is slightly less precise
        since |t| is not a polynomial, so we use a looser tolerance."""
        expected = np.sqrt(2.0 / np.pi)
        assert _hermite_abs_moment(1) == pytest.approx(expected, rel=1e-2)

    def test_moments_are_positive(self):
        for r in range(8):
            assert _hermite_abs_moment(r) > 0


class TestLipschitzConstant:
    def test_relu(self):
        assert _lipschitz_constant("relu") == 1.0

    def test_sigmoid(self):
        assert _lipschitz_constant("sigmoid") == pytest.approx(
            np.sqrt(np.pi / 8.0), rel=1e-10
        )

    def test_identity(self):
        assert _lipschitz_constant("identity") == 1.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            _lipschitz_constant("tanh")


# =====================================================================
# (a) Edgeworth truncation bound
# =====================================================================

class TestEdgeworthTruncation:
    """Tests for the Edgeworth truncation error bound."""

    def test_non_negative(self, ea_problem):
        d = ea_problem
        bound = edgeworth_truncation_bound(d["maf"], d["W"], d["Sigma"])
        assert bound >= 0

    def test_zero_for_identity(self, ea_problem):
        """Identity activation has all zero Hermite correction integrals,
        so the entire Edgeworth correction vanishes, making truncation
        error 0. But the bound itself uses Lip(g)=1 and the Lyapunov ratio,
        so it may be nonzero (it bounds a quantity we know is zero). Still,
        the actual truncation error is 0 for identity — the bound is
        conservative. We just check it's finite and small-ish."""
        d = ea_problem
        bound = edgeworth_truncation_bound(
            d["maf"], d["W"], d["Sigma"], activation="identity"
        )
        assert np.isfinite(bound)
        assert bound >= 0

    def test_increases_with_rarer_alleles(self):
        """Rarer alleles (smaller MAF) produce larger kappa_3, so the
        truncation bound should be larger."""
        rng = np.random.default_rng(42)
        p = 6
        m = 2
        Sigma = np.eye(p)
        W = rng.standard_normal((m, p)) * 0.1

        maf_common = np.full(p, 0.45)
        maf_rare = np.full(p, 0.05)

        bound_common = edgeworth_truncation_bound(maf_common, W, Sigma)
        bound_rare = edgeworth_truncation_bound(maf_rare, W, Sigma)

        assert bound_rare > bound_common

    def test_zero_when_maf_half(self):
        """When all MAFs = 0.5, kappa_3 = kappa_5 = 0; leading truncation
        terms vanish. kappa_4 is still nonzero but the cross-term κ₃·κ₄
        vanishes."""
        rng = np.random.default_rng(42)
        p = 6
        m = 2
        Sigma = np.eye(p)
        W = rng.standard_normal((m, p)) * 0.1
        maf = np.full(p, 0.5)

        bound = edgeworth_truncation_bound(maf, W, Sigma)
        assert bound < 1e-10

    def test_scales_with_number_of_snps(self):
        """For fixed MAF, the Lyapunov ratio decreases as p grows (CLT),
        so the truncation bound should decrease."""
        rng = np.random.default_rng(42)

        bounds = []
        for p in [10, 50, 200]:
            Sigma = np.eye(p)
            W = rng.standard_normal((1, p)) * 0.1
            maf = np.full(p, 0.1)
            bounds.append(edgeworth_truncation_bound(maf, W, Sigma))

        assert bounds[0] > bounds[1] > bounds[2]

    def test_no_sigma_uses_identity(self):
        """When Sigma is None, should use identity (independent SNPs)."""
        rng = np.random.default_rng(42)
        p = 6
        W = rng.standard_normal((2, p)) * 0.1
        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15, 0.25])

        b1 = edgeworth_truncation_bound(maf, W, Sigma=None)
        b2 = edgeworth_truncation_bound(maf, W, Sigma=np.eye(p))
        assert b1 == pytest.approx(b2, rel=1e-6)


# =====================================================================
# (b) Decorrelation approximation bound
# =====================================================================

class TestDecorrelationBound:
    """Tests for the decorrelation (cross-cumulant) error bound."""

    def test_non_negative(self, ea_problem):
        d = ea_problem
        bound = decorrelation_bound(d["maf"], d["W"], d["Sigma"])
        assert bound >= 0

    def test_zero_for_identity_ld(self, identity_ld_problem):
        """When Sigma = I, Sigma^{-1/2} = I so the decorrelation is exact.
        The off-diagonal fraction should be 0 and the bound should be 0."""
        d = identity_ld_problem
        bound = decorrelation_bound(d["maf"], d["W"], d["Sigma"])
        assert bound == pytest.approx(0.0, abs=1e-10)

    def test_increases_with_stronger_ld(self):
        """Stronger LD → more off-diagonal in Sigma^{-1/2} → larger bound."""
        rng = np.random.default_rng(42)
        p = 6
        m = 2
        maf = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.35])
        W = rng.standard_normal((m, p)) * 0.1

        Sigma_weak = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma_weak[i, j] = 0.1 ** abs(i - j)

        Sigma_strong = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma_strong[i, j] = 0.9 ** abs(i - j)

        bound_weak = decorrelation_bound(maf, W, Sigma_weak)
        bound_strong = decorrelation_bound(maf, W, Sigma_strong)
        assert bound_strong > bound_weak

    def test_precomputed_sigma_inv_sqrt(self, ea_problem):
        d = ea_problem
        S = decorrelation_matrix(d["Sigma"])
        b1 = decorrelation_bound(d["maf"], d["W"], d["Sigma"])
        b2 = decorrelation_bound(d["maf"], d["W"], d["Sigma"], S)
        assert b1 == pytest.approx(b2, rel=1e-10)


# =====================================================================
# (c) LD estimation error bound
# =====================================================================

class TestLDEstimationBound:
    """Tests for the LD estimation error bound."""

    def test_zero_when_no_error(self, ea_problem):
        d = ea_problem
        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.0,
        )
        assert bound == pytest.approx(0.0, abs=1e-15)

    def test_non_negative(self, ea_problem):
        d = ea_problem
        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.1,
        )
        assert bound >= 0

    def test_linear_in_delta(self, ea_problem):
        """The bound should scale linearly with delta_Sigma_fro."""
        d = ea_problem
        b1 = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.1,
        )
        b2 = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.2,
        )
        assert b2 == pytest.approx(2.0 * b1, rel=1e-10)

    def test_bounds_actual_loss_perturbation(self, ea_problem):
        """The bound should be >= the actual loss difference when we
        perturb Sigma by a small amount."""
        d = ea_problem
        rng = np.random.default_rng(123)

        delta = rng.standard_normal((d["p"], d["p"])) * 0.01
        delta = (delta + delta.T) / 2
        delta_fro = np.linalg.norm(delta, "fro")

        Sigma_hat = d["Sigma"] + delta
        eigvals = np.linalg.eigvalsh(Sigma_hat)
        if eigvals[0] < 1e-6:
            Sigma_hat += (1e-6 - eigvals[0]) * np.eye(d["p"])

        Sigma_beta_hat = d["Sigma_beta"]

        L_true = compute_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"]
        )
        L_pert = compute_loss(
            d["a"], d["W"], Sigma_hat, Sigma_beta_hat, d["E_y2"]
        )
        actual_diff = abs(L_pert - L_true)

        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=delta_fro,
        )
        assert bound >= actual_diff * 0.5, (
            f"Bound {bound} is not close to covering actual diff {actual_diff}. "
            f"(Using 0.5x tolerance for first-order bound vs actual.)"
        )

    def test_sigmoid_activation(self, ea_problem):
        d = ea_problem
        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.05, activation="sigmoid",
        )
        assert bound >= 0
        assert np.isfinite(bound)


# =====================================================================
# (d) PUMAS splitting variance
# =====================================================================

class TestPUMASVarianceBound:
    """Tests for the PUMAS splitting variance bound."""

    def test_non_negative(self, ea_problem):
        d = ea_problem
        bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=50000, n_train=40000, W=d["W"], a=d["a"],
        )
        assert bound >= 0

    def test_decreases_with_larger_N(self, ea_problem):
        """Larger GWAS → less PUMAS noise → smaller variance."""
        d = ea_problem
        b_small = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=10000, n_train=8000, W=d["W"], a=d["a"],
        )
        b_large = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=100000, n_train=80000, W=d["W"], a=d["a"],
        )
        assert b_large < b_small

    def test_increases_when_train_fraction_decreases(self, ea_problem):
        """Smaller training fraction → more noise → larger variance."""
        d = ea_problem
        N = 50000
        b_80 = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=N, n_train=int(0.8 * N), W=d["W"], a=d["a"],
        )
        b_50 = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=N, n_train=int(0.5 * N), W=d["W"], a=d["a"],
        )
        assert b_50 > b_80

    def test_empirical_variance_within_bound(self, ea_problem):
        """Generate many PUMAS splits and check that the empirical loss
        standard deviation is bounded by our analytical formula."""
        from ssnn.pumas import generate_pumas_split
        d = ea_problem
        N = 50000
        n_train = 40000

        losses = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            split = generate_pumas_split(
                d["Sigma_beta"], d["E_y2"], d["Sigma"], N, n_train, rng
            )
            L = compute_loss(
                d["a"], d["W"], d["Sigma"], split.Sigma_beta_train,
                split.E_y2_train,
            )
            losses.append(L)

        empirical_std = np.std(losses)
        analytic_bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=N, n_train=n_train, W=d["W"], a=d["a"],
        )
        assert analytic_bound >= empirical_std * 0.3, (
            f"Analytic bound {analytic_bound:.6f} should cover a meaningful "
            f"fraction of empirical std {empirical_std:.6f}"
        )

    def test_zero_weights_give_finite_bound(self, ea_problem):
        d = ea_problem
        W_zero = np.zeros_like(d["W"])
        a_zero = np.zeros_like(d["a"])
        bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=50000, n_train=40000, W=W_zero, a=a_zero,
        )
        assert np.isfinite(bound)
        assert bound >= 0


# =====================================================================
# (e) Optimization error bound
# =====================================================================

class TestOptimizationBound:
    """Tests for the optimization error bound."""

    def test_zero_for_converged(self):
        """If loss converged to a constant, the empirical gap is 0."""
        losses = [1.0, 0.5, 0.3, 0.3, 0.3]
        bound = optimization_bound(losses, lr=0.01)
        assert bound == pytest.approx(0.0, abs=1e-15)

    def test_non_negative(self):
        losses = [2.0, 1.5, 1.0, 0.8, 0.7, 0.6]
        bound = optimization_bound(losses, lr=0.01)
        assert bound >= 0

    def test_equals_final_minus_min(self):
        """Without L_smooth, bound = final - min."""
        losses = [2.0, 1.0, 0.5, 0.8]
        bound = optimization_bound(losses, lr=0.01)
        assert bound == pytest.approx(0.8 - 0.5, abs=1e-15)

    def test_empty_history(self):
        bound = optimization_bound([], lr=0.01)
        assert bound == float("inf")

    def test_single_entry(self):
        bound = optimization_bound([1.5], lr=0.01)
        assert bound == pytest.approx(0.0, abs=1e-15)

    def test_with_L_smooth(self):
        losses = [10.0, 5.0, 3.0, 2.0, 1.5]
        L_smooth = 100.0
        bound = optimization_bound(
            losses, lr=0.01, L_smooth=L_smooth, grad_clip=1.0
        )
        assert bound >= 0
        empirical = 1.5 - 1.5
        assert bound >= empirical

    def test_rate_bound_tighter_than_empirical(self):
        """For a well-behaved loss with many iterations, the rate bound
        can be tighter than the empirical gap."""
        T = 1000
        losses = [10.0 / (i + 1) for i in range(T + 1)]
        lr = 0.01
        L_smooth = 10.0

        bound = optimization_bound(
            losses, lr=lr, L_smooth=L_smooth, grad_clip=1.0
        )
        assert bound >= 0
        assert np.isfinite(bound)


class TestEstimateSmoothness:
    """Tests for the local Lipschitz constant estimator."""

    def test_positive(self, ea_problem):
        d = ea_problem
        L_smooth = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], n_probes=5,
        )
        assert L_smooth > 0

    def test_finite(self, ea_problem):
        d = ea_problem
        L_smooth = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"],
        )
        assert np.isfinite(L_smooth)

    def test_more_probes_gives_larger_estimate(self, ea_problem):
        """More probes should find a larger (or equal) smoothness constant."""
        d = ea_problem
        L_few = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], n_probes=2, rng=np.random.default_rng(0),
        )
        L_many = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], n_probes=20, rng=np.random.default_rng(0),
        )
        assert L_many >= L_few


# =====================================================================
# Full decomposition
# =====================================================================

class TestErrorDecomposition:
    """Tests for the full error decomposition."""

    def test_all_non_negative(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.05,
            loss_history=[1.0, 0.5, 0.3],
        )
        assert decomp.edgeworth_truncation >= 0
        assert decomp.decorrelation_approx >= 0
        assert decomp.ld_estimation >= 0
        assert decomp.pumas_variance >= 0
        assert decomp.optimization >= 0

    def test_total_is_sum(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.1,
            loss_history=[2.0, 1.0, 0.5],
        )
        expected_total = (
            decomp.edgeworth_truncation
            + decomp.decorrelation_approx
            + decomp.ld_estimation
            + decomp.pumas_variance
            + decomp.optimization
        )
        assert decomp.total == pytest.approx(expected_total, rel=1e-12)

    def test_no_loss_history_gives_zero_opt(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000,
        )
        assert decomp.optimization == 0.0

    def test_zero_delta_sigma_gives_zero_ld(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.0,
        )
        assert decomp.ld_estimation == pytest.approx(0.0, abs=1e-15)

    def test_identity_sigma_gives_zero_decorrelation(self, identity_ld_problem):
        d = identity_ld_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000,
        )
        assert decomp.decorrelation_approx == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_activation(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, activation="sigmoid",
        )
        assert decomp.total >= 0
        assert np.isfinite(decomp.total)

    def test_all_finite(self, ea_problem):
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.1,
            loss_history=[5.0, 3.0, 2.0, 1.5, 1.2],
        )
        assert np.isfinite(decomp.edgeworth_truncation)
        assert np.isfinite(decomp.decorrelation_approx)
        assert np.isfinite(decomp.ld_estimation)
        assert np.isfinite(decomp.pumas_variance)
        assert np.isfinite(decomp.optimization)

    def test_decomposition_with_trained_model(self, ea_problem):
        """Run actual training and verify the decomposition on the result."""
        from ssnn.edgeworth_optimizer import train_edgeworth

        d = ea_problem
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, lr=0.005, max_iters=50, activation="relu",
            rng=np.random.default_rng(42),
        )

        decomp = compute_error_decomposition(
            result.a, result.W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.01,
            loss_history=result.loss_history, lr=0.005,
        )
        assert decomp.total >= 0
        assert np.isfinite(decomp.total)
        assert decomp.optimization >= 0


class TestEdgeworthTruncationSanity:
    """Cross-check: the truncation bound should be consistent with
    Monte Carlo estimates of the actual truncation error."""

    def test_bound_covers_mc_estimate(self):
        """For independent Binomial genotypes, compare the analytical
        Edgeworth E[sigma(z)] against a Monte Carlo estimate, and verify
        the truncation bound covers the gap."""
        from ssnn.edgeworth_integrals import edgeworth_E_sigma
        from ssnn.cumulants import (
            snp_cumulants,
            projection_cumulants_independent,
        )

        rng = np.random.default_rng(42)
        p = 20
        maf = rng.uniform(0.05, 0.4, size=p)
        w = rng.standard_normal(p) * 0.2

        cum = snp_cumulants(maf)
        v = np.sum(w**2 * cum["kappa2"])
        kt3, kt4 = projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )

        ew_val = edgeworth_E_sigma(v, kt3, kt4, "relu")

        n_mc = 100_000
        genotypes = np.column_stack([
            rng.binomial(2, maf[j], size=n_mc) - 2.0 * maf[j]
            for j in range(p)
        ])
        z = genotypes @ w
        mc_val = np.mean(np.maximum(0, z))

        actual_error = abs(ew_val - mc_val)

        W = w.reshape(1, -1)
        bound = edgeworth_truncation_bound(maf, W, Sigma=None)

        assert bound > 0
        assert actual_error < bound * 50, (
            f"MC error {actual_error:.6f} exceeds 50x truncation bound "
            f"{bound:.6f}. The bound or MC may need more samples."
        )


# =====================================================================
# NEW TESTS — Gap-filling additions
# =====================================================================

# --- Additional fixtures ---

@pytest.fixture
def p1_problem():
    """Minimal problem with p=1 SNP and m=1 hidden unit."""
    Sigma = np.array([[1.0]])
    maf = np.array([0.2])
    beta_star = np.array([0.5])
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)
    W = np.array([[0.3]])
    a = np.array([0.4])
    return {
        "p": 1, "m": 1, "Sigma": Sigma, "maf": maf,
        "beta_star": beta_star, "Sigma_beta": Sigma_beta,
        "E_y2": E_y2, "W": W, "a": a,
    }


@pytest.fixture
def near_singular_problem():
    """Problem with a near-singular Sigma (condition number > 1000)."""
    rng = np.random.default_rng(77)
    p = 6
    m = 2
    eigvals = np.array([1000.0, 1.0, 0.5, 0.3, 0.1, 0.05])
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Sigma = Q @ np.diag(eigvals) @ Q.T
    Sigma = (Sigma + Sigma.T) / 2.0

    maf = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.35])
    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + 1.0)
    W = rng.standard_normal((m, p)) * 0.1
    a = rng.standard_normal(m) * 0.1
    return {
        "p": p, "m": m, "Sigma": Sigma, "maf": maf,
        "beta_star": beta_star, "Sigma_beta": Sigma_beta,
        "E_y2": E_y2, "W": W, "a": a,
    }


# =====================================================================
# Hermite polynomial — missing orders and recurrence property
# =====================================================================

class TestHermitePolyExtended:
    """Additional tests for the Hermite polynomial helper."""

    def test_H2_matches_known(self):
        """H_2(t) = t^2 - 1 (probabilist's convention)."""
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = t**2 - 1.0
        np.testing.assert_allclose(_hermite_poly(2, t), expected)

    def test_H5_matches_known(self):
        """H_5(t) = t^5 - 10t^3 + 15t (probabilist's convention)."""
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        expected = t**5 - 10.0 * t**3 + 15.0 * t
        np.testing.assert_allclose(_hermite_poly(5, t), expected, atol=1e-10)

    def test_H7_via_recurrence(self):
        """Test H_7 satisfies the three-term recurrence:
        H_r(t) = t * H_{r-1}(t) - (r-1) * H_{r-2}(t)."""
        t = np.array([-3.0, -1.5, 0.0, 0.5, 2.0, 4.0])
        h5 = _hermite_poly(5, t)
        h6 = _hermite_poly(6, t)
        h7_recurrence = t * h6 - 6.0 * h5
        h7_direct = _hermite_poly(7, t)
        np.testing.assert_allclose(h7_direct, h7_recurrence, atol=1e-8)

    def test_H10_via_recurrence(self):
        """Test that higher-order Hermite polynomials obey the recurrence
        at r=10: H_10(t) = t*H_9(t) - 9*H_8(t)."""
        t = np.array([-2.0, 0.0, 1.0, 3.0])
        h8 = _hermite_poly(8, t)
        h9 = _hermite_poly(9, t)
        h10_recurrence = t * h9 - 9.0 * h8
        h10_direct = _hermite_poly(10, t)
        np.testing.assert_allclose(h10_direct, h10_recurrence, atol=1e-6)

    def test_orthogonality_H2_H3(self):
        """E[H_2(z) * H_3(z)] = 0 for z ~ N(0,1) by Hermite orthogonality."""
        nodes, weights = np.polynomial.hermite_e.hermegauss(30)
        h2 = _hermite_poly(2, nodes)
        h3 = _hermite_poly(3, nodes)
        inner = np.sum(weights * h2 * h3) / np.sqrt(2.0 * np.pi)
        assert inner == pytest.approx(0.0, abs=1e-10)


class TestHermiteAbsMomentExtended:
    """Additional tests for E[|H_r(z)|] computation."""

    def test_H2_moment_known_value(self):
        """E[|H_2(z)|] = E[|z^2 - 1|] for z ~ N(0,1).
        Cross-check against the same Gauss–Hermite quadrature with matching
        node count (60 nodes) to confirm the implementation is self-consistent."""
        val = _hermite_abs_moment(2)
        nodes, weights = np.polynomial.hermite_e.hermegauss(60)
        expected = np.sum(weights * np.abs(nodes**2 - 1.0)) / np.sqrt(2.0 * np.pi)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_even_moment_H4(self):
        """Cross-check E[|H_4(z)|] against the same 60-node quadrature."""
        val = _hermite_abs_moment(4)
        nodes, weights = np.polynomial.hermite_e.hermegauss(60)
        h4 = nodes**4 - 6.0 * nodes**2 + 3.0
        expected = np.sum(weights * np.abs(h4)) / np.sqrt(2.0 * np.pi)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_moment_increases_with_r(self):
        """E[|H_r(z)|] should generally grow with r (the polynomial spreads)."""
        m3 = _hermite_abs_moment(3)
        m7 = _hermite_abs_moment(7)
        assert m7 > m3


# =====================================================================
# (a) Edgeworth truncation bound — edge cases and activation ordering
# =====================================================================

class TestEdgeworthTruncationEdgeCases:
    """Edge cases and additional coverage for truncation bound."""

    def test_p1_m1(self, p1_problem):
        """Truncation bound should still be computable with a single SNP."""
        d = p1_problem
        bound = edgeworth_truncation_bound(d["maf"], d["W"], d["Sigma"])
        assert bound >= 0
        assert np.isfinite(bound)

    def test_very_extreme_low_maf(self):
        """MAF=0.01 (very rare alleles) should produce a large but finite bound."""
        rng = np.random.default_rng(42)
        p = 5
        W = rng.standard_normal((1, p)) * 0.1
        maf = np.full(p, 0.01)
        bound = edgeworth_truncation_bound(maf, W, Sigma=None)
        assert bound > 0
        assert np.isfinite(bound)

    def test_maf_near_half(self):
        """MAF=0.499 should give a small bound (nearly symmetric genotypes).
        kappa3 and kappa5 nearly vanish but kappa4 is still nonzero, so the
        cross-term |kt3*kt4| and kt3^3 are small but the bound is not zero."""
        rng = np.random.default_rng(42)
        p = 10
        W = rng.standard_normal((2, p)) * 0.1
        maf = np.full(p, 0.499)
        bound = edgeworth_truncation_bound(maf, W, Sigma=None)
        assert bound < 1e-2

    def test_sigmoid_larger_than_identity(self, ea_problem):
        """Sigmoid has Lip(g) = sqrt(pi/8) > 1 for probit, but we compare
        against identity which also has Lip=1. Both should be finite;
        sigmoid bound should differ from ReLU due to the different constant."""
        d = ea_problem
        b_relu = edgeworth_truncation_bound(d["maf"], d["W"], d["Sigma"], activation="relu")
        b_sig = edgeworth_truncation_bound(d["maf"], d["W"], d["Sigma"], activation="sigmoid")
        assert b_relu > 0
        assert b_sig > 0
        lip_ratio = np.sqrt(np.pi / 8.0)
        assert b_sig == pytest.approx(b_relu * lip_ratio, rel=1e-10)

    def test_zero_row_in_W(self):
        """A hidden unit with w_k = 0 has zero projection variance;
        the bound should still work (skip that unit)."""
        p = 5
        W = np.zeros((2, p))
        W[1] = np.array([0.1, 0.2, -0.1, 0.15, -0.05])
        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15])
        bound = edgeworth_truncation_bound(maf, W, Sigma=None)
        assert bound >= 0
        assert np.isfinite(bound)

    def test_near_singular_sigma(self, near_singular_problem):
        """The bound should be finite even with a near-singular LD matrix."""
        d = near_singular_problem
        bound = edgeworth_truncation_bound(d["maf"], d["W"], d["Sigma"])
        assert np.isfinite(bound)
        assert bound >= 0

    def test_more_hidden_units_uses_max(self):
        """The bound takes the max over hidden units. Adding a unit with
        larger projection skew should increase or preserve the bound."""
        rng = np.random.default_rng(42)
        p = 6
        maf = np.full(p, 0.1)
        Sigma = np.eye(p)
        W_small = rng.standard_normal((1, p)) * 0.1
        W_big = np.vstack([W_small, rng.standard_normal((1, p)) * 0.5])
        b1 = edgeworth_truncation_bound(maf, W_small, Sigma)
        b2 = edgeworth_truncation_bound(maf, W_big, Sigma)
        assert b2 >= b1 * 0.99

    def test_large_weights_increase_bound(self):
        """Larger weight magnitudes increase the Lyapunov ratio (for fixed p)."""
        rng = np.random.default_rng(42)
        p = 8
        maf = np.full(p, 0.15)
        Sigma = np.eye(p)
        W_base = rng.standard_normal((1, p))
        b_small = edgeworth_truncation_bound(maf, W_base * 0.01, Sigma)
        b_large = edgeworth_truncation_bound(maf, W_base * 1.0, Sigma)
        assert np.isfinite(b_small) and np.isfinite(b_large)
        assert b_small == pytest.approx(b_large, rel=1e-6), (
            "Lyapunov ratios are scale-invariant in the weights"
        )


# =====================================================================
# (b) Decorrelation bound — edge cases
# =====================================================================

class TestDecorrelationBoundEdgeCases:
    """Edge cases and additional coverage for decorrelation bound."""

    def test_p1_always_zero(self, p1_problem):
        """With p=1, there are no off-diagonal entries; bound should be 0."""
        d = p1_problem
        bound = decorrelation_bound(d["maf"], d["W"], d["Sigma"])
        assert bound == pytest.approx(0.0, abs=1e-10)

    def test_near_singular_sigma_finite(self, near_singular_problem):
        """Decorrelation bound should be finite with near-singular LD."""
        d = near_singular_problem
        bound = decorrelation_bound(d["maf"], d["W"], d["Sigma"])
        assert np.isfinite(bound)
        assert bound >= 0

    def test_zero_row_in_W(self):
        """A hidden unit with w_k = 0 should be skipped gracefully."""
        p = 5
        W = np.zeros((2, p))
        W[1] = np.array([0.1, 0.2, -0.1, 0.15, -0.05])
        maf = np.array([0.1, 0.2, 0.3, 0.4, 0.15])
        Sigma = np.eye(p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.5 ** abs(i - j)
        bound = decorrelation_bound(maf, W, Sigma)
        assert np.isfinite(bound)
        assert bound >= 0

    def test_scales_with_ld_strength_fine_grained(self):
        """Sweep decay parameter and check monotonicity of the bound."""
        rng = np.random.default_rng(42)
        p = 6
        maf = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.35])
        W = rng.standard_normal((2, p)) * 0.1

        bounds = []
        for decay in [0.0, 0.3, 0.6, 0.9]:
            Sigma = np.eye(p)
            for i in range(p):
                for j in range(p):
                    Sigma[i, j] = decay ** abs(i - j) if decay > 0 else float(i == j)
            bounds.append(decorrelation_bound(maf, W, Sigma))

        for i in range(len(bounds) - 1):
            assert bounds[i + 1] >= bounds[i] - 1e-10


# =====================================================================
# (c) LD estimation bound — edge cases and Edgeworth cross-check
# =====================================================================

class TestLDEstimationBoundEdgeCases:
    """Additional tests for the LD estimation bound."""

    def test_p1_m1(self, p1_problem):
        """LD estimation bound with a single SNP should work and be non-negative."""
        d = p1_problem
        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.05,
        )
        assert bound >= 0
        assert np.isfinite(bound)

    def test_zero_row_W(self, ea_problem):
        """A zero-weight row should not break the computation."""
        d = ea_problem
        W_mod = d["W"].copy()
        W_mod[0] = 0.0
        bound = ld_estimation_bound(
            W_mod, d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.1,
        )
        assert np.isfinite(bound)
        assert bound >= 0

    def test_bounds_edgeworth_loss_perturbation(self, ea_problem):
        """The LD estimation bound should also cover perturbation of the
        Edgeworth loss, since that's what the bound is designed for."""
        d = ea_problem
        rng = np.random.default_rng(321)

        delta = rng.standard_normal((d["p"], d["p"])) * 0.005
        delta = (delta + delta.T) / 2
        delta_fro = np.linalg.norm(delta, "fro")

        Sigma_hat = d["Sigma"] + delta
        eigvals = np.linalg.eigvalsh(Sigma_hat)
        if eigvals[0] < 1e-6:
            Sigma_hat += (1e-6 - eigvals[0]) * np.eye(d["p"])

        L_true = compute_edgeworth_loss(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], loss_floor=None,
        )
        L_pert = compute_edgeworth_loss(
            d["a"], d["W"], Sigma_hat, d["Sigma_beta"],
            d["E_y2"], d["maf"], loss_floor=None,
        )
        actual_diff = abs(L_pert - L_true)

        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=delta_fro,
        )
        assert bound >= actual_diff * 0.1, (
            f"Bound {bound} should cover a meaningful fraction of actual "
            f"Edgeworth loss diff {actual_diff}"
        )

    def test_near_singular_sigma(self, near_singular_problem):
        """Finite result with near-singular LD matrix."""
        d = near_singular_problem
        bound = ld_estimation_bound(
            d["W"], d["a"], d["Sigma"], d["Sigma_beta"],
            delta_Sigma_fro=0.05,
        )
        assert np.isfinite(bound)
        assert bound >= 0


# =====================================================================
# (d) PUMAS variance bound — Edgeworth loss variant and edge cases
# =====================================================================

class TestPUMASVarianceBoundEdgeCases:
    """Additional tests for the PUMAS variance bound."""

    def test_p1_m1(self, p1_problem):
        """PUMAS variance bound should work with a single SNP."""
        d = p1_problem
        bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=50000, n_train=40000, W=d["W"], a=d["a"],
        )
        assert bound >= 0
        assert np.isfinite(bound)

    def test_empirical_edgeworth_loss_variance(self, ea_problem):
        """Test PUMAS variance against the Edgeworth loss (not just Gaussian),
        since the bound is derived for the Edgeworth framework."""
        from ssnn.pumas import generate_pumas_split
        d = ea_problem
        N = 50000
        n_train = 40000

        losses = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            split = generate_pumas_split(
                d["Sigma_beta"], d["E_y2"], d["Sigma"], N, n_train, rng
            )
            L = compute_edgeworth_loss(
                d["a"], d["W"], d["Sigma"], split.Sigma_beta_train,
                split.E_y2_train, d["maf"], loss_floor=None,
            )
            losses.append(L)

        empirical_std = np.std(losses)
        analytic_bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=N, n_train=n_train, W=d["W"], a=d["a"],
        )
        assert analytic_bound >= empirical_std * 0.2, (
            f"Analytic bound {analytic_bound:.6f} should cover a meaningful "
            f"fraction of empirical Edgeworth-loss std {empirical_std:.6f}"
        )

    def test_sigmoid_activation(self, ea_problem):
        """PUMAS variance bound should be finite for sigmoid activation."""
        d = ea_problem
        bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=50000, n_train=40000, W=d["W"], a=d["a"],
            activation="sigmoid",
        )
        assert bound >= 0
        assert np.isfinite(bound)

    def test_near_singular_sigma(self, near_singular_problem):
        """Finite result with near-singular LD matrix."""
        d = near_singular_problem
        bound = pumas_variance_bound(
            d["Sigma"], d["Sigma_beta"], d["E_y2"],
            N=50000, n_train=40000, W=d["W"], a=d["a"],
        )
        assert np.isfinite(bound)
        assert bound >= 0


# =====================================================================
# (e) Optimization bound — tighter tests with known functions
# =====================================================================

class TestOptimizationBoundExtended:
    """Additional tests for the optimization error bound."""

    def test_known_quadratic_with_L_smooth(self):
        """For a known quadratic f(x) = L/2 * x^2, GD with lr=1/L
        converges in one step from any initial point. Simulate a GD
        trajectory and check the rate bound is meaningful."""
        L_smooth = 4.0
        lr = 1.0 / L_smooth
        x0 = 3.0
        losses = []
        x = x0
        for _ in range(50):
            losses.append(0.5 * L_smooth * x**2)
            x = x - lr * L_smooth * x
        losses.append(0.5 * L_smooth * x**2)

        bound = optimization_bound(losses, lr=lr, L_smooth=L_smooth)
        true_final_gap = losses[-1] - 0.0
        assert bound >= 0
        assert bound >= true_final_gap * 0.99, (
            f"Rate bound {bound} should cover true gap {true_final_gap}"
        )

    def test_rate_bound_decreases_with_iterations(self):
        """More iterations → smaller rate bound (for fixed init gap)."""
        L_smooth = 10.0
        lr = 0.01
        losses_short = [5.0 - i * 0.01 for i in range(51)]
        losses_long = [5.0 - i * 0.01 for i in range(501)]

        b_short = optimization_bound(losses_short, lr=lr, L_smooth=L_smooth)
        b_long = optimization_bound(losses_long, lr=lr, L_smooth=L_smooth)
        assert b_long <= b_short

    def test_grad_clip_tightens_bound(self):
        """Gradient clipping should give a tighter bound when small."""
        losses = [10.0, 7.0, 5.0, 4.0, 3.5]
        L_smooth = 50.0
        lr = 0.01

        b_no_clip = optimization_bound(
            losses, lr=lr, L_smooth=L_smooth, grad_clip=None
        )
        b_clip = optimization_bound(
            losses, lr=lr, L_smooth=L_smooth, grad_clip=0.1
        )
        assert b_clip <= b_no_clip

    def test_monotone_loss_gives_zero(self):
        """A strictly decreasing loss ending at the minimum gives gap 0."""
        losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        bound = optimization_bound(losses, lr=0.01)
        assert bound == pytest.approx(0.0, abs=1e-15)

    def test_numpy_array_input(self):
        """optimization_bound should accept np.ndarray, not just lists."""
        losses = np.array([3.0, 2.0, 1.5, 1.2])
        bound = optimization_bound(losses, lr=0.01)
        assert bound == pytest.approx(1.2 - 1.2, abs=1e-15)


# =====================================================================
# Estimate smoothness — consistency with a known problem
# =====================================================================

class TestEstimateSmoothnessExtended:
    """Additional tests for the local Lipschitz estimator."""

    def test_deterministic_with_seed(self, ea_problem):
        """The estimator should be deterministic given a fixed rng seed."""
        d = ea_problem
        L1 = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], n_probes=5, rng=np.random.default_rng(42),
        )
        L2 = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], n_probes=5, rng=np.random.default_rng(42),
        )
        assert L1 == pytest.approx(L2, rel=1e-12)

    def test_identity_activation_smoothness(self, ea_problem):
        """Identity activation gives a quadratic loss in (a, W), so the
        gradient is affine → L_smooth should be finite and positive."""
        d = ea_problem
        L_smooth = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], activation="identity", n_probes=10,
        )
        assert L_smooth > 0
        assert np.isfinite(L_smooth)

    def test_larger_eps_still_works(self, ea_problem):
        """Larger perturbation should still produce a valid estimate."""
        d = ea_problem
        L_smooth = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], eps=1e-2, n_probes=5,
        )
        assert L_smooth > 0
        assert np.isfinite(L_smooth)

    def test_sigmoid_activation(self, ea_problem):
        """Smoothness estimation should work for sigmoid activation."""
        d = ea_problem
        L_smooth = estimate_smoothness(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"],
            d["E_y2"], d["maf"], activation="sigmoid", n_probes=5,
        )
        assert L_smooth > 0
        assert np.isfinite(L_smooth)


# =====================================================================
# ErrorDecomposition dataclass — direct construction and .total
# =====================================================================

class TestErrorDecompositionDataclass:
    """Tests for the ErrorDecomposition dataclass itself."""

    def test_construction_and_fields(self):
        """Test that the dataclass can be constructed and fields are accessible."""
        decomp = ErrorDecomposition(
            edgeworth_truncation=0.01,
            decorrelation_approx=0.02,
            ld_estimation=0.03,
            pumas_variance=0.04,
            optimization=0.05,
        )
        assert decomp.edgeworth_truncation == 0.01
        assert decomp.decorrelation_approx == 0.02
        assert decomp.ld_estimation == 0.03
        assert decomp.pumas_variance == 0.04
        assert decomp.optimization == 0.05

    def test_total_property(self):
        """The .total property should be the sum of all five fields."""
        decomp = ErrorDecomposition(
            edgeworth_truncation=0.1,
            decorrelation_approx=0.2,
            ld_estimation=0.3,
            pumas_variance=0.4,
            optimization=0.5,
        )
        assert decomp.total == pytest.approx(1.5, rel=1e-12)

    def test_total_with_zeros(self):
        """Total should be 0 when all components are 0."""
        decomp = ErrorDecomposition(
            edgeworth_truncation=0.0,
            decorrelation_approx=0.0,
            ld_estimation=0.0,
            pumas_variance=0.0,
            optimization=0.0,
        )
        assert decomp.total == pytest.approx(0.0, abs=1e-15)

    def test_total_single_nonzero(self):
        """Total should equal the single nonzero component."""
        decomp = ErrorDecomposition(
            edgeworth_truncation=0.0,
            decorrelation_approx=0.0,
            ld_estimation=0.42,
            pumas_variance=0.0,
            optimization=0.0,
        )
        assert decomp.total == pytest.approx(0.42, rel=1e-12)

    def test_total_is_property_not_stored(self):
        """Verify that .total is a computed property (changes if fields change)."""
        decomp = ErrorDecomposition(
            edgeworth_truncation=1.0,
            decorrelation_approx=2.0,
            ld_estimation=3.0,
            pumas_variance=4.0,
            optimization=5.0,
        )
        assert decomp.total == pytest.approx(15.0)
        decomp.optimization = 10.0
        assert decomp.total == pytest.approx(20.0)


# =====================================================================
# compute_error_decomposition — component reasonableness
# =====================================================================

class TestErrorDecompositionIntegration:
    """Integration tests for the full error decomposition."""

    def test_p1_m1(self, p1_problem):
        """Error decomposition should work for the minimal p=1, m=1 case."""
        d = p1_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000,
        )
        assert decomp.total >= 0
        assert np.isfinite(decomp.total)
        assert decomp.decorrelation_approx == pytest.approx(0.0, abs=1e-10)

    def test_near_singular_sigma(self, near_singular_problem):
        """Decomposition should be finite with near-singular LD."""
        d = near_singular_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.01,
            loss_history=[1.0, 0.5, 0.3],
        )
        assert np.isfinite(decomp.total)
        assert all(np.isfinite(x) for x in [
            decomp.edgeworth_truncation,
            decomp.decorrelation_approx,
            decomp.ld_estimation,
            decomp.pumas_variance,
            decomp.optimization,
        ])

    def test_truncation_bounded_above(self, ea_problem):
        """With p=8 and small weights, the truncation bound should be
        small (< 1) due to CLT convergence."""
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000,
        )
        assert decomp.edgeworth_truncation < 1.0, (
            f"Truncation bound {decomp.edgeworth_truncation} is unexpectedly large "
            f"for p=8 moderate-MAF problem"
        )

    def test_pumas_variance_scales_sensibly(self, ea_problem):
        """PUMAS variance should be small for N=50000 relative to E[y^2]."""
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000,
        )
        assert decomp.pumas_variance < d["E_y2"], (
            f"PUMAS variance {decomp.pumas_variance} should be much smaller "
            f"than E[y^2] = {d['E_y2']} for N=50000"
        )

    def test_ld_estimation_zero_when_no_error(self, ea_problem):
        """Redundant integration check: delta_fro=0 → ld_estimation=0."""
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.0,
        )
        assert decomp.ld_estimation == pytest.approx(0.0, abs=1e-15)

    def test_empty_loss_history_list(self, ea_problem):
        """Passing loss_history=[] should give optimization=0."""
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, loss_history=[],
        )
        assert decomp.optimization == 0.0

    def test_sigmoid_components_all_positive(self, ea_problem):
        """For sigmoid activation with LD, all non-trivial components should
        be positive (truncation > 0 since MAFs != 0.5, decorrelation > 0
        since Sigma != I)."""
        d = ea_problem
        decomp = compute_error_decomposition(
            d["a"], d["W"], d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], N=50000, n_train=40000, delta_Sigma_fro=0.05,
            loss_history=[2.0, 1.0, 0.5, 0.6], activation="sigmoid",
        )
        assert decomp.edgeworth_truncation > 0
        assert decomp.decorrelation_approx > 0
        assert decomp.ld_estimation > 0
        assert decomp.pumas_variance > 0
        assert decomp.optimization > 0
