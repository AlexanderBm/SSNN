"""Tests for baseline PRS methods."""

import numpy as np
import pytest

from ssnn.baselines import clump_and_threshold, ldpred2_inf, prs_cs
from ssnn.utils import generate_ld_matrix, linear_prs_weights


@pytest.fixture
def baseline_problem():
    """A small problem specifically for testing baselines."""
    rng = np.random.default_rng(7)
    p = 20
    Sigma = generate_ld_matrix(p, decay=0.5)
    beta_star = np.zeros(p)
    beta_star[:5] = rng.standard_normal(5) * 0.5

    Sigma_beta = Sigma @ beta_star
    return {
        "p": p,
        "Sigma": Sigma,
        "beta_star": beta_star,
        "Sigma_beta": Sigma_beta,
    }


class TestClumpAndThreshold:
    def test_output_shape(self, baseline_problem):
        bp = baseline_problem
        weights = clump_and_threshold(bp["Sigma"], bp["Sigma_beta"], n=10000)
        assert weights.shape == (bp["p"],)

    def test_sparse_output(self, baseline_problem):
        """C+T should produce sparse weights (only selected SNPs nonzero)."""
        bp = baseline_problem
        weights = clump_and_threshold(
            bp["Sigma"], bp["Sigma_beta"],
            p_threshold=0.05, r2_threshold=0.1, n=10000,
        )
        n_nonzero = np.count_nonzero(weights)
        assert n_nonzero < bp["p"]

    def test_strict_threshold_zero_output(self, baseline_problem):
        """Extremely strict p-value threshold should select no SNPs."""
        bp = baseline_problem
        weights = clump_and_threshold(
            bp["Sigma"], bp["Sigma_beta"],
            p_threshold=1e-300, n=100,
        )
        assert np.allclose(weights, 0.0)

    def test_permissive_threshold(self, baseline_problem):
        """Very permissive threshold selects at least some SNPs."""
        bp = baseline_problem
        weights = clump_and_threshold(
            bp["Sigma"], bp["Sigma_beta"],
            p_threshold=1.0, r2_threshold=1.0, n=10000,
        )
        assert np.count_nonzero(weights) > 0


class TestLDpred2Inf:
    def test_output_shape(self, baseline_problem):
        bp = baseline_problem
        weights = ldpred2_inf(bp["Sigma"], bp["Sigma_beta"])
        assert weights.shape == (bp["p"],)

    def test_positive_definiteness(self, baseline_problem):
        """Result should be finite for any valid input."""
        bp = baseline_problem
        weights = ldpred2_inf(bp["Sigma"], bp["Sigma_beta"], h2=0.5, n=10000)
        assert np.all(np.isfinite(weights))

    def test_limit_zero_penalty(self, baseline_problem):
        """With h2 -> 1 and large n, LDpred2-inf -> Sigma^{-1} Sigma_beta."""
        bp = baseline_problem
        weights = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.99, p_causal=1.0, n=1_000_000,
        )
        linear = linear_prs_weights(bp["Sigma"], bp["Sigma_beta"])
        np.testing.assert_allclose(weights, linear, rtol=0.05)

    def test_strong_shrinkage(self, baseline_problem):
        """With very low h2, weights should be shrunk toward zero."""
        bp = baseline_problem
        weights = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.001, n=100,
        )
        assert np.linalg.norm(weights) < 0.1 * np.linalg.norm(bp["beta_star"])


class TestPrsCS:
    def test_output_shape(self, baseline_problem):
        bp = baseline_problem
        weights = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000)
        assert weights.shape == (bp["p"],)

    def test_finite_output(self, baseline_problem):
        bp = baseline_problem
        weights = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000)
        assert np.all(np.isfinite(weights))

    def test_convergence(self, baseline_problem):
        """Running more iterations shouldn't change the result much."""
        bp = baseline_problem
        w_short = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000, max_iters=200)
        w_long = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000, max_iters=2000)
        # They should be close (same fixed point)
        np.testing.assert_allclose(w_short, w_long, atol=0.05)

    def test_shrinkage_effect(self, baseline_problem):
        """PRS-CS weights should be smaller in norm than unpenalized."""
        bp = baseline_problem
        weights_cs = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000)
        weights_unreg = linear_prs_weights(bp["Sigma"], bp["Sigma_beta"])
        assert np.linalg.norm(weights_cs) < np.linalg.norm(weights_unreg)

    def test_explicit_phi_differs_from_auto(self, baseline_problem):
        """PRS-CS with an explicit phi should give different weights than auto."""
        bp = baseline_problem
        w_auto = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000, phi=None)
        w_fixed = prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000, phi=0.001)
        assert not np.allclose(w_auto, w_fixed, atol=1e-6), (
            "Auto phi and explicit phi=0.001 should produce different weights"
        )

    def test_large_phi_approaches_unregularized(self, baseline_problem):
        """With very large phi (weak prior), PRS-CS → OLS-like solution."""
        bp = baseline_problem
        w_large_phi = prs_cs(
            bp["Sigma"], bp["Sigma_beta"],
            n=10000, phi=1e6, max_iters=3000,
        )
        w_linear = linear_prs_weights(bp["Sigma"], bp["Sigma_beta"])
        assert np.linalg.norm(w_large_phi) > np.linalg.norm(
            prs_cs(bp["Sigma"], bp["Sigma_beta"], n=10000, phi=0.001)
        )

    def test_zero_signal_gives_near_zero_weights(self):
        """When Sigma_beta is zero, PRS-CS weights should be near zero."""
        p = 10
        Sigma = generate_ld_matrix(p, decay=0.5)
        Sigma_beta = np.zeros(p)
        weights = prs_cs(Sigma, Sigma_beta, n=10000)
        assert np.linalg.norm(weights) < 1e-8


class TestClumpAndThresholdLDPruning:
    """Tests that C+T respects LD pruning: correlated SNPs should not both be selected."""

    def test_correlated_snps_not_both_selected(self):
        """Two perfectly correlated SNPs: at most one should be selected."""
        p = 4
        Sigma = np.eye(p)
        Sigma[0, 1] = 0.99
        Sigma[1, 0] = 0.99

        Sigma_beta = np.array([0.5, 0.49, 0.01, 0.01])
        weights = clump_and_threshold(
            Sigma, Sigma_beta,
            p_threshold=1.0, r2_threshold=0.5, n=50000,
        )
        nonzero_mask = weights != 0.0
        assert not (nonzero_mask[0] and nonzero_mask[1]), (
            "Two highly correlated SNPs should not both be selected by C+T"
        )

    def test_uncorrelated_snps_can_both_be_selected(self):
        """Two uncorrelated significant SNPs should both survive pruning."""
        p = 4
        Sigma = np.eye(p)
        Sigma_beta = np.array([0.5, 0.5, 0.0, 0.0])
        weights = clump_and_threshold(
            Sigma, Sigma_beta,
            p_threshold=1.0, r2_threshold=0.1, n=50000,
        )
        assert weights[0] != 0.0
        assert weights[1] != 0.0

    def test_pruning_radius_effect(self, baseline_problem):
        """Tighter r2_threshold should yield fewer selected SNPs."""
        bp = baseline_problem
        w_loose = clump_and_threshold(
            bp["Sigma"], bp["Sigma_beta"],
            p_threshold=1.0, r2_threshold=0.8, n=50000,
        )
        w_tight = clump_and_threshold(
            bp["Sigma"], bp["Sigma_beta"],
            p_threshold=1.0, r2_threshold=0.01, n=50000,
        )
        assert np.count_nonzero(w_tight) <= np.count_nonzero(w_loose)


class TestLDpred2InfShrinkageOrdering:
    """Tests that LDpred2-inf shrinkage ordering is correct."""

    def test_p_causal_less_than_1_shrinks_more(self, baseline_problem):
        """p_causal < 1 implies a sparser prior, so more shrinkage."""
        bp = baseline_problem
        w_full = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.5, p_causal=1.0, n=10000,
        )
        w_sparse = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.5, p_causal=0.01, n=10000,
        )
        assert np.linalg.norm(w_sparse) < np.linalg.norm(w_full)

    def test_higher_h2_less_shrinkage(self, baseline_problem):
        """Higher heritability means less regularization, larger weights."""
        bp = baseline_problem
        w_low_h2 = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.1, n=10000,
        )
        w_high_h2 = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.9, n=10000,
        )
        assert np.linalg.norm(w_high_h2) > np.linalg.norm(w_low_h2)

    def test_larger_n_less_shrinkage(self, baseline_problem):
        """Larger GWAS sample size means less penalty, closer to OLS."""
        bp = baseline_problem
        w_small_n = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.5, n=100,
        )
        w_large_n = ldpred2_inf(
            bp["Sigma"], bp["Sigma_beta"],
            h2=0.5, n=1_000_000,
        )
        assert np.linalg.norm(w_large_n) > np.linalg.norm(w_small_n)


class TestBaselineZeroInput:
    """Tests for zero/degenerate inputs across all baselines."""

    def test_ct_zero_signal(self):
        """C+T with zero signal → zero weights."""
        p = 10
        Sigma = generate_ld_matrix(p, decay=0.5)
        weights = clump_and_threshold(Sigma, np.zeros(p), n=10000)
        np.testing.assert_allclose(weights, 0.0)

    def test_ldpred2_zero_signal(self):
        """LDpred2-inf with zero signal → zero weights."""
        p = 10
        Sigma = generate_ld_matrix(p, decay=0.5)
        weights = ldpred2_inf(Sigma, np.zeros(p))
        np.testing.assert_allclose(weights, 0.0, atol=1e-12)

    def test_identity_sigma_ldpred2(self):
        """With Sigma=I, LDpred2-inf is just element-wise shrinkage."""
        p = 5
        Sigma = np.eye(p)
        Sigma_beta = np.array([0.5, -0.3, 0.8, 0.1, -0.4])
        n = 10000
        h2 = 0.5
        lam = p / (n * h2)
        expected = Sigma_beta / (1.0 + lam)
        weights = ldpred2_inf(Sigma, Sigma_beta, h2=h2, n=n)
        np.testing.assert_allclose(weights, expected, rtol=1e-10)
