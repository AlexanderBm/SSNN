"""Tests for the PUMAS pseudo-subset summary statistics module."""

import numpy as np
import pytest

from ssnn.pumas import (
    generate_pumas_split,
    generate_pumas_splits,
    pumas_summary_r2,
    pumas_nn_summary_r2,
)
from ssnn.utils import linear_prs_weights


class TestGeneratePumasSplit:
    def test_shapes(self, small_problem):
        sp = small_problem
        rng = np.random.default_rng(0)

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=rng,
        )

        assert split.Sigma_beta_train.shape == (sp["p"],)
        assert split.Sigma_beta_val.shape == (sp["p"],)
        assert split.n_train == 8000
        assert split.n_val == 2000
        assert split.E_y2_train > 0
        assert split.E_y2_val > 0

    def test_residual_consistency(self, small_problem):
        """Weighted sum of train + val should reconstruct the full-sample stats."""
        sp = small_problem
        rng = np.random.default_rng(1)
        N = 10000
        n_train = 8000

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=N, n_train=n_train, rng=rng,
        )

        # n_train * Sb_train + n_val * Sb_val = N * Sb_full
        reconstructed = (n_train * split.Sigma_beta_train
                         + split.n_val * split.Sigma_beta_val) / N
        np.testing.assert_allclose(
            reconstructed, sp["Sigma_beta"], atol=1e-10,
        )

    def test_error_if_n_train_too_large(self, small_problem):
        sp = small_problem
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="n_train"):
            generate_pumas_split(
                sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
                N=100, n_train=100, rng=rng,
            )

    def test_splits_are_stochastic(self, small_problem):
        """Different seeds produce different splits."""
        sp = small_problem
        s1 = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(10),
        )
        s2 = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(11),
        )
        assert not np.allclose(s1.Sigma_beta_train, s2.Sigma_beta_train)


class TestGeneratePumasSplits:
    def test_n_splits(self, small_problem):
        sp = small_problem
        splits = generate_pumas_splits(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_splits=3,
        )
        assert len(splits) == 3

    def test_splits_differ(self, small_problem):
        sp = small_problem
        splits = generate_pumas_splits(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_splits=3,
        )
        assert not np.allclose(
            splits[0].Sigma_beta_train, splits[1].Sigma_beta_train,
        )


class TestPumasSummaryR2:
    def test_oracle_weights_positive_r2(self, small_problem):
        """The true beta* should get positive summary-stat R² on validation."""
        sp = small_problem
        rng = np.random.default_rng(42)
        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=50000, n_train=40000, rng=rng,
        )
        r2 = pumas_summary_r2(
            split.Sigma_beta_val, sp["beta_star"],
            sp["Sigma"], split.E_y2_val,
        )
        assert r2 > 0.0

    def test_zero_weights_zero_r2(self, small_problem):
        sp = small_problem
        rng = np.random.default_rng(42)
        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=rng,
        )
        r2 = pumas_summary_r2(
            split.Sigma_beta_val, np.zeros(sp["p"]),
            sp["Sigma"], split.E_y2_val,
        )
        assert abs(r2) < 1e-10

    def test_linear_prs_positive_r2(self, small_problem):
        """Linear PRS from training stats should get positive val R²."""
        sp = small_problem
        rng = np.random.default_rng(42)
        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=50000, n_train=40000, rng=rng,
        )
        beta_hat = linear_prs_weights(sp["Sigma"], split.Sigma_beta_train)
        r2 = pumas_summary_r2(
            split.Sigma_beta_val, beta_hat,
            sp["Sigma"], split.E_y2_val,
        )
        assert r2 > 0.0


class TestPumasNNSummaryR2:
    def test_gaussian_nn_r2(self, small_problem):
        """Basic smoke test that NN R² computation runs without error."""
        sp = small_problem
        m = 3
        rng = np.random.default_rng(42)
        a = rng.standard_normal(m) * 0.01
        W = rng.standard_normal((m, sp["p"])) * 0.01

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(0),
        )
        r2 = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"], activation="relu",
        )
        assert np.isfinite(r2)

    def test_edgeworth_raises_without_maf(self, small_problem):
        """use_edgeworth=True with maf=None must raise ValueError."""
        sp = small_problem
        m = 3
        rng = np.random.default_rng(42)
        a = rng.standard_normal(m) * 0.01
        W = rng.standard_normal((m, sp["p"])) * 0.01

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match="maf required"):
            pumas_nn_summary_r2(
                split.Sigma_beta_val, split.E_y2_val,
                a, W, sp["Sigma"],
                maf=None, activation="relu", use_edgeworth=True,
            )

    def test_edgeworth_differs_from_gaussian_skewed_maf(self, small_problem):
        """With skewed MAFs, Edgeworth R² should differ from Gaussian R²."""
        sp = small_problem
        m = 3
        rng = np.random.default_rng(42)
        a = rng.standard_normal(m) * 0.1
        W = rng.standard_normal((m, sp["p"])) * 0.1
        maf = np.full(sp["p"], 0.05)

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(0),
        )
        r2_gauss = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"],
            activation="relu", use_edgeworth=False,
        )
        r2_ew = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"],
            maf=maf, activation="relu", use_edgeworth=True,
        )
        assert np.isfinite(r2_gauss)
        assert np.isfinite(r2_ew)
        assert r2_gauss != pytest.approx(r2_ew, abs=1e-10), (
            "Edgeworth and Gaussian R² should differ for skewed MAFs"
        )

    def test_edgeworth_matches_gaussian_maf_0_5(self, small_problem):
        """With symmetric MAFs (0.5), Edgeworth R² ≈ Gaussian R²."""
        sp = small_problem
        m = 3
        rng = np.random.default_rng(42)
        a = rng.standard_normal(m) * 0.01
        W = rng.standard_normal((m, sp["p"])) * 0.01
        maf = np.full(sp["p"], 0.5)

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(0),
        )
        r2_gauss = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"],
            activation="relu", use_edgeworth=False,
        )
        r2_ew = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"],
            maf=maf, activation="relu", use_edgeworth=True,
        )
        assert r2_ew == pytest.approx(r2_gauss, abs=1e-2)

    def test_zero_nn_weights_r2_near_zero(self, small_problem):
        """Zero NN weights should give R² ≈ 0 (loss ≈ E_y2)."""
        sp = small_problem
        m = 3
        a = np.zeros(m)
        W = np.zeros((m, sp["p"]))

        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=np.random.default_rng(0),
        )
        r2 = pumas_nn_summary_r2(
            split.Sigma_beta_val, split.E_y2_val,
            a, W, sp["Sigma"], activation="relu",
        )
        assert abs(r2) < 1e-8


class TestPumasCovarianceScaling:
    """Tests that the PUMAS split covariance scales correctly with n_train/N."""

    def test_noise_variance_scales_with_fraction(self, small_problem):
        """The variance of Sigma_beta_train around its mean should scale
        as (N - n_train) / N^2 * trace(Sigma)."""
        sp = small_problem
        N = 100000
        n_samples = 500

        for n_train in [20000, 50000, 80000]:
            trains = []
            for i in range(n_samples):
                rng = np.random.default_rng(i)
                s = generate_pumas_split(
                    sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
                    N=N, n_train=n_train, rng=rng,
                )
                trains.append(s.Sigma_beta_train)

            trains = np.array(trains)
            frac = n_train / N
            empirical_mean = trains.mean(axis=0)
            np.testing.assert_allclose(
                empirical_mean, frac * sp["Sigma_beta"], atol=0.01,
            )

            n_val = N - n_train
            expected_cov_scale = float(n_val) / (float(N) ** 2)
            empirical_var_trace = np.var(trains, axis=0).sum()
            expected_var_trace = expected_cov_scale * np.trace(sp["Sigma"])
            assert empirical_var_trace == pytest.approx(
                expected_var_trace, rel=0.15,
            )


class TestPumasSplitsEdgeCases:
    """Edge case tests for PUMAS splitting."""

    def test_single_split(self, small_problem):
        sp = small_problem
        splits = generate_pumas_splits(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_splits=1,
        )
        assert len(splits) == 1
        assert splits[0].n_train + splits[0].n_val == 10000

    def test_extreme_train_fraction_low(self, small_problem):
        """Very low train_fraction is clamped to at least 1."""
        sp = small_problem
        splits = generate_pumas_splits(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=100, n_splits=1, train_fraction=0.001,
        )
        assert splits[0].n_train >= 1
        assert splits[0].n_val == 100 - splits[0].n_train

    def test_extreme_train_fraction_high(self, small_problem):
        """Very high train_fraction is clamped to N-1."""
        sp = small_problem
        splits = generate_pumas_splits(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=100, n_splits=1, train_fraction=0.999,
        )
        assert splits[0].n_train <= 99

    def test_summary_r2_negative_for_bad_weights(self, small_problem):
        """Random garbage weights should give negative or near-zero R²."""
        sp = small_problem
        rng = np.random.default_rng(42)
        split = generate_pumas_split(
            sp["Sigma_beta"], sp["E_y2"], sp["Sigma"],
            N=10000, n_train=8000, rng=rng,
        )
        bad_weights = rng.standard_normal(sp["p"]) * 100.0
        r2 = pumas_summary_r2(
            split.Sigma_beta_val, bad_weights, sp["Sigma"], split.E_y2_val,
        )
        assert r2 < 0.5

    def test_summary_r2_zero_E_y2_returns_zero(self, small_problem):
        """E_y2_val <= 0 should return 0.0."""
        sp = small_problem
        r2 = pumas_summary_r2(
            sp["Sigma_beta"], sp["beta_star"], sp["Sigma"], E_y2_val=0.0,
        )
        assert r2 == 0.0

        r2_neg = pumas_summary_r2(
            sp["Sigma_beta"], sp["beta_star"], sp["Sigma"], E_y2_val=-1.0,
        )
        assert r2_neg == 0.0

    def test_nn_r2_zero_E_y2_returns_zero(self, small_problem):
        """pumas_nn_summary_r2 with E_y2_val <= 0 returns 0.0."""
        sp = small_problem
        m = 2
        rng = np.random.default_rng(0)
        a = rng.standard_normal(m) * 0.1
        W = rng.standard_normal((m, sp["p"])) * 0.1

        r2 = pumas_nn_summary_r2(
            sp["Sigma_beta"], 0.0,
            a, W, sp["Sigma"], activation="relu",
        )
        assert r2 == 0.0
