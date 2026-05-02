"""Tests for utils.py: LD generation, GWAS simulation, prediction utilities."""

import numpy as np
import pytest

from ssnn.utils import (
    generate_ld_matrix,
    generate_gwas_summary_stats,
    linear_prs_weights,
    prediction_r2,
    nn_predict,
    nn_prediction_r2,
)


# ---------------------------------------------------------------------------
# generate_ld_matrix
# ---------------------------------------------------------------------------

class TestGenerateLDMatrix:

    @pytest.mark.parametrize("p", [1, 5, 10, 20, 50])
    def test_positive_definite(self, p):
        Sigma = generate_ld_matrix(p)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min()}"

    @pytest.mark.parametrize("p", [1, 5, 10, 20])
    def test_symmetric(self, p):
        Sigma = generate_ld_matrix(p)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-15)

    def test_unit_diagonal(self):
        """Correlation-type LD matrix should have 1s on the diagonal."""
        Sigma = generate_ld_matrix(10, decay=0.5)
        np.testing.assert_allclose(np.diag(Sigma), 1.0, atol=1e-15)

    def test_block_structure(self):
        """With n_blocks=2 and p=10, off-block entries should be zero."""
        Sigma = generate_ld_matrix(10, n_blocks=2)
        assert Sigma[0, 5] == 0.0
        assert Sigma[4, 5] == 0.0
        assert Sigma[0, 1] != 0.0
        assert Sigma[5, 6] != 0.0

    def test_decay_controls_correlation(self):
        """Higher decay should produce stronger off-diagonal correlations."""
        Sigma_low = generate_ld_matrix(10, n_blocks=1, decay=0.2)
        Sigma_high = generate_ld_matrix(10, n_blocks=1, decay=0.9)
        assert abs(Sigma_high[0, 1]) > abs(Sigma_low[0, 1])

    def test_single_snp(self):
        """p=1 should be a 1x1 identity."""
        Sigma = generate_ld_matrix(1)
        np.testing.assert_allclose(Sigma, np.array([[1.0]]))

    @pytest.mark.parametrize("p,n_blocks", [(7, 3), (11, 4), (3, 2)])
    def test_non_divisible_block_sizes(self, p, n_blocks):
        """Block allocation should work when p is not divisible by n_blocks."""
        Sigma = generate_ld_matrix(p, n_blocks=n_blocks)
        assert Sigma.shape == (p, p)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

    def test_identity_ld(self):
        """decay=0 should give a block-diagonal with only 1s on diag (identity)."""
        Sigma = generate_ld_matrix(10, n_blocks=2, decay=0.0)
        np.testing.assert_allclose(Sigma, np.eye(10), atol=1e-15)


# ---------------------------------------------------------------------------
# generate_gwas_summary_stats
# ---------------------------------------------------------------------------

class TestGenerateGWASSummaryStats:

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        Sigma = generate_ld_matrix(5)
        beta = rng.standard_normal(5) * 0.3
        stats = generate_gwas_summary_stats(Sigma, beta, n=100, rng=rng)

        expected_keys = {"Sigma_beta", "Sigma_beta_hat", "E_y2", "E_y2_hat",
                         "Sigma", "beta_star", "n", "sigma_eps"}
        assert set(stats.keys()) == expected_keys

    def test_individual_data_returned_when_requested(self):
        rng = np.random.default_rng(0)
        Sigma = generate_ld_matrix(5)
        beta = rng.standard_normal(5) * 0.3
        stats = generate_gwas_summary_stats(
            Sigma, beta, n=100, rng=rng, return_individual_data=True
        )
        assert "X" in stats and "y" in stats
        assert stats["X"].shape == (100, 5)
        assert stats["y"].shape == (100,)

    def test_population_sigma_beta_exact(self):
        """Sigma_beta should be exactly Sigma @ beta_star."""
        rng = np.random.default_rng(1)
        Sigma = generate_ld_matrix(8)
        beta = rng.standard_normal(8) * 0.3
        stats = generate_gwas_summary_stats(Sigma, beta, n=100, rng=rng)
        np.testing.assert_allclose(stats["Sigma_beta"], Sigma @ beta, atol=1e-14)

    def test_population_E_y2_exact(self):
        """E_y2 should be exactly beta^T Sigma beta + sigma_eps^2."""
        rng = np.random.default_rng(2)
        p = 8
        Sigma = generate_ld_matrix(p)
        beta = rng.standard_normal(p) * 0.3
        sigma_eps = 1.5
        stats = generate_gwas_summary_stats(Sigma, beta, n=100, sigma_eps=sigma_eps, rng=rng)
        expected = beta @ Sigma @ beta + sigma_eps**2
        assert stats["E_y2"] == pytest.approx(expected, abs=1e-12)

    def test_finite_sample_converges_to_population(self):
        """Sigma_beta_hat and E_y2_hat should converge to population values as n grows."""
        rng = np.random.default_rng(42)
        p = 6
        Sigma = generate_ld_matrix(p, n_blocks=2)
        beta = rng.standard_normal(p) * 0.3

        errors_sbeta = []
        errors_ey2 = []
        for n in [1_000, 10_000, 100_000]:
            stats = generate_gwas_summary_stats(Sigma, beta, n=n, rng=np.random.default_rng(42))
            err_sb = np.linalg.norm(stats["Sigma_beta_hat"] - stats["Sigma_beta"])
            err_ey = abs(stats["E_y2_hat"] - stats["E_y2"])
            errors_sbeta.append(err_sb)
            errors_ey2.append(err_ey)

        assert errors_sbeta[-1] < errors_sbeta[0]
        assert errors_ey2[-1] < errors_ey2[0]

        # With n=100k the estimates should be within ~1% of population
        stats_big = generate_gwas_summary_stats(
            Sigma, beta, n=500_000, rng=np.random.default_rng(99)
        )
        np.testing.assert_allclose(
            stats_big["Sigma_beta_hat"], stats_big["Sigma_beta"], atol=0.02
        )
        assert stats_big["E_y2_hat"] == pytest.approx(stats_big["E_y2"], rel=0.02)


# ---------------------------------------------------------------------------
# linear_prs_weights
# ---------------------------------------------------------------------------

class TestLinearPRSWeights:

    def test_recovers_beta_star(self):
        """Sigma^{-1} Sigma_beta = Sigma^{-1} Sigma beta* = beta*."""
        rng = np.random.default_rng(10)
        p = 8
        Sigma = generate_ld_matrix(p, n_blocks=2)
        beta = rng.standard_normal(p) * 0.3
        Sigma_beta = Sigma @ beta
        recovered = linear_prs_weights(Sigma, Sigma_beta)
        np.testing.assert_allclose(recovered, beta, atol=1e-10)


# ---------------------------------------------------------------------------
# prediction_r2
# ---------------------------------------------------------------------------

class TestPredictionR2:

    def test_perfect_prediction(self):
        rng = np.random.default_rng(20)
        n, p = 1000, 5
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p)
        y = X @ beta
        r2 = prediction_r2(X, y, beta)
        assert r2 == pytest.approx(1.0, abs=1e-10)

    def test_zero_prediction(self):
        """Predicting zeros should give R^2 ~ 0 when weights are zero."""
        rng = np.random.default_rng(21)
        n, p = 1000, 5
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        r2 = prediction_r2(X, y, np.zeros(p))
        # R^2 = 1 - MSE/Var(y); MSE = Var(y) + mean(y)^2, so R^2 can be negative
        assert r2 < 0.1

    def test_noisy_prediction(self):
        """With noise, R^2 should be between 0 and 1."""
        rng = np.random.default_rng(22)
        n, p = 10000, 5
        Sigma = generate_ld_matrix(p, n_blocks=1)
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        beta = rng.standard_normal(p) * 0.5
        y = X @ beta + rng.normal(0, 1, n)
        r2 = prediction_r2(X, y, beta)
        assert 0 < r2 < 1


# ---------------------------------------------------------------------------
# nn_predict
# ---------------------------------------------------------------------------

class TestNNPredict:

    def test_relu_activation(self):
        rng = np.random.default_rng(30)
        n, m, p = 50, 3, 4
        X = rng.standard_normal((n, p))
        W = rng.standard_normal((m, p))
        a = rng.standard_normal(m)

        y_pred = nn_predict(X, a, W, "relu")
        expected = np.maximum(0, X @ W.T) @ a
        np.testing.assert_allclose(y_pred, expected, atol=1e-14)

    def test_sigmoid_activation(self):
        rng = np.random.default_rng(31)
        n, m, p = 50, 3, 4
        X = rng.standard_normal((n, p))
        W = rng.standard_normal((m, p))
        a = rng.standard_normal(m)

        y_pred = nn_predict(X, a, W, "sigmoid")
        expected = (1.0 / (1.0 + np.exp(-(X @ W.T)))) @ a
        np.testing.assert_allclose(y_pred, expected, atol=1e-14)

    def test_identity_activation(self):
        rng = np.random.default_rng(32)
        n, m, p = 50, 3, 4
        X = rng.standard_normal((n, p))
        W = rng.standard_normal((m, p))
        a = rng.standard_normal(m)

        y_pred = nn_predict(X, a, W, "identity")
        expected = (X @ W.T) @ a
        np.testing.assert_allclose(y_pred, expected, atol=1e-14)

    def test_unknown_activation_raises(self):
        X = np.zeros((2, 3))
        W = np.zeros((1, 3))
        a = np.zeros(1)
        with pytest.raises(ValueError, match="Unknown activation"):
            nn_predict(X, a, W, "tanh")

    def test_single_hidden_unit(self):
        """m=1 should still produce correct output."""
        rng = np.random.default_rng(33)
        n, p = 20, 5
        X = rng.standard_normal((n, p))
        W = rng.standard_normal((1, p))
        a = np.array([2.0])

        y_pred = nn_predict(X, a, W, "relu")
        expected = 2.0 * np.maximum(0, X @ W[0])
        np.testing.assert_allclose(y_pred, expected, atol=1e-14)

    def test_zero_weights_give_zero_output(self):
        n, m, p = 10, 3, 4
        X = np.random.default_rng(0).standard_normal((n, p))
        y_pred = nn_predict(X, np.zeros(m), np.zeros((m, p)), "relu")
        np.testing.assert_allclose(y_pred, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# nn_prediction_r2
# ---------------------------------------------------------------------------

class TestNNPredictionR2:

    def test_identity_matches_linear(self):
        """For identity activation, nn_prediction_r2 should match linear prediction_r2
        when Wa = weights."""
        rng = np.random.default_rng(40)
        n, p = 5000, 5
        Sigma = generate_ld_matrix(p, n_blocks=1)
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        beta = rng.standard_normal(p) * 0.3
        y = X @ beta + rng.normal(0, 1, n)

        # Use identity NN with W=I, a=beta
        W = np.eye(p)
        a = beta
        r2_nn = nn_prediction_r2(X, y, a, W, "identity")
        r2_lin = prediction_r2(X, y, beta)
        assert r2_nn == pytest.approx(r2_lin, abs=1e-10)
