"""Tests for optimizer.py: convergence, loss monotonicity, edge cases."""

import numpy as np
import pytest

from ssnn.optimizer import train, TrainResult
from ssnn.utils import generate_ld_matrix, generate_gwas_summary_stats


@pytest.fixture
def optimizer_problem():
    """A small problem for optimizer tests."""
    rng = np.random.default_rng(55)
    p = 6
    Sigma = generate_ld_matrix(p, n_blocks=2, decay=0.5)
    beta_star = rng.standard_normal(p) * 0.3
    sigma_eps = 1.0
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + sigma_eps**2)
    return Sigma, Sigma_beta, E_y2, sigma_eps, beta_star


# ---------------------------------------------------------------------------
# Basic convergence
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_identity_converges(self, optimizer_problem):
        Sigma, Sigma_beta, E_y2, sigma_eps, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=6, activation="identity",
            lr=0.005, max_iters=10000, tol=1e-10,
            rng=np.random.default_rng(0),
        )
        assert result.converged, (
            f"Identity optimizer did not converge in {result.n_iters} iters, "
            f"final loss={result.loss_history[-1]:.8f}"
        )

    def test_relu_converges(self, optimizer_problem):
        Sigma, Sigma_beta, E_y2, sigma_eps, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=6, activation="relu",
            lr=0.01, max_iters=10000, tol=1e-8,
            rng=np.random.default_rng(0),
        )
        assert result.converged, (
            f"ReLU optimizer did not converge in {result.n_iters} iters, "
            f"final loss={result.loss_history[-1]:.8f}"
        )

    def test_sigmoid_converges(self, optimizer_problem):
        Sigma, Sigma_beta, E_y2, sigma_eps, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=6, activation="sigmoid",
            lr=0.05, max_iters=10000, tol=1e-7,
            rng=np.random.default_rng(0),
        )
        assert result.converged, (
            f"Sigmoid optimizer did not converge in {result.n_iters} iters, "
            f"final loss={result.loss_history[-1]:.8f}"
        )


# ---------------------------------------------------------------------------
# Loss monotonicity
# ---------------------------------------------------------------------------

class TestLossMonotonicity:

    @pytest.mark.parametrize("activation", ["relu", "identity"])
    def test_loss_monotonically_decreasing(self, optimizer_problem, activation):
        """With a small enough learning rate, loss should decrease at every step."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=4, activation=activation,
            lr=0.001, max_iters=200, tol=1e-15,
            rng=np.random.default_rng(0),
        )
        losses = result.loss_history
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-12, (
                f"Loss increased at step {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}"
            )


# ---------------------------------------------------------------------------
# TrainResult structure
# ---------------------------------------------------------------------------

class TestTrainResult:

    def test_result_fields(self, optimizer_problem):
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=3, activation="relu",
            lr=0.01, max_iters=100, tol=1e-15,
            rng=np.random.default_rng(0),
        )
        assert isinstance(result, TrainResult)
        assert result.a.shape == (3,)
        assert result.W.shape == (3, 6)
        assert len(result.loss_history) > 0
        assert isinstance(result.converged, bool)
        assert result.n_iters == len(result.loss_history)

    def test_max_iters_not_converged(self, optimizer_problem):
        """With very few iterations and tight tolerance, should not converge."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=3, activation="relu",
            lr=0.001, max_iters=5, tol=1e-30,
            rng=np.random.default_rng(0),
        )
        assert not result.converged


# ---------------------------------------------------------------------------
# Edge cases: m=1, m >> p
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_hidden_unit(self, optimizer_problem):
        """m=1 should run without errors and produce a scalar a."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=1, activation="relu",
            lr=0.005, max_iters=2000, tol=1e-8,
            rng=np.random.default_rng(0),
        )
        assert result.a.shape == (1,)
        assert result.W.shape == (1, 6)
        assert len(result.loss_history) > 0

    def test_overparameterized(self, optimizer_problem):
        """m >> p should still run and converge."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        p = Sigma.shape[0]
        m = p * 4
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=m, activation="identity",
            lr=0.002, max_iters=5000, tol=1e-8,
            rng=np.random.default_rng(0),
        )
        assert result.a.shape == (m,)
        assert result.W.shape == (m, p)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_loss_improves_over_initial(self, optimizer_problem):
        """Final loss should be strictly less than initial loss."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=4, activation="relu",
            lr=0.005, max_iters=1000, tol=1e-15,
            rng=np.random.default_rng(0),
        )
        assert result.loss_history[-1] < result.loss_history[0]


# ---------------------------------------------------------------------------
# Sigmoid full pipeline
# ---------------------------------------------------------------------------

class TestSigmoidPipeline:

    def test_sigmoid_loss_decreases(self, optimizer_problem):
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=4, activation="sigmoid",
            lr=0.01, max_iters=500, tol=1e-15,
            rng=np.random.default_rng(0),
        )
        assert result.loss_history[-1] < result.loss_history[0]

    def test_sigmoid_final_loss_bounded(self, optimizer_problem):
        """Sigmoid final loss should be at most the zero-predictor loss (E[y^2])."""
        Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
        result = train(
            Sigma, Sigma_beta, E_y2,
            m=6, activation="sigmoid",
            lr=0.01, max_iters=3000, tol=1e-10,
            rng=np.random.default_rng(0),
        )
        assert result.loss_history[-1] < E_y2 + 1e-6


# ---------------------------------------------------------------------------
# Reproducibility with fixed rng
# ---------------------------------------------------------------------------

def test_deterministic_with_same_rng(optimizer_problem):
    Sigma, Sigma_beta, E_y2, _, _ = optimizer_problem
    result1 = train(
        Sigma, Sigma_beta, E_y2,
        m=3, activation="relu",
        lr=0.01, max_iters=100, tol=1e-15,
        rng=np.random.default_rng(42),
    )
    result2 = train(
        Sigma, Sigma_beta, E_y2,
        m=3, activation="relu",
        lr=0.01, max_iters=100, tol=1e-15,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_allclose(result1.a, result2.a, atol=1e-15)
    np.testing.assert_allclose(result1.W, result2.W, atol=1e-15)
    np.testing.assert_allclose(result1.loss_history, result2.loss_history, atol=1e-15)
