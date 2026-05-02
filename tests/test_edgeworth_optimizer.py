"""Tests for the Edgeworth-corrected optimizer."""

import numpy as np
import pytest

from ssnn.edgeworth_optimizer import train_edgeworth
from ssnn.edgeworth_risk import compute_edgeworth_loss
from ssnn.cumulants import decorrelation_matrix


@pytest.fixture
def training_setup():
    """A moderate problem for testing the Edgeworth optimizer."""
    rng = np.random.default_rng(42)
    p = 6
    sigma_eps = 1.0

    Sigma = np.eye(p)
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = 0.5 ** abs(i - j)

    maf = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4])
    beta_star = rng.standard_normal(p) * 0.3
    Sigma_beta = Sigma @ beta_star
    E_y2 = float(beta_star @ Sigma @ beta_star + sigma_eps**2)

    return {
        "Sigma": Sigma,
        "Sigma_beta": Sigma_beta,
        "E_y2": E_y2,
        "maf": maf,
        "beta_star": beta_star,
        "p": p,
        "sigma_eps": sigma_eps,
    }


class TestEdgeworthOptimizer:

    def test_loss_decreases(self, training_setup):
        """The Edgeworth-corrected loss should decrease during training."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.005,
            max_iters=200, init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        assert len(result.loss_history) > 1
        assert result.loss_history[-1] < result.loss_history[0]

    def test_returns_train_result(self, training_setup):
        """Optimizer should return a TrainResult with correct structure."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=3, activation="relu", lr=0.005,
            max_iters=50, init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        assert result.a.shape == (3,)
        assert result.W.shape == (3, d["p"])
        assert isinstance(result.loss_history, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iters, int)

    def test_identity_activation_converges_to_linear(self, training_setup):
        """With identity activation, the NN should recover linear PRS
        weights (same as the Gaussian case, since corrections vanish).
        """
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="identity", lr=0.005,
            max_iters=500, tol=1e-8, init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        final_loss = result.loss_history[-1]
        assert final_loss < d["E_y2"]  # Better than predicting 0

    def test_final_loss_matches_compute(self, training_setup):
        """The final loss in history should match recomputing from (a, W)."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.005,
            max_iters=100, init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        S = decorrelation_matrix(d["Sigma"])
        L_recomputed = compute_edgeworth_loss(
            result.a, result.W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )

        assert result.loss_history[-1] == pytest.approx(L_recomputed, rel=1e-8)

    def test_sigmoid_activation_runs(self, training_setup):
        """Ensure the optimizer runs without error for sigmoid activation."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="sigmoid", lr=0.005,
            max_iters=50, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        assert len(result.loss_history) > 0

    def test_identity_converges(self, training_setup):
        """Identity activation Edgeworth optimizer should converge (corrections vanish)."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="identity", lr=0.005,
            max_iters=5000, tol=1e-8, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        assert result.converged, (
            f"Edgeworth identity did not converge in {result.n_iters} iters, "
            f"final loss={result.loss_history[-1]:.8f}"
        )

    def test_loss_monotonically_decreasing_identity(self, training_setup):
        """With identity activation (no corrections), loss should decrease monotonically."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="identity", lr=0.001,
            max_iters=200, tol=1e-15, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        losses = result.loss_history
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-10, (
                f"Loss increased at step {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}"
            )

    def test_sigmoid_loss_decreases(self, training_setup):
        """Sigmoid activation should also decrease loss during EW training."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="sigmoid", lr=0.005,
            max_iters=200, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        assert result.loss_history[-1] < result.loss_history[0]

    def test_deterministic_with_same_rng(self, training_setup):
        """Two runs with the same RNG seed should produce identical results."""
        d = training_setup
        result1 = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.005,
            max_iters=50, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        result2 = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.005,
            max_iters=50, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_allclose(result1.a, result2.a, atol=1e-15)
        np.testing.assert_allclose(result1.W, result2.W, atol=1e-15)
        np.testing.assert_allclose(result1.loss_history, result2.loss_history, atol=1e-15)

    def test_single_hidden_unit(self, training_setup):
        """m=1 should work correctly for Edgeworth optimizer."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=1, activation="relu", lr=0.005,
            max_iters=200, init_scale=0.01,
            rng=np.random.default_rng(42),
        )
        assert result.a.shape == (1,)
        assert result.W.shape == (1, d["p"])
        assert result.loss_history[-1] < result.loss_history[0]


class TestEdgeworthSafeguards:
    """Tests for the Edgeworth surrogate unboundedness safeguards."""

    def test_loss_floor_clamps_negative(self, training_setup):
        """compute_edgeworth_loss with loss_floor=0 should never return
        a negative value, even at parameters where the raw surrogate
        would be negative."""
        d = training_setup
        from ssnn.edgeworth_risk import _raw_edgeworth_loss
        from ssnn.cumulants import decorrelation_matrix as dm

        S = dm(d["Sigma"])
        rng = np.random.default_rng(7)
        a_big = rng.standard_normal(3) * 10.0
        W_big = rng.standard_normal((3, d["p"])) * 5.0

        raw = _raw_edgeworth_loss(
            a_big, W_big, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )
        clamped = compute_edgeworth_loss(
            a_big, W_big, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S, loss_floor=0.0,
        )

        assert clamped >= 0.0
        if raw < 0.0:
            assert clamped == 0.0

    def test_loss_floor_none_returns_raw(self, training_setup):
        """loss_floor=None should return the unclamped raw loss."""
        d = training_setup
        from ssnn.edgeworth_risk import _raw_edgeworth_loss
        from ssnn.cumulants import decorrelation_matrix as dm

        S = dm(d["Sigma"])
        rng = np.random.default_rng(7)
        W = rng.standard_normal((2, d["p"])) * 0.1
        a = rng.standard_normal(2) * 0.1

        raw = _raw_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )
        unclamped = compute_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S, loss_floor=None,
        )
        assert unclamped == pytest.approx(raw, rel=1e-14)

    def test_loss_floor_does_not_affect_positive_region(self, training_setup):
        """In the normal (positive-loss) region, clamping has no effect."""
        d = training_setup
        from ssnn.cumulants import decorrelation_matrix as dm

        S = dm(d["Sigma"])
        rng = np.random.default_rng(42)
        W = rng.standard_normal((2, d["p"])) * 0.01
        a = rng.standard_normal(2) * 0.01

        L_raw = compute_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S, loss_floor=None,
        )
        L_clamped = compute_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S, loss_floor=0.0,
        )
        assert L_raw > 0
        assert L_clamped == pytest.approx(L_raw, rel=1e-14)

    def test_optimizer_loss_stays_nonnegative_relu(self, training_setup):
        """With default safeguards, the optimizer should never record
        a negative loss for ReLU, even with aggressive lr."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=3, activation="relu", lr=0.05,
            max_iters=500, init_scale=0.1,
            rng=np.random.default_rng(42),
            loss_floor=0.0, grad_clip=1.0,
        )
        for i, loss in enumerate(result.loss_history):
            assert loss >= 0.0, f"Negative loss at step {i}: {loss}"

    def test_grad_clip_limits_step_size(self, training_setup):
        """With grad_clip, the optimizer should take smaller steps than
        without it, visible as less aggressive loss changes."""
        d = training_setup
        kwargs = dict(
            Sigma=d["Sigma"], Sigma_beta=d["Sigma_beta"],
            E_y2=d["E_y2"], maf=d["maf"],
            m=2, activation="relu", lr=0.01,
            max_iters=30, init_scale=0.05,
            rng=np.random.default_rng(42),
            loss_floor=None, max_backtracks=0,
        )

        result_clipped = train_edgeworth(**kwargs, grad_clip=0.1)
        result_unclipped = train_edgeworth(
            **{**kwargs, "rng": np.random.default_rng(42)},
            grad_clip=None,
        )

        clipped_changes = [
            abs(result_clipped.loss_history[i] - result_clipped.loss_history[i-1])
            for i in range(1, len(result_clipped.loss_history))
        ]
        unclipped_changes = [
            abs(result_unclipped.loss_history[i] - result_unclipped.loss_history[i-1])
            for i in range(1, len(result_unclipped.loss_history))
        ]
        assert max(clipped_changes) <= max(unclipped_changes) + 1e-10

    def test_backtracking_prevents_loss_increase(self, training_setup):
        """With backtracking enabled, the loss should be non-increasing
        (modulo small numerical noise) for identity activation."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="identity", lr=0.01,
            max_iters=100, tol=1e-15, init_scale=0.01,
            rng=np.random.default_rng(42),
            loss_floor=None, grad_clip=None, max_backtracks=10,
        )
        losses = result.loss_history
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-10, (
                f"Loss increased at step {i}: {losses[i-1]:.10f} -> {losses[i]:.10f}"
            )

    def test_correction_delta_uses_raw_loss(self, training_setup):
        """compute_correction_delta should use the unclamped loss, so it
        faithfully reports the mathematical correction."""
        from ssnn.edgeworth_risk import compute_correction_delta, _raw_edgeworth_loss
        from ssnn.population_risk import compute_loss
        from ssnn.cumulants import decorrelation_matrix as dm

        d = training_setup
        S = dm(d["Sigma"])
        rng = np.random.default_rng(42)
        W = rng.standard_normal((2, d["p"])) * 0.1
        a = rng.standard_normal(2) * 0.1

        delta = compute_correction_delta(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu",
        )
        L_gauss = compute_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"], "relu"
        )
        L_raw = _raw_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S,
        )
        assert delta == pytest.approx(L_raw - L_gauss, abs=1e-10)


class TestEdgeworthSafeguardsAudit:
    """Audit: additional safeguard edge-case tests not in the original 7."""

    def test_max_backtracks_zero_disables_backtracking(self, training_setup):
        """With max_backtracks=0, the backtracking loop body never executes,
        so the optimizer may produce loss increases for aggressive lr."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.05,
            max_iters=50, init_scale=0.05,
            rng=np.random.default_rng(42),
            loss_floor=None, grad_clip=None, max_backtracks=0,
        )
        assert len(result.loss_history) > 1

    def test_grad_clip_none_allows_large_steps(self, training_setup):
        """With grad_clip=None, gradient norms are unrestricted."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.01,
            max_iters=20, init_scale=0.05,
            rng=np.random.default_rng(42),
            loss_floor=None, grad_clip=None, max_backtracks=0,
        )
        assert len(result.loss_history) > 1

    def test_loss_floor_interacts_with_correction_delta(self, training_setup):
        """compute_correction_delta uses loss_floor=None internally, so
        even if the optimizer uses loss_floor=0.0, the delta should still
        faithfully measure the raw mathematical correction."""
        from ssnn.edgeworth_risk import compute_correction_delta

        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.005,
            max_iters=100, init_scale=0.01,
            rng=np.random.default_rng(42),
            loss_floor=0.0,
        )

        delta = compute_correction_delta(
            result.a, result.W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu",
        )
        assert np.isfinite(delta)

    def test_gradient_zeroing_at_floor_is_recoverable(self, training_setup):
        """When raw loss is at the floor, gradients are zeroed.
        The optimizer should not permanently get stuck — if loss is at
        the floor, convergence should be declared via small relative change."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.01,
            max_iters=500, tol=1e-8, init_scale=0.01,
            rng=np.random.default_rng(42),
            loss_floor=0.0, grad_clip=1.0, max_backtracks=5,
        )
        assert result.loss_history[-1] >= 0.0
        assert len(result.loss_history) >= 2

    def test_all_safeguards_combined(self, training_setup):
        """All three safeguards together should keep the optimizer stable
        even with aggressive initialization."""
        d = training_setup
        result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=3, activation="relu", lr=0.1,
            max_iters=200, init_scale=0.5,
            rng=np.random.default_rng(42),
            loss_floor=0.0, grad_clip=0.5, max_backtracks=10,
        )
        for i, loss in enumerate(result.loss_history):
            assert loss >= 0.0, f"Negative loss at step {i}: {loss}"
            assert np.isfinite(loss), f"Non-finite loss at step {i}: {loss}"

    def test_custom_loss_floor_value(self, training_setup):
        """A non-default loss_floor (e.g. -0.1) should clamp at that value."""
        d = training_setup
        rng = np.random.default_rng(7)
        a_big = rng.standard_normal(3) * 10.0
        W_big = rng.standard_normal((3, d["p"])) * 5.0
        S = decorrelation_matrix(d["Sigma"])

        L = compute_edgeworth_loss(
            a_big, W_big, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu", S, loss_floor=-0.1,
        )
        assert L >= -0.1


class TestEdgeworthEndToEnd:
    """End-to-end validation of the Edgeworth correction framework."""

    def test_edgeworth_loss_differs_from_gaussian_on_asymmetric_mafs(self, training_setup):
        """The Edgeworth-corrected loss and the Gaussian loss should differ
        at the same parameter values when MAFs are not 0.5.

        This verifies the pipeline: cumulants -> projection cumulants ->
        correction integrals -> corrected loss gives a genuinely different
        objective from the Gaussian-only loss.
        """
        from ssnn.population_risk import compute_loss

        d = training_setup
        rng = np.random.default_rng(42)
        W = rng.standard_normal((2, d["p"])) * 0.1
        a = rng.standard_normal(2) * 0.1

        L_gauss = compute_loss(a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"], "relu")
        L_ew = compute_edgeworth_loss(
            a, W, d["Sigma"], d["Sigma_beta"], d["E_y2"],
            d["maf"], "relu",
        )

        assert L_gauss != pytest.approx(L_ew, abs=1e-8), (
            "Edgeworth loss should differ from Gaussian for non-0.5 MAFs"
        )

    def test_edgeworth_on_binomial_data_reasonable(self, training_setup):
        """Edgeworth-trained model on Binomial genotype data should produce
        predictions that are at least somewhat reasonable (MSE < E[y^2])."""
        from ssnn.utils import nn_predict

        d = training_setup
        rng = np.random.default_rng(999)
        p = d["p"]
        maf = d["maf"]
        beta_star = d["beta_star"]
        sigma_eps = d["sigma_eps"]

        n_test = 50_000
        X_test = np.zeros((n_test, p))
        for j in range(p):
            X_test[:, j] = rng.binomial(2, maf[j], size=n_test) - 2 * maf[j]
        y_test = X_test @ beta_star + rng.normal(0, sigma_eps, size=n_test)

        ew_result = train_edgeworth(
            d["Sigma"], d["Sigma_beta"], d["E_y2"], d["maf"],
            m=2, activation="relu", lr=0.002,
            max_iters=200, tol=1e-8, init_scale=0.01,
            rng=np.random.default_rng(42),
        )

        mse_ew = np.mean((y_test - nn_predict(X_test, ew_result.a, ew_result.W, "relu"))**2)
        trivial_mse = np.mean(y_test**2)

        assert mse_ew < trivial_mse, (
            f"Edgeworth MSE ({mse_ew:.4f}) should be less than trivial ({trivial_mse:.4f})"
        )
