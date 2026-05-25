"""Tests for J-S denoised interaction gradient (rank-1 collapse)."""

from __future__ import annotations

import numpy as np
import pytest

from ssnn.interaction_integrals import (
    _js_denoise_q,
    interaction_cross_moment_grad,
    interaction_cross_moment_grad_denoised,
)
from ssnn.interaction_optimizer import train_interaction
from ssnn.optimizer import train


# ---------------------------------------------------------------------------
# _js_denoise_q
# ---------------------------------------------------------------------------

def test_js_denoise_zero_when_low_snr():
    """J-S shrinkage collapses q_k to zero when |q_k| << sigma_q."""
    # sigma_q^2 = 3 * sigma_other2 * v_k^2 / n
    # With v_k=1, sigma_other2=10, n=100: sigma_q^2 = 0.3, sigma_q = 0.548
    # Use q_k = 0.1 << sigma_q -> js_weight = 0
    q_denoised = _js_denoise_q(q_k=0.1, v_k=1.0, sigma_other2=10.0, n=100)
    assert q_denoised == 0.0


def test_js_denoise_near_identity_when_high_snr():
    """J-S shrinkage barely shrinks q_k when SNR >> 1."""
    # sigma_q^2 = 3 * 0.001 * 1.0 / 10000 = 3e-7
    # q_k = 2.0 >> sigma_q = ~5.5e-4 -> js_weight ≈ 1
    q_denoised = _js_denoise_q(q_k=2.0, v_k=1.0, sigma_other2=0.001, n=10000)
    assert abs(q_denoised - 2.0) < 0.01


def test_js_denoise_passthrough_no_noise():
    """When sigma_other2=0, returns q_k unchanged."""
    q_denoised = _js_denoise_q(q_k=0.5, v_k=1.0, sigma_other2=0.0, n=100)
    assert q_denoised == 0.5


def test_js_denoise_passthrough_tiny_v():
    """When v_k ≈ 0, returns q_k unchanged (prevents division by zero)."""
    q_denoised = _js_denoise_q(q_k=0.5, v_k=1e-20, sigma_other2=1.0, n=100)
    assert q_denoised == 0.5


# ---------------------------------------------------------------------------
# interaction_cross_moment_grad_denoised
# ---------------------------------------------------------------------------

def test_denoised_grad_vanishes_low_snr():
    """Denoised gradient is exactly zero when SNR < 1 for all hidden units."""
    rng = np.random.default_rng(42)
    p = 10
    Sigma = np.eye(p)
    # q_k from this Gamma will be tiny; noise is huge
    Gamma = rng.standard_normal((p, p)) * 1e-5
    Gamma = (Gamma + Gamma.T) / 2.0
    w_k = rng.standard_normal(p)
    w_k /= np.linalg.norm(w_k)

    grad = interaction_cross_moment_grad_denoised(
        Sigma, Gamma, w_k, "relu", sigma_other2=1000.0, n=100
    )
    np.testing.assert_allclose(grad, 0.0, atol=1e-10)


def test_denoised_grad_proportional_to_sigma_w():
    """Denoised gradient is always proportional to Sigma @ w_k (rank-1 collapse)."""
    rng = np.random.default_rng(7)
    p = 8
    A = rng.standard_normal((p, p))
    Sigma = A.T @ A / p + 0.5 * np.eye(p)  # SPD
    Gamma = rng.standard_normal((p, p)) * 2.0
    Gamma = (Gamma + Gamma.T) / 2.0
    w_k = rng.standard_normal(p)

    grad = interaction_cross_moment_grad_denoised(
        Sigma, Gamma, w_k, "relu", sigma_other2=0.3, n=500
    )
    if np.linalg.norm(grad) < 1e-12:
        pytest.skip("gradient vanished at low SNR — rank-1 property trivially satisfied")

    Sigma_w = Sigma @ w_k
    cos_sim = abs(np.dot(grad, Sigma_w)) / (np.linalg.norm(grad) * np.linalg.norm(Sigma_w))
    assert cos_sim > 0.9999, f"Gradient not parallel to Sigma @ w_k: cos_sim={cos_sim:.6f}"


def test_denoised_grad_matches_standard_high_snr():
    """Denoised gradient has same sign / direction as standard gradient when SNR >> 1."""
    rng = np.random.default_rng(123)
    p = 6
    Sigma = np.eye(p)
    # Rank-1 Gamma: w_true @ w_true^T * scalar, guarantees large q_k at w_true
    w_true = rng.standard_normal(p)
    w_true /= np.linalg.norm(w_true)
    q_true = 3.0
    Gamma = q_true * np.outer(w_true, w_true)

    # Negligible noise
    grad_std = interaction_cross_moment_grad(Sigma, Gamma, w_true, "relu")
    grad_den = interaction_cross_moment_grad_denoised(
        Sigma, Gamma, w_true, "relu", sigma_other2=1e-8, n=100000
    )

    assert np.linalg.norm(grad_den) > 1e-10, "Denoised gradient should be nonzero at high SNR"
    cos_sim = np.dot(grad_std, grad_den) / (
        np.linalg.norm(grad_std) * np.linalg.norm(grad_den)
    )
    assert cos_sim > 0.99, f"Directions diverge at high SNR: cos_sim={cos_sim:.4f}"


def test_denoised_grad_negative_q():
    """J-S shrinkage works correctly for negative q_k (symmetric phenotype)."""
    rng = np.random.default_rng(55)
    p = 6
    Sigma = np.eye(p)
    # Negative diagonal dominates -> q_k < 0
    w_k = np.ones(p) / np.sqrt(p)
    Gamma = -5.0 * np.eye(p)  # q_k = -5

    # Low noise: should keep q_k (shrunk a bit)
    grad = interaction_cross_moment_grad_denoised(
        Sigma, Gamma, w_k, "relu", sigma_other2=0.001, n=100000
    )
    assert np.linalg.norm(grad) > 0, "Non-zero gradient expected at high SNR"

    # High noise: should vanish
    grad_low_snr = interaction_cross_moment_grad_denoised(
        Sigma, Gamma, w_k, "relu", sigma_other2=1000.0, n=10
    )
    np.testing.assert_allclose(grad_low_snr, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# train_interaction with denoising: fallback to Gaussian NN
# ---------------------------------------------------------------------------

def test_train_interaction_denoised_fallback():
    """When sigma_other2 is huge, denoised optimizer finds the same minimum as Gaussian NN."""
    rng = np.random.default_rng(99)
    p = 12
    A = rng.standard_normal((p, p))
    Sigma = A.T @ A / p + 0.3 * np.eye(p)
    beta_true = rng.standard_normal(p) * 0.2
    Sigma_beta = Sigma @ beta_true
    E_y2 = float(beta_true @ Sigma @ beta_true) + 0.5
    Gamma = rng.standard_normal((p, p)) * 0.1
    Gamma = (Gamma + Gamma.T) / 2.0

    m = 3
    # Both train() and train_interaction() generate W~N(0, init_scale), a~N(0, init_scale)
    # with the same layout.  Using the same seed gives identical initialization.
    gauss_res = train(
        Sigma, Sigma_beta, E_y2, m=m, activation="relu",
        lr=0.05, max_iters=2000, tol=1e-10,
        rng=np.random.default_rng(7),
    )

    # Denoised with enormous sigma_other2 -> q_k_denoised = 0 -> pure Gaussian gradient
    den_res = train_interaction(
        Sigma, Sigma_beta, E_y2, Gamma,
        m=m, activation="relu",
        lr=0.05, max_iters=2000, tol=1e-10,
        sigma_other2=1e6, n_train=10,
        rng=np.random.default_rng(7),  # same seed -> same init
    )

    # Both should converge to the same Gaussian minimum; final losses should match closely
    gauss_final_loss = gauss_res.loss_history[-1]
    den_final_loss = den_res.loss_history[-1]
    assert abs(gauss_final_loss - den_final_loss) < 1e-4, (
        f"Loss mismatch: Gaussian={gauss_final_loss:.8f}, Denoised={den_final_loss:.8f}"
    )
