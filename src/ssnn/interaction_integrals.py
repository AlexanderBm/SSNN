"""
Interaction integral machinery for the interaction-SSNN.

The standard SSNN cross-moment is:
    E[y sigma(w^T x)] = s_k * E[sigma'(z_k)]

where s_k = Sigma_beta^T w_k (linear in Sigma_beta).

With the interaction tensor Gamma_ij = E[x_i x_j y], a second-order
Stein identity gives an additional term:
    E[y sigma(w^T x)] = s_k * E[sigma'(z_k)] + q_k * E[sigma''(z_k)]

where q_k = w_k^T Gamma w_k is quadratic in w_k and couples to phenotype
through Gamma, breaking the linear-in-Sigma_beta structure.
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation_double_prime
from .gaussian_integrals import projection_variance


def interaction_cross_moment(
    Sigma: np.ndarray,
    Gamma: np.ndarray,
    w_k: np.ndarray,
    activation: str,
) -> float:
    """Compute the interaction correction to E[y sigma(w_k^T x)].

    Returns q_k * E[sigma''(z_k)] where:
        q_k = w_k^T Gamma w_k
        z_k ~ N(0, v_k), v_k = w_k^T Sigma w_k

    Args:
        Sigma: (p, p) LD covariance matrix.
        Gamma: (p, p) interaction tensor E[x_i x_j y].
        w_k: (p,) weight vector.
        activation: activation function name.

    Returns:
        Scalar interaction correction term.
    """
    E_sigma_pp, _ = get_activation_double_prime(activation)
    v_k = projection_variance(Sigma, w_k)
    q_k = float(w_k @ Gamma @ w_k)
    return q_k * E_sigma_pp(v_k)


def interaction_cross_moment_grad(
    Sigma: np.ndarray,
    Gamma: np.ndarray,
    w_k: np.ndarray,
    activation: str,
) -> np.ndarray:
    r"""Gradient of q_k * E[sigma''(z_k)] w.r.t. w_k.

    The interaction term is T_k = q_k * E_pp(v_k) where
        q_k = w_k^T Gamma w_k,  v_k = w_k^T Sigma w_k.

    By the product rule:
        dT_k/dw_k = E_pp(v_k) * dq_k/dw_k + q_k * dE_pp/dv_k * dv_k/dw_k
                   = E_pp(v_k) * 2 Gamma w_k + q_k * dE_pp/dv_k * 2 Sigma w_k

    Args:
        Sigma: (p, p) LD covariance matrix.
        Gamma: (p, p) interaction tensor.
        w_k: (p,) weight vector.
        activation: activation function name.

    Returns:
        (p,) gradient vector.
    """
    E_sigma_pp, dE_sigma_pp_dv = get_activation_double_prime(activation)
    v_k = projection_variance(Sigma, w_k)
    q_k = float(w_k @ Gamma @ w_k)

    Gamma_w = Gamma @ w_k
    Sigma_w = Sigma @ w_k

    return (
        E_sigma_pp(v_k) * 2.0 * Gamma_w
        + q_k * dE_sigma_pp_dv(v_k) * 2.0 * Sigma_w
    )


def interaction_cross_moment_denoised(
    Sigma: np.ndarray,
    Gamma: np.ndarray,
    w_k: np.ndarray,
    activation: str,
    sigma_other2: float,
    n: int,
) -> float:
    """Compute the interaction correction to E[y sigma(w_k^T x)] with J-S denoising.

    Returns q_denoised * E[sigma''(z_k)] where q_denoised is the J-S shrunk
    scalar.  Consistent with interaction_cross_moment_grad_denoised so that
    the denoised loss and denoised gradient form a valid Lyapunov pair for the
    backtracking line search.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Gamma: (p, p) interaction tensor.
        w_k: (p,) weight vector.
        activation: activation function name.
        sigma_other2: Cross-block phenotype noise variance.
        n: Sample size used to estimate Gamma.

    Returns:
        Scalar denoised interaction correction term.
    """
    E_sigma_pp, _ = get_activation_double_prime(activation)
    v_k = projection_variance(Sigma, w_k)
    q_k = float(w_k @ Gamma @ w_k)
    q_denoised = _js_denoise_q(q_k, v_k, sigma_other2, n)
    return q_denoised * E_sigma_pp(v_k)


def _js_denoise_q(q_k: float, v_k: float, sigma_other2: float, n: int) -> float:
    """James-Stein shrinkage for the scalar q_k = w_k^T Gamma w_k.

    Via the Isserlis-Wick theorem, the cross-block noise in q_k satisfies:
        Var(q_k^noise) = 3 * sigma_other2 * v_k^2 / n

    The positive-part J-S estimator shrinks q_k toward zero:
        q_denoised = q_k * max(0, 1 - sigma_q^2 / q_k^2)

    When |q_k| < sigma_q (SNR < 1), returns 0 exactly, causing the
    interaction gradient to vanish and the optimizer to recover the
    Gaussian NN solution gracefully.

    Args:
        q_k: Raw scalar w_k^T Gamma w_k.
        v_k: Projection variance w_k^T Sigma w_k.
        sigma_other2: Cross-block phenotype noise variance.
        n: Sample size used to estimate Gamma.

    Returns:
        Shrunk scalar.
    """
    if sigma_other2 <= 0 or v_k <= 1e-15 or n <= 0:
        return q_k
    sigma_q_sq = 3.0 * sigma_other2 * (v_k ** 2) / n
    q_k_sq = q_k ** 2
    if q_k_sq < 1e-30:
        return 0.0
    js_weight = max(0.0, 1.0 - sigma_q_sq / q_k_sq)
    return q_k * js_weight


def interaction_cross_moment_grad_denoised(
    Sigma: np.ndarray,
    Gamma: np.ndarray,
    w_k: np.ndarray,
    activation: str,
    sigma_other2: float,
    n: int,
) -> np.ndarray:
    r"""Rank-1 collapsed gradient with James-Stein denoising.

    When Gamma contains cross-block noise, the standard gradient
    E_pp * 2 * Gamma @ w_k is corrupted by noise in the off-axis
    directions. This function:

    1. Applies J-S shrinkage to q_k = w_k^T Gamma w_k.
    2. Replaces Gamma @ w_k with (q_denoised / v_k) * Sigma @ w_k,
       collapsing the gradient into the noise-free Sigma @ w_k direction.

    The result is always proportional to Sigma @ w_k:
        grad = 2 * q_denoised * (E_pp(v_k) / v_k + dE_pp_dv(v_k)) * Sigma @ w_k

    When SNR < 1, q_denoised = 0, the gradient vanishes, and the
    optimizer falls back exactly to the Gaussian NN gradient — no
    separate fallback logic needed.

    Args:
        Sigma: (p, p) LD covariance matrix.
        Gamma: (p, p) interaction tensor.
        w_k: (p,) weight vector.
        activation: activation function name.
        sigma_other2: Cross-block phenotype noise variance
            = max(0, E[y^2] - block_b_variance_explained).
        n: Sample size used to estimate Gamma.

    Returns:
        (p,) denoised gradient vector.
    """
    E_sigma_pp, dE_sigma_pp_dv = get_activation_double_prime(activation)
    v_k = projection_variance(Sigma, w_k)
    q_k = float(w_k @ Gamma @ w_k)
    Sigma_w = Sigma @ w_k

    q_denoised = _js_denoise_q(q_k, v_k, sigma_other2, n)

    v_k_safe = max(v_k, 1e-15)
    return (
        E_sigma_pp(v_k) * 2.0 * (q_denoised / v_k_safe) * Sigma_w
        + q_denoised * dE_sigma_pp_dv(v_k) * 2.0 * Sigma_w
    )
