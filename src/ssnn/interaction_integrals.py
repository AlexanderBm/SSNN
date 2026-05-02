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
