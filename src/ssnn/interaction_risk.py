"""
Interaction-extended population risk L_int(a, W) and its gradients.

    L_int = L_gauss + Delta_int

where Delta_int adds the interaction correction term from the
second-order summary statistic Gamma_ij = E[x_i x_j y].

The extended cross-moment is:
    E[y sigma(w_k^T x)] = s_k E[sigma'(z_k)] + q_k E[sigma''(z_k)]

where q_k = w_k^T Gamma w_k.  The E[f(x)^2] term is unchanged.

For identity activation, E[sigma''(z)] = 0, so the interaction term
vanishes and L_int = L_gauss, preserving the linear baseline.
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation, get_activation_derivs, get_activation_double_prime
from .gaussian_integrals import (
    projection_variance,
    pairwise_covariance,
    stein_cross_moment,
    activation_cross_moment,
)
from .interaction_integrals import (
    interaction_cross_moment,
    interaction_cross_moment_grad,
)


def compute_interaction_loss(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    Gamma: np.ndarray,
    activation: str = "relu",
) -> float:
    """Compute the interaction-extended population risk.

    L_int = E[y^2] - 2 E_int[y f(x)] + E[f(x)^2]

    where E_int[y f(x)] includes both the Stein (first-order) and
    interaction (second-order) cross-moment terms.

    Args:
        a: (m,) second-layer weights.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) = E[x y], marginal associations.
        E_y2: scalar E[y^2].
        Gamma: (p, p) interaction tensor E[x_i x_j y].
        activation: activation function name.

    Returns:
        Scalar population risk.
    """
    m = len(a)

    E_y_f = 0.0
    for k in range(m):
        E_y_f += a[k] * (
            stein_cross_moment(Sigma, W[k], Sigma_beta, activation)
            + interaction_cross_moment(Sigma, Gamma, W[k], activation)
        )

    E_f2 = 0.0
    for k in range(m):
        for l in range(m):
            E_f2 += a[k] * a[l] * activation_cross_moment(Sigma, W[k], W[l], activation)

    return E_y2 - 2.0 * E_y_f + E_f2


def compute_interaction_grad_a(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    Gamma: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """Gradient of L_int w.r.t. second-layer weights a.

    dL/da_k = -2 [E_stein[y sigma_k] + E_int[y sigma_k]]
              + 2 sum_l a_l E[sigma_k sigma_l]
    """
    m = len(a)
    grad = np.zeros(m)

    for k in range(m):
        E_y_sigma_k = (
            stein_cross_moment(Sigma, W[k], Sigma_beta, activation)
            + interaction_cross_moment(Sigma, Gamma, W[k], activation)
        )

        E_f_sigma_k = 0.0
        for l in range(m):
            E_f_sigma_k += a[l] * activation_cross_moment(Sigma, W[l], W[k], activation)

        grad[k] = -2.0 * E_y_sigma_k + 2.0 * E_f_sigma_k

    return grad


def compute_interaction_grad_W(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    Gamma: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    r"""Gradient of L_int w.r.t. first-layer weights W.

    dL/dw_k = -2 a_k d/dw_k [s_k E[sigma'(z_k)] + q_k E[sigma''(z_k)]]
              + 2 sum_l a_k a_l d/dw_k E[sigma_k sigma_l]

    The Stein gradient is the same as in the Gaussian case.  The
    interaction gradient adds d/dw_k [q_k E[sigma''(z_k)]].
    """
    _, E_sigma_prime_fn, _ = get_activation(activation)
    dE_sp_dv, grad_E_ss = get_activation_derivs(activation)

    m, p = W.shape
    grad_W = np.zeros_like(W)

    Sw = Sigma @ W.T  # (p, m)

    for k in range(m):
        Sw_k = Sw[:, k]
        v_k = float(W[k] @ Sw_k)
        s_k = float(Sigma_beta @ W[k])

        E_sp = E_sigma_prime_fn(v_k)
        dEsp_dv = dE_sp_dv(v_k)

        # Stein gradient (same as population_risk.compute_grad_W)
        stein_grad = -2.0 * a[k] * (
            Sigma_beta * E_sp + s_k * dEsp_dv * 2.0 * Sw_k
        )

        # Interaction gradient: -2 a_k d/dw_k [q_k E[sigma''(z_k)]]
        int_grad = -2.0 * a[k] * interaction_cross_moment_grad(
            Sigma, Gamma, W[k], activation,
        )

        # Cross terms: d/dw_k E[sigma_k sigma_l] (same as Gaussian)
        cross_grad = np.zeros(p)
        for l in range(m):
            Sw_l = Sw[:, l]
            C_kl = pairwise_covariance(Sigma, W[k], W[l])
            dF = grad_E_ss(C_kl)
            d_Ess = dF[0, 0] * 2.0 * Sw_k + dF[0, 1] * Sw_l
            cross_grad += 2.0 * a[k] * a[l] * d_Ess

        grad_W[k] = stein_grad + int_grad + cross_grad

    return grad_W


def compute_interaction_gradients(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    Gamma: np.ndarray,
    activation: str = "relu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both gradients (dL/da, dL/dW) for the interaction loss.

    Returns:
        (grad_a, grad_W) -- shapes (m,) and (m, p).
    """
    grad_a = compute_interaction_grad_a(a, W, Sigma, Sigma_beta, Gamma, activation)
    grad_W = compute_interaction_grad_W(a, W, Sigma, Sigma_beta, Gamma, activation)
    return grad_a, grad_W
