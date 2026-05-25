"""
Population risk L(a, W) and its gradients, computed entirely from
summary statistics (Sigma, Sigma_beta = Sigma @ beta*).

The population risk under squared loss is:

    L(a, W) = E[y^2] - 2 E[y f(x)] + E[f(x)^2]

where f(x) = sum_k a_k sigma(w_k^T x) is a one-hidden-layer network.

Each term reduces to Gaussian integrals via Stein's lemma (see
gaussian_integrals.py for the building blocks).

Gradients:
    dL/da_k = -2 E[(y - f(x)) sigma(w_k^T x)]
    dL/dw_k = -2 a_k E[(y - f(x)) sigma'(w_k^T x) x]

These also reduce to the same types of Gaussian integrals.
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation, get_activation_derivs
from .gaussian_integrals import (
    projection_variance,
    pairwise_covariance,
    stein_cross_moment,
    activation_cross_moment,
)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _compute_E_y_f(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str,
) -> float:
    """E[y f(x)] = sum_k a_k E[y sigma(w_k^T x)]."""
    m = len(a)
    total = 0.0
    for k in range(m):
        total += a[k] * stein_cross_moment(Sigma, W[k], Sigma_beta, activation)
    return total


def _compute_E_f_squared(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    activation: str,
    Cov_ref: np.ndarray | None = None,
) -> float:
    """E[f(x)^2] = sum_{k,l} a_k a_l E[sigma(w_k^T x) sigma(w_l^T x)].

    Uses Cov_ref for the arc-cosine kernel when provided (corrects binomial bias).
    """
    cov = Cov_ref if Cov_ref is not None else Sigma
    m = len(a)
    total = 0.0
    for k in range(m):
        for l in range(m):
            total += a[k] * a[l] * activation_cross_moment(cov, W[k], W[l], activation)
    return total


def compute_loss(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    activation: str = "relu",
    reg_a: float = 0.0,
    reg_W: float = 0.0,
    Cov_ref: np.ndarray | None = None,
) -> float:
    """Compute the population risk L(a, W) = E[(y - f(x))^2].

    = E[y^2] - 2 E[y f(x)] + E[f(x)^2] + reg_a*||a||^2 + reg_W*||W||_F^2

    All terms are computed from summary statistics only.

    Args:
        a: (m,) second-layer weights.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix (used for Stein terms E[y f]).
        Sigma_beta: (p,) = Sigma @ beta*, from GWAS marginal associations.
        E_y2: scalar E[y^2] = beta*^T Sigma beta* + sigma_eps^2.
        activation: name of activation function.
        reg_a: L2 regularization strength for second-layer weights.
        reg_W: L2 regularization strength for first-layer weights.
        Cov_ref: (p, p) empirical covariance for E[f^2] term (corrects
            binomial bias). Uses Sigma when None.

    Returns:
        Scalar population risk (with optional regularization).
    """
    E_y_f = _compute_E_y_f(a, W, Sigma, Sigma_beta, activation)
    E_f2 = _compute_E_f_squared(a, W, Sigma, activation, Cov_ref)
    loss = E_y2 - 2.0 * E_y_f + E_f2
    if reg_a > 0:
        loss += reg_a * np.sum(a ** 2)
    if reg_W > 0:
        loss += reg_W * np.sum(W ** 2)
    return loss


# ---------------------------------------------------------------------------
# Loss restricted to a single hidden unit (for efficient W gradient)
# ---------------------------------------------------------------------------

def _loss_terms_involving_k(
    k: int,
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str,
    Cov_ref: np.ndarray | None = None,
) -> float:
    """Compute only the parts of L that depend on w_k."""
    cov = Cov_ref if Cov_ref is not None else Sigma
    m = len(a)
    result = -2.0 * a[k] * stein_cross_moment(Sigma, W[k], Sigma_beta, activation)

    for l in range(m):
        result += 2.0 * a[k] * a[l] * activation_cross_moment(cov, W[k], W[l], activation)

    result -= a[k]**2 * activation_cross_moment(cov, W[k], W[k], activation)

    return result


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def compute_grad_a(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str = "relu",
    reg_a: float = 0.0,
    Cov_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of L w.r.t. second-layer weights a.

    dL/da_k = -2 E[y sigma(w_k^T x)] + 2 sum_l a_l E[sigma(w_k^T x) sigma(w_l^T x)]
              + 2 * reg_a * a_k
    """
    cov = Cov_ref if Cov_ref is not None else Sigma
    m = len(a)
    grad = np.zeros(m)

    for k in range(m):
        E_y_sigma_k = stein_cross_moment(Sigma, W[k], Sigma_beta, activation)

        E_f_sigma_k = 0.0
        for l in range(m):
            E_f_sigma_k += a[l] * activation_cross_moment(cov, W[l], W[k], activation)

        grad[k] = -2.0 * E_y_sigma_k + 2.0 * E_f_sigma_k

    if reg_a > 0:
        grad += 2.0 * reg_a * a

    return grad


def compute_grad_W(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str = "relu",
    reg_W: float = 0.0,
    Cov_ref: np.ndarray | None = None,
) -> np.ndarray:
    r"""Analytic gradient of L w.r.t. first-layer weights W.

    The full loss is:

        L = E[y^2] - 2 sum_k a_k E[y sigma_k]
            + sum_{k,l} a_k a_l E[sigma_k sigma_l]
            + reg_W * ||W||_F^2

    Only terms involving w_k contribute to dL/dw_k:

        dL/dw_{k,j} = -2 a_k d/dw_{k,j} E[y sigma_k]
                     + 2 sum_l a_k a_l d/dw_{k,j} E[sigma_k sigma_l]
                     + 2 * reg_W * w_{k,j}

    (the factor 2 in the cross-term comes from combining the (k,l) and
    (l,k) contributions, using symmetry of E[sigma_k sigma_l].)

    Chain rule through v_k = w_k^T Sigma w_k, c_{kl} = w_k^T Sigma w_l,
    s_k = (Sigma beta*)^T w_k gives:

        dv_k/dw_{k,j} = 2 (Sigma w_k)_j
        dc_{kl}/dw_{k,j} = (Sigma w_l)_j
        ds_k/dw_{k,j} = (Sigma beta*)_j
    """
    _, E_sigma_prime_fn, _ = get_activation(activation)
    dE_sp_dv, grad_E_ss = get_activation_derivs(activation)

    cov = Cov_ref if Cov_ref is not None else Sigma
    m, p = W.shape
    grad_W = np.zeros_like(W)

    Sw = Sigma @ W.T     # (p, m) — Sigma @ w_k for Stein terms
    Cw = cov @ W.T       # (p, m) — cov @ w_k for E[f^2] terms

    for k in range(m):
        Sw_k = Sw[:, k]
        Cw_k = Cw[:, k]
        v_k = float(W[k] @ Sw_k)
        s_k = float(Sigma_beta @ W[k])

        E_sp = E_sigma_prime_fn(v_k)
        dEsp_dv = dE_sp_dv(v_k)

        # Stein term uses Sigma
        stein_grad = -2.0 * a[k] * (
            Sigma_beta * E_sp + s_k * dEsp_dv * 2.0 * Sw_k
        )

        # Cross terms use cov (Cov_ref when provided)
        # E[sigma_k sigma_l] depends on C_kl = [[v_k, c_kl], [c_kl, v_l]]
        # computed from cov, not Sigma
        cross_grad = np.zeros(p)
        for l in range(m):
            Cw_l = Cw[:, l]
            C_kl = pairwise_covariance(cov, W[k], W[l])
            dF = grad_E_ss(C_kl)

            d_Ess = dF[0, 0] * 2.0 * Cw_k + dF[0, 1] * Cw_l
            cross_grad += 2.0 * a[k] * a[l] * d_Ess

        grad_W[k] = stein_grad + cross_grad

    if reg_W > 0:
        grad_W += 2.0 * reg_W * W

    return grad_W


def compute_gradients(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    activation: str = "relu",
    reg_a: float = 0.0,
    reg_W: float = 0.0,
    Cov_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both gradients (dL/da, dL/dW) in a single call.

    Returns:
        (grad_a, grad_W) -- shapes (m,) and (m, p).
    """
    grad_a = compute_grad_a(a, W, Sigma, Sigma_beta, activation, reg_a, Cov_ref)
    grad_W = compute_grad_W(a, W, Sigma, Sigma_beta, activation, reg_W, Cov_ref)
    return grad_a, grad_W
