"""
Interaction-extended population risk L_int(a, W) and its gradients.

    L_int = L_gauss + Delta_int

where Delta_int adds the interaction correction term from the
second-order summary statistic Gamma_ij = E[x_i x_j y].

The extended cross-moment is:
    E[y sigma(w_k^T x)] = s_k E[sigma'(z_k)] + q_k E[sigma''(z_k)]

where q_k = w_k^T Gamma w_k.

The E[f(x)^2] term uses the arc-cosine kernel formula, which requires
the covariance of projections w^T x.  When genotypes are binomial
(not Gaussian), the LD matrix Sigma (from the Gaussian latent model)
overestimates projection variances by ~2.5x, causing ~100% E[f^2]
bias.  An optional ``Cov_ref`` parameter allows using the empirical
covariance from a reference panel for the E[f^2] term, while keeping
Sigma for the Stein-based E[y*f] term where it is appropriate.

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
    Cov_ref: np.ndarray | None = None,
    reg_a: float = 0.0,
    reg_W: float = 0.0,
) -> float:
    """Compute the interaction-extended population risk.

    L_int = E[y^2] - 2 E_int[y f(x)] + E[f(x)^2] + reg_a*||a||^2 + reg_W*||W||_F^2

    where E_int[y f(x)] includes both the Stein (first-order) and
    interaction (second-order) cross-moment terms.

    Args:
        a: (m,) second-layer weights.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix (used for Stein/interaction terms).
        Sigma_beta: (p,) = E[x y], marginal associations.
        E_y2: scalar E[y^2].
        Gamma: (p, p) interaction tensor E[x_i x_j y].
        activation: activation function name.
        Cov_ref: (p, p) empirical covariance from a reference panel.
            When provided, used for the E[f^2] arc-cosine kernel instead
            of Sigma, correcting the ~100% bias on binomial data.
        reg_a: L2 regularization strength for second-layer weights.
        reg_W: L2 regularization strength for first-layer weights.

    Returns:
        Scalar population risk (with optional regularization).
    """
    m = len(a)
    Sigma_f2 = Cov_ref if Cov_ref is not None else Sigma

    E_y_f = 0.0
    for k in range(m):
        E_y_f += a[k] * (
            stein_cross_moment(Sigma, W[k], Sigma_beta, activation)
            + interaction_cross_moment(Sigma, Gamma, W[k], activation)
        )

    E_f2 = 0.0
    for k in range(m):
        for l in range(m):
            E_f2 += a[k] * a[l] * activation_cross_moment(Sigma_f2, W[k], W[l], activation)

    loss = E_y2 - 2.0 * E_y_f + E_f2
    if reg_a > 0:
        loss += reg_a * np.sum(a ** 2)
    if reg_W > 0:
        loss += reg_W * np.sum(W ** 2)
    return loss


def compute_interaction_grad_a(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    Gamma: np.ndarray,
    activation: str = "relu",
    Cov_ref: np.ndarray | None = None,
    reg_a: float = 0.0,
) -> np.ndarray:
    """Gradient of L_int w.r.t. second-layer weights a.

    dL/da_k = -2 [E_stein[y sigma_k] + E_int[y sigma_k]]
              + 2 sum_l a_l E[sigma_k sigma_l]
              + 2 * reg_a * a_k

    When Cov_ref is provided, the E[sigma_k sigma_l] terms use
    Cov_ref for projection covariances instead of Sigma.
    """
    m = len(a)
    Sigma_f2 = Cov_ref if Cov_ref is not None else Sigma
    grad = np.zeros(m)

    for k in range(m):
        E_y_sigma_k = (
            stein_cross_moment(Sigma, W[k], Sigma_beta, activation)
            + interaction_cross_moment(Sigma, Gamma, W[k], activation)
        )

        E_f_sigma_k = 0.0
        for l in range(m):
            E_f_sigma_k += a[l] * activation_cross_moment(Sigma_f2, W[l], W[k], activation)

        grad[k] = -2.0 * E_y_sigma_k + 2.0 * E_f_sigma_k

    if reg_a > 0:
        grad += 2.0 * reg_a * a

    return grad


def compute_interaction_grad_W(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    Gamma: np.ndarray,
    activation: str = "relu",
    Cov_ref: np.ndarray | None = None,
    reg_W: float = 0.0,
) -> np.ndarray:
    r"""Gradient of L_int w.r.t. first-layer weights W.

    dL/dw_k = -2 a_k d/dw_k [s_k E[sigma'(z_k)] + q_k E[sigma''(z_k)]]
              + 2 sum_l a_k a_l d/dw_k E[sigma_k sigma_l]
              + 2 * reg_W * w_k

    The Stein gradient uses Sigma (appropriate for Stein's lemma).
    The interaction gradient uses Sigma for projection variance v_k.
    The cross-term gradient uses Cov_ref (if provided) for the E[f^2]
    arc-cosine kernel, since Cov_ref gives the true projection
    covariances on binomial data.

    The chain rule for the cross term goes through C_kl:
        dv_k/dw_{k,j} = 2 (Cov_ref @ w_k)_j
        dc_{kl}/dw_{k,j} = (Cov_ref @ w_l)_j
    """
    _, E_sigma_prime_fn, _ = get_activation(activation)
    dE_sp_dv, grad_E_ss = get_activation_derivs(activation)

    m, p = W.shape
    Sigma_f2 = Cov_ref if Cov_ref is not None else Sigma
    grad_W = np.zeros_like(W)

    Sw = Sigma @ W.T      # (p, m) — for Stein/interaction terms
    Cw = Sigma_f2 @ W.T   # (p, m) — for E[f^2] cross terms

    for k in range(m):
        Sw_k = Sw[:, k]
        v_k = float(W[k] @ Sw_k)
        s_k = float(Sigma_beta @ W[k])

        E_sp = E_sigma_prime_fn(v_k)
        dEsp_dv = dE_sp_dv(v_k)

        # Stein gradient (uses Sigma — appropriate for Stein's lemma)
        stein_grad = -2.0 * a[k] * (
            Sigma_beta * E_sp + s_k * dEsp_dv * 2.0 * Sw_k
        )

        # Interaction gradient (uses Sigma for v_k)
        int_grad = -2.0 * a[k] * interaction_cross_moment_grad(
            Sigma, Gamma, W[k], activation,
        )

        # Cross terms: d/dw_k E[sigma_k sigma_l] using Cov_ref
        Cw_k = Cw[:, k]
        cross_grad = np.zeros(p)
        for l in range(m):
            Cw_l = Cw[:, l]
            C_kl = pairwise_covariance(Sigma_f2, W[k], W[l])
            dF = grad_E_ss(C_kl)
            d_Ess = dF[0, 0] * 2.0 * Cw_k + dF[0, 1] * Cw_l
            cross_grad += 2.0 * a[k] * a[l] * d_Ess

        grad_W[k] = stein_grad + int_grad + cross_grad

    if reg_W > 0:
        grad_W += 2.0 * reg_W * W

    return grad_W


def compute_interaction_gradients(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    Gamma: np.ndarray,
    activation: str = "relu",
    Cov_ref: np.ndarray | None = None,
    reg_a: float = 0.0,
    reg_W: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both gradients (dL/da, dL/dW) for the interaction loss.

    Args:
        Cov_ref: (p, p) empirical covariance for E[f^2] correction.
            Passed through to both gradient functions.
        reg_a: L2 regularization strength for second-layer weights.
        reg_W: L2 regularization strength for first-layer weights.

    Returns:
        (grad_a, grad_W) -- shapes (m,) and (m, p).
    """
    grad_a = compute_interaction_grad_a(a, W, Sigma, Sigma_beta, Gamma, activation, Cov_ref, reg_a)
    grad_W = compute_interaction_grad_W(a, W, Sigma, Sigma_beta, Gamma, activation, Cov_ref, reg_W)
    return grad_a, grad_W
