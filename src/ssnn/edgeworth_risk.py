"""
Edgeworth-corrected population risk L_EW(a, W) and its gradients.

    L_EW = L_gauss + Delta_L(a, W; kappa_tilde_3, kappa_tilde_4)

where Delta_L collects all Edgeworth correction terms.  The corrected
loss depends only on summary-recoverable quantities: Sigma, Sigma_beta,
and allele frequencies (which determine the cumulants).

The key structural result (Theorem 1 in the research plan):
    - For linear activations (sigma = id), Delta_L = 0 identically.
    - For nonlinear activations (ReLU, sigmoid), Delta_L != 0 whenever
      kappa_tilde_3 != 0 (i.e., allele frequencies differ from 0.5).
    - The Edgeworth-corrected optimum differs from the Gaussian one,
      and this shift is accessible only to nonlinear models.
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation, get_activation_derivs
from .cumulants import (
    snp_cumulants,
    projection_cumulants_independent,
    projection_cumulants_ld,
    projection_cumulant_gradients_independent,
    projection_cumulant_gradients_ld,
    decorrelation_matrix,
)
from .edgeworth_integrals import (
    edgeworth_E_sigma_prime,
    edgeworth_E_sigma_sigma,
    get_sigma_prime_correction_values,
)
from .gaussian_integrals import (
    projection_variance,
    pairwise_covariance,
)


# ===================================================================
# Edgeworth-corrected loss
# ===================================================================

def _projection_kt(
    w: np.ndarray,
    maf: np.ndarray,
    Sigma: np.ndarray | None,
    Sigma_inv_sqrt: np.ndarray | None,
) -> tuple[float, float]:
    """Get the projection cumulants for weight vector w."""
    if Sigma is not None and Sigma_inv_sqrt is not None:
        return projection_cumulants_ld(w, maf, Sigma, Sigma_inv_sqrt)
    else:
        cum = snp_cumulants(maf)
        return projection_cumulants_independent(
            w, cum["kappa2"], cum["kappa3"], cum["kappa4"]
        )


def _projection_kt_with_grad(
    w: np.ndarray,
    maf: np.ndarray,
    Sigma: np.ndarray,
    Sigma_inv_sqrt: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Get projection cumulants AND their gradients w.r.t. w.

    Returns:
        (kt3, kt4, grad_kt3, grad_kt4) where grads are (p,) arrays.
    """
    kt3, kt4 = projection_cumulants_ld(w, maf, Sigma, Sigma_inv_sqrt)
    grad_kt3, grad_kt4 = projection_cumulant_gradients_ld(
        w, maf, Sigma, Sigma_inv_sqrt
    )
    return kt3, kt4, grad_kt3, grad_kt4


def _raw_edgeworth_loss(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    activation: str,
    Sigma_inv_sqrt: np.ndarray,
) -> float:
    """Unclamped Edgeworth-corrected loss (internal)."""
    m = len(a)

    E_y_f = 0.0
    for k in range(m):
        v_k = projection_variance(Sigma, W[k])
        kt3_k, kt4_k = _projection_kt(W[k], maf, Sigma, Sigma_inv_sqrt)
        beta_Sigma_w = float(Sigma_beta @ W[k])

        E_sp = edgeworth_E_sigma_prime(v_k, kt3_k, kt4_k, activation)
        E_y_f += a[k] * beta_Sigma_w * E_sp

    E_f2 = 0.0
    for k in range(m):
        kt3_k, kt4_k = _projection_kt(W[k], maf, Sigma, Sigma_inv_sqrt)
        for l in range(m):
            kt3_l, kt4_l = _projection_kt(W[l], maf, Sigma, Sigma_inv_sqrt)
            C_kl = pairwise_covariance(Sigma, W[k], W[l])

            E_ss = edgeworth_E_sigma_sigma(
                C_kl, kt3_k, kt4_k, kt3_l, kt4_l, activation
            )
            E_f2 += a[k] * a[l] * E_ss

    return E_y2 - 2.0 * E_y_f + E_f2


def compute_edgeworth_loss(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    activation: str = "relu",
    Sigma_inv_sqrt: np.ndarray | None = None,
    loss_floor: float | None = 0.0,
) -> float:
    """Compute the Edgeworth-corrected population risk L_EW(a, W).

    L_EW = E[y^2] - 2 E_EW[y f(x)] + E_EW[f(x)^2]

    where the expectations use the Edgeworth-corrected distribution.

    The Edgeworth expansion can produce negative surrogate "densities,"
    which means L_EW is not guaranteed to be bounded below.  For ReLU
    especially, the optimizer can exploit this and drive the loss to
    -infinity.  The ``loss_floor`` parameter clamps the returned loss
    to prevent this.  The true population MSE is always >= 0, so the
    default floor of 0.0 is a conservative safeguard that does not
    distort the landscape in the physically meaningful region.

    Args:
        a: (m,) second-layer weights.
        W: (m, p) first-layer weight matrix.
        Sigma: (p, p) LD covariance matrix.
        Sigma_beta: (p,) = Sigma @ beta*.
        E_y2: scalar E[y^2].
        maf: (p,) minor allele frequencies.
        activation: name of activation function.
        Sigma_inv_sqrt: (p, p) precomputed Sigma^{-1/2} (optional).
        loss_floor: lower bound for the returned loss. None disables
            clamping (raw surrogate). Default 0.0.

    Returns:
        Scalar Edgeworth-corrected population risk (clamped if loss_floor set).
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    raw = _raw_edgeworth_loss(
        a, W, Sigma, Sigma_beta, E_y2, maf, activation, Sigma_inv_sqrt
    )

    if loss_floor is not None and raw < loss_floor:
        return loss_floor
    return raw


# ===================================================================
# Edgeworth correction Delta_L = L_EW - L_gauss
# ===================================================================

def compute_correction_delta(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    activation: str = "relu",
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> float:
    """Compute Delta_L = L_EW - L_gauss.

    This is the quantity from Theorem 1 (Non-Gaussian Gap):
    - Delta_L = 0 for identity activation (linear models)
    - Delta_L != 0 for nonlinear activations when kappa_tilde_3 != 0

    Always uses the *unclamped* Edgeworth loss so the delta faithfully
    reflects the mathematical correction, not optimizer safeguards.
    """
    from .population_risk import compute_loss

    L_gauss = compute_loss(a, W, Sigma, Sigma_beta, E_y2, activation)
    L_ew = compute_edgeworth_loss(
        a, W, Sigma, Sigma_beta, E_y2, maf, activation, Sigma_inv_sqrt,
        loss_floor=None,
    )
    return L_ew - L_gauss


# ===================================================================
# Edgeworth-corrected gradients
# ===================================================================

def _ew_loss_terms_involving_k(
    k: int,
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    maf: np.ndarray,
    activation: str,
    Sigma_inv_sqrt: np.ndarray,
) -> float:
    """Compute only the parts of L_EW that depend on w_k.

    Analogous to population_risk._loss_terms_involving_k but with
    Edgeworth corrections.
    """
    m = len(a)

    v_k = projection_variance(Sigma, W[k])
    kt3_k, kt4_k = _projection_kt(W[k], maf, Sigma, Sigma_inv_sqrt)
    beta_Sigma_w = float(Sigma_beta @ W[k])

    E_sp = edgeworth_E_sigma_prime(v_k, kt3_k, kt4_k, activation)
    result = -2.0 * a[k] * beta_Sigma_w * E_sp

    for l in range(m):
        kt3_l, kt4_l = _projection_kt(W[l], maf, Sigma, Sigma_inv_sqrt)
        C_kl = pairwise_covariance(Sigma, W[k], W[l])

        E_ss = edgeworth_E_sigma_sigma(
            C_kl, kt3_k, kt4_k, kt3_l, kt4_l, activation
        )
        result += 2.0 * a[k] * a[l] * E_ss

    C_kk = pairwise_covariance(Sigma, W[k], W[k])
    E_ss_kk = edgeworth_E_sigma_sigma(
        C_kk, kt3_k, kt4_k, kt3_k, kt4_k, activation
    )
    result -= a[k]**2 * E_ss_kk

    return result


def compute_edgeworth_grad_a(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    maf: np.ndarray,
    activation: str = "relu",
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of L_EW w.r.t. second-layer weights a.

    dL_EW/da_k = -2 E_EW[y sigma(w_k^T x)]
                 + 2 sum_l a_l E_EW[sigma(w_k^T x) sigma(w_l^T x)]
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    m = len(a)
    grad = np.zeros(m)

    for k in range(m):
        v_k = projection_variance(Sigma, W[k])
        kt3_k, kt4_k = _projection_kt(W[k], maf, Sigma, Sigma_inv_sqrt)
        beta_Sigma_w = float(Sigma_beta @ W[k])

        E_sp = edgeworth_E_sigma_prime(v_k, kt3_k, kt4_k, activation)
        E_y_sigma_k = beta_Sigma_w * E_sp

        E_f_sigma_k = 0.0
        for l in range(m):
            kt3_l, kt4_l = _projection_kt(W[l], maf, Sigma, Sigma_inv_sqrt)
            C_lk = pairwise_covariance(Sigma, W[l], W[k])

            E_ss = edgeworth_E_sigma_sigma(
                C_lk, kt3_l, kt4_l, kt3_k, kt4_k, activation
            )
            E_f_sigma_k += a[l] * E_ss

        grad[k] = -2.0 * E_y_sigma_k + 2.0 * E_f_sigma_k

    return grad


def compute_edgeworth_grad_W(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    maf: np.ndarray,
    activation: str = "relu",
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> np.ndarray:
    r"""Analytic gradient of L_EW w.r.t. first-layer weights W.

    Like the Gaussian case, the full loss is:

        L_EW = E[y^2] - 2 sum_k a_k E_EW[y sigma_k]
               + sum_{k,l} a_k a_l E_EW[sigma_k sigma_l]

    The Edgeworth expectations depend on w_k through three paths:
      (a) the Gaussian intermediate variables (v_k, c_{kl}, s_k)
      (b) the projection cumulants (kt3_k, kt4_k)

    Path (a) uses the same derivatives as the Gaussian analytic gradient.
    Path (b) adds correction terms from the Edgeworth expansion coefficients.

    For the Stein term E_EW[y sigma_k] = s_k * E_EW[sigma'(z_k)]:

        E_EW[sigma'] = E_G[sigma'] + (kt3/6)*I3' + (kt4/24)*I4' + (kt3^2/72)*I6'

    where I3', I4', I6' are constants (Hermite correction integrals for sigma').

        d/dw_{k,j} E_EW[sigma'] = dE_G[sigma']/dv * dv/dw_{k,j}
            + (I3'/6) * dkt3/dw_{k,j}
            + (I4'/24) * dkt4/dw_{k,j}
            + (kt3/36) * I6' * dkt3/dw_{k,j}

    For the cross-term E_EW[sigma_k sigma_l], the correction is
    computed by 2D quadrature and its dependence on kt3_k, kt4_k is
    handled via finite differences on the correction piece only (the
    Gaussian baseline part uses exact analytic derivatives).
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    _, E_sigma_prime_fn, _ = get_activation(activation)
    _, grad_E_ss = get_activation_derivs(activation)

    m, p = W.shape
    grad_W = np.zeros_like(W)

    Sw = Sigma @ W.T  # (p, m)

    _eps_v = 1e-5  # step for finite-difference of dE_EW[sigma']/dv

    for k in range(m):
        Sw_k = Sw[:, k]
        s_k = float(Sigma_beta @ W[k])
        v_k = float(W[k] @ Sw_k)

        kt3_k, kt4_k, gkt3_k, gkt4_k = _projection_kt_with_grad(
            W[k], maf, Sigma, Sigma_inv_sqrt
        )

        # --- Stein term: d/dw_{k,j} [-2 a_k s_k E_EW[sigma'(z_k)]] ---

        E_sp_ew = edgeworth_E_sigma_prime(v_k, kt3_k, kt4_k, activation)

        # v-dependent correction integrals I_k = E[sigma'(sqrt(v)*z) H_k(z)]
        # (constants for ReLU; v-dependent for sigmoid via quadrature)
        I3p, I4p, I6p = get_sigma_prime_correction_values(v_k, activation)

        # Full d E_EW[sigma'(z_k)] / dv_k via central finite difference.
        # This correctly captures the path-(a) contribution for all activations,
        # including the v-dependence of the correction integrals for sigmoid.
        dEsp_ew_dv = (
            edgeworth_E_sigma_prime(v_k + _eps_v, kt3_k, kt4_k, activation)
            - edgeworth_E_sigma_prime(v_k - _eps_v, kt3_k, kt4_k, activation)
        ) / (2.0 * _eps_v)

        # d E_EW[sigma'] / d w_{k,j}  (paths a + b)
        dEsp_dw = (
            dEsp_ew_dv * 2.0 * Sw_k           # path (a): through v_k (full EW)
            + (I3p / 6.0) * gkt3_k            # path (b): through kt3
            + (I4p / 24.0) * gkt4_k           # path (b): through kt4
            + (kt3_k / 36.0) * I6p * gkt3_k   # path (b): through kt3^2
        )

        # d/dw_{k,j} [s_k * E_EW[sigma']] = Sigma_beta_j * E_EW[sigma'] + s_k * dEsp_dw_j
        stein_grad = -2.0 * a[k] * (
            Sigma_beta * E_sp_ew + s_k * dEsp_dw
        )

        # --- Cross terms: d/dw_{k,j} [sum_{k,l} a_k a_l E_EW[sigma_k sigma_l]] ---
        # Decompose E_EW[sigma_k sigma_l] = E_G[sigma_k sigma_l] + correction(kt3_k, kt4_k, kt3_l, kt4_l)
        # The Gaussian part has analytic derivatives (path a).
        # The correction depends on kt3_k, kt4_k, C_{kl} — we differentiate
        # the correction w.r.t. kt3_k and kt4_k analytically (via finite
        # differences on the small correction), and w.r.t. C_{kl} via the
        # Gaussian grad_E_ss plus a correction finite-diff on the 2D quadrature.

        cross_grad = np.zeros(p)
        for l in range(m):
            Sw_l = Sw[:, l]
            C_kl = pairwise_covariance(Sigma, W[k], W[l])
            kt3_l, kt4_l = _projection_kt(W[l], maf, Sigma, Sigma_inv_sqrt)

            # Path (a): analytic Gaussian derivatives through C_{kl}
            dF_gauss = grad_E_ss(C_kl)
            d_Ess_gauss = dF_gauss[0, 0] * 2.0 * Sw_k + dF_gauss[0, 1] * Sw_l

            # Path (b): derivatives of the 2D correction through kt3_k, kt4_k
            # Use finite differences on the correction piece (cheap — just
            # the _cross_term_correction_quadrature call, not the full loss)
            eps_kt = 1e-6
            E_ss_full = edgeworth_E_sigma_sigma(
                C_kl, kt3_k, kt4_k, kt3_l, kt4_l, activation
            )
            _, _, E_ss_gauss_fn = get_activation(activation)
            E_ss_gauss_val = E_ss_gauss_fn(C_kl)
            correction_base = E_ss_full - E_ss_gauss_val

            E_ss_kt3p = edgeworth_E_sigma_sigma(
                C_kl, kt3_k + eps_kt, kt4_k, kt3_l, kt4_l, activation
            )
            E_ss_kt3m = edgeworth_E_sigma_sigma(
                C_kl, kt3_k - eps_kt, kt4_k, kt3_l, kt4_l, activation
            )
            dcorr_dkt3 = (E_ss_kt3p - E_ss_kt3m) / (2.0 * eps_kt)

            E_ss_kt4p = edgeworth_E_sigma_sigma(
                C_kl, kt3_k, kt4_k + eps_kt, kt3_l, kt4_l, activation
            )
            E_ss_kt4m = edgeworth_E_sigma_sigma(
                C_kl, kt3_k, kt4_k - eps_kt, kt3_l, kt4_l, activation
            )
            dcorr_dkt4 = (E_ss_kt4p - E_ss_kt4m) / (2.0 * eps_kt)

            # Path (a) contribution through C_{kl}'s dependence on v_k and c_{kl}:
            # We also need the correction's dependence on C_{kl} entries.
            # The Gaussian grad_E_ss gives dE_G[sigma_k sigma_l]/dC.
            # For the correction, finite-diff on v_k (C[0,0]):
            C_vkp = C_kl.copy(); C_vkp[0, 0] += eps_kt
            C_vkm = C_kl.copy(); C_vkm[0, 0] -= eps_kt
            corr_vkp = edgeworth_E_sigma_sigma(C_vkp, kt3_k, kt4_k, kt3_l, kt4_l, activation) - E_ss_gauss_fn(C_vkp)
            corr_vkm = edgeworth_E_sigma_sigma(C_vkm, kt3_k, kt4_k, kt3_l, kt4_l, activation) - E_ss_gauss_fn(C_vkm)
            dcorr_dvk = (corr_vkp - corr_vkm) / (2.0 * eps_kt)

            C_cp = C_kl.copy(); C_cp[0, 1] += eps_kt; C_cp[1, 0] += eps_kt
            C_cm = C_kl.copy(); C_cm[0, 1] -= eps_kt; C_cm[1, 0] -= eps_kt
            corr_cp = edgeworth_E_sigma_sigma(C_cp, kt3_k, kt4_k, kt3_l, kt4_l, activation) - E_ss_gauss_fn(C_cp)
            corr_cm = edgeworth_E_sigma_sigma(C_cm, kt3_k, kt4_k, kt3_l, kt4_l, activation) - E_ss_gauss_fn(C_cm)
            dcorr_dc = (corr_cp - corr_cm) / (2.0 * eps_kt)

            d_Ess_corr_via_C = dcorr_dvk * 2.0 * Sw_k + dcorr_dc * Sw_l

            d_Ess = (
                d_Ess_gauss
                + d_Ess_corr_via_C
                + dcorr_dkt3 * gkt3_k
                + dcorr_dkt4 * gkt4_k
            )

            cross_grad += 2.0 * a[k] * a[l] * d_Ess

        grad_W[k] = stein_grad + cross_grad

    return grad_W


def compute_edgeworth_gradients(
    a: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    Sigma_beta: np.ndarray,
    E_y2: float,
    maf: np.ndarray,
    activation: str = "relu",
    Sigma_inv_sqrt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both Edgeworth-corrected gradients (dL_EW/da, dL_EW/dW).

    Returns:
        (grad_a, grad_W) -- shapes (m,) and (m, p).
    """
    if Sigma_inv_sqrt is None:
        Sigma_inv_sqrt = decorrelation_matrix(Sigma)

    grad_a = compute_edgeworth_grad_a(
        a, W, Sigma, Sigma_beta, maf, activation, Sigma_inv_sqrt
    )
    grad_W = compute_edgeworth_grad_W(
        a, W, Sigma, Sigma_beta, maf, activation, Sigma_inv_sqrt
    )
    return grad_a, grad_W
