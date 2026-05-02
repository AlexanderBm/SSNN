"""
Edgeworth-corrected expectations of activation functions.

Under the true (non-Gaussian) genotype distribution, the density of
the standardized projection z = w^T x / sqrt(w^T Sigma w) is:

    f_z(t) = phi(t) [1 + (kt3/6) H_3(t) + (kt4/24) H_4(t)
                        + (kt3^2/72) H_6(t) + ...]

where phi is the standard Gaussian density, H_k are probabilist's Hermite
polynomials, and kt3, kt4 are the standardized projection cumulants.

For any test function g, the corrected expectation is:
    E_true[g(z)] = E_gauss[g(z)]
                 + (kt3/6) E_gauss[g(z) H_3(z)]
                 + (kt4/24) E_gauss[g(z) H_4(z)]
                 + (kt3^2/72) E_gauss[g(z) H_6(z)]
                 + ...

All correction integrals are still 1D Gaussian expectations (with
polynomial weights) and admit closed forms for ReLU, sigmoid, etc.

This module provides those correction integrals and the Edgeworth-corrected
versions of the three key expectations needed by the population risk:
    E[sigma(z)], E[sigma'(z)], E[sigma(z_k) sigma(z_l)]
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ===================================================================
# Hermite polynomials (probabilist's convention)
# ===================================================================

def hermite_3(t: float | np.ndarray) -> float | np.ndarray:
    """H_3(t) = t^3 - 3t"""
    return t**3 - 3.0 * t


def hermite_4(t: float | np.ndarray) -> float | np.ndarray:
    """H_4(t) = t^4 - 6t^2 + 3"""
    return t**4 - 6.0 * t**2 + 3.0


def hermite_6(t: float | np.ndarray) -> float | np.ndarray:
    """H_6(t) = t^6 - 15t^4 + 45t^2 - 15"""
    return t**6 - 15.0 * t**4 + 45.0 * t**2 - 15.0


# ===================================================================
# Gaussian expectations of activation * Hermite polynomial
#
# These are the correction integrals: E_gauss[g(z) H_k(z)]
# where z ~ N(0, 1) and g is an activation function or its derivative.
# ===================================================================

# -------------------------------------------------------------------
# ReLU: sigma(z) = max(0, z)
# -------------------------------------------------------------------

def _relu_E_sigma_H3() -> float:
    r"""E[ReLU(z) H_3(z)] for z ~ N(0,1).

    = \int_0^\infty t (t^3 - 3t) phi(t) dt
    = \int_0^\infty (t^4 - 3t^2) phi(t) dt

    Using half-Gaussian moments: E[t^n | t>0] * P(t>0):
        int_0^inf t^2 phi(t) dt = 1/2
        int_0^inf t^4 phi(t) dt = 3/2
    Result: 3/2 - 3*1/2 = 0
    """
    return 0.0


def _relu_E_sigma_H4() -> float:
    r"""E[ReLU(z) H_4(z)] for z ~ N(0,1).

    = \int_0^\infty t (t^4 - 6t^2 + 3) phi(t) dt
    = \int_0^\infty (t^5 - 6t^3 + 3t) phi(t) dt

    Half-Gaussian moments:
        int_0^inf t phi(t) dt = 1/sqrt(2pi)
        int_0^inf t^3 phi(t) dt = 2/sqrt(2pi)
        int_0^inf t^5 phi(t) dt = 8/sqrt(2pi)
    Result: 8/sqrt(2pi) - 12/sqrt(2pi) + 3/sqrt(2pi) = -1/sqrt(2pi)
    """
    return -1.0 / np.sqrt(2.0 * np.pi)


def _relu_E_sigma_H6() -> float:
    r"""E[ReLU(z) H_6(z)] for z ~ N(0,1).

    = \int_0^\infty t (t^6 - 15t^4 + 45t^2 - 15) phi(t) dt

    Half-Gaussian moments:
        int_0^inf t^{2k+1} phi(t) dt = (2k)!! / sqrt(2pi) = (2k)! / (2^k k! sqrt(2pi))
        k=0: 1/sqrt(2pi)
        k=1: 2/sqrt(2pi)
        k=2: 8/sqrt(2pi)
        k=3: 48/sqrt(2pi)
    Result: 48 - 15*8 + 45*2 - 15*1 = 48 - 120 + 90 - 15 = 3, all / sqrt(2pi)
    """
    return 3.0 / np.sqrt(2.0 * np.pi)


def _relu_E_sigma_prime_H3() -> float:
    r"""E[sigma'(z) H_3(z)] for ReLU, z ~ N(0,1).

    sigma'(z) = 1(z>0), so:
    = \int_0^\infty (t^3 - 3t) phi(t) dt

    Half-Gaussian moments:
        int_0^inf t phi(t) dt = 1/sqrt(2pi)
        int_0^inf t^3 phi(t) dt = 2/sqrt(2pi)
    Result: 2/sqrt(2pi) - 3/sqrt(2pi) = -1/sqrt(2pi)
    """
    return -1.0 / np.sqrt(2.0 * np.pi)


def _relu_E_sigma_prime_H4() -> float:
    r"""E[sigma'(z) H_4(z)] for ReLU, z ~ N(0,1).

    = \int_0^\infty (t^4 - 6t^2 + 3) phi(t) dt

    Half-Gaussian moments:
        int_0^inf t^0 phi(t) dt = 1/2
        int_0^inf t^2 phi(t) dt = 1/2
        int_0^inf t^4 phi(t) dt = 3/2
    Result: 3/2 - 6/2 + 3/2 = 0
    """
    return 0.0


def _relu_E_sigma_prime_H6() -> float:
    r"""E[sigma'(z) H_6(z)] for ReLU, z ~ N(0,1).

    = \int_0^\infty (t^6 - 15t^4 + 45t^2 - 15) phi(t) dt

    Half-Gaussian moments (even powers):
        int_0^inf t^{2k} phi(t) dt = (2k-1)!! / 2
        k=0: 1/2
        k=1: 1/2
        k=2: 3/2
        k=3: 15/2
    Result: 15/2 - 15*3/2 + 45*1/2 - 15*1/2 = (15 - 45 + 45 - 15)/2 = 0
    """
    return 0.0


# -------------------------------------------------------------------
# Sigmoid: probit approximation sigma(z) ~ Phi(lambda z)
# -------------------------------------------------------------------
# Closed forms are harder for sigmoid Hermite integrals; we use
# numerical quadrature with Gauss-Hermite nodes which is exact for
# polynomial * Gaussian integrands up to high order.

_LAMBDA = np.sqrt(np.pi / 8.0)

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)


def _gauss_hermite_expect(func) -> float:
    """E[func(z)] for z ~ N(0,1) via Gauss-Hermite quadrature.

    hermegauss uses the probabilist's weight w(x) = exp(-x^2/2),
    so E[f(z)] = (1/sqrt(2pi)) * sum_i w_i f(x_i).
    """
    return float(np.sum(_GH_WEIGHTS * func(_GH_NODES)) / np.sqrt(2.0 * np.pi))


def _sigmoid_approx(z):
    """Probit-approximated sigmoid: Phi(lambda * z)."""
    return norm.cdf(_LAMBDA * z)


def _sigmoid_approx_prime(z):
    """Derivative of probit-approximated sigmoid: lambda * phi(lambda * z)."""
    return _LAMBDA * norm.pdf(_LAMBDA * z)


def _sigmoid_E_sigma_H3() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx(z) * hermite_3(z))


def _sigmoid_E_sigma_H4() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx(z) * hermite_4(z))


def _sigmoid_E_sigma_H6() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx(z) * hermite_6(z))


def _sigmoid_E_sigma_prime_H3() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx_prime(z) * hermite_3(z))


def _sigmoid_E_sigma_prime_H4() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx_prime(z) * hermite_4(z))


def _sigmoid_E_sigma_prime_H6() -> float:
    return _gauss_hermite_expect(lambda z: _sigmoid_approx_prime(z) * hermite_6(z))


# -------------------------------------------------------------------
# Identity: sigma(z) = z  (linear baseline)
# -------------------------------------------------------------------
# By Hermite orthogonality, E[z^j H_k(z)] = 0 for k > j.
# Since sigma(z) = z (degree 1) and sigma'(z) = 1 (degree 0),
# all corrections with H_k for k >= 3 vanish identically.

def _identity_E_sigma_H3() -> float:
    """E[z H_3(z)] = E[z(z^3 - 3z)] = E[z^4] - 3E[z^2] = 3 - 3 = 0."""
    return 0.0


def _identity_E_sigma_H4() -> float:
    """E[z H_4(z)] = 0 by Hermite orthogonality (deg 1 < 4)."""
    return 0.0


def _identity_E_sigma_H6() -> float:
    return 0.0


def _identity_E_sigma_prime_H3() -> float:
    """E[1 * H_3(z)] = E[H_3(z)] = 0 by orthogonality."""
    return 0.0


def _identity_E_sigma_prime_H4() -> float:
    return 0.0


def _identity_E_sigma_prime_H6() -> float:
    return 0.0


# ===================================================================
# Dispatch tables
# ===================================================================

_EW_SIGMA_CORRECTIONS = {
    "relu": (_relu_E_sigma_H3, _relu_E_sigma_H4, _relu_E_sigma_H6),
    "sigmoid": (_sigmoid_E_sigma_H3, _sigmoid_E_sigma_H4, _sigmoid_E_sigma_H6),
    "identity": (_identity_E_sigma_H3, _identity_E_sigma_H4, _identity_E_sigma_H6),
}

_EW_SIGMA_PRIME_CORRECTIONS = {
    "relu": (_relu_E_sigma_prime_H3, _relu_E_sigma_prime_H4, _relu_E_sigma_prime_H6),
    "sigmoid": (_sigmoid_E_sigma_prime_H3, _sigmoid_E_sigma_prime_H4, _sigmoid_E_sigma_prime_H6),
    "identity": (_identity_E_sigma_prime_H3, _identity_E_sigma_prime_H4, _identity_E_sigma_prime_H6),
}


def _get_corrections(activation: str):
    """Return (sigma_corrections, sigma_prime_corrections) for the activation."""
    if activation not in _EW_SIGMA_CORRECTIONS:
        raise ValueError(
            f"Unknown activation: {activation!r}. "
            f"Choose from {list(_EW_SIGMA_CORRECTIONS)}"
        )
    return _EW_SIGMA_CORRECTIONS[activation], _EW_SIGMA_PRIME_CORRECTIONS[activation]


# ===================================================================
# Edgeworth-corrected expectations: the public API
# ===================================================================

def edgeworth_E_sigma(
    v: float,
    kt3: float,
    kt4: float,
    activation: str,
) -> float:
    """Edgeworth-corrected E[sigma(z_raw)] where z_raw ~ N(0, v).

    The Edgeworth expansion is expressed in terms of the *standardized*
    projection z_std = z_raw / sqrt(v) ~ N(0, 1):

        E_true[sigma(z_raw)] = E_gauss[sigma(sqrt(v) z_std)]
            + (kt3/6)    E_gauss[sigma(sqrt(v) z_std) H_3(z_std)]
            + (kt4/24)   E_gauss[sigma(sqrt(v) z_std) H_4(z_std)]
            + (kt3^2/72) E_gauss[sigma(sqrt(v) z_std) H_6(z_std)]

    For ReLU: sigma(sqrt(v) z_std) = sqrt(v) ReLU(z_std), so the
    correction integrals each scale by sqrt(v).

    For sigmoid: sigma(sqrt(v) z_std) = sigmoid(sqrt(v) z_std), which
    depends on v in a non-trivial way; computed via Gauss-Hermite quadrature.

    For identity: all corrections vanish by Hermite orthogonality.
    """
    from .activations import get_activation
    E_sigma_gauss, _, _ = get_activation(activation)

    sqrtv = np.sqrt(max(v, 0.0))

    if activation == "relu":
        # sigma(sqrt(v) z) = sqrt(v) ReLU(z), corrections scale by sqrt(v)
        sigma_corr, _ = _get_corrections(activation)
        H3_corr, H4_corr, H6_corr = sigma_corr
        return (
            E_sigma_gauss(v)
            + sqrtv * (kt3 / 6.0) * H3_corr()
            + sqrtv * (kt4 / 24.0) * H4_corr()
            + sqrtv * (kt3**2 / 72.0) * H6_corr()
        )
    elif activation == "sigmoid":
        def sig(z):
            return norm.cdf(_LAMBDA * sqrtv * z)
        H3_val = _gauss_hermite_expect(lambda z: sig(z) * hermite_3(z))
        H4_val = _gauss_hermite_expect(lambda z: sig(z) * hermite_4(z))
        H6_val = _gauss_hermite_expect(lambda z: sig(z) * hermite_6(z))
        return (
            E_sigma_gauss(v)
            + (kt3 / 6.0) * H3_val
            + (kt4 / 24.0) * H4_val
            + (kt3**2 / 72.0) * H6_val
        )
    else:
        # identity: corrections vanish by Hermite orthogonality
        return float(E_sigma_gauss(v))


def edgeworth_E_sigma_prime(
    v: float,
    kt3: float,
    kt4: float,
    activation: str,
) -> float:
    """Edgeworth-corrected E[sigma'(z_raw)] where z_raw ~ N(0, v).

    The expansion is in terms of the standardized variable z_std = z_raw / sqrt(v):

        E_true[sigma'(z_raw)] = E_gauss[sigma'(sqrt(v) z_std)]
            + (kt3/6)    E_gauss[sigma'(sqrt(v) z_std) H_3(z_std)]
            + (kt4/24)   E_gauss[sigma'(sqrt(v) z_std) H_4(z_std)]
            + (kt3^2/72) E_gauss[sigma'(sqrt(v) z_std) H_6(z_std)]

    For ReLU: sigma'(sqrt(v) z_std) = 1(z_std > 0), which is v-independent,
    so the correction integrals reduce to v-independent constants.

    For sigmoid: sigma'(sqrt(v) z_std) = lambda phi(lambda sqrt(v) z_std),
    which depends on v; correction integrals are computed via Gauss-Hermite
    quadrature at the correct scale.

    For identity: all corrections vanish by Hermite orthogonality.
    """
    from .activations import get_activation
    _, E_sigma_prime_gauss, _ = get_activation(activation)

    if activation == "relu":
        # ReLU'(sqrt(v) z) = 1(z > 0), independent of v — use precomputed constants
        _, sigma_prime_corr = _get_corrections(activation)
        H3_corr, H4_corr, H6_corr = sigma_prime_corr
        return (
            E_sigma_prime_gauss(v)
            + (kt3 / 6.0) * H3_corr()
            + (kt4 / 24.0) * H4_corr()
            + (kt3**2 / 72.0) * H6_corr()
        )
    elif activation == "sigmoid":
        sqrtv = np.sqrt(max(v, 0.0))
        # sigma'(sqrt(v) z) = lambda * phi(lambda * sqrt(v) * z)
        def sp(z):
            return _LAMBDA * norm.pdf(_LAMBDA * sqrtv * z)
        H3_val = _gauss_hermite_expect(lambda z: sp(z) * hermite_3(z))
        H4_val = _gauss_hermite_expect(lambda z: sp(z) * hermite_4(z))
        H6_val = _gauss_hermite_expect(lambda z: sp(z) * hermite_6(z))
        return (
            E_sigma_prime_gauss(v)
            + (kt3 / 6.0) * H3_val
            + (kt4 / 24.0) * H4_val
            + (kt3**2 / 72.0) * H6_val
        )
    else:
        # identity: corrections vanish
        return float(E_sigma_prime_gauss(v))


def edgeworth_E_sigma_sigma(
    C: np.ndarray,
    kt3_k: float,
    kt4_k: float,
    kt3_l: float,
    kt4_l: float,
    activation: str,
) -> float:
    """Edgeworth-corrected E[sigma(z_k) sigma(z_l)].

    The cross-term correction is more involved because the Edgeworth
    expansion applies to each projection separately.  We use a first-order
    product expansion:

    E_true[sigma(z_k) sigma(z_l)]
        ≈ E_gauss[sigma(z_k) sigma(z_l)]
          + (kt3_k/6) E_gauss[sigma(z_k) H_3(z_k) * sigma(z_l)]
          + (kt3_l/6) E_gauss[sigma(z_k) * sigma(z_l) H_3(z_l)]
          + (kt4_k/24) E_gauss[sigma(z_k) H_4(z_k) * sigma(z_l)]
          + (kt4_l/24) E_gauss[sigma(z_k) * sigma(z_l) H_4(z_l)]
          + ...

    The 2D correction integrals E_gauss[sigma(z_k) H_r(z_k) sigma(z_l)]
    are computed via Gauss-Hermite quadrature over the joint distribution
    of (z_k, z_l).

    For efficiency, we compute these via the conditional:
        z_k | z_l = rho * z_l + sqrt(1 - rho^2) * eps,  eps ~ N(0,1)

    This reduces 2D integrals to products of 1D integrals.
    """
    from .activations import get_activation
    _, _, E_sigma_sigma_gauss = get_activation(activation)

    gauss_term = E_sigma_sigma_gauss(C)

    v_k = C[0, 0]
    v_l = C[1, 1]
    if v_k <= 0 or v_l <= 0:
        return gauss_term

    c_kl = C[0, 1]
    rho = c_kl / np.sqrt(v_k * v_l)
    rho = np.clip(rho, -1.0 + 1e-10, 1.0 - 1e-10)

    correction = _cross_term_correction_quadrature(
        v_k, v_l, rho, kt3_k, kt4_k, kt3_l, kt4_l, activation
    )

    return gauss_term + correction


def _cross_term_correction_quadrature(
    v_k: float,
    v_l: float,
    rho: float,
    kt3_k: float,
    kt4_k: float,
    kt3_l: float,
    kt4_l: float,
    activation: str,
) -> float:
    """Compute the 2D Edgeworth correction terms via quadrature.

    Uses the conditional factorization of the bivariate Gaussian:
        z_k = rho * z_l + sqrt(1-rho^2) * eps

    where z_k and z_l are the *standardized* projections (variance 1).
    The activation must be applied to the *raw* projections z_raw = sqrt(v) * z_std,
    not to the standardized nodes directly.

    Vectorized over the Gauss-Hermite nodes for efficiency.
    """
    sigma_func = _get_sigma_func(activation)

    n_nodes = 20
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
    norm_factor = 1.0 / np.sqrt(2.0 * np.pi)

    sqrt_1_minus_rho2 = np.sqrt(max(1.0 - rho**2, 1e-15))
    sqrt_vk = np.sqrt(max(v_k, 0.0))
    sqrt_vl = np.sqrt(max(v_l, 0.0))

    # Outer product of nodes: zl[i], eps[j]
    zl_grid = nodes[:, None]       # (n, 1)
    eps_grid = nodes[None, :]      # (1, n)
    wl_grid = weights[:, None]     # (n, 1)
    wk_grid = weights[None, :]     # (1, n)

    # Standardized conditional projections
    zk_grid = rho * zl_grid + sqrt_1_minus_rho2 * eps_grid  # (n, n)

    # Apply activation to the raw (unscaled) projections: z_raw = sqrt(v) * z_std
    sig_zk = sigma_func(sqrt_vk * zk_grid)  # (n, n)
    sig_zl = sigma_func(sqrt_vl * zl_grid)  # (n, 1) broadcast

    joint_w = wl_grid * wk_grid * norm_factor**2  # (n, n)

    product = joint_w * sig_zk * sig_zl

    # Correction from z_k's non-Gaussianity
    h3_zk = hermite_3(zk_grid)
    h4_zk = hermite_4(zk_grid)
    corr_k = np.sum(product * ((kt3_k / 6.0) * h3_zk + (kt4_k / 24.0) * h4_zk))

    # Correction from z_l's non-Gaussianity
    h3_zl = hermite_3(zl_grid)
    h4_zl = hermite_4(zl_grid)
    corr_l = np.sum(product * ((kt3_l / 6.0) * h3_zl + (kt4_l / 24.0) * h4_zl))

    return float(corr_k + corr_l)


def get_sigma_prime_correction_values(
    v: float,
    activation: str,
) -> tuple[float, float, float]:
    """Return (I3p, I4p, I6p) = E[sigma'(sqrt(v) z) H_k(z)] for k = 3, 4, 6.

    These are the correction integrals used in the Edgeworth gradient.
    For ReLU they are v-independent constants; for sigmoid they depend on v
    and are computed via Gauss-Hermite quadrature.
    """
    if activation == "relu":
        _, sigma_prime_corr = _get_corrections(activation)
        return sigma_prime_corr[0](), sigma_prime_corr[1](), sigma_prime_corr[2]()
    elif activation == "sigmoid":
        sqrtv = np.sqrt(max(v, 0.0))
        def sp(z):
            return _LAMBDA * norm.pdf(_LAMBDA * sqrtv * z)
        I3p = float(_gauss_hermite_expect(lambda z: sp(z) * hermite_3(z)))
        I4p = float(_gauss_hermite_expect(lambda z: sp(z) * hermite_4(z)))
        I6p = float(_gauss_hermite_expect(lambda z: sp(z) * hermite_6(z)))
        return I3p, I4p, I6p
    else:
        # identity: all corrections are zero
        return 0.0, 0.0, 0.0


def _get_sigma_func(activation: str):
    """Return the pointwise activation function."""
    if activation == "relu":
        return lambda z: np.maximum(0.0, z)
    elif activation == "sigmoid":
        return lambda z: norm.cdf(_LAMBDA * z)
    elif activation == "identity":
        return lambda z: z
    else:
        raise ValueError(f"Unknown activation: {activation!r}")
