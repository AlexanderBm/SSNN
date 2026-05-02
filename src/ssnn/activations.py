"""
Closed-form expectations of activation functions under Gaussian inputs.

For a one-hidden-layer network f(x) = sum_k a_k sigma(w_k^T x) with
x ~ N(0, Sigma), the population risk L(a, W) = E[(y - f(x))^2] requires
three types of expectations for each activation sigma:

    E[sigma(z)]           where z ~ N(0, v)
    E[sigma'(z)]          where z ~ N(0, v)
    E[sigma(z_k) sigma(z_l)]  where (z_k, z_l) ~ N(0, C_kl)

This module provides closed forms for ReLU, sigmoid (probit approx),
and identity activations.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# ReLU: sigma(z) = max(0, z)
# ---------------------------------------------------------------------------

def relu_E_sigma(v: float) -> float:
    """E[ReLU(z)] for z ~ N(0, v).

    = sqrt(v / (2 pi))
    """
    return np.sqrt(v / (2.0 * np.pi))


def relu_E_sigma_prime(v: float) -> float:
    """E[ReLU'(z)] = E[1(z > 0)] = 1/2 by Gaussian symmetry."""
    return 0.5


def relu_E_sigma_sigma(C: np.ndarray) -> float:
    """E[ReLU(z_k) ReLU(z_l)] for (z_k, z_l) ~ N(0, C).

    Arc-cosine kernel of degree 1 (Cho & Saul 2009):

        = sqrt(v_k v_l) / (2 pi) * (sin(theta) + (pi - theta) cos(theta))

    where theta = arccos(C_01 / sqrt(v_k v_l)),  v_k = C_00, v_l = C_11.
    """
    v_k = C[0, 0]
    v_l = C[1, 1]
    c_kl = C[0, 1]

    if v_k <= 0 or v_l <= 0:
        return 0.0

    rho = np.clip(c_kl / np.sqrt(v_k * v_l), -1.0, 1.0)
    theta = np.arccos(rho)

    return np.sqrt(v_k * v_l) / (2.0 * np.pi) * (
        np.sin(theta) + (np.pi - theta) * np.cos(theta)
    )


# ---------------------------------------------------------------------------
# Sigmoid: sigma(z) = 1 / (1 + exp(-z)),  probit approximation
# ---------------------------------------------------------------------------
#
# The probit approximation sigma(z) ≈ Phi(lambda z) with lambda = sqrt(pi/8)
# gives analytically tractable Gaussian expectations.  This is the standard
# approximation used in Bayesian neural network / GP literature.

_LAMBDA = np.sqrt(np.pi / 8.0)


def sigmoid_E_sigma(v: float) -> float:
    """E[sigmoid(z)] = 1/2 for z ~ N(0, v), by the antisymmetry
    sigma(-z) = 1 - sigma(z)."""
    return 0.5


def sigmoid_E_sigma_prime(v: float) -> float:
    """E[sigmoid'(z)] under the probit approximation.

    sigmoid'(z) ≈ lambda phi(lambda z), so
    E[sigmoid'(z)] = lambda / sqrt(1 + lambda^2 v) * (1 / sqrt(2 pi))

    More precisely:
        E[lambda phi(lambda z)] = lambda / sqrt(2 pi (1 + lambda^2 v))
    """
    return _LAMBDA / np.sqrt(2.0 * np.pi * (1.0 + _LAMBDA**2 * v))


def sigmoid_E_sigma_sigma(C: np.ndarray) -> float:
    """E[sigmoid(z_k) sigmoid(z_l)] under the probit approximation.

    Under sigmoid(z) ≈ Phi(lambda z), this is the bivariate normal CDF
    Phi_2(0, 0; rho_eff) = 1/4 + arcsin(rho_eff) / (2 pi), where
    rho_eff = lambda^2 C_01 / sqrt((1 + lambda^2 v_k)(1 + lambda^2 v_l)).
    """
    v_k = C[0, 0]
    v_l = C[1, 1]
    c_kl = C[0, 1]

    denom = np.sqrt((1.0 + _LAMBDA**2 * v_k) * (1.0 + _LAMBDA**2 * v_l))
    rho_eff = np.clip(_LAMBDA**2 * c_kl / denom, -1.0, 1.0)

    return 0.25 + np.arcsin(rho_eff) / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Identity: sigma(z) = z  (linear baseline, for the reduction theorem)
# ---------------------------------------------------------------------------

def identity_E_sigma(v: float) -> float:
    """E[z] = 0 for z ~ N(0, v)."""
    return 0.0


def identity_E_sigma_prime(v: float) -> float:
    """E[1] = 1."""
    return 1.0


def identity_E_sigma_sigma(C: np.ndarray) -> float:
    """E[z_k z_l] = Cov(z_k, z_l) = C_01 for zero-mean Gaussians."""
    return C[0, 1]


# ---------------------------------------------------------------------------
# Derivatives of Gaussian expectations w.r.t. variance / covariance
# (needed for analytic W-gradients)
# ---------------------------------------------------------------------------

def relu_dE_sigma_prime_dv(v: float) -> float:
    """d/dv E[ReLU'(z)] = d/dv (1/2) = 0."""
    return 0.0


def sigmoid_dE_sigma_prime_dv(v: float) -> float:
    r"""d/dv E[sigmoid'(z)] for z ~ N(0,v).

    E[sigma'(z)] = lambda / sqrt(2 pi (1 + lambda^2 v))

    d/dv = -lambda^3 / (2 * (2 pi)^{1/2} * (1 + lambda^2 v)^{3/2})
    """
    denom = (1.0 + _LAMBDA**2 * v)
    return -_LAMBDA**3 / (2.0 * np.sqrt(2.0 * np.pi) * denom**1.5)


def identity_dE_sigma_prime_dv(v: float) -> float:
    """d/dv E[1] = 0."""
    return 0.0


def relu_grad_E_sigma_sigma(C: np.ndarray) -> np.ndarray:
    r"""Gradient of E[ReLU(z_k)ReLU(z_l)] w.r.t. C = [[v_k, c], [c, v_l]].

    Returns a 2x2 array [[dF/dv_k, dF/dc], [dF/dc, dF/dv_l]].

    F = sqrt(v_k v_l)/(2pi) * g(theta),
    g(theta) = sin(theta) + (pi-theta)cos(theta),
    theta = arccos(rho), rho = c / sqrt(v_k v_l).

    Differentiation uses g'(theta) = (theta - pi)sin(theta) and
    d theta/d rho = -1/sin(theta), combined with
    d rho/d v_k = -rho/(2 v_k),  d rho/d c = 1/sqrt(v_k v_l).
    """
    v_k = C[0, 0]
    v_l = C[1, 1]
    c_kl = C[0, 1]

    if v_k <= 0 or v_l <= 0:
        return np.zeros((2, 2))

    sv = np.sqrt(v_k * v_l)
    rho = np.clip(c_kl / sv, -1.0, 1.0)
    theta = np.arccos(rho)
    sin_th = np.sin(theta)

    g = sin_th + (np.pi - theta) * np.cos(theta)

    # dF/dc:  dtheta/dc = -1/(sin(theta) * sqrt(v_k v_l))
    # dF/dc = sqrt(v_k v_l)/(2pi) * g'(theta) * dtheta/dc
    #       = sqrt(v_k v_l)/(2pi) * (theta - pi)*sin(theta) * (-1/(sin(theta)*sqrt(v_k v_l)))
    #       = (pi - theta) / (2 pi)
    dF_dc = (np.pi - theta) / (2.0 * np.pi)

    # dF/dv_k has two pieces: through the sqrt(v_k v_l) prefactor and through theta.
    # Prefactor piece: d/dv_k [sqrt(v_k v_l)] = sqrt(v_l)/(2 sqrt(v_k))
    #   => sqrt(v_l)/(2 sqrt(v_k)) * g(theta) / (2 pi)
    # Theta piece: dtheta/dv_k = (-1/sin(theta)) * drho/dv_k
    #   drho/dv_k = -c / (2 v_k sqrt(v_k v_l)) = -rho / (2 v_k)
    #   dtheta/dv_k = rho / (2 v_k sin(theta))
    #   => sqrt(v_k v_l)/(2pi) * (theta - pi)*sin(theta) * rho/(2 v_k sin(theta))
    #    = sqrt(v_k v_l)/(2pi) * (theta - pi) * rho / (2 v_k)
    #    = rho * (theta - pi) * sqrt(v_l) / (4 pi sqrt(v_k))

    prefactor_k = np.sqrt(v_l) / (4.0 * np.pi * np.sqrt(v_k)) * g
    theta_k = rho * (theta - np.pi) * np.sqrt(v_l) / (4.0 * np.pi * np.sqrt(v_k))
    dF_dvk = prefactor_k + theta_k

    prefactor_l = np.sqrt(v_k) / (4.0 * np.pi * np.sqrt(v_l)) * g
    theta_l = rho * (theta - np.pi) * np.sqrt(v_k) / (4.0 * np.pi * np.sqrt(v_l))
    dF_dvl = prefactor_l + theta_l

    return np.array([
        [dF_dvk, dF_dc],
        [dF_dc,  dF_dvl],
    ])


def sigmoid_grad_E_sigma_sigma(C: np.ndarray) -> np.ndarray:
    r"""Gradient of E[sigmoid(z_k)sigmoid(z_l)] w.r.t. C.

    F = 1/4 + arcsin(rho_eff) / (2 pi)
    rho_eff = lambda^2 c / sqrt((1+lambda^2 v_k)(1+lambda^2 v_l))

    Returns 2x2 [[dF/dv_k, dF/dc], [dF/dc, dF/dv_l]].
    """
    v_k = C[0, 0]
    v_l = C[1, 1]
    c_kl = C[0, 1]

    A = 1.0 + _LAMBDA**2 * v_k
    B = 1.0 + _LAMBDA**2 * v_l
    sAB = np.sqrt(A * B)
    rho_eff = np.clip(_LAMBDA**2 * c_kl / sAB, -1.0 + 1e-10, 1.0 - 1e-10)

    # d arcsin(x)/dx = 1/sqrt(1-x^2)
    darcsin = 1.0 / np.sqrt(1.0 - rho_eff**2)

    # d rho_eff / d c = lambda^2 / sAB
    drho_dc = _LAMBDA**2 / sAB

    # d rho_eff / d v_k = -lambda^4 c / (2 A sAB)
    drho_dvk = -_LAMBDA**4 * c_kl / (2.0 * A * sAB)
    drho_dvl = -_LAMBDA**4 * c_kl / (2.0 * B * sAB)

    factor = darcsin / (2.0 * np.pi)

    dF_dc = factor * drho_dc
    dF_dvk = factor * drho_dvk
    dF_dvl = factor * drho_dvl

    return np.array([
        [dF_dvk, dF_dc],
        [dF_dc,  dF_dvl],
    ])


def identity_grad_E_sigma_sigma(C: np.ndarray) -> np.ndarray:
    """Gradient of E[z_k z_l] = C_01 w.r.t. C.

    dF/dv_k = 0, dF/dv_l = 0, dF/dc = 1.
    """
    return np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])


# ---------------------------------------------------------------------------
# Dispatch by name
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "relu": (relu_E_sigma, relu_E_sigma_prime, relu_E_sigma_sigma),
    "sigmoid": (sigmoid_E_sigma, sigmoid_E_sigma_prime, sigmoid_E_sigma_sigma),
    "identity": (identity_E_sigma, identity_E_sigma_prime, identity_E_sigma_sigma),
}

ACTIVATION_DERIVS = {
    "relu": (relu_dE_sigma_prime_dv, relu_grad_E_sigma_sigma),
    "sigmoid": (sigmoid_dE_sigma_prime_dv, sigmoid_grad_E_sigma_sigma),
    "identity": (identity_dE_sigma_prime_dv, identity_grad_E_sigma_sigma),
}


def get_activation(name: str):
    """Return (E_sigma, E_sigma_prime, E_sigma_sigma) for the named activation."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


def get_activation_derivs(name: str):
    """Return (dE_sigma_prime_dv, grad_E_sigma_sigma) for the named activation.

    dE_sigma_prime_dv(v) -> float:
        Derivative of E[sigma'(z)] w.r.t. projection variance v.

    grad_E_sigma_sigma(C) -> (2, 2) array:
        Gradient of E[sigma(z_k) sigma(z_l)] w.r.t. the 2x2
        covariance matrix C = [[v_k, c], [c, v_l]].
    """
    if name not in ACTIVATION_DERIVS:
        raise ValueError(f"Unknown activation: {name!r}. Choose from {list(ACTIVATION_DERIVS)}")
    return ACTIVATION_DERIVS[name]
