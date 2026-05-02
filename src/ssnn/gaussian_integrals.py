"""
Gaussian integral machinery: Stein's lemma and pairwise covariance helpers.

Stein's lemma (multivariate):
    For x ~ N(0, Sigma) and weakly differentiable g with E[|g'|] < inf,
        E[x * g(w^T x)] = Sigma @ w * E[g'(w^T x)]

This module provides the building blocks that population_risk.py uses to
express the neural network loss and gradients entirely in terms of
summary statistics (Sigma, Sigma @ beta*).
"""

from __future__ import annotations

import numpy as np

from .activations import get_activation


def projection_variance(Sigma: np.ndarray, w: np.ndarray) -> float:
    """Var(w^T x) = w^T Sigma w for x ~ N(0, Sigma)."""
    return float(w @ Sigma @ w)


def pairwise_covariance(Sigma: np.ndarray, w_k: np.ndarray, w_l: np.ndarray) -> np.ndarray:
    """2x2 covariance matrix of (w_k^T x, w_l^T x) for x ~ N(0, Sigma).

    C = [[w_k^T Sigma w_k,  w_k^T Sigma w_l],
         [w_l^T Sigma w_k,  w_l^T Sigma w_l]]
    """
    Sw_k = Sigma @ w_k
    Sw_l = Sigma @ w_l

    return np.array([
        [w_k @ Sw_k, w_k @ Sw_l],
        [w_l @ Sw_k, w_l @ Sw_l],
    ])


def stein_cross_moment(
    Sigma: np.ndarray,
    w_k: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str,
) -> float:
    """Compute E[y * sigma(w_k^T x)] using Stein's lemma.

    = beta*^T Sigma w_k * E[sigma'(z_k)]
    = (Sigma_beta)^T w_k * E[sigma'(z_k)]

    where z_k ~ N(0, w_k^T Sigma w_k).

    Args:
        Sigma: (p, p) LD covariance matrix.
        w_k: (p,) weight vector for hidden unit k.
        Sigma_beta: (p,) vector = Sigma @ beta*, recoverable from GWAS.
        activation: name of activation function.

    Returns:
        Scalar E[y * sigma(w_k^T x)].
    """
    _, E_sigma_prime, _ = get_activation(activation)

    v_k = projection_variance(Sigma, w_k)
    beta_Sigma_w = float(Sigma_beta @ w_k)

    return beta_Sigma_w * E_sigma_prime(v_k)


def activation_cross_moment(
    Sigma: np.ndarray,
    w_k: np.ndarray,
    w_l: np.ndarray,
    activation: str,
) -> float:
    """Compute E[sigma(w_k^T x) * sigma(w_l^T x)].

    This is a 2D Gaussian integral determined by the 2x2 covariance
    matrix C_kl of the projections (w_k^T x, w_l^T x).

    Args:
        Sigma: (p, p) LD covariance matrix.
        w_k, w_l: (p,) weight vectors.
        activation: name of activation function.

    Returns:
        Scalar E[sigma(w_k^T x) sigma(w_l^T x)].
    """
    _, _, E_sigma_sigma = get_activation(activation)
    C_kl = pairwise_covariance(Sigma, w_k, w_l)
    return E_sigma_sigma(C_kl)


def stein_gradient_helper(
    Sigma: np.ndarray,
    w_k: np.ndarray,
    Sigma_beta: np.ndarray,
    activation: str,
) -> np.ndarray:
    """Compute E[x * sigma'(w_k^T x)] via Stein's lemma applied to sigma'.

    For the w_k gradient we need E[(y - f(x)) sigma'(w_k^T x) x].
    The x-dependent part factors via Stein's lemma:

        E[x sigma'(w_k^T x)] = Sigma w_k * E[sigma''(w_k^T x)]

    For ReLU, sigma''(z) = delta(0) in distributional sense, but
    we only need this through the chain:
        dL/dw_k uses E[y sigma'(w_k^T x) x] = Sigma w_k * (beta*^T Sigma w_k) * E[sigma''(z_k)]
    which is handled at the population_risk level.

    This function returns Sigma @ w_k, the vector that Stein's lemma
    extracts, to be combined with the appropriate scalar expectations
    in the gradient computation.
    """
    return Sigma @ w_k
