"""Shared fixtures for the SSNN test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_problem(rng):
    """A small (p=10, m=3) test problem with known ground truth.

    Returns a dict with Sigma, beta_star, Sigma_beta, sigma_eps, E_y2,
    and a function to sample individual-level data.
    """
    p = 10
    m = 3
    sigma_eps = 1.0

    # Block-diagonal LD matrix: two 5x5 blocks with moderate correlation
    block = np.eye(5)
    for i in range(5):
        for j in range(5):
            block[i, j] = 0.5 ** abs(i - j)
    Sigma = np.block([[block, np.zeros((5, 5))],
                      [np.zeros((5, 5)), block]])

    beta_star = rng.standard_normal(p) * 0.3

    Sigma_beta = Sigma @ beta_star
    E_y2 = beta_star @ Sigma @ beta_star + sigma_eps**2

    def sample_data(n):
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        eps = rng.normal(0, sigma_eps, size=n)
        y = X @ beta_star + eps
        return X, y

    return {
        "p": p,
        "m": m,
        "Sigma": Sigma,
        "beta_star": beta_star,
        "Sigma_beta": Sigma_beta,
        "sigma_eps": sigma_eps,
        "E_y2": E_y2,
        "sample_data": sample_data,
    }
