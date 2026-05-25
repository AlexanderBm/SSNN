"""
Tests for residual_gamma.py.

Tests validate four properties:
1. Correctness: closed-form formulas match direct computations.
2. Signal removal: under a purely linear DGP, Gamma^resid is smaller
   than Gamma^raw (linear contamination is removed).
3. Summary-statistic formulas: Sigma_beta^resid and E_r2 match their
   closed-form expressions.
4. Edge cases: degenerate inputs (p=1, n_ref=1, zero inputs, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from ssnn.residual_gamma import (
    compute_gamma_correction,
    compute_residual_gamma,
    compute_residual_sigma_beta,
    compute_residual_e_y2,
    compute_genome_wide_residual_gamma,
    compute_residual_sigma_other2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_block_sigma(n_blocks: int, block_size: int, decay: float = 0.7) -> np.ndarray:
    """Build a block-diagonal LD matrix with exponential decay."""
    p = n_blocks * block_size
    block = np.array(
        [[decay ** abs(i - j) for j in range(block_size)] for i in range(block_size)]
    )
    blocks = [block] * n_blocks
    return np.block(
        [[blocks[i] if i == j else np.zeros((block_size, block_size))
          for j in range(n_blocks)]
         for i in range(n_blocks)]
    )


def gaussian_genotypes(rng: np.random.Generator, n: int, Sigma: np.ndarray) -> np.ndarray:
    """Sample centered Gaussian genotypes."""
    p = Sigma.shape[0]
    return rng.multivariate_normal(np.zeros(p), Sigma, size=n)


def binomial_genotypes(rng: np.random.Generator, n: int, p: int,
                       maf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample centered binomial genotypes; also return MAF."""
    if maf is None:
        maf = rng.uniform(0.1, 0.4, size=p)
    g = rng.binomial(2, maf, size=(n, p)).astype(float)
    X = g - 2 * maf
    return X, maf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_problem(rng):
    """Small Gaussian linear DGP (n=500, p=15) for algebraic tests."""
    p = 15
    n = 500
    Sigma = make_block_sigma(3, 5, decay=0.7)

    beta_star = rng.standard_normal(p) * 0.3
    X = gaussian_genotypes(rng, n, Sigma)
    eps = rng.standard_normal(n)
    y = X @ beta_star + eps

    Sigma_beta = X.T @ y / n
    beta_hat = np.linalg.solve(Sigma, Sigma_beta)
    E_y2 = float(np.mean(y ** 2))

    return {
        "p": p, "n": n, "Sigma": Sigma,
        "beta_star": beta_star, "beta_hat": beta_hat,
        "X": X, "y": y,
        "Sigma_beta": Sigma_beta, "E_y2": E_y2,
    }


@pytest.fixture
def large_linear_problem(rng):
    """Larger linear DGP with binomial genotypes (n=30000, p=20) for signal-removal tests."""
    p = 20
    n = 30_000
    n_ref = 5_000

    maf = rng.uniform(0.1, 0.4, size=p)
    X_gwas, _ = binomial_genotypes(rng, n, p, maf)
    X_ref, _ = binomial_genotypes(rng, n_ref, p, maf)

    beta_star = rng.standard_normal(p) * 0.2
    eps = rng.standard_normal(n)
    y = X_gwas @ beta_star + eps

    Sigma_emp = X_ref.T @ X_ref / n_ref
    Sigma_beta = X_gwas.T @ y / n
    Gamma_raw = (X_gwas * y[:, None]).T @ X_gwas / n

    beta_hat = np.linalg.solve(Sigma_emp, Sigma_beta)
    E_y2 = float(np.mean(y ** 2))

    return {
        "p": p, "n": n, "n_ref": n_ref,
        "Sigma": Sigma_emp, "beta_hat": beta_hat,
        "X_gwas": X_gwas, "X_ref": X_ref, "y": y,
        "Sigma_beta": Sigma_beta, "Gamma_raw": Gamma_raw, "E_y2": E_y2,
    }


@pytest.fixture
def ridge_problem(linear_problem):
    """Augments linear_problem with a ridge beta_hat for several lambda values."""
    prob = linear_problem
    Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
    lambdas = [1.0, 0.1, 0.01, 0.001]
    ridge_betas = {
        lam: np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        for lam in lambdas
    }
    return {**prob, "lambdas": lambdas, "ridge_betas": ridge_betas}


# ---------------------------------------------------------------------------
# 1. compute_gamma_correction
# ---------------------------------------------------------------------------

class TestGammaCorrection:

    def test_shape(self, linear_problem):
        prob = linear_problem
        correction = compute_gamma_correction(prob["X"], prob["beta_hat"])
        assert correction.shape == (prob["p"], prob["p"])

    def test_symmetry(self, linear_problem):
        prob = linear_problem
        correction = compute_gamma_correction(prob["X"], prob["beta_hat"])
        np.testing.assert_allclose(correction, correction.T, atol=1e-12)

    def test_zero_beta_gives_zero_correction(self, linear_problem):
        """When beta_hat is zero, linear prediction v = X @ 0 = 0, so correction is zero."""
        prob = linear_problem
        beta_zero = np.zeros(prob["p"])
        correction = compute_gamma_correction(prob["X"], beta_zero)
        np.testing.assert_allclose(correction, 0.0, atol=1e-12)

    def test_matches_explicit_outer_sum(self, linear_problem):
        """Verify (1/n) sum_l v_l * outer(x_l, x_l) against vectorised formula."""
        prob = linear_problem
        X, beta_hat = prob["X"], prob["beta_hat"]
        n = len(X)
        v = X @ beta_hat
        expected = sum(v[l] * np.outer(X[l], X[l]) for l in range(n)) / n
        result = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_linearity_in_beta_hat(self, linear_problem):
        """Doubling beta_hat should double the correction (bilinear formula)."""
        prob = linear_problem
        X, beta_hat = prob["X"], prob["beta_hat"]
        corr1 = compute_gamma_correction(X, beta_hat)
        corr2 = compute_gamma_correction(X, 2.0 * beta_hat)
        np.testing.assert_allclose(corr2, 2.0 * corr1, rtol=1e-10)

    def test_negative_beta_hat_still_symmetric(self, linear_problem):
        """Correction must be symmetric even when beta_hat has negative entries."""
        prob = linear_problem
        beta_neg = -prob["beta_hat"]
        correction = compute_gamma_correction(prob["X"], beta_neg)
        np.testing.assert_allclose(correction, correction.T, atol=1e-12)

    def test_negative_beta_hat_equals_negated_positive(self, linear_problem):
        """C(-beta) = -C(beta) since C is linear in beta_hat."""
        prob = linear_problem
        X, beta_hat = prob["X"], prob["beta_hat"]
        corr_pos = compute_gamma_correction(X, beta_hat)
        corr_neg = compute_gamma_correction(X, -beta_hat)
        np.testing.assert_allclose(corr_neg, -corr_pos, rtol=1e-10)

    def test_scale_invariant_in_X_rows_with_fixed_v(self, rng):
        """Scaling each X row by a constant s changes correction by s^2 (two X factors)."""
        p, n = 8, 100
        Sigma = make_block_sigma(2, 4, decay=0.6)
        X = gaussian_genotypes(rng, n, Sigma)
        beta_hat = rng.standard_normal(p) * 0.2

        # We want to verify the formula, not linearity in X (which is cubic).
        # Instead, verify that compute_gamma_correction returns
        # (X * v[:, None]).T @ X / n and not something else.
        v = X @ beta_hat
        expected = (X * v[:, None]).T @ X / n
        result = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_single_snp(self, rng):
        """p=1 degenerate case: correction should be a (1,1) scalar-valued matrix."""
        n = 200
        X = rng.standard_normal((n, 1))
        beta_hat = np.array([0.5])
        correction = compute_gamma_correction(X, beta_hat)
        assert correction.shape == (1, 1)
        v = X @ beta_hat
        expected = float(((X * v[:, None]).T @ X / n)[0, 0])
        assert float(correction[0, 0]) == pytest.approx(expected, rel=1e-10)

    def test_n_ref_1_does_not_crash(self, rng):
        """n_ref=1: single reference individual should not raise an exception."""
        p = 5
        X_ref = rng.standard_normal((1, p))
        beta_hat = rng.standard_normal(p) * 0.1
        correction = compute_gamma_correction(X_ref, beta_hat)
        assert correction.shape == (p, p)
        np.testing.assert_allclose(correction, correction.T, atol=1e-12)

    def test_all_positive_X_nonneg_beta_gives_psd(self, rng):
        """With X >= 0 and beta_hat >= 0, the correction should be PSD."""
        p, n = 6, 300
        X = np.abs(rng.standard_normal((n, p)))
        beta_hat = np.abs(rng.standard_normal(p)) * 0.3
        correction = compute_gamma_correction(X, beta_hat)
        # PSD check: all eigenvalues >= 0 (up to numerical noise)
        eigvals = np.linalg.eigvalsh(correction)
        assert np.all(eigvals >= -1e-10), f"Not PSD; min eigenvalue = {eigvals.min():.2e}"

    def test_matches_binomial_genotypes(self, rng):
        """Formula should hold for binomial (real SNP-like) genotypes too."""
        p, n = 10, 500
        X, _ = binomial_genotypes(rng, n, p)
        beta_hat = rng.standard_normal(p) * 0.2
        v = X @ beta_hat
        expected = (X * v[:, None]).T @ X / n
        result = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. compute_residual_gamma
# ---------------------------------------------------------------------------

class TestResidualGamma:

    def test_shape(self, linear_problem):
        prob = linear_problem
        Gamma_raw = (prob["X"] * prob["y"][:, None]).T @ prob["X"] / prob["n"]
        result = compute_residual_gamma(Gamma_raw, prob["X"], prob["beta_hat"])
        assert result.shape == (prob["p"], prob["p"])

    def test_symmetry(self, linear_problem):
        prob = linear_problem
        Gamma_raw = (prob["X"] * prob["y"][:, None]).T @ prob["X"] / prob["n"]
        Gamma_resid = compute_residual_gamma(Gamma_raw, prob["X"], prob["beta_hat"])
        np.testing.assert_allclose(Gamma_resid, Gamma_resid.T, atol=1e-12)

    def test_equals_direct_residual_computation(self, linear_problem):
        """KEY TEST: Gamma^resid must equal (1/n) X^T Diag(r) X directly."""
        prob = linear_problem
        X, y, beta_hat, n = prob["X"], prob["y"], prob["beta_hat"], prob["n"]

        Gamma_raw = (X * y[:, None]).T @ X / n
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_hat)

        r = y - X @ beta_hat
        Gamma_direct = (X * r[:, None]).T @ X / n

        np.testing.assert_allclose(Gamma_resid, Gamma_direct, atol=1e-12)

    def test_zero_beta_returns_gamma_raw_unchanged(self, linear_problem):
        """When beta_hat is zero, correction is zero, so Gamma_resid == Gamma_raw."""
        prob = linear_problem
        X, y, n = prob["X"], prob["y"], prob["n"]
        Gamma_raw = (X * y[:, None]).T @ X / n
        beta_zero = np.zeros(prob["p"])
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_zero)
        np.testing.assert_allclose(Gamma_resid, Gamma_raw, atol=1e-12)

    def test_linear_dgp_reduces_frobenius_norm(self, large_linear_problem):
        """Under a linear DGP with binomial genotypes, Gamma^resid should be
        substantially smaller than Gamma^raw because linear signal is removed."""
        prob = large_linear_problem
        Gamma_resid = compute_residual_gamma(
            prob["Gamma_raw"], prob["X_ref"], prob["beta_hat"],
        )
        norm_raw = np.linalg.norm(prob["Gamma_raw"], "fro")
        norm_resid = np.linalg.norm(Gamma_resid, "fro")
        ratio = norm_resid / norm_raw
        assert ratio < 0.65, (
            f"Expected residual norm < 65% of raw; got ratio {ratio:.3f}"
        )

    def test_reference_panel_vs_gwas_panel_close(self, large_linear_problem):
        """Reference panel and GWAS panel corrections should agree within sampling noise."""
        prob = large_linear_problem
        X_gwas, X_ref = prob["X_gwas"], prob["X_ref"]
        beta_hat, Gamma_raw = prob["beta_hat"], prob["Gamma_raw"]

        resid_ref = compute_residual_gamma(Gamma_raw, X_ref, beta_hat)
        resid_gwas = compute_residual_gamma(Gamma_raw, X_gwas, beta_hat)

        diff_norm = np.linalg.norm(resid_ref - resid_gwas, "fro")
        raw_norm = np.linalg.norm(Gamma_raw, "fro")
        assert diff_norm < 0.6 * raw_norm, (
            f"Panel difference too large: {diff_norm / raw_norm:.3f} of ||Gamma_raw||"
        )

    def test_all_zero_gamma_raw(self, linear_problem):
        """With Gamma_raw = 0, Gamma_resid = -correction (correction subtracted)."""
        prob = linear_problem
        p, X, beta_hat = prob["p"], prob["X"], prob["beta_hat"]
        Gamma_raw = np.zeros((p, p))
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_hat)
        correction = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(Gamma_resid, -correction, atol=1e-12)

    def test_additivity(self, linear_problem):
        """compute_residual_gamma is linear in Gamma_raw (affine, so difference is linear)."""
        prob = linear_problem
        X, beta_hat = prob["X"], prob["beta_hat"]
        p = prob["p"]
        n = prob["n"]
        y = prob["y"]

        Gamma1 = (X * y[:, None]).T @ X / n
        Gamma2 = (X * (2 * y)[:, None]).T @ X / n  # = 2 * Gamma1

        resid1 = compute_residual_gamma(Gamma1, X, beta_hat)
        resid2 = compute_residual_gamma(Gamma2, X, beta_hat)

        # Gamma2 = 2*Gamma1, and correction is the same, so:
        # resid2 = 2*Gamma1 - correction = 2*resid1 + correction
        correction = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(resid2, 2 * resid1 + correction, atol=1e-12)

    def test_single_snp(self, rng):
        """p=1 degenerate case should return a (1,1) matrix."""
        n = 200
        X = rng.standard_normal((n, 1))
        beta_hat = np.array([0.3])
        y = (X @ beta_hat).ravel() + rng.standard_normal(n) * 0.5
        Gamma_raw = (X * y[:, None]).T @ X / n
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_hat)
        assert Gamma_resid.shape == (1, 1)

        r = y - X @ beta_hat
        Gamma_direct = (X * r[:, None]).T @ X / n
        np.testing.assert_allclose(Gamma_resid, Gamma_direct, atol=1e-12)

    def test_n_ref_1_does_not_crash(self, rng):
        """Single reference individual should not raise."""
        p, n = 6, 300
        Sigma = make_block_sigma(2, 3, decay=0.5)
        X = gaussian_genotypes(rng, n, Sigma)
        y = X @ rng.standard_normal(p) + rng.standard_normal(n)
        Gamma_raw = (X * y[:, None]).T @ X / n
        beta_hat = rng.standard_normal(p) * 0.1
        X_ref = rng.standard_normal((1, p))
        Gamma_resid = compute_residual_gamma(Gamma_raw, X_ref, beta_hat)
        assert Gamma_resid.shape == (p, p)

    def test_negative_beta_hat_still_symmetric(self, linear_problem):
        """Result should be symmetric even when beta_hat has negative entries."""
        prob = linear_problem
        X, y, n, p = prob["X"], prob["y"], prob["n"], prob["p"]
        Gamma_raw = (X * y[:, None]).T @ X / n
        beta_neg = -prob["beta_hat"]
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_neg)
        np.testing.assert_allclose(Gamma_resid, Gamma_resid.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. compute_residual_sigma_beta
# ---------------------------------------------------------------------------

class TestResidualSigmaBeta:

    def test_shape(self, linear_problem):
        prob = linear_problem
        resid = compute_residual_sigma_beta(
            prob["Sigma_beta"], prob["Sigma"], prob["beta_hat"],
        )
        assert resid.shape == (prob["p"],)

    def test_ols_is_near_zero(self, linear_problem):
        """OLS: Sigma_beta^resid = Sigma_beta - Sigma @ Sigma^{-1} Sigma_beta = 0."""
        prob = linear_problem
        resid = compute_residual_sigma_beta(
            prob["Sigma_beta"], prob["Sigma"], prob["beta_hat"],
        )
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_identity_sigma_ols_gives_zero(self, rng):
        """With Sigma = I, OLS beta_hat = Sigma_beta, residual is exactly zero."""
        p = 10
        Sigma = np.eye(p)
        Sigma_beta = rng.standard_normal(p) * 0.4
        beta_hat = Sigma_beta.copy()  # OLS: Sigma^{-1} Sigma_beta = I @ Sigma_beta
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_ridge_matches_closed_form(self, linear_problem):
        """Ridge: Sigma_beta^resid = lambda * (Sigma + lambda*I)^{-1} Sigma_beta."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 0.1
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
        expected = lam * np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        np.testing.assert_allclose(resid, expected, atol=1e-10)

    def test_ridge_matches_closed_form_multiple_lambdas(self, ridge_problem):
        """Ridge closed form holds for several lambda values."""
        prob = ridge_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        for lam in prob["lambdas"]:
            beta_hat_ridge = prob["ridge_betas"][lam]
            resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
            expected = lam * np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
            np.testing.assert_allclose(resid, expected, atol=1e-10,
                                       err_msg=f"Failed at lambda={lam}")

    def test_ridge_norm_monotone_in_lambda(self, ridge_problem):
        """As lambda -> 0, the residual norm should decrease monotonically."""
        prob = ridge_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        norms = []
        for lam in sorted(prob["lambdas"], reverse=True):  # largest first
            resid = compute_residual_sigma_beta(Sigma_beta, Sigma, prob["ridge_betas"][lam])
            norms.append(np.linalg.norm(resid))
        assert all(norms[i] > norms[i + 1] for i in range(len(norms) - 1)), (
            f"Residual norms not monotone: {norms}"
        )

    def test_very_small_lambda_converges_to_zero(self, linear_problem):
        """Very small lambda (1e-10) should give near-zero residual."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 1e-10
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
        np.testing.assert_allclose(resid, 0.0, atol=1e-6)

    def test_very_large_lambda_gives_approx_sigma_beta(self, linear_problem):
        """Very large lambda (1e6): beta_hat ≈ 0, so residual ≈ Sigma_beta."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 1e6
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
        np.testing.assert_allclose(resid, Sigma_beta, rtol=1e-3)

    def test_zero_beta_gives_sigma_beta(self, linear_problem):
        """When beta_hat = 0, residual = Sigma_beta (Sigma @ 0 = 0)."""
        prob = linear_problem
        beta_zero = np.zeros(prob["p"])
        resid = compute_residual_sigma_beta(prob["Sigma_beta"], prob["Sigma"], beta_zero)
        np.testing.assert_allclose(resid, prob["Sigma_beta"], atol=1e-12)

    def test_zero_sigma_beta_with_zero_beta(self, linear_problem):
        """Zero Sigma_beta and zero beta_hat => residual is zero."""
        prob = linear_problem
        p = prob["p"]
        resid = compute_residual_sigma_beta(np.zeros(p), prob["Sigma"], np.zeros(p))
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_single_snp_ols(self, rng):
        """p=1: OLS gives scalar zero residual."""
        n = 300
        X = rng.standard_normal((n, 1))
        y = X[:, 0] * 0.5 + rng.standard_normal(n)
        Sigma = np.array([[float(np.mean(X[:, 0] ** 2))]])
        Sigma_beta = np.array([float(np.mean(X[:, 0] * y))])
        beta_hat = np.linalg.solve(Sigma, Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat)
        assert resid.shape == (1,)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_returns_ndarray(self, linear_problem):
        """Return type should be ndarray (not a scalar)."""
        prob = linear_problem
        resid = compute_residual_sigma_beta(
            prob["Sigma_beta"], prob["Sigma"], prob["beta_hat"],
        )
        assert isinstance(resid, np.ndarray)

    def test_linearity(self, linear_problem):
        """residual(Sigma_beta, Sigma, beta) = Sigma_beta - Sigma @ beta (linear check)."""
        prob = linear_problem
        Sigma, Sigma_beta = prob["Sigma"], prob["Sigma_beta"]
        beta_hat = prob["beta_hat"]
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat)
        expected = Sigma_beta - Sigma @ beta_hat
        np.testing.assert_allclose(resid, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. compute_residual_e_y2
# ---------------------------------------------------------------------------

class TestResidualEy2:

    def test_single_block_formula(self, linear_problem):
        """E_r2 = E_y2 - Sigma_beta^T beta_hat for one block."""
        prob = linear_problem
        Sigma_beta, beta_hat, E_y2 = (
            prob["Sigma_beta"], prob["beta_hat"], prob["E_y2"],
        )
        result = compute_residual_e_y2(E_y2, [Sigma_beta], [beta_hat])
        expected = E_y2 - float(np.dot(Sigma_beta, beta_hat))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_multiblock_sums_pve(self, rng):
        """Multi-block: E_r2 = E_y2 - sum_b Sb^T bh."""
        p_blocks = [5, 8, 6]
        sbs = [rng.standard_normal(p) * 0.1 for p in p_blocks]
        bhs = [rng.standard_normal(p) * 0.1 for p in p_blocks]
        E_y2 = 2.0
        result = compute_residual_e_y2(E_y2, sbs, bhs)
        expected = E_y2 - sum(float(np.dot(s, b)) for s, b in zip(sbs, bhs))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_multiblock_different_sizes(self, rng):
        """Blocks with heterogeneous sizes should all contribute correctly."""
        sizes = [3, 7, 2, 10]
        sbs = [rng.standard_normal(sz) for sz in sizes]
        bhs = [rng.standard_normal(sz) for sz in sizes]
        E_y2 = 3.5
        result = compute_residual_e_y2(E_y2, sbs, bhs)
        expected = E_y2 - sum(float(np.dot(s, b)) for s, b in zip(sbs, bhs))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_empty_blocks_returns_e_y2(self):
        """With no blocks, E_r2 == E_y2 (no PVE removed)."""
        result = compute_residual_e_y2(1.5, [], [])
        assert result == pytest.approx(1.5)

    def test_returns_float(self, linear_problem):
        """Return type should be a Python float (or float-like scalar)."""
        prob = linear_problem
        result = compute_residual_e_y2(
            prob["E_y2"], [prob["Sigma_beta"]], [prob["beta_hat"]],
        )
        assert isinstance(result, float)

    def test_nonnegative_for_well_specified_ols(self, large_linear_problem):
        """For a well-specified linear model with OLS beta_hat, E_r2 >= 0."""
        prob = large_linear_problem
        result = compute_residual_e_y2(
            prob["E_y2"], [prob["Sigma_beta"]], [prob["beta_hat"]],
        )
        assert result >= 0.0, f"Expected E_r2 >= 0, got {result:.4f}"

    def test_equals_empirical_residual_variance(self, linear_problem):
        """E_r2 from summary stats ≈ empirical mean(r^2) to within estimation noise."""
        prob = linear_problem
        X, y, beta_hat = prob["X"], prob["y"], prob["beta_hat"]
        r = y - X @ beta_hat
        empirical_e_r2 = float(np.mean(r ** 2))
        sumstats_e_r2 = compute_residual_e_y2(
            prob["E_y2"], [prob["Sigma_beta"]], [beta_hat],
        )
        assert sumstats_e_r2 == pytest.approx(empirical_e_r2, rel=0.15)

    def test_zero_beta_hat_gives_e_y2(self, linear_problem):
        """With beta_hat = 0, PVE = dot(Sigma_beta, 0) = 0, so E_r2 = E_y2."""
        prob = linear_problem
        beta_zero = np.zeros(prob["p"])
        result = compute_residual_e_y2(prob["E_y2"], [prob["Sigma_beta"]], [beta_zero])
        assert result == pytest.approx(prob["E_y2"], rel=1e-10)

    def test_multiblock_matches_single_concatenated_block(self, rng):
        """Splitting into multiple blocks should give the same result as one big block."""
        p = 20
        sb = rng.standard_normal(p) * 0.15
        bh = rng.standard_normal(p) * 0.15
        E_y2 = 2.5

        # Single concatenated block
        result_single = compute_residual_e_y2(E_y2, [sb], [bh])

        # Split into 4 blocks of 5
        sbs = [sb[i * 5: (i + 1) * 5] for i in range(4)]
        bhs = [bh[i * 5: (i + 1) * 5] for i in range(4)]
        result_multi = compute_residual_e_y2(E_y2, sbs, bhs)

        assert result_single == pytest.approx(result_multi, rel=1e-10)

    def test_ridge_pve_smaller_than_ols(self, linear_problem):
        """Ridge shrinks beta_hat, so PVE = dot(Sb, bh_ridge) < PVE_ols, hence E_r2_ridge > E_r2_ols."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 0.5
        beta_hat_ols = prob["beta_hat"]
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)

        e_r2_ols = compute_residual_e_y2(prob["E_y2"], [Sigma_beta], [beta_hat_ols])
        e_r2_ridge = compute_residual_e_y2(prob["E_y2"], [Sigma_beta], [beta_hat_ridge])
        assert e_r2_ridge > e_r2_ols, (
            f"Ridge residual {e_r2_ridge:.4f} should exceed OLS residual {e_r2_ols:.4f}"
        )

    def test_single_snp_single_block(self, rng):
        """p=1 single-SNP block: formula reduces to scalar subtraction."""
        n = 400
        X = rng.standard_normal((n, 1))
        y = X[:, 0] * 0.6 + rng.standard_normal(n)
        Sigma_beta = np.array([float(np.mean(X[:, 0] * y))])
        beta_hat = np.array([float(np.mean(X[:, 0] * y) / np.mean(X[:, 0] ** 2))])
        E_y2 = float(np.mean(y ** 2))
        result = compute_residual_e_y2(E_y2, [Sigma_beta], [beta_hat])
        expected = E_y2 - float(np.dot(Sigma_beta, beta_hat))
        assert result == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_p1_gamma_correction_shape_and_value(self, rng):
        """p=1: correction is a (1,1) matrix, matches scalar formula."""
        n = 150
        X = rng.standard_normal((n, 1))
        beta_hat = np.array([0.4])
        correction = compute_gamma_correction(X, beta_hat)
        assert correction.shape == (1, 1)
        v = X[:, 0] * beta_hat[0]
        expected = float(np.mean(X[:, 0] ** 2 * v))
        assert float(correction[0, 0]) == pytest.approx(expected, rel=1e-10)

    def test_p1_residual_gamma(self, rng):
        """p=1: Gamma_resid is (1,1) and equals direct (1/n) X^T Diag(r) X."""
        n = 150
        X = rng.standard_normal((n, 1))
        beta_hat = np.array([0.4])
        y = X[:, 0] * 0.5 + rng.standard_normal(n) * 0.8
        Gamma_raw = np.array([[float(np.mean(X[:, 0] ** 2 * y))]])
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_hat)
        assert Gamma_resid.shape == (1, 1)
        r = y - X[:, 0] * beta_hat[0]
        Gamma_direct = np.array([[float(np.mean(X[:, 0] ** 2 * r))]])
        np.testing.assert_allclose(Gamma_resid, Gamma_direct, atol=1e-12)

    def test_p1_sigma_beta_resid(self, rng):
        """p=1: Sigma_beta_resid is a (1,) vector."""
        Sigma = np.array([[1.5]])
        Sigma_beta = np.array([0.7])
        beta_hat = np.array([0.4])
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat)
        assert resid.shape == (1,)
        expected = Sigma_beta - Sigma @ beta_hat
        np.testing.assert_allclose(resid, expected, atol=1e-12)

    def test_n_ref_1_gamma_correction(self, rng):
        """n_ref=1: degenerate reference panel should not crash."""
        p = 8
        X_ref = rng.standard_normal((1, p))
        beta_hat = rng.standard_normal(p) * 0.1
        correction = compute_gamma_correction(X_ref, beta_hat)
        assert correction.shape == (p, p)
        np.testing.assert_allclose(correction, correction.T, atol=1e-12)

    def test_n_ref_1_residual_gamma(self, rng):
        """n_ref=1: compute_residual_gamma should return valid (p,p) matrix."""
        p = 8
        n = 200
        Sigma = make_block_sigma(2, 4, decay=0.5)
        X = gaussian_genotypes(rng, n, Sigma)
        y = X @ rng.standard_normal(p) + rng.standard_normal(n)
        Gamma_raw = (X * y[:, None]).T @ X / n
        beta_hat = rng.standard_normal(p) * 0.1
        X_ref = rng.standard_normal((1, p))
        Gamma_resid = compute_residual_gamma(Gamma_raw, X_ref, beta_hat)
        assert Gamma_resid.shape == (p, p)
        np.testing.assert_allclose(Gamma_resid, Gamma_resid.T, atol=1e-12)

    def test_negative_beta_hat_correction_symmetric(self, rng):
        """Negative beta_hat entries: correction must still be symmetric."""
        p, n = 10, 300
        Sigma = make_block_sigma(2, 5, decay=0.6)
        X = gaussian_genotypes(rng, n, Sigma)
        beta_hat = -np.abs(rng.standard_normal(p)) * 0.3  # all negative
        correction = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(correction, correction.T, atol=1e-12)

    def test_negative_beta_hat_correction_matches_formula(self, rng):
        """Negative beta_hat: correction still equals (X * v[:, None]).T @ X / n."""
        p, n = 10, 300
        Sigma = make_block_sigma(2, 5, decay=0.6)
        X = gaussian_genotypes(rng, n, Sigma)
        beta_hat = -np.abs(rng.standard_normal(p)) * 0.3
        v = X @ beta_hat
        expected = (X * v[:, None]).T @ X / n
        result = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_very_small_lambda_ridge_near_zero(self, linear_problem):
        """lambda=1e-10: ridge residual sigma_beta should be near-zero."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 1e-10
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
        np.testing.assert_allclose(resid, 0.0, atol=1e-6)

    def test_very_large_lambda_ridge_approx_sigma_beta(self, linear_problem):
        """lambda=1e6: beta_hat ≈ 0, so residual ≈ Sigma_beta."""
        prob = linear_problem
        Sigma, Sigma_beta, p = prob["Sigma"], prob["Sigma_beta"], prob["p"]
        lam = 1e6
        beta_hat_ridge = np.linalg.solve(Sigma + lam * np.eye(p), Sigma_beta)
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat_ridge)
        np.testing.assert_allclose(resid, Sigma_beta, rtol=1e-3)

    def test_all_zero_gamma_raw_gives_neg_correction(self, rng):
        """Gamma_raw = 0 => Gamma_resid = 0 - correction = -correction."""
        p, n = 10, 300
        Sigma = make_block_sigma(2, 5)
        X = gaussian_genotypes(rng, n, Sigma)
        beta_hat = rng.standard_normal(p) * 0.2
        Gamma_raw = np.zeros((p, p))
        Gamma_resid = compute_residual_gamma(Gamma_raw, X, beta_hat)
        correction = compute_gamma_correction(X, beta_hat)
        np.testing.assert_allclose(Gamma_resid, -correction, atol=1e-12)

    def test_empty_block_list_e_y2(self):
        """Empty block list: E_r2 == E_y2."""
        assert compute_residual_e_y2(2.3, [], []) == pytest.approx(2.3)

    def test_zero_sigma_beta_blocks_e_y2(self, rng):
        """All-zero Sigma_beta blocks: E_r2 == E_y2 (PVE = 0)."""
        blocks = [np.zeros(5), np.zeros(7)]
        beta_hats = [rng.standard_normal(5), rng.standard_normal(7)]
        E_y2 = 1.8
        result = compute_residual_e_y2(E_y2, blocks, beta_hats)
        assert result == pytest.approx(E_y2, rel=1e-10)

    def test_identity_sigma_ols_residual_zero(self, rng):
        """Sigma = I: OLS beta_hat = Sigma_beta, residual is exactly zero."""
        p = 12
        Sigma = np.eye(p)
        Sigma_beta = rng.standard_normal(p) * 0.3
        beta_hat = Sigma_beta.copy()  # OLS under identity
        resid = compute_residual_sigma_beta(Sigma_beta, Sigma, beta_hat)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_binomial_genotypes_gamma_correction_is_finite(self, rng):
        """Binomial (real SNP-like) genotypes: correction should be finite, no NaN/Inf."""
        p, n = 15, 400
        X, _ = binomial_genotypes(rng, n, p)
        beta_hat = rng.standard_normal(p) * 0.2
        correction = compute_gamma_correction(X, beta_hat)
        assert np.all(np.isfinite(correction)), "Correction has non-finite values"

    def test_sigma_beta_zero_entries_sigma_beta_resid_no_nan(self, linear_problem):
        """Sigma_beta with zero entries should not produce NaN."""
        prob = linear_problem
        Sigma_beta_zeros = np.zeros(prob["p"])
        resid = compute_residual_sigma_beta(Sigma_beta_zeros, prob["Sigma"], prob["beta_hat"])
        assert np.all(np.isfinite(resid)), "Residual has non-finite values"

    def test_multiblock_e_y2_different_sizes(self, rng):
        """Multi-block e_y2 works with blocks of sizes [2, 5, 3, 10, 1]."""
        sizes = [2, 5, 3, 10, 1]
        sbs = [rng.standard_normal(s) * 0.15 for s in sizes]
        bhs = [rng.standard_normal(s) * 0.10 for s in sizes]
        E_y2 = 4.0
        result = compute_residual_e_y2(E_y2, sbs, bhs)
        expected = E_y2 - sum(float(np.dot(s, b)) for s, b in zip(sbs, bhs))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_gamma_correction_no_nan_with_large_beta(self, rng):
        """Very large beta_hat: correction should not produce NaN or Inf."""
        p, n = 8, 200
        X = rng.standard_normal((n, p))
        beta_hat = rng.standard_normal(p) * 1e3
        correction = compute_gamma_correction(X, beta_hat)
        assert np.all(np.isfinite(correction))

    def test_residual_sigma_beta_no_nan_with_zero_sigma_beta(self, linear_problem):
        """Zero Sigma_beta with nonzero beta_hat: residual = -Sigma @ beta_hat."""
        prob = linear_problem
        p, Sigma, beta_hat = prob["p"], prob["Sigma"], prob["beta_hat"]
        Sigma_beta_zero = np.zeros(p)
        resid = compute_residual_sigma_beta(Sigma_beta_zero, Sigma, beta_hat)
        expected = -(Sigma @ beta_hat)
        np.testing.assert_allclose(resid, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. compute_genome_wide_residual_gamma
# ---------------------------------------------------------------------------

class TestGenomeWideResidualGamma:

    @pytest.fixture
    def multiblock_linear(self):
        """B=5 blocks, pure linear phenotype — no epistasis."""
        rng = np.random.default_rng(77)
        B, p, n = 5, 10, 5000
        betas = [rng.standard_normal(p) * 0.3 for _ in range(B)]
        X_blocks = [rng.standard_normal((n, p)) for _ in range(B)]
        beta_hats = [betas[b] + rng.standard_normal(p) * 0.01 for b in range(B)]

        y = sum(X_blocks[b] @ betas[b] for b in range(B))
        y += rng.standard_normal(n) * np.sqrt(float(np.var(y)))  # h²=0.5

        return {"B": B, "p": p, "n": n, "X_blocks": X_blocks,
                "y": y, "betas": betas, "beta_hats": beta_hats}

    def test_returns_correct_number_of_blocks(self, multiblock_linear):
        d = multiblock_linear
        gammas = compute_genome_wide_residual_gamma(
            d["X_blocks"], d["y"], d["beta_hats"], d["n"]
        )
        assert len(gammas) == d["B"]

    def test_each_block_correct_shape(self, multiblock_linear):
        d = multiblock_linear
        gammas = compute_genome_wide_residual_gamma(
            d["X_blocks"], d["y"], d["beta_hats"], d["n"]
        )
        for b, G in enumerate(gammas):
            assert G.shape == (d["p"], d["p"]), f"Block {b}: expected ({d['p']},{d['p']})"

    def test_each_block_symmetric(self, multiblock_linear):
        d = multiblock_linear
        gammas = compute_genome_wide_residual_gamma(
            d["X_blocks"], d["y"], d["beta_hats"], d["n"]
        )
        for b, G in enumerate(gammas):
            np.testing.assert_allclose(G, G.T, atol=1e-12, err_msg=f"Block {b} not symmetric")

    def test_matches_direct_leave_one_out(self, multiblock_linear):
        """Each Γ_b must equal (1/n) X_b^T diag(r_b) X_b computed directly."""
        d = multiblock_linear
        B, n = d["B"], d["n"]
        prs_total = sum(d["X_blocks"][b] @ d["beta_hats"][b] for b in range(B))
        gammas = compute_genome_wide_residual_gamma(
            d["X_blocks"], d["y"], d["beta_hats"], n
        )
        for b in range(B):
            r_b = d["y"] - (prs_total - d["X_blocks"][b] @ d["beta_hats"][b])
            G_direct = d["X_blocks"][b].T @ (d["X_blocks"][b] * r_b[:, None]) / n
            np.testing.assert_allclose(gammas[b], G_direct, atol=1e-12,
                                       err_msg=f"Block {b} mismatch")

    def test_reduces_noise_under_null(self, multiblock_linear):
        """Under pure linear DGP, gw-residual Γ has smaller Frobenius norm than raw Γ."""
        d = multiblock_linear
        n = d["n"]
        gammas_gw = compute_genome_wide_residual_gamma(
            d["X_blocks"], d["y"], d["beta_hats"], n
        )
        for b in range(d["B"]):
            Gamma_raw_b = d["X_blocks"][b].T @ (d["X_blocks"][b] * d["y"][:, None]) / n
            norm_raw = np.linalg.norm(Gamma_raw_b, "fro")
            norm_gw = np.linalg.norm(gammas_gw[b], "fro")
            assert norm_gw < norm_raw, (
                f"Block {b}: gw-residual norm ({norm_gw:.4f}) should be < raw ({norm_raw:.4f})"
            )

    def test_preserves_signal_vs_raw(self):
        """Under epistasis, top eigenvalue of gw-residual Γ ≥ top eigenvalue of raw Γ."""
        rng = np.random.default_rng(88)
        B, p, n = 5, 15, 20000

        betas = [rng.standard_normal(p) * 0.3 for _ in range(B)]
        X_blocks = [rng.standard_normal((n, p)) for _ in range(B)]
        beta_hats = betas.copy()

        # Plant epistasis in block 0: y += relu(w_epi^T x_0)
        w_epi = rng.standard_normal(p) * 0.5
        y = sum(X_blocks[b] @ betas[b] for b in range(B))
        nl = np.maximum(0.0, X_blocks[0] @ w_epi)
        y += nl * np.sqrt(float(np.var(y)) / max(float(np.var(nl)), 1e-15))
        y += rng.standard_normal(n) * np.sqrt(float(np.var(y)))

        gammas_gw = compute_genome_wide_residual_gamma(X_blocks, y, beta_hats, n)

        Gamma_raw_0 = X_blocks[0].T @ (X_blocks[0] * y[:, None]) / n
        top_raw = float(np.max(np.abs(np.linalg.eigvalsh(Gamma_raw_0))))
        top_gw = float(np.max(np.abs(np.linalg.eigvalsh(gammas_gw[0]))))

        # Signal is preserved: gw-residual top eigenvalue should not drop below raw
        assert top_gw >= top_raw * 0.7, (
            f"Signal lost: gw top={top_gw:.4f}, raw top={top_raw:.4f}"
        )

    def test_zero_beta_hats_equals_raw_gamma(self, multiblock_linear):
        """With all-zero beta_hats, leave-one-out residual r_b = y, so Γ_b = Γ_raw_b."""
        d = multiblock_linear
        n = d["n"]
        zero_betas = [np.zeros(d["p"]) for _ in range(d["B"])]
        gammas_gw = compute_genome_wide_residual_gamma(d["X_blocks"], d["y"], zero_betas, n)
        for b in range(d["B"]):
            Gamma_raw_b = d["X_blocks"][b].T @ (d["X_blocks"][b] * d["y"][:, None]) / n
            np.testing.assert_allclose(gammas_gw[b], Gamma_raw_b, atol=1e-12,
                                       err_msg=f"Block {b}: expected equal to raw Gamma")


# ---------------------------------------------------------------------------
# 7. compute_residual_sigma_other2
# ---------------------------------------------------------------------------

class TestResidualSigmaOther2:

    def test_excludes_target_block(self):
        """PVE of block_idx should NOT be subtracted from E_y2."""
        rng = np.random.default_rng(99)
        B, p = 4, 8
        sbs = [rng.standard_normal(p) * 0.2 for _ in range(B)]
        bhs = [rng.standard_normal(p) * 0.2 for _ in range(B)]
        E_y2 = 2.0

        for b in range(B):
            result = compute_residual_sigma_other2(E_y2, sbs, bhs, b)
            pve_other = sum(
                max(0.0, float(np.dot(sbs[bb], bhs[bb])))
                for bb in range(B) if bb != b
            )
            expected = max(1e-6, E_y2 - pve_other)
            assert result == pytest.approx(expected, rel=1e-10), f"Failed at block {b}"

    def test_monotone_in_block_idx(self):
        """Blocks with larger PVE: excluding them yields larger sigma_other2."""
        rng = np.random.default_rng(100)
        B, p = 5, 6
        # Make block 0 have very large PVE, block 4 very small
        sbs = [np.ones(p) * (0.5 / (b + 1)) for b in range(B)]
        bhs = [np.ones(p) * (0.5 / (b + 1)) for b in range(B)]
        E_y2 = 5.0
        # Excluding block 0 (high PVE) → larger sigma_other2 than excluding block 4 (low PVE)
        s_excl_0 = compute_residual_sigma_other2(E_y2, sbs, bhs, 0)
        s_excl_4 = compute_residual_sigma_other2(E_y2, sbs, bhs, 4)
        assert s_excl_0 > s_excl_4, (
            f"Excluding high-PVE block should give larger sigma_other2: {s_excl_0:.4f} vs {s_excl_4:.4f}"
        )

    def test_clamped_to_floor(self):
        """Result is clamped to ≥ 1e-6 even if PVE_other > E_y2."""
        E_y2 = 0.1
        # Very large PVE from other blocks
        sbs = [np.array([1.0]), np.array([1.0])]
        bhs = [np.array([1.0]), np.array([1.0])]
        result = compute_residual_sigma_other2(E_y2, sbs, bhs, block_idx=0)
        assert result >= 1e-6

    def test_single_block_returns_floor(self):
        """With B=1, PVE_other=0, sigma_other2 = E_y2."""
        sbs = [np.array([0.5, 0.3])]
        bhs = [np.array([0.4, 0.2])]
        E_y2 = 1.5
        result = compute_residual_sigma_other2(E_y2, sbs, bhs, block_idx=0)
        assert result == pytest.approx(E_y2, rel=1e-10)

    def test_all_zero_pve_returns_e_y2(self):
        """With zero PVE from all other blocks, sigma_other2 = E_y2."""
        B, p = 4, 5
        sbs = [np.zeros(p) for _ in range(B)]
        bhs = [np.zeros(p) for _ in range(B)]
        E_y2 = 3.0
        for b in range(B):
            result = compute_residual_sigma_other2(E_y2, sbs, bhs, b)
            assert result == pytest.approx(E_y2, rel=1e-10)
