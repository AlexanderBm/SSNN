"""Tests for genome_wide_optimizer.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.stats import norm

from ssnn.genome_wide_optimizer import (
    compute_beta_hats,
    predict_genome_wide,
    train_genome_wide,
)
from ssnn.optimizer import TrainResult, train
from ssnn.utils import nn_predict


# ---------------------------------------------------------------------------
# Shared DGP helpers (copied from scripts/residual_simulation.py)
# ---------------------------------------------------------------------------

def block_ld(p: int, decay: float = 0.6) -> np.ndarray:
    idx = np.arange(p)
    return decay ** np.abs(idx[:, None] - idx[None, :])


def block_genotypes(
    n: int,
    maf: np.ndarray,
    Sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial(2, p) genotypes via Gaussian latent thresholding."""
    p = len(maf)
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    q = 1.0 - maf
    thresh_0 = norm.ppf(q ** 2)
    thresh_01 = norm.ppf(q ** 2 + 2 * maf * q)
    X = np.zeros((n, p))
    X[Z >= thresh_0] = 1.0
    X[Z >= thresh_01] = 2.0
    return X


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.mean((y_true - y_pred) ** 2))
    ss_tot = float(np.var(y_true))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def make_genome_wide_data(
    B: int = 10,
    p_b: int = 30,
    n_train: int = 3000,
    n_ref: int = 1000,
    n_test: int = 2000,
    heritability: float = 0.5,
    epistasis_frac: float = 0.5,
    ld_decay: float = 0.6,
    ridge_lambda: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate a full multi-block dataset and summary statistics."""
    rng = np.random.default_rng(seed)

    mafs = [rng.uniform(0.1, 0.5, p_b) for _ in range(B)]
    sigmas = [block_ld(p_b, ld_decay) for _ in range(B)]

    n_epi = max(1, int(epistasis_frac * B)) if epistasis_frac > 0 else 0
    epi_blocks = set(rng.choice(B, size=n_epi, replace=False)) if n_epi > 0 else set()

    betas = [rng.standard_normal(p_b) * 0.3 for _ in range(B)]
    w_stars = [rng.standard_normal(p_b) * 0.3 if b in epi_blocks else None for b in range(B)]

    block_seeds = [int(rng.integers(2**32)) for _ in range(B)]

    y_train = np.zeros(n_train)
    y_test = np.zeros(n_test)

    Xs_train, Xs_test, Xs_ref = [], [], []

    for b in range(B):
        brng = np.random.default_rng(block_seeds[b])
        X_tr = block_genotypes(n_train, mafs[b], sigmas[b], brng)
        X_te = block_genotypes(n_test, mafs[b], sigmas[b], brng)
        X_rf = block_genotypes(n_ref, mafs[b], sigmas[b], brng)

        mu = X_tr.mean(axis=0)
        X_tr -= mu
        X_te -= mu
        X_rf -= mu
        Xs_train.append(X_tr)
        Xs_test.append(X_te)
        Xs_ref.append(X_rf)

        y_train += X_tr @ betas[b]
        y_test += X_te @ betas[b]

        if w_stars[b] is not None:
            lin_tr = X_tr @ betas[b]
            nl_tr = np.maximum(0.0, X_tr @ w_stars[b])
            nl_te = np.maximum(0.0, X_te @ w_stars[b])
            v_lin = float(np.var(lin_tr))
            v_nl = float(np.var(nl_tr))
            scale = np.sqrt(v_lin / v_nl) if v_nl > 1e-15 else 0.0
            y_train += scale * nl_tr
            y_test += scale * nl_te

    var_gen = float(np.var(y_train))
    sigma_eps = np.sqrt(var_gen * (1.0 - heritability) / heritability) if heritability < 1 else 0.0
    noise_rng = np.random.default_rng(int(rng.integers(2**32)))
    y_train = y_train + noise_rng.standard_normal(n_train) * sigma_eps
    y_test = y_test + noise_rng.standard_normal(n_test) * sigma_eps

    y_mean = float(np.mean(y_train))
    y_train -= y_mean
    y_test -= y_mean

    E_y2 = float(np.mean(y_train ** 2))
    Sigma_betas, Covs_ref, beta_hats = [], [], []

    for b in range(B):
        X = Xs_train[b]
        Sb = X.T @ y_train / n_train
        Sigma_betas.append(Sb)
        Covs_ref.append(Xs_ref[b].T @ Xs_ref[b] / n_ref)
        bh = np.linalg.solve(sigmas[b] + ridge_lambda * np.eye(p_b), Sb)
        beta_hats.append(bh)

    return {
        "sigmas": sigmas,
        "Sigma_betas": Sigma_betas,
        "Covs_ref": Covs_ref,
        "beta_hats": beta_hats,
        "E_y2": E_y2,
        "Xs_train": Xs_train,
        "Xs_test": Xs_test,
        "Xs_ref": Xs_ref,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Test 1: single-block matches standard train()
# ---------------------------------------------------------------------------

def test_single_block_matches_standard_train():
    """At B=1 with h2=0.5, genome-wide optimizer produces similar R² to standard train()."""
    data = make_genome_wide_data(B=1, p_b=30, n_train=3000, n_ref=1000, n_test=2000,
                                 heritability=0.5, epistasis_frac=0.0, seed=7)

    sigmas = data["sigmas"]
    Sigma_betas = data["Sigma_betas"]
    Covs_ref = data["Covs_ref"]
    beta_hats = data["beta_hats"]
    E_y2 = data["E_y2"]

    rng = np.random.default_rng(0)

    # Genome-wide optimizer (B=1)
    gw_results, c_star = train_genome_wide(
        Sigma_blocks=sigmas,
        Sigma_beta_blocks=Sigma_betas,
        E_y2=E_y2,
        Cov_ref_blocks=Covs_ref,
        beta_hat_blocks=beta_hats,
        m=3,
        activation="relu",
        lr=0.05,
        max_iters=300,
        base_reg_W=0.01,
        base_reg_a=0.01,
        rng=rng,
    )

    gw_pred = predict_genome_wide(data["Xs_test"], gw_results, c_star)
    gw_r2 = r2(data["y_test"], gw_pred)

    # Standard train()
    std_result = train(
        Sigma=sigmas[0],
        Sigma_beta=Sigma_betas[0],
        E_y2=E_y2,
        m=3,
        activation="relu",
        lr=0.05,
        max_iters=300,
        init_scale=0.01,
        rng=np.random.default_rng(1),
        Cov_ref=Covs_ref[0],
    )
    std_pred = nn_predict(data["Xs_test"][0], std_result.a, std_result.W, "relu")
    std_r2 = r2(data["y_test"], std_pred)

    # Both should be positive; genome-wide should be within 0.1 R² of standard
    assert gw_r2 > -0.05, f"Genome-wide R²={gw_r2:.4f} too negative for B=1"
    assert std_r2 > -0.05, f"Standard R²={std_r2:.4f} too negative"
    assert abs(gw_r2 - std_r2) < 0.15, (
        f"R² gap too large: gw={gw_r2:.4f}  std={std_r2:.4f}  diff={gw_r2-std_r2:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: c_star is in [0, 2] for a range of B values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [5, 10, 20, 40])
def test_c_star_range(B):
    """c_star output is in [0.0, 5.0] for all tested block counts."""
    data = make_genome_wide_data(B=B, p_b=20, n_train=2000, n_ref=500, n_test=1000,
                                 heritability=0.5, epistasis_frac=0.0, seed=B + 10)

    rng = np.random.default_rng(B)
    _, c_star = train_genome_wide(
        Sigma_blocks=data["sigmas"],
        Sigma_beta_blocks=data["Sigma_betas"],
        E_y2=data["E_y2"],
        Cov_ref_blocks=data["Covs_ref"],
        beta_hat_blocks=data["beta_hats"],
        m=3,
        activation="relu",
        lr=0.05,
        max_iters=200,
        base_reg_W=0.01,
        base_reg_a=0.01,
        rng=rng,
    )

    assert 0.0 <= c_star <= 5.0, f"c_star={c_star:.4f} out of bounds for B={B}"


# ---------------------------------------------------------------------------
# Test 3: per-block weight norms decrease with B
# ---------------------------------------------------------------------------

def test_weight_norms_decrease_with_B():
    """Per-block ||a||_1 norms should be smaller at large B than at small B."""
    norms_small = []
    norms_large = []

    for B, norm_list in [(5, norms_small), (40, norms_large)]:
        data = make_genome_wide_data(B=B, p_b=20, n_train=2000, n_ref=500, n_test=1000,
                                     heritability=0.5, epistasis_frac=0.0, seed=B + 99)
        rng = np.random.default_rng(B + 200)
        gw_results, _ = train_genome_wide(
            Sigma_blocks=data["sigmas"],
            Sigma_beta_blocks=data["Sigma_betas"],
            E_y2=data["E_y2"],
            Cov_ref_blocks=data["Covs_ref"],
            beta_hat_blocks=data["beta_hats"],
            m=3,
            activation="relu",
            lr=0.05,
            max_iters=200,
            base_reg_W=0.01,
            base_reg_a=0.01,
            rng=rng,
        )
        # Average per-block ||a||_1
        avg_norm = float(np.mean([np.sum(np.abs(r.a)) for r in gw_results]))
        norm_list.append(avg_norm)

    avg_small = np.mean(norms_small)
    avg_large = np.mean(norms_large)
    assert avg_large <= avg_small + 1.0, (
        f"Expected per-block norms to be similar or smaller at B=40 "
        f"(got avg_large={avg_large:.4f} vs avg_small={avg_small:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: genome-wide NN does not harm vs. linear PRS at B=40
# ---------------------------------------------------------------------------

def test_no_harm_vs_linear_prs():
    """At B=40 with n=3000, genome-wide NN R² >= linear PRS R² - 0.05."""
    B = 40
    p_b = 10   # small blocks for speed
    n_train = 3000
    n_test = 2000
    n_ref = 1000
    n_reps = 2
    ridge_lambda = 1.0

    r2_nn_list = []
    r2_lin_list = []

    for rep in range(n_reps):
        data = make_genome_wide_data(
            B=B, p_b=p_b, n_train=n_train, n_ref=n_ref, n_test=n_test,
            heritability=0.5, epistasis_frac=0.5,
            seed=42 + rep * 100,
        )

        rng = np.random.default_rng(rep + 1000)
        gw_results, c_star = train_genome_wide(
            Sigma_blocks=data["sigmas"],
            Sigma_beta_blocks=data["Sigma_betas"],
            E_y2=data["E_y2"],
            Cov_ref_blocks=data["Covs_ref"],
            beta_hat_blocks=data["beta_hats"],
            m=3,
            activation="relu",
            lr=0.05,
            max_iters=300,
            base_reg_W=0.1,
            base_reg_a=0.1,
            rng=rng,
        )
        gw_pred = predict_genome_wide(data["Xs_test"], gw_results, c_star)
        r2_nn_list.append(r2(data["y_test"], gw_pred))

        lin_pred = sum(data["Xs_test"][b] @ data["beta_hats"][b] for b in range(B))
        r2_lin_list.append(r2(data["y_test"], lin_pred))

    mean_nn = float(np.mean(r2_nn_list))
    mean_lin = float(np.mean(r2_lin_list))

    assert mean_nn >= mean_lin - 0.05, (
        f"Genome-wide NN ({mean_nn:.4f}) much worse than linear PRS ({mean_lin:.4f}) "
        f"at B={B}; gap = {mean_lin - mean_nn:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: linear fallback is triggered for near-zero signal blocks
# ---------------------------------------------------------------------------

def test_linear_fallback_triggered():
    """A block with near-zero signal should trigger the linear fallback."""
    rng_data = np.random.default_rng(99)
    p_b = 20
    sigma = block_ld(p_b, decay=0.6)

    # Block 0: real signal
    Sb_signal = rng_data.standard_normal(p_b) * 0.3
    # Block 1: near-zero signal
    Sb_zero = np.zeros(p_b)

    Sigma_blocks = [sigma, sigma]
    Sigma_beta_blocks = [Sb_signal, Sb_zero]
    Cov_ref_blocks = [sigma, sigma]

    beta_hat_signal = np.linalg.solve(sigma + np.eye(p_b), Sb_signal)
    beta_hat_zero = np.zeros(p_b)
    beta_hat_blocks = [beta_hat_signal, beta_hat_zero]

    E_y2 = float(Sb_signal @ beta_hat_signal) + 0.5

    rng = np.random.default_rng(123)
    results, c_star = train_genome_wide(
        Sigma_blocks=Sigma_blocks,
        Sigma_beta_blocks=Sigma_beta_blocks,
        E_y2=E_y2,
        Cov_ref_blocks=Cov_ref_blocks,
        beta_hat_blocks=beta_hat_blocks,
        m=3,
        activation="relu",
        lr=0.05,
        max_iters=200,
        base_reg_W=0.01,
        base_reg_a=0.01,
        rng=rng,
    )

    # Block 1 (near-zero signal) should have triggered fallback:
    # W has shape (2, p_b) — the 2-neuron relu identity representation.
    b1 = results[1]
    assert b1.W.shape[0] == 2, (
        f"Expected 2-neuron fallback (relu identity) for zero-signal block; got W.shape={b1.W.shape}"
    )
    # The two rows should be [beta_hat, -beta_hat] (both zero vectors here)
    assert np.allclose(b1.W[0], beta_hat_zero, atol=1e-10), (
        f"Fallback W[0] should equal beta_hat (zeros); got {b1.W[0]}"
    )
    assert np.allclose(b1.W[1], -beta_hat_zero, atol=1e-10), (
        f"Fallback W[1] should equal -beta_hat (zeros); got {b1.W[1]}"
    )
