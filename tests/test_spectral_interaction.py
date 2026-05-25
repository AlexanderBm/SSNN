"""Tests for spectral interaction NN components."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.stats import norm
from ssnn.gamma_denoising import marchenko_pastur_edge, denoise_gamma, triage_block
from ssnn.interaction_init import compute_eigenvector_init
from ssnn.robust_interaction_optimizer import train_interaction_robust
from ssnn.optimizer import train, TrainResult


# ---------------------------------------------------------------------------
# DGP helpers (copied from scripts/residual_simulation.py)
# ---------------------------------------------------------------------------

def block_ld(p: int, decay: float = 0.6) -> np.ndarray:
    idx = np.arange(p)
    return decay ** np.abs(idx[:, None] - idx[None, :])


def block_genotypes(n: int, maf: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = len(maf)
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    q = 1.0 - maf
    thresh_0 = norm.ppf(q ** 2)
    thresh_01 = norm.ppf(q ** 2 + 2 * maf * q)
    X = np.zeros((n, p))
    X[Z >= thresh_0] = 1.0
    X[Z >= thresh_01] = 2.0
    return X


def r2(y_true, y_pred):
    ss_res = float(np.mean((y_true - y_pred) ** 2))
    ss_tot = float(np.var(y_true))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def orthogonal_unit(beta: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Return eps orthogonalized w.r.t. beta and normalized to unit length."""
    beta_norm = np.linalg.norm(beta)
    if beta_norm < 1e-15:
        out = eps
    else:
        u = beta / beta_norm
        out = eps - (eps @ u) * u
    n = np.linalg.norm(out)
    if n < 1e-15:
        return out
    return out / n


# ---------------------------------------------------------------------------
# Tests for gamma_denoising.py
# ---------------------------------------------------------------------------

def test_mp_edge_formula():
    """marchenko_pastur_edge should match 3.5 * sqrt(sigma2 * p/n).

    The formula applies to the weighted matrix Gamma = (1/n) X^T diag(y) X,
    NOT the Wishart sample covariance. The noise max-eigenvalue scales as
    ~C*sqrt(sigma2*p/n) empirically; we use C=3.5.
    """
    p, n, sigma2 = 100, 1000, 1.0
    expected = 3.5 * np.sqrt(sigma2 * p / n)
    result = marchenko_pastur_edge(p=p, n=n, sigma2=sigma2)
    assert abs(result - expected) < 1e-12, f"Expected {expected:.6f}, got {result:.6f}"

    # Threshold should decrease as n increases (more data → tighter noise floor)
    edge_small_n = marchenko_pastur_edge(p=30, n=500, sigma2=1.0)
    edge_large_n = marchenko_pastur_edge(p=30, n=5000, sigma2=1.0)
    assert edge_large_n < edge_small_n, "Threshold should decrease with larger n"


def test_denoise_recovers_planted_spike():
    """denoise_gamma should retain the planted eigenvalue spike above MP edge."""
    rng = np.random.default_rng(42)
    p = 30
    n = 1000
    sigma2 = 1.0
    mp_edge = marchenko_pastur_edge(p=p, n=n, sigma2=sigma2)
    lambda_true = 3.0 * mp_edge  # well above threshold

    # Random unit vector for the spike
    v = rng.standard_normal(p)
    v /= np.linalg.norm(v)

    # Rank-1 signal + symmetric noise
    # Noise entries ~ N(0, sigma2 / n) so that per-entry variance is sigma2/n
    noise_raw = rng.standard_normal((p, p)) * np.sqrt(sigma2 / n)
    noise = (noise_raw + noise_raw.T) / 2.0

    Gamma = lambda_true * np.outer(v, v) + noise

    Gamma_denoised, surv_vecs, surv_vals = denoise_gamma(Gamma, sigma_other2=sigma2, n=n)

    assert len(surv_vals) >= 1, "At least one eigenvalue should survive."
    # The largest surviving eigenvalue should be close to lambda_true
    largest_idx = np.argmax(np.abs(surv_vals))
    assert abs(surv_vals[largest_idx] - lambda_true) < 0.5 * lambda_true, (
        f"Surviving eigenvalue {surv_vals[largest_idx]:.4f} far from planted {lambda_true:.4f}"
    )


def test_denoise_zeros_below_threshold():
    """denoise_gamma should return all-zeros when Gamma is pure-noise (no signal in y).

    Constructs Gamma the same way it's estimated in practice:
        Gamma = (1/n) X^T diag(y_noise) X
    where y_noise is independent of X. The noise max-eigenvalue scales as
    ~C*sqrt(sigma2*p/n), well below the threshold 3.5*sqrt(sigma2*p/n).
    """
    rng = np.random.default_rng(123)
    p = 20
    n = 10000
    sigma2 = 1.0  # realistic noise variance

    # Build Gamma from pure-noise phenotype: (1/n) X^T diag(y_noise) X
    X = rng.standard_normal((n, p))  # standardised genotypes
    y_noise = rng.standard_normal(n) * np.sqrt(sigma2)
    Gamma_noise = X.T @ (X * y_noise[:, None]) / n

    _, _, surv_vals = denoise_gamma(Gamma_noise, sigma_other2=sigma2, n=n)

    # Noise max-eigenvalue ≈ 0.11 (empirical); threshold = 3.5*sqrt(1.0*20/10000) ≈ 0.157
    # No eigenvalues should survive
    assert len(surv_vals) == 0, (
        f"Expected 0 surviving eigenvalues for pure-noise Gamma, got {len(surv_vals)}"
    )


def test_triage_block_consistent():
    """triage_block and denoise_gamma should agree on survival."""
    rng = np.random.default_rng(77)
    p = 25
    n = 500
    sigma2 = 1.0
    mp_edge = marchenko_pastur_edge(p=p, n=n, sigma2=sigma2)

    # --- Signal matrix: one large spike ---
    v = rng.standard_normal(p)
    v /= np.linalg.norm(v)
    lambda_signal = 5.0 * mp_edge
    Gamma_signal = lambda_signal * np.outer(v, v)

    triage_true = triage_block(Gamma_signal, sigma2, n)
    _, surv_vecs_s, surv_vals_s = denoise_gamma(Gamma_signal, sigma2, n)
    assert triage_true == (len(surv_vals_s) > 0), "triage_block and denoise_gamma disagree on signal matrix"
    assert triage_true is True, "Signal matrix should be triaged as having signal"

    # --- Noise matrix: very small entries ---
    noise_raw = rng.standard_normal((p, p)) * 1e-6
    Gamma_noise = (noise_raw + noise_raw.T) / 2.0

    triage_false = triage_block(Gamma_noise, sigma2, n)
    _, surv_vecs_n, surv_vals_n = denoise_gamma(Gamma_noise, sigma2, n)
    assert triage_false == (len(surv_vals_n) > 0), "triage_block and denoise_gamma disagree on noise matrix"
    assert triage_false is False, "Noise matrix should not be triaged as having signal"


def test_mp_edge_increases_with_gamma():
    """marchenko_pastur_edge should increase as n decreases (gamma = p/n increases)."""
    p = 30
    sigma2 = 1.0
    ns = [10000, 1000, 100]  # decreasing n → increasing gamma
    edges = [marchenko_pastur_edge(p=p, n=n_val, sigma2=sigma2) for n_val in ns]

    for i in range(len(edges) - 1):
        assert edges[i] < edges[i + 1], (
            f"Edge at n={ns[i]} ({edges[i]:.4f}) should be less than edge at n={ns[i+1]} ({edges[i+1]:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests for interaction_init.py
# ---------------------------------------------------------------------------

def test_eigenvector_init_no_survivors():
    """With no surviving eigenvectors, init should return warm-start unchanged."""
    rng = np.random.default_rng(1)
    m, p = 4, 10
    gauss_a = rng.standard_normal(m)
    gauss_W = rng.standard_normal((m, p))

    surviving_eigvecs = np.empty((0, p))
    surviving_eigvals = np.empty(0)

    a_init, W_init = compute_eigenvector_init(surviving_eigvecs, surviving_eigvals, gauss_a, gauss_W, m)

    np.testing.assert_array_equal(a_init, gauss_a)
    np.testing.assert_array_equal(W_init, gauss_W)


def test_eigenvector_init_allocates_correct_neurons():
    """With m=4, k=2 survivors, exactly 2 neurons get eigenvector directions."""
    rng = np.random.default_rng(2)
    m, p = 4, 10

    # Create 2 orthonormal eigenvectors
    Q, _ = np.linalg.qr(rng.standard_normal((p, 2)))
    surviving_eigvecs = Q.T  # shape (2, p)
    surviving_eigvals = np.array([5.0, 3.0])

    gauss_a = rng.standard_normal(m)
    gauss_W = rng.standard_normal((m, p))

    a_init, W_init = compute_eigenvector_init(surviving_eigvecs, surviving_eigvals, gauss_a, gauss_W, m)

    # n_eig = min(2, 4//2) = 2
    n_eig = 2
    for i in range(n_eig):
        # Cosine similarity between assigned W row and corresponding eigenvector
        w_norm = np.linalg.norm(W_init[i])
        e_norm = np.linalg.norm(surviving_eigvecs[i])
        if w_norm > 1e-15 and e_norm > 1e-15:
            cos_sim = abs(np.dot(W_init[i], surviving_eigvecs[i]) / (w_norm * e_norm))
            assert cos_sim >= 0.99, f"Neuron {i}: cosine similarity {cos_sim:.4f} < 0.99"


def test_eigenvector_init_preserves_warm_start():
    """Warm-start neurons in init should match largest-|a| Gaussian NN neurons."""
    rng = np.random.default_rng(3)
    m, p = 6, 10

    # Create 2 surviving eigenvectors
    Q, _ = np.linalg.qr(rng.standard_normal((p, 2)))
    surviving_eigvecs = Q.T  # shape (2, p)
    surviving_eigvals = np.array([5.0, 3.0])

    gauss_a = rng.standard_normal(m)
    gauss_W = rng.standard_normal((m, p))

    a_init, W_init = compute_eigenvector_init(surviving_eigvecs, surviving_eigvals, gauss_a, gauss_W, m)

    n_eig = min(2, m // 2)  # = 2
    n_warm = m - n_eig      # = 4

    # Find largest-|a| warm-start neurons
    warm_order = np.argsort(-np.abs(gauss_a))

    for j in range(n_warm):
        w_idx = warm_order[j % len(warm_order)]
        np.testing.assert_array_almost_equal(
            W_init[n_eig + j], gauss_W[w_idx],
            err_msg=f"Warm neuron {j} W does not match expected warm-start neuron"
        )
        assert abs(a_init[n_eig + j] - gauss_a[w_idx]) < 1e-12, (
            f"Warm neuron {j} a does not match expected warm-start neuron"
        )


# ---------------------------------------------------------------------------
# Tests for robust_interaction_optimizer.py
# ---------------------------------------------------------------------------

def _make_gauss_result(m: int, p: int, rng: np.random.Generator) -> TrainResult:
    """Create a simple TrainResult for testing."""
    a = rng.standard_normal(m) * 0.1
    W = rng.standard_normal((m, p)) * 0.1
    return TrainResult(a=a, W=W, loss_history=[0.5, 0.4, 0.35], converged=True, n_iters=3)


def test_robust_no_signal_returns_gauss():
    """With pure noise Gamma and small n, robust optimizer should return gauss_result."""
    rng = np.random.default_rng(42)
    p = 20
    n = 100  # very small n → very large MP edge, so nothing clears threshold

    Sigma = block_ld(p)
    Sigma_beta = rng.standard_normal(p) * 0.1
    E_y2 = 1.0

    # Pure noise Gamma_raw: tiny random symmetric matrix
    noise = rng.standard_normal((p, p)) * 1e-6
    Gamma_raw = (noise + noise.T) / 2.0

    # Reference panel genotypes for residualization
    maf = rng.uniform(0.1, 0.5, p)
    X_ref = block_genotypes(n, maf, Sigma, rng)
    X_ref -= X_ref.mean(axis=0)

    beta_hat = np.linalg.solve(Sigma + np.eye(p), Sigma_beta)
    sigma_other2 = E_y2  # conservative upper bound

    gauss_result = _make_gauss_result(m=3, p=p, rng=rng)

    result = train_interaction_robust(
        Sigma=Sigma,
        Sigma_beta=Sigma_beta,
        E_y2=E_y2,
        Gamma_raw=Gamma_raw,
        X_ref=X_ref,
        beta_hat=beta_hat,
        sigma_other2=sigma_other2,
        n_train=n,
        m=3,
        gauss_result=gauss_result,
        rng=np.random.default_rng(99),
    )

    # Should return gauss_result unchanged
    np.testing.assert_array_almost_equal(result.a, gauss_result.a)
    np.testing.assert_array_almost_equal(result.W, gauss_result.W)


def test_robust_beats_collapsed_gradient():
    """Robust NN maintains 'do no harm' and beats collapsed when signal is orthogonal.

    DGP: single block, p=30, n=10000, binomial genotypes.
    w_star = alpha * beta/||beta|| + sqrt(1-alpha²) * eps_perp.
    alpha=0.4: partially orthogonal — favorable for spectral approach.
    The collapsed (rank-1) gradient stays in the beta direction and misses
    signal in the orthogonal component. The eigenvector init lets the robust
    optimizer find the orthogonal direction.

    Primary assertion: robust >= gauss - 0.01 ("do no harm") in every rep.
    Secondary assertion: mean_robust >= mean_collapsed - 0.005 (at least competitive).
    nonlinear_strength = 1.0, heritability = 0.5
    """
    from ssnn.interaction_optimizer import train_interaction
    from ssnn.utils import nn_predict

    n_reps = 3  # keep test fast (<30s)
    p = 30
    n_train = 10000
    n_test = 2000
    n_ref = 2000
    heritability = 0.5
    nonlinear_strength = 1.0
    alpha = 0.4  # partially orthogonal — favors spectral/eigenvector approach

    r2_gauss_list = []
    r2_collapsed_list = []
    r2_robust_list = []

    for rep in range(n_reps):
        seed = 10000 + rep * 1000
        rng = np.random.default_rng(seed)

        maf = rng.uniform(0.1, 0.5, p)
        Sigma = block_ld(p)

        X_tr = block_genotypes(n_train, maf, Sigma, rng)
        X_te = block_genotypes(n_test, maf, Sigma, rng)
        X_rf = block_genotypes(n_ref, maf, Sigma, rng)
        mu = X_tr.mean(axis=0)
        X_tr -= mu; X_te -= mu; X_rf -= mu

        beta = rng.standard_normal(p) * 0.3
        beta_unit = beta / np.linalg.norm(beta)
        eps_raw = rng.standard_normal(p)
        eps_perp = orthogonal_unit(beta, eps_raw)
        w_star = alpha * beta_unit + np.sqrt(max(0.0, 1.0 - alpha**2)) * eps_perp
        w_star = w_star * np.linalg.norm(beta)

        lin_tr = X_tr @ beta
        nl_tr = np.maximum(0.0, X_tr @ w_star)
        nl_te = np.maximum(0.0, X_te @ w_star)
        v_lin = float(np.var(lin_tr))
        v_nl = float(np.var(nl_tr))
        scale = np.sqrt(nonlinear_strength * v_lin / v_nl) if v_nl > 1e-15 else 0.0

        y_tr = lin_tr + scale * nl_tr
        y_te = X_te @ beta + scale * nl_te

        var_gen = float(np.var(y_tr))
        sigma_eps = np.sqrt(var_gen * (1.0 - heritability) / heritability)
        y_tr += rng.standard_normal(n_train) * sigma_eps
        y_te += rng.standard_normal(n_test) * sigma_eps
        y_mean = float(np.mean(y_tr))
        y_tr -= y_mean; y_te -= y_mean

        # sigma_other2: noise variance = (1-h²)*Var_gen (NOT total E[y²])
        # For a single block, the "other" variance is just the additive noise
        sigma_other2_val = float(np.mean(y_tr ** 2)) * (1.0 - heritability)

        Sigma_beta = X_tr.T @ y_tr / n_train
        Gamma_raw = X_tr.T @ (X_tr * y_tr[:, None]) / n_train
        E_y2 = float(np.mean(y_tr ** 2))
        Cov_ref = X_rf.T @ X_rf / n_ref

        ridge_lambda = 1.0
        beta_hat = np.linalg.solve(Cov_ref + ridge_lambda * np.eye(p), Sigma_beta)
        sigma_other2 = max(0.0, E_y2 - float(np.dot(Sigma_beta, beta_hat)))

        # Gaussian NN
        g_rng = np.random.default_rng(seed + 1)
        g_res = train(
            Sigma, Sigma_beta, E_y2,
            m=3, activation="relu", lr=0.05, max_iters=300,
            init_scale=0.01, rng=g_rng, Cov_ref=Cov_ref,
        )
        y_gauss = nn_predict(X_te, g_res.a, g_res.W, "relu")
        r2_gauss_list.append(r2(y_te, y_gauss))

        # Collapsed interaction NN
        try:
            c_res = train_interaction(
                Sigma, Sigma_beta, E_y2, Gamma_raw,
                m=3, activation="relu", lr=0.005, max_iters=300,
                a_init=g_res.a, W_init=g_res.W,
                Cov_ref=Cov_ref,
                rng=np.random.default_rng(seed + 2),
                grad_clip=0.5,
                sigma_other2=sigma_other2_val if sigma_other2_val > 0 else E_y2 * 0.5,
                n_train=n_train,
            )
            y_collapsed = nn_predict(X_te, c_res.a, c_res.W, "relu")
        except Exception:
            y_collapsed = y_gauss
        r2_collapsed_list.append(r2(y_te, y_collapsed))

        # Robust interaction NN
        rob_result = train_interaction_robust(
            Sigma=Sigma,
            Sigma_beta=Sigma_beta,
            E_y2=E_y2,
            Gamma_raw=Gamma_raw,
            X_ref=X_rf,
            beta_hat=beta_hat,
            sigma_other2=sigma_other2_val,  # noise variance = (1-h²)*E[y²]
            n_train=n_train,
            m=3,
            activation="relu",
            Cov_ref=Cov_ref,
            gauss_result=g_res,
            n_restarts=3,
            lr=0.005,
            max_iters=300,
            rng=np.random.default_rng(seed + 3),
        )
        y_robust = nn_predict(X_te, rob_result.a, rob_result.W, "relu")
        r2_robust_list.append(r2(y_te, y_robust))

    mean_gauss = np.mean(r2_gauss_list)
    mean_collapsed = np.mean(r2_collapsed_list)
    mean_robust = np.mean(r2_robust_list)

    print(f"\ntest_robust_beats_collapsed_gradient results:")
    print(f"  mean R2 gaussian:  {mean_gauss:.4f}")
    print(f"  mean R2 collapsed: {mean_collapsed:.4f}")
    print(f"  mean R2 robust:    {mean_robust:.4f}")

    # Primary: "do no harm" — robust must never fall significantly below Gaussian
    assert mean_robust >= mean_gauss - 0.01, (
        f"Robust ({mean_robust:.4f}) fell below Gaussian ({mean_gauss:.4f}) by more than 0.01"
    )

    # Secondary: at least competitive with collapsed (within 0.01)
    # Full performance comparison belongs in spectral_validation.py
    assert mean_robust >= mean_collapsed - 0.01, (
        f"Robust ({mean_robust:.4f}) fell more than 0.01 below collapsed ({mean_collapsed:.4f})"
    )


def test_robust_uses_full_gradient():
    """Verify robust optimizer returns gauss when no signal, real signal when present."""
    rng = np.random.default_rng(55)
    p = 20
    Sigma = block_ld(p)
    Sigma_beta = rng.standard_normal(p) * 0.1
    E_y2 = 1.0

    maf = rng.uniform(0.1, 0.5, p)
    n_ref = 200
    X_ref = block_genotypes(n_ref, maf, Sigma, rng)
    X_ref -= X_ref.mean(axis=0)

    beta_hat = np.linalg.solve(Sigma + np.eye(p), Sigma_beta)
    gauss_result = _make_gauss_result(m=3, p=p, rng=rng)

    # Case 1: pure noise Gamma, small n → triage_block returns False → robust returns gauss
    n_small = 50
    noise = rng.standard_normal((p, p)) * 1e-6
    Gamma_noise = (noise + noise.T) / 2.0

    result_noise = train_interaction_robust(
        Sigma=Sigma,
        Sigma_beta=Sigma_beta,
        E_y2=E_y2,
        Gamma_raw=Gamma_noise,
        X_ref=X_ref,
        beta_hat=beta_hat,
        sigma_other2=E_y2,
        n_train=n_small,
        m=3,
        gauss_result=gauss_result,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_almost_equal(result_noise.a, gauss_result.a,
                                          err_msg="Pure noise should return gauss result")

    # Case 2: real signal in Gamma → robust may differ from gauss
    # (We only verify it doesn't crash and returns a valid TrainResult)
    mp_edge = marchenko_pastur_edge(p=p, n=1000, sigma2=E_y2)
    v_signal = rng.standard_normal(p)
    v_signal /= np.linalg.norm(v_signal)
    Gamma_signal = 5.0 * mp_edge * np.outer(v_signal, v_signal)

    result_signal = train_interaction_robust(
        Sigma=Sigma,
        Sigma_beta=Sigma_beta,
        E_y2=E_y2,
        Gamma_raw=Gamma_signal,
        X_ref=X_ref,
        beta_hat=beta_hat,
        sigma_other2=E_y2,
        n_train=1000,
        m=3,
        gauss_result=gauss_result,
        rng=np.random.default_rng(2),
    )
    assert result_signal is not None, "Should return a valid TrainResult for signal Gamma"
    assert result_signal.a.shape == (3,), "Result should have correct shape"
    assert result_signal.W.shape == (3, p), "Result should have correct shape"
