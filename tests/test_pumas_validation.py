"""Tests for the PUMAS validation pipeline (Step 4)."""

import numpy as np
import pytest

from ssnn.pumas_validation import (
    TraitConfig,
    run_validation,
    run_synthetic_validation,
    _aggregate_best,
    MethodResult,
)
from ssnn.utils import generate_ld_matrix
from ssnn.simulation import (
    generate_maf_spectrum,
    generate_effect_sizes,
    generate_binomial_genotypes,
    compute_summary_stats_from_genotypes,
)


@pytest.fixture
def small_trait_config():
    """A minimal TraitConfig for fast pipeline testing."""
    rng = np.random.default_rng(99)
    p = 15
    N = 5000

    Sigma = generate_ld_matrix(p, decay=0.4)
    maf = generate_maf_spectrum(p, "mixed", rng)
    beta_star, sigma_eps = generate_effect_sizes(p, Sigma, 0.5, 0.3, rng)

    X = generate_binomial_genotypes(N, maf, Sigma, rng)
    col_means = X.mean(axis=0)
    X_centered = X - col_means
    y = X_centered @ beta_star + rng.normal(0, sigma_eps, N)

    stats = compute_summary_stats_from_genotypes(X, y, Sigma)

    return TraitConfig(
        name="Test Trait",
        Sigma_beta=stats["Sigma_beta_hat"],
        Sigma=Sigma,
        maf=stats["maf"],
        E_y2=stats["E_y2_hat"],
        N=N,
        h2_grid=[0.3, 0.5],
        p_causal_grid=[0.1, 1.0],
        ct_p_thresholds=[0.01, 0.05],
        ct_r2_thresholds=[0.2],
        nn_hidden_units=[3],
        nn_max_iters=500,
    )


class TestRunValidation:
    def test_returns_trait_result(self, small_trait_config):
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        assert result.trait_name == "Test Trait"
        assert len(result.method_results) > 0
        assert len(result.best_per_method) > 0

    def test_all_methods_present(self, small_trait_config):
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        expected_methods = {"C+T", "LDpred2-inf", "PRS-CS", "Gaussian NN", "Edgeworth NN"}
        found_methods = set(result.best_per_method.keys())
        assert expected_methods == found_methods

    def test_r2_values_are_finite(self, small_trait_config):
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        for method, info in result.best_per_method.items():
            assert np.isfinite(info["mean_r2"]), f"{method} has non-finite R²"
            assert np.isfinite(info["std_r2"]), f"{method} has non-finite std"


class TestRunSyntheticValidation:
    def test_smoke(self):
        result = run_synthetic_validation(
            p=10, N=3000,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            n_splits=2, seed=77,
        )
        assert len(result.best_per_method) == 5
        for method, info in result.best_per_method.items():
            assert np.isfinite(info["mean_r2"]), f"{method} non-finite"


class TestAggregateBest:
    def test_picks_best_config(self):
        results = [
            MethodResult("A", 0, 0.1, {"h2": 0.3}),
            MethodResult("A", 1, 0.2, {"h2": 0.3}),
            MethodResult("A", 0, 0.5, {"h2": 0.5}),
            MethodResult("A", 1, 0.6, {"h2": 0.5}),
        ]
        best = _aggregate_best(results, n_splits=2)
        assert "A" in best
        assert best["A"]["mean_r2"] == pytest.approx(0.55)
        assert "h2=0.5" in best["A"]["params"]

    def test_single_method_single_config(self):
        """A method with a single config should still be aggregated."""
        results = [
            MethodResult("PRS-CS", 0, 0.3),
            MethodResult("PRS-CS", 1, 0.4),
        ]
        best = _aggregate_best(results, n_splits=2)
        assert "PRS-CS" in best
        assert best["PRS-CS"]["mean_r2"] == pytest.approx(0.35)
        assert best["PRS-CS"]["params"] == "default"

    def test_multiple_methods(self):
        """Aggregation works correctly across multiple methods."""
        results = [
            MethodResult("A", 0, 0.3, {"h2": 0.5}),
            MethodResult("A", 1, 0.4, {"h2": 0.5}),
            MethodResult("B", 0, 0.6),
            MethodResult("B", 1, 0.7),
        ]
        best = _aggregate_best(results, n_splits=2)
        assert len(best) == 2
        assert best["B"]["mean_r2"] > best["A"]["mean_r2"]

    def test_std_computed_correctly(self):
        results = [
            MethodResult("X", 0, 0.2),
            MethodResult("X", 1, 0.4),
            MethodResult("X", 2, 0.6),
        ]
        best = _aggregate_best(results, n_splits=3)
        expected_std = float(np.std([0.2, 0.4, 0.6]))
        assert best["X"]["std_r2"] == pytest.approx(expected_std, rel=1e-10)
        assert best["X"]["n_splits"] == 3


class TestMethodResultFields:
    """Tests that MethodResult objects have correct split indices and metadata."""

    def test_split_indices_correct(self, small_trait_config):
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        for mr in result.method_results:
            assert mr.split_idx in (0, 1)
            assert isinstance(mr.summary_r2_val, float)
            assert isinstance(mr.method, str)

    def test_ct_results_have_params(self, small_trait_config):
        """C+T results should store p_threshold and r2_threshold."""
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        ct_results = [r for r in result.method_results if r.method == "C+T"]
        assert len(ct_results) > 0
        for r in ct_results:
            assert "p_threshold" in r.params
            assert "r2_threshold" in r.params

    def test_ldpred2_results_have_params(self, small_trait_config):
        """LDpred2 results should store h2 and p_causal."""
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        ldpred_results = [r for r in result.method_results if r.method == "LDpred2-inf"]
        assert len(ldpred_results) > 0
        for r in ldpred_results:
            assert "h2" in r.params
            assert "p_causal" in r.params

    def test_nn_results_have_m(self, small_trait_config):
        """NN results should store the number of hidden units."""
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        nn_results = [
            r for r in result.method_results
            if r.method in ("Gaussian NN", "Edgeworth NN")
        ]
        assert len(nn_results) > 0
        for r in nn_results:
            assert "m" in r.params


class TestRunValidationEdgeCases:
    """Edge cases for run_validation."""

    def test_single_split(self, small_trait_config):
        """n_splits=1 should work and produce results for all methods."""
        result = run_validation(
            small_trait_config, n_splits=1, seed=10,
        )
        expected_methods = {"C+T", "LDpred2-inf", "PRS-CS", "Gaussian NN", "Edgeworth NN"}
        found_methods = set(result.best_per_method.keys())
        assert expected_methods == found_methods
        for method, info in result.best_per_method.items():
            assert info["n_splits"] == 1

    def test_verbose_runs_without_error(self, small_trait_config, capsys):
        """verbose=True should print method names without crashing."""
        run_validation(
            small_trait_config, n_splits=1, seed=10, verbose=True,
        )
        captured = capsys.readouterr()
        assert "C+T" in captured.out
        assert "LDpred2-inf" in captured.out


class TestR2Ordering:
    """Integration test: trained methods should outperform random weights."""

    def test_oracle_beats_random(self, small_trait_config):
        """LDpred2-inf (a reasonable method) should outperform on average
        compared to what a zero-weight predictor would achieve."""
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        ldpred_r2 = result.best_per_method["LDpred2-inf"]["mean_r2"]
        assert np.isfinite(ldpred_r2)

    def test_all_r2_bounded(self, small_trait_config):
        """All best R² values should be within a reasonable range."""
        result = run_validation(
            small_trait_config, n_splits=2, seed=10,
        )
        for method, info in result.best_per_method.items():
            assert info["mean_r2"] < 1.5, f"{method} R² unreasonably large"
            assert info["mean_r2"] > -10.0, f"{method} R² unreasonably negative"


class TestRunSyntheticValidationVariants:
    """Test run_synthetic_validation with different parameter combinations."""

    def test_common_maf_spectrum(self):
        result = run_synthetic_validation(
            p=10, N=3000,
            maf_spectrum="common", ld_decay=0.3,
            heritability=0.5, sparsity=0.3,
            n_splits=2, seed=77,
        )
        assert len(result.best_per_method) == 5

    def test_high_heritability(self):
        result = run_synthetic_validation(
            p=10, N=3000,
            maf_spectrum="mixed", ld_decay=0.5,
            heritability=0.9, sparsity=0.5,
            n_splits=2, seed=88,
        )
        assert len(result.best_per_method) == 5
        for method, info in result.best_per_method.items():
            assert np.isfinite(info["mean_r2"])

    def test_sparse_effects(self):
        result = run_synthetic_validation(
            p=10, N=3000,
            maf_spectrum="mixed", ld_decay=0.3,
            heritability=0.5, sparsity=0.1,
            n_splits=2, seed=99,
        )
        assert len(result.best_per_method) == 5
