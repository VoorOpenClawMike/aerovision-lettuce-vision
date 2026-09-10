"""Unit tests for sampling-design statistics (Module 5)."""
import pytest

from src.sampling.stats import (
    bootstrap_proportion_ci,
    cochran_sample_size,
    evaluate_sampling_fractions,
    finite_population_correction,
    recommend_sample_percentage,
    required_sample_size,
    z_score,
)


def test_z_score_known_values():
    assert z_score(0.95) == pytest.approx(1.959963, abs=1e-4)
    assert z_score(0.99) == pytest.approx(2.575829, abs=1e-4)
    with pytest.raises(ValueError):
        z_score(1.5)


def test_cochran_worstcase_384():
    # classic worst-case (p=0.5, 95%, 5%) sample size ~= 384
    n0 = cochran_sample_size(0.95, 0.05, p=0.5)
    assert n0 == pytest.approx(384.15, abs=1.0)
    # smaller margin -> larger sample
    assert cochran_sample_size(0.95, 0.025) > n0
    with pytest.raises(ValueError):
        cochran_sample_size(0.95, 0)


def test_fpc_reduces_sample_size():
    n0 = cochran_sample_size(0.95, 0.05)
    n_small = finite_population_correction(n0, 500)
    n_large = finite_population_correction(n0, 100000)
    assert n_small < n0                      # correction always shrinks
    assert n_small < n_large < n0            # smaller population -> smaller n
    assert required_sample_size(5000, 0.95, 0.05) == 357


def test_recommendation_picks_smallest_meeting_fraction():
    rec = recommend_sample_percentage(5000, 0.95, 0.05)
    assert rec.required_n == 357
    assert rec.recommended_label == "10%"     # 500 >= 357
    assert rec.options["10%"]["meets"] is True

    # tiny population with a tight margin: even 50% may not meet -> None
    rec2 = recommend_sample_percentage(200, 0.99, 0.02)
    assert rec2.recommended_fraction in (None, 0.5)
    # required n never exceeds population
    assert rec2.required_n <= 200


def test_bootstrap_proportion_ci_bounds_and_degenerate():
    # degenerate: all ones -> zero-width interval at 1.0
    p, lo, hi = bootstrap_proportion_ci([1, 1, 1, 1])
    assert p == 1.0 and lo == 1.0 and hi == 1.0
    # mixed sample: low <= point <= high, all within [0,1]
    data = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 5
    p, lo, hi = bootstrap_proportion_ci(data, n_resamples=500, seed=1)
    assert 0.0 <= lo <= p <= hi <= 1.0
    assert p == pytest.approx(0.5, abs=0.01)


def test_evaluate_sampling_fractions_structure():
    labels = (["light"] * 60 + ["medium"] * 25 + ["heavy"] * 15) * 3
    out = evaluate_sampling_fractions(labels, confidence=0.95,
                                      n_resamples=400, seed=0)
    assert set(out.keys()) == {"10%", "25%", "50%"}
    for frac_key, block in out.items():
        assert block["sample_size"] >= 1
        for cls in ("light", "medium", "heavy"):
            c = block["classes"][cls]
            assert 0.0 <= c["ci_low"] <= c["proportion"] <= c["ci_high"] <= 1.0
            assert c["ci_halfwidth"] >= 0.0
    # larger fraction => at least as many samples
    assert out["50%"]["sample_size"] > out["10%"]["sample_size"]
