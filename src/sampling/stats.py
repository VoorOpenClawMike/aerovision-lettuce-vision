"""Sampling-design statistics for weight-class estimation (Module 5).

Answers: *what fraction of the field do we need to inspect (10 / 25 / 50 %) to
estimate the weight-class distribution with a chosen confidence and margin?*

Two complementary tools:

1. **Cochran's sample-size formula** (for a proportion) with the **finite
   population correction (FPC)** — the analytic minimum sample size.

       n0 = z² · p·(1-p) / e²          (Cochran, infinite population)
       n  = n0 / (1 + (n0 - 1)/N)      (finite population correction)

   ``p`` defaults to 0.5 (maximum variance → most conservative sample size).

2. **Bootstrap BCa confidence intervals** (``scipy.stats.bootstrap``) on the
   observed class proportions for a candidate sampling fraction — an empirical
   check of the CI width you actually get, complementing the analytic formula.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_FRACTIONS = (0.10, 0.25, 0.50)


def z_score(confidence: float) -> float:
    """Two-sided z critical value for a confidence level (e.g. 0.95 -> 1.96)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    from scipy.stats import norm
    return float(norm.ppf(0.5 + confidence / 2.0))


def cochran_sample_size(confidence: float, margin_of_error: float, p: float = 0.5) -> float:
    """Cochran's ``n0`` for an infinite population."""
    if not 0.0 < margin_of_error < 1.0:
        raise ValueError("margin_of_error must be in (0, 1)")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    z = z_score(confidence)
    return (z ** 2 * p * (1.0 - p)) / (margin_of_error ** 2)


def finite_population_correction(n0: float, population: int) -> float:
    """Apply the FPC to an infinite-population sample size ``n0``."""
    if population <= 0:
        raise ValueError("population must be > 0")
    return n0 / (1.0 + (n0 - 1.0) / population)


def required_sample_size(
    population: int,
    confidence: float = 0.95,
    margin_of_error: float = 0.05,
    p: float = 0.5,
) -> int:
    """Minimum sample size (rounded up) for the given precision, with FPC."""
    n0 = cochran_sample_size(confidence, margin_of_error, p)
    n = finite_population_correction(n0, population)
    return min(population, int(math.ceil(n)))


@dataclass
class SamplingRecommendation:
    population: int
    confidence: float
    margin_of_error: float
    required_n: int
    required_fraction: float
    options: Dict[str, dict]        # "10%" -> {count, meets}
    recommended_fraction: float | None
    recommended_label: str | None


def recommend_sample_percentage(
    population: int,
    confidence: float = 0.95,
    margin_of_error: float = 0.05,
    p: float = 0.5,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> SamplingRecommendation:
    """Pick the smallest candidate fraction whose count meets the required n."""
    req_n = required_sample_size(population, confidence, margin_of_error, p)
    options: Dict[str, dict] = {}
    recommended: float | None = None
    for frac in sorted(fractions):
        count = int(math.floor(population * frac))
        meets = count >= req_n
        options[f"{int(frac * 100)}%"] = {"count": count, "meets": meets}
        if meets and recommended is None:
            recommended = frac
    return SamplingRecommendation(
        population=population,
        confidence=confidence,
        margin_of_error=margin_of_error,
        required_n=req_n,
        required_fraction=req_n / population if population else 0.0,
        options=options,
        recommended_fraction=recommended,
        recommended_label=(f"{int(recommended * 100)}%" if recommended else None),
    )


# --------------------------------------------------------------------------- #
# Bootstrap BCa on class proportions                                          #
# --------------------------------------------------------------------------- #
def bootstrap_proportion_ci(
    indicator: Sequence[int],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """BCa bootstrap CI for a proportion (mean of a 0/1 indicator).

    Returns ``(point_estimate, ci_low, ci_high)``. Degenerate samples
    (all-0 / all-1, or n < 2) return a zero-width interval at the estimate.
    """
    arr = np.asarray(indicator, dtype=float)
    point = float(arr.mean()) if arr.size else 0.0
    if arr.size < 2 or arr.min() == arr.max():
        return point, point, point
    from scipy.stats import bootstrap
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (arr,), np.mean, method="BCa", confidence_level=confidence,
        n_resamples=n_resamples, random_state=rng,
    )
    lo = float(res.confidence_interval.low)
    hi = float(res.confidence_interval.high)
    return point, lo, hi


def evaluate_sampling_fractions(
    labels: Sequence[str],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    confidence: float = 0.95,
    classes: Sequence[str] = ("light", "medium", "heavy"),
    n_resamples: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, dict]]:
    """For each fraction, draw a sample and report BCa CI per class proportion.

    The returned CI half-width shrinks as the fraction grows — the empirical
    counterpart to Cochran's formula.
    """
    labels = list(labels)
    N = len(labels)
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, dict]] = {}
    for frac in sorted(fractions):
        k = max(1, int(round(N * frac)))
        idx = rng.choice(N, size=min(k, N), replace=False)
        sample = [labels[i] for i in idx]
        per_class: Dict[str, dict] = {}
        for c in classes:
            indicator = [1 if s == c else 0 for s in sample]
            point, lo, hi = bootstrap_proportion_ci(
                indicator, confidence, n_resamples, seed=seed)
            per_class[c] = {
                "proportion": round(point, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "ci_halfwidth": round((hi - lo) / 2.0, 4),
            }
        out[f"{int(frac * 100)}%"] = {"sample_size": len(sample), "classes": per_class}
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _load_labels(csv_path: str, column: str = "weight_class") -> List[str]:
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        return [row[column] for row in csv.DictReader(f)]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cochran sample-size + bootstrap sampling advice.")
    p.add_argument("--population", type=int, required=True)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.05, help="margin of error (0-1)")
    p.add_argument("--p", type=float, default=0.5, help="expected proportion (0.5 = worst case)")
    p.add_argument("--labels-csv", default=None,
                   help="optional CSV with a weight_class column for bootstrap CIs")
    p.add_argument("--n-resamples", type=int, default=2000)
    return p


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    rec = recommend_sample_percentage(
        args.population, args.confidence, args.margin, args.p)
    result = {"recommendation": asdict(rec)}
    if args.labels_csv:
        labels = _load_labels(args.labels_csv)
        result["bootstrap"] = evaluate_sampling_fractions(
            labels, confidence=args.confidence, n_resamples=args.n_resamples)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
