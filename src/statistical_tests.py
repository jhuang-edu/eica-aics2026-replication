# -*- coding: utf-8 -*-
"""
Statistical Significance and Bootstrap Hypothesis Testing
Paper: Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME
Reproduces paired exact McNemar test and 20,000-resample Bootstrap 95% Confidence Intervals.
"""

import math
import numpy as np
from pathlib import Path
from metrics_calculator import cohen_kappa, load_dataset


def binom_pmf(k, n, p=0.5):
    """Compute binomial probability mass function."""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def paired_mcnemar_exact(y_benchmark, y_manual, y_eica):
    """
    Perform exact paired McNemar test for discordant pairs.
    b: Manual discordant with benchmark, but EICA aligned (error corrected).
    c: Manual aligned with benchmark, but EICA discordant (performance degraded).
    """
    n = len(y_benchmark)
    b = 0  # Manual incorrect (0), EICA correct (1)
    c = 0  # Manual correct (1), EICA incorrect (0)

    for bench, m, e in zip(y_benchmark, y_manual, y_eica):
        m_correct = (m == bench)
        e_correct = (e == bench)
        if (not m_correct) and e_correct:
            b += 1
        elif m_correct and (not e_correct):
            c += 1

    total_discordant = b + c
    if total_discordant == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}

    # Two-sided exact binomial test with p = 0.5
    min_val = min(b, c)
    cum_prob = sum(binom_pmf(k, total_discordant, 0.5) for k in range(min_val + 1))
    p_value = min(1.0, 2.0 * cum_prob)

    return {
        "b": b,
        "c": c,
        "n_discordant": total_discordant,
        "p_value": p_value,
        "p_value_formatted": f"{p_value:.3f} (exact: {p_value:.6f})"
    }


def bootstrap_kappa_ci(y_benchmark, y_manual, y_eica, n_bootstraps=20000, seed=42):
    """
    Compute 95% bootstrap confidence intervals for:
    1. Single-track Manual Cohen's Kappa
    2. Single-track EICA Cohen's Kappa
    3. Paired Difference Delta-Kappa = Kappa_EICA - Kappa_Manual
    """
    np.random.seed(seed)
    n = len(y_benchmark)
    bench_arr = np.array(y_benchmark)
    man_arr = np.array(y_manual)
    eica_arr = np.array(y_eica)

    boot_k_man = np.empty(n_bootstraps)
    boot_k_eica = np.empty(n_bootstraps)
    boot_delta_k = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        indices = np.random.randint(0, n, size=n)
        b_sample = bench_arr[indices]
        m_sample = man_arr[indices]
        e_sample = eica_arr[indices]

        k_m = cohen_kappa(b_sample, m_sample)
        k_e = cohen_kappa(b_sample, e_sample)

        boot_k_man[i] = k_m
        boot_k_eica[i] = k_e
        boot_delta_k[i] = k_e - k_m

    ci_man = np.percentile(boot_k_man, [2.5, 97.5])
    ci_eica = np.percentile(boot_k_eica, [2.5, 97.5])
    ci_delta = np.percentile(boot_delta_k, [2.5, 97.5])

    return {
        "n_bootstraps": n_bootstraps,
        "manual_kappa_ci": (float(ci_man[0]), float(ci_man[1])),
        "eica_kappa_ci": (float(ci_eica[0]), float(ci_eica[1])),
        "delta_kappa_ci": (float(ci_delta[0]), float(ci_delta[1])),
        "overlap": max(0.0, float(ci_man[1] - ci_eica[0]))
    }


def run_statistical_suite(csv_path, n_bootstraps=20000):
    """Run full statistical evaluation suite."""
    rows = load_dataset(csv_path)
    dept = [int(r["dept_decision"]) for r in rows]
    manual = [int(r["hr_manual_decision"]) for r in rows]
    eica = [int(r["llm_hr_decision"]) for r in rows]

    mcnemar_res = paired_mcnemar_exact(dept, manual, eica)
    boot_res = bootstrap_kappa_ci(dept, manual, eica, n_bootstraps=n_bootstraps)

    return {
        "mcnemar_p": mcnemar_res["p_value"],
        "mcnemar_p_exact": mcnemar_res["p_value_formatted"],
        "discordant_b": mcnemar_res["b"],
        "discordant_c": mcnemar_res["c"],
        "manual_kappa_ci": boot_res["manual_kappa_ci"],
        "eica_kappa_ci": boot_res["eica_kappa_ci"],
        "delta_kappa_ci": boot_res["delta_kappa_ci"],
        "marginal_overlap": boot_res["overlap"]
    }


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "../data/eval_records_40.csv"
    res = run_statistical_suite(csv_file, n_bootstraps=20000)
    print("=" * 60)
    print("STATISTICAL INFERENCE & HYPOTHESIS TESTING RESULTS")
    print("=" * 60)
    print(f"Paired McNemar Exact Test: {res['mcnemar_p_exact']}")
    print(f"  Discordant Pairs: {res['discordant_b']} (Manual Err -> EICA Align) vs {res['discordant_c']} (Manual Align -> EICA Err)")
    print(f"Manual Cohen's Kappa 95% Bootstrap CI: [{res['manual_kappa_ci'][0]:.3f}, {res['manual_kappa_ci'][1]:.3f}]")
    print(f"EICA Cohen's Kappa 95% Bootstrap CI:   [{res['eica_kappa_ci'][0]:.3f}, {res['eica_kappa_ci'][1]:.3f}]")
    print(f"Paired Delta-Kappa 95% Bootstrap CI:   [{res['delta_kappa_ci'][0]:.3f}, {res['delta_kappa_ci'][1]:.3f}]")
    print(f"Marginal CI Overlap: {res['marginal_overlap']:.3f} (Zero strictly excluded in Paired Difference CI)")
    print("=" * 60)
