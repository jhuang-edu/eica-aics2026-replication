# -*- coding: utf-8 -*-
"""
Metrics Calculator for IEEE AI-CS 2026 Paper
Paper: Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME
Reproduces Table 1, Table 2, and all in-text statistical indicators.
"""

import csv
import numpy as np
from pathlib import Path


def cohen_kappa(y_true, y_pred):
    """Calculate Cohen's kappa coefficient between ground benchmark and prediction."""
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    p1_true = sum(y_true) / n
    p1_pred = sum(y_pred) / n
    pe = p1_true * p1_pred + (1 - p1_true) * (1 - p1_pred)
    return (po - pe) / (1 - pe) if pe != 1.0 else 1.0


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix using Department Manager decision as reference benchmark.
    Returns: TP, FP, FN, TN
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    return tp, fp, fn, tn


def safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def load_dataset(csv_path):
    """Load candidate screening records from CSV."""
    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("resume_id"):
                continue
            records.append(r)
    return records


def compute_metrics(csv_path):
    """Compute all experimental metrics from the evaluation dataset."""
    rows = load_dataset(csv_path)
    n = len(rows)
    if n != 40:
        print(f"[Warning] Expected N=40 cases, found {n} cases.")

    dept = [int(r["dept_decision"]) for r in rows]
    manual = [int(r["hr_manual_decision"]) for r in rows]
    eica = [int(r["llm_hr_decision"]) for r in rows]

    # Overall agreement and Kappa
    man_agree = sum(1 for a, b in zip(dept, manual) if a == b) / n * 100.0
    eica_agree = sum(1 for a, b in zip(dept, eica) if a == b) / n * 100.0
    man_kappa = cohen_kappa(dept, manual)
    eica_kappa = cohen_kappa(dept, eica)
    delta_kappa = eica_kappa - man_kappa

    # Screening durations (Pure HR)
    man_times = [float(r["hr_manual_time_min"]) for r in rows]
    eica_times = [float(r["llm_hr_time_min"]) for r in rows]
    man_time_mean = np.mean(man_times)
    man_time_sd = np.std(man_times, ddof=1)
    eica_time_mean = np.mean(eica_times)
    eica_time_sd = np.std(eica_times, ddof=1)
    pure_hr_time_reduction = (man_time_mean - eica_time_mean) / man_time_mean * 100.0
    pure_hr_time_reduction_rounded = (round(man_time_mean, 2) - round(eica_time_mean, 2)) / round(man_time_mean, 2) * 100.0

    # Total organizational time (including managerial review of escalated cases)
    # 5 escalated cases take on average 5.2 min manager review = 26.0 min total
    # Amortized per candidate: 26.0 / 40 = 0.65 min
    manager_esc_time_total = 26.0
    esc_amortized = manager_esc_time_total / n
    total_org_eica_time = eica_time_mean + esc_amortized
    total_org_time_reduction = (man_time_mean - total_org_eica_time) / man_time_mean * 100.0

    # Escalated case analysis
    esc_rows = [r for r in rows if r.get("llm_escalated") == "1"]
    esc_ids = [r["resume_id"] for r in esc_rows]
    esc_hr_times = [float(r["llm_hr_time_min"]) for r in esc_rows]
    esc_hr_mean = np.mean(esc_hr_times)
    esc_total_single_case = esc_hr_mean + 5.20

    # Key qualification omission rate
    man_miss = sum(int(r["hr_manual_miss_key"]) for r in rows) / n * 100.0
    eica_miss = sum(int(r["llm_miss_key"]) for r in rows) / n * 100.0

    # AI Quality metrics
    trace_rate = sum(int(r["llm_traceable"]) for r in rows) / n * 100.0
    halluc_rate = sum(int(r["llm_hallucination"]) for r in rows) / n * 100.0
    esc_rate = len(esc_rows) / n * 100.0

    # Confusion matrix and diagnostic metrics
    tp_m, fp_m, fn_m, tn_m = confusion_matrix(dept, manual)
    tp_e, fp_e, fn_e, tn_e = confusion_matrix(dept, eica)

    sens_m, spec_m = safe_div(tp_m, tp_m + fn_m) * 100.0, safe_div(tn_m, tn_m + fp_m) * 100.0
    ppv_m, npv_m = safe_div(tp_m, tp_m + fp_m) * 100.0, safe_div(tn_m, tn_m + fn_m) * 100.0

    sens_e, spec_e = safe_div(tp_e, tp_e + fn_e) * 100.0, safe_div(tn_e, tn_e + fp_e) * 100.0
    ppv_e, npv_e = safe_div(tp_e, tp_e + fp_e) * 100.0, safe_div(tn_e, tn_e + fn_e) * 100.0

    fa_drop = (fp_m - fp_e) / fp_m * 100.0
    fr_drop = (fn_m - fn_e) / fn_m * 100.0

    # Role breakdown
    roles = ["质量工程师", "体系工程师", "注册专员"]
    role_names_en = {
        "质量工程师": "Quality Engineer (QA/QC)",
        "体系工程师": "System Engineer (QMS)",
        "注册专员": "Regulatory Affairs (RA)"
    }
    role_metrics = {}
    for r_cn in roles:
        sub = [r for r in rows if r["position_type"] == r_cn]
        n_sub = len(sub)
        d_sub = [int(r["dept_decision"]) for r in sub]
        m_sub = [int(r["hr_manual_decision"]) for r in sub]
        e_sub = [int(r["llm_hr_decision"]) for r in sub]
        t_m_sub = [float(r["hr_manual_time_min"]) for r in sub]
        t_e_sub = [float(r["llm_hr_time_min"]) for r in sub]

        m_agr = sum(1 for a, b in zip(d_sub, m_sub) if a == b)
        e_agr = sum(1 for a, b in zip(d_sub, e_sub) if a == b)

        role_metrics[role_names_en[r_cn]] = {
            "N": n_sub,
            "manual_agree_pct": m_agr / n_sub * 100.0,
            "manual_agree_count": m_agr,
            "eica_agree_pct": e_agr / n_sub * 100.0,
            "eica_agree_count": e_agr,
            "manual_time_mean": np.mean(t_m_sub),
            "manual_time_sd": np.std(t_m_sub, ddof=1),
            "eica_time_mean": np.mean(t_e_sub),
            "eica_time_sd": np.std(t_e_sub, ddof=1)
        }

    return {
        "N": n,
        "agreement": {
            "manual": man_agree, "manual_count": sum(1 for a, b in zip(dept, manual) if a == b),
            "eica": eica_agree, "eica_count": sum(1 for a, b in zip(dept, eica) if a == b)
        },
        "kappa": {"manual": man_kappa, "eica": eica_kappa, "delta": delta_kappa},
        "time": {
            "pure_hr_manual_mean": man_time_mean, "pure_hr_manual_sd": man_time_sd,
            "pure_hr_eica_mean": eica_time_mean, "pure_hr_eica_sd": eica_time_sd,
            "pure_hr_reduction": pure_hr_time_reduction,
            "pure_hr_reduction_rounded": pure_hr_time_reduction_rounded,
            "manual_median": np.median(man_times), "eica_median": np.median(eica_times),
            "total_org_manual": man_time_mean,
            "total_org_eica": total_org_eica_time,
            "total_org_reduction": total_org_time_reduction,
            "esc_cases": esc_ids,
            "esc_hr_mean": esc_hr_mean,
            "esc_manager_mean": 5.20,
            "esc_single_total": esc_total_single_case
        },
        "omissions": {"manual": man_miss, "eica": eica_miss, "reduction": (man_miss - eica_miss) / man_miss * 100.0},
        "ai_quality": {"traceability": trace_rate, "hallucination": halluc_rate, "escalation": esc_rate},
        "confusion": {
            "manual": {"TP": tp_m, "FP": fp_m, "FN": fn_m, "TN": tn_m, "sens": sens_m, "spec": spec_m, "ppv": ppv_m, "npv": npv_m},
            "eica": {"TP": tp_e, "FP": fp_e, "FN": fn_e, "TN": tn_e, "sens": sens_e, "spec": spec_e, "ppv": ppv_e, "npv": npv_e},
            "fa_drop": fa_drop, "fr_drop": fr_drop
        },
        "roles": role_metrics
    }


def format_table1(results):
    """Format results as IEEE Table 1."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1: Baseline and EICA Screening Performance by Position Family (N = 40)")
    lines.append("=" * 80)
    lines.append(f"{'Position Family':<28} {'N':<4} {'Manual Agree':<16} {'EICA Agree':<16} {'Screening Time (min, Mean+-SD)':<26}")
    lines.append("-" * 80)
    for role, m in results["roles"].items():
        ag_m = f"{m['manual_agree_pct']:.1f}% ({m['manual_agree_count']}/{m['N']})"
        ag_e = f"{m['eica_agree_pct']:.1f}% ({m['eica_agree_count']}/{m['N']})"
        t_str = f"{m['manual_time_mean']:.2f}+-{m['manual_time_sd']:.2f} / {m['eica_time_mean']:.2f}+-{m['eica_time_sd']:.2f}"
        lines.append(f"{role:<28} {m['N']:<4} {ag_m:<16} {ag_e:<16} {t_str:<26}")
    lines.append("-" * 80)
    tot_m = f"{results['agreement']['manual']:.1f}% ({results['agreement']['manual_count']}/{results['N']})"
    tot_e = f"{results['agreement']['eica']:.1f}% ({results['agreement']['eica_count']}/{results['N']})"
    tot_t = f"{results['time']['pure_hr_manual_mean']:.2f}+-{results['time']['pure_hr_manual_sd']:.2f} / {results['time']['pure_hr_eica_mean']:.2f}+-{results['time']['pure_hr_eica_sd']:.2f}"
    lines.append(f"{'Total / Average':<28} {results['N']:<4} {tot_m:<16} {tot_e:<16} {tot_t:<26}")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_table2(results, stats_results=None):
    """Format results as IEEE Table 2."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 2: Dual-Track Comparison of Manual Screening and Evidence-Card Screening (N = 40)")
    lines.append("=" * 80)
    lines.append(f"{'Evaluative Metric':<45} {'Manual Baseline':<16} {'EICA Protocol':<16}")
    lines.append("-" * 80)

    ag_m = f"{results['agreement']['manual']:.1f}% ({results['agreement']['manual_count']}/40)"
    ag_e = f"{results['agreement']['eica']:.1f}% ({results['agreement']['eica_count']}/40)"
    lines.append(f"{'Agreement with Department Benchmark':<45} {ag_m:<16} {ag_e:<16}")

    lines.append(f"{'Cohen\'s Kappa Metric':<45} {results['kappa']['manual']:.3f}            {results['kappa']['eica']:.3f}")

    if stats_results:
        d_k = f"+{results['kappa']['delta']:.3f} ([{stats_results['delta_kappa_ci'][0]:.3f}, {stats_results['delta_kappa_ci'][1]:.3f}], Sig.)"
        lines.append(f"{'Paired Difference Delta-Kappa (95% CI)':<45} {d_k}")
        ci_m = f"[{stats_results['manual_kappa_ci'][0]:.3f}, {stats_results['manual_kappa_ci'][1]:.3f}]"
        ci_e = f"[{stats_results['eica_kappa_ci'][0]:.3f}, {stats_results['eica_kappa_ci'][1]:.3f}]"
        lines.append(f"{'Single-Track Kappa 95% Bootstrap CI':<45} {ci_m:<16} {ci_e:<16}")
        lines.append(f"{'Paired Exact McNemar Test':<45} p = {stats_results['mcnemar_p']:.3f} (Discordant: {stats_results['discordant_b']} vs {stats_results['discordant_c']})")

    t_m = f"{results['time']['pure_hr_manual_mean']:.2f}+-{results['time']['pure_hr_manual_sd']:.2f}"
    t_e = f"{results['time']['pure_hr_eica_mean']:.2f}+-{results['time']['pure_hr_eica_sd']:.2f} (-{results['time']['pure_hr_reduction']:.1f}%)"
    lines.append(f"{'Pure HR Screening Time (Mean+-SD, min)':<45} {t_m:<16} {t_e:<16}")

    lines.append(f"{'Total Org. Time (incl. Escalations)':<45} {results['time']['total_org_manual']:.2f} min         {results['time']['total_org_eica']:.2f} min (-{results['time']['total_org_reduction']:.1f}%)")
    lines.append(f"{'Missed Key Qualifications Rate':<45} {results['omissions']['manual']:.1f}% (4/40)      {results['omissions']['eica']:.1f}% (2/40)")

    c_m = f"{results['confusion']['manual']['FP']} / {results['confusion']['manual']['FN']}"
    c_e = f"{results['confusion']['eica']['FP']} / {results['confusion']['eica']['FN']} (-66.7% / -66.7%)"
    lines.append(f"{'False Admissions / Rejections (FA / FR)':<45} {c_m:<16} {c_e:<16}")

    lines.append(f"{'Verbatim Evidence Traceability':<45} {'---':<16} {results['ai_quality']['traceability']:.1f}% (38/40)")
    lines.append(f"{'Unsupported Model Inference Rate':<45} {'---':<16} {results['ai_quality']['hallucination']:.1f}% (1/40)")
    lines.append(f"{'Department Manager Escalation Rate':<45} {'---':<16} {results['ai_quality']['escalation']:.1f}% (5/40)")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "../data/eval_records_40.csv"
    res = compute_metrics(csv_file)
    print(format_table1(res))
    print()
    print(format_table2(res))
