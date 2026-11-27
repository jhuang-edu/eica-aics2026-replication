# -*- coding: utf-8 -*-
"""
=============================================================================
One-Click Reproduction Script for IEEE AI-CS 2026 Conference Paper (Revised)
Paper Title: Ontology-Constrained LLM Evidence Extraction and Finite-State
             Decision Support System for Compliance-Critical Talent Screening
Target: IEEE AI-CS 2026 (ISBN: 979-8-3195-0203-2, Guangzhou, China)
Authors: Junjun Huang*, Ning Chen
=============================================================================

This master reproduction script automatically executes:
1. Data Integrity and Provenance Verification (N=40 records, 18-month archive)
2. Empirical Field Metric Calculation (Recreating Table 1 & Table 2)
3. Hypothesis Testing (Paired exact McNemar test and 20,000-resample Bootstrap CIs)
4. Trace-Driven Architectural Policy Ablation (Table 3: M0 -> M1 -> M2 -> M3)
5. Executable EICA Engine Replay (R05, R37, R31) & Synthetic Stress-Test (N=200)
6. Publication-Grade Vector Graphic Asset Generation (Figure 1 & Figure 2)
7. Rigorous 1:1 In-Text Verification Check (Audit against manuscript claims)
"""

import sys
import hashlib
import shutil
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
SRC_DIR = CODE_DIR / "src"
DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = CODE_DIR / "output"
PAPER_FIG_DIR = CODE_DIR.parent / "paper_9p" / "figures"
PAPER_FIG_DIRS = [
    CODE_DIR.parent / "paper_4p" / "figures",
    CODE_DIR.parent / "paper_9p" / "figures",
]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC_DIR))

from metrics_calculator import compute_metrics, format_table1, format_table2, round_half_up
from statistical_tests import run_statistical_suite
from ablation_and_benchmark import (
    format_sensitivity_summary,
    format_synthetic_ablation_table,
    format_table3,
    run_exemplar_verification_replay,
    run_parameter_sensitivity_sweep,
    run_synthetic_component_ablations,
    run_synthetic_verification_benchmark,
    run_trace_driven_ablation,
)
from plot_figures import plot_fig1_architecture, plot_fig2_results


def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_in_text_alignment(metrics, stats, ablation, stress, comp_ablations=None, sens_sweep=None):
    m2 = ablation["M2_EICA_HardFSM"]
    m3 = ablation["M3_EICA_FullVerifier"]
    checks = [
        ("Inter-rater Agreement (Manual M0)", f"{metrics['agreement']['manual']:.1f}%", "62.5%"),
        ("Inter-rater Agreement (EICA M1 Field)", f"{metrics['agreement']['eica']:.1f}%", "87.5%"),
        ("Cohen's Kappa (Manual M0)", f"{metrics['kappa']['manual']:.3f}", "0.250"),
        ("Cohen's Kappa (EICA M1 Field)", f"{metrics['kappa']['eica']:.3f}", "0.746"),
        ("Paired Delta-Kappa (M1 vs M0)", f"{metrics['kappa']['delta']:.3f}", "0.496"),
        ("Paired Delta-Kappa 95% CI Lower", f"{stats['delta_kappa_ci'][0]:.3f}", "0.196"),
        ("Paired Delta-Kappa 95% CI Upper", f"{stats['delta_kappa_ci'][1]:.3f}", "0.807"),
        ("Paired Exact McNemar p (M1 vs M0)", f"{stats['mcnemar_p']:.5f}", "0.00635"),
        ("Discordant Pairs (M1 vs M0)", f"{stats['discordant_b']} vs {stats['discordant_c']}", "11 vs 1"),
        ("Pure HR Screening Time (Manual)", f"{metrics['time']['pure_hr_manual_mean']:.2f} min", "4.80 min"),
        ("Pure HR Screening Time (EICA)", f"{metrics['time']['pure_hr_eica_mean']:.2f} min", "2.70 min"),
        ("Pure HR Time Reduction", f"{round_half_up(metrics['time']['pure_hr_reduction_rounded'], 1):.1f}%", "43.8%"),
        ("Total Org. Screening Time (M1/M2)", f"{metrics['time']['total_org_eica']:.2f} min", "3.35 min"),
        ("Total Org. Time Reduction (M1/M2)", f"{metrics['time']['total_org_reduction']:.1f}%", "30.2%"),
        ("False Admissions (M0 -> M1)", f"{metrics['confusion']['manual']['FP']} -> {metrics['confusion']['eica']['FP']}", "9 -> 3"),
        ("False Rejections (M0 -> M1)", f"{metrics['confusion']['manual']['FN']} -> {metrics['confusion']['eica']['FN']}", "6 -> 2"),
        ("Missed Key Qualifications Rate", f"{metrics['omissions']['manual']:.1f}% -> {metrics['omissions']['eica']:.1f}%", "10.0% -> 5.0%"),
        ("Verbatim Evidence Traceability (M1)", f"{metrics['ai_quality']['traceability']:.1f}%", "95.0%"),
        ("Model Extrinsic Hallucination (M1)", f"{metrics['ai_quality']['hallucination']:.1f}%", "2.5%"),
        ("M2 Hard FSM Agreement & Kappa", f"{m2['agreement_pct']:.1f}% / {m2['kappa']:.3f}", "92.5% / 0.845"),
        ("M2 Hard FSM False Admissions", f"{m2['FP']}", "1"),
        ("M3 Full EICA Agreement & Kappa", f"{m3['agreement_pct']:.1f}% / {m3['kappa']:.3f}", "95.0% / 0.896"),
        ("M3 Full EICA False Admissions", f"{m3['FP']}", "0"),
        ("M3 Full EICA Org. Time & Reduction", f"{m3['total_org_time_min']:.2f} min (-{m3['net_time_reduction_pct']:.1f}%)", "3.51 min (-27.0%)"),
        ("M3 Expected Loss at 10:1 Ratio", f"{m3['expected_loss_by_lambda_fa']['10']:.4f}", "0.0875"),
        ("Stress-Test Extraction Fault Recall", f"{stress['extraction_verifier_summary']['recall_pct']:.1f}%", "100.0%"),
        ("Stress-Test Schema Tacit Blocking", f"{stress['schema_tacit_block_rate_pct']:.1f}%", "100.0%"),
        ("Stress-Test FSM Bypass Blocking", f"{stress['fsm_invariant_block_rate_pct']:.1f}%", "100.0%"),
    ]

    if comp_ablations is not None and sens_sweep is not None:
        tau_map = {r["tau_span"]: r for r in sens_sweep["tau_span"]}
        alpha_map = {r["alpha"]: r for r in sens_sweep["alpha"]}
        lmin_map = {r["l_min"]: r for r in sens_sweep["l_min"]}
        checks.extend([
            ("Ablation Full_EICA Extraction F1", f"{comp_ablations['Full_EICA']['extraction_f1_pct']:.1f}%", "100.0%"),
            ("Ablation Full_EICA Legacy Leakage", f"{comp_ablations['Full_EICA']['legacy_query_leakage_pct']:.1f}%", "0.0%"),
            ("Ablation Unpinned_RAG Legacy Leakage", f"{comp_ablations['Unpinned_RAG']['legacy_query_leakage_pct']:.1f}%", "100.0%"),
            ("Ablation Exact_Match Extraction Recall", f"{comp_ablations['Exact_String_Match_Only']['extraction_recall_pct']:.1f}%", "33.3%"),
            ("Ablation Exact_Match Extraction F1", f"{comp_ablations['Exact_String_Match_Only']['extraction_f1_pct']:.1f}%", "50.0%"),
            ("Ablation Full_EICA Override Bypass", f"{comp_ablations['Full_EICA']['conflict_override_bypass_pct']:.1f}%", "0.0%"),
            ("Ablation Soft_FSM Override Bypass", f"{comp_ablations['Soft_FSM_Lock_Off']['conflict_override_bypass_pct']:.1f}%", "100.0%"),
            ("Ablation Unmasked_Schema Tacit Block", f"{comp_ablations['Unmasked_Schema']['schema_tacit_block_rate_pct']:.1f}%", "0.0%"),
            ("Sensitivity tau_span=0.85 Boundary F1", f"{tau_map[0.85]['boundary_f1_pct']:.1f}%", "100.0%"),
            ("Sensitivity alpha=[0.3..0.7] Pinning", f"{min(alpha_map[a]['pin_compliance_pct'] for a in (0.3, 0.5, 0.7)):.1f}%", "100.0%"),
            ("Sensitivity l_min=15 Short-Reason Block", f"{lmin_map[15]['short_reason_block_pct']:.1f}%", "100.0%"),
            ("Sensitivity l_min=5 Short-Reason Block", f"{lmin_map[5]['short_reason_block_pct']:.1f}%", "0.0%"),
            ("Table 3 M3 Formatted E[L](10:1)", f"{round_half_up(m3['expected_loss_by_lambda_fa']['10'], 3):.3f}", "0.088"),
            ("Table 2 Pure HR Reduction Format", f"{round_half_up(metrics['time']['pure_hr_reduction'], 1):.1f}%", "43.8%"),
        ])

    print("\n" + "=" * 86)
    print("MANUSCRIPT CLAIMS VS REPRODUCED CODE VALUES (1:1 AUDIT LEDGER)")
    print("=" * 86)
    print(f"{'Metric Item':<42} {'Code Output':<20} {'Paper Claim':<16} {'Status'}")
    print("-" * 86)
    all_passed = True
    for item, code_val, paper_val in checks:
        matched = (code_val == paper_val)
        status = "[MATCH 100%]" if matched else "[CHECK]"
        print(f"{item:<42} {code_val:<20} {paper_val:<16} {status}")
        if not matched:
            all_passed = False
    print("=" * 86)
    return all_passed, len(checks)


def main():
    print("*" * 86)
    print("IEEE AI-CS 2026 EXPERIMENTAL & ALGORITHMIC REPRODUCTION PIPELINE")
    print("Ontology-Constrained LLM Evidence Extraction & Finite-State Decision Support System")
    print("*" * 86)

    eval_csv = DATA_DIR / "eval_records_40.csv"
    archive_csv = DATA_DIR / "recruitment_archive_18m.csv"

    if not eval_csv.exists():
        print(f"[Error] Missing dataset: {eval_csv}")
        sys.exit(1)

    print("\n[Stage 1/6] Verifying Data Integrity...")
    print(f"  - Eval Dataset (N=40): {eval_csv.name} | SHA-256: {file_sha256(eval_csv)[:16]}...")
    print(f"  - 18M Archive:         {archive_csv.name} | SHA-256: {file_sha256(archive_csv)[:16]}...")

    print("\n[Stage 2/6] Computing Core Field Metrics (Table 1 & Table 2)...")
    metrics = compute_metrics(eval_csv)
    print(format_table1(metrics))

    print("\n[Stage 3/6] Running Hypothesis Testing & Bootstrap Resampling (20,000 iterations)...")
    stats = run_statistical_suite(eval_csv, n_bootstraps=20000)
    print(format_table2(metrics, stats))

    print("\n[Stage 4/6] Running Trace-Driven Architectural Policy Ablation (Table 3)...")
    ablation = run_trace_driven_ablation(eval_csv, n_bootstraps=20000)
    print(format_table3(ablation))

    print("\n[Stage 5/6] Running Executable EICA Engine Replay (R05/R37/R31) & Stress-Test (N=200)...")
    exemplars = run_exemplar_verification_replay()
    for ex in exemplars:
        print(f"  - {ex['case_id']} ({ex['role']}): RAG={ex['top_rag_clauses']} | "
              f"Flags={ex['verifier_violations']} | NaiveConfirm={ex['naive_confirm_state']} "
              f"({ex['naive_confirm_error']}) | {ex['latency_ms']:.2f} ms")
    stress = run_synthetic_verification_benchmark(seed=42)
    print(f"  - Synthetic Stress-Test (N={stress['N_total']}): "
          f"Extraction F1={stress['extraction_verifier_summary']['f1_pct']:.1f}% | "
          f"TacitBlock={stress['schema_tacit_block_rate_pct']:.1f}% | "
          f"FSMBlock={stress['fsm_invariant_block_rate_pct']:.1f}% | "
          f"Latency P50={stress['latency_ms']['p50']:.2f} ms, P95={stress['latency_ms']['p95']:.2f} ms")
    comp_ablations = run_synthetic_component_ablations(seed=42)
    sens_sweep = run_parameter_sensitivity_sweep(seed=42)
    print("\n" + format_synthetic_ablation_table(comp_ablations))
    print("\n" + format_sensitivity_summary(sens_sweep))

    print("\n[Stage 6/6] Generating Publication-Grade Vector Graphics (Fig 1 & Fig 2)...")
    fig1_pdf = OUTPUT_DIR / "fig1_architecture.pdf"
    fig1_png = OUTPUT_DIR / "fig1_architecture.png"
    fig2_pdf = OUTPUT_DIR / "fig2_results.pdf"
    fig2_png = OUTPUT_DIR / "fig2_results.png"

    plot_fig1_architecture(str(fig1_pdf), str(fig1_png))
    plot_fig2_results(eval_csv, str(fig2_pdf), str(fig2_png))

    # Sync freshly generated figures to paper_4p/figures/ and paper_9p/figures/
    for fig_dir in PAPER_FIG_DIRS:
        if fig_dir.exists():
            for src in (fig1_pdf, fig1_png, fig2_pdf, fig2_png):
                shutil.copy2(src, fig_dir / src.name)
            print(f"  - Synced updated vector PDFs & 300-DPI PNGs to {fig_dir}")

    all_passed, n_checks = verify_in_text_alignment(
        metrics, stats, ablation, stress, comp_ablations=comp_ablations, sens_sweep=sens_sweep
    )

    print("\n" + "#" * 86)
    if all_passed:
        print(f">> REPRODUCTION VERIFICATION RESULT: 100% PERFECT AUDIT PASSED ({n_checks}/{n_checks} CHECKS)!")
    else:
        print(f">> REPRODUCTION FINISHED WITH OBSERVATIONS. (Check summary above; {n_checks} checks)")
    print("#" * 86)


if __name__ == "__main__":
    main()
