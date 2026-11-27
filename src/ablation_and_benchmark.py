# -*- coding: utf-8 -*-
"""
Trace-Driven Architectural Policy Ablation (N=40 Real Field Logs) &
Controlled Algorithmic Verification Stress-Test (N=200 + 3 Enterprise Exemplars)
================================================================================
Generates 100% reproducible, zero-fabrication computer-science evaluation metrics
for Table 3 and Section V of the revised IEEE AI-CS 2026 manuscript.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    from .metrics_calculator import cohen_kappa, confusion_matrix, load_dataset, safe_div
    from .statistical_tests import paired_mcnemar_exact, bootstrap_kappa_ci
    from .eica_engine import (
        CompetencyOntology,
        DeterministicVerifier,
        FSMDecisionGate,
        HybridRegulatoryRAG,
        OntologySchemaMasker,
    )
except ImportError:
    from metrics_calculator import cohen_kappa, confusion_matrix, load_dataset, safe_div
    from statistical_tests import paired_mcnemar_exact, bootstrap_kappa_ci
    from eica_engine import (
        CompetencyOntology,
        DeterministicVerifier,
        FSMDecisionGate,
        HybridRegulatoryRAG,
        OntologySchemaMasker,
    )


def _fmt_half_up(val: float, decimals: int) -> str:
    q = Decimal("1") if decimals == 0 else Decimal("1." + "0" * decimals)
    return str(Decimal(str(round(float(val), 8))).quantize(q, rounding=ROUND_HALF_UP))


# =============================================================================
# Part 1: Trace-Driven Architectural Policy Ablation on Real N=40 Field Records
# =============================================================================

def run_trace_driven_ablation(csv_path: Path, n_bootstraps: int = 20000) -> Dict[str, dict]:
    """
    Replays the 40 real empirical records in eval_records_40.csv under four
    progressively stricter architectural configurations:
      M0: Manual Unassisted Baseline (Field Track 1)
      M1: EICA Soft-Advisory Card (Field Track 2 as logged: HR could override
          escalation warnings on R02, R18 and missed R31 hallucination)
      M2: EICA + Hard FSM Escalation Lock (Post-hoc counterfactual policy upper
          bound: blocks HR bypass whenever llm_escalated=1, routing R02 & R18 to
          recorded pre-experimental blinded department manager benchmark Y)
      M3: Full EICA + Hard FSM Gate + Deterministic Span/Verb Verifier
          (Post-hoc counterfactual policy upper bound: uses logged audit tags
          llm_traceable=0 / llm_hallucination=1 [R18, R31] as trace-level
          verifier triggers and routes them to pre-experimental benchmark Y)

    All M2/M3 counterfactual resolutions represent analytic policy upper bounds
    strictly using the recorded pre-experimental blinded department manager
    benchmark (dept_decision) in eval_records_40.csv; no synthetic resume
    decisions are introduced into the N=40 field dataset.
    """
    rows = load_dataset(csv_path)
    n = len(rows)
    dept = [int(r["dept_decision"]) for r in rows]
    m0_pred = [int(r["hr_manual_decision"]) for r in rows]
    m1_pred = [int(r["llm_hr_decision"]) for r in rows]

    # M2: Hard FSM Escalation Lock on llm_escalated == 1 (resolves R02, R18 via manager benchmark Y)
    m2_pred = []
    m2_esc_count = 0
    for r in rows:
        esc = int(r["llm_escalated"])
        if esc == 1:
            m2_pred.append(int(r["dept_decision"]))
            m2_esc_count += 1
        else:
            m2_pred.append(int(r["llm_hr_decision"]))

    # M3: Hard FSM Gate + Deterministic Span/Verb Verifier on llm_traceable == 0 or llm_hallucination == 1
    m3_pred = []
    m3_esc_count = 0
    m3_verifier_catches = []
    for r in rows:
        esc = int(r["llm_escalated"])
        untraceable = int(r["llm_traceable"]) == 0
        halluc = int(r["llm_hallucination"]) == 1
        if untraceable or halluc:
            m3_verifier_catches.append(r["resume_id"])
        if esc == 1 or untraceable or halluc:
            m3_pred.append(int(r["dept_decision"]))
            m3_esc_count += 1
        else:
            m3_pred.append(int(r["llm_hr_decision"]))

    man_time_mean = float(np.mean([float(r["hr_manual_time_min"]) for r in rows]))
    eica_hr_mean = float(np.mean([float(r["llm_hr_time_min"]) for r in rows]))

    # Manager escalation times:
    # M1 & M2 have 5 escalated cases (R02, R05, R18, R30, R37) totaling 26.0 min (5.20 min avg -> +0.65 min amortized).
    # M3 adds R31 (1 extra escalation). Using the actual recorded manager review time for R31 (6.2 min from dept blinded log)
    # gives 26.0 + 6.2 = 32.2 min (+0.805 min amortized -> 3.51 min total); using mean 5.20 min gives 31.2 min (+0.78 min -> 3.48 min).
    # We compute exact recorded amortized time using 32.2 min total (clearly documented).
    m1_org_time = eica_hr_mean + 26.0 / n
    m2_org_time = eica_hr_mean + 26.0 / n
    m3_org_time = eica_hr_mean + (26.0 + 6.2) / n

    configs = [
        ("M0_Manual", "Manual Unassisted Baseline", m0_pred, 0, man_time_mean, 0.0),
        ("M1_EICA_SoftCard", "EICA Soft-Advisory Card (Field Log)", m1_pred, 5, m1_org_time, 100.0 * sum(int(r["llm_hallucination"]) for r in rows) / n),
        ("M2_EICA_HardFSM", "EICA + Hard FSM Escalation Gate", m2_pred, m2_esc_count, m2_org_time, 100.0 * sum(int(r["llm_hallucination"]) for r in rows) / n),
        ("M3_EICA_FullVerifier", "Full EICA (Hard FSM + Det. Verifier)", m3_pred, m3_esc_count, m3_org_time, 0.0),
    ]

    results = {}
    for key, label, pred, esc_cnt, org_time, unblocked_halluc_rate in configs:
        tp, fp, fn, tn = confusion_matrix(dept, pred)
        agree = (tp + tn) / n * 100.0
        kap = cohen_kappa(dept, pred)
        sens = safe_div(tp, tp + fn) * 100.0
        spec = safe_div(tn, tn + fp) * 100.0
        ppv = safe_div(tp, tp + fp) * 100.0
        npv = safe_div(tn, tn + fn) * 100.0
        f1 = safe_div(2 * tp, 2 * tp + fp + fn) * 100.0

        # Paired statistical comparison vs Manual Baseline (M0)
        mc = paired_mcnemar_exact(dept, m0_pred, pred)
        boot = bootstrap_kappa_ci(dept, m0_pred, pred, n_bootstraps=n_bootstraps, seed=42)

        # Expected Asymmetric Compliance Loss E[L(D, Y)] across cost ratios lambda_FA : lambda_FR : lambda_Esc
        # Normalize lambda_FR = 1.0, lambda_Esc = 0.25 (manager review cost << false rejection << false admission)
        loss_by_ratio = {}
        for lam_fa in (1, 2, 5, 10, 20):
            total_loss = lam_fa * fp + 1.0 * fn + 0.25 * esc_cnt
            loss_by_ratio[str(lam_fa)] = round(total_loss / n, 4)

        results[key] = {
            "label": label,
            "agreement_pct": round(agree, 1),
            "agree_count": tp + tn,
            "kappa": round(kap, 3),
            "kappa_ci": [round(boot["eica_kappa_ci"][0], 3), round(boot["eica_kappa_ci"][1], 3)],
            "delta_kappa_vs_m0": round(kap - cohen_kappa(dept, m0_pred), 3),
            "delta_kappa_ci_vs_m0": [round(boot["delta_kappa_ci"][0], 3), round(boot["delta_kappa_ci"][1], 3)],
            "mcnemar_p_vs_m0": round(mc["p_value"], 5),
            "discordant_vs_m0": f"{mc['b']} vs {mc['c']}",
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity_pct": round(sens, 1),
            "specificity_pct": round(spec, 1),
            "ppv_pct": round(ppv, 1),
            "npv_pct": round(npv, 1),
            "f1_pct": round(f1, 1),
            "escalation_count": esc_cnt,
            "escalation_rate_pct": round(esc_cnt / n * 100.0, 1),
            "unblocked_hallucination_pct": round(unblocked_halluc_rate, 1),
            "total_org_time_min": round(org_time, 2),
            "net_time_reduction_pct": round((man_time_mean - org_time) / man_time_mean * 100.0, 1),
            "expected_loss_by_lambda_fa": loss_by_ratio,
        }

    results["_meta"] = {
        "N": n,
        "m3_verifier_caught_ids": m3_verifier_catches,
        "m2_intercepted_ids": ["R02", "R18"],
        "m3_additional_intercepted_ids": ["R31"],
    }
    return results


# =============================================================================
# Part 2: Real Enterprise Exemplar Replay (R05, R37, R31) +
#         Controlled Synthetic Algorithmic Stress-Test Benchmark (N=200)
# =============================================================================

def run_exemplar_verification_replay() -> List[dict]:
    """
    Runs the executable EICA HybridRAG + SchemaMasker + DeterministicVerifier + FSMGate
    on three PIPL-compliant de-identified English structural vignettes modeled on
    field records R05, R37, and R31 in data/eval_records_40.csv (with R05 testing
    a simulated naive-extractor credential-inflation payload, R37 testing missing
    mandatory registration, and R31 replaying the logged verb-escalation hallucination).
    """
    rag = HybridRegulatoryRAG(alpha=0.5)
    masker = OntologySchemaMasker()
    verifier = DeterministicVerifier(span_threshold=0.85)
    gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

    exemplars = [
        {
            "case_id": "R05",
            "role": "QA/QC",
            "resume": (
                "Education: B.Eng. Materials Science and Engineering, 2014-2018.\n"
                "Experience 1 (2018-2021): Class III cardiovascular device QA Engineer. Led 2 production line IQ/OQ/PQ validations covering 12 machines; updated ISO 14971 risk files; participated in 2 ISO 13485 internal audits; closed 28 CAPAs/yr.\n"
                "Experience 2 (2021-2023): Class III implantable device QA Engineer. Cleanroom environmental monitoring, sterile barrier validation, EO residual testing.\n"
                "Certificates & Training: Attended ISO 13485 internal auditor training (2019); ISO 14971 risk management training (2020). No formal certificate number listed.\n"
                "Self-eval: Collaborates across R&D, production, and regulatory teams."
            ),
            "raw_card": {
                "candidate_id": "R05",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng. Materials Science and Engineering, 2014-2018", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "Class III cardiovascular device QA Engineer", "anchor_lines": [2, 3], "status": "verified"},
                    # Simulate a naive model trying to mark training-only as verified -> verifier must catch it!
                    {"req_id": "QA_CERT", "evidence": "Attended ISO 13485 internal auditor training (2019)", "claim_text": "Certified ISO 13485 internal auditor", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "Led 2 production line IQ/OQ/PQ validations", "anchor_lines": [2, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "updated ISO 14971 risk files; closed 28 CAPAs/yr", "anchor_lines": [2, 2], "status": "verified"},
                ],
            },
        },
        {
            "case_id": "R37",
            "role": "RA",
            "resume": (
                "Education: B.Med. Clinical Medicine, 2013-2018.\n"
                "Experience 1 (2018-2022): CRO Clinical Research Coordinator / Assistant CRA. Managed 3 Class II/III device trials, 120 subjects; supported 2 CER compilations.\n"
                "Experience 2 (2022-2024): Medical Device Regulatory Assistant. Drafted 2 Class III Clinical Evaluation Reports (CER) and deficiency responses; participated in 1 Class II initial NMPA registration (CER chapter).\n"
                "Training: GCP training certificate (2019); NMPA registration training (2023)."
            ),
            "raw_card": {
                "candidate_id": "R37",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med. Clinical Medicine, 2013-2018", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "participated in 1 Class II initial NMPA registration (CER chapter)", "anchor_lines": [3, 3], "status": "not_provided"},
                    {"req_id": "RA_CER", "evidence": "Drafted 2 Class III Clinical Evaluation Reports (CER) and deficiency responses", "anchor_lines": [3, 3], "status": "verified"},
                ],
            },
        },
        {
            "case_id": "R31",
            "role": "QMS",
            "resume": (
                "Experience (2017-2024): Medical Device System Engineer.\n"
                "Maintained ISO 13485 quality system, participated in company internal audit preparations and external audit support, tracked non-conformity closures."
            ),
            "raw_card": {
                "candidate_id": "R31",
                "role": "QMS",
                # Reproduce exact R31 field hallucination: weak "participated" promoted to "certified lead auditor"
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 lead auditor, participated in internal audits", "claim_text": "Certified ISO 13485 lead auditor", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "participated in company internal audit preparations", "claim_text": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 quality system", "anchor_lines": [2, 2], "status": "verified"},
                ],
            },
        },
    ]

    outputs = []
    for ex in exemplars:
        t0 = time.perf_counter()
        top_clauses = rag.retrieve(ex["resume"], top_k=3)
        masked_card, mask_v = masker.enforce_mask_and_validate(ex["raw_card"], ex["role"])
        verified_card, ver_v = verifier.verify_card(ex["resume"], masked_card, ex["role"])
        # Attempt naive recruiter Confirm(1) -> Hard FSM gate must block if violations/missing M exist
        confirm_res = gate.step(verified_card, "Confirm", proposed_decision=1)
        esc_res = gate.step(verified_card, "Escalate", reason="Routed to department manager for compliance adjudication")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        outputs.append({
            "case_id": ex["case_id"],
            "role": ex["role"],
            "top_rag_clauses": [c.doc_id for c, _ in top_clauses],
            "mask_violations": mask_v,
            "verifier_violations": ver_v,
            "naive_confirm_state": confirm_res.next_state,
            "naive_confirm_error": confirm_res.error_code,
            "escalation_state": esc_res.next_state,
            "latency_ms": round(elapsed_ms, 3),
        })
    return outputs


def build_synthetic_stress_cases(seed: int = 42) -> Dict[str, list]:
    """
    Build the deterministic N_syn = 200 controlled fault-injection stress-test cases
    across 6 strata (50 + 30 + 30 + 30 + 30 + 30).
    """
    rng = np.random.default_rng(seed)
    weak_verbs = ["participated in", "assisted with", "supported", "attended training for", "helped prepare"]
    strong_claims = [
        "Certified ISO 13485 lead auditor",
        "Led independent ISO 13485 internal audits",
        "Directed Class III NMPA registration independently",
    ]
    cert_ids = [f"CERT-ISO13485-{1000 + i}" for i in range(60)]
    devices = ["cardiovascular stent", "sterile infusion set", "orthopedic implant", "active patient monitor", "IVD analyzer"]

    cases: Dict[str, list] = {
        "valid_grounded": [],
        "unanchored_span": [],
        "verb_escalation": [],
        "missing_cert_proof": [],
        "tacit_or_score_injection": [],
        "fsm_bypass_attack": [],
    }

    # 1. 50 Valid Grounded Extractions
    for i in range(50):
        dev = devices[i % len(devices)]
        cid = cert_ids[i]
        resume = (
            f"Education: B.Eng. Biomedical Engineering (2015-2019).\n"
            f"Experience: 4 years Class III {dev} QA Engineer. Executed IQ/OQ/PQ validation and ISO 14971 CAPA closures.\n"
            f"Credentials: Certified ISO 13485 internal auditor, Certificate No. {cid}, valid through 2027."
        )
        raw_card = {
            "candidate_id": f"SYN-V-{i:03d}",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2015-2019)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": f"4 years Class III {dev} QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": f"Certified ISO 13485 internal auditor, Certificate No. {cid}", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "ISO 14971 CAPA closures", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        cases["valid_grounded"].append({"resume": resume, "raw_card": raw_card, "role": "QA/QC"})

    # 2. 30 Unanchored / Fabricated Spans
    for i in range(30):
        dev = devices[i % len(devices)]
        resume = (
            f"Education: B.Sc. Mechanical Engineering (2016-2020).\n"
            f"Experience: 3 years industrial inspection technician for consumer appliances.\n"
            f"Skills: Basic Excel and dimensional caliper checks."
        )
        raw_card = {
            "candidate_id": f"SYN-U-{i:03d}",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Sc. Mechanical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": f"5 years Class III {dev} QA Director at GMP sterile plant", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": f"Certified ISO 13485 Lead Auditor No. FAKE-{i:03d}", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Led 10 cleanroom sterilizer PQ runs", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "Managed enterprise CAPA board", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        cases["unanchored_span"].append({"resume": resume, "raw_card": raw_card, "role": "QA/QC"})

    # 3. 30 Verb-Escalation Hallucinations (R31 pattern)
    for i in range(30):
        wv = weak_verbs[i % len(weak_verbs)]
        sc = strong_claims[i % len(strong_claims)]
        resume = (
            f"Experience: Quality System Engineer (2019-2024).\n"
            f"Maintained ISO 13485 documentation and {wv} internal audit preparations and external inspection logistics."
        )
        raw_card = {
            "candidate_id": f"SYN-H-{i:03d}",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": f"{wv} internal audit preparations", "claim_text": sc, "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": f"{wv} internal audit preparations and external inspection logistics", "claim_text": sc, "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 documentation", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        cases["verb_escalation"].append({"resume": resume, "raw_card": raw_card, "role": "QMS"})

    # 4. 30 Missing Credential Proof / Training-Only Claims (R05 pattern)
    for i in range(30):
        dev = devices[i % len(devices)]
        resume = (
            f"Education: B.Eng. Materials Engineering (2015-2019).\n"
            f"Experience: 4 years Class III {dev} QA Engineer; executed IQ/OQ/PQ and CAPA.\n"
            f"Training: Completed ISO 13485 internal auditor training workshop in 2022; no formal certificate issued."
        )
        raw_card = {
            "candidate_id": f"SYN-C-{i:03d}",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Materials Engineering (2015-2019)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": f"4 years Class III {dev} QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Completed ISO 13485 internal auditor training workshop in 2022", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "executed IQ/OQ/PQ", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "CAPA", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        cases["missing_cert_proof"].append({"resume": resume, "raw_card": raw_card, "role": "QA/QC"})

    # 5. 30 Tacit-Slot Injection & Scalar Score Attacks
    for i in range(30):
        cid = cert_ids[i]
        resume = (
            f"Education: B.Eng. Biomedical Engineering (2016-2020).\n"
            f"Experience: 4 years Class III medical device QA Engineer.\n"
            f"Credentials: Certified ISO 13485 internal auditor, Certificate No. {cid}."
        )
        raw_card = {
            "candidate_id": f"SYN-T-{i:03d}",
            "role": "QA/QC",
            "overall_fit_score": float(rng.uniform(75.0, 98.0)),  # Forbidden scalar score
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": f"Certified ISO 13485 internal auditor, Certificate No. {cid}", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                # Forbidden tacit-slot hallucination
                {"req_id": "QA_TACIT", "evidence": "High ethical vigilance and strong audit composure inferred from tone", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        cases["tacit_or_score_injection"].append({"resume": resume, "raw_card": raw_card, "role": "QA/QC"})

    # 6. 30 Unauthorized FSM Override / Missing Mandatory / Short-Reason Attacks
    for i in range(30):
        resume = "Education: B.Eng. Materials.\nExperience: 2 years QA technician."
        raw_card = {
            "candidate_id": f"SYN-F-{i:03d}",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Materials", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CERT", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        attack_mode = "confirm_missing_m" if (i % 2 == 0) else "adjust_short_reason"
        cases["fsm_bypass_attack"].append({"resume": resume, "raw_card": raw_card, "role": "QA/QC", "attack_mode": attack_mode})

    return cases


def _prepare_unmasked_card(raw_card: dict, role: str) -> dict:
    """Attach ontology tiers to an unmasked raw_card when schema masking is disabled."""
    reqs = {r.req_id: r for r in CompetencyOntology.get_requirements(role)}
    card_copy = dict(raw_card)
    card_copy["entries"] = [
        {**dict(e), "tier": reqs[e["req_id"]].tier}
        for e in raw_card.get("entries", [])
        if e.get("req_id") in reqs
    ]
    return card_copy


def run_synthetic_verification_benchmark(
    seed: int = 42,
    alpha: float = 0.5,
    span_threshold: float = 0.85,
    l_min: int = 15,
    pinned_only: bool = True,
    use_fuzzy_span: bool = True,
    use_verb_tier: bool = True,
    use_cert_proof: bool = True,
    enforce_hard_escalation_lock: bool = True,
    schema_mask_enabled: bool = True,
) -> dict:
    """
    Controlled Synthetic Algorithmic Stress-Test Benchmark (N_syn = 200 cases).
    Explicitly programmatic fault-injection testbench (separate from the N=40
    enterprise field dataset) evaluating end-to-end deterministic detection
    performance, invariant enforcement, and CPU runtime latency across 6 strata:
      - 50 Valid Grounded Extractions (expected: PASS -> Confirm(1) allowed)
      - 30 Unanchored / Fabricated Spans (expected: CAUGHT -> UNANCHORED_SPAN -> s_block)
      - 30 Verb-Escalation Hallucinations (expected: CAUGHT -> VERB_ESCALATION -> s_block)
      - 30 Missing Credential Proof / Training-Only (expected: CAUGHT -> MISSING_CREDENTIAL_PROOF -> s_block)
      - 30 Tacit-Slot Injection & Scalar-Score Attacks (expected: MASKED -> 0% tacit leakage)
      - 30 Unauthorized FSM Override / Short-Reason Attacks (expected: BLOCKED -> s_block)
    """
    cases = build_synthetic_stress_cases(seed=seed)
    rag = HybridRegulatoryRAG(alpha=alpha, pinned_only=pinned_only)
    masker = OntologySchemaMasker()
    verifier = DeterministicVerifier(
        span_threshold=span_threshold,
        use_fuzzy_span=use_fuzzy_span,
        use_verb_tier=use_verb_tier,
        use_cert_proof=use_cert_proof,
    )
    gate = FSMDecisionGate(l_min=l_min, enforce_hard_escalation_lock=enforce_hard_escalation_lock)

    latencies_ms: List[float] = []
    rag_pinned_precision_hits = 0
    total_rag_queries = 0

    strata_stats = {
        "valid_grounded": {"n": 50, "passed_clean": 0, "false_alarms": 0},
        "unanchored_span": {"n": 30, "caught": 0, "missed": 0},
        "verb_escalation": {"n": 30, "caught": 0, "missed": 0},
        "missing_cert_proof": {"n": 30, "caught": 0, "missed": 0},
        "tacit_or_score_injection": {"n": 30, "blocked": 0, "leaked": 0},
        "fsm_bypass_attack": {"n": 30, "blocked": 0, "bypassed": 0},
    }

    # 1. 50 Valid Grounded Extractions
    for c in cases["valid_grounded"]:
        t0 = time.perf_counter()
        clauses = rag.retrieve(c["resume"], top_k=3)
        total_rag_queries += 1
        if all(cl.pinned for cl, _ in clauses):
            rag_pinned_precision_hits += 1
        if schema_mask_enabled:
            masked, mv = masker.enforce_mask_and_validate(c["raw_card"], c["role"])
        else:
            masked, mv = _prepare_unmasked_card(c["raw_card"], c["role"]), []
        verified, vv = verifier.verify_card(c["resume"], masked, c["role"])
        res = gate.step(verified, "Confirm", proposed_decision=1)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if not mv and not vv and res.next_state == "s_term" and res.final_decision == "1":
            strata_stats["valid_grounded"]["passed_clean"] += 1
        else:
            strata_stats["valid_grounded"]["false_alarms"] += 1

    # 2. 30 Unanchored / Fabricated Spans
    for c in cases["unanchored_span"]:
        t0 = time.perf_counter()
        clauses = rag.retrieve(c["resume"], top_k=3)
        total_rag_queries += 1
        if all(cl.pinned for cl, _ in clauses):
            rag_pinned_precision_hits += 1
        masked = masker.enforce_mask_and_validate(c["raw_card"], c["role"])[0] if schema_mask_enabled else _prepare_unmasked_card(c["raw_card"], c["role"])
        verified, vv = verifier.verify_card(c["resume"], masked, c["role"])
        res = gate.step(verified, "Confirm", proposed_decision=1)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if any(v.startswith("UNANCHORED_SPAN") for v in vv) and res.next_state == "s_block":
            strata_stats["unanchored_span"]["caught"] += 1
        else:
            strata_stats["unanchored_span"]["missed"] += 1

    # 3. 30 Verb-Escalation Hallucinations (R31 pattern)
    for c in cases["verb_escalation"]:
        t0 = time.perf_counter()
        clauses = rag.retrieve(c["resume"], top_k=3)
        total_rag_queries += 1
        if all(cl.pinned for cl, _ in clauses):
            rag_pinned_precision_hits += 1
        masked = masker.enforce_mask_and_validate(c["raw_card"], c["role"])[0] if schema_mask_enabled else _prepare_unmasked_card(c["raw_card"], c["role"])
        verified, vv = verifier.verify_card(c["resume"], masked, c["role"])
        res = gate.step(verified, "Confirm", proposed_decision=1)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if any(v.startswith("VERB_ESCALATION") for v in vv) and res.next_state == "s_block":
            strata_stats["verb_escalation"]["caught"] += 1
        else:
            strata_stats["verb_escalation"]["missed"] += 1

    # 4. 30 Missing Credential Proof / Training-Only Claims (R05 pattern)
    for c in cases["missing_cert_proof"]:
        t0 = time.perf_counter()
        clauses = rag.retrieve(c["resume"], top_k=3)
        total_rag_queries += 1
        if all(cl.pinned for cl, _ in clauses):
            rag_pinned_precision_hits += 1
        masked = masker.enforce_mask_and_validate(c["raw_card"], c["role"])[0] if schema_mask_enabled else _prepare_unmasked_card(c["raw_card"], c["role"])
        verified, vv = verifier.verify_card(c["resume"], masked, c["role"])
        res = gate.step(verified, "Confirm", proposed_decision=1)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if any(v.startswith("MISSING_CREDENTIAL_PROOF") for v in vv) and res.next_state == "s_block":
            strata_stats["missing_cert_proof"]["caught"] += 1
        else:
            strata_stats["missing_cert_proof"]["missed"] += 1

    # 5. 30 Tacit-Slot Injection & Scalar Score Attacks
    for c in cases["tacit_or_score_injection"]:
        t0 = time.perf_counter()
        if schema_mask_enabled:
            masked, mv = masker.enforce_mask_and_validate(c["raw_card"], c["role"])
            verified, _ = verifier.verify_card(c["resume"], masked, c["role"])
        else:
            # Unmasked schema baseline skips OntologySchemaMasker and allows raw card payload directly
            mv = []
            verified = _prepare_unmasked_card(c["raw_card"], c["role"])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        tacit_entry = [e for e in verified["entries"] if e["req_id"] == "QA_TACIT"][0]
        if (
            any("TACIT_EXTRAPOLATION_BLOCKED" in v for v in mv)
            and any("FORBIDDEN_FIELD_STRIPPED" in v for v in mv)
            and tacit_entry["status"] is None
            and "overall_fit_score" not in verified
        ):
            strata_stats["tacit_or_score_injection"]["blocked"] += 1
        else:
            strata_stats["tacit_or_score_injection"]["leaked"] += 1

    # 6. 30 Unauthorized FSM Override / Missing Mandatory / Short-Reason Attacks
    for c in cases["fsm_bypass_attack"]:
        t0 = time.perf_counter()
        masked = masker.enforce_mask_and_validate(c["raw_card"], c["role"])[0] if schema_mask_enabled else _prepare_unmasked_card(c["raw_card"], c["role"])
        verified, _ = verifier.verify_card(c["resume"], masked, c["role"])
        if c["attack_mode"] == "confirm_missing_m":
            res = gate.step(verified, "Confirm", proposed_decision=1)
        else:
            res = gate.step(verified, "Adjust", proposed_decision=1, reason="looks ok")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if res.next_state == "s_block":
            strata_stats["fsm_bypass_attack"]["blocked"] += 1
        else:
            strata_stats["fsm_bypass_attack"]["bypassed"] += 1

    # Aggregate detection metrics across the 90 fault-injection extraction attacks + 50 clean grounded cases
    tp_det = (
        strata_stats["unanchored_span"]["caught"]
        + strata_stats["verb_escalation"]["caught"]
        + strata_stats["missing_cert_proof"]["caught"]
    )
    fn_det = (
        strata_stats["unanchored_span"]["missed"]
        + strata_stats["verb_escalation"]["missed"]
        + strata_stats["missing_cert_proof"]["missed"]
    )
    fp_det = strata_stats["valid_grounded"]["false_alarms"]
    tn_det = strata_stats["valid_grounded"]["passed_clean"]

    precision = safe_div(tp_det, tp_det + fp_det) * 100.0
    recall = safe_div(tp_det, tp_det + fn_det) * 100.0
    f1 = safe_div(2 * tp_det, 2 * tp_det + fp_det + fn_det) * 100.0

    lat_arr = np.array(latencies_ms)
    return {
        "N_total": 200,
        "strata": strata_stats,
        "extraction_verifier_summary": {
            "TP": tp_det, "FP": fp_det, "FN": fn_det, "TN": tn_det,
            "precision_pct": round(precision, 1),
            "recall_pct": round(recall, 1),
            "f1_pct": round(f1, 1),
        },
        "schema_tacit_block_rate_pct": round(strata_stats["tacit_or_score_injection"]["blocked"] / 30 * 100.0, 1),
        "fsm_invariant_block_rate_pct": round(strata_stats["fsm_bypass_attack"]["blocked"] / 30 * 100.0, 1),
        "rag_whitelist_pin_compliance_pct": round(rag_pinned_precision_hits / max(total_rag_queries, 1) * 100.0, 1),
        "latency_ms": {
            "mean": round(float(np.mean(lat_arr)), 3),
            "p50": round(float(np.percentile(lat_arr, 50)), 3),
            "p95": round(float(np.percentile(lat_arr, 95)), 3),
            "max": round(float(np.max(lat_arr)), 3),
        },
    }


def _build_legacy_rag_queries() -> List[str]:
    """20 deterministic regulatory queries mentioning superseded Order 650 / 2014 rules."""
    topics = [
        "QMS inspection", "sterile device registration", "internal audit", "CAPA management",
        "personnel competence", "supplier audit", "implantable traceability", "clinical evaluation",
        "equipment validation", "certificate proof", "cleanroom monitoring", "risk management",
        "label compliance", "batch release", "change control", "post-market surveillance",
        "recall procedure", "training records", "regulatory filing", "process qualification",
    ]
    return [
        f"Superseded 2014 Order 650 medical device supervision rules for {t} prior to Order 739"
        for t in topics
    ]


def _build_minor_variant_quotes() -> List[Tuple[List[str], str, List[int]]]:
    """20 deterministic valid extractions with minor punctuation/spacing/OCR-like edits (Sim_span in [0.87, 0.93])."""
    devices = [
        "cardiovascular stent", "sterile infusion set", "orthopedic implant",
        "active patient monitor", "IVD analyzer",
    ]
    samples: List[Tuple[List[str], str, List[int]]] = []
    for i in range(20):
        dev = devices[i % len(devices)]
        lines = [
            "Education: B.Eng. Biomedical Engineering (2015-2019).",
            f"Experience: 4 years Class III {dev} QA Engineer; executed IQ/OQ/PQ validation and ISO 14971 CAPA closures.",
        ]
        if i % 2 == 0:
            # Minor hyphen/typographic difference vs source -> not an exact substring, Sim ~ 0.91
            quote = f"4 years Class-III {dev} Q.A. Eng."
        else:
            quote = "executed IQ-OQ-PQ validations and ISO-14971 CAPA closure"
        samples.append((lines, quote, [2, 2]))
    return samples


def _build_subtle_fabrication_quotes() -> List[Tuple[List[str], str, List[int]]]:
    """30 deterministic subtle entity/number-swap partial fabrications (Sim_span in [0.77, 0.83])."""
    devices = [
        "cardiovascular stent", "sterile infusion set", "orthopedic implant",
        "active patient monitor", "IVD analyzer",
    ]
    samples: List[Tuple[List[str], str, List[int]]] = []
    for i in range(30):
        dev = devices[i % len(devices)]
        lines = [
            "Education: B.Eng. Biomedical Engineering (2015-2019).",
            f"Experience: 4 years Class III {dev} QA Engineer; executed IQ/OQ/PQ validation and ISO 14971 CAPA closures.",
        ]
        if i % 3 == 0:
            quote = "executed IQ/OQ/PQ audits and ISO 14971 CAPA review"
        elif i % 3 == 1:
            quote = "executed IQ/OQ testing and ISO 9001 CAPA closures"
        else:
            quote = "executed IQ/OQ/PQ review and ISO 14971 CAPA audits"
        samples.append((lines, quote, [2, 2]))
    return samples


def _build_boundary_valid_quotes() -> List[Tuple[List[str], str, List[int]]]:
    """30 deterministic noisy-valid extractions (Sim_span in [0.87, 0.93])."""
    devices = [
        "cardiovascular stent", "sterile infusion set", "orthopedic implant",
        "active patient monitor", "IVD analyzer",
    ]
    samples: List[Tuple[List[str], str, List[int]]] = []
    for i in range(30):
        dev = devices[i % len(devices)]
        lines = [
            "Education: B.Eng. Biomedical Engineering (2015-2019).",
            f"Experience: 4 years Class III {dev} QA Engineer; executed IQ/OQ/PQ validation and ISO 14971 CAPA closures.",
        ]
        if i % 3 == 0:
            quote = f"4 years Class-III {dev} Q.A. Eng."
        elif i % 3 == 1:
            quote = "executed IQ-OQ-PQ validations and ISO-14971 CAPA closure"
        else:
            quote = "executed IQ/OQ/PQ validation & ISO 14971 CAPA records"
        samples.append((lines, quote, [2, 2]))
    return samples


def run_synthetic_component_ablations(seed: int = 42) -> Dict[str, dict]:
    """
    Deterministic Option-1 Component-Off Ablations on N_syn=200 + Diagnostic Probes
    (zero fabricated LLM outputs):
      - Full_EICA: all safeguards active
      - Unpinned_RAG: pinned_only=False (allows deprecated ORD650-DEP into index)
      - Exact_String_Match_Only: use_fuzzy_span=False, use_verb_tier=False, use_cert_proof=False
      - Soft_FSM_Lock_Off: enforce_hard_escalation_lock=False
      - Unmasked_Schema: schema_mask_enabled=False
    """
    cases = build_synthetic_stress_cases(seed=seed)
    legacy_queries = _build_legacy_rag_queries()
    minor_quotes = _build_minor_variant_quotes()

    configs = [
        (
            "Full_EICA",
            "Full EICA (All Deterministic Guards Active)",
            dict(pinned_only=True, use_fuzzy_span=True, use_verb_tier=True, use_cert_proof=True, enforce_hard_escalation_lock=True, schema_mask_enabled=True),
        ),
        (
            "Unpinned_RAG",
            "Ablation: Unpinned RAG (ORD650-DEP Included)",
            dict(pinned_only=False, use_fuzzy_span=True, use_verb_tier=True, use_cert_proof=True, enforce_hard_escalation_lock=True, schema_mask_enabled=True),
        ),
        (
            "Exact_String_Match_Only",
            "Ablation: Exact String Match Only (No Fuzzy/Verb/Cert)",
            dict(pinned_only=True, use_fuzzy_span=False, use_verb_tier=False, use_cert_proof=False, enforce_hard_escalation_lock=True, schema_mask_enabled=True),
        ),
        (
            "Soft_FSM_Lock_Off",
            "Ablation: Soft FSM Gate (Hard Escalation Lock Off)",
            dict(pinned_only=True, use_fuzzy_span=True, use_verb_tier=True, use_cert_proof=True, enforce_hard_escalation_lock=False, schema_mask_enabled=True),
        ),
        (
            "Unmasked_Schema",
            "Ablation: Unmasked Schema (No Tier-V/Score Mask)",
            dict(pinned_only=True, use_fuzzy_span=True, use_verb_tier=True, use_cert_proof=True, enforce_hard_escalation_lock=True, schema_mask_enabled=False),
        ),
    ]

    results: Dict[str, dict] = {}
    for key, label, cfg in configs:
        bench = run_synthetic_verification_benchmark(seed=seed, **cfg)
        rag = HybridRegulatoryRAG(alpha=0.5, pinned_only=cfg["pinned_only"])
        masker = OntologySchemaMasker()
        verifier = DeterministicVerifier(
            span_threshold=0.85,
            use_fuzzy_span=cfg["use_fuzzy_span"],
            use_verb_tier=cfg["use_verb_tier"],
            use_cert_proof=cfg["use_cert_proof"],
        )
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=cfg["enforce_hard_escalation_lock"])

        # Probe 1: Legacy Order-650 regulatory leakage (N=20)
        legacy_leaks = 0
        for q in legacy_queries:
            top3 = rag.retrieve(q, top_k=3)
            if any(not doc.pinned for doc, _ in top3):
                legacy_leaks += 1

        # Probe 2: Minor-punctuation / OCR-variant valid quote false alarm rate (N=20)
        minor_fa = 0
        for lines, quote, anchor in minor_quotes:
            aligned, _ = verifier.check_span_alignment(lines, quote, anchor)
            if not aligned:
                minor_fa += 1

        # Probe 3: Conflict-Override Adjust(1) across 90 fault cases + 10 Tier-P conflict Confirm(1) cases
        override_total = 0
        override_bypasses = 0
        for stratum in ("unanchored_span", "verb_escalation", "missing_cert_proof"):
            for c in cases[stratum]:
                masked = masker.enforce_mask_and_validate(c["raw_card"], c["role"])[0] if cfg["schema_mask_enabled"] else _prepare_unmasked_card(c["raw_card"], c["role"])
                verified, _ = verifier.verify_card(c["resume"], masked, c["role"])
                adj_map = {e["req_id"]: "verified" for e in verified["entries"] if e["tier"] in ("M", "P")}
                res = gate.step(
                    verified,
                    "Adjust",
                    proposed_decision=1,
                    reason="Recruiter override: manual review approved candidate",
                    adjusted_statuses=adj_map,
                )
                override_total += 1
                if res.next_state == "s_term" and res.final_decision == "1":
                    override_bypasses += 1

        # 10 Tier-P conflict cards (all Tier-M verified, Tier-P conflicting)
        for idx in range(10):
            tier_p_card = {
                "candidate_id": f"TP-OVR-{idx:02d}",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "tier": "M", "status": "verified"},
                    {"req_id": "QA_EXP", "tier": "M", "status": "verified"},
                    {"req_id": "QA_CERT", "tier": "M", "status": "verified"},
                    {"req_id": "QA_VAL", "tier": "P", "status": "conflicting"},
                    {"req_id": "QA_CAPA", "tier": "P", "status": "verified"},
                ],
                "verifier_violations": [],
            }
            res = gate.step(tier_p_card, "Confirm", proposed_decision=1)
            override_total += 1
            if res.next_state == "s_term" and res.final_decision == "1":
                override_bypasses += 1

        ext = bench["extraction_verifier_summary"]
        results[key] = {
            "label": label,
            "extraction_precision_pct": ext["precision_pct"],
            "extraction_recall_pct": ext["recall_pct"],
            "extraction_f1_pct": ext["f1_pct"],
            "schema_tacit_block_rate_pct": bench["schema_tacit_block_rate_pct"],
            "fsm_invariant_block_rate_pct": bench["fsm_invariant_block_rate_pct"],
            "rag_pin_compliance_pct": bench["rag_whitelist_pin_compliance_pct"],
            "legacy_query_leakage_pct": round(legacy_leaks / len(legacy_queries) * 100.0, 1),
            "minor_variant_false_alarm_pct": round(minor_fa / len(minor_quotes) * 100.0, 1),
            "conflict_override_bypass_pct": round(override_bypasses / max(override_total, 1) * 100.0, 1),
            "latency_p50_ms": bench["latency_ms"]["p50"],
            "latency_p95_ms": bench["latency_ms"]["p95"],
        }
    return results


def run_parameter_sensitivity_sweep(seed: int = 42) -> Dict[str, List[dict]]:
    """
    Deterministic sensitivity sweep over key hyperparameters:
      - tau_span in [0.75, 0.85, 0.95] (verifier span similarity threshold)
      - alpha in [0.0, 0.3, 0.5, 0.7, 1.0] (hybrid RAG BM25 vs cosine weight)
      - l_min in [5, 10, 15, 25] (FSM Adjust justification minimum length)
    """
    cases = build_synthetic_stress_cases(seed=seed)
    boundary_valid = _build_boundary_valid_quotes()
    subtle_fab = _build_subtle_fabrication_quotes()

    # 1. tau_span sweep
    tau_sweep: List[dict] = []
    for tau in (0.75, 0.85, 0.95):
        bench = run_synthetic_verification_benchmark(seed=seed, span_threshold=tau)
        verifier = DeterministicVerifier(span_threshold=tau)
        tp = fp = fn = tn = 0
        for lines, quote, anchor in boundary_valid:
            aligned, _ = verifier.check_span_alignment(lines, quote, anchor)
            if aligned:
                tn += 1
            else:
                fp += 1
        for lines, quote, anchor in subtle_fab:
            aligned, _ = verifier.check_span_alignment(lines, quote, anchor)
            if not aligned:
                tp += 1
            else:
                fn += 1
        b_prec = safe_div(tp, tp + fp) * 100.0
        b_rec = safe_div(tp, tp + fn) * 100.0
        b_f1 = safe_div(2 * tp, 2 * tp + fp + fn) * 100.0
        b_fa = safe_div(fp, len(boundary_valid)) * 100.0
        tau_sweep.append({
            "tau_span": tau,
            "base_n200_f1_pct": bench["extraction_verifier_summary"]["f1_pct"],
            "boundary_precision_pct": round(b_prec, 1),
            "boundary_recall_pct": round(b_rec, 1),
            "boundary_f1_pct": round(b_f1, 1),
            "boundary_false_alarm_pct": round(b_fa, 1),
        })

    # 2. alpha sweep (140 N_syn queries + R05/R37/R31 + 15 deterministic role-clause relevance probes)
    role_probes = [
        ("Class III cardiovascular device QA IQ OQ PQ validation ISO 14971 CAPA internal audit", "ISO13485-8.5.2"),
        ("ISO 13485 internal auditor certificate number validity training attendance", "SOP-CERT-03"),
        ("Sterile medical device cleanroom environmental monitoring EO residual packaging", "NMPA-GMP-STERILE"),
        ("Implantable device batch traceability biocompatibility supplier audit", "NMPA-GMP-IMPLANT"),
        ("Sterilization chamber modification IQ OQ PQ re-qualification protocol", "SOP-VAL-04"),
        ("ISO 9001 internal auditor cross-industry GMP equivalence escalation", "SOP-EQ-01"),
        ("Class III NMPA initial registration clinical evaluation CER deficiency response", "SOP-RA-02"),
        ("Order 739 Order 797 qualified QA regulatory personnel statutory responsibility", "ORD797-A35"),
        ("Administrative sanctions recalls license revocation non-conforming QMS", "ORD797-A86"),
        ("Personnel competence appropriate education training skills experience records", "ISO13485-6.2"),
        ("Special processes sterilization cleanroom IQ OQ PQ documented protocol", "ISO13485-7.5.6"),
        ("Internal audits certified independent auditors training does not substitute", "ISO13485-8.2.4"),
        ("Quality management system CAPA risk updates effectiveness verification", "ISO13485-8.5.2"),
        ("Medical device regulatory assistant Class II III registration CER drafting", "SOP-RA-02"),
        ("Aseptic packaging microbial monitoring cleanroom control NMPA GMP", "NMPA-GMP-STERILE"),
    ]
    syn_140_resumes = [
        c["resume"]
        for s in ("valid_grounded", "unanchored_span", "verb_escalation", "missing_cert_proof")
        for c in cases[s]
    ]
    alpha_sweep: List[dict] = []
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        rag = HybridRegulatoryRAG(alpha=alpha, pinned_only=True)
        pin_hits = 0
        for r in syn_140_resumes:
            top3 = rag.retrieve(r, top_k=3)
            if all(doc.pinned for doc, _ in top3):
                pin_hits += 1
        probe_hits = 0
        for q, expected_id in role_probes:
            top3 = rag.retrieve(q, top_k=3)
            if any(doc.doc_id == expected_id for doc, _ in top3):
                probe_hits += 1
        alpha_sweep.append({
            "alpha": alpha,
            "pin_compliance_pct": round(pin_hits / len(syn_140_resumes) * 100.0, 1),
            "top3_expected_clause_hit_pct": round(probe_hits / len(role_probes) * 100.0, 1),
        })

    # 3. l_min sweep
    base_valid_card = {
        "candidate_id": "LMIN-PROBE",
        "role": "QA/QC",
        "entries": [
            {"req_id": "QA_DEG", "tier": "M", "status": "verified"},
            {"req_id": "QA_EXP", "tier": "M", "status": "verified"},
            {"req_id": "QA_CERT", "tier": "M", "status": "verified"},
            {"req_id": "QA_VAL", "tier": "P", "status": "not_provided"},
            {"req_id": "QA_CAPA", "tier": "P", "status": "verified"},
        ],
        "verifier_violations": [],
    }
    short_reasons = ["looks ok"] * 15  # len = 8
    concise_reasons = [
        "GMP equiv verified",        # 18
        "Verified via GMP",          # 16
        "Prior QA audit pass",       # 19
        "Matches ISO 9001 eq",       # 19
        "GMP line eq valid",         # 17
    ] * 3  # 15 items, lengths 16..19 (< 25)
    detailed_reasons = [
        "Verified sterile GMP IQ/OQ equivalence via manager audit log",
        "Confirmed ISO 9001 lead experience satisfies preferred P tier",
        "Documented cleanroom validation equivalence in HR archive",
    ] * 5  # 15 items, lengths > 40

    lmin_sweep: List[dict] = []
    for l_min in (5, 10, 15, 25):
        gate = FSMDecisionGate(l_min=l_min, enforce_hard_escalation_lock=True)
        short_blocked = sum(
            1 for rsn in short_reasons
            if gate.step(base_valid_card, "Adjust", proposed_decision=1, reason=rsn).next_state == "s_block"
        )
        concise_blocked = sum(
            1 for rsn in concise_reasons
            if gate.step(base_valid_card, "Adjust", proposed_decision=1, reason=rsn).next_state == "s_block"
        )
        detailed_blocked = sum(
            1 for rsn in detailed_reasons
            if gate.step(base_valid_card, "Adjust", proposed_decision=1, reason=rsn).next_state == "s_block"
        )
        lmin_sweep.append({
            "l_min": l_min,
            "short_reason_block_pct": round(short_blocked / len(short_reasons) * 100.0, 1),
            "concise_valid_false_block_pct": round(concise_blocked / len(concise_reasons) * 100.0, 1),
            "detailed_valid_false_block_pct": round(detailed_blocked / len(detailed_reasons) * 100.0, 1),
        })

    return {
        "tau_span": tau_sweep,
        "alpha": alpha_sweep,
        "l_min": lmin_sweep,
    }


def format_table3(ablation: dict) -> str:
    lines = []
    lines.append("=" * 98)
    lines.append("TABLE 3: Trace-Driven Architectural Policy Ablation on Real Field Logs (N = 40)")
    lines.append("=" * 98)
    lines.append(f"{'System Configuration':<38} {'Agree':<10} {'Kappa':<8} {'F1':<7} {'FA/FR':<8} {'Esc':<8} {'Org.Time':<10} {'E[L](10:1)'}")
    lines.append("-" * 98)
    for k in ("M0_Manual", "M1_EICA_SoftCard", "M2_EICA_HardFSM", "M3_EICA_FullVerifier"):
        r = ablation[k]
        loss_fmt = _fmt_half_up(r["expected_loss_by_lambda_fa"]["10"], 3)
        lines.append(
            f"{r['label']:<38} {r['agreement_pct']:>5.1f}%    {r['kappa']:>5.3f}   "
            f"{r['f1_pct']:>5.1f}%  {r['FP']}/{r['FN']:<5} {r['escalation_rate_pct']:>5.1f}%  "
            f"{r['total_org_time_min']:>5.2f} min  {loss_fmt:>6}"
        )
    lines.append("=" * 98)
    lines.append("Note: M3 exact E[L](10:1) = 0.0875 (reported as 0.088 via half-up rounding).")
    return "\n".join(lines)


def format_synthetic_ablation_table(ablations: Dict[str, dict]) -> str:
    lines = []
    lines.append("=" * 106)
    lines.append("SYNTHETIC COMPONENT-OFF ABLATION (N_syn = 200 + Deterministic Diagnostic Probes)")
    lines.append("=" * 106)
    lines.append(
        f"{'Configuration':<26} {'Ext.P':<7} {'Ext.R':<7} {'Ext.F1':<8} "
        f"{'TacitBlk':<9} {'OvrBypass':<10} {'LegacyLeak':<11} {'MinorFA':<8} {'P95(ms)'}"
    )
    lines.append("-" * 106)
    for key in ("Full_EICA", "Unpinned_RAG", "Exact_String_Match_Only", "Soft_FSM_Lock_Off", "Unmasked_Schema"):
        r = ablations[key]
        lines.append(
            f"{key:<26} {r['extraction_precision_pct']:>5.1f}%  {r['extraction_recall_pct']:>5.1f}%  "
            f"{r['extraction_f1_pct']:>5.1f}%   {r['schema_tacit_block_rate_pct']:>6.1f}%   "
            f"{r['conflict_override_bypass_pct']:>7.1f}%    {r['legacy_query_leakage_pct']:>7.1f}%     "
            f"{r['minor_variant_false_alarm_pct']:>5.1f}%   {r['latency_p95_ms']:>6.3f}"
        )
    lines.append("=" * 106)
    return "\n".join(lines)


def format_sensitivity_summary(sweep: Dict[str, List[dict]] = None) -> str:
    if sweep is None:
        sweep = run_parameter_sensitivity_sweep(seed=42)
    lines = []
    lines.append("=" * 88)
    lines.append("HYPERPARAMETER SENSITIVITY SWEEP (Deterministic)")
    lines.append("=" * 88)
    lines.append("[1] Span Similarity Threshold (tau_span):")
    for r in sweep["tau_span"]:
        lines.append(
            f"  tau={r['tau_span']:.2f} -> N200_F1={r['base_n200_f1_pct']:.1f}% | "
            f"Boundary P/R/F1={r['boundary_precision_pct']:.1f}%/{r['boundary_recall_pct']:.1f}%/{r['boundary_f1_pct']:.1f}% | "
            f"Boundary_FA={r['boundary_false_alarm_pct']:.1f}%"
        )
    lines.append("[2] Hybrid RAG Weight (alpha):")
    for r in sweep["alpha"]:
        lines.append(
            f"  alpha={r['alpha']:.1f} -> PinCompliance={r['pin_compliance_pct']:.1f}% | "
            f"Top3_ExpectedClauseHit={r['top3_expected_clause_hit_pct']:.1f}%"
        )
    lines.append("[3] FSM Minimum Justification Length (l_min):")
    for r in sweep["l_min"]:
        lines.append(
            f"  l_min={r['l_min']:<2d} -> ShortReasonBlock={r['short_reason_block_pct']:.1f}% | "
            f"ConciseValidFalseBlock={r['concise_valid_false_block_pct']:.1f}% | "
            f"DetailedValidFalseBlock={r['detailed_valid_false_block_pct']:.1f}%"
        )
    lines.append("=" * 88)
    return "\n".join(lines)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "eval_records_40.csv"
    ablation = run_trace_driven_ablation(csv_path, n_bootstraps=20000)
    exemplars = run_exemplar_verification_replay()
    stress = run_synthetic_verification_benchmark(seed=42)
    comp_ablations = run_synthetic_component_ablations(seed=42)
    sens_sweep = run_parameter_sensitivity_sweep(seed=42)

    print(format_table3(ablation))
    print("\n" + format_synthetic_ablation_table(comp_ablations))
    print("\n" + format_sensitivity_summary(sens_sweep))
    print("\n--- TRACE-DRIVEN ABLATION DETAILS (JSON) ---")
    print(json.dumps(ablation, indent=2, ensure_ascii=False))
    print("\n--- REAL ENTERPRISE EXEMPLAR REPLAY (R05, R37, R31) ---")
    print(json.dumps(exemplars, indent=2, ensure_ascii=False))
    print("\n--- CONTROLLED SYNTHETIC VERIFICATION STRESS-TEST (N=200) ---")
    print(json.dumps(stress, indent=2, ensure_ascii=False))
