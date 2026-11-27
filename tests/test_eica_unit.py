# -*- coding: utf-8 -*-
"""
Unit Tests for Hardened EICA Engine, Metrics, Statistical Tests, and Ablations
Run from `code/` directory:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = CODE_DIR / "src"
DATA_DIR = CODE_DIR / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eica_engine import (
    CompetencyOntology,
    DeterministicVerifier,
    FSMDecisionGate,
    HybridRegulatoryRAG,
    OntologySchemaMasker,
)
from metrics_calculator import compute_metrics, format_table2, round_half_up
from statistical_tests import bootstrap_kappa_ci, paired_mcnemar_exact
from ablation_and_benchmark import (
    format_table3,
    run_exemplar_verification_replay,
    run_parameter_sensitivity_sweep,
    run_synthetic_component_ablations,
    run_synthetic_verification_benchmark,
    run_trace_driven_ablation,
)


class TestOntologyAndSchemaMasker(unittest.TestCase):
    def test_compile_schema_omits_tier_v_all_roles(self):
        for role in ("QA/QC", "QMS", "RA"):
            schema = OntologySchemaMasker.compile_generation_schema(role)
            reqs = CompetencyOntology.get_requirements(role)
            expected_mp = [r.req_id for r in reqs if r.tier in ("M", "P")]
            expected_v = [r.req_id for r in reqs if r.tier == "V"]
            req_enum = schema["properties"]["entries"]["items"]["properties"]["req_id"]["enum"]
            self.assertEqual(req_enum, expected_mp)
            self.assertEqual(schema["masked_tacit_slots"], expected_v)
            for v_id in expected_v:
                self.assertNotIn(v_id, req_enum)

    def test_enforce_mask_strips_scalar_scores_and_blocks_tacit(self):
        raw_card = {
            "candidate_id": "T-001",
            "role": "QA/QC",
            "overall_fit_score": 95.0,
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_TACIT", "evidence": "High composure", "anchor_lines": [1, 1], "status": "verified"},
            ],
        }
        clean, violations = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        self.assertNotIn("overall_fit_score", clean)
        self.assertTrue(any("FORBIDDEN_FIELD_STRIPPED:overall_fit_score" in v for v in violations))
        self.assertTrue(any("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT" in v for v in violations))
        tacit = [e for e in clean["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tacit["status"])
        self.assertEqual(tacit["evidence"], "")

    def test_missing_req_filled_not_provided_and_invalid_status_conflicting(self):
        raw_card = {
            "candidate_id": "T-002",
            "role": "RA",
            "entries": [
                {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "maybe"},
            ],
        }
        clean, violations = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "RA")
        by_id = {e["req_id"]: e for e in clean["entries"]}
        self.assertEqual(by_id["RA_DEG"]["status"], "conflicting")
        self.assertEqual(by_id["RA_REG"]["status"], "not_provided")
        self.assertEqual(by_id["RA_CER"]["status"], "not_provided")
        self.assertIsNone(by_id["RA_TACIT"]["status"])
        self.assertTrue(any("INVALID_ENUM_STATUS:RA_DEG:maybe" in v for v in violations))
        self.assertTrue(any("MISSING_REQ_FILLED_NOT_PROVIDED:RA_REG" in v for v in violations))

    def test_masker_handles_null_anchor_lines_evidence_and_rejects_duplicate_req_id(self):
        raw_card = {
            "candidate_id": "T-NULL-DUP",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": None, "anchor_lines": None, "status": "verified"},
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years QA", "anchor_lines": 1, "status": "verified"},
            ],
        }
        clean, violations = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        self.assertIn("DUPLICATE_REQ_ID:QA_DEG", violations)
        deg_entries = [e for e in clean["entries"] if e["req_id"] == "QA_DEG"]
        self.assertEqual(len(deg_entries), 1)
        self.assertEqual(deg_entries[0]["evidence"], "")
        self.assertEqual(deg_entries[0]["anchor_lines"], [])
        exp_entry = [e for e in clean["entries"] if e["req_id"] == "QA_EXP"][0]
        self.assertEqual(exp_entry["anchor_lines"], [])

    def test_masker_and_verifier_block_tier_v_claim_text_only_tampering(self):
        raw_card = {
            "candidate_id": "T-TACIT-CLAIM",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_TACIT", "evidence": "", "anchor_lines": [], "status": None, "claim_text": "High integrity"},
            ],
        }
        clean, violations = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT", violations)
        tacit_clean = [e for e in clean["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tacit_clean["status"])
        self.assertEqual(tacit_clean["evidence"], "")
        self.assertEqual(tacit_clean["claim_text"], "")

        # Direct DeterministicVerifier call with unmasked Tier-V claim_text-only tampering
        tampered_card = {
            "candidate_id": "V-TACIT-CLAIM",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "tier": "M", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_TACIT", "tier": "V", "evidence": "", "anchor_lines": [], "status": None, "claim_text": "High integrity"},
            ],
        }
        verifier = DeterministicVerifier(span_threshold=0.85, max_anchor_span=3)
        verified, vv = verifier.verify_card("B.Eng.", tampered_card, "QA/QC")
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT", vv)
        tacit_ver = [e for e in verified["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tacit_ver["status"])
        self.assertEqual(tacit_ver["claim_text"], "")


class TestHybridRegulatoryRAGAndCorpus(unittest.TestCase):
    def test_default_corpus_12_pinned_and_ord650_excluded(self):
        rag = HybridRegulatoryRAG(alpha=0.5, pinned_only=True)
        expected_12 = {
            "ORD797-A35",
            "ORD797-A86",
            "ISO13485-6.2",
            "ISO13485-7.5.6",
            "ISO13485-8.2.4",
            "ISO13485-8.5.2",
            "SOP-EQ-01",
            "SOP-RA-02",
            "SOP-CERT-03",
            "NMPA-GMP-STERILE",
            "NMPA-GMP-IMPLANT",
            "SOP-VAL-04",
        }
        pinned_ids = {d.doc_id for d in rag.pinned_docs}
        all_ids = {d.doc_id for d in rag.all_docs}
        self.assertEqual(len(rag.pinned_docs), 12)
        self.assertEqual(pinned_ids, expected_12)
        self.assertEqual(len(rag.all_docs), 13)
        self.assertIn("ORD650-DEP", all_ids)
        self.assertNotIn("ORD650-DEP", pinned_ids)

    def test_pinned_only_blocks_ord650_on_legacy_query(self):
        rag = HybridRegulatoryRAG(alpha=0.5, pinned_only=True)
        res = rag.retrieve("Superseded 2014 Order 650 medical device supervision rules prior to Order 739", top_k=3)
        self.assertEqual(len(res), 3)
        self.assertTrue(all(doc.pinned and doc.doc_id != "ORD650-DEP" for doc, _ in res))

    def test_unpinned_rag_leaks_ord650_on_legacy_query(self):
        rag = HybridRegulatoryRAG(alpha=0.5, pinned_only=False)
        res = rag.retrieve("Superseded 2014 Order 650 medical device supervision rules prior to Order 739", top_k=3)
        self.assertTrue(any(doc.doc_id == "ORD650-DEP" and not doc.pinned for doc, _ in res))

    def test_empty_corpus_does_not_fallback_to_default(self):
        rag = HybridRegulatoryRAG(corpus=[], alpha=0.5, pinned_only=True)
        self.assertEqual(rag.all_docs, [])
        self.assertEqual(rag.pinned_docs, [])
        self.assertEqual(rag.retrieve("ISO 13485", top_k=3), [])


class TestDeterministicVerifierHardening(unittest.TestCase):
    def setUp(self):
        self.verifier = DeterministicVerifier(span_threshold=0.85, max_anchor_span=3)

    def test_rule2_tier_p_weak_verb_caught(self):
        resume = (
            "Education: B.Eng. Biomedical Engineering (2016-2020).\n"
            "Experience: Executed Class III QA work; participated in IQ/OQ/PQ validations.\n"
            "Credentials: Certified ISO 13485 internal auditor, Certificate No. CERT-101."
        )
        raw_card = {
            "candidate_id": "V-P1",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "Executed Class III QA work", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "participated in IQ/OQ/PQ validations", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        # Isolate line with only weak verb for QA_VAL
        resume_weak_val = (
            "Education: B.Eng. Biomedical Engineering (2016-2020).\n"
            "Experience: 4 years Class III medical device QA Engineer.\n"
            "Validation: participated in IQ/OQ/PQ validations.\n"
            "Credentials: Certified ISO 13485 internal auditor, Certificate No. CERT-101."
        )
        raw_card["entries"][1]["evidence"] = "4 years Class III medical device QA Engineer"
        raw_card["entries"][2]["anchor_lines"] = [4, 4]
        raw_card["entries"][3]["anchor_lines"] = [3, 3]
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        verified, vv = self.verifier.verify_card(resume_weak_val, masked, "QA/QC")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_VAL:") for v in vv))
        val_entry = [e for e in verified["entries"] if e["req_id"] == "QA_VAL"][0]
        self.assertEqual(val_entry["status"], "conflicting")

    def test_rule2_tier_m_degree_v_src_zero_allowed_and_escalated_claim_caught(self):
        resume = (
            "Education: B.Eng. Biomedical Engineering (2016-2020).\n"
            "Experience: 4 years Class III medical device QA Engineer.\n"
            "Credentials: Certified ISO 13485 internal auditor, Certificate No. CERT-102."
        )
        raw_card = {
            "candidate_id": "V-D1",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "claim_text": "Led Class III QA audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-102", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
        # QA_DEG has v_src=0, v_claim_raw=0 -> must NOT false alarm
        self.assertFalse(any("QA_DEG" in v for v in vv))
        # QA_EXP has v_src=0, v_claim_raw=3 ("Led") -> MUST be caught
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_EXP:") for v in vv))
        exp_entry = [e for e in verified["entries"] if e["req_id"] == "QA_EXP"][0]
        self.assertEqual(exp_entry["status"], "conflicting")

    def test_local_negation_and_certificate_no_handling(self):
        # Local negation blocks strong cert claim in same clause
        self.assertEqual(
            self.verifier.classify_verb_tier("Attended ISO 13485 training; no formal certificate listed."),
            1,
        )
        # Unrelated negation in another clause must NOT suppress Tier 3 on "Led"
        self.assertEqual(
            self.verifier.classify_verb_tier("Led 2 IQ/OQ/PQ validations; no formal certificate issued"),
            3,
        )
        # "Certificate No." must NOT be misclassified as negation
        cert_text = "Certified ISO 13485 internal auditor, Certificate No. CERT-1000"
        self.assertEqual(self.verifier.classify_verb_tier(cert_text), 3)
        self.assertTrue(self.verifier.has_credential_proof(cert_text))

    def test_anchor_lines_strictly_validated(self):
        lines = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5"]
        # Reversed span [3, 2] must be rejected
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 2", [3, 2]), (False, 0.0))
        # Out of bounds [0, 1] and [1, 99] must be rejected
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 1", [0, 1]), (False, 0.0))
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 1", [1, 99]), (False, 0.0))
        # Span exceeding max_anchor_span=3 ([1, 5] has 5 lines) must be rejected
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 1", [1, 5]), (False, 0.0))
        # Boolean anchor indices must be rejected
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 1", [True, 2]), (False, 0.0))
        # Valid single and multi-line (<=3) slices succeed
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 2", [2, 2]), (True, 1.0))
        self.assertEqual(self.verifier.check_span_alignment(lines, "Line 2 Line 3", [2, 3]), (True, 1.0))

    def test_scattered_containment_rejected_and_r05_multi_segment_allowed(self):
        lines = ["5 years retail sales manager; no medical device qa; basic excel"]
        aligned, sim = self.verifier.check_span_alignment(lines, "5 years medical device qa", [1, 1])
        self.assertFalse(aligned)
        self.assertLess(sim, 0.85)

        # Semicolon-separated negated segment also rejected by pre-segment negation guard
        aligned_neg, _ = self.verifier.check_span_alignment(lines, "5 years; medical device qa", [1, 1])
        self.assertFalse(aligned_neg)

        # Legitimate R05 QA_CAPA multi-segment extraction succeeds
        r05_lines = [
            "Led 2 production line IQ/OQ/PQ validations; updated ISO 14971 risk files; participated in 2 ISO 13485 internal audits; closed 28 CAPAs/yr."
        ]
        aligned_r05, sim_r05 = self.verifier.check_span_alignment(
            r05_lines, "updated ISO 14971 risk files; closed 28 CAPAs/yr", [1, 1]
        )
        self.assertTrue(aligned_r05)
        self.assertGreaterEqual(sim_r05, 0.85)

    def test_rule2_and_rule3_both_execute_and_join_verifier_note(self):
        resume = (
            "Education: B.Eng. Biomedical Engineering (2015-2019).\n"
            "Experience: 4 years Class III stent QA Engineer. Executed IQ/OQ/PQ.\n"
            "Training: Attended ISO 13485 internal auditor training; no formal certificate listed."
        )
        raw_card = {
            "candidate_id": "R05-JOIN",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2015-2019)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III stent QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Attended ISO 13485 internal auditor training", "claim_text": "Certified ISO 13485 internal auditor", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_CERT") for v in vv))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", vv)
        cert_entry = [e for e in verified["entries"] if e["req_id"] == "QA_CERT"][0]
        self.assertIn(";", cert_entry["verifier_note"])
        self.assertIn("tier", cert_entry["verifier_note"].lower())
        self.assertIn("credential proof", cert_entry["verifier_note"].lower())

    def test_rule4_tier_v_tampering_reset(self):
        card = {
            "candidate_id": "V-TACIT",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "tier": "M", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_TACIT", "tier": "V", "evidence": "High integrity", "anchor_lines": [1, 1], "status": "verified"},
            ],
        }
        verified, vv = self.verifier.verify_card("B.Eng.", card, "QA/QC")
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT", vv)
        tacit = [e for e in verified["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tacit["status"])
        self.assertEqual(tacit["evidence"], "")
        self.assertEqual(tacit["anchor_lines"], [])

    def test_component_off_toggles_exact_string_only(self):
        exact_only = DeterministicVerifier(
            span_threshold=0.85, use_fuzzy_span=False, use_verb_tier=False, use_cert_proof=False
        )
        lines = ["Executed IQ/OQ/PQ validation and ISO 14971 CAPA closures."]
        self.assertEqual(exact_only.check_span_alignment(lines, "Executed IQ/OQ/PQ validation", [1, 1]), (True, 1.0))
        self.assertEqual(exact_only.check_span_alignment(lines, "Executed IQ-OQ-PQ validation", [1, 1]), (False, 0.0))

    def test_multi_clause_line_rule2_scoped_to_evidence_clause(self):
        resume = "Header\nLed ISO 13485 internal audits; attended auditor training; Maintained ISO 13485 system."
        raw = {
            "candidate_id": "MC-1",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "attended auditor training", "claim_text": "Certified ISO 13485 auditor", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
        verified, vv = self.verifier.verify_card(resume, masked, "QMS")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_CERT") for v in vv))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QMS_CERT", vv)
        self.assertFalse(any("QMS_LEAD" in v for v in vv))

    def test_bug1_negated_qa_cert_clauses_caught_and_fsm_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        cases = [
            ("Certified ISO 13485 auditor; no formal certificate issued.", "Certified ISO 13485 auditor"),
            ("Certified internal auditor, no formal certificate issued.", "Certified internal auditor"),
            ("Certified lead auditor. No formal certificate number listed.", "Certified lead auditor"),
            ("Certified internal auditor; lacking certificate number.", "Certified internal auditor"),
            ("No formal certificate issued; Certified internal auditor.", "Certified internal auditor"),
        ]
        for line1, cert_ev in cases:
            resume = f"{line1}\nB.Eng. Executed 4 years QA"
            raw = {
                "candidate_id": "NEG-CERT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "Executed 4 years QA", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": cert_ev, "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", vv, f"Failed for {line1}")
            cert_entry = [e for e in verified["entries"] if e["req_id"] == "QA_CERT"][0]
            self.assertEqual(cert_entry["status"], "conflicting")
            res = gate.step(verified, "Confirm", proposed_decision=1)
            self.assertEqual(res.next_state, "s_block")
            self.assertEqual(res.error_code, "ERR_UNRESOLVED_CONFLICT_LOCK")

        # Ontology label phrasings without negation pass has_credential_proof
        self.assertTrue(self.verifier.has_credential_proof("Certified ISO 13485 internal auditor"))
        self.assertTrue(self.verifier.has_credential_proof("Certified ISO 13485 lead auditor"))

    def test_bug1_comma_separated_rule2_negation_not_dropped(self):
        resume_pre = (
            "Header\n"
            "Not authorized to lead audits, led minor documentation prep\n"
            "Certified ISO 13485 auditor, Certificate No. CERT-1\n"
            "Maintained ISO 13485 system."
        )
        raw_pre = {
            "candidate_id": "COMMA-PRE",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Certificate No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "led minor documentation prep", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        masked_pre, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_pre, "QMS")
        verified_pre, vv_pre = self.verifier.verify_card(resume_pre, masked_pre, "QMS")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_LEAD") for v in vv_pre))
        self.assertEqual([e for e in verified_pre["entries"] if e["req_id"] == "QMS_LEAD"][0]["status"], "conflicting")

        resume_post = (
            "Header\n"
            "Led ISO 13485 internal audits, not verified by formal record\n"
            "Certified ISO 13485 auditor, Certificate No. CERT-1\n"
            "Maintained ISO 13485 system."
        )
        raw_post = {
            "candidate_id": "COMMA-POST",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Certificate No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        masked_post, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_post, "QMS")
        verified_post, vv_post = self.verifier.verify_card(resume_post, masked_post, "QMS")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_LEAD") for v in vv_post))
        self.assertEqual([e for e in verified_post["entries"] if e["req_id"] == "QMS_LEAD"][0]["status"], "conflicting")

    def test_auditor_chinese_weichizheng_caveat(self):
        self.assertEqual(self.verifier.classify_verb_tier("未持证，仅参与内审培训"), 1)
        self.assertFalse(self.verifier.has_credential_proof("未持证，仅参与内审培训"))

    def test_bug2_span_negation_multiseg_singleseg_and_tier12_verb_negation(self):
        # Multi-segment with high whole-string similarity must still fail pre-negation guard
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["5 years retail; no medical device qa"], "5 years retail; medical device qa", [1, 1]
            )[0]
        )
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["5 years retail; no prior medical device qa"], "5 years retail; medical device qa", [1, 1]
            )[0]
        )
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["5 years retail sales manager at big store; other duties; without prior medical device qa"],
                "5 years retail sales manager; medical device qa",
                [1, 1],
            )[0]
        )
        # Single-segment pre-negation guard
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["5 years retail; no medical device qa"], "medical device qa", [1, 1]
            )[0]
        )
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["No Class III medical device QA Engineer experience."],
                "Class III medical device QA Engineer",
                [1, 1],
            )[0]
        )
        # "Certificate No." must not trigger false negation alarm
        self.assertTrue(
            self.verifier.check_span_alignment(
                ["Certified ISO 13485 auditor, Certificate No. CERT-101"], "CERT-101", [1, 1]
            )[0]
        )
        # Tier 2 & Tier 1 negation scoping
        self.assertEqual(self.verifier.classify_verb_tier("Never executed IQ/OQ/PQ validation."), 0)
        self.assertEqual(self.verifier.classify_verb_tier("Never participated in IQ/OQ/PQ validation."), 0)

        # End-to-end verify_card catches negated Tier-P validation (both stripped and verbatim-negated quotes)
        for ev_text in ("executed IQ/OQ/PQ validation", "Never executed IQ/OQ/PQ validation"):
            resume_neg_val = (
                "Education: B.Eng. Biomedical Engineering (2016-2020).\n"
                "Experience: 4 years Class III medical device QA Engineer.\n"
                "Validation: Never executed IQ/OQ/PQ validation.\n"
                "Credentials: Certified ISO 13485 internal auditor, Certificate No. CERT-101."
            )
            raw_neg_val = {
                "candidate_id": "NEG-VAL-E2E",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": ev_text, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked_nv, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_neg_val, "QA/QC")
            verified_nv, vv_nv = self.verifier.verify_card(resume_neg_val, masked_nv, "QA/QC")
            self.assertTrue(vv_nv, f"Expected violation for {ev_text!r}")
            self.assertEqual([e for e in verified_nv["entries"] if e["req_id"] == "QA_VAL"][0]["status"], "conflicting")

    def test_bug3_tier_p_min_verb_tier2_enforced_when_v_src_zero(self):
        resume = (
            "Education: B.Eng. Biomedical Engineering (2016-2020).\n"
            "Experience: 4 years Class III medical device QA Engineer.\n"
            "Validation: Observed IQ/OQ/PQ validation.\n"
            "Credentials: Certified ISO 13485 internal auditor, Certificate No. CERT-101."
        )
        raw = {
            "candidate_id": "TIER-P-V0",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering (2016-2020)", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Observed IQ/OQ/PQ validation", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
        verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_VAL:") for v in vv))
        self.assertFalse(any("QA_EXP" in v for v in vv))
        self.assertEqual([e for e in verified["entries"] if e["req_id"] == "QA_VAL"][0]["status"], "conflicting")

    def test_cert_dot_no_abbreviation_does_not_false_alarm(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        resume = "Certified ISO 13485 internal auditor, Cert. No. CERT-101\nB.Eng. Executed 4 years QA"
        raw = {
            "candidate_id": "CERT-DOT-NO",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "Executed 4 years QA", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
        verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
        self.assertEqual(vv, [])
        res = gate.step(verified, "Confirm", proposed_decision=1)
        self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"))

    def test_verbatim_negated_tier_m_quotes_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        neg_exp_cases = [
            ("No Class III medical device QA Engineer experience.", "No Class III medical device QA Engineer experience"),
            ("Lacks Class III medical device QA Engineer experience.", "Lacks Class III medical device QA Engineer experience"),
            ("Without Class III medical device QA Engineer experience.", "Without Class III medical device QA Engineer experience"),
            ("无三类医疗器械QA工程师经验。", "无三类医疗器械QA工程师经验"),
            ("Experience: No Class III medical device QA Engineer experience", "Experience: No Class III medical device QA Engineer experience"),
            ("Candidate lacks Class III medical device QA Engineer experience", "Candidate lacks Class III medical device QA Engineer experience"),
            ("QA: 没有三类医疗器械QA工程师经验", "QA: 没有三类医疗器械QA工程师经验"),
            ("Class III medical device QA Engineer experience (pending)", "Class III medical device QA Engineer experience"),
        ]
        for line2, ev in neg_exp_cases:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Certificate No. CERT-101"
            raw = {
                "candidate_id": "VERBATIM-NEG-M",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, f"Expected violation for {line2!r}")
            res = gate.step(verified, "Confirm", proposed_decision=1)
            self.assertEqual(res.next_state, "s_block")

        # Verbatim negated QA_DEG also blocked
        resume_deg = (
            "No Bachelor degree in biomedical engineering.\n"
            "4 years Class III medical device QA Engineer\n"
            "Certified ISO 13485 internal auditor, Certificate No. CERT-101"
        )
        raw_deg = {
            "candidate_id": "VERBATIM-NEG-DEG",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "No Bachelor degree in biomedical engineering", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked_deg, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_deg, "QA/QC")
        verified_deg, vv_deg = self.verifier.verify_card(resume_deg, masked_deg, "QA/QC")
        self.assertTrue(vv_deg)
        self.assertEqual(gate.step(verified_deg, "Confirm", proposed_decision=1).next_state, "s_block")

        # Positive QA_EXP sharing line with sibling "; no formal certificate issued." must NOT false alarm
        resume_sib = (
            "B.Eng.\n"
            "4 years Class III medical device QA Engineer; no formal certificate issued.\n"
            "Certified ISO 13485 internal auditor, Certificate No. CERT-101"
        )
        raw_sib = {
            "candidate_id": "VALID-SIBLING-EXP",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Certificate No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked_sib, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_sib, "QA/QC")
        verified_sib, vv_sib = self.verifier.verify_card(resume_sib, masked_sib, "QA/QC")
        self.assertEqual(vv_sib, [])
        self.assertEqual(gate.step(verified_sib, "Confirm", proposed_decision=1).final_decision, "1")

    def test_credential_invalidation_and_chinese_negations_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        cases = [
            ("Certified ISO 13485 internal auditor; lacks formal certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; lacks certificate number", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; without valid certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; no valid certificate issued", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; no current certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate expired in 2021", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate has expired", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor (expired)", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor (revoked)", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor (pending)", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; failed to obtain certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; failed to obtain formal certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate not issued", "Certified ISO 13485 internal auditor"),
            ("Certified internal auditor, missing certificate.", "Certified internal auditor"),
            ("Certified internal auditor; certificate not issued.", "Certified internal auditor"),
            ("非持证，仅参加内审培训", "非持证"),
            ("未正式持证，仅参与内审培训", "未正式持证"),
            ("不再持证，仅参加内审培训", "不再持证"),
            ("持证（已过期），内审员", "持证"),
            ("Certified ISO 13485 internal auditor（未获得证书）", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor（证书已过期）", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; 无有效证书", "Certified ISO 13485 internal auditor"),
        ]
        for txt, ev in cases:
            self.assertFalse(self.verifier.has_credential_proof(txt), txt)
            self.assertLess(self.verifier.classify_verb_tier(txt), 3, txt)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{txt}"
            raw = {
                "candidate_id": "CERT-INV",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, txt)
            self.assertEqual(gate.step(verified, "Confirm", proposed_decision=1).next_state, "s_block", txt)

    def test_span_alignment_internal_negation_modifier_gap_and_fuzzy_off(self):
        # Internal-negation mismatches inside matched span
        self.assertEqual(
            self.verifier.check_span_alignment(["5 years retail sales no medical device qa"], "retail sales medical device qa", [1, 1]),
            (False, 0.0),
        )
        self.assertEqual(
            self.verifier.check_span_alignment(["5 years retail sales; no medical device qa"], "retail sales medical device qa", [1, 1]),
            (False, 0.0),
        )
        self.assertEqual(
            self.verifier.check_span_alignment(["Executed 4 years QA; no IQ/OQ/PQ validation"], "Executed 4 years QA IQ/OQ/PQ validation", [1, 1]),
            (False, 0.0),
        )
        # Modifier gaps in _SPAN_PRE_NEG_RE
        for src, ev in [
            ("5 years retail; no direct medical device qa", "medical device qa"),
            ("5 years retail; lacks required medical device qa", "medical device qa"),
            ("5 years retail; no hands-on medical device qa", "medical device qa"),
            ("5 years retail; no Class III medical device qa", "medical device qa"),
            ("5 years retail; failed to complete IQ/OQ/PQ validation", "IQ/OQ/PQ validation"),
            ("5 years retail; no longer performs IQ/OQ/PQ validation", "IQ/OQ/PQ validation"),
        ]:
            self.assertEqual(self.verifier.check_span_alignment([src], ev, [1, 1]), (False, 0.0), src)

        # use_fuzzy_span=False still enforces pre-negation & internal negation guards
        exact_verifier = DeterministicVerifier(use_fuzzy_span=False)
        self.assertEqual(
            exact_verifier.check_span_alignment(["5 years retail; no medical device qa"], "medical device qa", [1, 1]),
            (False, 0.0),
        )
        self.assertEqual(
            exact_verifier.check_span_alignment(["Never executed IQ/OQ/PQ validation"], "executed IQ/OQ/PQ validation", [1, 1]),
            (False, 0.0),
        )

    def test_rule2_long_clause_pre_negation_and_trailing_negation_blocked(self):
        negated_phrases = [
            "Not qualified to lead audits, led minor documentation prep",
            "Led ISO 13485 internal audits, without formal verification",
            "Never in any medical device manufacturing setting independently led internal audits",
            "Never allowed to direct audits, directed minor prep",
            "No formal certificate issued, Certified internal auditor",
            "Executed IQ/OQ/PQ validation (pending)",
            "Executed IQ/OQ/PQ validation, failed to complete protocol",
            "Executed IQ/OQ/PQ validation, expired in 2021",
            "Led ISO 13485 internal audits, compiled checklists, not verified by formal record",
        ]
        for txt in negated_phrases:
            self.assertEqual(self.verifier.classify_verb_tier(txt), 0, txt)

        # End-to-end QMS_LEAD blocked for pre/post comma-clause negations
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line2, ev in [
            ("Not qualified to lead audits, led minor documentation prep", "led minor documentation prep"),
            ("Led ISO 13485 internal audits, without formal verification", "Led ISO 13485 internal audits"),
        ]:
            resume = (
                "Header\n"
                f"{line2}\n"
                "Certified ISO 13485 auditor, Certificate No. CERT-1\n"
                "Maintained ISO 13485 system."
            )
            raw = {
                "candidate_id": "QMS-NEG",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Certificate No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            verified, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_LEAD") for v in vv), (line2, vv))
            self.assertEqual(gate.step(verified, "Confirm", proposed_decision=1).next_state, "s_block")

    def test_r4_fn_prefix_contractions_placeholder_training_and_lifecycle_cert_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        fn_cases = [
            ("Non-certified ISO 13485 internal auditor", "Non-certified ISO 13485 internal auditor"),
            ("Non certified ISO 13485 internal auditor", "Non certified ISO 13485 internal auditor"),
            ("Un-certified ISO 13485 internal auditor", "Un-certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; doesn't hold a formal certificate.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; didn't obtain certificate.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; hasn't renewed certificate.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; no cert.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor, Certificate No. N/A", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor, Cert. No. N/A", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor, Certificate No. None", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor, Cert. No. ---", "Certified ISO 13485 internal auditor"),
            ("Attended ISO 13485 internal auditor training; Training Certificate No. TRN-101", "Training Certificate No. TRN-101"),
            ("Training Certificate No. CERT-TRN-101", "Training Certificate No. CERT-TRN-101"),
            ("Certified ISO 13485 internal auditor; certificate lapsed.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate withdrawn.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; invalid certificate", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor (invalid)", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate void", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor (in progress)", "Certified ISO 13485 internal auditor"),
            ("持证内审员（待取证）", "持证内审员"),
            ("ISO 13485 持证内审员（备考中）", "ISO 13485 持证内审员"),
            ("持证内审员（已注销）", "持证内审员"),
            ("持证内审员；证书无效", "持证内审员"),
            ("Certified ISO 13485 internal auditor; 未通过内审员认证", "Certified ISO 13485 internal auditor"),
            ("Expired ISO 13485 internal auditor certificate, Cert. No. CERT-101", "Expired ISO 13485 internal auditor certificate, Cert. No. CERT-101"),
            ("Revoked ISO 13485 internal auditor certificate", "Revoked ISO 13485 internal auditor certificate"),
        ]
        for line3, ev in fn_cases:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "R4-FN-CERT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            verified, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(verified, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Positive controls preserved (Re-certified, non-conformity, Training + Internal Auditor Cert)
        for pos in [
            "Re-certified ISO 13485 internal auditor, Cert. No. CERT-202",
            "Certified ISO 13485 internal auditor, tracked non-conformity closures, Cert. No. CERT-101",
            "Attended ISO 13485 internal auditor training; Internal Auditor Certificate No. CERT-101",
        ]:
            self.assertTrue(self.verifier.has_credential_proof(pos), pos)
            self.assertEqual(self.verifier.classify_verb_tier(pos), 3, pos)

    def test_r4_fn_semicolon_no_formal_verification_and_zero_years_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        # QA_CERT with "; no formal verification"
        res_cert = "B.Eng.\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor; no formal verification"
        raw_cert = {
            "candidate_id": "R4-VERIF-CERT",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked_cert, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_cert, "QA/QC")
        ver_cert, vv_cert = self.verifier.verify_card(res_cert, masked_cert, "QA/QC")
        self.assertTrue(vv_cert)
        self.assertEqual(gate.step(ver_cert, "Confirm", proposed_decision=1).next_state, "s_block")

        # QMS_LEAD with "; no formal verification"
        res_qms = (
            "Header\n"
            "Led ISO 13485 internal audits; no formal verification\n"
            "Certified ISO 13485 auditor, Certificate No. CERT-1\n"
            "Maintained ISO 13485 system."
        )
        raw_qms = {
            "candidate_id": "R4-VERIF-QMS",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Certificate No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        masked_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qms, "QMS")
        ver_qms, vv_qms = self.verifier.verify_card(res_qms, masked_qms, "QMS")
        self.assertTrue(vv_qms)
        self.assertEqual(gate.step(ver_qms, "Confirm", proposed_decision=1).next_state, "s_block")

        # 0 years / zero years QA_EXP
        for ev in ("Class III medical device QA Engineer experience", "0 years Class III medical device QA Engineer experience"):
            res_0 = "B.Eng.\n0 years Class III medical device QA Engineer experience.\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw_0 = {
                "candidate_id": "R4-0YR",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked_0, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_0, "QA/QC")
            ver_0, vv_0 = self.verifier.verify_card(res_0, masked_0, "QA/QC")
            self.assertTrue(vv_0, ev)
            self.assertEqual(gate.step(ver_0, "Confirm", proposed_decision=1).next_state, "s_block", ev)

        self.assertFalse(self.verifier.check_span_alignment(["zero years of QA experience"], "QA experience", [1, 1])[0])
        self.assertFalse(self.verifier.check_span_alignment(["0个月医疗器械QA经验"], "医疗器械QA经验", [1, 1])[0])
        # Positive controls
        self.assertEqual(self.verifier.classify_verb_tier("Led 2 IQ/OQ/PQ validations; no formal certificate issued"), 3)
        self.assertTrue(self.verifier.check_span_alignment(["0 defects; zero nonconformities"], "0 defects; zero nonconformities", [1, 1])[0])

    def test_r4_fp_quality_domain_compliance_phrasings_pass(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        # QA/QC positive quality-compliance outcomes
        qa_cases = [
            ("QA_VAL", "Executed IQ/OQ/PQ validation with no nonconformity", "Executed IQ/OQ/PQ validation with no nonconformity"),
            ("QA_EXP", "4 years Class III medical device QA Engineer, passed audit with no nonconformities", "4 years Class III medical device QA Engineer, passed audit with no nonconformities"),
            ("QA_EXP", "4 years Class III medical device QA Engineer, passed audit with no nonconformities", "4 years Class III medical device QA Engineer"),
            ("QA_EXP", "4年三类无菌医疗器械QA工程师，连续三年无重大不符合项", "4年三类无菌医疗器械QA工程师"),
            ("QA_VAL", "执行IQ/OQ/PQ验证，无偏差", "执行IQ/OQ/PQ验证"),
        ]
        for rid, line, ev in qa_cases:
            if rid == "QA_EXP":
                resume = f"B.Eng.\n{line}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
                entries = [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ]
            else:
                resume = f"B.Eng.\n4 years Class III QA Engineer\n{line}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
                entries = [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ]
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate({"candidate_id": "FP-QA", "role": "QA/QC", "entries": entries}, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (rid, line, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), (rid, line))

        # QMS positive quality-compliance outcomes
        for line2, ev in [
            ("Led 3 ISO 13485 internal audits with no nonconformity", "Led 3 ISO 13485 internal audits"),
            ("Led 3 internal audits with no record of nonconformity", "Led 3 internal audits"),
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "FP-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertEqual(vv, [], (line2, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1")

        # RA Chinese positive quality-compliance outcomes
        for line3 in [
            "撰写临床评价报告(CER)及发补回复，连续三年无重大不符合项。",
            "撰写临床评价报告(CER)及发补回复，无发补逾期记录。",
        ]:
            resume = f"B.Med.\n独立主导Class III NMPA注册\n{line3}"
            raw = {
                "candidate_id": "FP-RA",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "独立主导Class III NMPA注册", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": "撰写临床评价报告(CER)及发补回复", "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertEqual(vv, [], (line3, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1")

    def test_r4_fp_chinese_sentence_boundaries_and_sibling_clause_scoping(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        # Chinese sentence boundaries preserved across 。
        res_exp = "B.Eng.\n参与无菌车间管理。未取得主任审核员证。4年三类医疗器械QA工程师。\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
        raw_exp = {
            "candidate_id": "CN-BOUND-EXP",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4年三类医疗器械QA工程师", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_exp, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_exp, "QA/QC")
        v_exp, vv_exp = self.verifier.verify_card(res_exp, m_exp, "QA/QC")
        self.assertEqual(vv_exp, [])
        self.assertEqual(gate.step(v_exp, "Confirm", proposed_decision=1).final_decision, "1")

        for line3 in ["三年无重大不良事件。执行IQ/OQ/PQ验证。", "未参加无菌车间管理。执行IQ/OQ/PQ验证。"]:
            res_val = f"B.Eng.\n4年三类医疗器械QA工程师\n{line3}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw_val = {
                "candidate_id": "CN-BOUND-VAL",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4年三类医疗器械QA工程师", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "执行IQ/OQ/PQ验证", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            m_val, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_val, "QA/QC")
            v_val, vv_val = self.verifier.verify_card(res_val, m_val, "QA/QC")
            self.assertEqual(vv_val, [], (line3, vv_val))
            self.assertEqual(gate.step(v_val, "Confirm", proposed_decision=1).final_decision, "1")

        # Sibling negation sentence must NOT poison QA_EXP with weak verb from Attended
        res_sib = "B.Eng.\n4 years Class III medical device QA Engineer. Attended audit workshop, no formal certificate."
        raw_sib = {
            "candidate_id": "AUD-C5",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III medical device QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Attended audit workshop", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_sib, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_sib, "QA/QC")
        _, vv_sib = self.verifier.verify_card(res_sib, m_sib, "QA/QC")
        self.assertFalse(any("QA_EXP" in x for x in vv_sib), vv_sib)
        self.assertTrue(any("QA_CERT" in x for x in vv_sib), vv_sib)

        # Cross-sentence overlapping authority negation still caught
        res_auth = "Header\nLed ISO 13485 internal audits; not authorized to lead internal audits\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
        raw_auth = {
            "candidate_id": "SIB-AUTH",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_auth, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_auth, "QMS")
        _, vv_auth = self.verifier.verify_card(res_auth, m_auth, "QMS")
        self.assertTrue(any("QMS_LEAD" in x for x in vv_auth), vv_auth)

    def test_r4_auditor_cn_negation_cannot_declined_and_suspended_pending_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for src, ev in [
            ("并未执行IQ/OQ/PQ验证。", "并未执行IQ/OQ/PQ验证"),
            ("并未执行IQ/OQ/PQ验证。", "执行IQ/OQ/PQ验证"),
            ("Experience: cannot verify ISO 14971 CAPA closures", "ISO 14971 CAPA closures"),
            ("Experience: declined to lead internal audit", "lead internal audit"),
            ("Led IQ/OQ/PQ validation, suspended pending audit", "Led IQ/OQ/PQ validation"),
        ]:
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{src}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "AUD-C124",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (src, ev))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (src, ev))

        self.assertFalse(self.verifier.check_span_alignment(["未曾担任Class III QA工程师。"], "Class III QA工程师", [1, 1])[0])
        self.assertLess(self.verifier.classify_verb_tier("尚未独立主导Class III NMPA注册"), 3)
        self.assertEqual(self.verifier.classify_verb_tier("Led IQ/OQ/PQ validation, suspended pending audit"), 0)

    def test_r5_fn_abbreviated_cert_and_bare_status_sentences_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line3 in [
            "Certified ISO 13485 internal auditor; cert. void.",
            "Certified ISO 13485 internal auditor; cert. lapsed.",
            "Certified ISO 13485 internal auditor; cert. expired.",
            "Certified ISO 13485 internal auditor; cert. revoked.",
            "Certified ISO 13485 internal auditor; cert. invalid.",
            "Certified ISO 13485 internal auditor; cert. pending.",
            "Certified ISO 13485 internal auditor; cert. not issued.",
            "Certified ISO 13485 internal auditor; cert void",
            "Certified ISO 13485 internal auditor; cert. not renewed.",
            "Certified ISO 13485 internal auditor; expired",
            "Certified ISO 13485 internal auditor; cancelled",
            "Certified ISO 13485 internal auditor; revoked.",
        ]:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "R5-C1",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Positive preservation: unrelated project status sentence must NOT trigger standalone status lock
        res_ok = "B.Eng.\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-101; prior 2020 project suspended"
        raw_ok = {
            "candidate_id": "R5-C1-POS",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_ok, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_ok, "QA/QC")
        v_ok, vv_ok = self.verifier.verify_card(res_ok, m_ok, "QA/QC")
        self.assertEqual(vv_ok, [])
        self.assertEqual(gate.step(v_ok, "Confirm", proposed_decision=1).final_decision, "1")

    def test_r5_fn_training_cert_no_cn_placeholders_credential_self_not_certified_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line3, ev in [
            ("Attended ISO 13485 internal auditor training; Training Certificate, Cert. No. TRN-101", "Training Certificate, Cert. No. TRN-101"),
            ("ISO 13485 internal auditor training certificate, Cert. No. TRN-101", "ISO 13485 internal auditor training certificate, Cert. No. TRN-101"),
            ("Attended ISO 13485 internal auditor training; Certificate of Training, Cert. No. TRN-101", "Certificate of Training, Cert. No. TRN-101"),
            ("Attended ISO 13485 internal auditor workshop; Certificate of Attendance, Cert. No. ATT-101", "Certificate of Attendance, Cert. No. ATT-101"),
            ("Training Certificate, Cert. No. TRN-101", "Training Certificate, Cert. No. TRN-101"),
            ("training certificate, Cert. No. TRN-101", "training certificate, Cert. No. TRN-101"),
            ("持证内审员，证书编号：N/A", "持证内审员"),
            ("持证内审员，证书编号：待补", "持证内审员"),
            ("Certified ISO 13485 internal auditor, Cert. No. 暂无", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; lacks credential.", "Certified ISO 13485 internal auditor"),
            ("Self-certified ISO 13485 internal auditor", "Self-certified ISO 13485 internal auditor"),
            ("Not-certified ISO 13485 internal auditor", "Not-certified ISO 13485 internal auditor"),
        ]:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "R5-C2",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Positive statutory auditor certificate after training preserved
        for pos in [
            "Attended ISO 13485 internal auditor training; Internal Auditor Certificate No. CERT-101",
            "Completed training course; Lead Auditor Certificate No. LA-442",
        ]:
            self.assertTrue(self.verifier.has_credential_proof(pos), pos)
            self.assertEqual(self.verifier.classify_verb_tier(pos), 3, pos)

    def test_r5_fn_sibling_authority_verification_contradictions_and_zero_year_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line2, ev in [
            ("Led 3 ISO 13485 internal audits. Not approved by QA.", "Led 3 ISO 13485 internal audits"),
            ("Led 3 ISO 13485 internal audits; not authorized to do so.", "Led 3 ISO 13485 internal audits"),
            ("Led 3 ISO 13485 internal audits; did not obtain formal verification.", "Led 3 ISO 13485 internal audits"),
            ("Led 3 ISO 13485 internal audits; verification pending.", "Led 3 ISO 13485 internal audits"),
            ("Led ISO 13485 internal audits; not authorized to lead ISO 13485 internal audits", "Led ISO 13485 internal audits"),
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "R5-C3-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": ev, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        for line3 in [
            "Executed IQ/OQ/PQ validation; did not obtain formal verification.",
            "Executed IQ/OQ/PQ validation; not permitted to execute IQ/OQ/PQ validation",
        ]:
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R5-C3-VAL",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Hyphenated 0-year experience blocked
        res_0 = "B.Eng.\n0-year Class III medical device QA Engineer experience.\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
        raw_0 = {
            "candidate_id": "R5-C3-0YR",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "Class III medical device QA Engineer experience", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_0, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_0, "QA/QC")
        v_0, vv_0 = self.verifier.verify_card(res_0, m_0, "QA/QC")
        self.assertTrue(vv_0)
        self.assertEqual(gate.step(v_0, "Confirm", proposed_decision=1).next_state, "s_block")

    def test_r5_fp_cn_quantifier_cert_split_autonomy_without_weak_disclaimer_and_not_only_pass(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line2, ev2 in [
            ("4年三类医疗器械QA工程师，连续三年无任何不符合项", "4年三类医疗器械QA工程师"),
            ("4年三类医疗器械QA工程师，连续三年无任何重大偏差", "4年三类医疗器械QA工程师"),
            ("Attended ISO 13485 workshop, no cert. 4 years Class III medical device QA Engineer.", "4 years Class III medical device QA Engineer"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R5-C4-EXP",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), line2)

        # Independently led without supervision (QMS_LEAD)
        res_qms = "Header\nIndependently led ISO 13485 internal audits without supervision\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
        raw_qms = {
            "candidate_id": "R5-C4-QMS",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Independently led ISO 13485 internal audits without supervision", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qms, "QMS")
        v_qms, vv_qms = self.verifier.verify_card(res_qms, m_qms, "QMS")
        self.assertEqual(vv_qms, [])
        self.assertEqual(gate.step(v_qms, "Confirm", proposed_decision=1).final_decision, "1")

        # QA_VAL positive phrasings: without external support, unrelated weak disclaimer, not only...
        for line3, ev3 in [
            ("Executed IQ/OQ/PQ validation without external support", "Executed IQ/OQ/PQ validation without external support"),
            ("Executed IQ/OQ/PQ validation. Not trained on EO sterilization validation.", "Executed IQ/OQ/PQ validation"),
            ("Not only executed IQ/OQ/PQ validation, but also led CAPA", "executed IQ/OQ/PQ validation"),
        ]:
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R5-C4-VAL",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": ev3, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line3, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), line3)

    def test_r6_fn_lifecycle_no_longer_valid_placeholders_and_training_modifiers_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        fn_cases = [
            ("Certified ISO 13485 internal auditor; no longer valid.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; certificate no longer valid.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; cert. is no longer valid.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; not valid.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; not renewed.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; renewal pending.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; cert: not renewed.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor, Cert. No. [Pending]", "Certified ISO 13485 internal auditor"),
            ("持证内审员，证书编号：待发", "持证内审员"),
            ("Certificate of Course Completion, Cert. No. TRN-101", "Certificate of Course Completion, Cert. No. TRN-101"),
            ("Certificate of Workshop Attendance, Cert. No. ATT-202", "Certificate of Workshop Attendance, Cert. No. ATT-202"),
            ("Training Internal Auditor Certificate, Cert. No. TRN-909", "Training Internal Auditor Certificate, Cert. No. TRN-909"),
        ]
        for line3, ev in fn_cases:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "R6-C1",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Positive preservation for genuine statutory certificates after training
        for pos in [
            "Attended ISO 13485 internal auditor training; Internal Auditor Certificate No. CERT-101",
            "Completed training course; Lead Auditor Certificate No. LA-442",
        ]:
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{pos}"
            raw = {
                "candidate_id": "R6-C1-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": pos, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (pos, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1", pos)

    def test_r6_fn_verif_auth_zero_exp_and_degree_bypasses_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line2 in [
            "Led 3 ISO 13485 internal audits; could not obtain formal verification.",
            "Led 3 ISO 13485 internal audits; lacks formal verification.",
            "Led 3 ISO 13485 internal audits; was not formally verified.",
            "Led 3 ISO 13485 internal audits; not-authorized to do so.",
            "Led 3 ISO 13485 internal audits; unapproved by QA.",
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "R6-C2-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led 3 ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # QA_VAL failure
        res_val = "B.Eng.\n4 years Class III QA Engineer\nExecuted IQ/OQ/PQ validation; validation failed.\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
        raw_val = {
            "candidate_id": "R6-C2-VAL",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_val, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_val, "QA/QC")
        v_val, vv_val = self.verifier.verify_card(res_val, m_val, "QA/QC")
        self.assertTrue(vv_val)
        self.assertEqual(gate.step(v_val, "Confirm", proposed_decision=1).next_state, "s_block")

        # QA_EXP zero-experience boundary variants
        for line2, ev2 in [
            ("0.0 years Class III medical device QA Engineer experience.", "Class III medical device QA Engineer experience"),
            ("Zero relevant medical device QA experience.", "medical device QA experience"),
            ("Less than 1 year Class III medical device QA Engineer experience.", "Class III medical device QA Engineer experience"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R6-C2-EXP",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # Positive decimal boundary preservation (0.5 and 4.0)
        for pos_exp in [
            "0.5 years cleanroom + 4 years Class III medical device QA Engineer experience.",
            "4.0 years Class III medical device QA Engineer experience.",
        ]:
            resume = f"B.Eng.\n{pos_exp}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R6-C2-EXP-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": pos_exp, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (pos_exp, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1", pos_exp)

        # QA_DEG (v_src=0, min_verb_tier=1) degree pending / did not graduate
        for line1 in [
            "B.Eng. in Biomedical Engineering; degree pending.",
            "B.Eng. in Biomedical Engineering; did not graduate.",
        ]:
            resume = f"{line1}\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R6-C2-DEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng. in Biomedical Engineering", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line1)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line1)

    def test_r6_fp_without_external_guidance_and_cn_quality_safety_incidents_pass(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        res_val = "B.Eng.\n4 years Class III QA Engineer\nExecuted IQ/OQ/PQ validation without external guidance\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
        raw_val = {
            "candidate_id": "R6-C3-VAL",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation without external guidance", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_val, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_val, "QA/QC")
        v_val, vv_val = self.verifier.verify_card(res_val, m_val, "QA/QC")
        self.assertEqual(vv_val, [], vv_val)
        self.assertEqual(gate.step(v_val, "Confirm", proposed_decision=1).final_decision, "1")

        for line2 in [
            "4年三类医疗器械QA工程师，连续三年无任何质量事故",
            "4年三类医疗器械QA工程师，连续三年无重大安全事件",
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "R6-C3-CN",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4年三类医疗器械QA工程师", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1", line2)

    def test_m4_p2_gap3_fn_semicolon_lifecycle_training_verif_and_under_1yr_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # Semicolon lifecycle statuses + Chinese invalidation + training cert modifier (QA_CERT)
        cert_fns = [
            ("Certified ISO 13485 internal auditor; status: expired.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; revoked in 2024.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; expired in 2023.", "Certified ISO 13485 internal auditor"),
            ("Certified ISO 13485 internal auditor; suspended pending audit", "Certified ISO 13485 internal auditor"),
            ("持证内审员，不再有效", "持证内审员"),
            ("Certificate of ISO 13485 Internal Auditor Training, Cert. No. TRN-505", "Certificate of ISO 13485 Internal Auditor Training, Cert. No. TRN-505"),
        ]
        for line3, ev3 in cert_fns:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "M4-FN-CERT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev3, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # Positive statutory certs after training remain intact
        for pos in [
            "Attended ISO 13485 internal auditor training; Internal Auditor Certificate No. CERT-101",
            "Completed training course; Lead Auditor Certificate No. LA-442",
        ]:
            self.assertTrue(self.verifier.has_credential_proof(pos), pos)
            self.assertEqual(self.verifier.classify_verb_tier(pos), 3, pos)

        # QMS_LEAD semicolon suspended & QA review failure
        for line2 in [
            "Led 3 ISO 13485 internal audits; suspended pending audit",
            "Led 3 ISO 13485 internal audits; did not pass QA review.",
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "M4-FN-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led 3 ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # QA_VAL protocol failed & validation rejected by QA
        for line3 in [
            "Executed IQ/OQ/PQ validation; protocol failed.",
            "Executed IQ/OQ/PQ validation; validation rejected by QA.",
        ]:
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "M4-FN-VAL",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # QA_DEG incomplete / did not complete / expected year
        for line1 in [
            "B.Eng. in Biomedical Engineering; did not complete degree.",
            "B.Eng. in Biomedical Engineering; degree incomplete.",
            "B.Eng. in Biomedical Engineering; expected 2027.",
        ]:
            resume = f"{line1}\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "M4-FN-DEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng. in Biomedical Engineering", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line1)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line1)

        # QA_EXP < 1 year (English months & Chinese 不足1年)
        for line2, ev2 in [
            ("Less than 6 months Class III medical device QA Engineer experience.", "Class III medical device QA Engineer experience"),
            ("不足1年三类医疗器械QA工程师经验。", "三类医疗器械QA工程师经验"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "M4-FN-EXP",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

    def test_m4_p2_gap3_fp_cn_adverse_events_audit_findings_and_third_party_supervision_pass(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        for line2, ev2 in [
            ("4年三类医疗器械QA工程师，无医疗器械不良事件", "4年三类医疗器械QA工程师"),
            ("4年三类医疗器械QA工程师，无任何审计发现项", "4年三类医疗器械QA工程师"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "M4-FP-CN",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), line2)

        res_val = (
            "B.Eng.\n"
            "4 years Class III QA Engineer\n"
            "Executed IQ/OQ/PQ validation without third-party supervision\n"
            "Certified ISO 13485 internal auditor, Cert. No. CERT-101"
        )
        raw_val = {
            "candidate_id": "M4-FP-VAL",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation without third-party supervision", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_val, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_val, "QA/QC")
        v_val, vv_val = self.verifier.verify_card(res_val, m_val, "QA/QC")
        self.assertEqual(vv_val, [], vv_val)
        self.assertEqual(gate.step(v_val, "Confirm", proposed_decision=1).final_decision, "1")

        # Negative controls still blocked: "无医疗器械经验" and "without verification"
        self.assertFalse(self.verifier.check_span_alignment(["无医疗器械经验"], "医疗器械经验", [1, 1])[0])
        self.assertEqual(self.verifier.classify_verb_tier("Executed IQ/OQ/PQ validation without verification"), 0)

    def test_m4_p2_gap4_shorter_contiguous_span_alignment(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        src_line = "4 yrs Class III QA Eng; executed IQ/OQ/PQ validation and ISO 14971 CAPA closures."
        aligned, sim = self.verifier.check_span_alignment([src_line], "4 years Class III QA Engineer", [1, 1])
        self.assertTrue(aligned)
        self.assertGreaterEqual(sim, 0.85)

        resume = (
            "B.Eng.\n"
            f"{src_line}\n"
            "Certified ISO 13485 internal auditor, Cert. No. CERT-101"
        )
        raw = {
            "candidate_id": "M4-GAP4",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
        ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
        self.assertEqual(vv, [], vv)
        self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1")

        # Negated shorter source span must still be rejected
        self.assertFalse(
            self.verifier.check_span_alignment(
                ["No 4 yrs Class III QA Eng; executed IQ/OQ/PQ validation."],
                "4 years Class III QA Engineer",
                [1, 1],
            )[0]
        )

    def test_m4_p2_r2_gap1_comma_license_rejected_credential_degree_and_sub1yr_exp_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. QA_CERT FN probes (comma & semicolon license/credential rejected, incomplete, denied, status:, parenthetical, adjacent variants)
        qa_cert_fns = [
            "Certified ISO 13485 internal auditor, license rejected by registrar.",
            "Certified ISO 13485 internal auditor; license rejected by registrar.",
            "Certified ISO 13485 internal auditor; credential incomplete.",
            "Certified ISO 13485 internal auditor; credential rejected by registrar.",
            "Certified ISO 13485 internal auditor; license incomplete.",
            "Certified ISO 13485 internal auditor; status: incomplete.",
            "Certified ISO 13485 internal auditor, status: incomplete.",
            "Certified ISO 13485 internal auditor; status: rejected.",
            "Certified ISO 13485 internal auditor; cert. incomplete.",
            "Certified ISO 13485 internal auditor (incomplete)",
            "Certified ISO 13485 internal auditor (rejected)",
            "Certified ISO 13485 internal auditor; license denied.",
            "Certified ISO 13485 internal auditor; credential not completed.",
            "Certified ISO 13485 internal auditor; license revoked.",
            "Certified ISO 13485 internal auditor; license suspended.",
            "Certified ISO 13485 internal auditor; license withdrawn.",
            "Certified ISO 13485 internal auditor; pending approval.",
            "Certified ISO 13485 internal auditor; unverified.",
        ]
        for line3 in qa_cert_fns:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "GAP-R2-CERT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # 2. RA_REG FN probes (requires_cert_proof=False, min_verb_tier=3)
        for line2 in [
            "Independent Class II/III NMPA registration; license rejected by registrar.",
            "Independent Class II/III NMPA registration, license rejected by registrar.",
            "Independent Class II/III NMPA registration; credential incomplete.",
            "Independent Class II/III NMPA registration; license denied.",
        ]:
            resume = f"B.Med.\n{line2}\nDrafted 2 Class III CERs"
            raw = {
                "candidate_id": "GAP-R2-RA",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "Independent Class II/III NMPA registration", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": "Drafted 2 Class III CERs", "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 3. QA_DEG FN probes
        for line1 in [
            "B.Sc. Biomedical Engineering; credential incomplete.",
            "B.Sc. Biomedical Engineering; degree in progress.",
            "B.Sc. Biomedical Engineering; expected graduation in 2027.",
            "B.Sc. Biomedical Engineering, expected graduation in 2027.",
            "B.Sc. Biomedical Engineering; status: incomplete.",
            "B.Sc. Biomedical Engineering; degree not completed.",
            "B.Sc. Biomedical Engineering (incomplete)",
        ]:
            resume = f"{line1}\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "GAP-R2-DEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Sc. Biomedical Engineering", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line1)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line1)

        # 4. QA_EXP sub-1-year decimal & "< 1 year" FN probes (both stripped & verbatim quotes, plus trailing duration)
        for line2, ev2 in [
            ("0.5 years Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("0.5 years Class III QA Engineer experience.", "0.5 years Class III QA Engineer experience"),
            ("< 1 year Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("< 1 year Class III QA Engineer experience.", "< 1 year Class III QA Engineer experience"),
            ("Executed Class III QA work, 0.5 years.", "Executed Class III QA work"),
            ("Executed Class III QA work (< 1 year).", "Executed Class III QA work"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-101"
            raw = {
                "candidate_id": "GAP-R2-EXP",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-101", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line2, ev2))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, ev2))

        # 5. Positive controls preserved (s_term, vv=[])
        for line3, ev3 in [
            ("Certified ISO 13485 internal auditor, Cert. No. CERT-999; prior college application rejected.", "Certified ISO 13485 internal auditor, Cert. No. CERT-999"),
            ("Certified ISO 13485 internal auditor, Cert. No. CERT-999, prior college application rejected.", "Certified ISO 13485 internal auditor, Cert. No. CERT-999"),
        ]:
            resume = f"B.Sc. Biomedical Engineering\n10.5 years Class III QA Engineer experience\n{line3}"
            raw = {
                "candidate_id": "GAP-R2-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Sc. Biomedical Engineering", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "10.5 years Class III QA Engineer experience", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": ev3, "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line3, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), line3)

    def test_m4_p2_r3_gap1_domain_lexical_and_leading_dot_parity(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. RA_REG domain invalidation parity (EN & CN, comma & semicolon)
        for line2 in [
            "Independent Class II/III NMPA registration, registration rejected by NMPA.",
            "Independent Class II/III NMPA registration; registration suspended.",
            "Independent Class II/III NMPA registration; filing rejected.",
            "Independent Class II/III NMPA registration; submission rejected.",
            "Independent Class II/III NMPA registration, approval denied.",
        ]:
            resume = f"B.Med.\n{line2}\nDrafted 2 Class III CERs"
            raw = {
                "candidate_id": "GAP-R3-RA-EN",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "Independent Class II/III NMPA registration", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": "Drafted 2 Class III CERs", "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        for line2 in [
            "独立主导II/III类NMPA注册申报，注册被驳回。",
            "独立主导II/III类NMPA注册申报，申报被拒绝。",
            "独立主导II/III类NMPA注册申报，注册未获批。",
            "独立主导II/III类NMPA注册申报，注册已撤销。",
            "独立主导II/III类NMPA注册申报，注册被暂停。",
            "独立主导II/III类NMPA注册申报，注册失败。",
            "独立主导II/III类NMPA注册申报，审批未通过。",
            "独立主导II/III类NMPA注册申报，未完成注册。",
            "独立主导II/III类NMPA注册申报，已作废。",
        ]:
            resume = f"医学本科\n{line2}\n撰写2份三类CER"
            raw = {
                "candidate_id": "GAP-R3-RA-CN",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "医学本科", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "独立主导II/III类NMPA注册申报", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": "撰写2份三类CER", "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 2. QMS_LEAD audit approval / authorization / audit failure parity
        for line2 in [
            "Led ISO 13485 internal audits, audit approval revoked.",
            "Led ISO 13485 internal audits; authorization revoked.",
            "Led ISO 13485 internal audits; audit failed.",
            "Led ISO 13485 internal audits; audit incomplete.",
            "Led ISO 13485 internal audits; authorization suspended.",
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "GAP-R3-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 3. QA_CERT temporal adverbs before invalidation states
        for line3 in [
            "Certified ISO 13485 internal auditor, Cert. No. CERT-999; later revoked.",
            "Certified ISO 13485 internal auditor, Cert. No. CERT-999; subsequently suspended.",
            "Certified ISO 13485 internal auditor, Cert. No. CERT-999, later revoked.",
            "Certified ISO 13485 internal auditor, Cert. No. CERT-999; recently expired.",
            "Certified ISO 13485 internal auditor, Cert. No. CERT-999; renewal rejected.",
        ]:
            self.assertFalse(self.verifier.has_credential_proof(line3), line3)
            self.assertLess(self.verifier.classify_verb_tier(line3), 3, line3)
            resume = f"B.Eng.\n4 years Class III QA Engineer\n{line3}"
            raw = {
                "candidate_id": "GAP-R3-CERT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-999", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line3)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line3)

        # 4. QA_DEG parenthetical and comma expected year (both stripped and verbatim)
        for line1, ev1 in [
            ("B.Sc. Biomedical Engineering (expected 2027)", "B.Sc. Biomedical Engineering"),
            ("B.Sc. Biomedical Engineering (expected 2027)", "B.Sc. Biomedical Engineering (expected 2027)"),
            ("B.Sc. Biomedical Engineering, expected 2027", "B.Sc. Biomedical Engineering"),
            ("B.Sc. Biomedical Engineering (expected May 2027)", "B.Sc. Biomedical Engineering"),
        ]:
            resume = f"{line1}\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-999"
            raw = {
                "candidate_id": "GAP-R3-DEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": ev1, "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-999", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line1, ev1))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line1, ev1))

        # 5. QA_EXP leading-dot <1yr decimals & standalone <12mo / 0 yrs
        for line2, ev2 in [
            (".5 years Class III QA Engineer experience.", "Class III QA Engineer experience"),
            (".5 years Class III QA Engineer experience.", ".5 years Class III QA Engineer experience"),
            (".5 yrs Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("0 yrs Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("Executed Class III QA work, .5 yrs.", "Executed Class III QA work"),
            ("6个月三类医疗器械QA工程师经验。", "三类医疗器械QA工程师经验"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-999"
            raw = {
                "candidate_id": "GAP-R3-EXP",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-999", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line2, ev2))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, ev2))

        # 6. Positive controls preserved (s_term, vv=[])
        pos_qa_cases = [
            (
                "B.Sc. Biomedical Engineering (graduated 2019)",
                "10.5 years Class III QA Engineer experience.",
                "10.5 years Class III QA Engineer experience",
                "Certified ISO 13485 internal auditor, Cert. No. CERT-999; prior college application rejected.",
            ),
            (
                "B.Sc. Biomedical Engineering, completed 2019",
                ".5 years cleanroom + 4 years Class III medical device QA Engineer experience.",
                ".5 years cleanroom + 4 years Class III medical device QA Engineer experience",
                "Certified ISO 13485 internal auditor, Cert. No. CERT-999, prior college application rejected.",
            ),
            (
                "B.Sc. Biomedical Engineering (graduated 2019)",
                "3 years 6 months Class III medical device QA Engineer experience.",
                "3 years 6 months Class III medical device QA Engineer experience",
                "Certified ISO 13485 internal auditor, Cert. No. CERT-999",
            ),
            (
                "B.Sc. Biomedical Engineering (graduated 2019)",
                "6 months cleanroom + 4 years Class III medical device QA Engineer experience.",
                "6 months cleanroom + 4 years Class III medical device QA Engineer experience",
                "Certified ISO 13485 internal auditor, Cert. No. CERT-999",
            ),
        ]
        for line1, line2, ev2, line3 in pos_qa_cases:
            resume = f"{line1}\n{line2}\n{line3}"
            raw = {
                "candidate_id": "GAP-R3-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": line1, "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-999", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line1, line2, line3, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), (line1, line2, line3))

        # Positive QMS & RA controls preserved
        pos_qms_res = "Header\nLed 2 ISO 13485 internal audits; zero audit findings\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
        pos_qms_raw = {
            "candidate_id": "GAP-R3-POS-QMS",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led 2 ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(pos_qms_raw, "QMS")
        v_qms, vv_qms = self.verifier.verify_card(pos_qms_res, m_qms, "QMS")
        self.assertEqual(vv_qms, [])
        self.assertEqual(gate.step(v_qms, "Confirm", proposed_decision=1).final_decision, "1")

        pos_ra_res = "B.Med.\nIndependent Class II/III NMPA registration; 3 Class III approvals obtained.\nDrafted 2 Class III CERs"
        pos_ra_raw = {
            "candidate_id": "GAP-R3-POS-RA",
            "role": "RA",
            "entries": [
                {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "RA_REG", "evidence": "Independent Class II/III NMPA registration", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "RA_CER", "evidence": "Drafted 2 Class III CERs", "anchor_lines": [3, 3], "status": "verified"},
            ],
        }
        m_ra, _ = OntologySchemaMasker.enforce_mask_and_validate(pos_ra_raw, "RA")
        v_ra, vv_ra = self.verifier.verify_card(pos_ra_res, m_ra, "RA")
        self.assertEqual(vv_ra, [])
        self.assertEqual(gate.step(v_ra, "Confirm", proposed_decision=1).final_decision, "1")

    def test_m4_p2_r4_gap1_multi_year_sub_rotation_and_duration_boundary_parity(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. All 11 GAP-R4-1 realistic multi-year positive controls + prior positive controls must pass
        multi_year_pos = [
            ("4 years Class III QA Engineer; 6 months internship in cleanroom.", "4 years Class III QA Engineer; 6 months internship in cleanroom"),
            ("4 years Class III QA Engineer; 6 months internship in cleanroom.", "4 years Class III QA Engineer"),
            ("4 years Class III QA Engineer, 6 months internship.", "4 years Class III QA Engineer, 6 months internship"),
            ("4 years Class III QA Engineer, 6 months internship.", "4 years Class III QA Engineer"),
            ("4 years QA Engineer after 6 months cleanroom training.", "4 years QA Engineer after 6 months cleanroom training"),
            ("4 years QA Engineer after 6 months cleanroom training.", "4 years QA Engineer"),
            ("4 years Class III QA Engineer, including 6 months cleanroom monitoring.", "4 years Class III QA Engineer, including 6 months cleanroom monitoring"),
            ("4 years Class III QA Engineer, including 6 months cleanroom monitoring.", "4 years Class III QA Engineer"),
            ("4 years Class III QA Engineer (6 months sterile packaging validation).", "4 years Class III QA Engineer (6 months sterile packaging validation)"),
            ("4 years Class III QA Engineer (6 months sterile packaging validation).", "4 years Class III QA Engineer"),
            ("5 years Class III medical device QA Engineer, with 6 months ISO 13485 supplier audit.", "5 years Class III medical device QA Engineer, with 6 months ISO 13485 supplier audit"),
            ("5 years Class III medical device QA Engineer, with 6 months ISO 13485 supplier audit.", "5 years Class III medical device QA Engineer"),
            ("4 years Class III medical device QA Engineer + 6 months cleanroom.", "4 years Class III medical device QA Engineer + 6 months cleanroom"),
            ("4 years Class III medical device QA Engineer + .5 years cleanroom.", "4 years Class III medical device QA Engineer + .5 years cleanroom"),
            ("4 years QA Engineer (6 months cleanroom training).", "4 years QA Engineer (6 months cleanroom training)"),
            ("4年三类医疗器械QA工程师，含6个月洁净室轮岗经验。", "4年三类医疗器械QA工程师，含6个月洁净室轮岗经验"),
            ("4年三类医疗器械QA工程师（含6个月实习）。", "4年三类医疗器械QA工程师（含6个月实习）"),
            ("6个月实习+4年三类医疗器械QA工程师经验。", "6个月实习+4年三类医疗器械QA工程师经验"),
            ("3 years 6 months Class III QA Engineer experience.", "3 years 6 months Class III QA Engineer experience"),
            ("6 months cleanroom + 4 years Class III medical device QA Engineer experience.", "6 months cleanroom + 4 years Class III medical device QA Engineer experience"),
            ("0.5 years cleanroom + 4 years Class III medical device QA Engineer experience.", "0.5 years cleanroom + 4 years Class III medical device QA Engineer experience"),
            (".5 years cleanroom + 4 years Class III medical device QA Engineer experience.", ".5 years cleanroom + 4 years Class III medical device QA Engineer experience"),
            ("2020.05 - 2025.05: 5 years Class III QA Engineer.", "2020.05 - 2025.05: 5 years Class III QA Engineer"),
        ]
        for line2, ev2 in multi_year_pos:
            resume = (
                "B.Eng. Biomedical Engineering\n"
                f"{line2}\n"
                "Certified ISO 13485 internal auditor, Cert. No. CERT-1001, issued 2021.05."
            )
            raw = {
                "candidate_id": "GAP-R4-1-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001, issued 2021.05", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, ev2, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), (line2, ev2))

        # 2. Standalone insufficient durations & strict <1yr/0yr clauses remain blocked
        insufficient_neg = [
            ("6 months Class III QA Engineer experience.", "6 months Class III QA Engineer experience"),
            ("6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            (".5 years Class III QA Engineer experience.", ".5 years Class III QA Engineer experience"),
            (".5 years Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("0 years Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("< 1 year Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("4 years retail sales, 6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("4 years retail sales; 6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("4 years retail sales, < 1 year Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("6个月三类医疗器械QA工程师经验。", "三类医疗器械QA工程师经验"),
        ]
        for line2, ev2 in insufficient_neg:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001."
            raw = {
                "candidate_id": "GAP-R4-1-NEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line2, ev2))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, ev2))

    def test_m4_p2_r4_gap2_domain_report_dossier_application_invalidation_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. QMS_LEAD audit/CAPA/inspection report rejection/suspension
        for line2 in [
            "Led ISO 13485 internal audits; audit report rejected.",
            "Led ISO 13485 internal audits, audit report suspended.",
            "Led ISO 13485 internal audits; CAPA report rejected.",
            "Led ISO 13485 internal audits; inspection report rejected.",
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "GAP-R4-2-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 2. RA_REG & RA_CER dossier / application / CER report rejections
        for line2, line3, ev2, ev3 in [
            ("Independent Class II/III NMPA registration; dossier rejected.", "Drafted 2 Class III CERs", "Independent Class II/III NMPA registration", "Drafted 2 Class III CERs"),
            ("Independent Class II/III NMPA registration, registration application denied.", "Drafted 2 Class III CERs", "Independent Class II/III NMPA registration", "Drafted 2 Class III CERs"),
            ("Independent Class II/III NMPA registration; NMPA application rejected.", "Drafted 2 Class III CERs", "Independent Class II/III NMPA registration", "Drafted 2 Class III CERs"),
            ("Independent Class II/III NMPA registration", "Drafted Class III CER; CER report rejected.", "Independent Class II/III NMPA registration", "Drafted Class III CER"),
        ]:
            resume = f"B.Med.\n{line2}\n{line3}"
            raw = {
                "candidate_id": "GAP-R4-2-RA",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": ev3, "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, (line2, line3))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, line3))

        # 3. QA_VAL validation/test report rejection
        for line4 in [
            "Executed IQ/OQ/PQ validation; validation report rejected.",
            "Executed IQ/OQ/PQ validation; test report rejected.",
        ]:
            resume = (
                "B.Eng.\n"
                "4 years Class III QA Engineer\n"
                "Certified ISO 13485 internal auditor, Cert. No. CERT-1001\n"
                f"{line4}"
            )
            raw = {
                "candidate_id": "GAP-R4-2-VAL",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [4, 4], "status": "verified"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line4)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line4)

        # 4. Unrelated "prior college application rejected" positive control preserved
        pos_res = "B.Eng.\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001; prior college application rejected."
        pos_raw = {
            "candidate_id": "GAP-R4-2-POS",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
            ],
        }
        m_pos, _ = OntologySchemaMasker.enforce_mask_and_validate(pos_raw, "QA/QC")
        v_pos, vv_pos = self.verifier.verify_card(pos_res, m_pos, "QA/QC")
        self.assertEqual(vv_pos, [])
        self.assertEqual(gate.step(v_pos, "Confirm", proposed_decision=1).final_decision, "1")

    def test_m4_p2_r5_gap1_plural_copula_colon_and_bare_domain_noun_invalidation_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. QMS_LEAD plural copulas (were / have been / have not been) & colon status
        for line2 in [
            "Led ISO 13485 internal audits; audit reports were rejected.",
            "Led ISO 13485 internal audits; CAPA reports have been rejected.",
            "Led ISO 13485 internal audits; audit report: rejected.",
            "Led ISO 13485 internal audits; audit reports have not been approved.",
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "GAP-R5-1-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 2. RA_REG & RA_CER plural copulas + bare CER invalidation
        for line2, line3, ev2, ev3 in [
            ("Independent Class II/III NMPA registration; dossiers were rejected.", "Drafted 2 Class III CERs", "Independent Class II/III NMPA registration", "Drafted 2 Class III CERs"),
            ("Independent Class II/III NMPA registration; registration applications were denied.", "Drafted 2 Class III CERs", "Independent Class II/III NMPA registration", "Drafted 2 Class III CERs"),
            ("Independent Class II/III NMPA registration", "Drafted Class III CER; CER rejected.", "Independent Class II/III NMPA registration", "Drafted Class III CER"),
            ("Independent Class II/III NMPA registration", "Drafted Class III CER; CER: rejected.", "Independent Class II/III NMPA registration", "Drafted Class III CER"),
        ]:
            resume = f"B.Med.\n{line2}\n{line3}"
            raw = {
                "candidate_id": "GAP-R5-1-RA",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": ev3, "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, (line2, line3))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, line3))

        # 3. QA_VAL & QA_CAPA bare IQ/OQ/PQ, bare CAPA, and plural report rejections
        for rid, line4, ev4 in [
            ("QA_VAL", "Executed IQ/OQ/PQ validation; validation reports were rejected.", "Executed IQ/OQ/PQ validation"),
            ("QA_VAL", "Executed IQ/OQ/PQ validation; IQ/OQ/PQ failed.", "Executed IQ/OQ/PQ validation"),
            ("QA_CAPA", "Executed ISO 14971 CAPA; CAPA rejected.", "Executed ISO 14971 CAPA"),
            ("QA_CAPA", "Executed ISO 14971 CAPA; CAPA reports were rejected.", "Executed ISO 14971 CAPA"),
        ]:
            resume = (
                "B.Eng.\n"
                "4 years Class III QA Engineer\n"
                "Certified ISO 13485 internal auditor, Cert. No. CERT-1001\n"
                f"{line4}"
            )
            raw = {
                "candidate_id": "GAP-R5-1-QA",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": ev4 if rid == "QA_VAL" else "", "anchor_lines": [4, 4] if rid == "QA_VAL" else [], "status": "verified" if rid == "QA_VAL" else "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": ev4 if rid == "QA_CAPA" else "", "anchor_lines": [4, 4] if rid == "QA_CAPA" else [], "status": "verified" if rid == "QA_CAPA" else "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, line4)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line4)

        # 4. Positive controls preserved (s_term, 1, vv=[])
        # CAPA closed & validation report approved
        res_cap_pos = (
            "B.Eng.\n"
            "4 years Class III QA Engineer\n"
            "Certified ISO 13485 internal auditor, Cert. No. CERT-1001; prior college application rejected.\n"
            "Executed IQ/OQ/PQ validation; validation report approved.\n"
            "Executed ISO 14971 CAPA; CAPA closed."
        )
        raw_cap_pos = {
            "candidate_id": "GAP-R5-1-POS-QA",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "Executed ISO 14971 CAPA", "anchor_lines": [5, 5], "status": "verified"},
            ],
        }
        m_cap, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_cap_pos, "QA/QC")
        v_cap, vv_cap = self.verifier.verify_card(res_cap_pos, m_cap, "QA/QC")
        self.assertEqual(vv_cap, [], vv_cap)
        self.assertEqual(gate.step(v_cap, "Confirm", proposed_decision=1).final_decision, "1")

        # CER report approved & audit report approved
        res_ra_pos = "B.Med.\nIndependent Class II/III NMPA registration\nDrafted Class III CER; CER report approved."
        raw_ra_pos = {
            "candidate_id": "GAP-R5-1-POS-RA",
            "role": "RA",
            "entries": [
                {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "RA_REG", "evidence": "Independent Class II/III NMPA registration", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "RA_CER", "evidence": "Drafted Class III CER", "anchor_lines": [3, 3], "status": "verified"},
            ],
        }
        m_ra, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_ra_pos, "RA")
        v_ra, vv_ra = self.verifier.verify_card(res_ra_pos, m_ra, "RA")
        self.assertEqual(vv_ra, [], vv_ra)
        self.assertEqual(gate.step(v_ra, "Confirm", proposed_decision=1).final_decision, "1")

        res_qms_pos = "Header\nLed ISO 13485 internal audits; audit report approved.\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
        raw_qms_pos = {
            "candidate_id": "GAP-R5-1-POS-QMS",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qms_pos, "QMS")
        v_qms, vv_qms = self.verifier.verify_card(res_qms_pos, m_qms, "QMS")
        self.assertEqual(vv_qms, [], vv_qms)
        self.assertEqual(gate.step(v_qms, "Confirm", proposed_decision=1).final_decision, "1")

    def test_m4_p2_r5_gap2_subyear_qa_exp_for_past_last_fewer_blocked_and_positive_preserved(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. All GAP-R5-2 sub-year QA_EXP FN probes must block
        subyear_qa_exp_fns = [
            ("Worked as Class III QA Engineer for 6 months.", "Worked as Class III QA Engineer for 6 months"),
            ("Worked as Class III QA Engineer for 6 months.", "Class III QA Engineer"),
            ("Class III QA Engineer for 8 months.", "Class III QA Engineer"),
            ("Served as Class III QA Engineer over the past 6 months.", "Class III QA Engineer"),
            ("Class III QA Engineer in the last 6 months.", "Class III QA Engineer"),
            ("Fewer than 6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("Worked as Class III QA Engineer for six months.", "Class III QA Engineer"),
            ("Class III QA Engineer over the last 8 months.", "Class III QA Engineer"),
        ]
        for line2, ev2 in subyear_qa_exp_fns:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001"
            raw = {
                "candidate_id": "GAP-R5-2-NEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line2, ev2))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, ev2))

        # 2. Positive non-QA_EXP duration controls preserved: valid for 6 months (QA_CERT),
        #    every 6 months (QMS_LEAD), Executed IQ/OQ/PQ validation for 6 months (QA_VAL)
        for ev_cert in (
            "Certified ISO 13485 internal auditor, Cert. No. CERT-1001",
            "Certified ISO 13485 internal auditor, Cert. No. CERT-1001, valid for 6 months",
        ):
            for ev_val in ("Executed IQ/OQ/PQ validation", "Executed IQ/OQ/PQ validation for 6 months"):
                resume = (
                    "B.Eng.\n"
                    "4 years Class III QA Engineer\n"
                    "Certified ISO 13485 internal auditor, Cert. No. CERT-1001, valid for 6 months.\n"
                    "Executed IQ/OQ/PQ validation for 6 months."
                )
                raw = {
                    "candidate_id": "GAP-R5-2-POS-QA",
                    "role": "QA/QC",
                    "entries": [
                        {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                        {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                        {"req_id": "QA_CERT", "evidence": ev_cert, "anchor_lines": [3, 3], "status": "verified"},
                        {"req_id": "QA_VAL", "evidence": ev_val, "anchor_lines": [4, 4], "status": "verified"},
                        {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    ],
                }
                masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
                ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
                self.assertEqual(vv, [], (ev_cert, ev_val, vv))
                res = gate.step(ver, "Confirm", proposed_decision=1)
                self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"))

        res_qms = "Header\nLed ISO 13485 internal audits every 6 months.\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
        raw_qms = {
            "candidate_id": "GAP-R5-2-POS-QMS",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits every 6 months", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qms, "QMS")
        v_qms, vv_qms = self.verifier.verify_card(res_qms, m_qms, "QMS")
        self.assertEqual(vv_qms, [], vv_qms)
        self.assertEqual(gate.step(v_qms, "Confirm", proposed_decision=1).final_decision, "1")

        # 3. Unseen R4/R5 multi-year sub-rotation positive controls preserved
        for line2, ev2 in [
            ("4 years Class III QA Engineer, 6 months' internship.", "4 years Class III QA Engineer, 6 months' internship"),
            ("3 yrs 6 mos Class III QA Engineer.", "3 yrs 6 mos Class III QA Engineer"),
            ("4 years Class III QA Engineer (0.5 years cleanroom rotation).", "4 years Class III QA Engineer (0.5 years cleanroom rotation)"),
            ("4年三类医疗器械QA工程师，包括6个月洁净室见习。", "4年三类医疗器械QA工程师，包括6个月洁净室见习"),
        ]:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001"
            raw = {
                "candidate_id": "GAP-R5-2-POS-ROT",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, vv))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).final_decision, "1", line2)

    def test_m4_p2_r6_gap1_copula_plural_contraction_lifecycle_invalidation_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. QMS_LEAD plural copulas, lifecycle states (not valid/active), contractions, temporal adv, status without colon
        for line2, line3 in [
            ("Led ISO 13485 internal audits; audit reports remain pending.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit reports remain rejected.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report is not valid.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit approval is not active.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report hasn't been approved.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report wasn't approved.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report is still pending.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report had been rejected.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; audit report will not be approved.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits, audit reports remain pending.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits (audit report not valid).", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits; status is not valid.", "Certified ISO 13485 auditor, Cert. No. CERT-1"),
            ("Led ISO 13485 internal audits", "Certified ISO 13485 auditor, Cert. No. CERT-1; certificates remain invalid."),
            ("Led ISO 13485 internal audits", "Certified ISO 13485 auditor, Cert. No. CERT-1; licenses remain suspended."),
            ("Led ISO 13485 internal audits", "Certified ISO 13485 auditor, Cert. No. CERT-1; certificate is not valid."),
            ("Led ISO 13485 internal audits", "Certified ISO 13485 auditor, Cert. No. CERT-1; certificates were revoked."),
        ]:
            resume = f"Header\n{line2}\n{line3}\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "GAP-R6-1-QMS",
                "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, (line2, line3))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, line3))

        # 2. RA_REG plural copulas & lifecycle states
        for line2 in [
            "Independent Class II/III NMPA registration; dossiers remain incomplete.",
            "Independent Class II/III NMPA registration; dossier is not valid.",
            "Independent Class II/III NMPA registration; registrations remain pending.",
        ]:
            resume = f"B.Med.\n{line2}\nDrafted 2 Class III CERs"
            raw = {
                "candidate_id": "GAP-R6-1-RA",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "Independent Class II/III NMPA registration", "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "RA_CER", "evidence": "Drafted 2 Class III CERs", "anchor_lines": [3, 3], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "RA")
            ver, vv = self.verifier.verify_card(resume, masked, "RA")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 3. QA_CAPA have not been closed
        res_capa = (
            "B.Eng.\n"
            "4 years Class III QA Engineer\n"
            "Certified ISO 13485 internal auditor, Cert. No. CERT-1001\n"
            "Executed ISO 14971 CAPA; CAPA reports have not been closed."
        )
        raw_capa = {
            "candidate_id": "GAP-R6-1-CAPA",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QA_CAPA", "evidence": "Executed ISO 14971 CAPA", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_capa, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_capa, "QA/QC")
        v_capa, vv_capa = self.verifier.verify_card(res_capa, m_capa, "QA/QC")
        self.assertTrue(vv_capa)
        self.assertEqual(gate.step(v_capa, "Confirm", proposed_decision=1).next_state, "s_block")

    def test_m4_p2_r6_gap1_chinese_neither_nor_and_positive_controls(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        # 1. Chinese invalidation parity (lowercase capa/cer/iq in .lower() regexes)
        for role, line_txt, req_id, ev_txt in [
            ("QMS", "Led ISO 13485 internal audits; 审计报告无效。", "QMS_LEAD", "Led ISO 13485 internal audits"),
            ("QMS", "Led ISO 13485 internal audits; 审计报告已驳回。", "QMS_LEAD", "Led ISO 13485 internal audits"),
            ("RA", "Independent Class II/III NMPA registration; 注册申请已撤回。", "RA_REG", "Independent Class II/III NMPA registration"),
            ("QA/QC", "Executed ISO 14971 CAPA; CAPA未关闭。", "QA_CAPA", "Executed ISO 14971 CAPA"),
        ]:
            if role == "QMS":
                resume = f"Header\n{line_txt}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
                raw = {
                    "candidate_id": "CN-NEG-QMS", "role": "QMS",
                    "entries": [
                        {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                        {"req_id": "QMS_LEAD", "evidence": ev_txt, "anchor_lines": [2, 2], "status": "verified"},
                        {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                    ],
                }
            elif role == "RA":
                resume = f"B.Med.\n{line_txt}\nDrafted 2 Class III CERs"
                raw = {
                    "candidate_id": "CN-NEG-RA", "role": "RA",
                    "entries": [
                        {"req_id": "RA_DEG", "evidence": "B.Med.", "anchor_lines": [1, 1], "status": "verified"},
                        {"req_id": "RA_REG", "evidence": ev_txt, "anchor_lines": [2, 2], "status": "verified"},
                        {"req_id": "RA_CER", "evidence": "Drafted 2 Class III CERs", "anchor_lines": [3, 3], "status": "verified"},
                    ],
                }
            else:
                resume = f"B.Eng.\n4 years Class III QA Engineer\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001\n{line_txt}"
                raw = {
                    "candidate_id": "CN-NEG-QA", "role": "QA/QC",
                    "entries": [
                        {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                        {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                        {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                        {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                        {"req_id": "QA_CAPA", "evidence": ev_txt, "anchor_lines": [4, 4], "status": "verified"},
                    ],
                }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, role)
            ver, vv = self.verifier.verify_card(resume, masked, role)
            self.assertTrue(vv, (role, line_txt))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (role, line_txt))

        # 2. Sibling negation: neither/nor & 没有 in classify_verb_tier & verify_card
        self.assertEqual(self.verifier.classify_verb_tier("Neither led nor directed ISO 13485 internal audits."), 0)
        self.assertEqual(self.verifier.classify_verb_tier("没有实际参加或者独立主导ISO 13485内审。"), 0)
        for line2, ev2 in [
            ("Neither led nor directed ISO 13485 internal audits.", "ISO 13485 internal audits"),
            ("没有实际参加或者独立主导ISO 13485内审。", "ISO 13485内审"),
        ]:
            resume = f"Header\n{line2}\nCertified ISO 13485 auditor, Cert. No. CERT-1\nMaintained ISO 13485 system."
            raw = {
                "candidate_id": "SIB-NEG", "role": "QMS",
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QMS_LEAD", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
            ver, vv = self.verifier.verify_card(resume, masked, "QMS")
            self.assertTrue(vv, line2)
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", line2)

        # 3. Positive controls preserved (including _CERT_NEG_RE safety on unrelated "not approved")
        self.assertFalse(
            self.verifier._CERT_NEG_RE.search("certified iso 13485 internal auditor, prior college application not approved.")
        )
        pos_qa = (
            "B.Eng.\n"
            "4 years Class III QA Engineer\n"
            "Certified ISO 13485 internal auditor, Cert. No. CERT-1001, certificates valid and active; prior college application not approved.\n"
            "Executed IQ/OQ/PQ validation for 6 months; validation reports were approved; IQ/OQ/PQ passed.\n"
            "Executed ISO 14971 CAPA; CAPAs closed; CAPA均已关闭，无未关闭不符合项。"
        )
        raw_qa = {
            "candidate_id": "R6-POS-QA", "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA Engineer", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_VAL", "evidence": "Executed IQ/OQ/PQ validation", "anchor_lines": [4, 4], "status": "verified"},
                {"req_id": "QA_CAPA", "evidence": "Executed ISO 14971 CAPA", "anchor_lines": [5, 5], "status": "verified"},
            ],
        }
        m_qa, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qa, "QA/QC")
        v_qa, vv_qa = self.verifier.verify_card(pos_qa, m_qa, "QA/QC")
        self.assertEqual(vv_qa, [], vv_qa)
        self.assertEqual(gate.step(v_qa, "Confirm", proposed_decision=1).final_decision, "1")

        pos_qms = (
            "Header\n"
            "Led ISO 13485 internal audits; audit reports were approved; audit report was not rejected; 无CAPA未关闭；验证报告已批准。\n"
            "Certified ISO 13485 auditor, Cert. No. CERT-1; 证书有效\n"
            "Maintained ISO 13485 system."
        )
        raw_qms = {
            "candidate_id": "R6-POS-QMS", "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Cert. No. CERT-1", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [4, 4], "status": "verified"},
            ],
        }
        m_qms, _ = OntologySchemaMasker.enforce_mask_and_validate(raw_qms, "QMS")
        v_qms, vv_qms = self.verifier.verify_card(pos_qms, m_qms, "QMS")
        self.assertEqual(vv_qms, [], vv_qms)
        self.assertEqual(gate.step(v_qms, "Confirm", proposed_decision=1).final_decision, "1")

    def test_m4_p2_r6_gap2_subyear_fp_apostrophe_plus_and_cn_positives(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        positives = [
            ("4 years Class III QA Engineer. Promoted to QA team lead within 6 months.", "Class III QA Engineer"),
            ("4 years Class III QA Engineer. Promoted to QA team lead 6 months later.", "Class III QA Engineer"),
            ("4 years Class III QA Engineer. Promoted to QA team lead 6 months ago.", "Class III QA Engineer"),
            ("4 years Class III QA Engineer with 9 months' cleanroom training.", "4 years Class III QA Engineer with 9 months' cleanroom training"),
            ("3+ years Class III QA Engineer, including 6 months cleanroom monitoring.", "3+ years Class III QA Engineer, including 6 months cleanroom monitoring"),
            ("3+ years Class III QA Engineer, including 6 months cleanroom monitoring.", "3+ years Class III QA Engineer"),
            ("3 years and 6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("十二个月三类医疗器械QA工程师工作经验。", "三类医疗器械QA工程师工作经验"),
            ("二十四个月三类医疗器械QA工程师工作经验。", "三类医疗器械QA工程师工作经验"),
        ]
        for line2, ev2 in positives:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001"
            raw = {
                "candidate_id": "GAP-R6-2-POS",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertEqual(vv, [], (line2, ev2, vv))
            res = gate.step(ver, "Confirm", proposed_decision=1)
            self.assertEqual((res.next_state, res.final_decision), ("s_term", "1"), (line2, ev2))

    def test_m4_p2_r6_gap2_subyear_fns_and_cross_clause_bypasses_blocked(self):
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

        negatives = [
            ("Worked as Class III QA Engineer for 6 months before moving.", "Class III QA Engineer"),
            ("Class III QA Engineer 6 months ago.", "Class III QA Engineer"),
            ("Worked as Class III QA Engineer for 6 months prior to graduation.", "Class III QA Engineer"),
            ("Completed onboarding and 6 months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("After 6 months as Class III QA Engineer, left the company.", "Class III QA Engineer"),
            ("6+ months Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("0.5+ years Class III QA Engineer experience.", "Class III QA Engineer experience"),
            ("六个月三类医疗器械QA工程师工作经验。", "三类医疗器械QA工程师工作经验"),
            ("4 years retail sales, worked as Class III QA Engineer for 6 months.", "Class III QA Engineer"),
            ("4年销售经验，担任三类医疗器械QA工程师6个月。", "三类医疗器械QA工程师"),
            ("4 years retail sales. Worked as Class III QA Engineer. QA Engineer role lasted 6 months.", "Worked as Class III QA Engineer"),
        ]
        for line2, ev2 in negatives:
            resume = f"B.Eng.\n{line2}\nCertified ISO 13485 internal auditor, Cert. No. CERT-1001"
            raw = {
                "candidate_id": "GAP-R6-2-NEG",
                "role": "QA/QC",
                "entries": [
                    {"req_id": "QA_DEG", "evidence": "B.Eng.", "anchor_lines": [1, 1], "status": "verified"},
                    {"req_id": "QA_EXP", "evidence": ev2, "anchor_lines": [2, 2], "status": "verified"},
                    {"req_id": "QA_CERT", "evidence": "Certified ISO 13485 internal auditor, Cert. No. CERT-1001", "anchor_lines": [3, 3], "status": "verified"},
                    {"req_id": "QA_VAL", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QA_CAPA", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            }
            masked, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
            ver, vv = self.verifier.verify_card(resume, masked, "QA/QC")
            self.assertTrue(vv, (line2, ev2))
            self.assertEqual(gate.step(ver, "Confirm", proposed_decision=1).next_state, "s_block", (line2, ev2))


class TestFSMDecisionGateHardening(unittest.TestCase):
    def setUp(self):
        self.gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        self.clean_card = {
            "candidate_id": "FSM-1",
            "role": "QA/QC",
            "entries": [
                {"req_id": "QA_DEG", "tier": "M", "status": "verified"},
                {"req_id": "QA_EXP", "tier": "M", "status": "verified"},
                {"req_id": "QA_CERT", "tier": "M", "status": "verified"},
                {"req_id": "QA_VAL", "tier": "P", "status": "verified"},
                {"req_id": "QA_CAPA", "tier": "P", "status": "not_provided"},
            ],
            "verifier_violations": [],
        }

    def test_invalid_proposed_decision_blocked(self):
        for bad in (None, 2, -1, True, False, "1", 1.0, 0.0):
            for action in ("Confirm", "Adjust"):
                res = self.gate.step(self.clean_card, action, proposed_decision=bad, reason="Valid long justification text here")
                self.assertEqual(res.next_state, "s_block")
                self.assertEqual(res.error_code, "ERR_INVALID_DECISION")

    def test_empty_entries_mandatory_guard_blocked(self):
        empty_card = {"candidate_id": "EMPTY", "role": "QA/QC", "entries": [], "verifier_violations": []}
        res_confirm = self.gate.step(empty_card, "Confirm", proposed_decision=1)
        self.assertEqual(res_confirm.next_state, "s_block")
        self.assertEqual(res_confirm.error_code, "ERR_MANDATORY_GUARD_FAILED")
        res_adjust = self.gate.step(
            empty_card, "Adjust", proposed_decision=1, reason="Valid long justification text here"
        )
        self.assertEqual(res_adjust.next_state, "s_block")
        self.assertEqual(res_adjust.error_code, "ERR_MANDATORY_GUARD_FAILED")

    def test_adjust_invalid_enum_and_tier_p_conflicting_blocked(self):
        res_bad_enum = self.gate.step(
            self.clean_card,
            "Adjust",
            proposed_decision=1,
            reason="Manager approved equivalent qualification",
            adjusted_statuses={"QA_VAL": "waived"},
        )
        self.assertEqual(res_bad_enum.next_state, "s_block")
        self.assertEqual(res_bad_enum.error_code, "ERR_INVALID_ENUM_STATUS")

        res_tier_p = self.gate.step(
            self.clean_card,
            "Adjust",
            proposed_decision=1,
            reason="Manager approved equivalent qualification",
            adjusted_statuses={"QA_VAL": "conflicting"},
        )
        self.assertEqual(res_tier_p.next_state, "s_block")
        self.assertEqual(res_tier_p.error_code, "ERR_UNRESOLVED_CONFLICT_LOCK")

    def test_adjust_short_reason_and_valid_transitions(self):
        res_short = self.gate.step(self.clean_card, "Adjust", proposed_decision=1, reason="too short")
        self.assertEqual(res_short.next_state, "s_block")
        self.assertEqual(res_short.error_code, "ERR_JUSTIFICATION_TOO_SHORT")

        res_c0 = self.gate.step(self.clean_card, "Confirm", proposed_decision=0)
        self.assertEqual(res_c0.next_state, "s_term")
        self.assertEqual(res_c0.final_decision, "0")

        res_esc = self.gate.step(self.clean_card, "Escalate", reason="Need manager review")
        self.assertEqual(res_esc.next_state, "s_esc")
        self.assertEqual(res_esc.final_decision, "Esc")


class TestMetricsStatsAblationAndParity(unittest.TestCase):
    def test_metrics_n40_and_table2_43_8_rounding(self):
        csv_path = DATA_DIR / "eval_records_40.csv"
        metrics = compute_metrics(csv_path)
        table2 = format_table2(metrics)
        self.assertEqual(round_half_up(metrics["time"]["pure_hr_reduction"], 1), 43.8)
        self.assertIn("-43.8%", table2)
        self.assertNotIn("-43.7%", table2)

    def test_mcnemar_zero_discordant_and_bootstrap_rng_isolation(self):
        mc = paired_mcnemar_exact([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(mc["n_discordant"], 0)
        self.assertEqual(mc["p_value"], 1.0)
        self.assertIn("p_value_formatted", mc)

        # Verify bootstrap_kappa_ci does not mutate global numpy.random state
        np.random.seed(999)
        before = np.random.randint(0, 100000)
        np.random.seed(999)
        _ = np.random.randint(0, 100000)
        expected_next = np.random.randint(0, 100000)

        np.random.seed(999)
        _ = np.random.randint(0, 100000)
        bootstrap_kappa_ci([1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], n_bootstraps=50, seed=42)
        actual_next = np.random.randint(0, 100000)
        self.assertEqual(before, before)
        self.assertEqual(actual_next, expected_next)

    def test_trace_ablation_and_table3_0_088_0_0875(self):
        csv_path = DATA_DIR / "eval_records_40.csv"
        abl = run_trace_driven_ablation(csv_path, n_bootstraps=2000)
        self.assertEqual(abl["M3_EICA_FullVerifier"]["expected_loss_by_lambda_fa"]["10"], 0.0875)
        t3 = format_table3(abl)
        self.assertIn("0.088", t3)
        self.assertIn("0.0875", t3)

    def test_exemplars_and_synthetic_n200_and_ablations(self):
        exemplars = {e["case_id"]: e for e in run_exemplar_verification_replay()}
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_CERT") for v in exemplars["R05"]["verifier_violations"]))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", exemplars["R05"]["verifier_violations"])
        self.assertEqual(exemplars["R37"]["verifier_violations"], [])
        self.assertTrue(any(v.startswith("UNANCHORED_SPAN:QMS_CERT") for v in exemplars["R31"]["verifier_violations"]))
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_LEAD") for v in exemplars["R31"]["verifier_violations"]))

        stress = run_synthetic_verification_benchmark(seed=42)
        self.assertEqual(stress["extraction_verifier_summary"]["f1_pct"], 100.0)
        self.assertEqual(stress["rag_whitelist_pin_compliance_pct"], 100.0)

        comp = run_synthetic_component_ablations(seed=42)
        self.assertEqual(comp["Full_EICA"]["extraction_f1_pct"], 100.0)
        self.assertEqual(comp["Unpinned_RAG"]["legacy_query_leakage_pct"], 100.0)
        self.assertEqual(comp["Exact_String_Match_Only"]["extraction_recall_pct"], 33.3)
        self.assertEqual(comp["Exact_String_Match_Only"]["extraction_f1_pct"], 50.0)
        self.assertEqual(comp["Soft_FSM_Lock_Off"]["conflict_override_bypass_pct"], 100.0)
        self.assertEqual(comp["Unmasked_Schema"]["schema_tacit_block_rate_pct"], 0.0)

        sweep = run_parameter_sensitivity_sweep(seed=42)
        tau_map = {r["tau_span"]: r for r in sweep["tau_span"]}
        self.assertEqual(tau_map[0.85]["boundary_f1_pct"], 100.0)
        self.assertGreater(tau_map[0.95]["boundary_false_alarm_pct"], 0.0)
        self.assertLess(tau_map[0.75]["boundary_recall_pct"], 100.0)

    def test_m4_p2_gap2_concise_reasons_length_16_to_19_and_lmin_sweep_boundary(self):
        import ast
        import inspect

        src = inspect.getsource(run_parameter_sensitivity_sweep)
        tree = ast.parse(src)
        concise_reasons = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == "concise_reasons":
                    if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                        concise_reasons = ast.literal_eval(node.value.left) * ast.literal_eval(node.value.right)
                    else:
                        concise_reasons = ast.literal_eval(node.value)
                    break
        self.assertIsNotNone(concise_reasons)
        self.assertEqual(len(concise_reasons), 15)
        for rsn in concise_reasons:
            self.assertGreaterEqual(len(rsn.strip()), 16, rsn)
            self.assertLessEqual(len(rsn.strip()), 19, rsn)

        base_card = {
            "candidate_id": "LMIN-CHK",
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
        gate_15 = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        gate_20 = FSMDecisionGate(l_min=20, enforce_hard_escalation_lock=True)
        blk_15 = sum(1 for r in concise_reasons if gate_15.step(base_card, "Adjust", proposed_decision=1, reason=r).next_state == "s_block")
        blk_20 = sum(1 for r in concise_reasons if gate_20.step(base_card, "Adjust", proposed_decision=1, reason=r).next_state == "s_block")
        self.assertEqual(blk_15 / len(concise_reasons) * 100.0, 0.0)
        self.assertEqual(blk_20 / len(concise_reasons) * 100.0, 100.0)


class TestPromptAndVLLMExtractor(unittest.TestCase):
    """Unit tests for M38 reference prompt template, prompt assembler, and VLLMExtractor."""

    def test_reference_prompt_file_and_builder_omit_tacit_slots(self):
        from eica_engine import HybridRegulatoryRAG, build_extraction_prompt

        prompt_doc = CODE_DIR / "prompts" / "eica_extraction_reference_prompt.md"
        self.assertTrue(prompt_doc.exists(), "Reference prompt template file missing")
        doc_text = prompt_doc.read_text(encoding="utf-8")
        self.assertIn("Reference Prompt Template", doc_text)
        self.assertIn("withheld under enterprise confidentiality", doc_text)

        rag = HybridRegulatoryRAG()
        clauses = rag.retrieve("ISO 13485 internal auditor sterile validation", top_k=3)
        assembled = build_extraction_prompt(
            "R05",
            "QA/QC",
            ["Bachelor in Biomedical Engineering.", "Attended ISO 13485 internal audit training."],
            clauses,
        )
        self.assertEqual(len(assembled["messages"]), 2)
        user_msg = assembled["messages"][1]["content"]
        self.assertNotIn("QA_TACIT", user_msg)
        self.assertIn("QA_CERT", user_msg)
        self.assertIn(clauses[0][0].doc_id, user_msg)
        self.assertEqual(assembled["json_schema"]["masked_tacit_slots"], ["QA_TACIT"])

    def test_vllm_extractor_enforces_mask_and_strips_scalar_scores(self):
        import json
        from eica_engine import HybridRegulatoryRAG, VLLMExtractor

        captured = {}

        def fake_transport(payload: dict) -> str:
            captured["payload"] = payload
            return json.dumps({
                "candidate_id": "R31",
                "role": "QMS",
                "overall_fit_score": 98.5,
                "entries": [
                    {"req_id": "QMS_CERT", "evidence": "ISO 13485", "anchor_lines": [1], "status": "verified", "score": 99},
                    {"req_id": "QMS_LEAD", "evidence": "participated", "anchor_lines": [1], "status": "verified"},
                    {"req_id": "QMS_EQUIV", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "QMS_TACIT", "evidence": "High ethics", "anchor_lines": [1], "status": "verified"},
                ],
            })

        rag = HybridRegulatoryRAG()
        clauses = rag.retrieve("ISO 13485 QMS internal audit", top_k=3)
        extractor = VLLMExtractor(transport=fake_transport)
        card, flags = extractor.extract("R31", "QMS", ["Participated in ISO 13485 audit."], clauses)

        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["seed"], 42)
        self.assertEqual(captured["payload"]["max_tokens"], 2048)
        self.assertNotIn("overall_fit_score", card)
        self.assertTrue(any("TACIT_EXTRAPOLATION_BLOCKED" in f for f in flags))
        tacit_entry = [e for e in card["entries"] if e["req_id"] == "QMS_TACIT"][0]
        self.assertIsNone(tacit_entry["status"])
        cert_entry = [e for e in card["entries"] if e["req_id"] == "QMS_CERT"][0]
        self.assertNotIn("score", cert_entry)

    def test_vllm_extractor_retries_up_to_two_times_and_fails_cleanly(self):
        import json
        from eica_engine import HybridRegulatoryRAG, VLLMExtractor

        rag = HybridRegulatoryRAG()
        clauses = rag.retrieve("NMPA registration", top_k=3)

        calls = {"n": 0}

        def flaky_transport(_payload: dict) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                return "{invalid json"
            return json.dumps({
                "candidate_id": "R37",
                "role": "RA",
                "entries": [
                    {"req_id": "RA_DEG", "evidence": "BSc Pharmacy", "anchor_lines": [1], "status": "verified"},
                    {"req_id": "RA_REG", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                    {"req_id": "RA_CER", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                ],
            })

        extractor = VLLMExtractor(max_retries=2, transport=flaky_transport)
        card, _flags = extractor.extract("R37", "RA", ["BSc Pharmacy."], clauses)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(card["candidate_id"], "R37")

        always_bad = VLLMExtractor(max_retries=2, transport=lambda _p: "not-json")
        with self.assertRaises(RuntimeError):
            always_bad.extract("R37", "RA", ["BSc Pharmacy."], clauses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
