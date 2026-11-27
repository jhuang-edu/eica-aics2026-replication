# -*- coding: utf-8 -*-
"""
Tier 2 E2E Verification Suite: Boundary & Corner Cases (60 Boundary Tests).

Covers boundary/corner cases across F1..F11 per TEST_INFRA.md and e2e_spec_miner_r1/handoff.md:
  - F1 (5): Empty CSV, single-row CSV, Cohen's kappa edges, safe_div(0), zero discordant McNemar
  - F2 (6): Page count/Letter tolerances, 3..10 keywords, title colon/math, abstract cite/math, mid-sentence Eq.(, Eq.(1..6) alignment
  - F3 (5): Font family diversity, 6.8pt table threshold, figure font legibility, baseline-aware orphan punct + badness 10000/UTF-8, 72pt column balance
  - F4 (5): 43.75%->-43.8% rounding, 0.0875/0.088 rounding, no data/06 docstring, README local paths, no stale 23/34 counts
  - F5 (5): No fabricated commercial/Qwen3 LLM runs, M2/M3 counterfactual disclosure, N_syn=200 separation, Tectonic claim honesty, latency bounds
  - F6 (6): T2-A1..A6 SchemaMasker boundaries
  - F7 (5): T2-B1..B5 HybridRegulatoryRAG boundaries
  - F8 (8): T2-C1..C8 DeterministicVerifier boundaries (incl. local negation, Tier P v_src==0, independent Rule 2+3 + verifier_note, anchor bounds, contiguous span, tau=0.85)
  - F9 (5): T2-D1..D5 FSMDecisionGate boundaries
  - F10 (5): Extreme synthetic CSV in plot_fig2_results, no PUA glyphs, Fig1/Fig2 design font sizes, isolated output files
  - F11 (5): Different seeds, component-off ablations (fuzzy span off, verb tier off, cert proof off, unpinned RAG)
"""

from __future__ import annotations

import csv
import re
import tempfile
import unittest
from pathlib import Path

try:
    from . import e2e_helpers as h
except ImportError:
    import e2e_helpers as h

fitz = h.fitz


class TestTier2F1MetricsAndStatsBoundaries(unittest.TestCase):
    """F1 (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()

    def test_t2_f1_01_compute_metrics_empty_csv_raises(self):
        from metrics_calculator import compute_metrics

        with tempfile.TemporaryDirectory(prefix="eica_t2_") as td:
            p = Path(td) / "empty.csv"
            p.write_text(
                "resume_id,position_type,dept_decision,hr_manual_decision,hr_manual_time_min,"
                "hr_manual_miss_key,llm_hr_decision,llm_hr_time_min,llm_traceable,"
                "llm_miss_key,llm_hallucination,llm_escalated\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                compute_metrics(p)

    def test_t2_f1_02_compute_metrics_single_row_boundary(self):
        from metrics_calculator import compute_metrics

        with tempfile.TemporaryDirectory(prefix="eica_t2_") as td:
            p = Path(td) / "one.csv"
            p.write_text(
                "resume_id,position_type,dept_decision,hr_manual_decision,hr_manual_time_min,"
                "hr_manual_miss_key,llm_hr_decision,llm_hr_time_min,llm_traceable,"
                "llm_miss_key,llm_hallucination,llm_escalated\n"
                "R01,质量工程师,1,1,4.5,0,1,2.5,1,0,0,0\n",
                encoding="utf-8",
            )
            m = compute_metrics(p)
            self.assertEqual(m["N"], 1)
            self.assertEqual(m["time"]["pure_hr_manual_sd"], 0.0)
            self.assertEqual(m["agreement"]["eica"], 100.0)

    def test_t2_f1_03_cohen_kappa_edge_cases(self):
        from metrics_calculator import cohen_kappa

        self.assertEqual(cohen_kappa([], []), 0.0)
        self.assertEqual(cohen_kappa([1, 1, 1], [1, 1, 1]), 1.0)
        self.assertEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_t2_f1_04_safe_div_zero_denominator(self):
        from metrics_calculator import safe_div

        self.assertEqual(safe_div(5.0, 0.0), 0.0)
        self.assertEqual(safe_div(0, 0), 0.0)

    def test_t2_f1_05_mcnemar_exact_zero_discordant_pairs(self):
        from statistical_tests import paired_mcnemar_exact

        res = paired_mcnemar_exact([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(res["b"], 0)
        self.assertEqual(res["c"], 0)
        self.assertEqual(res["p_value"], 1.0)


@h.requires_full_workspace
class TestTier2F2ChecklistBoundaries(unittest.TestCase):
    """F2 (6 boundary cases)."""

    def test_t2_f2_01_page_count_and_letter_tolerance_boundaries(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        self.assertTrue(4 <= tel.page_count <= 20)
        for w, ht in tel.page_sizes:
            self.assertTrue(abs(w - 612.0) <= 1.5 and abs(ht - 792.0) <= 1.5)

    def test_t2_f2_02_keywords_count_boundary_3_to_10(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        m = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", main_clean, re.DOTALL)
        self.assertIsNotNone(m)
        kws = [k.strip() for k in m.group(1).split(",") if k.strip()]
        self.assertTrue(3 <= len(kws) <= 10, f"Keyword count {len(kws)} outside [3, 10]")

    def test_t2_f2_03_title_colon_and_math_forbidden_boundary(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        title = h.extract_tex_brace_arg(main_clean, r"\title{")[0]
        self.assertNotIn(":", title)
        self.assertNotIn("$", title)

    def test_t2_f2_04_abstract_no_cite_math_footnote_boundary(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_clean, re.DOTALL)
        self.assertIsNotNone(m)
        abs_txt = m.group(1)
        self.assertNotIn(r"\cite", abs_txt)
        self.assertNotIn("$", abs_txt)
        self.assertNotIn(r"\footnote", abs_txt)

    def test_t2_f2_05_no_mid_sentence_eq_paren_variants(self):
        body = h.combined_body_tex()
        bad = re.findall(r"(?<!Equation~)(?<!Equation )Eq\.\s*~?(?:\(\\ref|\\eqref)\{[^}]+\}\)?", body)
        self.assertEqual(bad, [], f"Mid-sentence Eq.(ref|eqref) variants found: {bad}")

    def test_t2_f2_06_equation_numbers_1_to_6_vertical_overlap_boundary(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        for n in range(1, 7):
            self.assertIn(f"({n})", tel.full_text, f"Missing equation number ({n}) in main.pdf")
        self.assertEqual(tel.wrapped_eq_numbers, [], f"Wrapped equation numbers: {tel.wrapped_eq_numbers}")


@h.requires_full_workspace
class TestTier2F3TypographyBoundaries(unittest.TestCase):
    """F3 (5 boundary cases)."""

    def test_t2_f3_01_font_family_diversity_and_no_tu_ptm_pcr(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertTrue(tel.has_times_font)
        self.assertTrue(tel.has_mono_font)
        self.assertFalse(tel.has_lmroman_fallback)
        self.assertNotIn("TU/ptm/", build.log_text)
        self.assertNotIn("TU/pcr/", build.log_text)

    def test_t2_f3_02_table_font_size_6_8pt_boundary(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        self.assertEqual(tel.table_small_spans, [], f"Table font < 6.8pt spans: {tel.table_small_spans[:6]}")

    def test_t2_f3_03_figure_font_size_boundary(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        self.assertTrue(tel.figure_spans)
        too_small = h.find_tiny_figure_spans(tel.figure_spans, min_body_pt=4.8, min_sub_pt=None)
        self.assertEqual(too_small, [], f"Rendered non-subscript figure spans < 4.8pt: {too_small[:6]}")
        for fig_name in ("fig1_architecture.pdf", "fig2_results.pdf"):
            fig_bytes = (h.PAPER_FIG_DIR / fig_name).read_bytes()
            native_spans = [(pg, sz, txt) for pg, sz, _, txt in h.collect_all_pdf_spans(fig_bytes)]
            self.assertTrue(native_spans, f"Expected vector text spans in {fig_name}")
            tiny_native = h.find_tiny_figure_spans(native_spans, min_body_pt=7.5, min_sub_pt=5.5)
            self.assertEqual(tiny_native, [], f"Native design font too small in {fig_name}: {tiny_native[:6]}")

    def test_t2_f3_04_baseline_aware_orphan_punct_and_log_limits(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(tel.orphaned_punct, [])
        self.assertEqual(re.findall(r"Underfull \\hbox \(badness 10000\)", build.log_text), [])
        self.assertEqual(re.findall(r"Invalid UTF-8", build.log_text), [])

    def test_t2_f3_05_last_page_column_balance_72pt_boundary(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        self.assertIsNotNone(tel.last_page_col_diff_pt)
        self.assertLessEqual(tel.last_page_col_diff_pt, 72.0)


class TestTier2F4RoundingAndDocsBoundaries(unittest.TestCase):
    """F4 (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()

    def test_t2_f4_01_rounding_43_75_pct_boundary(self):
        from metrics_calculator import format_table2

        m = h.get_cached_metrics()
        s = h.get_cached_stats_20k()
        t2 = format_table2(m, s)
        self.assertIn("-43.8%", t2)
        self.assertNotIn("-43.7%", t2)

    def test_t2_f4_02_rounding_0_0875_precision_tier_boundary(self):
        from ablation_and_benchmark import format_table3

        abl = h.get_cached_ablation_20k()
        self.assertEqual(abl["M3_EICA_FullVerifier"]["expected_loss_by_lambda_fa"]["10"], 0.0875)
        t3 = format_table3(abl)
        self.assertTrue("0.088" in t3 or "0.0875" in t3)
        self.assertNotRegex(t3, r"\b0\.087\b")

    def test_t2_f4_03_no_data06_stale_docstring_in_ablation(self):
        from ablation_and_benchmark import run_exemplar_verification_replay

        doc = run_exemplar_verification_replay.__doc__ or ""
        self.assertNotIn("data/06", doc, "Remove stale data/06 docstring reference")

    @h.requires_full_workspace
    def test_t2_f4_04_no_broken_local_paths_in_readmes(self):
        root_rm = (h.REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("EICA-AI-CS-2026-核查最终总结-muse.pdf", root_rm)
        self.assertNotIn("references.bib", root_rm)

    @h.requires_full_workspace
    def test_t2_f4_05_no_stale_counts_23_or_34_in_docs(self):
        root_rm = (h.REPO_ROOT / "README.md").read_text(encoding="utf-8")
        code_rm = (h.REPO_ROOT / "code" / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("34 篇", root_rm)
        stale_23_re = r"(?i)(?<!\d)23\s*(?:项|checks|metrics|indicators|assertions|core|項目|个|の)"
        self.assertNotRegex(root_rm, stale_23_re)
        self.assertNotRegex(code_rm, stale_23_re)


class TestTier2F5IntegrityAndLatencyBoundaries(unittest.TestCase):
    """F5 (5 boundary cases)."""

    @h.requires_full_workspace
    def test_t2_f5_01_no_unexecuted_commercial_llm_claims(self):
        body = h.combined_body_tex()
        self.assertNotRegex(
            body,
            r"(?:GPT-4o?|Claude(?:-3)?|Gemini|Llama-3|Qwen3|DeepSeek(?:-[A-Za-z0-9.]+)?|Mistral)[^\n.]{0,60}(?:40 dossiers|40 resumes|accuracy|kappa|achieved \d+\.\d+%)",
        )

    @h.requires_full_workspace
    def test_t2_f5_02_counterfactual_m2_m3_explicitly_labeled(self):
        r04 = h.strip_tex_comments(h.read_paper_sources()["04_setup.tex"]).lower()
        r05 = h.strip_tex_comments(h.read_paper_sources()["05_results.tex"]).lower()
        self.assertIn("trace-driven", r04)
        self.assertIn("counterfactual", r05)

    @h.requires_full_workspace
    def test_t2_f5_03_synthetic_n200_explicitly_separated_from_n40(self):
        r04 = h.strip_tex_comments(h.read_paper_sources()["04_setup.tex"]).lower()
        self.assertIn("synthetic", r04)
        self.assertIn("200", r04)

    @h.requires_full_workspace
    def test_t2_f5_04_tectonic_verification_claim_honesty(self):
        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        if "verified cleanly under tectonic" in rl.lower():
            log = h.get_tectonic_build().log_text
            self.assertEqual(re.findall(r"TU/ptm/", log), [])
            self.assertEqual(re.findall(r"Invalid UTF-8", log), [])
            self.assertEqual(re.findall(r"badness 10000", log), [])

    def test_t2_f5_05_latency_sanity_upper_bounds(self):
        lat = h.get_cached_synthetic_42()["latency_ms"]
        self.assertGreater(lat["p50"], 0.0)
        self.assertLess(lat["p50"], 25.0)
        self.assertLess(lat["p95"], 100.0)


class TestTier2F6SchemaMaskingBoundaries(unittest.TestCase):
    """F6: T2-A1..A6 OntologySchemaMasker Boundaries (6 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import CompetencyOntology, OntologySchemaMasker
        self.masker = OntologySchemaMasker
        self.onto = CompetencyOntology

    def test_t2_f6_01_compile_schema_masks_tacit_for_all_roles(self):
        for role, tacit in [("QA/QC", "QA_TACIT"), ("QMS", "QMS_TACIT"), ("RA", "RA_TACIT")]:
            sch = self.masker.compile_generation_schema(role)
            reqs = self.onto.get_requirements(role)
            n_ext = sum(1 for r in reqs if r.tier in ("M", "P"))
            self.assertNotIn(tacit, sch["properties"]["entries"]["items"]["properties"]["req_id"]["enum"])
            self.assertIn(tacit, sch["masked_tacit_slots"])
            self.assertEqual(sch["properties"]["entries"]["minItems"], n_ext)
            self.assertEqual(sch["properties"]["entries"]["maxItems"], n_ext)

    def test_t2_f6_02_strip_forbidden_scalar_score_and_rank_keys(self):
        raw = {"candidate_id": "A2", "role": "QA/QC", "overall_fit_score": 92.5, "rank": 1, "entries": []}
        clean, viols = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        self.assertNotIn("overall_fit_score", clean)
        self.assertNotIn("rank", clean)
        self.assertIn("FORBIDDEN_FIELD_STRIPPED:overall_fit_score", viols)
        self.assertIn("FORBIDDEN_FIELD_STRIPPED:rank", viols)

    def test_t2_f6_03_block_tacit_slot_injection_with_verified_status(self):
        raw = {
            "candidate_id": "A3", "role": "QA/QC",
            "entries": [{"req_id": "QA_TACIT", "evidence": "calm", "anchor_lines": [1, 1], "status": "verified"}],
        }
        clean, viols = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT", viols)
        tac = [e for e in clean["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tac["status"])
        self.assertEqual(tac["evidence"], "")

    def test_t2_f6_04_invalid_enum_status_downgraded_to_conflicting(self):
        raw = {
            "candidate_id": "A4", "role": "QA/QC",
            "entries": [{"req_id": "QA_EXP", "evidence": "3 yrs", "anchor_lines": [1, 1], "status": "partially_verified"}],
        }
        clean, viols = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        exp = [e for e in clean["entries"] if e["req_id"] == "QA_EXP"][0]
        self.assertEqual(exp["status"], "conflicting")
        self.assertTrue(any(v.startswith("INVALID_ENUM_STATUS:QA_EXP:") for v in viols))

    def test_t2_f6_05_missing_mandatory_req_auto_filled_not_provided(self):
        raw = {"candidate_id": "A5", "role": "QA/QC", "entries": []}
        clean, viols = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        cert = [e for e in clean["entries"] if e["req_id"] == "QA_CERT"][0]
        self.assertEqual(cert["status"], "not_provided")
        self.assertIn("MISSING_REQ_FILLED_NOT_PROVIDED:QA_CERT", viols)

    def test_t2_f6_06_unknown_req_id_flagged_and_dropped(self):
        raw = {
            "candidate_id": "A6", "role": "QA/QC",
            "entries": [{"req_id": "FAKE_REQ", "evidence": "x", "anchor_lines": [1, 1], "status": "verified"}],
        }
        clean, viols = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        self.assertIn("UNKNOWN_REQ_ID:FAKE_REQ", viols)
        self.assertFalse(any(e["req_id"] == "FAKE_REQ" for e in clean["entries"]))


class TestTier2F7HybridRAGBoundaries(unittest.TestCase):
    """F7: T2-B1..B5 HybridRegulatoryRAG Boundaries (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import HybridRegulatoryRAG
        self.rag_cls = HybridRegulatoryRAG

    def test_t2_f7_01_pinned_docs_12_and_order_650_excluded(self):
        rag = self.rag_cls()
        self.assertEqual(len(rag.pinned_docs), 12)
        self.assertTrue(all(d.pinned for d in rag.pinned_docs))
        self.assertFalse(any("650" in d.doc_id or "650" in d.title for d in rag.pinned_docs))

    def test_t2_f7_02_exact_order_650_query_still_excludes_deprecated(self):
        rag = self.rag_cls()
        res = rag.retrieve("Order 650 superseded medical device supervision rules 2014", top_k=12)
        self.assertTrue(all(d.pinned and "650" not in d.doc_id for d, _ in res))

    def test_t2_f7_03_empty_and_punctuation_only_queries_safe(self):
        rag = self.rag_cls()
        for q in ("", "---", "   "):
            res = rag.retrieve(q, top_k=3)
            self.assertEqual(len(res), 3)
            self.assertTrue(all(sc == 0.0 for _, sc in res))

    def test_t2_f7_04_extreme_alpha_0_and_1_and_half_combination(self):
        q = "ISO 13485 internal auditor certificate IQ/OQ/PQ"
        r1 = dict((d.doc_id, s) for d, s in self.rag_cls(alpha=1.0).retrieve(q, top_k=12))
        r0 = dict((d.doc_id, s) for d, s in self.rag_cls(alpha=0.0).retrieve(q, top_k=12))
        r5 = dict((d.doc_id, s) for d, s in self.rag_cls(alpha=0.5).retrieve(q, top_k=12))
        for doc_id in r5:
            self.assertGreaterEqual(r1[doc_id], 0.0)
            self.assertLessEqual(r1[doc_id], 1.0 + 1e-9)
            self.assertGreaterEqual(r0[doc_id], 0.0)
            self.assertLessEqual(r0[doc_id], 1.0 + 1e-9)
            self.assertAlmostEqual(r5[doc_id], 0.5 * r1[doc_id] + 0.5 * r0[doc_id], places=9)

    def test_t2_f7_05_bilingual_cjk_alnum_query_retrieval(self):
        rag = self.rag_cls()
        res = rag.retrieve("内审员证书 ISO 13485 灭菌验证", top_k=3)
        self.assertEqual(len(res), 3)
        self.assertTrue(any(d.doc_id.startswith("ISO13485") for d, _ in res))


class TestTier2F8VerifierBoundaries(unittest.TestCase):
    """F8: T2-C1..C8 DeterministicVerifier Boundaries (8 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import DeterministicVerifier, OntologySchemaMasker
        self.v = DeterministicVerifier(span_threshold=0.85)
        self.masker = OntologySchemaMasker

    def test_t2_f8_01_local_negation_does_not_suppress_prior_certified_clause(self):
        src = "Certified ISO 13485 internal auditor, Certificate No. CERT-101; no adverse audit findings."
        self.assertEqual(self.v.classify_verb_tier(src), 3)

    def test_t2_f8_02_true_local_negation_fails_rule2_and_rule3(self):
        src = "Not a certified ISO 13485 auditor; attended training only without formal certificate."
        self.assertLess(self.v.classify_verb_tier(src), 3)
        self.assertFalse(self.v.has_credential_proof(src))

    def test_t2_f8_03_certificate_no_identifier_non_negation_passes(self):
        src = "Certified ISO 13485 internal auditor, Certificate No. CERT-2025-88."
        self.assertEqual(self.v.classify_verb_tier(src), 3)
        self.assertTrue(self.v.has_credential_proof(src))

    def test_t2_f8_04_tier_p_v_src_zero_cannot_bypass_rule2(self):
        resume = "Header\nCleanroom monitoring"
        raw = {
            "candidate_id": "C4", "role": "QA/QC",
            "entries": [
                {"req_id": "QA_VAL", "evidence": "Cleanroom monitoring", "claim_text": "Led cleanroom validation", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        card, _ = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        vcard, viols = self.v.verify_card(resume, card, "QA/QC")
        self.assertTrue(any(x.startswith("VERB_ESCALATION:QA_VAL") for x in viols))
        val_e = [e for e in vcard["entries"] if e["req_id"] == "QA_VAL"][0]
        self.assertEqual(val_e["status"], "conflicting")

    def test_t2_f8_05_independent_rule2_and_rule3_preserve_verifier_note(self):
        resume = "Header\nAttended ISO 13485 internal auditor training (2019)"
        raw = {
            "candidate_id": "C5", "role": "QA/QC",
            "entries": [
                {"req_id": "QA_CERT", "evidence": "Attended ISO 13485 internal auditor training (2019)", "claim_text": "Certified ISO 13485 internal auditor", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        card, _ = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        vcard, viols = self.v.verify_card(resume, card, "QA/QC")
        self.assertTrue(any(x.startswith("VERB_ESCALATION:QA_CERT") for x in viols))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", viols)
        cert_e = [e for e in vcard["entries"] if e["req_id"] == "QA_CERT"][0]
        note = cert_e.get("verifier_note", "")
        self.assertIn("verb", note.lower())
        self.assertIn("credential", note.lower())

    def test_t2_f8_06_reversed_oob_and_wide_anchor_lines_rejected(self):
        lines = [f"Line {i}: some content here" for i in range(1, 11)]
        self.assertFalse(self.v.check_span_alignment(lines, "Line 2: some content here", [3, 1])[0])
        self.assertFalse(self.v.check_span_alignment(lines, "Line 2: some content here", [99, 100])[0])
        self.assertFalse(self.v.check_span_alignment(lines, "Line 2: some content here", [1, 10])[0])

    def test_t2_f8_07_scattered_subsequence_fails_contiguous_span(self):
        lines = ["Alpha beta gamma delta epsilon zebra"]
        aligned, _ = self.v.check_span_alignment(lines, "a b g d e z", [1, 1])
        self.assertFalse(aligned)

    def test_t2_f8_08_span_threshold_tau_0_85_exact_boundary(self):
        # 20-char target vs 18-char match (18/20 = 0.90 >= 0.85 -> pass) vs 15-char (15/20 = 0.75 < 0.85 -> fail)
        lines = ["Executed IQ OQ PQ validation runs"]
        ok, sim_ok = self.v.check_span_alignment(lines, "Executed IQ OQ PQ validation", [1, 1])
        self.assertTrue(ok)
        self.assertGreaterEqual(sim_ok, 0.85)
        bad, sim_bad = self.v.check_span_alignment(lines, "Executed IQ OQ PQ XXXXX YYYYY", [1, 1])
        self.assertFalse(bad)
        self.assertLess(sim_bad, 0.85)


class TestTier2F9FSMBoundaries(unittest.TestCase):
    """F9: T2-D1..D5 FSMDecisionGate Boundaries (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import FSMDecisionGate
        self.gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

    def _clean_card(self):
        return {
            "candidate_id": "D", "role": "QA/QC", "verifier_violations": [],
            "entries": [
                {"req_id": "QA_DEG", "tier": "M", "status": "verified"},
                {"req_id": "QA_EXP", "tier": "M", "status": "verified"},
                {"req_id": "QA_CERT", "tier": "M", "status": "verified"},
                {"req_id": "QA_VAL", "tier": "P", "status": "verified"},
            ],
        }

    def test_t2_f9_01_invalid_proposed_decision_rejected(self):
        c = self._clean_card()
        for bad_dec in (None, 2, -1, True):
            r1 = self.gate.step(c, "Confirm", proposed_decision=bad_dec)
            self.assertEqual(r1.next_state, "s_block")
            self.assertIsNone(r1.final_decision)
            self.assertEqual(r1.error_code, "ERR_INVALID_DECISION")
            r2 = self.gate.step(c, "Adjust", proposed_decision=bad_dec, reason="Long enough justification string")
            self.assertEqual(r2.next_state, "s_block")
            self.assertIsNone(r2.final_decision)

    def test_t2_f9_02_adjust_to_conflicting_tier_p_blocked(self):
        c = self._clean_card()
        r = self.gate.step(
            c, "Adjust", proposed_decision=1,
            reason="Valid override justification exceeding fifteen characters",
            adjusted_statuses={"QA_VAL": "conflicting"},
        )
        self.assertEqual((r.next_state, r.final_decision, r.error_code), ("s_block", None, "ERR_UNRESOLVED_CONFLICT_LOCK"))

    def test_t2_f9_03_justification_length_l_min_15_boundary(self):
        c = self._clean_card()
        r14 = self.gate.step(c, "Adjust", proposed_decision=1, reason="  14_chars_exact  ")
        self.assertEqual((r14.next_state, r14.error_code), ("s_block", "ERR_JUSTIFICATION_TOO_SHORT"))
        r15 = self.gate.step(c, "Adjust", proposed_decision=1, reason="  15_chars_exact!  ")
        self.assertEqual((r15.next_state, r15.final_decision), ("s_term", "1"))

    def test_t2_f9_04_confirm_0_allowed_even_with_conflicts_and_missing_m(self):
        c = self._clean_card()
        c["entries"][2]["status"] = "conflicting"
        c["entries"][1]["status"] = "not_provided"
        c["verifier_violations"] = ["MISSING_CREDENTIAL_PROOF:QA_CERT"]
        r = self.gate.step(c, "Confirm", proposed_decision=0)
        self.assertEqual((r.next_state, r.final_decision), ("s_term", "0"))

    def test_t2_f9_05_unknown_fsm_action_returns_err_invalid_action(self):
        r = self.gate.step(self._clean_card(), "Bypass", proposed_decision=1)
        self.assertEqual((r.next_state, r.final_decision, r.error_code), ("s_block", None, "ERR_INVALID_ACTION"))


class TestTier2F10FigureBoundaries(unittest.TestCase):
    """F10 (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()

    def test_t2_f10_01_fig2_handles_extreme_synthetic_csv_0pct_and_100pct(self):
        from plot_figures import plot_fig2_results

        cols, rows = h.load_csv_rows(h.EVAL_CSV_PATH)
        with tempfile.TemporaryDirectory(prefix="eica_f10_") as td:
            td_p = Path(td)
            mod_csv = td_p / "all_pass.csv"
            with open(mod_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    r2 = dict(r)
                    r2["llm_hr_decision"] = r2["dept_decision"]
                    w.writerow(r2)
            out_pdf = td_p / "fig2.pdf"
            out_png = td_p / "fig2.png"
            plot_fig2_results(str(mod_csv), str(out_pdf), str(out_png))
            self.assertTrue(out_pdf.exists() and out_pdf.stat().st_size > 5000)
            self.assertTrue(out_png.exists() and out_png.stat().st_size > 20000)

    @h.requires_full_workspace
    def test_t2_f10_02_fig2_no_stixnonunicode_pua_glyphs(self):
        tel = h.analyze_pdf_file(h.PAPER_FIG_DIR / "fig2_results.pdf")
        self.assertEqual(tel.pua_chars, [], f"PUA glyphs in fig2_results.pdf: {tel.pua_chars}")
        self.assertFalse(any("stixnonunicode" in f.lower() for f in tel.fonts_base))

    @h.requires_full_workspace
    def test_t2_f10_03_fig1_design_text_readable_no_tiny_fonts(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        fig1_pages, _ = h.locate_figure_pages(tel)
        self.assertTrue(fig1_pages, "Could not locate Figure 1 page dynamically in main.pdf")
        fig1_spans = [sp for sp in tel.figure_spans if sp[0] in fig1_pages]
        self.assertTrue(fig1_spans, f"No figure spans found on Figure 1 pages {sorted(fig1_pages)}")
        tiny_rendered = h.find_tiny_figure_spans(fig1_spans, min_body_pt=4.8, min_sub_pt=None)
        self.assertEqual(tiny_rendered, [], f"Fig. 1 rendered non-subscript font < 4.8 pt in main.pdf: {tiny_rendered[:6]}")
        fig1_bytes = (h.PAPER_FIG_DIR / "fig1_architecture.pdf").read_bytes()
        self.assertEqual(h.analyze_pdf_bytes(fig1_bytes).pua_chars, [])
        native_spans = [(pg, sz, txt) for pg, sz, _, txt in h.collect_all_pdf_spans(fig1_bytes)]
        self.assertTrue(native_spans)
        tiny_native = h.find_tiny_figure_spans(native_spans, min_body_pt=7.5, min_sub_pt=5.5)
        self.assertEqual(tiny_native, [], f"Fig. 1 native design font too small: {tiny_native[:6]}")

    @h.requires_full_workspace
    def test_t2_f10_04_fig2_design_text_readable_no_tiny_footnotes(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        _, fig2_pages = h.locate_figure_pages(tel)
        self.assertTrue(fig2_pages, "Could not locate Figure 2 page dynamically in main.pdf")
        fig2_spans = [sp for sp in tel.figure_spans if sp[0] in fig2_pages]
        self.assertTrue(fig2_spans, f"No figure spans found on Figure 2 pages {sorted(fig2_pages)}")
        tiny_rendered = h.find_tiny_figure_spans(fig2_spans, min_body_pt=4.8, min_sub_pt=None)
        self.assertEqual(tiny_rendered, [], f"Fig. 2 rendered non-subscript font < 4.8 pt in main.pdf: {tiny_rendered[:6]}")
        fig2_bytes = (h.PAPER_FIG_DIR / "fig2_results.pdf").read_bytes()
        self.assertEqual(h.analyze_pdf_bytes(fig2_bytes).pua_chars, [])
        native_spans = [(pg, sz, txt) for pg, sz, _, txt in h.collect_all_pdf_spans(fig2_bytes)]
        self.assertTrue(native_spans)
        tiny_native = h.find_tiny_figure_spans(native_spans, min_body_pt=7.5, min_sub_pt=5.5)
        self.assertEqual(tiny_native, [], f"Fig. 2 native design font too small: {tiny_native[:6]}")

    def test_t2_f10_05_fig_outputs_nonempty_pdf_and_png_in_isolated_dir(self):
        res = h.get_reproduce_run()
        self.assertGreater(len(res.fig1_pdf_bytes), 5000)
        self.assertGreater(len(res.fig2_pdf_bytes), 5000)
        self.assertGreater(len(res.fig1_png_bytes), 50000)
        self.assertGreater(len(res.fig2_png_bytes), 50000)


class TestTier2F11AblationBoundaries(unittest.TestCase):
    """F11 (5 boundary cases)."""

    def setUp(self):
        h.ensure_src_on_path()

    def test_t2_f11_01_synthetic_benchmark_different_seeds_stable_counts(self):
        from ablation_and_benchmark import run_synthetic_verification_benchmark

        s1 = run_synthetic_verification_benchmark(seed=7)
        s2 = run_synthetic_verification_benchmark(seed=99)
        self.assertEqual(s1["N_total"], 200)
        self.assertEqual(s2["N_total"], 200)
        self.assertEqual(s1["strata"]["valid_grounded"]["passed_clean"], 50)
        self.assertEqual(s2["strata"]["valid_grounded"]["passed_clean"], 50)

    def test_t2_f11_02_ablation_disable_fuzzy_span_fails_minor_typos(self):
        from eica_engine import DeterministicVerifier, OntologySchemaMasker

        resume = "Education\nLed 2 production line IQ/OQ/PQ validations"
        raw = {
            "candidate_id": "FZ", "role": "QA/QC",
            "entries": [{"req_id": "QA_VAL", "evidence": "Led 2 production line IQ/OQ/PQ validation", "anchor_lines": [2, 2], "status": "verified"}],
        }
        card, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
        v_on = DeterministicVerifier(use_fuzzy_span=True)
        v_off = DeterministicVerifier(use_fuzzy_span=False)
        self.assertEqual(v_on.verify_card(resume, card, "QA/QC")[1], [])
        self.assertFalse(v_off.check_span_alignment(resume.splitlines(), "Led 2 production line IQ/OQ/PQ validaton", [2, 2])[0])

    def test_t2_f11_03_ablation_disable_verb_tier_misses_r31_escalation(self):
        from eica_engine import DeterministicVerifier, OntologySchemaMasker

        resume = "Experience\nparticipated in internal audit preparations"
        raw = {
            "candidate_id": "VT", "role": "QMS",
            "entries": [{"req_id": "QMS_LEAD", "evidence": "participated in internal audit preparations", "claim_text": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"}],
        }
        card, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QMS")
        _, viol_on = DeterministicVerifier(use_verb_tier=True).verify_card(resume, card, "QMS")
        _, viol_off = DeterministicVerifier(use_verb_tier=False).verify_card(resume, card, "QMS")
        self.assertTrue(any("VERB_ESCALATION" in v for v in viol_on))
        self.assertEqual(viol_off, [])

    def test_t2_f11_04_ablation_disable_cert_proof_misses_r05_training_only(self):
        from eica_engine import DeterministicVerifier, OntologySchemaMasker

        resume = "Experience\nLed ISO 13485 internal auditor training workshop"
        raw = {
            "candidate_id": "CP", "role": "QA/QC",
            "entries": [{"req_id": "QA_CERT", "evidence": "Led ISO 13485 internal auditor training workshop", "anchor_lines": [2, 2], "status": "verified"}],
        }
        card, _ = OntologySchemaMasker.enforce_mask_and_validate(raw, "QA/QC")
        _, viol_on = DeterministicVerifier(use_cert_proof=True).verify_card(resume, card, "QA/QC")
        _, viol_off = DeterministicVerifier(use_cert_proof=False).verify_card(resume, card, "QA/QC")
        self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", viol_on)
        self.assertNotIn("MISSING_CREDENTIAL_PROOF:QA_CERT", viol_off)

    def test_t2_f11_05_ablation_unpinned_rag_retrieves_deprecated_order_650(self):
        from eica_engine import HybridRegulatoryRAG

        q = "Order 650 superseded medical device rules 2014"
        res_unpinned = HybridRegulatoryRAG(pinned_only=False).retrieve(q, top_k=3)
        self.assertTrue(any(d.doc_id == "ORD650-DEP" for d, _ in res_unpinned))


if __name__ == "__main__":
    unittest.main(verbosity=2)
