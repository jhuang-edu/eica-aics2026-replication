# -*- coding: utf-8 -*-
"""
Tier 1 E2E Verification Suite: Requirement & IEEE/Checklist Compliance (64 Predicates).

Covers F1..F11 per TEST_INFRA.md and e2e_spec_miner_r1/handoff.md:
  - F1 (6): Reproduction & eval_records_40.csv invariants
  - F2 (10): Checking list.pdf & IEEEtran_HOWTO.pdf compliance
  - F3 (7): Tectonic compilation & PDF font/layout telemetry
  - F4 (6): Multi-artifact numeric, prose & repo hygiene parity
  - F5 (5): Option 1 academic integrity & Reviewer C1-C3 rubric
  - F6 (5): OntologySchemaMasker contracts
  - F7 (5): HybridRegulatoryRAG contracts
  - F8 (5): DeterministicVerifier contracts
  - F9 (5): FSMDecisionGate contracts
  - F10 (5): Dynamic vector & raster figure generation
  - F11 (5): Synthetic stress-test & component-off ablations
"""

from __future__ import annotations

import inspect
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fitz

try:
    from . import e2e_helpers as h
except ImportError:
    import e2e_helpers as h


class TestTier1F1ReproductionAndCsv(unittest.TestCase):
    """F1: reproduce_all.py & eval_records_40.csv Invariants (6 predicates)."""

    def test_t1_f1_01_eval_csv_immutability_and_schema(self):
        self.assertEqual(
            h.compute_sha256(h.EVAL_CSV_PATH),
            h.EXPECTED_EVAL_CSV_SHA256,
            "eval_records_40.csv SHA-256 drifted from immutable baseline",
        )
        cols, rows = h.load_csv_rows(h.EVAL_CSV_PATH)
        expected_cols = [
            "resume_id", "position_type", "dept_decision", "hr_manual_decision",
            "hr_manual_time_min", "hr_manual_miss_key", "llm_hr_decision",
            "llm_hr_time_min", "llm_traceable", "llm_miss_key",
            "llm_hallucination", "llm_escalated",
        ]
        self.assertEqual(cols, expected_cols)
        self.assertEqual(len(rows), 40)
        self.assertEqual([r["resume_id"] for r in rows], [f"R{i:02d}" for i in range(1, 41)])
        self.assertEqual(sum(int(r["dept_decision"]) for r in rows), 17)
        self.assertEqual(sum(int(r["hr_manual_decision"]) for r in rows), 20)
        self.assertEqual([r["resume_id"] for r in rows if r["llm_escalated"] == "1"], ["R02", "R05", "R18", "R30", "R37"])
        self.assertEqual([r["resume_id"] for r in rows if r["llm_hallucination"] == "1"], ["R31"])
        self.assertEqual([r["resume_id"] for r in rows if r["llm_traceable"] == "0"], ["R18", "R31"])

    def test_t1_f1_02_reproduce_all_exit_and_28_ledger(self):
        res = h.get_reproduce_run()
        self.assertEqual(res.returncode, 0, f"reproduce_all.py failed:\n{res.stderr}\n{res.stdout}")
        ledger = h.parse_reproduce_ledger(res.stdout)
        self.assertGreaterEqual(len(ledger), 28, f"Expected >= 28 ledger rows, got {len(ledger)}")
        mismatches = [row for row in ledger if row[3] != "[MATCH 100%]"]
        self.assertEqual(mismatches, [], f"Audit ledger mismatches detected: {mismatches}")

    def test_t1_f1_03_table1_and_table2_empirical_metrics(self):
        h.ensure_src_on_path()
        from metrics_calculator import round_half_up

        m = h.get_cached_metrics()
        s = h.get_cached_stats_20k()

        self.assertAlmostEqual(m["agreement"]["manual"], 62.5, places=1)
        self.assertAlmostEqual(m["agreement"]["eica"], 87.5, places=1)
        self.assertAlmostEqual(round(m["kappa"]["manual"], 3), 0.250, places=3)
        self.assertAlmostEqual(round(m["kappa"]["eica"], 3), 0.746, places=3)
        self.assertAlmostEqual(round(m["kappa"]["delta"], 3), 0.496, places=3)
        self.assertAlmostEqual(round(s["delta_kappa_ci"][0], 3), 0.196, places=3)
        self.assertAlmostEqual(round(s["delta_kappa_ci"][1], 3), 0.807, places=3)
        self.assertAlmostEqual(round(s["mcnemar_p"], 5), 0.00635, places=5)
        self.assertEqual((s["discordant_b"], s["discordant_c"]), (11, 1))
        self.assertEqual((m["confusion"]["manual"]["FP"], m["confusion"]["manual"]["FN"]), (9, 6))
        self.assertEqual((m["confusion"]["eica"]["FP"], m["confusion"]["eica"]["FN"]), (3, 2))
        self.assertAlmostEqual(m["omissions"]["manual"], 10.0, places=1)
        self.assertAlmostEqual(m["omissions"]["eica"], 5.0, places=1)
        self.assertAlmostEqual(round(m["time"]["pure_hr_manual_mean"], 2), 4.80, places=2)
        self.assertAlmostEqual(round(m["time"]["pure_hr_eica_mean"], 2), 2.70, places=2)
        self.assertAlmostEqual(round_half_up(m["time"]["pure_hr_reduction"], 1), 43.8, places=1)
        self.assertAlmostEqual(round(m["time"]["total_org_eica"], 2), 3.35, places=2)
        self.assertAlmostEqual(round(m["time"]["total_org_reduction"], 1), 30.2, places=1)

    def test_t1_f1_04_table3_ablation_and_loss_invariants(self):
        abl = h.get_cached_ablation_20k()
        m2 = abl["M2_EICA_HardFSM"]
        m3 = abl["M3_EICA_FullVerifier"]

        self.assertEqual(m2["agreement_pct"], 92.5)
        self.assertEqual(m2["kappa"], 0.845)
        self.assertEqual(m2["mcnemar_p_vs_m0"], 0.00183)
        self.assertEqual(m2["discordant_vs_m0"], "13 vs 1")
        self.assertEqual(m2["f1_pct"], 90.9)
        self.assertEqual((m2["FP"], m2["FN"]), (1, 2))
        self.assertEqual(m2["total_org_time_min"], 3.35)
        self.assertEqual(m2["expected_loss_by_lambda_fa"]["10"], 0.3312)

        self.assertEqual(m3["agreement_pct"], 95.0)
        self.assertEqual(m3["kappa"], 0.896)
        self.assertEqual(m3["mcnemar_p_vs_m0"], 0.00098)
        self.assertEqual(m3["discordant_vs_m0"], "14 vs 1")
        self.assertEqual(m3["f1_pct"], 93.8)
        self.assertEqual((m3["FP"], m3["FN"]), (0, 2))
        self.assertEqual(m3["specificity_pct"], 100.0)
        self.assertEqual(m3["ppv_pct"], 100.0)
        self.assertEqual(m3["escalation_rate_pct"], 15.0)
        self.assertEqual(m3["total_org_time_min"], 3.51)
        self.assertEqual(m3["expected_loss_by_lambda_fa"]["10"], 0.0875)

    def test_t1_f1_05_exemplar_replay_invariants(self):
        h.ensure_src_on_path()
        from ablation_and_benchmark import run_exemplar_verification_replay

        out = {x["case_id"]: x for x in run_exemplar_verification_replay()}
        self.assertEqual(set(out.keys()), {"R05", "R37", "R31"})

        self.assertEqual(out["R05"]["naive_confirm_state"], "s_block")
        self.assertEqual(out["R05"]["naive_confirm_error"], "ERR_UNRESOLVED_CONFLICT_LOCK")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:") for v in out["R05"]["verifier_violations"]))
        self.assertTrue(any(v.startswith("MISSING_CREDENTIAL_PROOF:") for v in out["R05"]["verifier_violations"]))

        self.assertEqual(out["R37"]["naive_confirm_state"], "s_block")
        self.assertEqual(out["R37"]["naive_confirm_error"], "ERR_MANDATORY_GUARD_FAILED")

        self.assertEqual(out["R31"]["naive_confirm_state"], "s_block")
        self.assertEqual(out["R31"]["naive_confirm_error"], "ERR_UNRESOLVED_CONFLICT_LOCK")
        self.assertTrue(any(v.startswith("UNANCHORED_SPAN:") for v in out["R31"]["verifier_violations"]))
        self.assertTrue(any(v.startswith("VERB_ESCALATION:") for v in out["R31"]["verifier_violations"]))

    def test_t1_f1_06_synthetic_stress_test_200_and_unit_suite(self):
        stress = h.get_cached_synthetic_42()
        self.assertEqual(stress["N_total"], 200)
        self.assertEqual(stress["strata"]["valid_grounded"]["passed_clean"], 50)
        self.assertEqual(stress["strata"]["unanchored_span"]["caught"], 30)
        self.assertEqual(stress["strata"]["verb_escalation"]["caught"], 30)
        self.assertEqual(stress["strata"]["missing_cert_proof"]["caught"], 30)
        self.assertEqual(stress["extraction_verifier_summary"]["recall_pct"], 100.0)
        self.assertEqual(stress["extraction_verifier_summary"]["precision_pct"], 100.0)
        self.assertEqual(stress["extraction_verifier_summary"]["f1_pct"], 100.0)
        self.assertEqual(stress["schema_tacit_block_rate_pct"], 100.0)
        self.assertEqual(stress["fsm_invariant_block_rate_pct"], 100.0)
        self.assertEqual(stress["rag_whitelist_pin_compliance_pct"], 100.0)

        unit_tests = sorted(
            p for p in (h.CODE_DIR / "tests").glob("test_*.py") if p.is_file()
        )
        self.assertTrue(unit_tests, "Expected unit tests under code/tests/test_*.py (M1 deliverable)")
        env = h.build_subprocess_env()
        env["PYTHONPATH"] = str(h.SRC_DIR) + ";" + str(h.CODE_DIR / "tests")
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v"] + [p.stem for p in unit_tests],
            cwd=str(h.CODE_DIR / "tests"),
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(proc.returncode, 0, f"Unit test suite failed:\n{proc.stdout}\n{proc.stderr}")


class TestTier1F2IEEEAndChecklistCompliance(unittest.TestCase):
    """F2: Checking list.pdf & IEEEtran_HOWTO.pdf Compliance (10 predicates)."""

    def test_t1_f2_01_page_size_columns_and_length(self):
        build = h.get_tectonic_build()
        self.assertEqual(build.returncode, 0, f"Tectonic failed:\n{build.stderr}")
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertGreaterEqual(tel.page_count, 4)
        self.assertLessEqual(tel.page_count, 20)
        for i, (w, ht) in enumerate(tel.page_sizes, 1):
            self.assertAlmostEqual(w, 612.0, delta=1.5, msg=f"Page {i} width {w} != US Letter 612 pt")
            self.assertAlmostEqual(ht, 792.0, delta=1.5, msg=f"Page {i} height {ht} != US Letter 792 pt")
        main_tex = h.read_paper_sources()["main.tex"]
        self.assertIn(r"\documentclass[conference]{IEEEtran}", main_tex)
        self.assertFalse(tel.has_page_numbers_header_footer, "Conference mode must omit running page numbers")

    def test_t1_f2_02_title_abstract_keywords_hygiene(self):
        main_tex = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        titles = h.extract_tex_brace_arg(main_tex, r"\title{")
        self.assertEqual(len(titles), 1)
        title = titles[0]
        self.assertNotIn(":", title, "IEEE Xplore forbids subtitles/colons in title")
        self.assertNotIn("$", title, "Title must not contain math")
        self.assertNotIn(r"\footnote", title)
        self.assertNotIn(r"\thanks", title)

        m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.DOTALL)
        self.assertIsNotNone(m_abs)
        abstract = m_abs.group(1)
        self.assertNotIn("$", abstract, "Abstract must not contain inline math $...$")
        self.assertNotIn(r"\cite", abstract, "Abstract must not contain citations")
        self.assertNotIn(r"\footnote", abstract)

        m_kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", main_tex, re.DOTALL)
        self.assertIsNotNone(m_kw)
        kws = [k.strip() for k in m_kw.group(1).split(",") if k.strip()]
        self.assertGreaterEqual(len(kws), 3)
        self.assertLessEqual(len(kws), 10)
        self.assertNotIn("$", m_kw.group(1))

    def test_t1_f2_03_authors_affiliations_and_correspondence(self):
        main_tex = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        authors = h.extract_tex_brace_arg(main_tex, r"\author{")
        self.assertEqual(len(authors), 1)
        blk = authors[0]
        self.assertIn("Junjun Huang", blk)
        self.assertIn("Ning Chen", blk)
        self.assertIn(r"\textsuperscript{*}", blk)
        self.assertIn(r"\thanks{", blk)
        for req_tok in ("Indonesia", "Malaysia", "China", "90245", "50603", "100039", "200433", "junjun@ieee.org", "chelcc510@gmail.com"):
            self.assertIn(req_tok, blk, f"Missing author metadata token: {req_tok}")

    def test_t1_f2_04_figure_and_table_captions_and_citations(self):
        sources = h.read_paper_sources()
        main_clean = h.strip_tex_comments(sources["main.tex"])
        self.assertIn(r"\renewcommand{\figurename}{Figure}", main_clean)
        self.assertIn(r"\renewcommand{\tablename}{Table}", main_clean)
        self.assertIn(r"\renewcommand{\thetable}{\arabic{table}}", main_clean)

        body = h.combined_body_tex()
        caps = h.extract_caption_args(body)
        self.assertGreaterEqual(len(caps), 6)
        for cap in caps:
            self.assertTrue(cap.endswith("."), f"Caption missing trailing period: {cap!r}")
            self.assertFalse(re.match(r"^(?:Fig\.|Tab\.)", cap), f"Abbreviated caption disallowed: {cap!r}")

        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertIn("Figure 1", tel.full_text)
        self.assertIn("Figure 2", tel.full_text)
        fig_labels = re.findall(r"\\label\{(fig:[^}]+)\}", body)
        tab_labels = re.findall(r"\\label\{(tab:[^}]+)\}", body)
        self.assertGreaterEqual(len(fig_labels), 2)
        self.assertGreaterEqual(len(tab_labels), 4)
        for fl in fig_labels:
            self.assertIn(f"\\ref{{{fl}}}", body, f"Figure label {fl} never cited in body text")
        for tl in tab_labels:
            self.assertIn(f"\\ref{{{tl}}}", body, f"Table label {tl} never cited in body text")
        for tnum in range(1, 5):
            self.assertIn(f"Table {tnum}", tel.full_text)

    def test_t1_f2_05_editable_tables_and_legibility(self):
        body = h.combined_body_tex()
        self.assertIn(r"\begin{tabular}", body)
        self.assertGreaterEqual(body.count(r"\begin{tabular}"), 4)
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(
            tel.table_small_spans,
            [],
            f"Rendered table body contains micro-text (< 6.8 pt, likely \\resizebox): {tel.table_small_spans[:8]}",
        )

    def test_t1_f2_06_equations_alignment_and_eqref_convention(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(
            tel.wrapped_eq_numbers,
            [],
            f"Equation numbers wrapped below equations: {tel.wrapped_eq_numbers}",
        )
        body = h.combined_body_tex()
        bad_eq = re.findall(r"(?<!Equation~)(?<!Equation )Eq\.\s*~?(?:\(\\ref|\\eqref)\{", body)
        self.assertEqual(bad_eq, [], f"IEEEtran requires \\eqref{{...}} mid-sentence instead of Eq.~(\\ref|\\eqref): {bad_eq}")

    def test_t1_f2_07_ieeetran_algorithm_float_rule(self):
        sources = h.read_paper_sources()
        for name, txt in sources.items():
            clean = h.strip_tex_comments(txt)
            self.assertNotIn(r"\usepackage{algorithm}", clean, f"{name} must not load algorithm.sty")
            self.assertNotIn(r"\begin{algorithm}", clean, f"{name} must wrap algorithmic inside figure float")
        m3 = h.strip_tex_comments(sources["03_model.tex"])
        self.assertIn(r"\begin{algorithmic}", m3)

    def test_t1_f2_08_references_checklist_and_monotonic_cites(self):
        sources = h.read_paper_sources()
        main_clean = h.strip_tex_comments(sources["main.tex"])
        bibs = h.extract_bibitems(main_clean)
        self.assertGreaterEqual(len(bibs), 5)
        self.assertEqual(len(bibs), 32)

        for key, body in bibs:
            self.assertRegex(body, r"\b(?:19|20)\d{2}\b", f"Bibitem {key} missing publication year")
            self.assertTrue(
                ("doi:" in body.lower() or "http://" in body.lower() or "https://" in body.lower()),
                f"Bibitem {key} missing DOI or URL",
            )

        ordered_cites = h.extract_ordered_citations(sources)
        bib_keys = [k for k, _ in bibs]
        self.assertEqual(set(ordered_cites), set(bib_keys), "Mismatch between cited keys and bibliography")
        self.assertEqual(ordered_cites, bib_keys, "Citations must appear in strictly monotonic [1]..[N] order")
        self.assertRegex(main_clean, r"\\def\\@cite#1#2\{[^\n]*\\textsuperscript\{\[", "Missing superscript \\@cite definition")
        self.assertEqual(h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes).orphaned_punct, [])

    def test_t1_f2_09_zero_cjk_and_zero_unresolved_refs(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(tel.cjk_chars, [], f"CJK characters found in main.pdf: {tel.cjk_chars}")
        self.assertEqual(tel.unresolved_refs, [], f"Unresolved references (??) in main.pdf: {tel.unresolved_refs}")
        for fig_name in ("fig1_architecture.pdf", "fig2_results.pdf"):
            fig_tel = h.analyze_pdf_file(h.PAPER_FIG_DIR / fig_name)
            self.assertEqual(fig_tel.cjk_chars, [], f"CJK characters found in {fig_name}: {fig_tel.cjk_chars}")

    def test_t1_f2_10_figures_embedded_highres_and_no_lena(self):
        body = h.combined_body_tex()
        self.assertIn("figures/fig1_architecture.pdf", body)
        self.assertIn("figures/fig2_results.pdf", body)
        self.assertNotIn("lena", body.lower())
        for fig_name in ("fig1_architecture.png", "fig2_results.png"):
            p = h.PAPER_FIG_DIR / fig_name
            self.assertTrue(p.exists(), f"Missing raster figure: {p}")
            w_px, _ = h.png_pixel_size(p.read_bytes())
            self.assertGreaterEqual(w_px, 2000, f"{fig_name} width too low for 300 DPI: {w_px} px")


class TestTier1F3TectonicTypography(unittest.TestCase):
    """F3: Tectonic Compilation & PDF Typography Telemetry (7 predicates)."""

    def test_t1_f3_01_clean_tectonic_build(self):
        build = h.get_tectonic_build()
        self.assertEqual(build.returncode, 0, f"Tectonic exit != 0:\n{build.stderr}")
        self.assertNotIn("Undefined control sequence", build.log_text)
        self.assertNotIn("There were undefined references", build.log_text)
        self.assertNotIn("There were undefined citations", build.log_text)

    def test_t1_f3_02_prescribed_ieee_fonts_no_lmroman_fallback(self):
        build = h.get_tectonic_build()
        log = build.log_text
        ptm_warns = re.findall(r".*TU/ptm/.*undefined.*", log)
        pcr_warns = re.findall(r".*TU/pcr/.*undefined.*", log)
        self.assertEqual(ptm_warns, [], f"Found {len(ptm_warns)} TU/ptm font warnings in main.log")
        self.assertEqual(pcr_warns, [], f"Found {len(pcr_warns)} TU/pcr font warnings in main.log")

        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertTrue(tel.has_times_font, f"Missing Times-family serif font; found {sorted(tel.fonts_base)}")
        self.assertTrue(tel.has_mono_font, f"Missing monospace font; found {sorted(tel.fonts_base)}")
        self.assertFalse(tel.has_lmroman_fallback, "LMRoman fallback font detected in main.pdf")

    def test_t1_f3_03_no_forbidden_ieeetran_spacing_overrides(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        self.assertNotIn(r"\linespread{0.96}", main_clean)
        self.assertNotIn(r"\fontsize{7.5pt}", main_clean)
        self.assertNotIn(r"\itemsep}{-0.9pt}", main_clean)

    def test_t1_f3_04_clean_log_encoding_and_badness_10000(self):
        log = h.get_tectonic_build().log_text
        utf8_warns = re.findall(r".*Invalid UTF-8.*", log)
        badness_10k = re.findall(r".*Underfull \\hbox \(badness 10000\).*", log)
        self.assertEqual(utf8_warns, [], f"Invalid UTF-8 warnings in main.log: {utf8_warns}")
        self.assertEqual(badness_10k, [], f"Underfull \\hbox (badness 10000) in main.log: {badness_10k}")

    def test_t1_f3_05_zero_orphaned_citation_punctuation(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(tel.orphaned_punct, [], f"Orphaned citation punctuation detected: {tel.orphaned_punct}")

    def test_t1_f3_06_zero_pua_and_legible_figure_text(self):
        build = h.get_tectonic_build()
        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertEqual(tel.pua_chars, [], f"PUA glyphs found in main.pdf: {tel.pua_chars[:10]}")
        for fig_name in ("fig1_architecture.pdf", "fig2_results.pdf"):
            fig_bytes = (h.PAPER_FIG_DIR / fig_name).read_bytes()
            ftel = h.analyze_pdf_bytes(fig_bytes)
            self.assertEqual(ftel.pua_chars, [], f"PUA glyphs found in {fig_name}: {ftel.pua_chars}")
            native_spans = [(pg, sz, txt) for pg, sz, _, txt in h.collect_all_pdf_spans(fig_bytes)]
            self.assertTrue(native_spans, f"Expected vector text spans in {fig_name}")
            tiny_native = h.find_tiny_figure_spans(native_spans, min_body_pt=7.5, min_sub_pt=5.5)
            self.assertEqual(tiny_native, [], f"Native design font too small in {fig_name}: {tiny_native[:8]}")
        self.assertTrue(tel.figure_spans, "Expected embedded vector text spans in figures")
        tiny_fig_spans = h.find_tiny_figure_spans(tel.figure_spans, min_body_pt=4.8, min_sub_pt=None)
        self.assertEqual(tiny_fig_spans, [], f"Microscopic non-subscript figure text (< 4.8 pt) in main.pdf: {tiny_fig_spans[:8]}")

    def test_t1_f3_07_last_page_column_equalization(self):
        tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)
        self.assertIsNotNone(tel.last_page_col_diff_pt, "Last page does not have two populated columns")
        self.assertLessEqual(tel.last_page_col_diff_pt, 72.0, f"Last-page column diff {tel.last_page_col_diff_pt:.1f} pt > 72 pt")


class TestTier1F4CrossArtifactParity(unittest.TestCase):
    """F4: Multi-Artifact Numeric, Prose & Hygiene Parity (6 predicates)."""

    def test_t1_f4_01_paper_tables_and_prose_vs_reproduce(self):
        r05_res = h.read_paper_sources()["05_results.tex"]
        for tok in ("62.5\\%", "87.5\\%", "92.5\\%", "95.0\\%", "0.250", "0.746", "0.845", "0.896",
                    "0.00635", "0.00183", "0.00098", "-43.8\\%", "-30.2\\%", "-27.0\\%"):
            self.assertIn(tok, r05_res, f"Expected verified metric {tok!r} in 05_results.tex")
        self.assertTrue("0.088" in r05_res or "0.0875" in r05_res, "Expected 0.088 or 0.0875 in 05_results.tex")
        for pat in (r"9\s*/\s*6", r"3\s*/\s*2", r"1\s*/\s*2", r"0\s*/\s*2"):
            self.assertRegex(r05_res, pat, f"Expected FA/FR pattern {pat} in 05_results.tex")
        self.assertNotIn("-43.7", r05_res)
        self.assertNotRegex(r05_res, r"\b0\.087\b")

    def test_t1_f4_02_response_letter_parity(self):
        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        self.assertNotIn("reviewer's core", rl, "Grammar typo: use reviewers' core")
        self.assertRegex(rl, r"(?i)\b32\b[^\n]{0,35}references", "Missing 32 references count in RESPONSE_LETTER_AICS2026.md")
        self.assertRegex(rl, r"(?i)\b15\b[^\n]{0,45}(?:ieee|cs|peer-reviewed|references)", "Missing 15 new CS refs count in RESPONSE_LETTER_AICS2026.md")
        ledger = h.parse_reproduce_ledger(h.get_reproduce_run().stdout)
        self.assertRegex(
            rl,
            rf"(?i)\b(?:28|42|{len(ledger)})\b[^\n]{{0,40}}(?:claims|checks|assertions|metrics|audit)",
            "Missing verified check/claim count (28 core or 42 ledger) in RESPONSE_LETTER_AICS2026.md",
        )
        for tok in ("95.0%", "0.896"):
            self.assertIn(tok, rl, f"Missing synced token {tok!r} in RESPONSE_LETTER_AICS2026.md")
        self.assertTrue("0.088" in rl or "0.0875" in rl, "Expected 0.088 or 0.0875 in RESPONSE_LETTER_AICS2026.md")
        self.assertNotRegex(rl, r"\b0\.087\b")
        self.assertNotIn("-43.7", rl)
        self.assertRegex(rl.lower(), r"complexity|o\(", "Response letter should cite formal complexity analysis")
        self.assertRegex(rl.lower(), r"component-off|ablation|sensitivity", "Response letter should cite synthetic ablations")
        if "verified cleanly under tectonic" in rl.lower():
            log = h.get_tectonic_build().log_text
            self.assertNotIn("TU/ptm/", log, "RESPONSE_LETTER claims clean Tectonic build while TU/ptm warnings exist")
            self.assertNotIn("TU/pcr/", log, "RESPONSE_LETTER claims clean Tectonic build while TU/pcr warnings exist")
            self.assertNotIn("Invalid UTF-8", log, "RESPONSE_LETTER claims clean Tectonic build while UTF-8 warnings exist")
            self.assertNotIn("badness 10000", log, "RESPONSE_LETTER claims clean Tectonic build while badness 10000 warnings exist")

    def test_t1_f4_03_readme_and_code_readme_parity(self):
        root_rm = (h.REPO_ROOT / "README.md").read_text(encoding="utf-8")
        code_rm = (h.REPO_ROOT / "code" / "README.md").read_text(encoding="utf-8")
        for stale in ("EICA-AI-CS-2026-核查最终总结-muse.pdf", "references.bib", "34 篇", "23项", "23 checks", "23 metrics",
                      "23 個", "23項目", "seaborn", "pandas", "Human-AI Division of Labor", "Anonymous Authors"):
            self.assertNotIn(stale, root_rm, f"Stale token {stale!r} in README.md")
            self.assertNotIn(stale, code_rm, f"Stale token {stale!r} in code/README.md")
        stale_23_re = r"(?i)(?<!\d)23\s*(?:项|checks|metrics|indicators|assertions|core|項目|个|の)"
        self.assertNotRegex(root_rm, stale_23_re, "Stale 23 check count in README.md")
        self.assertNotRegex(code_rm, stale_23_re, "Stale 23 check count in code/README.md")
        ledger = h.parse_reproduce_ledger(h.get_reproduce_run().stdout)
        self.assertRegex(
            code_rm,
            rf"(?i)(?<!seed )\b(?:28|42|{len(ledger)})\b[^\n]{{0,30}}(?:checks|assertions|claims|metrics|audit|quantitative|项|項目|の)",
            "Missing updated check count (28 or 42) in code/README.md",
        )
        for req in ("eica_engine.py", "ablation_and_benchmark.py", "vllm_config.yaml"):
            self.assertIn(req, code_rm, f"Missing {req!r} in code/README.md")

        # GAP-1 & unit-test count sync checks
        vllm_cfg = (h.CODE_DIR / "vllm_config.yaml").read_text(encoding="utf-8")
        rl_txt = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        paper_txts = "\n".join(h.read_paper_sources().values())
        self.assertIn("Qwen3-32B", vllm_cfg)
        for label, txt in [
            ("code/vllm_config.yaml", vllm_cfg),
            ("README.md", root_rm),
            ("code/README.md", code_rm),
            ("RESPONSE_LETTER_AICS2026.md", rl_txt),
            ("paper/", paper_txts),
        ]:
            self.assertNotIn("Qwen3.8-27B", txt, f"Stale Qwen3.8-27B reference found in {label}")
        stale_unit_re = r"(?i)(?<!\d)(?:47|52|53|54|55|56|58)(?:`|\*|\s)*(?:项|件|std|unit|/)"
        for label, txt in [("README.md", root_rm), ("code/README.md", code_rm), ("RESPONSE_LETTER_AICS2026.md", rl_txt)]:
            self.assertNotRegex(txt, stale_unit_re, f"Stale unit-test count found in {label}")
            self.assertIn("62", txt, f"Expected updated 62 unit-test count in {label}")

    def test_t1_f4_04_dpin_and_algorithm1_parity(self):
        h.ensure_src_on_path()
        from eica_engine import HybridRegulatoryRAG

        rag = HybridRegulatoryRAG()
        self.assertEqual(len(rag.pinned_docs), 12)
        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        self.assertRegex(m3, r"\b12\s+active\b")
        fig1_txt = h.analyze_pdf_file(h.PAPER_FIG_DIR / "fig1_architecture.pdf").full_text
        self.assertIn("12 docs", fig1_txt)
        for tag in ("UNANCHORED_SPAN", "VERB_ESCALATION", "MISSING_CREDENTIAL_PROOF"):
            self.assertIn(tag.replace("_", r"\_"), m3, f"Alg. 1 missing standardized tag {tag}")
        self.assertNotIn("MISSING\\_CERT", m3, "Alg. 1 still uses deprecated MISSING_CERT tag")
        self.assertNotIn(r"\ELSIF{$v_{\mathrm{src}}", m3, "Alg. 1 must execute Rule 2 and Rule 3 independently")
        self.assertRegex(m3, r"s_\{\\mathrm\{term\}\}\(D,\s*\\boldsymbol\{\\sigma\}\^\*\),\s*&\s*\\text\{if\s*\}\s*a\s*=\s*\\textit\{Confirm\}\(D\)")

    def test_t1_f4_05_repo_hygiene_no_locks_or_stale_src_figs(self):
        lock_files = [str(p) for p in h.REPO_ROOT.rglob("~$*")]
        self.assertEqual(lock_files, [], f"Office lock files present: {lock_files}")
        stale_src_figs = [str(p) for p in h.SRC_DIR.glob("fig*") if p.is_file()]
        self.assertEqual(stale_src_figs, [], f"Stale figures in code/src/: {stale_src_figs}")

    def test_t1_f4_06_output_rounding_helpers_parity(self):
        h.ensure_src_on_path()
        from ablation_and_benchmark import format_table3, run_trace_driven_ablation
        from metrics_calculator import compute_metrics, format_table2
        from statistical_tests import run_statistical_suite

        m = compute_metrics(h.EVAL_CSV_PATH)
        s = run_statistical_suite(h.EVAL_CSV_PATH, n_bootstraps=2000)
        t2 = format_table2(m, s)
        self.assertIn("-43.8%", t2)
        self.assertNotIn("-43.7%", t2)

        abl = run_trace_driven_ablation(h.EVAL_CSV_PATH, n_bootstraps=2000)
        t3 = format_table3(abl)
        self.assertTrue("0.088" in t3 or "0.0875" in t3, f"Expected 0.088/0.0875 in format_table3:\n{t3}")
        self.assertNotRegex(t3, r"\b0\.087\b")


class TestTier1F5Option1IntegrityAndRubric(unittest.TestCase):
    """F5: Option 1 Academic Integrity & Reviewer C1-C3 Rubric (5 predicates)."""

    def test_t1_f5_01_option1_zero_fabricated_llm_runs(self):
        body = h.combined_body_tex()
        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        for txt in (body, rl):
            self.assertNotRegex(
                txt,
                r"(?:GPT-4o?|Claude|Gemini|Llama-3|Qwen3|DeepSeek(?:-[A-Za-z0-9.]+)?|Mistral)[^\n.]{0,60}(?:40 dossiers|40 resumes|achieved \d+\.\d+%)",
                "Detected potential fabricated multi-LLM empirical claim",
            )
        self.assertIn("trace-driven", body.lower())
        self.assertIn("synthetic", body.lower())

    def test_t1_f5_02_cs_complexity_and_safety_invariant_proofs(self):
        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        self.assertRegex(m3, r"O\([^)]+\)", "03_model.tex must state explicit O(.) computational complexity bounds")
        self.assertRegex(m3, r"\\begin\{(?:proposition|theorem)\}|Proposition|Theorem", "03_model.tex must include formal safety invariant Proposition/Theorem")
        self.assertRegex(m3, r"\\begin\{proof\}|Proof\.|\\emph\{Proof", "03_model.tex must include formal proof(s) of safety invariants")

    def test_t1_f5_03_architectural_accuracy_no_overclaim(self):
        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        self.assertRegex(m3.lower(), r"host-side|post-assembly|strip", "Must clarify host-side validation/stripping")
        self.assertRegex(m3.lower(), r"tf-idf|dense-proxy|sparse", "Must clarify TF-IDF dense-proxy cosine accurately")

    def test_t1_f5_04_deterministic_ablations_and_sensitivity_present(self):
        h.ensure_src_on_path()
        import ablation_and_benchmark as ab

        funcs = [name for name, obj in inspect.getmembers(ab, inspect.isfunction)]
        has_comp_ablation = any("component" in n or "module" in n or "deterministic_ablation" in n for n in funcs)
        has_sensitivity = any("sensitiv" in n or "sweep" in n for n in funcs)
        self.assertTrue(has_comp_ablation, f"Missing component-off ablation API in ablation_and_benchmark.py; found {funcs}")
        self.assertTrue(has_sensitivity, f"Missing parameter sensitivity API in ablation_and_benchmark.py; found {funcs}")
        r05 = h.strip_tex_comments(h.read_paper_sources()["05_results.tex"])
        self.assertRegex(r05.lower(), r"sensitivity|tau_\{\\mathrm\{span\}\}|l_\{\\min\}", "05_results.tex must report sensitivity analysis")

    def test_t1_f5_05_ei_empirical_systems_and_cs_bib_ratio(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        bibs = h.extract_bibitems(main_clean)
        cs_tokens = ("ieee", "acm", "emnlp", "acl", "neurips", "arxiv", "comput", "intell", "llm", "language model",
                     "retrieval", "software", "system", "sosp", "cvpr", "icassp", "ijcnn", "getmobile", "hallucination",
                     "human-mach", "human-comput", "ai act", "order no.", "personal information")
        cs_count = sum(1 for _, b in bibs if any(tok in b.lower() for tok in cs_tokens))
        self.assertGreaterEqual(cs_count, 24, f"Expected >= 24/32 CS/AI/Systems/Gov refs, got {cs_count}/{len(bibs)}")


class TestTier1F6SchemaMasking(unittest.TestCase):
    """F6: OntologySchemaMasker Contracts (5 predicates)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import OntologySchemaMasker
        self.masker = OntologySchemaMasker

    def test_t1_f6_01_schema_omits_tacit_slots_all_roles(self):
        for role, tacit in [("QA/QC", "QA_TACIT"), ("QMS", "QMS_TACIT"), ("RA", "RA_TACIT")]:
            sch = self.masker.compile_generation_schema(role)
            enum_ids = sch["properties"]["entries"]["items"]["properties"]["req_id"]["enum"]
            self.assertNotIn(tacit, enum_ids)
            self.assertIn(tacit, sch["masked_tacit_slots"])

    def test_t1_f6_02_schema_enforces_closed_status_enum(self):
        sch = self.masker.compile_generation_schema("QA/QC")
        st_enum = sch["properties"]["entries"]["items"]["properties"]["status"]["enum"]
        self.assertEqual(st_enum, ["verified", "not_provided", "conflicting"])

    def test_t1_f6_03_schema_forbids_additional_properties(self):
        sch = self.masker.compile_generation_schema("QMS")
        self.assertFalse(sch["additionalProperties"])
        self.assertFalse(sch["properties"]["entries"]["items"]["additionalProperties"])

    def test_t1_f6_04_enforce_strips_extra_root_keys(self):
        raw = {"candidate_id": "C1", "role": "RA", "overall_fit_score": 88.0, "rank": 2, "entries": []}
        clean, viols = self.masker.enforce_mask_and_validate(raw, "RA")
        self.assertNotIn("overall_fit_score", clean)
        self.assertNotIn("rank", clean)
        self.assertIn("FORBIDDEN_FIELD_STRIPPED:overall_fit_score", viols)
        self.assertIn("FORBIDDEN_FIELD_STRIPPED:rank", viols)

    def test_t1_f6_05_post_assembly_tier_v_always_blank(self):
        raw = {"candidate_id": "C2", "role": "QA/QC", "entries": []}
        clean, _ = self.masker.enforce_mask_and_validate(raw, "QA/QC")
        tacit = [e for e in clean["entries"] if e["tier"] == "V"]
        self.assertEqual(len(tacit), 1)
        self.assertIsNone(tacit[0]["status"])
        self.assertEqual(tacit[0]["evidence"], "")
        self.assertEqual(tacit[0]["anchor_lines"], [])


class TestTier1F7HybridRAG(unittest.TestCase):
    """F7: Version-Pinned Hybrid Regulatory RAG Contracts (5 predicates)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import HybridRegulatoryRAG
        self.rag = HybridRegulatoryRAG()

    def test_t1_f7_01_pinned_corpus_has_12_active_docs(self):
        self.assertEqual(len(self.rag.pinned_docs), 12)
        self.assertTrue(all(d.pinned for d in self.rag.pinned_docs))

    def test_t1_f7_02_deprecated_order_650_exists_in_all_docs(self):
        dep = [d for d in self.rag.all_docs if not d.pinned]
        self.assertTrue(dep, "Expected deprecated Order 650 clause in all_docs")
        self.assertTrue(any("650" in d.title or "650" in d.text for d in dep))

    def test_t1_f7_03_retrieve_top_k_sorted_descending(self):
        res = self.rag.retrieve("ISO 13485 internal auditor certificate IQ/OQ/PQ", top_k=5)
        self.assertEqual(len(res), 5)
        scores = [sc for _, sc in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_t1_f7_04_retrieve_never_returns_unpinned_docs(self):
        res = self.rag.retrieve("Order 650 superseded medical device rules 2014", top_k=12)
        self.assertTrue(all(doc.pinned for doc, _ in res))

    def test_t1_f7_05_retrieve_is_bit_identical_deterministic(self):
        r1 = self.rag.retrieve("Class III NMPA registration CER", top_k=4)
        r2 = self.rag.retrieve("Class III NMPA registration CER", top_k=4)
        self.assertEqual([(d.doc_id, sc) for d, sc in r1], [(d.doc_id, sc) for d, sc in r2])


class TestTier1F8DeterministicVerifier(unittest.TestCase):
    """F8: DeterministicVerifier Contracts (5 predicates)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import DeterministicVerifier, OntologySchemaMasker
        self.verifier = DeterministicVerifier(span_threshold=0.85)
        self.masker = OntologySchemaMasker

    def _make_qms_card(self, cert_ev, cert_claim=None):
        raw = {
            "candidate_id": "T",
            "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": cert_ev, "claim_text": cert_claim or cert_ev, "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        return self.masker.enforce_mask_and_validate(raw, "QMS")[0]

    def test_t1_f8_01_rule1_and_independent_rules_2_3(self):
        resume = "Header line\nLed ISO 13485 internal audits; attended auditor training; Maintained ISO 13485 system."
        card = self._make_qms_card("attended auditor training", "Certified ISO 13485 auditor")
        vcard, viols = self.verifier.verify_card(resume, card, "QMS")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_CERT") for v in viols))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QMS_CERT", viols)
        cert_entry = [e for e in vcard["entries"] if e["req_id"] == "QMS_CERT"][0]
        self.assertEqual(cert_entry["status"], "conflicting")

    def test_t1_f8_02_rule1_failure_downgrades_and_skips_rules_2_3(self):
        resume = "Header line\nMaintained ISO 13485 system."
        card = self._make_qms_card("Completely fabricated auditor text not in resume", "Certified ISO 13485 auditor")
        vcard, viols = self.verifier.verify_card(resume, card, "QMS")
        self.assertIn("UNANCHORED_SPAN:QMS_CERT", viols)
        self.assertFalse(any("QMS_CERT" in v and "VERB_ESCALATION" in v for v in viols))

    def test_t1_f8_03_rule2_failure_downgrades_with_verb_escalation(self):
        resume = "Header line\nAssisted ISO 13485 internal audits; Certified ISO 13485 auditor, Certificate No. CERT-1; Maintained ISO 13485 system."
        raw = {
            "candidate_id": "T2", "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "evidence": "Certified ISO 13485 auditor, Certificate No. CERT-1", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_LEAD", "evidence": "Assisted ISO 13485 internal audits", "claim_text": "Led ISO 13485 internal audits", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QMS_EQUIV", "evidence": "Maintained ISO 13485 system", "anchor_lines": [2, 2], "status": "verified"},
            ],
        }
        masked, _ = self.masker.enforce_mask_and_validate(raw, "QMS")
        vcard, viols = self.verifier.verify_card(resume, masked, "QMS")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QMS_LEAD") for v in viols))
        lead = [e for e in vcard["entries"] if e["req_id"] == "QMS_LEAD"][0]
        self.assertEqual(lead["status"], "conflicting")

    def test_t1_f8_04_rule3_failure_downgrades_with_missing_credential(self):
        resume = "Header line\nLed ISO 13485 internal audits; internal audit training workshop; Maintained ISO 13485 system."
        card = self._make_qms_card("Led ISO 13485 internal audits; internal audit training workshop", "Led ISO 13485 internal audits")
        vcard, viols = self.verifier.verify_card(resume, card, "QMS")
        self.assertIn("MISSING_CREDENTIAL_PROOF:QMS_CERT", viols)
        cert = [e for e in vcard["entries"] if e["req_id"] == "QMS_CERT"][0]
        self.assertEqual(cert["status"], "conflicting")

    def test_t1_f8_05_non_verified_preserved_and_non_blank_v_reset(self):
        resume = "Line 1\nLine 2"
        card = {
            "candidate_id": "T5", "role": "QMS",
            "entries": [
                {"req_id": "QMS_CERT", "tier": "M", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QMS_LEAD", "tier": "M", "evidence": "bad", "anchor_lines": [], "status": "conflicting"},
                {"req_id": "QMS_EQUIV", "tier": "P", "evidence": "", "anchor_lines": [], "status": "not_provided"},
                {"req_id": "QMS_TACIT", "tier": "V", "evidence": "ethical", "anchor_lines": [1, 1], "status": "verified"},
            ],
        }
        vcard, viols = self.verifier.verify_card(resume, card, "QMS")
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QMS_TACIT", viols)
        tac = [e for e in vcard["entries"] if e["req_id"] == "QMS_TACIT"][0]
        self.assertIsNone(tac["status"])
        self.assertEqual(tac["evidence"], "")


class TestTier1F9FSMDecisionGate(unittest.TestCase):
    """F9: FSMDecisionGate Contracts (5 predicates)."""

    def setUp(self):
        h.ensure_src_on_path()
        from eica_engine import FSMDecisionGate
        self.gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)

    def _clean_card(self):
        return {
            "candidate_id": "C", "role": "QMS", "verifier_violations": [],
            "entries": [
                {"req_id": "QMS_CERT", "tier": "M", "status": "verified"},
                {"req_id": "QMS_LEAD", "tier": "M", "status": "verified"},
                {"req_id": "QMS_EQUIV", "tier": "P", "status": "not_provided"},
                {"req_id": "QMS_TACIT", "tier": "V", "status": None},
            ],
        }

    def test_t1_f9_01_confirm_1_clean_card_transitions_to_s_term(self):
        res = self.gate.step(self._clean_card(), "Confirm", proposed_decision=1)
        self.assertEqual(res.next_state, "s_term")
        self.assertEqual(res.final_decision, "1")
        self.assertIsNone(res.error_code)

    def test_t1_f9_02_confirm_1_with_conflict_or_violations_blocks(self):
        c1 = self._clean_card()
        c1["entries"][2]["status"] = "conflicting"
        r1 = self.gate.step(c1, "Confirm", proposed_decision=1)
        self.assertEqual((r1.next_state, r1.final_decision, r1.error_code), ("s_block", None, "ERR_UNRESOLVED_CONFLICT_LOCK"))

        c2 = self._clean_card()
        c2["verifier_violations"] = ["UNANCHORED_SPAN:QMS_CERT"]
        r2 = self.gate.step(c2, "Confirm", proposed_decision=1)
        self.assertEqual((r2.next_state, r2.final_decision, r2.error_code), ("s_block", None, "ERR_UNRESOLVED_CONFLICT_LOCK"))

    def test_t1_f9_03_confirm_1_with_missing_mandatory_blocks(self):
        c = self._clean_card()
        c["entries"][1]["status"] = "not_provided"
        r = self.gate.step(c, "Confirm", proposed_decision=1)
        self.assertEqual((r.next_state, r.final_decision, r.error_code), ("s_block", None, "ERR_MANDATORY_GUARD_FAILED"))

    def test_t1_f9_04_adjust_1_enforces_l_min_g_esc_and_g_m(self):
        c = self._clean_card()
        c["entries"][1]["status"] = "not_provided"
        r_short = self.gate.step(c, "Adjust", proposed_decision=1, reason="too short", adjusted_statuses={"QMS_LEAD": "verified"})
        self.assertEqual(r_short.error_code, "ERR_JUSTIFICATION_TOO_SHORT")

        r_ok = self.gate.step(c, "Adjust", proposed_decision=1, reason="Verified original physical diploma on file", adjusted_statuses={"QMS_LEAD": "verified"})
        self.assertEqual((r_ok.next_state, r_ok.final_decision), ("s_term", "1"))

    def test_t1_f9_05_escalate_unconditionally_transitions_to_s_esc(self):
        c = self._clean_card()
        c["verifier_violations"] = ["VERB_ESCALATION:QMS_LEAD"]
        r = self.gate.step(c, "Escalate", reason="Needs manager review")
        self.assertEqual((r.next_state, r.final_decision), ("s_esc", "Esc"))
        self.assertIn("violations", r.audit_entry)


class TestTier1F10DynamicFigures(unittest.TestCase):
    """F10: Dynamic Vector & Raster Figure Generation (5 predicates)."""

    def test_t1_f10_01_fig1_generates_valid_pdf_and_300dpi_png(self):
        res = h.get_reproduce_run()
        self.assertGreater(len(res.fig1_pdf_bytes), 5000)
        self.assertGreater(len(res.fig1_png_bytes), 50000)
        w_px, _ = h.png_pixel_size(res.fig1_png_bytes)
        self.assertGreaterEqual(w_px, 2000)

    def test_t1_f10_02_fig2_generates_valid_pdf_and_png(self):
        res = h.get_reproduce_run()
        self.assertGreater(len(res.fig2_pdf_bytes), 5000)
        self.assertGreater(len(res.fig2_png_bytes), 50000)

    def test_t1_f10_03_fig2_computes_dynamically_from_csv_path(self):
        h.ensure_src_on_path()
        from plot_figures import plot_fig2_results

        cols, rows = h.load_csv_rows(h.EVAL_CSV_PATH)
        with tempfile.TemporaryDirectory(prefix="eica_dynfig_") as td:
            td_p = Path(td)
            mod_csv = td_p / "mod_eval.csv"
            import csv
            with open(mod_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    r2 = dict(r)
                    r2["llm_hr_decision"] = r2["dept_decision"]
                    w.writerow(r2)
            p_orig = td_p / "orig.pdf"
            p_mod = td_p / "mod.pdf"
            plot_fig2_results(str(h.EVAL_CSV_PATH), str(p_orig), str(td_p / "orig.png"))
            plot_fig2_results(str(mod_csv), str(p_mod), str(td_p / "mod.png"))
            t_orig = h.analyze_pdf_file(p_orig).full_text
            t_mod = h.analyze_pdf_file(p_mod).full_text
            self.assertNotEqual(t_orig, t_mod, "plot_fig2_results ignored csv_path and produced identical hardcoded figure")

    def test_t1_f10_04_zero_pua_glyphs_in_generated_figure_pdfs(self):
        res = h.get_reproduce_run()
        t1 = h.analyze_pdf_bytes(res.fig1_pdf_bytes)
        t2 = h.analyze_pdf_bytes(res.fig2_pdf_bytes)
        self.assertEqual(t1.pua_chars, [], f"PUA chars in fig1: {t1.pua_chars}")
        self.assertEqual(t2.pua_chars, [], f"PUA chars in fig2: {t2.pua_chars}")

    def test_t1_f10_05_fig1_synced_12_docs_and_rule2_3(self):
        res = h.get_reproduce_run()
        txt = h.analyze_pdf_bytes(res.fig1_pdf_bytes).full_text
        self.assertIn("12 docs", txt)
        self.assertIn("Verb-Escalation", txt)
        self.assertIn("Credential Proof", txt)


class TestTier1F11SyntheticAblations(unittest.TestCase):
    """F11: Synthetic Stress-Test & Component-Off Ablations (5 predicates)."""

    def test_t1_f11_01_synthetic_benchmark_6_strata_sum_200(self):
        s = h.get_cached_synthetic_42()
        self.assertEqual(len(s["strata"]), 6)
        self.assertEqual(sum(v["n"] for v in s["strata"].values()), 200)

    def test_t1_f11_02_synthetic_benchmark_deterministic_seed_42(self):
        h.ensure_src_on_path()
        from ablation_and_benchmark import run_synthetic_verification_benchmark

        s1 = h.get_cached_synthetic_42()
        s2 = run_synthetic_verification_benchmark(seed=42)
        self.assertEqual(s1["strata"], s2["strata"])
        self.assertEqual(s1["extraction_verifier_summary"], s2["extraction_verifier_summary"])

    def test_t1_f11_03_unpinned_rag_ablation_exposes_order_650_leakage(self):
        h.ensure_src_on_path()
        from eica_engine import HybridRegulatoryRAG

        q = "Order 650 superseded medical device supervision rules 2014"
        pinned = HybridRegulatoryRAG(pinned_only=True).retrieve(q, top_k=3)
        unpinned = HybridRegulatoryRAG(pinned_only=False).retrieve(q, top_k=3)
        self.assertTrue(all(c.pinned for c, _ in pinned))
        self.assertTrue(any(not c.pinned for c, _ in unpinned))

    def test_t1_f11_04_soft_fsm_gate_ablation_allows_bypasses(self):
        h.ensure_src_on_path()
        from eica_engine import FSMDecisionGate

        card = {
            "candidate_id": "X", "role": "QMS", "verifier_violations": ["VERB_ESCALATION:QMS_LEAD"],
            "entries": [
                {"req_id": "QMS_CERT", "tier": "M", "status": "verified"},
                {"req_id": "QMS_LEAD", "tier": "M", "status": "verified"},
                {"req_id": "QMS_EQUIV", "tier": "P", "status": "not_provided"},
            ],
        }
        hard_res = FSMDecisionGate(enforce_hard_escalation_lock=True).step(card, "Confirm", proposed_decision=1)
        soft_res = FSMDecisionGate(enforce_hard_escalation_lock=False).step(card, "Confirm", proposed_decision=1)
        self.assertEqual(hard_res.next_state, "s_block")
        self.assertEqual((soft_res.next_state, soft_res.final_decision), ("s_term", "1"))

    def test_t1_f11_05_sensitivity_sweep_apis_and_bounds(self):
        h.ensure_src_on_path()
        import ablation_and_benchmark as ab

        sweep_fn = next(
            (obj for name, obj in inspect.getmembers(ab, inspect.isfunction)
             if "sensitiv" in name or "sweep" in name),
            None,
        )
        self.assertIsNotNone(sweep_fn, "Missing sensitivity sweep function in ablation_and_benchmark.py")
        out = sweep_fn()
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
