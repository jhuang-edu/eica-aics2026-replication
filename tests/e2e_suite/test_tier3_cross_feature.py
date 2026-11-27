# -*- coding: utf-8 -*-
"""
Tier 3 E2E Verification Suite: Pairwise Cross-Feature Combinations (11 Tests).

Verifies end-to-end interactions across F1..F11 per TEST_INFRA.md:
  - T3-01: F1 (CSV) + F10 (plot_fig2_results) + F4 (Tables 1-3)
  - T3-02: F6 (SchemaMasker) + F8 (Verifier) + F9 (FSMGate)
  - T3-03: F7 (Pinned RAG) + F4 (03_model.tex) + F10 (Fig. 1)
  - T3-04: F8 (Verifier) + F4 (Alg. 1 & 05_results.tex) + F2 (IEEE captions/tags)
  - T3-05: F1 (reproduce_all.py) + F4 (RESPONSE_LETTER & READMEs) + F5 (Integrity)
  - T3-06: F2 (IEEE Checklist) + F3 (Tectonic) + F10 (Fresh Figures)
  - T3-07: F1 (M0-M3 Ablation) + F10 (Fig. 2 Panel c) + F4 (05_results.tex)
  - T3-08: F11 (Synthetic + Component-off Ablation) + F4 (05_results.tex) + F5 (C2)
  - T3-09: F7 (HybridRAG) + F8 (Verifier) + F4 (R05/R37/R31 prose)
  - T3-10: F9 (FSMGate) + F2 (Eq. 5 & Eq. 6) + F4 (Audit log contract)
  - T3-11: F2 (32 monotonic bibitems) + F4 (READMEs/Response Letter) + F5 (CS ratio)
"""

from __future__ import annotations

import csv
import inspect
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from . import e2e_helpers as h
except ImportError:
    import e2e_helpers as h


class TestTier3CrossFeature(unittest.TestCase):
    """11 pairwise cross-feature combinatorial tests."""

    def setUp(self):
        h.ensure_src_on_path()

    @h.requires_pymupdf
    def test_t3_01_csv_mutation_propagates_to_plot_fig2_and_real_matches_tables(self):
        from plot_figures import plot_fig2_results

        cols, rows = h.load_csv_rows(h.EVAL_CSV_PATH)
        with tempfile.TemporaryDirectory(prefix="eica_t301_") as td:
            td_p = Path(td)
            mod_csv = td_p / "eval_mod.csv"
            with open(mod_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    r2 = dict(r)
                    r2["llm_hr_decision"] = r2["dept_decision"]
                    w.writerow(r2)
            p_real = td_p / "real.pdf"
            p_mod = td_p / "mod.pdf"
            plot_fig2_results(str(h.EVAL_CSV_PATH), str(p_real), str(td_p / "real.png"))
            plot_fig2_results(str(mod_csv), str(p_mod), str(td_p / "mod.png"))

            t_real = h.analyze_pdf_file(p_real).full_text
            t_mod = h.analyze_pdf_file(p_mod).full_text
            self.assertNotEqual(t_real, t_mod, "plot_fig2_results must compute dynamically from csv_path")

            m = h.get_cached_metrics()
            abl = h.get_cached_ablation_20k()
            for val_str in (
                f"{m['agreement']['manual']:.1f}%",
                f"{m['agreement']['eica']:.1f}%",
                f"{abl['M2_EICA_HardFSM']['agreement_pct']:.1f}%",
                f"{abl['M3_EICA_FullVerifier']['agreement_pct']:.1f}%",
            ):
                self.assertIn(val_str, t_real, f"Missing {val_str} in Fig. 2 PDF text")

    def test_t3_02_end_to_end_masker_verifier_fsm_pipeline(self):
        from eica_engine import DeterministicVerifier, FSMDecisionGate, OntologySchemaMasker

        resume = (
            "Education: B.Eng. Biomedical Engineering, 2016-2020.\n"
            "Experience: Participated in ISO 13485 internal audit preparations; 4 years Class III QA.\n"
            "Credentials: Attended ISO 13485 internal auditor training workshop."
        )
        raw_card = {
            "candidate_id": "PIPE-01",
            "role": "QA/QC",
            "overall_fit_score": 95.0,
            "entries": [
                {"req_id": "QA_DEG", "evidence": "B.Eng. Biomedical Engineering, 2016-2020", "anchor_lines": [1, 1], "status": "verified"},
                {"req_id": "QA_EXP", "evidence": "4 years Class III QA", "anchor_lines": [2, 2], "status": "verified"},
                {"req_id": "QA_CERT", "evidence": "Attended ISO 13485 internal auditor training workshop", "claim_text": "Certified ISO 13485 internal auditor", "anchor_lines": [3, 3], "status": "verified"},
                {"req_id": "QA_TACIT", "evidence": "Strong ethical composure", "anchor_lines": [1, 1], "status": "verified"},
            ],
        }
        masked, mv = OntologySchemaMasker.enforce_mask_and_validate(raw_card, "QA/QC")
        self.assertIn("FORBIDDEN_FIELD_STRIPPED:overall_fit_score", mv)
        self.assertIn("TACIT_EXTRAPOLATION_BLOCKED:QA_TACIT", mv)
        tacit_after_mask = [e for e in masked["entries"] if e["req_id"] == "QA_TACIT"][0]
        self.assertIsNone(tacit_after_mask["status"])

        verified, vv = DeterministicVerifier(span_threshold=0.85).verify_card(resume, masked, "QA/QC")
        self.assertTrue(any(v.startswith("VERB_ESCALATION:QA_CERT") for v in vv))
        self.assertIn("MISSING_CREDENTIAL_PROOF:QA_CERT", vv)

        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        res_block = gate.step(verified, "Confirm", proposed_decision=1)
        self.assertEqual((res_block.next_state, res_block.final_decision, res_block.error_code),
                         ("s_block", None, "ERR_UNRESOLVED_CONFLICT_LOCK"))
        res_esc = gate.step(verified, "Escalate", reason="Verifier flagged certification escalation")
        self.assertEqual((res_esc.next_state, res_esc.final_decision), ("s_esc", "Esc"))
        self.assertEqual(res_esc.audit_entry["violations"], vv)

    @h.requires_full_workspace
    def test_t3_03_dpin_corpus_matches_model_tex_and_fig1(self):
        from eica_engine import HybridRegulatoryRAG

        rag = HybridRegulatoryRAG()
        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        fig1_txt = h.analyze_pdf_file(h.PAPER_FIG_DIR / "fig1_architecture.pdf").full_text
        self.assertEqual(len(rag.pinned_docs), 12)
        self.assertRegex(m3, r"\b12\s+active\b")
        self.assertIn("12 docs", fig1_txt)
        self.assertIn(r"\widehat{\mathrm{BM25}}(q, d)", m3)
        self.assertIn(r"\cos(\mathbf{v}_q, \mathbf{v}_d)", m3)

    @h.requires_full_workspace
    def test_t3_04_verifier_r05_r31_matches_alg1_and_results_tex(self):
        from ablation_and_benchmark import run_exemplar_verification_replay

        replays = {x["case_id"]: x for x in run_exemplar_verification_replay()}
        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        r05_tex = h.strip_tex_comments(h.read_paper_sources()["05_results.tex"])

        for tag in ("UNANCHORED_SPAN", "VERB_ESCALATION", "MISSING_CREDENTIAL_PROOF"):
            self.assertIn(tag.replace("_", r"\_"), m3)
            self.assertIn(tag.replace("_", r"\_"), r05_tex)
        self.assertTrue(any("MISSING_CREDENTIAL_PROOF" in v for v in replays["R05"]["verifier_violations"]))
        self.assertTrue(any("VERB_ESCALATION" in v for v in replays["R05"]["verifier_violations"]))
        self.assertTrue(any("UNANCHORED_SPAN" in v for v in replays["R31"]["verifier_violations"]))
        self.assertTrue(any("VERB_ESCALATION" in v for v in replays["R31"]["verifier_violations"]))

    @h.requires_full_workspace
    def test_t3_05_reproduce_all_matches_response_letter_and_readmes(self):
        res = h.get_reproduce_run()
        self.assertEqual(res.returncode, 0)
        ledger = h.parse_reproduce_ledger(res.stdout)
        self.assertGreaterEqual(len(ledger), 28)
        self.assertTrue(all(r[3] == "[MATCH 100%]" for r in ledger))

        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        code_rm = (h.REPO_ROOT / "code" / "README.md").read_text(encoding="utf-8")
        for tok in ("87.5%", "92.5%", "95.0%", "0.746", "0.845", "0.896",
                    "0.00183", "0.00098"):
            self.assertIn(tok, rl, f"RESPONSE_LETTER missing {tok}")
        self.assertTrue("0.088" in rl or "0.0875" in rl, "RESPONSE_LETTER missing 0.088 or 0.0875")
        self.assertNotRegex(rl, r"\b0\.087\b")
        self.assertNotIn("-43.7", rl)
        self.assertRegex(
            rl,
            rf"(?i)\b(?:28|42|{len(ledger)})\b[^\n]{{0,40}}(?:claims|checks|assertions|metrics|audit)",
            "RESPONSE_LETTER missing check/claim count (28 or 42)",
        )
        self.assertRegex(
            code_rm,
            rf"(?i)(?<!seed )\b(?:28|42|{len(ledger)})\b[^\n]{{0,30}}(?:checks|assertions|claims|metrics|audit|quantitative|项|項目|の)",
            "code/README.md missing updated check count (28 or 42)",
        )
        self.assertIn("95.0%", code_rm)
        self.assertIn("0.896", code_rm)
        self.assertNotRegex(code_rm, r"\b0\.087\b")
        self.assertNotIn("-43.7", code_rm)

    @h.requires_full_workspace
    def test_t3_06_tectonic_compile_with_fresh_generated_figures(self):
        rep = h.get_reproduce_run()
        self.assertEqual(rep.returncode, 0)
        with tempfile.TemporaryDirectory(prefix="eica_t306_") as td:
            td_p = Path(td)
            paper_copy = td_p / "paper"
            shutil.copytree(h.PAPER_DIR, paper_copy)
            (paper_copy / "figures" / "fig1_architecture.pdf").write_bytes(rep.fig1_pdf_bytes)
            (paper_copy / "figures" / "fig2_results.pdf").write_bytes(rep.fig2_pdf_bytes)
            out_dir = td_p / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["tectonic", "--keep-logs", "-o", str(out_dir), str(paper_copy / "main.tex")],
                cwd=str(paper_copy),
                env=h.build_subprocess_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, f"Fresh-figure tectonic build failed:\n{proc.stderr}")
            tel = h.analyze_pdf_file(out_dir / "main.pdf")
            self.assertEqual(tel.pua_chars, [])
            self.assertEqual(tel.cjk_chars, [])
            self.assertEqual(tel.unresolved_refs, [])
            self.assertEqual(tel.wrapped_eq_numbers, [])
            self.assertFalse(tel.has_lmroman_fallback)

    @h.requires_full_workspace
    def test_t3_07_ablation_m0_m3_metrics_match_expected_loss_panel_c(self):
        abl = h.get_cached_ablation_20k()
        self.assertEqual(abl["M0_Manual"]["expected_loss_by_lambda_fa"]["10"], 2.4)
        self.assertAlmostEqual(abl["M1_EICA_SoftCard"]["expected_loss_by_lambda_fa"]["10"], 0.83125, delta=0.0001)
        self.assertAlmostEqual(abl["M2_EICA_HardFSM"]["expected_loss_by_lambda_fa"]["10"], 0.33125, delta=0.0001)
        self.assertEqual(abl["M3_EICA_FullVerifier"]["expected_loss_by_lambda_fa"]["10"], 0.0875)

        r05 = h.read_paper_sources()["05_results.tex"]
        for tok in ("2.400", "0.831", "0.331", "96.4\\%"):
            self.assertIn(tok, r05)
        self.assertTrue("0.088" in r05 or "0.0875" in r05, "Expected 0.088 or 0.0875 in 05_results.tex")
        self.assertNotRegex(r05, r"\b0\.087\b")

    @h.requires_full_workspace
    def test_t3_08_synthetic_benchmark_and_component_ablation_match_results_tex(self):
        import ablation_and_benchmark as ab

        stress = h.get_cached_synthetic_42()
        self.assertEqual(stress["extraction_verifier_summary"]["recall_pct"], 100.0)
        comp_fn = next(
            (obj for name, obj in inspect.getmembers(ab, inspect.isfunction)
             if "component" in name or "module" in name or "deterministic_ablation" in name),
            None,
        )
        self.assertIsNotNone(comp_fn, "Missing deterministic component-off ablation API")
        comp_out = comp_fn()
        self.assertTrue(comp_out)
        r05 = h.strip_tex_comments(h.read_paper_sources()["05_results.tex"]).lower()
        self.assertIn("ablation", r05)
        self.assertIn("200", r05)

    @h.requires_full_workspace
    def test_t3_09_rag_top_clauses_and_verifier_tags_consistent_across_engine_and_paper(self):
        from ablation_and_benchmark import run_exemplar_verification_replay

        replays = {x["case_id"]: x for x in run_exemplar_verification_replay()}
        r05_tex = h.strip_tex_comments(h.read_paper_sources()["05_results.tex"])
        self.assertIn("ISO13485-8.2.4", replays["R05"]["top_rag_clauses"])
        self.assertIn("ISO13485-8.2.4", r05_tex)
        self.assertIn("SOP-RA-02", replays["R37"]["top_rag_clauses"])
        self.assertIn("SOP-RA-02", r05_tex)

    @h.requires_full_workspace
    def test_t3_10_fsm_confirm_adjust_escalate_align_with_eq5_eq6_and_audit_log(self):
        from eica_engine import FSMDecisionGate

        m3 = h.strip_tex_comments(h.read_paper_sources()["03_model.tex"])
        self.assertIn("ERR_", repr(FSMDecisionGate().step({"entries": []}, "BadAction").error_code))
        self.assertIn(r"\mathcal{G}_{\mathrm{M}}", m3)
        self.assertIn(r"\mathcal{G}_{\mathrm{esc}}", m3)
        self.assertIn(r"L_{\min}", m3)
        gate = FSMDecisionGate(l_min=15, enforce_hard_escalation_lock=True)
        card = {
            "candidate_id": "T310", "role": "QMS", "verifier_violations": [],
            "entries": [
                {"req_id": "QMS_CERT", "tier": "M", "status": "verified"},
                {"req_id": "QMS_LEAD", "tier": "M", "status": "verified"},
                {"req_id": "QMS_EQUIV", "tier": "P", "status": "verified"},
            ],
        }
        r_c = gate.step(card, "Confirm", proposed_decision=1)
        r_a = gate.step(card, "Adjust", proposed_decision=1, reason="Documented cross-standard equivalence verified")
        r_e = gate.step(card, "Escalate", reason="Escalate to manager")
        self.assertEqual((r_c.next_state, r_a.next_state, r_e.next_state), ("s_term", "s_term", "s_esc"))
        for r in (r_c, r_a, r_e):
            self.assertIn("ts", r.audit_entry)
            self.assertIn("action", r.audit_entry)

    @h.requires_full_workspace
    def test_t3_11_bib_keys_inline_count_32_matches_response_letter_and_readmes(self):
        main_clean = h.strip_tex_comments(h.read_paper_sources()["main.tex"])
        bibs = h.extract_bibitems(main_clean)
        self.assertEqual(len(bibs), 32)
        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")
        root_rm = (h.REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertRegex(rl, r"(?i)\b32\b[^\n]{0,35}references")
        self.assertIn("32", root_rm)
        self.assertNotIn("34 篇", root_rm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
