# -*- coding: utf-8 -*-
"""
Tier 4 E2E Verification Suite: Real-World Submission Audit Scenarios (6 Scenarios).

Simulates full submission gatekeeping audits per TEST_INFRA.md:
  - T4-S1: Full Camera-Ready Compile & IEEE Checklist Audit (F2, F3)
  - T4-S2: Isolated One-Click Reproduction & Dynamic Plot Audit (F1, F10)
  - T4-S3: Reviewer Critique (C1-C3) & Option 1 Integrity Audit (F4, F5, F11)
  - T4-S4: Adversarial Input & Runtime Stress-Test Replay (F6, F7, F8, F9, F11)
  - T4-S5: Submission Repository Hygiene & Documentation Sync Audit (F4)
  - T4-S6: Non-Destructive Pre-Built PDF & Dataset Immutability Parity Check (F1, F2, F3, F10)
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


class TestTier4SubmissionScenarios(unittest.TestCase):
    """6 full end-to-end submission audit scenarios."""

    def test_t4_s1_full_camera_ready_compile_and_ieee_checklist_audit(self):
        build = h.get_tectonic_build()
        self.assertEqual(build.returncode, 0, f"Tectonic build failed:\n{build.stderr}")
        log = build.log_text
        self.assertEqual(re.findall(r".*TU/ptm/.*undefined.*", log), [], "TU/ptm font fallback warnings present")
        self.assertEqual(re.findall(r".*TU/pcr/.*undefined.*", log), [], "TU/pcr font fallback warnings present")
        self.assertEqual(re.findall(r".*Invalid UTF-8.*", log), [], "Invalid UTF-8 warnings present")
        self.assertEqual(re.findall(r".*Underfull \\hbox \(badness 10000\).*", log), [], "badness 10000 underfulls present")

        tel = h.analyze_pdf_bytes(build.pdf_bytes)
        self.assertTrue(4 <= tel.page_count <= 20)
        for w, ht in tel.page_sizes:
            self.assertAlmostEqual(w, 612.0, delta=1.5)
            self.assertAlmostEqual(ht, 792.0, delta=1.5)
        self.assertTrue(tel.has_times_font and tel.has_mono_font and not tel.has_lmroman_fallback)
        self.assertEqual(tel.cjk_chars, [])
        self.assertEqual(tel.pua_chars, [])
        self.assertEqual(tel.unresolved_refs, [])
        self.assertEqual(tel.orphaned_punct, [])
        self.assertEqual(tel.wrapped_eq_numbers, [])
        self.assertEqual(tel.table_small_spans, [])
        self.assertIsNotNone(tel.last_page_col_diff_pt)
        self.assertLessEqual(tel.last_page_col_diff_pt, 72.0)

        sources = h.read_paper_sources()
        body = h.combined_body_tex()
        self.assertEqual(re.findall(r"(?<!Equation~)(?<!Equation )Eq\.\s*~?(?:\(\\ref|\\eqref)\{", body), [])
        self.assertNotIn(r"\usepackage{algorithm}", h.strip_tex_comments(sources["main.tex"]))
        bibs = h.extract_bibitems(sources["main.tex"])
        self.assertEqual(len(bibs), 32)
        self.assertEqual(h.extract_ordered_citations(sources), [k for k, _ in bibs])

    def test_t4_s2_isolated_one_click_reproduction_and_dynamic_plot_audit(self):
        sha_before = h.compute_sha256(h.EVAL_CSV_PATH)
        rep = h.get_reproduce_run()
        self.assertEqual(rep.returncode, 0, f"reproduce_all.py failed:\n{rep.stderr}")
        ledger = h.parse_reproduce_ledger(rep.stdout)
        self.assertGreaterEqual(len(ledger), 28)
        self.assertTrue(all(r[3] == "[MATCH 100%]" for r in ledger), f"Ledger mismatch: {[r for r in ledger if r[3] != '[MATCH 100%]']}")

        for pdf_b in (rep.fig1_pdf_bytes, rep.fig2_pdf_bytes):
            ftel = h.analyze_pdf_bytes(pdf_b)
            self.assertEqual(ftel.pua_chars, [])
            self.assertEqual(ftel.cjk_chars, [])

        h.ensure_src_on_path()
        from plot_figures import plot_fig2_results
        cols, rows = h.load_csv_rows(h.EVAL_CSV_PATH)
        with tempfile.TemporaryDirectory(prefix="eica_s2_") as td:
            td_p = Path(td)
            mod_csv = td_p / "mod.csv"
            with open(mod_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    r2 = dict(r)
                    r2["llm_hr_decision"] = r2["dept_decision"]
                    w.writerow(r2)
            p_mod = td_p / "fig2_mod.pdf"
            plot_fig2_results(str(mod_csv), str(p_mod), str(td_p / "fig2_mod.png"))
            self.assertNotEqual(
                h.analyze_pdf_bytes(rep.fig2_pdf_bytes).full_text,
                h.analyze_pdf_file(p_mod).full_text,
                "plot_fig2_results must react to modified CSV",
            )
        self.assertEqual(h.compute_sha256(h.EVAL_CSV_PATH), sha_before)

    def test_t4_s3_reviewer_critique_c1_c3_and_option1_integrity_audit(self):
        sources = h.read_paper_sources()
        m3 = h.strip_tex_comments(sources["03_model.tex"])
        r05 = h.strip_tex_comments(sources["05_results.tex"])
        rl = (h.REPO_ROOT / "RESPONSE_LETTER_AICS2026.md").read_text(encoding="utf-8")

        # C1: Algorithmic depth, complexity, formal safety proofs, accurate RAG/schema framing
        self.assertRegex(m3, r"O\([^)]+\)")
        self.assertRegex(m3, r"\\begin\{(?:proposition|theorem)\}|Proposition|Theorem")
        self.assertRegex(m3.lower(), r"tf-idf|dense-proxy")
        self.assertRegex(m3.lower(), r"host-side|post-assembly|strip")

        # C2: Option 1 integrity + trace-driven ablation + synthetic component ablations & sensitivity
        self.assertNotRegex(
            m3 + r05 + rl,
            r"(?:GPT-4o?|Claude|Gemini|Llama-3|Qwen3|DeepSeek(?:-[A-Za-z0-9.]+)?|Mistral)[^\n.]{0,60}(?:40 dossiers|40 resumes|achieved \d+\.\d+%)",
        )
        self.assertIn("counterfactual", r05.lower())
        self.assertRegex(r05.lower(), r"sensitivity|ablation")

        # C3: CS bibliography ratio + accurate Tectonic statement + grammar
        bibs = h.extract_bibitems(sources["main.tex"])
        self.assertEqual(len(bibs), 32)
        self.assertNotIn("reviewer's core", rl)
        self.assertNotRegex(rl, r"\b0\.087\b")
        log = h.get_tectonic_build().log_text
        self.assertEqual(re.findall(r"TU/ptm/|TU/pcr/|Invalid UTF-8", log), [])

    def test_t4_s4_adversarial_input_and_runtime_stress_test_replay(self):
        h.ensure_src_on_path()
        from ablation_and_benchmark import run_exemplar_verification_replay
        from eica_engine import DeterministicVerifier, FSMDecisionGate, HybridRegulatoryRAG, OntologySchemaMasker

        stress = h.get_cached_synthetic_42()
        self.assertEqual(stress["extraction_verifier_summary"]["recall_pct"], 100.0)
        self.assertEqual(stress["extraction_verifier_summary"]["precision_pct"], 100.0)
        self.assertEqual(stress["schema_tacit_block_rate_pct"], 100.0)
        self.assertEqual(stress["fsm_invariant_block_rate_pct"], 100.0)
        self.assertEqual(stress["rag_whitelist_pin_compliance_pct"], 100.0)
        self.assertLess(stress["latency_ms"]["p50"], 25.0)
        self.assertLess(stress["latency_ms"]["p95"], 100.0)

        replays = {x["case_id"]: x for x in run_exemplar_verification_replay()}
        self.assertEqual(replays["R05"]["naive_confirm_state"], "s_block")
        self.assertEqual(replays["R37"]["naive_confirm_state"], "s_block")
        self.assertEqual(replays["R31"]["naive_confirm_state"], "s_block")

        # Adversarial pipeline checks: Order 650 exclusion, Tier-P v_src==0, Adjust conflicting lock
        rag = HybridRegulatoryRAG()
        self.assertTrue(all(d.pinned for d, _ in rag.retrieve("Order 650 2014", top_k=12)))
        v = DeterministicVerifier()
        card, _ = OntologySchemaMasker.enforce_mask_and_validate({
            "candidate_id": "ADV", "role": "QA/QC",
            "entries": [{"req_id": "QA_VAL", "evidence": "Cleanroom monitoring", "claim_text": "Led cleanroom validation", "anchor_lines": [1, 1], "status": "verified"}],
        }, "QA/QC")
        vc, vv = v.verify_card("Cleanroom monitoring", card, "QA/QC")
        self.assertTrue(any("VERB_ESCALATION:QA_VAL" in x for x in vv))
        res = FSMDecisionGate().step(vc, "Adjust", proposed_decision=1, reason="Attempt override > 15 chars", adjusted_statuses={"QA_VAL": "conflicting"})
        self.assertEqual(res.next_state, "s_block")

    def test_t4_s5_submission_repo_hygiene_and_documentation_sync_audit(self):
        lock_files = [str(p) for p in h.REPO_ROOT.rglob("~$*")]
        self.assertEqual(lock_files, [], f"Office lock files found: {lock_files}")
        stale_figs = [str(p) for p in h.SRC_DIR.glob("fig*") if p.is_file()]
        self.assertEqual(stale_figs, [], f"Stale figures in code/src/: {stale_figs}")

        root_rm = (h.REPO_ROOT / "README.md").read_text(encoding="utf-8")
        code_rm = (h.REPO_ROOT / "code" / "README.md").read_text(encoding="utf-8")
        for bad in ("EICA-AI-CS-2026-核查最终总结-muse.pdf", "references.bib", "34 篇", "23项", "pandas", "seaborn",
                    "Human-AI Division of Labor", "Anonymous Authors"):
            self.assertNotIn(bad, root_rm)
            self.assertNotIn(bad, code_rm)
        stale_23_re = r"(?i)(?<!\d)23\s*(?:项|checks|metrics|indicators|assertions|core|項目|个|の)"
        self.assertNotRegex(root_rm, stale_23_re)
        self.assertNotRegex(code_rm, stale_23_re)
        self.assertRegex(
            code_rm,
            r"(?i)(?<!seed )\b(?:28|42)\b[^\n]{0,30}(?:checks|assertions|claims|metrics|audit|quantitative|项|項目|の)",
        )
        for req in ("eica_engine.py", "ablation_and_benchmark.py", "vllm_config.yaml"):
            self.assertIn(req, code_rm)

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
            self.assertNotIn("Qwen3.8-27B", txt, f"Stale Qwen3.8-27B found in {label}")
        for label, txt in [("README.md", root_rm), ("code/README.md", code_rm), ("RESPONSE_LETTER_AICS2026.md", rl_txt)]:
            self.assertIn("62", txt, f"Expected synced 62 unit-test count in {label}")

    def test_t4_s6_non_destructive_prebuilt_pdf_and_dataset_immutability_parity(self):
        self.assertEqual(h.compute_sha256(h.EVAL_CSV_PATH), h.EXPECTED_EVAL_CSV_SHA256)
        self.assertEqual(h.compute_sha256(h.ARCHIVE_CSV_PATH), h.EXPECTED_ARCHIVE_CSV_SHA256)

        prebuilt_pdf = h.PAPER_DIR / "main.pdf"
        self.assertTrue(prebuilt_pdf.exists(), "Tracked paper/main.pdf missing")
        pre_tel = h.analyze_pdf_file(prebuilt_pdf)
        fresh_tel = h.analyze_pdf_bytes(h.get_tectonic_build().pdf_bytes)

        self.assertEqual(pre_tel.page_count, fresh_tel.page_count, "Prebuilt main.pdf page count out of sync with main.tex")
        self.assertTrue(pre_tel.has_times_font and not pre_tel.has_lmroman_fallback, "Prebuilt main.pdf has LMRoman fallback")
        self.assertEqual(pre_tel.pua_chars, [])
        self.assertEqual(pre_tel.cjk_chars, [])
        self.assertEqual(pre_tel.wrapped_eq_numbers, [])
        self.assertEqual(pre_tel.table_small_spans, [])
        self.assertEqual(pre_tel.orphaned_punct, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
