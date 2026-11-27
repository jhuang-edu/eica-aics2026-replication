# EICA De-Identified Structural Vignettes and Field Error Trace Diagnosis

> Companion case-trace appendix to *Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening* (IEEE AI-CS 2026).  
> Executable vignette replay: `run_exemplar_verification_replay()` in [`src/ablation_and_benchmark.py`](../src/ablation_and_benchmark.py).  
> Open English reference prompt template: [`prompts/eica_extraction_reference_prompt.md`](../prompts/eica_extraction_reference_prompt.md).

---

## 1. PIPL-Compliant Reconstructed English Vignettes (Tier B Software Conformance)

Because China PIPL prohibits publishing raw Chinese candidate resumes, Tier B replays Algorithm 1 (`eica_engine.py`) on three de-identified English structural vignettes modeled on field failure/equivalence archetypes. Workstation reference CPU: Intel Core i5-1130G7 (4 cores / 8 threads), CPython 3.14.

### Vignette 1 — R05 Pattern (QA/QC Engineer, Simulated Naive Credential Inflation)
- **Field Log Context (`eval_records_40.csv`):** `position_type=质量工程师`, `dept_decision=1`, `hr_manual_decision=0` (`miss_key=1`), `llm_hr_decision=1` (`traceable=1`, `hallucination=0`, `escalated=1`). In the actual field trial, R05 had **no hallucination** and was resolved via escalated *Adopted Equivalence*.
- **Tier-B Software Probe:** Tests a simulated naive-extractor payload claiming `verified` on `QA_CERT` from training-only phrasing without a statutory certificate identifier.
- **Observed Engine Output:**
  - Top-3 Pinned RAG Clauses: `ISO13485-8.2.4`, `SOP-VAL-04`, `ISO13485-7.5.6` (0 deprecated `ORD650-DEP`).
  - Verifier Flags: `VERB_ESCALATION:QA_CERT:src_v1_max_v3` + `MISSING_CREDENTIAL_PROOF:QA_CERT` (independent Rules 2 & 3).
  - Hard FSM Gate on `Confirm(1)`: Blocked into `s_block` (`ERR_UNRESOLVED_CONFLICT_LOCK`), compelling `Escalate` $\to$ `s_esc`.
  - Cold-start runtime (includes regex/BM25 first-call setup): `29.3 ms` on reference workstation.

### Vignette 2 — R37 Pattern (RA Specialist, Missing Mandatory Registration)
- **Field Log Context (`eval_records_40.csv`):** `position_type=注册专员`, `dept_decision=1`, `hr_manual_decision=0` (`miss_key=1`), `llm_hr_decision=1` (`traceable=1`, `hallucination=0`, `escalated=1`). Resolved via escalated *Adopted Equivalence*.
- **Tier-B Software Probe:** Tests unverified mandatory NMPA Class II/III initial registration (`RA_REG = not_provided`) under attempted `Confirm(1)`.
- **Observed Engine Output:**
  - Top-3 Pinned RAG Clauses: `SOP-RA-02`, `ISO13485-6.2`, `NMPA-GMP-STERILE`.
  - Verifier Flags: `[]` (no fabrication; honest `not_provided`).
  - Hard FSM Gate on `Confirm(1)`: Blocked into `s_block` (`ERR_MANDATORY_GUARD_FAILED`), compelling `Escalate` $\to$ `s_esc`.
  - Warm runtime: `2.6 ms` on reference workstation.

### Vignette 3 — R31 Pattern (QMS Engineer, Logged Field Verb-Escalation Hallucination)
- **Field Log Context (`eval_records_40.csv`):** `position_type=体系工程师`, `dept_decision=0`, `hr_manual_decision=1`, `llm_hr_decision=1`, `llm_traceable=0`, `llm_hallucination=1`, `llm_escalated=0`.
- **Failure Diagnosis:** In Track 2 (deployed soft-advisory M1 without Stage 2b verifier), the frozen LLM inflated *"participated in ISO 13485 audit preparations"* ($v_{\mathrm{src}}=1$) into *"certified ISO 13485 lead auditor"* ($v_{\mathrm{req}}=3$) with a `verified` flag. Sitting on the jagged capability frontier, HR accepted the card without escalation, causing a False Admission.
- **Tier-B Software Replay:**
  - Top-3 Pinned RAG Clauses: `ISO13485-8.2.4`, `ORD797-A35`, `ISO13485-6.2`.
  - Verifier Flags: `UNANCHORED_SPAN:QMS_CERT` + `VERB_ESCALATION:QMS_LEAD:src_v1_max_v3` ($v_{\mathrm{src}}=1 < v_{\mathrm{req}}=3$).
  - Hard FSM Gate on `Confirm(1)`: Blocked into `s_block` (`ERR_UNRESOLVED_CONFLICT_LOCK`), eliminating the residual False Admission under M3 counterfactual replay.
  - Warm runtime: `6.2 ms` on reference workstation.

---

## 2. Field Error & Equivalence Audit Notes (`data/eval_records_40.csv`)

| Resume ID | Role | Pre-Exp. $Y$ | M0 Manual | M1 Field | Audit Tags (`trace / miss / hall / esc`) | Diagnostic Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **R02** | QA/QC | `0` | `1` (FA) | `1` (FA) | `1 / 0 / 0 / 1` | **Soft-Advisory Override:** System triggered escalation, HR logged preliminary advance under soft UI; subsequent escalation review rejected ($Y=0$). Blocked by M2 Hard FSM Lock. |
| **R18** | QA/QC | `0` | `1` (FA) | `1` (FA) | `0 / 0 / 0 / 1` | **Soft-Advisory Override + Untraceable:** Triggered both untraceability and escalation; HR logged advance under soft UI; subsequent escalation review rejected ($Y=0$). Blocked by M2/M3. |
| **R31** | QMS | `0` | `1` (FA) | `1` (FA) | `0 / 0 / 1 / 0` | **Verb-Escalation Hallucination:** Sole model hallucination (`2.5%`). Missed by soft M1; intercepted by M3 deterministic verifier (`FA: 1 -> 0`). |
| **R05, R30, R37** | QA/QC, QMS, RA | `1` | `0, 0, 0` | `1, 1, 1` | `1 / 0 / 0 / 1` | **Adopted Equivalence:** Legitimate cross-standard equivalence (e.g., ISO 9001 audit leadership on R30) resolved via non-penalized escalation $s_{\mathrm{esc}}$. |
| **R28** | QMS | `1` | `0` (FR) | `1` (TP) | `1 / 0 / 0 / 0` | **Clean M1 Recovery:** Manual screening missed mandatory QMS scope (`miss_key=1`); M1 surfaced verbatim evidence and aligned with $Y=1$ without escalation. |
| **R24** | QA/QC | `1` | `0` (FR) | `0` (FR) | `1 / 1 / 0 / 0` | **Shared False Rejection:** Both tracks missed a mandatory qualification (`miss_key=1`) and rejected without escalation or hallucination. |
| **R13** | QA/QC | `1` | `1` (TP) | `0` (FR) | `1 / 1 / 0 / 0` | **Sole M1 Degradation:** Only discordant pair moving from M0 alignment to M1 false rejection (`11 vs. 1`, exact McNemar $p=0.00635$). |
