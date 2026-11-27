# EICA Supplementary Tables, Archive Provenance, and Ablation Ledgers

> Companion full-table appendix to *Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening* (IEEE AI-CS 2026).  
> All numbers below are reproduced programmatically by [`reproduce_all.py`](../reproduce_all.py) (42/42 automated assertions passing) from [`data/eval_records_40.csv`](../data/eval_records_40.csv) and [`data/recruitment_archive_18m.csv`](../data/recruitment_archive_18m.csv).

---

## S1. 18-Month Enterprise Recruitment Archive Provenance (`data/recruitment_archive_18m.csv`)

Period: Jan. 2025 – Jun. 2026 at a Zhejiang Class II/III medical device manufacturer (NMPA GMP, ISO 13485:2016 / GB/T 42061-2022, China State Council Order No. 739/797).

| Position Family (CSV Value) | Requisitions | Total Resumes | Forwarded by HR | Rejected $\le 5$ Bus. Days | Onboarded | Post-Cutoff Unlogged Closure | Sampled $N$ ($N=40$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| QA/QC Engineer (`质量工程师`) | 13 | 512 | 141 | 47 | 10 | 84 | 27 |
| QMS Engineer (`体系工程师`) | 3 | 96 | 26 | 12 | 2 | 12 | 5 |
| RA Specialist (`注册专员`) | 5 | 154 | 31 | 15 | 3 | 13 | 8 |
| **Total** | **21** | **762** | **198** | **74 (37.4%)** | **15** | **109** | **40** |

Sampling criteria: complete technical histories matching archive role shares, calibrated at an exact 20:20 advance/reject balance under Track 1 unassisted manual screening (20 advances, 20 rejections). Pre-experimental blinded manager benchmark $Y$ yielded 17 advances (42.5% positivity) and 23 rejections.

---

## S2. Full Table 1 — Field Screening Performance by Position Family ($N = 40$)

| Role Family | $N$ | M0 Manual Agree | M1 EICA Field Agree | Screening Time (M0 / M1, Mean $\pm$ SD, min) | Exploratory Two-Sided Fisher $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| QA/QC Engineer | 27 | 70.4% (19/27) | 85.2% (23/27) | $4.75 \pm 0.71$ / $2.72 \pm 0.42$ | 0.327 |
| QMS Engineer* | 5 | 40.0% (2/5) | 80.0% (4/5) | $5.22 \pm 0.77$ / $2.78 \pm 0.45$ | 0.524 |
| RA Specialist* | 8 | 50.0% (4/8) | 100.0% (8/8) | $4.71 \pm 0.56$ / $2.58 \pm 0.32$ | 0.077 |
| **Total / Avg.** | **40** | **62.5% (25/40)** | **87.5% (35/40)** | **$4.80 \pm 0.69$ / $2.70 \pm 0.40$** | — |

*\*Sub-samples ($N=5, 8$) provide exploratory indicators rather than population estimates.*

---

## S3. Full Table 2 — Field Paired Comparison of Manual Baseline (M0) vs. EICA Soft-Card (M1) ($N = 40$)

| Evaluative Metric | M0: Manual Baseline | M1: EICA Field Soft-Card |
| :--- | :---: | :---: |
| Benchmark Agreement | 62.5% (25/40) | 87.5% (35/40) |
| Unescalated Subset ($n = 35$, independent of manager escalation) | 71.4% (25/35), $\kappa=0.432$ | 91.4% (32/35), $\kappa=0.819$ |
| Cohen's $\kappa$ (95% Percentile Bootstrap CI, 20,000 resamples, seed=42) | 0.250 $[-0.053, 0.545]$ | 0.746 $[0.514, 0.947]$ |
| Paired $\Delta\kappa$ (95% CI) | — | $+0.496$ ($[0.196, 0.807]$, Sig.) |
| Exact McNemar Test | — | $p = 0.00635$ (Discordant: 11 vs. 1) |
| Sensitivity / Specificity / PPV / NPV (at 42.5% positivity, 17/40) | 64.7% / 60.9% / 55.0% / 70.0% | 88.2% / 87.0% / 83.3% / 90.9% |
| Pure HR Screening Time (Mean $\pm$ SD; Median [IQR], min) | $4.80 \pm 0.69$; 4.55 [IQR 1.15] | $2.70 \pm 0.40$ (-43.8%); 2.60 [IQR 0.63] |
| Organizational Time (incl. 5 escalated manager reviews = 26.0 min) | 4.80 min (excl. informal queries) | 3.35 min (-30.2%) |
| Missed Mandatory Qualification Rate | 10.0% (4/40: R05, R24, R28, R37) | 5.0% (2/40: R13, R24) |
| False Admissions / False Rejections (FA / FR) | 9 / 6 | 3 / 2 (-66.7% / -66.7%) |
| Verbatim Evidence Traceability | — | 95.0% (38/40) |
| Extrinsic Hallucination / Escalation Rate | — | 2.5% (1/40, R31) / 12.5% (5/40) |

---

## S4. Full Table 3 — Trace-Driven Counterfactual Policy Upper-Bound Ablation ($N = 40$)

| System Configuration | Agreement | Cohen's $\kappa$ (95% CI) | McNemar $p$ (vs. M0) | F1 | FA / FR | Esc. / Org. Time | $\mathbb{E}[\mathcal{L}]_{10:1}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M0: Manual Baseline (Track 1 Field) | 62.5% (25/40) | 0.250 $[-0.053, 0.545]$ | — | 59.5% | 9 / 6 | 0.0% / 4.80 min | 2.400 |
| M1: EICA Soft-Card (Track 2 Field) | 87.5% (35/40) | 0.746 $[0.514, 0.947]$ | 0.00635 (11 vs. 1) | 85.7% | 3 / 2 | 12.5% / 3.35 min | 0.831 |
| M2: +Hard FSM Lock (Counterfactual Replay) | 92.5% (37/40) | 0.845 $[0.650, 1.000]$ | 0.00183 (13 vs. 1) | 90.9% | 1 / 2 | 12.5% / 3.35 min | 0.331 |
| **M3: Full EICA + Det. Verifier (Counterfactual Replay)** | **95.0% (38/40)** | **0.896 $[0.734, 1.000]$** | **0.00098 (14 vs. 1)** | **93.8%** | **0 / 2** | **15.0% / 3.51 min** | **0.088** (`0.0875`) |

*Note: M2/M3 are analytic counterfactual policy upper bounds (not live re-runs on raw resumes): M2 enforces hard locks on logged `llm_escalated=1` (R02, R18); M3 additionally triggers on logged audit tags `llm_traceable=0` or `llm_hallucination=1` (R18, R31) and resolves enforced escalations to recorded pre-experimental benchmark $Y$. M3 amortized org. time includes R31's 6.2 min review ($2.70 + 32.2/40 = 3.51$ min, $-27.0\%$ vs. M0). Loss weights: $\lambda_{\mathrm{FA}}=10, \lambda_{\mathrm{FR}}=1, \lambda_{\mathrm{Esc}}=0.25$.*

---

## S5. Full Table 4 — Deterministic Rule-Conformance & Fault-Injection Testbench ($N_{\mathrm{syn}} = 200$)

Measured on workstation CPU: Intel Core i5-1130G7 (4 cores / 8 threads), CPython 3.14, warm steady-state loop (`src/ablation_and_benchmark.py`, seed=42). Absolute latency scales with host CPU; all ratio metrics are hardware-invariant.

| Stratum / Invariant Dimension | $n$ | Measured Outcome |
| :--- | :---: | :---: |
| Valid Grounded Extractions | 50 | 50/50 Pass (0.0% FA) |
| Unanchored / Fabricated Spans | 30 | 30/30 Caught (100.0% Rec.) |
| Verb-Escalation (R31 Pattern) | 30 | 30/30 Caught (100.0% Rec.) |
| Missing Credential Proof (R05 Pattern) | 30 | 30/30 Caught (100.0% Rec.) |
| Tacit ($\mathrm{V}$) & Score Injection | 30 | 30/30 Stripped (0.0% Leak) |
| FSM Override / Short-Reason | 30 | 30/30 Blocked ($s_{\mathrm{block}}$) |
| **Aggregate Fault P / R / F1**$^a$ | **140** | **100.0% / 100.0% / 100.0%** |
| **RAG Pinned Compliance**$^a$ | **140** | **100.0% (0 Deprecated `ORD650-DEP`)** |
| **CPU Latency (P50 / P95 / Warm Max)**$^b$ | **200** | **4.35 / 9.76 / 11.54 ms** |

$^a$$140 = 50$ valid $+ 90$ extraction/credential fault cases (excluding 30 tacit and 30 FSM strata). $^b$Warm steady-state loop across all 200 cases (excluding cold-start regex/BM25 initialization in vignette R05).

### Component-Off Ablations & Hyperparameter Sensitivity Probes (`src/ablation_and_benchmark.py`)
- **`Unpinned_RAG`**: 100.0% (20/20 legacy queries) leak deprecated Order No. 650 (`ORD650-DEP`) vs. 0.0% under `Full_EICA`.
- **`Exact_String_Match_Only`**: Extraction recall drops to 33.3% (50.0% F1), permits 60.0% (60/100 across 90 fault `Adjust` + 10 Tier-P `Confirm` probes) override bypasses, and triggers 100.0% (20/20 punctuation variants) false alarms.
- **`Soft_FSM_Lock_Off`**: Permits 100.0% (100/100) override bypasses.
- **`Unmasked_Schema`**: Drops tacit-slot blocking to 0.0% (0/30).
- **Span Threshold Sweep ($\tau_{\mathrm{span}}$)**: $\tau_{\mathrm{span}}=0.85$ achieves 100.0% boundary P/R/F1 separating 30 noisy valid quotes ($\mathrm{Sim}_{\mathrm{span}}\in[0.87,0.93]$) from 30 entity-swapped fabrications ($\in[0.77,0.83]$); $\tau_{\mathrm{span}}=0.75$ yields 0.0% F1; $\tau_{\mathrm{span}}=0.95$ triggers 100.0% (30/30) false alarms.
- **Hybrid RAG Weight ($\alpha$)**: $\alpha \le 0.5$ yields 100.0% (140/140) pin compliance and 100.0% (15/15) top-3 role-clause hits (drops to 14/15 = 93.3% at $\alpha \ge 0.7$).
- **Minimum Override Justification ($L_{\min}$)**: $L_{\min}=15$ blocks 100.0% (15/15) of 8-char excuses with 0.0% (0/15) false blocks on concise 16–19-char justifications (vs. 0.0% blocking at $L_{\min}=5$ and 100.0% false blocks at $L_{\min}=25$).

---

## S6. Full Table 5 — Systems Engineering Paradigm Comparison

| Architectural Paradigm | Tacit / Score Mask | Pinned RAG | Verb / Cert Guard | FSM Gate | Extra GPU VRAM | Determinism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Unconstrained Prompting | No | No | No | No | 0 GiB | Stochastic |
| Standard RAG + Citations | No | Unpinned | No | No | Emb. GPU/CPU | Stochastic |
| GCD / Schema-Only (Outlines) | Syntax only | Optional | No | No | $\approx 0$ GiB | Syntax only |
| Multi-Pass / Neural NLI | No | Optional | Probabilistic | Soft alert | $+1$–$8$ GiB / $k\times$ | Probabilistic |
| **EICA (Proposed System)** | **Dual-Layer** | **Yes ($\mathcal{D}_{\mathrm{pin}}$)** | **Deterministic** | **Hard ($\mathcal{G}_{\mathrm{esc}}$)** | **0 GiB (4.35 ms)** | **100% CPU Replay** |
