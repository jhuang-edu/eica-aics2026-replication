<div align="center">

# Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening
### Replication Package, Executable EICA Engine & Empirical Benchmark Suite

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](#)
[![Reproducibility](https://img.shields.io/badge/Core%20Suite-42%2F42%20%7C%2065%20Unit%20Tests-brightgreen.svg)](#)
[![Open Science](https://img.shields.io/badge/Open%20Science-Replication%20Package-purple.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

**[English](#english)** | **[中文说明](#chinese)** | **[日本語](#japanese)**

</div>

---

<a id="english"></a>
## English

### 1. Research Overview & Background

Recruitment and talent acquisition in high-compliance industries (e.g., medical devices, healthcare, pharmaceuticals, aerospace) present strict technical and regulatory hurdles. Small and Medium Enterprises (SMEs) face acute resource constraints: human recruiters often struggle with dense technical dossiers, while fully autonomous AI screening risks catastrophic compliance omissions and ungrounded hallucinations.

This open-science benchmark suite supports our IEEE AI-CS 2026 paper titled **"Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening"** (Official Website: [https://www.icai-cs.com/](https://www.icai-cs.com/) | Portal: [https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)). 

The research introduces the **EICA** system architecture, integrating four algorithmic modules implemented in `src/eica_engine.py` and evaluated in `src/ablation_and_benchmark.py`:
1. **Dual-Layer Ontology-Constrained Schema Masking**: Three-tier competency ontology ($\tau(r)\in\{\mathrm{M},\mathrm{P},\mathrm{V}\}$) compiled into JSON Schema (`additionalProperties: false`, omitting tacit $\mathrm{V}$ slots and scalar scores) paired with host-side `enforce_mask_and_validate()` stripping and backfill.
2. **Version-Pinned Sparse Hybrid Regulatory RAG**: Metadata-filtered normalized Okapi BM25 + sparse token/3-gram TF-IDF cosine retrieval on host CPU (zero extra GPU VRAM) over **12 active pinned clauses** in $\mathcal{D}_{\mathrm{pin}}$ (`ORD797-A35`, `ORD797-A86`, `ISO13485-6.2/7.5.6/8.2.4/8.5.2`, `SOP-EQ-01/RA-02/CERT-03/VAL-04`, `NMPA-GMP-STERILE/IMPLANT`), excluding deprecated `ORD650-DEP` (Order No. 650).
3. **Zero-VRAM Deterministic Span-Anchor & Verb-Conflict Verifier (Algorithm 1)**: Zero-stochasticity post-extraction CPU verifier checking contiguous verbatim span alignment ($\tau_{\mathrm{span}}=0.85$, `UNANCHORED_SPAN`), clause-scoped monotonic verb-strength escalation & negation (`VERB_ESCALATION`), and independent full-span statutory credential proof (`MISSING_CREDENTIAL_PROOF`).
4. **Hard Finite-State Decision Gate**: Auditable human-in-the-loop state machine with a non-overridable conflict-escalation lock ($\mathcal{G}_{\mathrm{esc}}$) and mandatory guard ($\mathcal{G}_{\mathrm{M}}$) across pre- and post-adjustment transitions ($L_{\min}=15$).

This repository provides the complete de-identified fixed-order sequential paired field dataset ($N=40$ real-world candidates, single-blind department-manager benchmark), 18-month talent acquisition baseline archives, trace-driven counterfactual policy upper-bound replay (M0–M3), PIPL-compliant reconstructed vignettes (R05/R37/R31), an open English reference prompt template (`prompts/eica_extraction_reference_prompt.md`) with OpenAI-compatible vLLM adapter (`VLLMExtractor`), $N_{\mathrm{syn}}=200$ deterministic unit-level rule-conformance fault-injection testbench + component-off ablations + parameter sensitivity sweeps, statistical routines (Bootstrap 95% CIs, exact McNemar tests, Cohen's $\kappa$), **42 automated audit checks**, **65 unit tests**, dual-mode **end-to-end (E2E) verification checks** (standalone code/data subset on public clones; full 141-check suite including LaTeX/PDF/DOCX audits in the complete author manuscript workspace), and dynamic figure generators. All deterministic metrics, figures, and rate-based benchmarks reproduce identically from `reproduce_all.py` (absolute CPU latency in milliseconds scales with host hardware; manuscript figures report an Intel i5-1130G7 4-core / 8-thread workstation).

---

### 2. Directory Structure

```text
.
├── README.md                         # Multilingual documentation (EN/CN/JP) and data dictionary
├── requirements.txt                  # Minimal scientific computing dependencies (numpy, matplotlib, scipy)
├── vllm_config.yaml                  # On-premise RTX 3090 vLLM deployment configuration
├── reproduce_all.py                  # One-click reproduction & 42-check assertion audit pipeline
├── data/                             # Empirical de-identified datasets
│   ├── eval_records_40.csv           # N=40 paired dual-track screening trial records
│   └── recruitment_archive_18m.csv   # 18-month SME talent acquisition baseline archive
├── docs/                             # Formal proofs, full supplementary tables (Tables 1–5), and vignette/trace appendices
│   ├── proofs.md                     # Complete proofs of Propositions 1–2, Eq. (5) advisory routing, and Big-O derivation
│   ├── supplementary_tables.md       # Full Tables 1–5, 18-month archive provenance, and ablation/sensitivity ledgers
│   └── vignettes.md                  # PIPL-compliant R05/R37/R31 vignettes and field error-trace diagnoses
├── prompts/                          # Open English reference prompt template for Algorithm 1 LLMExtract
│   └── eica_extraction_reference_prompt.md
├── src/                              # Core EICA engine, ablation, stats & graphics modules
│   ├── eica_engine.py                # OntologySchemaMasker, HybridRegulatoryRAG, DeterministicVerifier, FSMDecisionGate, VLLMExtractor
│   ├── ablation_and_benchmark.py     # Trace-driven M0–M3 ablation, N_syn=200 stress-test, component ablations & sensitivity
│   ├── metrics_calculator.py         # Precision, recall, kappa, F1, and efficiency calculator
│   ├── statistical_tests.py          # McNemar exact tests & 20,000-resample Bootstrap CIs
│   └── plot_figures.py               # Data-driven dynamic vector/raster figure generator (>=8pt, zero PUA)
├── tests/                            # Automated verification suites
│   ├── test_eica_unit.py             # 65 stdlib unittest cases
│   └── e2e_suite/                    # Dual-mode E2E suite (standalone code/data checks; full 141 checks in complete manuscript workspace)
└── output/                           # Publication figures & audit outputs
    ├── fig1_architecture.pdf         # Figure 1 in manuscript: EICA System Architecture (Vector)
    ├── fig1_architecture.png         # Figure 1 in manuscript: 300-DPI Raster
    ├── fig2_results.pdf              # Figure 3 in manuscript: Empirical & Ablation Results (Vector; Fig. 2 = Alg. 1)
    └── fig2_results.png              # Figure 3 in manuscript: 300-DPI Raster
```

---

### 3. Quick Start

#### Requirements
- Python 3.8 or higher
- Install dependencies (`numpy>=1.20.0`, `matplotlib>=3.4.0`, `scipy>=1.7.0`):
  ```bash
  pip install -r requirements.txt
  ```

#### One-Click Execution & Test Suites
Run the automated reproduction, unit, and E2E verification pipelines:
```bash
# 1. Run 42 automated audit checks (42/42 MATCH 100%) and regenerate figures
python reproduce_all.py

# 2. Run 65 unit tests (65/65 OK)
python -m unittest tests.test_eica_unit -v

# 3. Run E2E verification suite (standalone code/data subset on public clone; full 141/141 in complete manuscript workspace)
python tests/e2e_suite/run_e2e_suite.py --tier all --verbose
```
Upon executing `python reproduce_all.py`, the script verifies all datasets, executes trace-driven M0–M3 ablations, $N_{\mathrm{syn}}=200$ stress-tests, component-off ablations, and sensitivity sweeps, regenerates `fig1_architecture` (Figure 1) and `fig2_results` (Figure 3, alongside Figure 2 = Algorithm 1 in the paper), and validates **42 automated audit checks (42/42 MATCH 100%)**. Note: in standalone clones of this replication repository (where manuscript LaTeX sources `paper/` and `Templates/` are not bundled), `run_e2e_suite.py` automatically runs all code, CSV, engine, and ablation E2E checks using cross-platform LF-normalized SHA-256 verification and cleanly skips manuscript/Tectonic-only checks with 0 failures and 0 errors.

---

### 4. Data Dictionary (`data/eval_records_40.csv`)

| Field Name | Description | Values / Range |
| :--- | :--- | :--- |
| `resume_id` | De-identified candidate dossier identifier | `R01` – `R40` |
| `position_type` | Job family category in medical device SME (raw Chinese CSV values) | `质量工程师` (QA/QC Engineer), `体系工程师` (QMS Engineer), `注册专员` (RA Specialist) |
| `dept_decision` | Department Manager pre-experimental blinded operational benchmark | `1` (Advance), `0` (Reject) |
| `hr_manual_decision`| Recruiter baseline unassisted determination | `1` (Advance), `0` (Reject) |
| `hr_manual_time_min`| Unassisted manual screening duration (minutes) | Float (Mean: 4.80 min) |
| `hr_manual_miss_key`| Manual omission of mandatory qualifications | `1` (Omitted), `0` (Captured) |
| `llm_hr_decision` | EICA protocol human-AI collaborative determination | `1` (Advance), `0` (Reject) |
| `llm_hr_time_min` | Human recruiter screening duration under EICA (minutes) | Float (Mean: 2.70 min) |
| `llm_traceable` | Verbatim text evidence quotation traceability | `1` (Fully Traceable), `0` (Untraceable) |
| `llm_miss_key` | Omission of mandatory qualifications under EICA | `1` (Omitted), `0` (Captured) |
| `llm_hallucination` | Extrinsic LLM hallucination flag | `1` (Hallucinated, R31), `0` (None) |
| `llm_escalated` | Managerial conflict escalation triggered | `1` (Escalated: 5 cases), `0` (None) |

---

### 5. LLM Extraction Layer, VRAM Deployment & GenAI Compliance

The information extraction stage of the EICA pipeline was deployed in production on on-premise hardware:
- **Base Model**: `Qwen/Qwen2.5-32B-Instruct` (AWQ 4-bit quantization, static weight footprint $\approx$ 19.3 GiB; exact Jan. 2025 enterprise local checkpoint SHA was not archived in production logs).
- **Target Hardware**: Single `NVIDIA GeForce RTX 3090 GPU (24 GiB VRAM, Ampere architecture)`.
- **Inference Engine & Reference Prompt**: `vLLM` server (`--max-model-len 8192`, `--gpu-memory-utilization 0.92`, temperature = 0.0, seed 42, max output 2,048 tokens). An open English reference prompt template (`prompts/eica_extraction_reference_prompt.md`) and OpenAI-compatible client adapter (`VLLMExtractor` in `src/eica_engine.py`) reproduce the exact Algorithm 1 schema/RAG constraints (the partner firm's original internal Chinese prompt is withheld under SOP/PIPL confidentiality).
- **Context & VRAM Allocation Budget**:
  - Workflow mode: Single-request execution (batch size = 1).
  - Context window ceiling: Capped at 8,192 tokens (empirical mean context $\approx$ 2,200 tokens, historical maximum $\le$ 4,800 tokens).
  - Dynamic KV Cache: Allocated at $\approx$ 2.0 GiB.
  - Total Runtime VRAM: 19.3 GiB (weights) + 2.0 GiB (KV cache) + 0.2 GiB (runtime/activation overhead) $\approx$ 21.5 GiB $\le$ 24 GiB (safely within single 24 GiB GPU capacity).
- **Longitudinal Design Control**: Model weights were frozen upon system commissioning in January 2025. To guarantee controlled longitudinal comparability across the 18-month enterprise compliance archive (Jan 2025 -- Jun 2026), the system was deliberately maintained without in-flight upgrades to subsequent releases (Qwen3 in April 2025, Qwen3.5 in February 2026). Stages 1, 2b, and 3 execute deterministically on CPU; Stage 2 greedy GPU decoding (`temperature=0.0, seed=42`) had no multi-run stability sweep.
- **Future Pluggable Benchmark & GenAI Transparency Disclosure**: Future work (Section VI & Table 5) will release a public synthetic compliance benchmark comparing Vanilla LLM, Standard RAG, GCD-only, NLI-verified, and EICA pipelines across model families such as `Qwen3-32B`, while all empirical metrics in this study remain strictly benchmarked on frozen `Qwen2.5-32B-Instruct`. Per IEEE and AI-CS 2026 generative-AI transparency guidelines, `Qwen2.5-32B-Instruct` is the evaluated Tier-A extraction component whereas $N_{\mathrm{syn}}=200$ and component-off ablations are deterministic programmatic benchmarks (zero GenAI-fabricated empirical data); generative AI assistants (Claude/GPT-class models) were used solely for English/multilingual documentation & copy-editing and LaTeX/software debugging, with all empirical logs and benchmarks verified solely by the authors.

---

### 6. Empirical Verification, Ablation & Stress-Test Audit Ledger

| Metric Indicator | In-Text Manuscript Finding | Code Reproduced Value | Status |
| :--- | :--- | :--- | :---: |
| **Manual Agreement Rate (M0)** | 62.5% (25/40) | 62.5% (25/40) | **MATCH 100%** |
| **EICA Field Agreement Rate (M1)** | 87.5% (35/40) | 87.5% (35/40) | **MATCH 100%** |
| **Manual / M1 Cohen's $\kappa$** | 0.250 / 0.746 | 0.250 / 0.746 | **MATCH 100%** |
| **Paired $\Delta\kappa$ & 95% Bootstrap CI** | +0.496, [0.196, 0.807] | +0.496, [0.196, 0.807] (Sig., excludes 0) | **MATCH 100%** |
| **Paired Exact McNemar $p$-value (M1)** | $p = 0.00635$ | $p = 0.00635$ (Discordant: 11 vs 1) | **MATCH 100%** |
| **Pure HR Screening Time (M0 $\to$ M1)** | 4.80 min $\to$ 2.70 min (-43.8%) | 4.80 min $\to$ 2.70 min (-43.8%) | **MATCH 100%** |
| **Total Org. Time (M1, incl. 5 Esc.)** | 4.80 min $\to$ 3.35 min (-30.2%) | 4.80 min $\to$ 3.35 min (-30.2%) | **MATCH 100%** |
| **False Admissions / Rejections (M1)** | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | **MATCH 100%** |
| **Mandatory Miss / Traceability / Halluc.** | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | **MATCH 100%** |
| **Table 3 M2 (+Hard FSM Lock)** | 92.5% (37/40), $\kappa=0.845$, $p=0.00183$, F1 90.9%, FA/FR 1/2, 3.35 min, $\mathbb{E}[\mathcal{L}]_{10:1}=0.331$ (M0 `2.400` $\to$ M1 `0.831` [`0.8313`] $\to$ M2 `0.331`, -60.2% vs M1) | Exact Match | **MATCH 100%** |
| **Table 3 M3 (Full EICA + Det. Verifier)** | **95.0% (38/40)**, **$\kappa=0.896$**, $p=0.00098$, F1 **93.8%**, FA/FR **0/2**, Esc 15.0%, 3.51 min (-27.0%), **$\mathbb{E}[\mathcal{L}]_{10:1}=0.088$** (exact `0.0875`, -96.4% vs M0 2.400) | Exact Match | **MATCH 100%** |
| **Table 4 ($N_{\mathrm{syn}}=200$ Stress-Test)** | Fault P/R/F1 100.0%, Tacit/FSM 100.0%, Pinned RAG 100.0% (`12` active $\mathcal{D}_{\mathrm{pin}}$, `ORD650-DEP` excluded), CPU P50/P95/Max `4.35 / 9.76 / 11.54 ms` | Exact Match | **MATCH 100%** |
| **Component-Off Ablations & Sensitivity** | `Unpinned_RAG` 100.0% legacy leak; `Exact_String_Match_Only` 33.3% R / 50.0% F1 / 60.0% bypass / 100.0% minor FA; `Soft_FSM_Lock_Off` 100.0% bypass; `Unmasked_Schema` 0.0% tacit block; $\tau_{\mathrm{span}}=0.85$, $\alpha\le 0.5$, $L_{\min}=15$ verified | Exact Match | **MATCH 100%** |

---

### 7. Citation

```bibtex
@inproceedings{huang2026eica,
  title     = {Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening},
  author    = {Huang, Junjun and Chen, Ning},
  booktitle = {Proceedings of the 2026 International Academic Conference on Artificial Intelligence and Case Studies (AI-CS 2026)},
  year      = {2026},
  note      = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```

---

<a id="chinese"></a>
## 中文说明

### 1. 研究背景与核心介绍

在高合规壁垒行业（如医疗器械、生物医药、高端装备制造）中，企业招聘面临着双重严苛约束：一方面，中小企业（SME）缺乏大型跨国企业充沛的 HR 团队，普通招聘人员难以在极短时间内穿透专业技术壁垒与法规资质文件（如 GMP、ISO 13485、注册合规体系）；另一方面，若直接采用黑盒 AI 全自动初筛，极易出现**模型外生幻觉**与**假阴性误淘汰关键人才**的高昂合规代价。

本开源基准代码库对应 IEEE AI-CS 2026 学术论文 **《面向高合规人才筛选的本体约束大语言模型证据抽取与有限状态决策支持系统》**（会议官网：[https://www.icai-cs.com/](https://www.icai-cs.com/) ｜ 艾思科蓝入口：[https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)）。

本研究提出了 **EICA** 系统架构，将 `src/eica_engine.py` 中实现的四大核心算法模块集成于 `src/ablation_and_benchmark.py` 进行系统级评估：
1. **双层本体约束 Schema 硬掩码（Ontology-Constrained Schema Masking）**：将三层级胜任力本体（$\tau(r)\in\{\mathrm{M},\mathrm{P},\mathrm{V}\}$）编译为 JSON Schema（`additionalProperties: false`，彻底省略隐性 $\mathrm{V}$ 槽位及标量评分/排序字段），并辅以宿主侧 `enforce_mask_and_validate()` 清洗与补全；
2. **版本锁定混合法规检索（Version-Pinned Hybrid Regulatory RAG）**：基于元数据过滤的归一化 BM25 + Token/3-gram TF-IDF 余弦相似度检索，锁定 **12 份现行有效 $\mathcal{D}_{\mathrm{pin}}$ 法规与 SOP 条款**（含 NMPA GMP 附录），并严格排除废弃的 `ORD650-DEP`（650 号令）；
3. **确定性原文锚点与动词冲突校验器（DeterministicVerifier, Algorithm 1）**：零随机性后置校验层，独立执行逐字连续区间对齐（$\tau_{\mathrm{span}}=0.85$，`UNANCHORED_SPAN`）、从句作用域单调动词强度越级与否定校验（`VERB_ESCALATION`）以及全区间法定资质证明校验（`MISSING_CREDENTIAL_PROOF`）；
4. **硬有限状态决策闸门（Hard Finite-State Decision Gate）**：可审计的人机协同状态机，在调整前后严格执行不可绕过的冲突升级锁（$\mathcal{G}_{\mathrm{esc}}$）与强制资质守卫（$\mathcal{G}_{\mathrm{M}}$，$L_{\min}=15$）。

本仓库开源了完整的 $N=40$ 真实单盲（部门主管实验前盲评基准）序贯配对实测对照数据、18 个月企业招聘台账基准、轨迹反事实策略消融（M0–M3）、英文参考提示词模板（`prompts/eica_extraction_reference_prompt.md`）与 vLLM 调用适配器（`VLLMExtractor`）、$N_{\mathrm{syn}}=200$ 确定性故障注入压力测试、组件消融与参数敏感性扫描、统计检验算法（Bootstrap 20,000 次重采样、精确 McNemar 检验、Cohen's $\kappa$）、**42 项自动化审计核验**、**65 项单元测试**、双模式**端到端（E2E）核验**（公开克隆自动运行全部代码/数据核验；完整稿件工作区运行全量 141 项含 LaTeX/PDF/Word 核验）以及动态矢量图表生成代码。所有确定性统计、比例指标与矢量图均可通过 `python reproduce_all.py` 一键精确复现（绝对 CPU 毫秒延迟因宿主硬件而异，正文实测平台为 Intel i5-1130G7 4 核 8 线程）。

---

### 2. 目录架构

```text
.
├── README.md                         # 多语言详细说明文档（EN/CN/JP）与数据字典
├── requirements.txt                  # 运行环境依赖清单（numpy, matplotlib, scipy）
├── vllm_config.yaml                  # 本地 RTX 3090 vLLM 推理部署配置参数
├── reproduce_all.py                  # 一键端到端复现与 42 项自动化审计核验脚本
├── data/                             # 实证脱敏数据集
│   ├── eval_records_40.csv           # 40 份候选人双轨实测对照数据底表
│   └── recruitment_archive_18m.csv   # 18 个月企业级招聘台账基准
├── docs/                             # 完整形式化证明、补充表格（表 1–5）与脱敏案例/轨迹诊断附录
│   ├── proofs.md                     # 命题 1–2 完整证明、式 (5) 路由与 Big-O 复杂度推导
│   ├── supplementary_tables.md       # 表 1–5 完整明细、18 个月档案溯源与消融/敏感性台账
│   └── vignettes.md                  # 合规脱敏 R05/R37/R31 案例与现场误差诊断
├── prompts/                          # Algorithm 1 英文参考提示词模板（企业内部中文原稿因 SOP/PIPL 保密未公开）
│   └── eica_extraction_reference_prompt.md
├── src/                              # 核心算法引擎、消融基准、统计检验与绘图源码
│   ├── eica_engine.py                # 核心确定性 EICA 引擎（SchemaMasker, HybridRAG, DeterministicVerifier, FSMGate, VLLMExtractor）
│   ├── ablation_and_benchmark.py     # M0–M3 轨迹消融、N_syn=200 压力测试、组件消融与敏感性扫描
│   ├── metrics_calculator.py         # 混淆矩阵、一致性、F1 与效率指标计算引擎
│   ├── statistical_tests.py          # McNemar 精确检验与 20,000 次 Bootstrap 95% 置信区间
│   └── plot_figures.py               # 数据驱动高分辨率学术出版级矢量图生成模块（>=8pt、零 PUA）
├── tests/                            # 自动化单元与端到端测试套件
│   ├── test_eica_unit.py             # 65 项标准库单元测试
│   └── e2e_suite/                    # 双模式 E2E 验证套件（公开克隆运行代码/数据子集；完整稿件工作区运行全量 141 项）
└── output/                           # 生成的矢量图表与输出文件
    ├── fig1_architecture.pdf         # 对应论文 Figure 1：EICA 系统架构图（矢量）
    ├── fig1_architecture.png         # 对应论文 Figure 1：300-DPI 高清位图
    ├── fig2_results.pdf              # 对应论文 Figure 3：实证与消融结果图（矢量；Figure 2 为 Algorithm 1）
    └── fig2_results.png              # 对应论文 Figure 3：300-DPI 高清位图
```

---

### 3. 一键快速复现与测试套件

#### 环境准备
- Python 3.8 或更高版本
- 安装依赖库（`numpy>=1.20.0`, `matplotlib>=3.4.0`, `scipy>=1.7.0`）：
  ```bash
  pip install -r requirements.txt
  ```

#### 执行复现流水线与测试
在仓库根目录（或完整工作区 `code/` 目录）下运行：
```bash
# 1. 运行 42 项自动化审计核验（42/42）并生成 Figure 1 与 Figure 3
python reproduce_all.py

# 2. 运行 65 项单元测试（65/65 OK）
python -m unittest tests.test_eica_unit -v

# 3. 运行 E2E 验证套件（公开克隆自动跳过 LaTeX/排版项零报错通过；完整稿件工作区运行全量 141/141）
python tests/e2e_suite/run_e2e_suite.py --tier all --verbose
```
运行 `python reproduce_all.py` 后，终端将自动完成数据一致性检验、M0–M3 轨迹消融、$N_{\mathrm{syn}}=200$ 压力测试、组件消融与敏感性扫描、统计显著性检验与图表输出，并通过 **42 项自动化审计核验（42/42 MATCH 100%）**。

---

### 4. 数据字典 (`data/eval_records_40.csv`)

| 字段名称 | 业务含义 | 取值范围 / 数据特征 |
| :--- | :--- | :--- |
| `resume_id` | 候选人简历唯一脱敏标识 | `R01` 至 `R40` |
| `position_type` | 岗位类别（医疗器械企业核心专业岗） | `质量工程师`, `体系工程师`, `注册专员` |
| `dept_decision` | 用人部门主管实验前独立盲评基准（Pre-experimental blind benchmark） | `1` (通过), `0` (淘汰) |
| `hr_manual_decision`| HR 传统纯人工独立初筛结果 | `1` (通过), `0` (淘汰) |
| `hr_manual_time_min`| 纯人工初筛耗时（分钟） | 连续型浮点数（均值：4.80 分钟） |
| `hr_manual_miss_key`| 人工筛选是否漏检硬性资质 | `1` (发生漏检), `0` (未漏检) |
| `llm_hr_decision` | EICA 协议下人机协同最终决策 | `1` (通过), `0` (淘汰) |
| `llm_hr_time_min` | EICA 模式下 HR 审核耗时（分钟） | 连续型浮点数（均值：2.70 分钟） |
| `llm_traceable` | 证据卡原文逐字溯源完整性 | `1` (具备完整原文线索), `0` (无法溯源) |
| `llm_miss_key` | EICA 协同模式下是否漏检硬性资质 | `1` (发生漏检), `0` (未漏检) |
| `llm_hallucination` | 大模型外生事实幻觉标记 | `1` (出现幻觉，仅 R31 案例), `0` (无幻觉) |
| `llm_escalated` | 是否触发用人部门主管冲突升级复核 | `1` (触发升级，共 5 例), `0` (未触发) |

---

### 5. 大模型抽取层、显存基准与 GenAI 合规边界

EICA 协议的信息提取层在企业内部单卡服务器上部署运行：
- **主模型版本**：`Qwen/Qwen2.5-32B-Instruct`（AWQ 4-bit 量化，静态模型权重占用约 19.3 GiB；2025 年 1 月企业本地部署日志未归档具体权重 SHA）。
- **部署硬件平台**：单张消费级 `NVIDIA GeForce RTX 3090 GPU (24 GiB 显存, Ampere 架构)`。
- **推理运行引擎与参考提示词**：`vLLM` 高性能推理引擎（配置 `--max-model-len 8192`, `--gpu-memory-utilization 0.92`，采样参数 temperature = 0.0, seed 42, 最大输出 2,048 tokens）。仓库提供英文参考提示词模板（`prompts/eica_extraction_reference_prompt.md`）及 OpenAI 兼容调用封装（`src/eica_engine.py` 中的 `VLLMExtractor`）；合作企业内部原始中文生产提示词因 SOP 与 PIPL 合规约束未原样发布。
- **上下文与显存精细预算**：
  - 执行模式：单请求串行处理（Batch size = 1）。
  - 上下文窗口预算：单份简历最大上下文硬性限制在 8,192 tokens（实测均值约 2,200 tokens，历史峰值 $\le$ 4,800 tokens）。
  - 动态 KV Cache 显存：动态分配约 2.0 GiB。
  - 运行时总显存：静态权重 19.3 GiB + KV Cache 2.0 GiB + 激活与运行时开销 0.2 GiB $\approx$ 21.5 GiB $\le$ 24 GiB，稳定运行于单卡 24 GiB 显存阈值内。
- **纵向台账受控一致性设计**：系统于 2025 年 1 月正式上线并即刻冻结模型权重。为确保企业 18 个月全周期合规人事档案（2025.01--2026.06）纵向数据的可比性与受控一致性，期内严禁模型漂移，未随后续 Qwen3（2025 年 4 月）及 Qwen3.5（2026 年 2 月）等模型发布而擅自升级。阶段 1、2b 与 3 在 CPU 上严格确定性执行；阶段 2 贪婪解码（`temperature=0.0, seed=42`）未开展多次 GPU 重复采样稳定性扫描。
- **未来基准规划与 GenAI 透明披露**：论文第 VI 节（含 Table 5）规划在未来公开合成合规基准上，面向 `Qwen3-32B` 等模型家族开展 Vanilla LLM / Standard RAG / GCD-only / NLI-verified / EICA 对比评测；本课题所有实证基准均严格基于冻结权重的 `Qwen2.5-32B-Instruct` 产出。依据 IEEE 与 AI-CS 2026 生成式 AI 透明度规范，明确区分 Tier-A 被测抽取组件（`Qwen2.5-32B-Instruct`）与程序化确定性软件基准（$N_{\mathrm{syn}}=200$ 及组件消融），严禁使用 GenAI 伪造实测数据；生成式 AI 助手（Claude/GPT 类模型）仅用于英文/多语种文档润色与 LaTeX/软件调试，所有实测日志与基准结果均由作者独立核实。

---

### 6. 全要素对账核验表（实证配对、策略消融与压力测试）

| 评估指标 | 论文正文报告数值 | 代码复现计算数值 | 对账结果 |
| :--- | :--- | :--- | :---: |
| **纯人工决策一致率 (M0)** | 62.5% (25/40) | 62.5% (25/40) | **100% 吻合** |
| **EICA 现场决策一致率 (M1)** | 87.5% (35/40) | 87.5% (35/40) | **100% 吻合** |
| **人工 / M1 Cohen's $\kappa$** | 0.250 / 0.746 | 0.250 / 0.746 | **100% 吻合** |
| **配对 $\Delta\kappa$ 与 95% Bootstrap 区间** | +0.496, [0.196, 0.807] | +0.496, [0.196, 0.807] (显著排除 0) | **100% 吻合** |
| **配对 McNemar 精确检验 (M1)** | $p = 0.00635$ | $p = 0.00635$ (分歧对: 11 vs 1) | **100% 吻合** |
| **单份纯 HR 筛选耗时 (M0 $\to$ M1)** | 4.80 分钟 $\to$ 2.70 分钟 (-43.8%) | 4.80 分钟 $\to$ 2.70 分钟 (-43.8%) | **100% 吻合** |
| **组织总耗时 (M1 含 5 例主管升级)** | 4.80 分钟 $\to$ 3.35 分钟 (-30.2%) | 4.80 分钟 $\to$ 3.35 分钟 (-30.2%) | **100% 吻合** |
| **假阳性 / 假阴性变化 (M1)** | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | **100% 吻合** |
| **硬性条件漏检 / 原文可溯源 / 幻觉** | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | **100% 吻合** |
| **Table 3 M2 (+硬 FSM 锁回放)** | 92.5% (37/40), $\kappa=0.845$, $p=0.00183$, F1 90.9%, FA/FR 1/2, 3.35 分钟, $\mathbb{E}[\mathcal{L}]_{10:1}=0.331$（M0 `2.400` $\to$ M1 `0.831` [`0.8313`] $\to$ M2 `0.331`，较 M1 降 -60.2%） | 完全吻合 | **100% 吻合** |
| **Table 3 M3 (完整 EICA+校验器)** | **95.0% (38/40)**, **$\kappa=0.896$**, $p=0.00098$, F1 **93.8%**, FA/FR **0/2**, 升级率 15.0%, 3.51 分钟 (-27.0%), **$\mathbb{E}[\mathcal{L}]_{10:1}=0.088$** (精确值 `0.0875`, 较 M0 2.400 降 -96.4%) | 完全吻合 | **100% 吻合** |
| **Table 4 ($N_{\mathrm{syn}}=200$ 压力测试)** | 故障 P/R/F1 100.0%, 隐性/FSM 拦截 100.0%, 固定 RAG 合规 100.0% (`12` 激活 $\mathcal{D}_{\mathrm{pin}}$ 条款, 排除 `ORD650-DEP`), CPU P50/P95/Max `4.35 / 9.76 / 11.54 ms` | 完全吻合 | **100% 吻合** |
| **组件消融与参数敏感性扫描** | `Unpinned_RAG` 100.0% 废弃条款泄漏; `Exact_String_Match_Only` 33.3% R / 50.0% F1 / 60.0% 绕过 / 100.0% 误报; `Soft_FSM_Lock_Off` 100.0% 绕过; `Unmasked_Schema` 0.0% 拦截; $\tau_{\mathrm{span}}=0.85, \alpha\le 0.5, L_{\min}=15$ | 完全吻合 | **100% 吻合** |

---

### 7. 论文引用

```bibtex
@inproceedings{huang2026eica,
  title     = {Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening},
  author    = {Huang, Junjun and Chen, Ning},
  booktitle = {Proceedings of the 2026 International Academic Conference on Artificial Intelligence and Case Studies (AI-CS 2026)},
  year      = {2026},
  note      = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```

---

<a id="japanese"></a>
## 日本語

### 1. 研究概要と背景

医療機器、ヘルスケア、バイオ医薬品などの高い規制準拠（コンプライアンス）が求められる産業において、中小企業（SME）の人材採用は厳しい課題に直面しています。専任の技術人事担当者が不足しているため、複雑な専門技術用語や各種規制要件（ISO 13485、GMP、薬事認証など）を短時間で精査することは困難です。一方で、ブラックボックス化されたAIによる完全自動選考は、ハルシネーション（事実誤認）や法令上の必須資格の看過による重大なリスクを伴います。

本オープンサイエンス・リポジトリは、IEEE AI-CS 2026 論文 **「Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening」**（公式サイト：[https://www.icai-cs.com/](https://www.icai-cs.com/) ｜ 投稿ポータル：[https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)）の公式再現パッケージです。

本研究では、`src/eica_engine.py` に実装され `src/ablation_and_benchmark.py` で評価される4つのアルゴリズム・モジュールからなる **EICA** システムアーキテクチャを提案しています：
1. **二層オントロジー制約スキーマ・ハードマスキング**：3階層コンピテンシー・オントロジー（$\tau(r)\in\{\mathrm{M},\mathrm{P},\mathrm{V}\}$）を JSON Schema（`additionalProperties: false`、暗黙知 $\mathrm{V}$ スロットおよびスカラー評価値を除外）としてコンパイルし、ホスト側 `enforce_mask_and_validate()` による洗浄・補完を併用。
2. **バージョン固定ハイブリッド規制 RAG**：メタデータフィルタリング付き正規化 BM25 + Token/3-gram TF-IDF コサイン類似度により、**12件の有効な固定条項**（$\mathcal{D}_{\mathrm{pin}}$、NMPA GMP 附属書を含む）を検索し、廃止済み `ORD650-DEP`（国務院令第650号）を厳格に除外。
3. **決定論的スパン・動詞コンフリクト検証器（Algorithm 1）**：連続スパン一致（$\tau_{\mathrm{span}}=0.85$、`UNANCHORED_SPAN`）、節スコープ単調動詞強度エスカレーション・否定検査（`VERB_ESCALATION`）、および全スパン法定資格証明チェック（`MISSING_CREDENTIAL_PROOF`）を確率的揺らぎなしで独立実行。
4. **ハード有限状態意思決定ゲート（Hard FSM Gate）**：変更前後で回避不能なコンフリクト・エスカレーション・ロック（$\mathcal{G}_{\mathrm{esc}}$）および必須要件ガード（$\mathcal{G}_{\mathrm{M}}$、$L_{\min}=15$）を強制する監査可能な HITL 状態機械。

本リポジトリには、匿名化された $N=40$ 件の候補者単盲検（部門責任者による実験前ブラインド基準）逐次ペア評価データ、18ヶ月間の採用台帳ベースライン、反実仮想ポリシーアブレーション（M0–M3）、英語リファレンス・プロンプトテンプレート（`prompts/eica_extraction_reference_prompt.md`）および vLLM アダプタ（`VLLMExtractor`）、$N_{\mathrm{syn}}=200$ 決定論的フォールトインジェクション負荷試験・コンポーネント除去アブレーション・感度分析、統計検定（Bootstrap 20,000回、正確 McNemar 検定、Cohen's $\kappa$）、**42項目の自動監査チェック**、**65件の単体テスト**、デュアルモード**E2E検証チェック**（公開クローンではコード・データ項目を自動実行、論文原稿を含む完全ワークスペースでは全141項目を実行）、および動的図表生成コードが含まれています（CPU遅延の絶対値は実行環境に依存し、論文は Intel i5-1130G7 4コア/8スレッド環境で計測）。

---

### 2. ディレクトリ構成

```text
.
├── README.md                         # 多言語ドキュメント（EN/CN/JP）およびデータ辞書
├── requirements.txt                  # 必要最小限の依存ライブラリ（numpy, matplotlib, scipy）
├── vllm_config.yaml                  # オンプレミス RTX 3090 vLLM 推論デプロイ設定
├── reproduce_all.py                  # ワンクリック完全再現・42項目自動監査パイプライン
├── data/                             # 実証研究データセット（匿名化済み）
│   ├── eval_records_40.csv           # 候補者40名の評価記録データ
│   └── recruitment_archive_18m.csv   # 18ヶ月間の採用実績ベースライン
├── docs/                             # 形式的証明・補足表（Table 1–5）・匿名化事例トレース付録
│   ├── proofs.md                     # 命題 1–2 の完全証明、式 (5) ルーティング、計算量導出
│   ├── supplementary_tables.md       # Table 1–5 全明細、18ヶ月採用台帳来歴、アブレーション・感度分析台帳
│   └── vignettes.md                  # PIPL準拠 R05/R37/R31 事例と現場エラー診断
├── prompts/                          # Algorithm 1 英語リファレンス・プロンプトテンプレート
│   └── eica_extraction_reference_prompt.md
├── src/                              # コアエンジン・アブレーション・統計・可視化スクリプト
│   ├── eica_engine.py                # 決定論的 EICA エンジン（SchemaMasker, HybridRAG, DeterministicVerifier, FSMGate, VLLMExtractor）
│   ├── ablation_and_benchmark.py     # M0–M3 トレースアブレーション、N_syn=200 負荷試験、コンポーネント除去・感度分析
│   ├── metrics_calculator.py         # 一致率・Cohen's kappa・F1・効率性計算
│   ├── statistical_tests.py          # McNemar正確検定およびBootstrap 95%信頼区間
│   └── plot_figures.py               # データ駆動高解像度ベクター/ラスター図表生成（>=8pt、PUAなし）
├── tests/                            # 自動テストスイート
│   ├── test_eica_unit.py             # 65件の標準ライブラリ単体テスト
│   └── e2e_suite/                    # デュアルモードE2E検証スイート（公開クローンはコード項目、完全ワークスペースは全141項目）
└── output/                           # 生成図表・監査レポート出力先
    ├── fig1_architecture.pdf         # 論文 Figure 1：EICA システムアーキテクチャ（ベクター）
    ├── fig1_architecture.png         # 論文 Figure 1：300-DPI PNG
    ├── fig2_results.pdf              # 論文 Figure 3：実証評価・アブレーション結果（ベクター；Figure 2 = Alg. 1）
    └── fig2_results.png              # 論文 Figure 3：300-DPI PNG
```

---

### 3. クイックスタート

#### 動作環境
- Python 3.8 以上
- 依存パッケージのインストール（`numpy>=1.20.0`, `matplotlib>=3.4.0`, `scipy>=1.7.0`）：
  ```bash
  pip install -r requirements.txt
  ```

#### 再現スクリプト・テストスイートの実行
リポジトリルート（または完全ワークスペースの `code/` ディレクトリ）で以下を実行します：
```bash
# 1. 42項目の自動監査チェック（42/42）を実行し Figure 1 と Figure 3 を生成
python reproduce_all.py

# 2. 65件の単体テストを実行（65/65 OK）
python -m unittest tests.test_eica_unit -v

# 3. E2E検証スイートを実行（公開クローンではコード項目のみ自動実行しエラー0件、完全ワークスペースでは全141項目）
python tests/e2e_suite/run_e2e_suite.py --tier all --verbose
```
`python reproduce_all.py` の実行後、すべてのデータセット検証、M0–M3 ポリシーアブレーション、$N_{\mathrm{syn}}=200$ 負荷試験・コンポーネント除去・感度分析、統計的有意性検定、図表出力が自動で行われ、**42項目の自動監査チェック（42/42 MATCH 100%）** と **100% 一致** することが確認されます。

---

### 4. データ辞書 (`data/eval_records_40.csv`)

| カラム名 | 項目説明 | 設定値・範囲 |
| :--- | :--- | :--- |
| `resume_id` | 匿名化された履歴書管理番号 | `R01` ～ `R40` |
| `position_type` | 募集職種区分（CSV内の原文中国語値） | `质量工程师` (品質エンジニア), `体系工程师` (QMSエンジニア), `注册专员` (薬事申請担当) |
| `dept_decision` | 部門長による実験前ブラインド運用基準（operational benchmark） | `1` (通過), `0` (不採用) |
| `hr_manual_decision`| 人事担当者による単独判定結果 | `1` (通過), `0` (不採用) |
| `hr_manual_time_min`| 人事単独でのスクリーニング所要時間（分） | 浮動小数点数（平均：4.80分） |
| `hr_manual_miss_key`| 人事による必須要件の見落とし有無 | `1` (見落としあり), `0` (見落としなし) |
| `llm_hr_decision` | EICAプロトコルによる人機協調判定結果 | `1` (通過), `0` (不採用) |
| `llm_hr_time_min` | EICA協調時の人事作業時間（分） | 浮動小数点数（平均：2.70分） |
| `llm_traceable` | 抽出エビデンスの原文追跡可能性 | `1` (追跡可能), `0` (追跡不可) |
| `llm_miss_key` | EICA協調時の必須要件見落とし有無 | `1` (見落としあり), `0` (見落としなし) |
| `llm_hallucination` | LLMのハルシネーション（虚偽出力）フラグ | `1` (発生：R31事例のみ), `0` (なし) |
| `llm_escalated` | 部門長へのエスカレーション審議フラグ | `1` (エスカレーション発動：5件), `0` (なし) |

---

### 5. LLM抽出層、VRAM仕様および GenAI コンプライアンス

EICAプロトコルの情報抽出層は、企業内のオンプレミス環境にて稼働しています：
- **ベースモデル**：`Qwen/Qwen2.5-32B-Instruct`（AWQ 4-bit量子化、静的重みフットプリント 約19.3 GiB；2025年1月企業導入時の正確なチェックポイントSHAは運用ログに未記録）。
- **稼働ハードウェア**：単一の `NVIDIA GeForce RTX 3090 GPU (24 GiB VRAM, Ampereアーキテクチャ)`。
- **推論エンジンと参考プロンプト**：`vLLM`（`--max-model-len 8192`, `--gpu-memory-utilization 0.92`、サンプリング設定 temperature = 0.0, seed 42, 最大出力トークン 2,048）。英語リファレンス・プロンプトテンプレート（`prompts/eica_extraction_reference_prompt.md`）と `VLLMExtractor` を収録（企業内部の中国語実運用プロンプトはSOP/PIPL機密保持のため非公開）。
- **コンテキストおよびVRAMメモリ配分設計**：
  - 実行方式：単一リクエスト順次処理（Batch size = 1）。
  - コンテキスト上限：1件あたりの最大コンテキストを 8,192 トークンに制限（実績平均 約2,200トークン、過去最大値 $\le$ 4,800トークン）。
  - 動的 KV Cache 占有量：約 2.0 GiB。
  - 実行時総VRAM消費：静的重み 19.3 GiB + KV Cache 2.0 GiB + ランタイムオーバーヘッド 0.2 GiB $\approx$ 21.5 GiB $\le$ 24 GiB（24 GiB GPUの安全閾値内で安定稼働）。
- **縦断的台帳データの一貫性制御**：2025年1月の稼働開始時にモデル重みを厳格に凍結。18ヶ月間にわたる企業のコンプライアンス採用記録（2025年1月〜2026年6月）におけるデータ比較可能性を担保するため、その後にリリースされた Qwen3（2025年4月）や Qwen3.5（2026年2月）への更新を意図的に行っていません。ステージ 1・2b・3 はCPU上で厳密に決定論的実行を行い、ステージ 2 のGPU貪欲デコード（`temperature=0.0, seed=42`）については反復実行安定性検証は未実施です。
- **将来ベンチマーク計画および GenAI 透明性開示**：第 VI 節では将来の公開合成コンプライアンスベンチマークにおいて `Qwen3-32B` 等を含むモデル群の比較評価を計画していますが、本研究のすべての実証指標は凍結された `Qwen2.5-32B-Instruct` 上で計測されたものです。IEEE および AI-CS 2026 の生成AI透明性ガイドラインに基づき、Tier-A 評価対象コンポーネント（`Qwen2.5-32B-Instruct`）とプログラムによる決定論的ソフトウェア基準（$N_{\mathrm{syn}}=200$ およびコンポーネント除去実験）を明確に区別し（GenAI によるデータ捏造ゼロ）、生成AIアシスタント（Claude/GPT系モデル）は英語・多言語ドキュメントの校正および LaTeX/ソフトウェアのデバッグ補助のみに使用した旨を開示しています。

---

### 6. 監査台帳：論文記載値と再現値の照合

| 評価指標 | 論文内の記載値 | コードによる再現計算値 | 検証結果 |
| :--- | :--- | :--- | :---: |
| **手作業判定一致率 (M0)** | 62.5% (25/40) | 62.5% (25/40) | **100% 一致** |
| **EICA 現場協調一致率 (M1)** | 87.5% (35/40) | 87.5% (35/40) | **100% 一致** |
| **M0 / M1 Cohen's $\kappa$** | 0.250 / 0.746 | 0.250 / 0.746 | **100% 一致** |
| **$\Delta\kappa$ & 95% Bootstrap 信頼区間** | +0.496, [0.196, 0.807] | +0.496, [0.196, 0.807] (有意、0を含まず) | **100% 一致** |
| **正確 McNemar 検定 $p$ 値 (M1)** | $p = 0.00635$ | $p = 0.00635$ (不一致対: 11 vs 1) | **100% 一致** |
| **純人事審査所要時間 (M0 $\to$ M1)** | 4.80 分 $\to$ 2.70 分 (-43.8%) | 4.80 分 $\to$ 2.70 分 (-43.8%) | **100% 一致** |
| **組織全体の総審査時間 (M1)** | 4.80 分 $\to$ 3.35 分 (-30.2%) | 4.80 分 $\to$ 3.35 分 (-30.2%) | **100% 一致** |
| **Type I / Type II エラー (M1)** | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | 9 $\to$ 3 / 6 $\to$ 2 (-66.7%) | **100% 一致** |
| **必須見落とし / 追跡率 / 幻覚率** | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | 10.0% $\to$ 5.0% / 95.0% (38/40) / 2.5% (R31) | **100% 一致** |
| **Table 3 M2 (+Hard FSM Lock)** | 92.5% (37/40), $\kappa=0.845$, $p=0.00183$, F1 90.9%, FA/FR 1/2, 3.35分, $\mathbb{E}[\mathcal{L}]_{10:1}=0.331$（M0 `2.400` $\to$ M1 `0.831` [`0.8313`] $\to$ M2 `0.331`、M1比 -60.2%） | 完全一致 | **100% 一致** |
| **Table 3 M3 (Full EICA+Verifier)** | **95.0% (38/40)**, **$\kappa=0.896$**, $p=0.00098$, F1 **93.8%**, FA/FR **0/2**, Esc 15.0%, 3.51分 (-27.0%), **$\mathbb{E}[\mathcal{L}]_{10:1}=0.088$** (正確値 `0.0875`, M0 2.400比 -96.4%) | 完全一致 | **100% 一致** |
| **Table 4 ($N_{\mathrm{syn}}=200$ 負荷試験)** | Fault P/R/F1 100.0%, Tacit/FSM 100.0%, Pinned RAG 100.0% (`12`件有効 $\mathcal{D}_{\mathrm{pin}}$, `ORD650-DEP`除外), CPU P50/P95/Max `4.35 / 9.76 / 11.54 ms` | 完全一致 | **100% 一致** |
| **コンポーネント除去・感度分析** | `Unpinned_RAG` 100.0% 廃止条項漏洩; `Exact_String_Match_Only` 33.3% R / 50.0% F1 / 60.0% バイパス / 100.0% 誤報; `Soft_FSM_Lock_Off` 100.0% バイパス; `Unmasked_Schema` 0.0% 遮断; $\tau_{\mathrm{span}}=0.85, \alpha\le 0.5, L_{\min}=15$ | 完全一致 | **100% 一致** |

---

### 7. 引用情報

```bibtex
@inproceedings{huang2026eica,
  title     = {Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening},
  author    = {Huang, Junjun and Chen, Ning},
  booktitle = {Proceedings of the 2026 International Academic Conference on Artificial Intelligence and Case Studies (AI-CS 2026)},
  year      = {2026},
  note      = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```
