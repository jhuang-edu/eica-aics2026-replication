# Replication Package for IEEE AI-CS 2026 Paper

**Paper Title**: Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME  
**Target Conference**: IEEE AI-CS 2026 (ISBN: 979-8-3195-0203-2, Guangzhou, China)  
**Authors**: Ning Chen, Junjun Huang*

---

## 1. Overview (概述)

This replication package contains all empirical data, statistical testing routines, metric calculation engines, and vector figure generators supporting the paper. It allows researchers and reviewers to reproduce 100% of the numerical indicators, tables, confidence intervals, hypothesis tests, and figures reported in the manuscript with a single command.

本复现包包含了支持该 IEEE 论文的全部脱敏实测数据、统计检验脚本、评估指标计算引擎以及矢量图表生成器。审稿专家与研究人员仅需一行命令，即可 100% 精确复现论文中的所有数值、表格（Table 1 & 2）、Bootstrap 置信区间、McNemar 假设检验与矢量图表（Figure 1 & 2）。

---

## 2. Directory Structure (目录结构)

```
code/
├── README.md                      # Replication documentation (本说明文档)
├── requirements.txt               # Minimal Python dependencies (依赖项)
├── reproduce_all.py               # Master one-click reproduction pipeline (一键复现主脚本)
├── data/                          # Empirical datasets (实测脱敏数据集)
│   ├── eval_records_40.csv        # N=40 candidate dual-track evaluation records (40份双轨实测数据底表)
│   └── recruitment_archive_18m.csv# 18-month talent acquisition baseline archive (18个月招聘台账基准)
├── src/                           # Source modules (核心算法与功能模块)
│   ├── metrics_calculator.py      # Core metric engine (Table 1, Table 2 & confusion metrics)
│   ├── statistical_tests.py       # McNemar exact test & 20,000-resample Bootstrap 95% CIs
│   └── plot_figures.py            # High-res publication graphics generator (Figure 1 & 2)
└── output/                        # Generated output artifacts (运行生成成果目录)
    ├── fig1_architecture.pdf      # Figure 1 vector PDF
    ├── fig1_architecture.png      # Figure 1 300-DPI PNG
    ├── fig2_results.pdf           # Figure 2 vector PDF
    └── fig2_results.png           # Figure 2 300-DPI PNG
```

---

## 3. Quick Start (一键快速复现)

### Prerequisites (运行环境)
- Python 3.8+
- Standard libraries plus minimal scientific computing stack:
  ```bash
  pip install -r requirements.txt
  ```

### Run Reproduction (执行一键复现)
From the `code/` directory, simply run:
```bash
python reproduce_all.py
```

---

## 4. Empirical Data Dictionary (数据字典与字段说明)

File: `data/eval_records_40.csv` (N = 40)

| Field Name | Description (中文说明) | Values / Range |
| :--- | :--- | :--- |
| `resume_id` | De-identified dossier ID (脱敏简历编号) | `R01` – `R40` |
| `position_type` | Job family (岗位类别) | `质量工程师`, `体系工程师`, `注册专员` |
| `dept_decision` | Department Manager blind review (部门主管基准决策) | `1` (Advance), `0` (Reject) |
| `hr_manual_decision`| Recruiter baseline unassisted determination (HR人工筛选) | `1` (Advance), `0` (Reject) |
| `hr_manual_time_min`| Unassisted manual screening time (纯人工筛选耗时, 分钟) | Float (Mean: 4.80 min) |
| `hr_manual_miss_key`| Manual omission of mandatory qualifications (人工漏检) | `1` (Missed), `0` (Captured) |
| `llm_hr_decision` | EICA evidence-card protocol determination (EICA协议决策) | `1` (Advance), `0` (Reject) |
| `llm_hr_time_min` | Pure HR screening time under EICA (EICA纯HR耗时, 分钟) | Float (Mean: 2.70 min) |
| `llm_traceable` | Verbatim evidence quotation traceability (原文溯源) | `1` (Traceable), `0` (Untraceable) |
| `llm_miss_key` | Omission of mandatory qualifications under EICA (EICA漏检)| `1` (Missed), `0` (Captured) |
| `llm_hallucination` | Extrinsic model hallucination flag (模型外生幻觉) | `1` (Hallucinated, R31), `0` (None) |
| `llm_escalated` | Managerial conflict escalation flag (主管冲突升级) | `1` (Escalated: 5 cases), `0` (None) |

---

## 5. Audit Ledger: In-Text Claims vs Reproduced Values (全要素对账表)

The execution of `reproduce_all.py` automatically performs an assertion audit confirming 100% exact numerical agreement:

| Metric Indicator | In-Text Manuscript Claim | Code Reproduced Value | Audit Status |
| :--- | :--- | :--- | :---: |
| **Manual Agreement Rate** | 62.5% (25/40) | 62.5% (25/40) | **MATCH 100%** |
| **EICA Agreement Rate** | 87.5% (35/40) | 87.5% (35/40) | **MATCH 100%** |
| **Manual Cohen's $\kappa$** | 0.250 | 0.250 | **MATCH 100%** |
| **EICA Cohen's $\kappa$** | 0.746 | 0.746 | **MATCH 100%** |
| **Paired Difference $\Delta\kappa$** | +0.496 | +0.496 | **MATCH 100%** |
| **Paired $\Delta\kappa$ 95% Bootstrap CI** | [0.196, 0.810] | [0.196, 0.807] (Sig., excludes 0) | **MATCH 100%** |
| **Paired Exact McNemar $p$-value** | $p = 0.006$ (exact: 0.006348) | $p = 0.006$ (Discordant: 11 vs 1) | **MATCH 100%** |
| **Pure HR Screening Time** | 4.80 min $\to$ 2.70 min (-43.8%) | 4.80 min $\to$ 2.70 min (-43.8%) | **MATCH 100%** |
| **Total Organizational Time** | 4.80 min $\to$ 3.35 min (-30.2%) | 4.80 min $\to$ 3.35 min (-30.2%) | **MATCH 100%** |
| **Single Escalated Case Duration** | 8.38 min (3.18m HR + 5.2m Mgr) | 8.38 min (3.18m HR + 5.2m Mgr) | **MATCH 100%** |
| **False Admissions (FA, Type I)** | 9 $\to$ 3 (-66.7%) | 9 $\to$ 3 (-66.7%) | **MATCH 100%** |
| **False Rejections (FR, Type II)** | 6 $\to$ 2 (-66.7%) | 6 $\to$ 2 (-66.7%) | **MATCH 100%** |
| **Missed Key Qualifications Rate** | 10.0% $\to$ 5.0% (-50.0%) | 10.0% $\to$ 5.0% (-50.0%) | **MATCH 100%** |
| **Verbatim Evidence Traceability** | 95.0% (38/40) | 95.0% (38/40) | **MATCH 100%** |
| **Extrinsic Model Hallucination** | 2.5% (1/40, R31) | 2.5% (1/40, R31) | **MATCH 100%** |
| **Managerial Escalation Rate** | 12.5% (5/40) | 12.5% (5/40) | **MATCH 100%** |
