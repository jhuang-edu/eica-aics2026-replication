<div align="center">

# Human-AI Division of Labor in LLM-Assisted Resume Screening
### Replication Package & Empirical Benchmark Suite

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](#)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)](#)
[![Open Science](https://img.shields.io/badge/Open%20Science-Replication%20Package-purple.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[English](#english)** | **[中文说明](#chinese)** | **[日本語](#japanese)**

</div>

---

<a id="english"></a>
## English

### 1. Research Overview & Background

Recruitment and talent acquisition in high-compliance industries (e.g., medical devices, healthcare, pharmaceuticals, aerospace) present strict technical and regulatory hurdles. Small and Medium Enterprises (SMEs) face acute resource constraints: human recruiters often struggle with dense technical dossiers, while fully autonomous AI screening risks catastrophic compliance omissions and ungrounded hallucinations.

This open-science benchmark suite supports our empirical research paper titled **"Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME"** (currently **Under Review** for the *2026 International Academic Conference on Artificial Intelligence and Case Studies (AI-CS 2026)* — Official Website: [https://www.icai-cs.com/](https://www.icai-cs.com/) | Portal: [https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)). 

The research introduces the **EICA (Evidence Card & Interactive Conflict Escalation) Protocol**, establishing a resilient division of labor between Large Language Models (LLMs) and Human Resource professionals:
1. **Verbatim Evidence Grounding**: The LLM acts as an exhaustive evidentiary parser, extracting word-for-word citations from candidate resumes rather than generating subjective, hallucinated endorsements.
2. **Hard-Gate Compliance Verification**: Automated cross-checking against ISO 13485/GMP mandatory certification gates, eliminating accidental oversights.
3. **Managerial Conflict Escalation**: A calibrated triage mechanism that automatically escalates borderline or high-discrepancy candidate dossiers to department managers.

This repository provides the complete, de-identified dual-track evaluation dataset ($N=40$ real-world candidates), historical 18-month talent acquisition baseline archives, statistical analysis routines (Bootstrap 95% CIs, McNemar tests, Cohen's $\kappa$), and high-resolution figure generators. Everything reproduces in a single command.

---

### 2. Directory Structure

```text
.
├── README.md                      # Multilingual documentation and data dictionary
├── requirements.txt               # Minimal scientific computing dependencies
├── reproduce_all.py               # One-click reproduction and assertion audit pipeline
├── data/                          # Empirical de-identified datasets
│   ├── eval_records_40.csv        # N=40 paired dual-track screening trial records
│   └── recruitment_archive_18m.csv# 18-month SME talent acquisition baseline archive
├── src/                           # Core analytical and graphics engines
│   ├── metrics_calculator.py      # Precision, recall, kappa, and efficiency calculator
│   ├── statistical_tests.py       # McNemar exact tests & 20,000-resample Bootstrap CIs
│   └── plot_figures.py            # High-resolution vector graphic generator
└── output/                        # Publication figures & audit outputs
    ├── fig1_architecture.pdf      # Figure 1: EICA Protocol Architecture (Vector)
    ├── fig1_architecture.png      # Figure 1: 300-DPI Raster
    ├── fig2_results.pdf           # Figure 2: Empirical Performance & Cost (Vector)
    └── fig2_results.png           # Figure 2: 300-DPI Raster
```

---

### 3. Quick Start

#### Requirements
- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### One-Click Execution
Run the automated end-to-end pipeline:
```bash
python reproduce_all.py
```
Upon execution, the script verifies all datasets, calculates metrics, runs non-parametric hypothesis tests, generates vector figures, and compares all outputs against manuscript assertion thresholds with a **100% exact numerical match**.

---

### 4. Data Dictionary (`data/eval_records_40.csv`)

| Field Name | Description | Values / Range |
| :--- | :--- | :--- |
| `resume_id` | De-identified candidate dossier identifier | `R01` – `R40` |
| `position_type` | Job family category in medical device SME | `Quality Engineer`, `Systems Engineer`, `Regulatory Specialist` |
| `dept_decision` | Department Manager blind review benchmark | `1` (Advance), `0` (Reject) |
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

### 5. LLM Extraction Layer & VRAM Deployment Benchmark

The information extraction stage of the EICA pipeline was deployed in production on on-premise hardware:
- **Base Model**: `Qwen/Qwen2.5-32B-Instruct` (AWQ 4-bit quantization, static weight footprint $\approx$ 19.3 GB).
- **Target Hardware**: Single `NVIDIA GeForce RTX 3090 GPU (24GB VRAM, Ampere architecture)`.
- **Inference Engine**: `vLLM` server (`--max-model-len 8192`, `--gpu-memory-utilization 0.92`, temperature = 0.0, seed 42, max output 2,048 tokens).
- **Context & VRAM Allocation Budget**:
  - Workflow mode: Single-request execution (batch size = 1).
  - Context window ceiling: Capped at 8,192 tokens (empirical mean context $\approx$ 2,200 tokens, historical maximum $\le$ 4,800 tokens).
  - Dynamic KV Cache: Allocated at $\approx$ 2.0 GiB.
  - Total Runtime VRAM: 19.3 GB (weights) + 2.0 GiB (KV cache) + 0.2 GiB (runtime/activation overhead) $\approx$ 21.5 GiB $\le$ 24 GiB (safely within single 24GB GPU capacity).
- **Longitudinal Design Control**: Model weights were frozen upon system commissioning in January 2025. To guarantee controlled longitudinal comparability across the 18-month enterprise compliance archive (Jan 2025 -- Jun 2026), the system was deliberately maintained without in-flight upgrades to subsequent releases (Qwen3 in April 2025, Qwen3.5 in February 2026).
- **Future Pluggable Upgrades**: Post-period open-source releases (such as high-efficiency `Qwen3.8-27B` released in August 2026) provide a pluggable upgrade path to further compress inference latency, while all baseline empirical metrics in this study remain strictly benchmarked on the frozen Qwen2.5-32B-Instruct weights.

---

### 6. Empirical Verification & Audit Ledger

| Metric Indicator | In-Text Manuscript Finding | Code Reproduced Value | Status |
| :--- | :--- | :--- | :---: |
| **Manual Agreement Rate** | 62.5% (25/40) | 62.5% (25/40) | **MATCH 100%** |
| **EICA Agreement Rate** | 87.5% (35/40) | 87.5% (35/40) | **MATCH 100%** |
| **Manual Cohen's $\kappa$** | 0.250 (Slight agreement) | 0.250 | **MATCH 100%** |
| **EICA Cohen's $\kappa$** | 0.746 (Substantial agreement) | 0.746 | **MATCH 100%** |
| **Paired Difference $\Delta\kappa$** | +0.496 | +0.496 | **MATCH 100%** |
| **Paired $\Delta\kappa$ 95% Bootstrap CI** | [0.196, 0.807] | [0.196, 0.807] (Sig., excludes 0) | **MATCH 100%** |
| **Paired Exact McNemar $p$-value** | $p = 0.006$ | $p = 0.006$ (Discordant: 11 vs 1) | **MATCH 100%** |
| **HR Screening Time per Dossier** | 4.80 min $\to$ 2.70 min (-43.8%) | 4.80 min $\to$ 2.70 min (-43.8%) | **MATCH 100%** |
| **Total Organizational Time** | 4.80 min $\to$ 3.35 min (-30.2%) | 4.80 min $\to$ 3.35 min (-30.2%) | **MATCH 100%** |
| **False Admissions (Type I Error)** | 9 $\to$ 3 (-66.7%) | 9 $\to$ 3 (-66.7%) | **MATCH 100%** |
| **False Rejections (Type II Error)** | 6 $\to$ 2 (-66.7%) | 6 $\to$ 2 (-66.7%) | **MATCH 100%** |
| **Mandatory Qualification Miss Rate** | 10.0% $\to$ 5.0% (-50.0%) | 10.0% $\to$ 5.0% (-50.0%) | **MATCH 100%** |
| **Verbatim Evidence Traceability** | 95.0% (38/40) | 95.0% (38/40) | **MATCH 100%** |
| **Extrinsic Model Hallucination** | 2.5% (1/40, Case R31) | 2.5% (1/40, Case R31) | **MATCH 100%** |
| **Managerial Escalation Triage Rate** | 12.5% (5/40) | 12.5% (5/40) | **MATCH 100%** |

---

### 7. Citation

```bibtex
@article{human_ai_screening_2026,
  title   = {Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME},
  author  = {Anonymous Authors},
  journal = {Working Paper / Under Review},
  year    = {2026},
  note    = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```

---

<a id="chinese"></a>
## 中文说明

### 1. 研究背景与核心介绍

在高合规壁垒行业（如医疗器械、生物医药、高端装备制造）中，企业招聘面临着双重严苛约束：一方面，中小企业（SME）缺乏大型跨国企业充沛的 HR 团队，普通招聘人员难以在极短时间内穿透专业技术壁垒与法规资质文件（如 GMP、ISO 13485、注册合规体系）；另一方面，若直接采用黑盒 AI 全自动初筛，极易出现**模型外生幻觉**与**假阴性误淘汰关键人才**的高昂合规代价。

本开源基准代码库对应实证学术论文 **《大语言模型辅助简历初筛中的人机分工：基于医疗器械中小企业的实证研究》**（目前正在 **2026年人工智能与案例研究国际学术会议（2026 International Academic Conference on Artificial Intelligence and Case Studies, AI-CS 2026）** 审稿投稿中；会议官网：[https://www.icai-cs.com/](https://www.icai-cs.com/) ｜ 艾思科蓝入口：[https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)）。

本研究创新提出了 **EICA 证据卡与交互式冲突升级协议（Evidence Card & Interactive Conflict Escalation Protocol）**，重塑了大模型与招聘人员的分工边界：
1. **逐字证据回溯（Verbatim Grounding）**：大模型仅扮演“证据提取器与硬性条件核查器”，提取候选人简历中的逐字原文凭据，杜绝主观捏造；
2. **硬资质闸门校验（Hard-Gate Check）**：对关键法规证书、行业年限等硬性指标实施逻辑双重锁定，大幅降低误判漏检；
3. **分层分流与冲突升级（Triage & Escalation）**：对高置信度候选人实现极速分流通行，对争议、低分及模型与人工冲突案例自动升级至用人部门主管决策，兼顾筛选效率与组织审慎性。

本仓库开源了完整的 $N=40$ 真实双盲实测对照数据、18 个月企业招聘台账基准、全部统计检验算法（Bootstrap 20,000次重采样、McNemar 配对检验）及矢量图表自动生成代码，实现“一行命令全要素 100% 精确复现”。

---

### 2. 目录架构

```text
.
├── README.md                      # 多语言详细说明文档与数据字典
├── requirements.txt               # 运行环境依赖清单
├── reproduce_all.py               # 一键端到端复现与断言对账脚本
├── data/                          # 实证脱敏数据集
│   ├── eval_records_40.csv        # 40 份候选人双轨实测对照数据底表
│   └── recruitment_archive_18m.csv# 18 个月企业级招聘台账基准
├── src/                           # 核心计算与可视化源码
│   ├── metrics_calculator.py      # 混淆矩阵、一致性与效率指标计算引擎
│   ├── statistical_tests.py       # McNemar 检验与 20,000 次 Bootstrap 95% 置信区间
│   └── plot_figures.py            # 高分辨率学术出版级矢量图生成模块
└── output/                        # 生成的矢量图表与输出文件
    ├── fig1_architecture.pdf      # 图 1：EICA 协议人机分工架构图（矢量）
    ├── fig1_architecture.png      # 图 1：300-DPI 高清位图
    ├── fig2_results.pdf           # 图 2：实证性能与成本权衡图（矢量）
    └── fig2_results.png           # 图 2：300-DPI 高清位图
```

---

### 3. 一键快速复现

#### 环境准备
- Python 3.8 或更高版本
- 安装依赖库：
  ```bash
  pip install -r requirements.txt
  ```

#### 执行复现流水线
在仓库根目录下运行：
```bash
python reproduce_all.py
```
终端将自动输出全套数据一致性检验、统计显著性指标、生成高清图表，并对论文中的 23 项核心实证指标进行断言比对（对账完全一致）。

---

### 4. 数据字典 (`data/eval_records_40.csv`)

| 字段名称 | 业务含义 | 取值范围 / 数据特征 |
| :--- | :--- | :--- |
| `resume_id` | 候选人简历唯一脱敏标识 | `R01` 至 `R40` |
| `position_type` | 岗位类别（医疗器械企业核心专业岗） | `质量工程师`, `体系工程师`, `注册专员` |
| `dept_decision` | 用人部门主管双盲评审基准（Ground Truth） | `1` (通过), `0` (淘汰) |
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

### 5. 大模型抽取层与本地部署显存基准

EICA 协议的信息提取层在企业内部单卡服务器上部署运行：
- **主模型版本**：`Qwen/Qwen2.5-32B-Instruct`（AWQ 4-bit 量化，静态模型权重占用约 19.3 GB）。
- **部署硬件平台**：单张消费级 `NVIDIA GeForce RTX 3090 GPU (24GB 显存, Ampere 架构)`。
- **推理运行引擎**：`vLLM` 高性能推理引擎（配置 `--max-model-len 8192`, `--gpu-memory-utilization 0.92`，采样参数 temperature = 0.0, seed 42, 最大输出 2,048 tokens）。
- **上下文与显存精细预算**：
  - 执行模式：单请求串行处理（Batch size = 1）。
  - 上下文窗口预算：单份简历最大上下文硬性限制在 8,192 tokens（实测均值约 2,200 tokens，历史峰值 $\le$ 4,800 tokens）。
  - 动态 KV Cache 显存：动态分配约 2.0 GiB。
  - 运行时总显存：静态权重 19.3 GB + KV Cache 2.0 GiB + 激活与运行时开销 0.2 GiB $\approx$ 21.5 GiB $\le$ 24 GiB，稳定运行于单卡 24GB 显存阈值内。
- **纵向台账受控一致性设计**：系统于 2025 年 1 月正式上线并即刻冻结模型权重。为确保企业 18 个月全周期合规人事档案（2025.01--2026.06）纵向数据的可比性与受控一致性，期内严禁模型漂移，未随后续 Qwen3（2025 年 4 月）及 Qwen3.5（2026 年 2 月）等模型发布而擅自升级。
- **期后可插拔升级兼容性**：实验期后发布的开源模型（如 2026 年 8 月推出的高能效 `Qwen3.8-27B` 权重）为系统未来进一步压降推理延迟提供了即插即用（pluggable）的升级路径；本课题所有实证基准数据均严格基于冻结权重的 Qwen2.5-32B-Instruct 产出。

---

### 6. 全要素对账核验表

| 评估指标 | 论文正文报告数值 | 代码复现计算数值 | 对账结果 |
| :--- | :--- | :--- | :---: |
| **纯人工决策一致率** | 62.5% (25/40) | 62.5% (25/40) | **100% 吻合** |
| **EICA 决策一致率** | 87.5% (35/40) | 87.5% (35/40) | **100% 吻合** |
| **人工筛选 Cohen's $\kappa$** | 0.250 (低度一致) | 0.250 | **100% 吻合** |
| **EICA 协同 Cohen's $\kappa$** | 0.746 (高度一致) | 0.746 | **100% 吻合** |
| **配对 $\kappa$ 提升增量 $\Delta\kappa$** | +0.496 | +0.496 | **100% 吻合** |
| **$\Delta\kappa$ 95% Bootstrap 置信区间** | [0.196, 0.807] | [0.196, 0.807] (显著排除 0) | **100% 吻合** |
| **配对 McNemar 检验 $p$ 值** | $p = 0.006$ (精确值: 0.006348) | $p = 0.006$ (分歧对: 11 vs 1) | **100% 吻合** |
| **单份纯 HR 筛选耗时** | 4.80 分钟 $\to$ 2.70 分钟 (-43.8%) | 4.80 分钟 $\to$ 2.70 分钟 (-43.8%) | **100% 吻合** |
| **组织总耗时 (含主管升级)** | 4.80 分钟 $\to$ 3.35 分钟 (-30.2%) | 4.80 分钟 $\to$ 3.35 分钟 (-30.2%) | **100% 吻合** |
| **假阳性误通过数 (Type I 错误)** | 9 例 $\to$ 3 例 (-66.7%) | 9 例 $\to$ 3 例 (-66.7%) | **100% 吻合** |
| **假阴性误淘汰数 (Type II 错误)** | 6 例 $\to$ 2 例 (-66.7%) | 6 例 $\to$ 2 例 (-66.7%) | **100% 吻合** |
| **关键硬性条件漏检率** | 10.0% $\to$ 5.0% (-50.0%) | 10.0% $\to$ 5.0% (-50.0%) | **100% 吻合** |
| **逐字证据原文可溯源率** | 95.0% (38/40) | 95.0% (38/40) | **100% 吻合** |
| **模型外生幻觉率** | 2.5% (1/40, R31 案例) | 2.5% (1/40, R31 案例) | **100% 吻合** |
| **主管冲突升级流转率** | 12.5% (5/40) | 12.5% (5/40) | **100% 吻合** |

---

### 7. 论文引用

```bibtex
@article{human_ai_screening_2026,
  title   = {Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME},
  author  = {Anonymous Authors},
  journal = {Working Paper / Under Review},
  year    = {2026},
  note    = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```

---

<a id="japanese"></a>
## 日本語

### 1. 研究概要と背景

医療機器、ヘルスケア、バイオ医薬品などの高い規制準拠（コンプライアンス）が求められる産業において、中小企業（SME）の人材採用は厳しい課題に直面しています。専任の技術人事担当者が不足しているため、複雑な専門技術用語や各種規制要件（ISO 13485、GMP、薬事認証など）を短時間で精査することは困難です。一方で、ブラックボックス化されたAIによる完全自動選考は、ハルシネーション（事実誤認）や法令上の必須資格の看過による重大なリスクを伴います。

本オープンサイエンス・リポジトリは、学術論文 **「Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME」**（現在 **2026年 人工知能とケーススタディ国際学術会議（2026 International Academic Conference on Artificial Intelligence and Case Studies, AI-CS 2026）** へ投稿・査読中；公式サイト：[https://www.icai-cs.com/](https://www.icai-cs.com/) ｜ 投稿ポータル：[https://www.ais.cn/attendees/index/A2INZU](https://www.ais.cn/attendees/index/A2INZU)）を完全に再現するための公式検証パッケージです。

本研究では、大規模言語モデル（LLM）と人事担当者の最適な協調を実現する **EICA（エビデンスカードおよび対話型コンフリクト・エスカレーション）プロトコル** を提案しています：
1. **逐語的エビデンスの抽出（Verbatim Grounding）**：LLMは主観的評価を行わず、候補者の履歴書から必須条件に関する原文引用のみを正確に抽出します。
2. **ハードゲート（必須要件）の検証**：法令上の必須資格や職務年数を厳格に照合し、見落としを防止します。
3. **部門管理者へのエスカレーション**：AIと人間の判断が不一致の場合や境界事例について、現場の部門長へ自動的に審議を委ねるトリアージ機構を備えています。

本リポジトリには、匿名化された $N=40$ 件の候補者二重評価データ、18ヶ月間の採用台帳ベースライン、統計検定スクリプト（Bootstrap 20,000回リサンプリング、McNemar検定）、高解像度ベクトル図生成モジュールが含まれており、ワンクリックで論文の全結果を再現可能です。

---

### 2. ディレクトリ構成

```text
.
├── README.md                      # 多言語ドキュメントおよびデータ辞書
├── requirements.txt               # 必要最小限の依存ライブラリ
├── reproduce_all.py               # ワンクリック完全再現・監査パイプライン
├── data/                          # 実証研究データセット（匿名化済み）
│   ├── eval_records_40.csv        # 候補者40名の評価記録データ
│   └── recruitment_archive_18m.csv# 18ヶ月間の採用実績ベースライン
├── src/                           # コア分析・可視化スクリプト
│   ├── metrics_calculator.py      # 一致率・Cohen's kappa・効率性計算
│   ├── statistical_tests.py       # McNemar検定およびBootstrap 95%信頼区間
│   └── plot_figures.py            # 高解像度ベクターグラフィック生成
└── output/                        # 生成図表・監査レポート出力先
    ├── fig1_architecture.pdf      # 図1：EICAプロトコル構造図（ベクター）
    ├── fig1_architecture.png      # 図1：高解像度PNG
    ├── fig2_results.pdf           # 図2：実証結果・トレードオフ図（ベクター）
    └── fig2_results.png           # 図2：高解像度PNG
```

---

### 3. クイックスタート

#### 動作環境
- Python 3.8 以上
- 依存パッケージのインストール：
  ```bash
  pip install -r requirements.txt
  ```

#### 再現スクリプトの実行
リポジトリのルートディレクトリで以下を実行します：
```bash
python reproduce_all.py
```
実行後、すべてのデータセット検証、統計的有意性検定、図表出力が自動で行われ、論文内の23の主要指標と **100% 一致** することが確認されます。

---

### 4. データ辞書 (`data/eval_records_40.csv`)

| カラム名 | 項目説明 | 設定値・範囲 |
| :--- | :--- | :--- |
| `resume_id` | 匿名化された履歴書管理番号 | `R01` ～ `R40` |
| `position_type` | 募集職種区分 | `品質エンジニア`, `システムエンジニア`, `薬事申請担当` |
| `dept_decision` | 部門長によるブラインド判定基準（真値） | `1` (通過), `0` (不採用) |
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

### 5. LLM抽出層およびVRAMハードウェア仕様

EICAプロトコルの情報抽出層は、企業内のオンプレミス環境にて稼働しています：
- **ベースモデル**：`Qwen/Qwen2.5-32B-Instruct`（AWQ 4-bit量子化、静的重みフットプリント 約19.3 GB）。
- **稼働ハードウェア**：単一の `NVIDIA GeForce RTX 3090 GPU (24GB VRAM, Ampereアーキテクチャ)`。
- **推論エンジン**：`vLLM`（`--max-model-len 8192`, `--gpu-memory-utilization 0.92`、サンプリング設定 temperature = 0.0, seed 42, 最大出力トークン 2,048）。
- **コンテキストおよびVRAMメモリ配分設計**：
  - 実行方式：単一リクエスト順次処理（Batch size = 1）。
  - コンテキスト上限：1件あたりの最大コンテキストを 8,192 トークンに制限（実績平均 約2,200トークン、過去最大値 $\le$ 4,800トークン）。
  - 動的 KV Cache 占有量：約 2.0 GiB。
  - 実行時総VRAM消費：静的重み 19.3 GB + KV Cache 2.0 GiB + ランタイムオーバーヘッド 0.2 GiB $\approx$ 21.5 GiB $\le$ 24 GiB（24GB GPUの安全閾値内で安定稼働）。
  - **縦断的台帳データの一貫性制御**：2025年1月の稼働開始時にモデル重みを厳格に凍結。18ヶ月間にわたる企業のコンプライアンス採用記録（2025年1月〜2026年6月）におけるデータ比較可能性（longitudinal comparability）を担保するため、その後にリリースされた Qwen3（2025年4月）や Qwen3.5（2026年2月）への更新を意図的に行っていません。
  - **将来のプラグイン型アップグレード**：調査期間外に公開された最新オープンソースモデル（2026年8月リリースの高効率モデル `Qwen3.8-27B` など）は、業務パイプラインを変更せずに遅延を短縮するプラグイン型（pluggable）アップグレードパスを提供します。なお、本研究のすべてのベースライン評価指標は、凍結された Qwen2.5-32B-Instruct 上で計測されたものです。

---

### 6. 監査台帳：論文記載値と再現値の照合

| 評価指標 | 論文内の記載値 | コードによる再現計算値 | 検証結果 |
| :--- | :--- | :--- | :---: |
| **手作業判定一致率** | 62.5% (25/40) | 62.5% (25/40) | **100% 一致** |
| **EICA協調判定一致率** | 87.5% (35/40) | 87.5% (35/40) | **100% 一致** |
| **手作業 Cohen's $\kappa$** | 0.250 (低い一致) | 0.250 | **100% 一致** |
| **EICA協調 Cohen's $\kappa$** | 0.746 (高い一致) | 0.746 | **100% 一致** |
| **対応のある $\kappa$ 改善幅 $\Delta\kappa$** | +0.496 | +0.496 | **100% 一致** |
| **$\Delta\kappa$ 95% Bootstrap 信頼区間** | [0.196, 0.807] | [0.196, 0.807] (有意、0を含まず) | **100% 一致** |
| **McNemar 検定 $p$ 値** | $p = 0.006$ (正確値: 0.006348) | $p = 0.006$ (不一致対: 11 vs 1) | **100% 一致** |
| **人事1件あたり審査所要時間** | 4.80 分 $\to$ 2.70 分 (-43.8%) | 4.80 分 $\to$ 2.70 分 (-43.8%) | **100% 一致** |
| **組織全体の総審査時間 (審議含む)**| 4.80 分 $\to$ 3.35 分 (-30.2%) | 4.80 分 $\to$ 3.35 分 (-30.2%) | **100% 一致** |
| **不当通過件数 (Type I エラー)** | 9 件 $\to$ 3 件 (-66.7%) | 9 件 $\to$ 3 件 (-66.7%) | **100% 一致** |
| **誤判定除外件数 (Type II エラー)**| 6 件 $\to$ 2 件 (-66.7%) | 6 件 $\to$ 2 件 (-66.7%) | **100% 一致** |
| **必須条件見落とし率** | 10.0% $\to$ 5.0% (-50.0%) | 10.0% $\to$ 5.0% (-50.0%) | **100% 一致** |
| **原文エビデンス追跡率** | 95.0% (38/40) | 95.0% (38/40) | **100% 一致** |
| **モデル外因性ハルシネーション率** | 2.5% (1/40, R31事例) | 2.5% (1/40, R31事例) | **100% 一致** |
| **部門長エスカレーション率** | 12.5% (5/40) | 12.5% (5/40) | **100% 一致** |

---

### 7. 引用情報

```bibtex
@article{human_ai_screening_2026,
  title   = {Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME},
  author  = {Anonymous Authors},
  journal = {Working Paper / Under Review},
  year    = {2026},
  note    = {Replication package: https://github.com/jhuang-edu/eica-aics2026-replication}
}
```
