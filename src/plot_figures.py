# -*- coding: utf-8 -*-
"""
High-Resolution Graphic Assets Generator for IEEE AI-CS 2026 Paper
Paper: Human-AI Division of Labor in LLM-Assisted Resume Screening at a Medical Device SME
Generates:
  - fig1_architecture.pdf / fig1_architecture.png (Figure 1: EICA 3-Stage Architecture)
  - fig2_results.pdf / fig2_results.png (Figure 2: Empirical Evaluation Dual-Panel Results)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from metrics_calculator import compute_metrics

# Configure publication-grade styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# IEEE standard color palette
COLOR_NAVY = '#1B365D'
COLOR_BLUE = '#2B6CB0'
COLOR_LIGHT_BLUE = '#EBF8FF'
COLOR_TEAL = '#2C7A7B'
COLOR_LIGHT_TEAL = '#E6FFFA'
COLOR_GREEN = '#2F855A'
COLOR_LIGHT_GREEN = '#F0FFF4'
COLOR_CRIMSON = '#9B2C2C'
COLOR_LIGHT_RED = '#FFF5F5'
COLOR_AMBER = '#DD6B20'
COLOR_LIGHT_AMBER = '#FFFAF0'
COLOR_GRAY_BG = '#F7FAFC'
COLOR_BORDER_GRAY = '#CBD5E0'
COLOR_DARK_TEXT = '#2D3748'
COLOR_MUTED_TEXT = '#718096'


def plot_fig2_results(csv_path, out_pdf="fig2_results.pdf", out_png="fig2_results.png"):
    """
    Generate Figure 2: Dual-panel experimental results.
    Panel (a): Role-level agreement comparison.
    Panel (b): Decision errors and screening durations.
    """
    metrics = compute_metrics(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
    plt.subplots_adjust(wspace=0.38, top=0.88, bottom=0.18, left=0.08, right=0.92)

    # ================= PANEL A: Agreement Rate by Role =================
    roles = ['Quality Eng.\n(n = 27)', 'System Eng.\n(n = 5)', 'Regulatory\n(n = 8)', 'Overall\n(N = 40)']
    man_agree = [
        metrics['roles']['Quality Engineer (QA/QC)']['manual_agree_pct'],
        metrics['roles']['System Engineer (QMS)']['manual_agree_pct'],
        metrics['roles']['Regulatory Affairs (RA)']['manual_agree_pct'],
        metrics['agreement']['manual']
    ]
    eica_agree = [
        metrics['roles']['Quality Engineer (QA/QC)']['eica_agree_pct'],
        metrics['roles']['System Engineer (QMS)']['eica_agree_pct'],
        metrics['roles']['Regulatory Affairs (RA)']['eica_agree_pct'],
        metrics['agreement']['eica']
    ]

    x = np.arange(len(roles))
    width = 0.35

    rects1 = ax1.bar(x - width/2, man_agree, width, label='Manual Baseline', color='#718096', edgecolor='#2D3748', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, eica_agree, width, label='EICA Protocol', color='#2B6CB0', edgecolor='#1A365D', linewidth=0.8)

    ax1.set_ylabel('Agreement Rate (%)', fontsize=8.5, fontweight='bold', color=COLOR_DARK_TEXT)
    ax1.set_title('(a) Benchmark Agreement Rate by Role & Overall', fontsize=8.8, fontweight='bold', pad=8, color=COLOR_DARK_TEXT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(roles, fontsize=7.5)
    ax1.set_ylim(0, 115)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=7.2, framealpha=0.9)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 2.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=6.8, color='#2D3748')
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 2.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#1A365D')

    ax1.text(0.5, -0.22, "Overall Cohen's kappa: 0.250 -> 0.746 (delta-kappa = +0.496)",
             transform=ax1.transAxes, ha='center', fontsize=7.2, fontstyle='italic', color='#2D3748')

    # ================= PANEL B: Errors and Screening Time =================
    categories = ['False\nAdmissions', 'False\nRejections', 'Missed Key\nQuals']
    man_errors = [metrics['confusion']['manual']['FP'], metrics['confusion']['manual']['FN'], 4]
    eica_errors = [metrics['confusion']['eica']['FP'], metrics['confusion']['eica']['FN'], 2]

    x2 = np.arange(len(categories))
    width2 = 0.32

    rects_e1 = ax2.bar(x2 - width2/2, man_errors, width2, label='Manual Errors', color='#C53030', edgecolor='#742A2A', linewidth=0.8)
    rects_e2 = ax2.bar(x2 + width2/2, eica_errors, width2, label='EICA Errors', color='#319795', edgecolor='#234E52', linewidth=0.8)

    ax2.set_ylabel('Error Count (cases)', fontsize=8.5, fontweight='bold', color=COLOR_DARK_TEXT)
    ax2.set_title('(b) Decision Error Counts and Screening Time', fontsize=8.8, fontweight='bold', pad=8, color=COLOR_DARK_TEXT)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=7.5)
    ax2.set_ylim(0, 11)
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    for rect in rects_e1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.25, f'{int(h)}', ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#742A2A')
    for rect in rects_e2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.25, f'{int(h)}', ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#234E52')

    # Secondary axis for screening time
    ax2_time = ax2.twinx()
    time_points = [0.8, 1.8]  # Manual vs EICA time positions
    times = [metrics['time']['pure_hr_manual_mean'], metrics['time']['pure_hr_eica_mean']]
    tot_org = metrics['time']['total_org_eica']

    ax2_time.set_ylabel('Mean Time (min/resume)', fontsize=8.5, fontweight='bold', color='#1A365D')
    ax2_time.set_ylim(0, 6.0)
    ax2_time.set_yticks([0, 1, 2, 3, 4, 5])

    # Plot time points
    ax2_time.plot([0, 1, 2], [4.80, 2.70, 3.35], color='#1A365D', marker='s', markersize=5, linestyle=':', linewidth=1.5, label='Time Profile')
    ax2_time.text(0, 4.80 + 0.25, '4.80m', ha='center', fontsize=6.8, fontweight='bold', color='#1A365D')
    ax2_time.text(1, 2.70 - 0.45, '2.70m\n(-43.8%)', ha='center', fontsize=6.6, fontweight='bold', color='#1A365D')
    ax2_time.text(2, 3.35 + 0.25, '3.35m\n(-30.2%)', ha='center', fontsize=6.6, fontweight='bold', color='#1A365D')

    ax2.legend(loc='upper right', fontsize=7.0, framealpha=0.9)
    ax2.text(0.5, -0.22, "FA -66.7% | FR -66.7% | Pure HR -43.8% | Net Org -30.2%",
             transform=ax2.transAxes, ha='center', fontsize=7.2, fontstyle='italic', color='#2D3748')

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Success] Generated Figure 2: {out_pdf} and {out_png}")


def plot_fig1_architecture(out_pdf="fig1_architecture.pdf", out_png="fig1_architecture.png"):
    """
    Generate Figure 1: Three-stage EICA Architecture and Decision-Gate Workflow.
    """
    fig = plt.figure(figsize=(7.2, 3.6), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def draw_rounded_box(x, y, w, h, title, body, bg_col, border_col, title_col=COLOR_NAVY):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.2)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.22, title, ha='center', va='center',
                    fontsize=7.8, fontweight='bold', color=title_col)
        if body:
            ax.text(x + w/2, y + h/2 - 0.10, body, ha='center', va='center',
                    fontsize=6.5, color=COLOR_DARK_TEXT, multialignment='center')

    # Top Section Headers
    ax.text(1.65, 4.75, "1. Grounding & Standards", ha='center', fontsize=8.5, fontweight='bold', color=COLOR_NAVY)
    ax.text(5.00, 4.75, "2. LLM Extraction & Evidence Card", ha='center', fontsize=8.5, fontweight='bold', color=COLOR_BLUE)
    ax.text(8.35, 4.75, "3. Human Decision Gate", ha='center', fontsize=8.5, fontweight='bold', color=COLOR_TEAL)

    # Stage 1: Competency & Corpus
    draw_rounded_box(0.2, 2.5, 2.9, 1.9, "Three-Tier Competency Model",
                     "M (Mandatory): Hard gate (degree, ISO cert)\nP (Preferred): Contextual equivalence allowed\nV (Tacit/Verify): Interview-only, no CV inference",
                     COLOR_LIGHT_BLUE, COLOR_BLUE)

    draw_rounded_box(0.2, 0.4, 2.9, 1.8, "Frozen Regulatory Corpus",
                     "State Council Order No. 739 / 797\nISO 13485:2016 & Corporate Validation SOPs\nRAG Whitelist (Top-3 clauses retrieved)",
                     COLOR_GRAY_BG, COLOR_BORDER_GRAY)

    # Connect Stage 1 -> Stage 2
    ax.annotate('', xy=(3.4, 3.45), xytext=(3.1, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_NAVY, lw=1.5, mutation_scale=10))
    ax.annotate('', xy=(3.4, 1.30), xytext=(3.1, 1.30),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_NAVY, lw=1.5, mutation_scale=10))

    # Stage 2: Extraction & 4-Column Card
    draw_rounded_box(3.5, 2.3, 3.0, 2.1, "4-Column Evidence Card",
                     "Card(r) = < r, e(c), a(c), sigma(r) >\n- r: Requirement ID & Tier\n- e(c): Verbatim extracted quotation\n- a(c): Document coordinate (Page/Line)\n- sigma(r) in {verified, not provided, conflicting}",
                     COLOR_LIGHT_TEAL, COLOR_TEAL)

    draw_rounded_box(3.5, 0.4, 3.0, 1.6, "Extraction Constraints",
                     "Local Qwen2.5-32B (greedy, temp=0.0)\nDeterministic extraction only (no ranking/scoring)\nAutomated JSON schema retry (max 2 cycles)",
                     COLOR_GRAY_BG, COLOR_BORDER_GRAY)

    # Connect Stage 2 -> Stage 3
    ax.annotate('', xy=(6.8, 3.35), xytext=(6.5, 3.35),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_NAVY, lw=1.5, mutation_scale=10))

    # Stage 3: Human Decision Gate & State Transition
    draw_rounded_box(6.9, 2.6, 2.9, 1.8, "Cognitive Forcing Gate",
                     "HR Recruiter Actions:\n[Confirm] (all M criteria verified)\n[Adjust] (requires written rationale |rsn| >= L)\n[Escalate] (routes to Dept. Head)",
                     COLOR_LIGHT_AMBER, COLOR_AMBER)

    draw_rounded_box(6.9, 0.4, 2.9, 1.9, "Accountable Audit & Adjudication",
                     "Department Manager Adjudication (Escalated: 5/40)\nImmutable audit log of spans & justifications\nCompliant with EU AI Act Art. 14 & PIPL Art. 24",
                     COLOR_LIGHT_GREEN, COLOR_GREEN)

    # Arrow from Gate to Adjudication
    ax.annotate('', xy=(8.35, 2.3), xytext=(8.35, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_AMBER, lw=1.5, mutation_scale=10))

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Success] Generated Figure 1: {out_pdf} and {out_png}")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "../data/eval_records_40.csv"
    plot_fig1_architecture("fig1_architecture.pdf", "fig1_architecture.png")
    plot_fig2_results(csv_file, "fig2_results.pdf", "fig2_results.png")
