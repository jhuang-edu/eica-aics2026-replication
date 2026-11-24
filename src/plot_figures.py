# -*- coding: utf-8 -*-
"""
High-Resolution Publication Graphics Generator for EICA Paper (Compact IEEE Layout)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#2D3748'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_PRIMARY_NAVY = '#1A365D'
COLOR_ACCENT_BLUE  = '#2B6CB0'
COLOR_LIGHT_BLUE   = '#EBF8FF'
COLOR_TEAL_DARK    = '#234E52'
COLOR_TEAL_MID     = '#319795'
COLOR_LIGHT_TEAL   = '#E6FFFA'
COLOR_GREEN_DARK   = '#22543D'
COLOR_LIGHT_GREEN  = '#F0FFF4'
COLOR_CRIMSON_DARK = '#742A2A'
COLOR_CRIMSON_MID  = '#C53030'
COLOR_LIGHT_RED    = '#FFF5F5'
COLOR_AMBER_DARK   = '#7B341E'
COLOR_AMBER_MID    = '#DD6B20'
COLOR_LIGHT_AMBER  = '#FFFAF0'
COLOR_GRAY_BG      = '#F7FAFC'
COLOR_BORDER_GRAY  = '#CBD5E0'
COLOR_TEXT_MAIN    = '#1A202C'
COLOR_TEXT_MUTED   = '#4A5568'


def plot_fig1_architecture(out_pdf="fig1_architecture.pdf", out_png="fig1_architecture.png"):
    fig = plt.figure(figsize=(10.5, 4.4), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    def draw_card(x, y, w, h, title, body_lines, bg_col, border_col, header_bg=None, title_col=COLOR_PRIMARY_NAVY, line_step=0.23):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.1)
        ax.add_patch(box)

        if header_bg:
            h_bar = FancyBboxPatch((x, y + h - 0.38), w, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor=header_bg, edgecolor=border_col, linewidth=0.9)
            ax.add_patch(h_bar)
            ax.text(x + w / 2.0, y + h - 0.19, title, ha='center', va='center',
                    fontsize=8.8, fontweight='bold', color=title_col)
        else:
            if title:
                ax.text(x + w / 2.0, y + h - 0.21, title, ha='center', va='center',
                        fontsize=8.6, fontweight='bold', color=title_col)

        if body_lines:
            curr_y = y + h - 0.50
            for line, is_bold, fsize, col in body_lines:
                ax.text(x + 0.12, curr_y, line, ha='left', va='top',
                        fontsize=fsize, fontweight='bold' if is_bold else 'normal', color=col)
                curr_y -= line_step

    # Column Headers
    hdr1 = FancyBboxPatch((0.10, 5.32), 3.15, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_PRIMARY_NAVY, edgecolor=COLOR_PRIMARY_NAVY)
    ax.add_patch(hdr1)
    ax.text(1.675, 5.51, "Stage 1: Grounding & Standards", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='white')

    hdr2 = FancyBboxPatch((3.675, 5.32), 3.15, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_ACCENT_BLUE, edgecolor=COLOR_ACCENT_BLUE)
    ax.add_patch(hdr2)
    ax.text(5.25, 5.51, "Stage 2: LLM Extraction & Card", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='white')

    hdr3 = FancyBboxPatch((7.25, 5.32), 3.15, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_TEAL_DARK, edgecolor=COLOR_TEAL_DARK)
    ax.add_patch(hdr3)
    ax.text(8.825, 5.51, "Stage 3: Human Decision Gate", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='white')

    # Stage 1 Top & Bottom Cards (y_top=2.45..5.18 h=2.73; y_bot=0.10..2.28 h=2.18)
    draw_card(0.10, 2.45, 3.15, 2.73, "Three-Tier Competency Model", [
        ("M (Mandatory): Hard statutory gate", True, 7.6, COLOR_PRIMARY_NAVY),
        ("  Degree, license, ISO 13485 audit cert.", False, 7.0, COLOR_TEXT_MAIN),
        ("  Unverified M -> blocks admission", False, 7.0, COLOR_CRIMSON_DARK),
        ("P (Preferred): Contextual equivalence", True, 7.6, COLOR_PRIMARY_NAVY),
        ("  Cross-industry GMP / sterile validation", False, 7.0, COLOR_TEXT_MAIN),
        ("  Allows grounded equivalence claims", False, 7.0, COLOR_TEXT_MUTED),
        ("V (Tacit/Interview): Submerged traits", True, 7.6, COLOR_PRIMARY_NAVY),
        ("  Audit composure, ethical vigilance", False, 7.0, COLOR_TEXT_MAIN),
        ("  Enforced blank (empty); no inference", True, 7.0, COLOR_CRIMSON_DARK)
    ], COLOR_LIGHT_BLUE, COLOR_ACCENT_BLUE, header_bg='#DCEEFB', line_step=0.24)

    draw_card(0.10, 0.10, 3.15, 2.18, "Regulatory Grounding & Corpus", [
        ("Candidate Dossier Input (N = 40):", True, 7.5, COLOR_TEXT_MAIN),
        ("  De-identified resumes (R01-R40)", False, 6.9, COLOR_TEXT_MUTED),
        ("Version-Pinned Statutory Whitelist:", True, 7.5, COLOR_TEXT_MAIN),
        ("  Order 739/797 QMS & ISO 13485:2016", False, 6.9, COLOR_TEXT_MUTED),
        ("  12 internal SOPs & validation rules", False, 6.9, COLOR_TEXT_MUTED),
        ("RAG Grounding Layer:", True, 7.5, COLOR_TEXT_MAIN),
        ("  Top-3 regulatory clauses retrieved", False, 6.9, COLOR_ACCENT_BLUE)
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY, line_step=0.23)

    ax.annotate('', xy=(3.62, 3.80), xytext=(3.28, 3.80),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.8, mutation_scale=12))
    ax.annotate('', xy=(3.62, 1.20), xytext=(3.28, 1.20),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.8, mutation_scale=12))

    # Stage 2 Top & Bottom Cards (y_top=2.25..5.18 h=2.93; y_bot=0.10..2.08 h=1.98)
    draw_card(3.675, 2.25, 3.15, 2.93, "4-Column Evidence Card", [
        ("Card(c, r) = < r, e(c, r), a(c, r), sigma(c, r) >", True, 7.2, COLOR_PRIMARY_NAVY),
        ("  r: Target competency requirement", False, 6.9, COLOR_TEXT_MAIN),
        ("  e(c, r): Verbatim extracted quotation", False, 6.9, COLOR_TEXT_MAIN),
        ("  a(c, r): Citation anchor (page/line span)", False, 6.9, COLOR_TEXT_MAIN),
        ("  sigma: Closed flag in Sigma (empty if V):", False, 6.9, COLOR_TEXT_MAIN),
        ("    * verified: Grounded verbatim text", False, 6.9, COLOR_GREEN_DARK),
        ("    * not provided: Missing credential", False, 6.9, COLOR_AMBER_DARK),
        ("    * conflicting: Discrepancy vs. rules", False, 6.9, COLOR_CRIMSON_DARK),
        ("Traceability: 95.0% anchored (38/40)", True, 7.4, COLOR_TEAL_DARK)
    ], COLOR_LIGHT_TEAL, COLOR_TEAL_MID, header_bg='#CEF5EE', line_step=0.25)

    draw_card(3.675, 0.10, 3.15, 1.98, "Extraction Engine & Guardrails", [
        ("Qwen2.5-32B-Instruct (AWQ 4-bit)", True, 7.4, COLOR_TEXT_MAIN),
        ("  Frozen Jan 2025; RTX 3090 (21.5 GiB)", False, 6.9, COLOR_TEXT_MUTED),
        ("  vLLM (temp=0.0, batch=1, ctx <= 8,192)", False, 6.9, COLOR_TEXT_MUTED),
        ("Strict Operational Guardrails:", True, 7.4, COLOR_CRIMSON_DARK),
        ("  Zero scoring, ranking, or verdicts", False, 6.9, COLOR_CRIMSON_DARK),
        ("  Host controls state; no auto-rejection", False, 6.9, COLOR_TEXT_MAIN)
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY, line_step=0.23)

    ax.annotate('', xy=(7.20, 3.70), xytext=(6.86, 3.70),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.8, mutation_scale=12))

    # Stage 3 Top & Bottom Cards (y_top=2.45..5.18 h=2.73; y_bot=0.10..2.12 h=2.02)
    draw_card(7.25, 2.45, 3.15, 2.73, "Cognitive Forcing Decision Gate", [
        ("Recruiter Tri-State Action Space A:", True, 7.5, COLOR_PRIMARY_NAVY),
        ("[Confirm] -> Finalize (D = 0 or 1)", True, 7.2, COLOR_GREEN_DARK),
        ("  Guard: D = 1 requires all M verified", False, 6.9, COLOR_TEXT_MUTED),
        ("[Adjust] -> Override status vector sigma'", True, 7.2, COLOR_AMBER_DARK),
        ("  Mandates justification |rsn| >= Lmin", False, 6.9, COLOR_TEXT_MUTED),
        ("[Escalate] -> Route to Dept. Manager", True, 7.2, COLOR_CRIMSON_DARK),
        ("  Triggered by conflict or equivalence", False, 6.9, COLOR_TEXT_MUTED),
        ("State: delta(s, a) -> s_term | s_esc | s_blk", True, 7.1, COLOR_PRIMARY_NAVY)
    ], COLOR_LIGHT_AMBER, COLOR_AMBER_MID, header_bg='#FEEBC8', line_step=0.26)

    ax.annotate('', xy=(8.825, 2.15), xytext=(8.825, 2.43),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_AMBER_MID, lw=1.8, mutation_scale=12))

    draw_card(7.25, 0.10, 3.15, 2.02, "Manager Adjudication & Audit Log", [
        ("Department Manager Review (12.5%):", True, 7.4, COLOR_PRIMARY_NAVY),
        ("  5 escalated (R02, R05, R18, R30, R37)", False, 6.9, COLOR_TEXT_MAIN),
        ("  3 Adopted Equiv.; 2 Rejections (Y=0)", False, 6.9, COLOR_TEXT_MAIN),
        ("Immutable Compliance Audit Trail:", True, 7.4, COLOR_GREEN_DARK),
        ("  Logged spans, reasons, timestamps", False, 6.9, COLOR_TEXT_MUTED),
        ("  Aligns with EU AI Act Art. 14 / PIPL", False, 6.9, COLOR_TEXT_MUTED)
    ], COLOR_LIGHT_GREEN, COLOR_GREEN_DARK, line_step=0.24)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_fig2_results(csv_path="../data/eval_records_40.csv", out_pdf="fig2_results.pdf", out_png="fig2_results.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.05), dpi=300)
    plt.subplots_adjust(wspace=0.30, top=0.86, bottom=0.22, left=0.06, right=0.98)

    # Panel (a): Agreement Rate by Role and Overall
    roles = ['QA/QC\n(n=27)', 'QMS*\n(n=5)', 'RA*\n(n=8)', 'Overall\n(N=40)']
    man_agree = [70.4, 40.0, 50.0, 62.5]
    eica_agree = [85.2, 80.0, 100.0, 87.5]

    x = np.arange(len(roles))
    width = 0.36

    rects1 = ax1.bar(x - width/2, man_agree, width, label='Manual Baseline',
                     color='#A0AEC0', edgecolor='#4A5568', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, eica_agree, width, label='EICA Protocol',
                     color='#2B6CB0', edgecolor='#1A365D', linewidth=0.8)

    ax1.set_ylabel('Benchmark Agreement (%)', fontsize=8.0, fontweight='bold', color=COLOR_TEXT_MAIN)
    ax1.set_title('(a) Agreement by Role & Overall', fontsize=8.6, fontweight='bold', pad=8, color=COLOR_TEXT_MAIN)
    ax1.set_xticks(x)
    ax1.set_xticklabels(roles, fontsize=7.2)
    ax1.set_ylim(0, 118)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.tick_params(axis='y', labelsize=7.2)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', fontsize=6.8, framealpha=0.94)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 2.0, f'{h:.1f}%',
                 ha='center', va='bottom', fontsize=6.4, color='#2D3748')
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 2.0, f'{h:.1f}%',
                 ha='center', va='bottom', fontsize=6.4, fontweight='bold', color='#1A365D')

    ax1.text(0.5, -0.25, "Cohen's kappa: 0.250 -> 0.746 (p = 0.00635)\n*Small sub-samples (n=5, 8) are exploratory.",
             transform=ax1.transAxes, ha='center', fontsize=6.6, fontstyle='italic', color=COLOR_TEXT_MUTED)

    # Panel (b): Decision Error Counts
    categories = ['False\nAdmissions', 'False\nRejections', 'Missed Hard\nQuals']
    man_errors = [9, 6, 4]
    eica_errors = [3, 2, 2]

    x2 = np.arange(len(categories))
    width2 = 0.34

    rects_e1 = ax2.bar(x2 - width2/2, man_errors, width2, label='Manual Baseline',
                       color='#E53E3E', edgecolor='#742A2A', linewidth=0.8)
    rects_e2 = ax2.bar(x2 + width2/2, eica_errors, width2, label='EICA Protocol',
                       color='#319795', edgecolor='#234E52', linewidth=0.8)

    ax2.set_ylabel('Case Count (out of 40)', fontsize=8.0, fontweight='bold', color=COLOR_TEXT_MAIN)
    ax2.set_title('(b) Decision & Omission Errors', fontsize=8.6, fontweight='bold', pad=8, color=COLOR_TEXT_MAIN)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=7.2)
    ax2.set_ylim(0, 11.2)
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.tick_params(axis='y', labelsize=7.2)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', fontsize=6.8, framealpha=0.94)

    for rect in rects_e1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.22, f'{int(h)}',
                 ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='#742A2A')
    for rect in rects_e2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.22, f'{int(h)}',
                 ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='#234E52')

    ax2.text(0.5, -0.25, "FA reduced by 66.7% (9 -> 3); FR by 66.7% (6 -> 2)\nMandatory omissions halved (4 -> 2).",
             transform=ax2.transAxes, ha='center', fontsize=6.6, fontstyle='italic', color=COLOR_TEXT_MUTED)

    # Panel (c): Screening Time Profile (Dedicated X-axis)
    time_cats = ['Manual\nBaseline', 'EICA\n(Pure HR)', 'EICA\n(Total Org.)']
    time_vals = [4.80, 2.70, 3.35]
    bar_cols = ['#A0AEC0', '#2B6CB0', '#319795']
    edge_cols = ['#4A5568', '#1A365D', '#234E52']

    x3 = np.arange(len(time_cats))
    rects_t = ax3.bar(x3, time_vals, width=0.46, color=bar_cols, edgecolor=edge_cols, linewidth=0.9)

    ax3.set_ylabel('Mean Time (min / resume)', fontsize=8.0, fontweight='bold', color=COLOR_TEXT_MAIN)
    ax3.set_title('(c) Screening Time Profile', fontsize=8.6, fontweight='bold', pad=8, color=COLOR_TEXT_MAIN)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(time_cats, fontsize=7.2)
    ax3.set_ylim(0, 6.2)
    ax3.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax3.tick_params(axis='y', labelsize=7.2)
    ax3.grid(axis='y', linestyle='--', alpha=0.35)

    labels_t = ['4.80 min\n(Baseline)', '2.70 min\n(-43.8%)', '3.35 min\n(-30.2% net)']
    for rect, lbl in zip(rects_t, labels_t):
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2., h + 0.16, lbl,
                 ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#1A202C')

    ax3.text(0.5, -0.25, "Total Org. includes 5 manager escalations\n(8.38 min/escalation, +0.65 min amortized).",
             transform=ax3.transAxes, ha='center', fontsize=6.6, fontstyle='italic', color=COLOR_TEXT_MUTED)

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot_fig1_architecture("fig1_architecture.pdf", "fig1_architecture.png")
    plot_fig2_results("../data/eval_records_40.csv", "fig2_results.pdf", "fig2_results.png")
