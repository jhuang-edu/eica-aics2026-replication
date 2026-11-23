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
    fig = plt.figure(figsize=(7.2, 3.4), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    def draw_card(x, y, w, h, title, body_lines, bg_col, border_col, header_bg=None, title_col=COLOR_PRIMARY_NAVY):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.0)
        ax.add_patch(box)

        if header_bg:
            h_bar = FancyBboxPatch((x, y + h - 0.38), w, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor=header_bg, edgecolor=border_col, linewidth=0.8)
            ax.add_patch(h_bar)
            ax.text(x + w / 2.0, y + h - 0.20, title, ha='center', va='center',
                    fontsize=7.6, fontweight='bold', color=title_col)
        else:
            if title:
                ax.text(x + w / 2.0, y + h - 0.20, title, ha='center', va='center',
                        fontsize=7.6, fontweight='bold', color=title_col)

        if body_lines:
            curr_y = y + h - 0.48
            for line, is_bold, fsize, col in body_lines:
                ax.text(x + 0.10, curr_y, line, ha='left', va='top',
                        fontsize=fsize, fontweight='bold' if is_bold else 'normal', color=col)
                curr_y -= (fsize * 0.036 + 0.07)

    # Headers
    hdr1 = FancyBboxPatch((0.15, 4.75), 3.0, 0.34, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_PRIMARY_NAVY, edgecolor=COLOR_PRIMARY_NAVY)
    ax.add_patch(hdr1)
    ax.text(1.65, 4.92, "Stage 1: Grounding & Standards", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='white')

    hdr2 = FancyBboxPatch((3.5, 4.75), 3.0, 0.34, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_ACCENT_BLUE, edgecolor=COLOR_ACCENT_BLUE)
    ax.add_patch(hdr2)
    ax.text(5.0, 4.92, "Stage 2: LLM Extraction & Card", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='white')

    hdr3 = FancyBboxPatch((6.85, 4.75), 3.0, 0.34, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=COLOR_TEAL_DARK, edgecolor=COLOR_TEAL_DARK)
    ax.add_patch(hdr3)
    ax.text(8.35, 4.92, "Stage 3: Human Decision Gate", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='white')

    # Stage 1
    draw_card(0.15, 2.50, 3.0, 2.15, "Three-Tier Competency Model", [
        ("M (Mandatory): Hard statutory gate", True, 6.8, COLOR_PRIMARY_NAVY),
        ("  Formal degree, license, ISO audit cert.", False, 6.1, COLOR_TEXT_MAIN),
        ("  Missing evidence -> Mandates Esc / Rej", False, 6.1, COLOR_CRIMSON_DARK),
        ("P (Preferred): Contextual equivalence", True, 6.8, COLOR_PRIMARY_NAVY),
        ("  Cross-industry GMP/sterile validation", False, 6.1, COLOR_TEXT_MAIN),
        ("  Allows grounded equivalence claims", False, 6.1, COLOR_TEXT_MUTED),
        ("V (Tacit/Verify): Interview-only traits", True, 6.8, COLOR_PRIMARY_NAVY),
        ("  Audit composure, ethical vigilance", False, 6.1, COLOR_TEXT_MAIN),
        ("  Enforced blank; zero CV extrapolation", True, 6.1, COLOR_CRIMSON_DARK)
    ], COLOR_LIGHT_BLUE, COLOR_ACCENT_BLUE, header_bg='#DCEEFB')

    draw_card(0.15, 0.25, 3.0, 2.15, "Regulatory Grounding & Corpus", [
        ("Candidate Dossier Input (N=40):", True, 6.8, COLOR_TEXT_MAIN),
        ("  De-identified resumes (R01-R40)", False, 6.1, COLOR_TEXT_MUTED),
        ("  Stripped PII; preserved technical text", False, 6.1, COLOR_TEXT_MUTED),
        ("Version-Pinned Statutory Whitelist:", True, 6.8, COLOR_TEXT_MAIN),
        ("  Order 739/797 QMS & ISO 13485:2016", False, 6.1, COLOR_TEXT_MUTED),
        ("  12 internal SOPs & validation rules", False, 6.1, COLOR_TEXT_MUTED),
        ("RAG Grounding Layer:", True, 6.8, COLOR_TEXT_MAIN),
        ("  Top-3 regulatory clauses retrieved", False, 6.1, COLOR_ACCENT_BLUE)
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY)

    ax.annotate('', xy=(3.45, 3.55), xytext=(3.18, 3.55),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.6, mutation_scale=11))
    ax.annotate('', xy=(3.45, 1.35), xytext=(3.18, 1.35),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.6, mutation_scale=11))

    # Stage 2
    draw_card(3.5, 2.05, 3.0, 2.60, "4-Column Evidence Card", [
        ("Card(r) = < r, e(c), a(c), sigma(r) >", True, 7.0, COLOR_PRIMARY_NAVY),
        ("- r : Target competency requirement", False, 6.1, COLOR_TEXT_MAIN),
        ("- e(c) : Verbatim extracted quotation", False, 6.1, COLOR_TEXT_MAIN),
        ("- a(c) : Citation anchor (page/line span)", False, 6.1, COLOR_TEXT_MAIN),
        ("- sigma(r) : Closed status flag in Sigma:", False, 6.1, COLOR_TEXT_MAIN),
        ("  * verified    : Grounded verbatim text", False, 6.1, COLOR_GREEN_DARK),
        ("  * not provided: Missing credential", False, 6.1, COLOR_AMBER_DARK),
        ("  * conflicting : Discrepancy vs rules", False, 6.1, COLOR_CRIMSON_DARK),
        ("Auditable Traceability:", True, 6.6, COLOR_PRIMARY_NAVY),
        ("  95.0% verbatim text anchor rate (38/40)", False, 6.1, COLOR_TEAL_DARK)
    ], COLOR_LIGHT_TEAL, COLOR_TEAL_MID, header_bg='#CEF5EE')

    draw_card(3.5, 0.25, 3.0, 1.70, "Extraction Engine & Limits", [
        ("Local Qwen2.5-32B-Instruct instance", True, 6.6, COLOR_TEXT_MAIN),
        ("  Greedy decoding (temperature = 0.0)", False, 6.1, COLOR_TEXT_MUTED),
        ("  Fixed seed; context window 32,768", False, 6.1, COLOR_TEXT_MUTED),
        ("Strict Operational Guardrails:", True, 6.6, COLOR_CRIMSON_DARK),
        ("  Zero scoring, ranking, or verdicts", False, 6.1, COLOR_CRIMSON_DARK),
        ("  Host controls state; no auto-rejection", False, 6.1, COLOR_TEXT_MAIN)
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY)

    ax.annotate('', xy=(6.8, 3.35), xytext=(6.53, 3.35),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.6, mutation_scale=11))

    # Stage 3
    draw_card(6.85, 2.25, 3.0, 2.40, "Cognitive Forcing Decision Gate", [
        ("Recruiter Tri-State Action Space A:", True, 6.8, COLOR_PRIMARY_NAVY),
        ("[Confirm] -> Advance to interview", True, 6.4, COLOR_GREEN_DARK),
        ("  Guard: requires all M criteria verified", False, 6.0, COLOR_TEXT_MUTED),
        ("[Adjust]  -> Override status flag", True, 6.4, COLOR_AMBER_DARK),
        ("  Mandates written reason |rsn| >= Lmin", False, 6.0, COLOR_TEXT_MUTED),
        ("[Escalate] -> Route to Dept. Manager", True, 6.4, COLOR_CRIMSON_DARK),
        ("  Triggered by conflicting or ambiguity", False, 6.0, COLOR_TEXT_MUTED),
        ("State Transition: delta(s, a) -> s_term", True, 6.4, COLOR_PRIMARY_NAVY)
    ], COLOR_LIGHT_AMBER, COLOR_AMBER_MID, header_bg='#FEEBC8')

    ax.annotate('', xy=(8.35, 1.88), xytext=(8.35, 2.22),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_AMBER_MID, lw=1.6, mutation_scale=11))

    draw_card(6.85, 0.25, 3.0, 1.60, "Manager Adjudication & Audit Log", [
        ("Department Manager Review (12.5%):", True, 6.6, COLOR_PRIMARY_NAVY),
        ("  5 cases escalated (R02, R05, R18, R30, R37)", False, 6.0, COLOR_TEXT_MAIN),
        ("  3 Adopted Equivalence; 2 Rejections", False, 6.0, COLOR_TEXT_MAIN),
        ("Immutable Compliance Audit Trail:", True, 6.6, COLOR_GREEN_DARK),
        ("  Logged spans, justifications, timestamps", False, 6.0, COLOR_TEXT_MUTED),
        ("  Aligns with EU AI Act Art. 14 / PIPL", False, 6.0, COLOR_TEXT_MUTED)
    ], COLOR_LIGHT_GREEN, COLOR_GREEN_DARK)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_fig2_results(csv_path="../data/eval_records_40.csv", out_pdf="fig2_results.pdf", out_png="fig2_results.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300)
    plt.subplots_adjust(wspace=0.34, top=0.88, bottom=0.20, left=0.08, right=0.92)

    roles = ['Quality Eng.\n(n = 27)', 'System Eng.*\n(n = 5)', 'Regulatory*\n(n = 8)', 'Overall\n(N = 40)']
    man_agree = [70.4, 40.0, 50.0, 62.5]
    eica_agree = [85.2, 80.0, 100.0, 87.5]

    x = np.arange(len(roles))
    width = 0.34

    rects1 = ax1.bar(x - width/2, man_agree, width, label='Manual Baseline',
                     color='#A0AEC0', edgecolor='#4A5568', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, eica_agree, width, label='EICA Protocol',
                     color='#2B6CB0', edgecolor='#1A365D', linewidth=0.8)

    ax1.set_ylabel('Agreement with Dept. Benchmark (%)', fontsize=7.6, fontweight='bold', color=COLOR_TEXT_MAIN)
    ax1.set_title('(a) Benchmark Agreement Rate by Role & Overall', fontsize=8.0, fontweight='bold', pad=6, color=COLOR_TEXT_MAIN)
    ax1.set_xticks(x)
    ax1.set_xticklabels(roles, fontsize=6.8)
    ax1.set_ylim(0, 116)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', fontsize=6.6, framealpha=0.92)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 1.8, f'{h:.1f}%',
                 ha='center', va='bottom', fontsize=6.3, color='#2D3748')
    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 1.8, f'{h:.1f}%',
                 ha='center', va='bottom', fontsize=6.3, fontweight='bold', color='#1A365D')

    ax1.text(0.5, -0.24, "Overall Cohen's kappa: 0.250 -> 0.746 (delta-kappa = +0.496, p = 0.00635)\n*Note: Sub-samples (n=5, 8) provide directional indications.",
             transform=ax1.transAxes, ha='center', fontsize=6.2, fontstyle='italic', color=COLOR_TEXT_MUTED)

    categories = ['False\nAdmissions', 'False\nRejections', 'Missed Hard\nQuals']
    man_errors = [9, 6, 4]
    eica_errors = [3, 2, 2]

    x2 = np.arange(len(categories))
    width2 = 0.30

    rects_e1 = ax2.bar(x2 - width2/2, man_errors, width2, label='Manual Errors (cases)',
                       color='#E53E3E', edgecolor='#742A2A', linewidth=0.8)
    rects_e2 = ax2.bar(x2 + width2/2, eica_errors, width2, label='EICA Errors (cases)',
                       color='#319795', edgecolor='#234E52', linewidth=0.8)

    ax2.set_ylabel('Error Count (cases)', fontsize=7.6, fontweight='bold', color=COLOR_TEXT_MAIN)
    ax2.set_title('(b) Decision Error Counts and Screening Time Profile', fontsize=8.0, fontweight='bold', pad=6, color=COLOR_TEXT_MAIN)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=6.8)
    ax2.set_ylim(0, 11)
    ax2.set_yticks([0, 2, 4, 6, 8, 10])
    ax2.grid(axis='y', linestyle='--', alpha=0.35)

    for rect in rects_e1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.20, f'{int(h)}',
                 ha='center', va='bottom', fontsize=6.6, fontweight='bold', color='#742A2A')
    for rect in rects_e2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.20, f'{int(h)}',
                 ha='center', va='bottom', fontsize=6.6, fontweight='bold', color='#234E52')

    ax2_time = ax2.twinx()
    ax2_time.set_ylabel('Mean Time (min / resume)', fontsize=7.6, fontweight='bold', color='#1A365D')
    ax2_time.set_ylim(0, 6.0)
    ax2_time.set_yticks([0, 1, 2, 3, 4, 5])

    time_x = [0.0, 1.0, 2.0]
    time_y = [4.80, 2.70, 3.35]
    line_time = ax2_time.plot(time_x, time_y, color='#1A365D', marker='s', markersize=4.5,
                              linestyle=':', linewidth=1.4, label='Time Profile')

    ax2_time.text(0.0, 4.80 + 0.20, '4.80m\n(Manual)', ha='center', fontsize=6.0, fontweight='bold', color='#1A365D')
    ax2_time.text(1.0, 2.70 - 0.45, '2.70m\n(Pure HR, -43.8%)', ha='center', fontsize=6.0, fontweight='bold', color='#1A365D')
    ax2_time.text(2.0, 3.35 + 0.20, '3.35m\n(Total Org, -30.2%)', ha='center', fontsize=6.0, fontweight='bold', color='#1A365D')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_time.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=6.2, framealpha=0.92)

    ax2.text(0.5, -0.24, "FA: -66.7% | FR: -66.7% | Pure HR Time: -43.8% | Net Org Time: -30.2%\nTotal Org Time includes 5 manager reviews (8.38 min/escalation).",
             transform=ax2.transAxes, ha='center', fontsize=6.2, fontstyle='italic', color=COLOR_TEXT_MUTED)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot_fig1_architecture("fig1_architecture.pdf", "fig1_architecture.png")
    plot_fig2_results("../data/eval_records_40.csv", "fig2_results.pdf", "fig2_results.png")
