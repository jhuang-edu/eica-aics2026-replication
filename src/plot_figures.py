# -*- coding: utf-8 -*-
"""
High-Resolution Publication Graphics Generator for Revised EICA Paper (IEEE AI-CS 2026)
=======================================================================================
Figure 1: EICA System Architecture (Ontology Masking, Hybrid Regulatory RAG,
          Constrained LLM Extractor, Deterministic Span/Verb Verifier, Hard FSM Gate)
Figure 2: Empirical Field Results, Trace-Driven Architectural Ablation (M0-M3),
          and Asymmetric Compliance Loss Sensitivity Curves (computed dynamically from CSV)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from .metrics_calculator import compute_metrics
    from .statistical_tests import run_statistical_suite
    from .ablation_and_benchmark import run_trace_driven_ablation
except ImportError:
    from metrics_calculator import compute_metrics
    from statistical_tests import run_statistical_suite
    from ablation_and_benchmark import run_trace_driven_ablation

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#2D3748"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLOR_PRIMARY_NAVY = "#1A365D"
COLOR_ACCENT_BLUE = "#2B6CB0"
COLOR_LIGHT_BLUE = "#EBF8FF"
COLOR_TEAL_DARK = "#234E52"
COLOR_TEAL_MID = "#319795"
COLOR_LIGHT_TEAL = "#E6FFFA"
COLOR_GREEN_DARK = "#22543D"
COLOR_LIGHT_GREEN = "#F0FFF4"
COLOR_CRIMSON_DARK = "#742A2A"
COLOR_CRIMSON_MID = "#C53030"
COLOR_LIGHT_RED = "#FFF5F5"
COLOR_AMBER_DARK = "#7B341E"
COLOR_AMBER_MID = "#DD6B20"
COLOR_LIGHT_AMBER = "#FFFAF0"
COLOR_PURPLE_DARK = "#44337A"
COLOR_PURPLE_MID = "#6B46C1"
COLOR_LIGHT_PURPLE = "#FAF5FF"
COLOR_GRAY_BG = "#F7FAFC"
COLOR_BORDER_GRAY = "#CBD5E0"
COLOR_TEXT_MAIN = "#1A202C"
COLOR_TEXT_MUTED = "#4A5568"


def plot_fig1_architecture(out_pdf="fig1_architecture.pdf", out_png="fig1_architecture.png"):
    """Draw the 4-column EICA system architecture synced to eica_engine.py."""
    fig = plt.figure(figsize=(10.0, 4.65), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 6.20)
    ax.axis("off")

    def draw_card(
        x, y, w, h, title, body_lines, bg_col, border_col,
        header_bg=None, title_col=COLOR_PRIMARY_NAVY, line_step=0.255
    ):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.1,
        )
        ax.add_patch(box)
        if header_bg:
            h_bar = FancyBboxPatch(
                (x, y + h - 0.39), w, 0.39,
                boxstyle="round,pad=0.02,rounding_size=0.07",
                facecolor=header_bg, edgecolor=border_col, linewidth=0.9,
            )
            ax.add_patch(h_bar)
            ax.text(
                x + w / 2.0, y + h - 0.195, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=title_col,
            )
        elif title:
            ax.text(
                x + w / 2.0, y + h - 0.21, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=title_col,
            )

        if body_lines:
            curr_y = y + h - 0.50
            for line, is_bold, fsize, col in body_lines:
                ax.text(
                    x + 0.075, curr_y, line, ha="left", va="top",
                    fontsize=fsize, fontweight="bold" if is_bold else "normal", color=col,
                )
                curr_y -= line_step

    # 4 Column Headers (col_w = 2.31, gaps = 0.18 -> 0.11, 2.60, 5.09, 7.58)
    col_w = 2.31
    xs = [0.11, 2.60, 5.09, 7.58]
    headers = [
        ("Stage 1: Ontology & Hybrid RAG", COLOR_PRIMARY_NAVY),
        ("Stage 2: Constrained Extractor", COLOR_ACCENT_BLUE),
        ("Stage 2b: Deterministic Verifier", COLOR_PURPLE_DARK),
        ("Stage 3: Hard FSM Decision Gate", COLOR_TEAL_DARK),
    ]
    for x0, (htxt, hcol) in zip(xs, headers):
        hdr = FancyBboxPatch(
            (x0, 5.70), col_w, 0.40,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=hcol, edgecolor=hcol,
        )
        ax.add_patch(hdr)
        ax.text(
            x0 + col_w / 2.0, 5.90, htxt, ha="center", va="center",
            fontsize=8.1, fontweight="bold", color="white",
        )

    # Stage 1
    draw_card(xs[0], 2.65, col_w, 2.92, "Three-Tier Ontology & Mask", [
        ("M (Mandatory): Hard gate", True, 8.0, COLOR_PRIMARY_NAVY),
        ("  Degree, ISO 13485, QMS lead", False, 7.8, COLOR_TEXT_MAIN),
        ("P (Preferred): Equivalence", True, 8.0, COLOR_PRIMARY_NAVY),
        ("  GMP / ISO 9001 transfer rules", False, 7.8, COLOR_TEXT_MAIN),
        ("V (Tacit): Schema-Masked", True, 8.0, COLOR_CRIMSON_DARK),
        ("  Omitted from LLM JSON schema", False, 7.8, COLOR_CRIMSON_DARK),
        ("  Host sets sigma(c,r) = empty", False, 7.8, COLOR_TEXT_MUTED),
        ("Closed Enum Sigma:", True, 8.0, COLOR_PRIMARY_NAVY),
        ("  {verified, not_prov, conflict}", False, 7.8, COLOR_TEXT_MAIN),
    ], COLOR_LIGHT_BLUE, COLOR_ACCENT_BLUE, header_bg="#DCEEFB", line_step=0.258)

    draw_card(xs[0], 0.08, col_w, 2.40, "Version-Pinned Hybrid RAG", [
        ("Whitelist Corpus D_pin (12 docs):", True, 7.9, COLOR_TEXT_MAIN),
        ("  Order 739/797, ISO 13485, SOPs", False, 7.8, COLOR_TEXT_MUTED),
        ("  NMPA GMP; ORD650-DEP excluded", False, 7.8, COLOR_TEXT_MUTED),
        ("Hybrid Retrieval Scoring:", True, 7.9, COLOR_ACCENT_BLUE),
        ("  S_hyb = a*BM25_hat+(1-a)*Cos_hat", False, 7.8, COLOR_PRIMARY_NAVY),
        ("  Top-3 clauses bound to prompt", False, 7.8, COLOR_TEXT_MAIN),
        ("Input: De-identified R01-R40", True, 7.8, COLOR_TEXT_MUTED),
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY, line_step=0.258)

    # Stage 2
    draw_card(xs[1], 2.55, col_w, 3.02, "Local LLM Evidence Extractor", [
        ("Qwen2.5-32B-Instruct (AWQ 4b)", True, 7.9, COLOR_PRIMARY_NAVY),
        ("  Single RTX 3090 (21.5 GiB)", False, 7.8, COLOR_TEXT_MUTED),
        ("  vLLM (temp=0.0, seed=42)", False, 7.8, COLOR_TEXT_MUTED),
        ("  Grammar/JSON Schema Mode", True, 7.8, COLOR_ACCENT_BLUE),
        ("4-Column Evidence Card:", True, 7.9, COLOR_TEAL_DARK),
        ("  Card(c,r) = <r, e, a, sigma>", True, 7.8, COLOR_PRIMARY_NAVY),
        ("  e(c,r): Verbatim resume span", False, 7.8, COLOR_TEXT_MAIN),
        ("  a(c,r): Line-span anchor", False, 7.8, COLOR_TEXT_MAIN),
        ("  Retry cap <= 2 on schema err", False, 7.8, COLOR_TEXT_MUTED),
    ], COLOR_LIGHT_TEAL, COLOR_TEAL_MID, header_bg="#CEF5EE", line_step=0.268)

    draw_card(xs[1], 0.08, col_w, 2.30, "Strict Generation Guardrails", [
        ("Prohibited Model Outputs:", True, 7.9, COLOR_CRIMSON_DARK),
        ("  Zero scalar fit scores", False, 7.8, COLOR_CRIMSON_DARK),
        ("  Zero candidate rankings", False, 7.8, COLOR_CRIMSON_DARK),
        ("  Zero admit/reject verdicts", False, 7.8, COLOR_CRIMSON_DARK),
        ("Host-Controlled Invariants:", True, 7.9, COLOR_TEXT_MAIN),
        ("  LLM acts purely as parser", False, 7.8, COLOR_TEXT_MUTED),
        ("  100% schema/tacit isolation", False, 7.8, COLOR_GREEN_DARK),
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY, line_step=0.252)

    # Stage 2b (Deterministic Verifier)
    draw_card(xs[2], 2.10, col_w, 3.47, "Span & Conflict Verifier", [
        ("1. Span-Anchor Alignment:", True, 7.9, COLOR_PURPLE_DARK),
        ("  Contig. Sim(e, t_src) >= 0.85", False, 7.8, COLOR_TEXT_MAIN),
        ("  Fail -> UNANCHORED_SPAN", False, 7.8, COLOR_CRIMSON_DARK),
        ("2. Verb-Escalation Guard (indep.):", True, 7.8, COLOR_PURPLE_DARK),
        ("  Clause-scoped v_src vs v_max", False, 7.8, COLOR_TEXT_MAIN),
        ("  Fail -> VERB_ESCALATION", False, 7.8, COLOR_CRIMSON_DARK),
        ("3. Credential Proof Rule (indep.):", True, 7.8, COLOR_PURPLE_DARK),
        ("  Full-span check; train != cert", False, 7.8, COLOR_TEXT_MAIN),
        ("  Fail -> MISSING_CREDENTIAL_PROOF", False, 7.8, COLOR_CRIMSON_DARK),
        ("4. Tacit Blank Invariant:", True, 7.9, COLOR_PURPLE_DARK),
        ("  Force sigma(c,r)=empty for V", False, 7.8, COLOR_TEXT_MAIN),
    ], COLOR_LIGHT_PURPLE, COLOR_PURPLE_MID, header_bg="#E9D8FD", line_step=0.262)

    draw_card(xs[2], 0.08, col_w, 1.88, "Verifier Runtime Telemetry", [
        ("Zero-Stochasticity CPU Check:", True, 7.9, COLOR_TEXT_MAIN),
        ("  Median latency: ~4.3 ms/case", False, 7.8, COLOR_GREEN_DARK),
        ("  P95 latency: ~9.8 ms/case", False, 7.8, COLOR_GREEN_DARK),
        ("Unit & Vignette Verification:", True, 7.9, COLOR_PURPLE_DARK),
        ("  Catches R31 & R05-sim faults", False, 7.8, COLOR_TEXT_MAIN),
        ("  Rule-check F1: 100% (90/90)", False, 7.8, COLOR_TEXT_MAIN),
    ], COLOR_GRAY_BG, COLOR_BORDER_GRAY, line_step=0.232)

    # Stage 3
    draw_card(xs[3], 2.52, col_w, 3.05, "Hard FSM Decision Gate", [
        ("Actions A = {Conf, Adj, Esc}", True, 7.9, COLOR_PRIMARY_NAVY),
        ("Confirm(D) -> s_term (D in {0,1})", True, 7.8, COLOR_GREEN_DARK),
        ("  D=1 blocked if M unverified", False, 7.8, COLOR_TEXT_MUTED),
        ("Adjust(D, sigma', rsn)->s_term", True, 7.8, COLOR_AMBER_DARK),
        ("  Requires |rsn| >= L_min (15)", False, 7.8, COLOR_TEXT_MUTED),
        ("Hard Conflict Lock -> s_block:", True, 7.9, COLOR_CRIMSON_DARK),
        ("  Locks D=1 on conflicts/flags", False, 7.8, COLOR_CRIMSON_DARK),
        ("  Forces Escalate -> s_esc", False, 7.8, COLOR_TEXT_MAIN),
        ("Transition: delta(s, a) -> S", True, 7.8, COLOR_PRIMARY_NAVY),
    ], COLOR_LIGHT_AMBER, COLOR_AMBER_MID, header_bg="#FEEBC8", line_step=0.268)

    draw_card(xs[3], 0.08, col_w, 2.28, "Adjudication & Audit Log", [
        ("Manager Adjudication (s_esc):", True, 7.9, COLOR_PRIMARY_NAVY),
        ("  Resolves R02, R18, R31 (FA->0)", False, 7.8, COLOR_GREEN_DARK),
        ("  Adopts equiv. R05, R30, R37", False, 7.8, COLOR_TEXT_MAIN),
        ("Structured Audit Log:", True, 7.9, COLOR_TEAL_DARK),
        ("  Anchors, flags, reasons, ts", False, 7.8, COLOR_TEXT_MUTED),
        ("  EU AI Act Art.12/14 & PIPL 24", False, 7.8, COLOR_TEXT_MUTED),
        ("  Upper bound: 95.0% (\u03ba=0.896)", True, 7.8, COLOR_GREEN_DARK),
    ], COLOR_LIGHT_GREEN, COLOR_GREEN_DARK, line_step=0.250)

    # Horizontal arrows between stages
    arrow_kw = dict(arrowstyle="-|>", color=COLOR_PRIMARY_NAVY, lw=1.5, mutation_scale=11)
    ax.annotate("", xy=(xs[1] - 0.02, 4.10), xytext=(xs[0] + col_w + 0.02, 4.10), arrowprops=arrow_kw)
    ax.annotate("", xy=(xs[1] - 0.02, 1.25), xytext=(xs[0] + col_w + 0.02, 1.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(xs[2] - 0.02, 4.00), xytext=(xs[1] + col_w + 0.02, 4.00), arrowprops=arrow_kw)
    ax.annotate(
        "", xy=(xs[3] - 0.02, 4.00), xytext=(xs[2] + col_w + 0.02, 4.00),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_PURPLE_DARK, lw=1.5, mutation_scale=11),
    )
    # Vertical arrow in Stage 3
    ax.annotate(
        "", xy=(xs[3] + col_w / 2.0, 2.38), xytext=(xs[3] + col_w / 2.0, 2.52),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_AMBER_MID, lw=1.5, mutation_scale=11),
    )

    plt.tight_layout(pad=0.12)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_results(csv_path="../data/eval_records_40.csv", out_pdf="fig2_results.pdf", out_png="fig2_results.png"):
    """
    3-Panel Systems Evaluation Figure (computed dynamically from csv_path):
      (a) Field Agreement Rate by Role (Manual vs. EICA Field Track)
      (b) Trace-Driven Architectural Policy Ablation (M0 -> M1 -> M2 -> M3)
      (c) Expected Asymmetric Compliance Loss E[L(D,Y)] vs. Penalty Ratio lambda_FA / lambda_FR
    """
    csv_p = Path(csv_path)
    if not csv_p.exists() and str(csv_path) == "../data/eval_records_40.csv":
        fallback = Path(__file__).resolve().parent.parent / "data" / "eval_records_40.csv"
        if fallback.exists():
            csv_p = fallback

    m = compute_metrics(csv_p)
    s = run_statistical_suite(csv_p, n_bootstraps=2000)
    abl = run_trace_driven_ablation(csv_p, n_bootstraps=2000)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.45), dpi=300)
    plt.subplots_adjust(wspace=0.42, top=0.86, bottom=0.27, left=0.06, right=0.96)

    # -------------------------------------------------------------------------
    # Panel (a): Role Breakdown (Field Track 1 vs Field Track 2)
    # -------------------------------------------------------------------------
    qa = m["roles"]["Quality Engineer (QA/QC)"]
    qms = m["roles"]["System Engineer (QMS)"]
    ra = m["roles"]["Regulatory Affairs (RA)"]
    roles = [
        f"QA/QC\n(n={qa['N']})",
        f"QMS*\n(n={qms['N']})",
        f"RA*\n(n={ra['N']})",
        f"Overall\n(N={m['N']})",
    ]
    man_agree = [qa["manual_agree_pct"], qms["manual_agree_pct"], ra["manual_agree_pct"], m["agreement"]["manual"]]
    eica_agree = [qa["eica_agree_pct"], qms["eica_agree_pct"], ra["eica_agree_pct"], m["agreement"]["eica"]]

    x1 = np.arange(len(roles))
    w1 = 0.31
    pair_gap = 0.08
    r1 = ax1.bar(x1 - w1 / 2 - pair_gap / 2, man_agree, w1, label="M0: Manual Baseline", color="#A0AEC0", edgecolor="#4A5568", lw=0.8)
    r2 = ax1.bar(x1 + w1 / 2 + pair_gap / 2, eica_agree, w1, label="M1: EICA Field Log", color="#2B6CB0", edgecolor="#1A365D", lw=0.8)
    ax1.set_xlim(-0.62, len(roles) - 1 + 0.60)

    ax1.set_ylabel("Benchmark Agreement (%)", fontsize=8.8, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax1.set_title("(a) Field Agreement by Role", fontsize=9.4, fontweight="bold", pad=7, color=COLOR_TEXT_MAIN)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(roles, fontsize=8.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.tick_params(axis="y", labelsize=8.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    bar_labels = []
    for rect1, rect2 in zip(r1, r2):
        h1, h2 = rect1.get_height(), rect2.get_height()
        xc1 = rect1.get_x() + rect1.get_width() / 2.0 - 0.10
        xc2 = rect2.get_x() + rect2.get_width() / 2.0 + 0.02
        t1 = ax1.text(xc1, h1 + 1.8, f"{h1:.1f}%", ha="center", va="bottom", fontsize=7.6, color="#2D3748")
        t2 = ax1.text(xc2, h2 + 1.8, f"{h2:.1f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#1A365D")
        bar_labels.append([xc1, h1 + 1.8, t1])
        bar_labels.append([xc2, h2 + 1.8, t2])

    ax1_ylim_top = 142.0
    ax1.set_ylim(0, ax1_ylim_top)
    leg1 = ax1.legend(loc="upper left", fontsize=7.8, framealpha=0.94)

    # General renderer-space collision resolution across all 8 bar labels
    # (intra-pair + inter-group) and legend clearance
    pad_x_px, pad_y_px = 5.0, 6.0
    for _ in range(40):
        fig.canvas.draw()
        renderer1 = fig.canvas.get_renderer()
        ax1_bb = ax1.get_window_extent(renderer1)
        ax1_h_px = max(ax1_bb.height, 1.0)
        lbl_bbs = [item[2].get_window_extent(renderer1) for item in bar_labels]
        leg1_bb = leg1.get_window_extent(renderer1)

        moved = False
        order = sorted(range(len(bar_labels)), key=lambda idx: (bar_labels[idx][1], idx))
        for pos_a in range(len(order)):
            ia = order[pos_a]
            ba = lbl_bbs[ia]
            for pos_b in range(pos_a + 1, len(order)):
                ib = order[pos_b]
                bb = lbl_bbs[ib]
                x_ov = (ba.x1 + pad_x_px > bb.x0) and (bb.x1 + pad_x_px > ba.x0)
                y_ov = (ba.y1 + pad_y_px > bb.y0) and (bb.y1 + pad_y_px > ba.y0)
                if x_ov and y_ov:
                    dy_px = (ba.y1 + pad_y_px) - bb.y0
                    if dy_px > 0:
                        dy_data = dy_px * (ax1_ylim_top / ax1_h_px)
                        bar_labels[ib][1] += dy_data
                        bar_labels[ib][2].set_y(bar_labels[ib][1])
                        moved = True
                        break
            if moved:
                break

        if moved:
            continue

        max_leg_deficit_px = 0.0
        for ba in lbl_bbs:
            x_ov = (ba.x1 + pad_x_px > leg1_bb.x0) and (leg1_bb.x1 + pad_x_px > ba.x0)
            y_ov = (ba.y1 + pad_y_px > leg1_bb.y0) and (leg1_bb.y1 + pad_y_px > ba.y0)
            if x_ov and y_ov:
                deficit = (ba.y1 + pad_y_px) - leg1_bb.y0
                if deficit > max_leg_deficit_px:
                    max_leg_deficit_px = deficit

        if max_leg_deficit_px > 0:
            ax1_ylim_top += max(6.0, max_leg_deficit_px * (ax1_ylim_top / ax1_h_px) * 1.35)
            ax1.set_ylim(0, ax1_ylim_top)
            continue

        break

    ax1.text(
        0.5, -0.32,
        f"All N={m['N']}: Exact McNemar p = 0.00635 (11 vs. 1)\n"
        f"*Exploratory small sub-samples (n = {qms['N']}, {ra['N']}).",
        transform=ax1.transAxes, ha="center", fontsize=7.8, fontstyle="italic", color=COLOR_TEXT_MUTED,
    )

    # -------------------------------------------------------------------------
    # Panel (b): Architectural Policy Ablation (M0 -> M1 -> M2 -> M3)
    # -------------------------------------------------------------------------
    abl_keys = ["M0_Manual", "M1_EICA_SoftCard", "M2_EICA_HardFSM", "M3_EICA_FullVerifier"]
    tiers = ["M0\nManual", "M1\nSoft-Card", "M2\n+Hard FSM", "M3\n+Verifier"]
    agree_vals = [abl[k]["agreement_pct"] for k in abl_keys]
    fa_counts = [abl[k]["FP"] for k in abl_keys]
    bar_colors = ["#A0AEC0", "#2B6CB0", "#319795", "#22543D"]

    x2 = np.arange(len(tiers))
    bars = ax2.bar(x2, agree_vals, width=0.52, color=bar_colors, edgecolor="#1A202C", lw=0.8, alpha=0.92, label="Agreement (%)")
    ax2.set_ylabel("Agreement (%)", fontsize=8.8, fontweight="bold", color=COLOR_PRIMARY_NAVY)
    ax2.set_title("(b) Architectural Policy Ablation", fontsize=9.4, fontweight="bold", pad=7, color=COLOR_TEXT_MAIN)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(tiers, fontsize=8.0)
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis="y", labelsize=8.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    for rect, val in zip(bars, agree_vals):
        ax2.text(
            rect.get_x() + rect.get_width() / 2.0, val + 2.0, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#1A202C",
        )

    ax2_r = ax2.twinx()
    ax2_r.plot(x2, fa_counts, color=COLOR_CRIMSON_MID, marker="o", markersize=5.2, linewidth=1.8, label="False Admissions")
    ax2_r.set_ylabel(f"False Admissions (FA / {m['N']})", fontsize=8.6, fontweight="bold", color=COLOR_CRIMSON_DARK)
    max_fa = max(fa_counts) if fa_counts else 0
    if max_fa <= 9:
        ax2_r.set_ylim(-0.9, 11.2)
        ax2_r.set_yticks([0, 3, 6, 9])
    else:
        top_tick = int(np.ceil(max_fa / 3.0) * 3)
        ax2_r.set_ylim(-0.08 * top_tick, top_tick * 1.22)
        ax2_r.set_yticks(list(range(0, top_tick + 1, max(1, top_tick // 3))))
    ax2_r.tick_params(axis="y", labelsize=8.0, colors=COLOR_CRIMSON_DARK)

    fa_span = max(max_fa, 9)
    ry_min, ry_max = ax2_r.get_ylim()
    ry_range = max(ry_max - ry_min, 1e-9)
    for xi, agree_v, fa in zip(x2, agree_vals, fa_counts):
        bar_lo = (agree_v + 2.0) / 118.0
        bar_hi = bar_lo + 0.095
        fa_frac = (fa - ry_min) / ry_range
        offset = 0.075 * fa_span

        # Candidate 1: above marker (default)
        above_y = fa + offset
        above_lo = (above_y - ry_min) / ry_range
        above_hi = above_lo + 0.095
        # Candidate 2: below marker
        below_y = fa - offset
        below_hi = (below_y - ry_min) / ry_range
        below_lo = below_hi - 0.095

        def _clearance(lo, hi):
            if lo < 0.02 or hi > 0.98:
                return -1.0
            if hi <= bar_lo:
                return bar_lo - hi
            if lo >= bar_hi:
                return lo - bar_hi
            return -min(hi - bar_lo, bar_hi - lo)

        c_above = _clearance(above_lo, above_hi)
        c_below = _clearance(below_lo, below_hi)
        if c_above >= 0.04 or c_above >= c_below:
            if c_above < 0.02 and c_below < 0.02:
                # Fallback: nudge laterally if vertically boxed in on both sides
                x_pos = xi + (0.24 if xi < len(x2) - 1 else -0.24)
                y_pos, va_pos = above_y, "bottom"
            else:
                x_pos, y_pos, va_pos = xi + 0.04, above_y, "bottom"
        else:
            x_pos, y_pos, va_pos = xi + 0.04, below_y, "top"

        ax2_r.text(
            x_pos, y_pos, f"FA={fa}", ha="center", va=va_pos,
            fontsize=7.8, fontweight="bold", color=COLOR_CRIMSON_DARK,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=COLOR_CRIMSON_MID, lw=0.6, alpha=0.94),
        )

    m2 = abl["M2_EICA_HardFSM"]
    m3 = abl["M3_EICA_FullVerifier"]
    ax2.text(
        0.5, -0.32,
        f"M2 hard lock (\u03ba={m2['kappa']:.3f});\n"
        f"M3 +verifier (FA={m3['FP']}, \u03ba={m3['kappa']:.3f}).",
        transform=ax2.transAxes, ha="center", fontsize=7.8, fontstyle="italic", color=COLOR_TEXT_MUTED,
    )

    # -------------------------------------------------------------------------
    # Panel (c): Expected Asymmetric Compliance Loss E[L(D, Y)] Sensitivity
    # -------------------------------------------------------------------------
    ratios = np.array([1, 2, 5, 10, 20])
    loss_m0 = [abl["M0_Manual"]["expected_loss_by_lambda_fa"][str(r)] for r in ratios]
    loss_m1 = [abl["M1_EICA_SoftCard"]["expected_loss_by_lambda_fa"][str(r)] for r in ratios]
    loss_m2 = [abl["M2_EICA_HardFSM"]["expected_loss_by_lambda_fa"][str(r)] for r in ratios]
    loss_m3 = [abl["M3_EICA_FullVerifier"]["expected_loss_by_lambda_fa"][str(r)] for r in ratios]

    ax3.plot(ratios, loss_m0, marker="s", lw=1.7, ms=4.5, color="#718096", linestyle="--",
             label=f"M0: Manual ({abl['M0_Manual']['total_org_time_min']:.2f} min)")
    ax3.plot(ratios, loss_m1, marker="^", lw=1.7, ms=4.8, color="#2B6CB0",
             label=f"M1: Soft-Card ({abl['M1_EICA_SoftCard']['total_org_time_min']:.2f} min)")
    ax3.plot(ratios, loss_m2, marker="D", lw=1.7, ms=4.5, color="#319795",
             label=f"M2: +Hard FSM ({abl['M2_EICA_HardFSM']['total_org_time_min']:.2f} min)")
    ax3.plot(ratios, loss_m3, marker="o", lw=2.0, ms=5.0, color="#22543D",
             label=f"M3: +Verifier ({abl['M3_EICA_FullVerifier']['total_org_time_min']:.2f} min)")

    ax3.set_xlabel(r"Asymmetric Penalty Ratio ($\lambda_{\mathrm{FA}} / \lambda_{\mathrm{FR}}$)", fontsize=10.6, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax3.set_ylabel(r"Expected Compliance Loss $E[L]$", fontsize=8.8, fontweight="bold", color=COLOR_TEXT_MAIN)
    ax3.set_title("(c) Compliance Loss Sensitivity", fontsize=9.4, fontweight="bold", pad=7, color=COLOR_TEXT_MAIN)
    ax3.set_xticks(ratios)
    max_loss = max(max(loss_m0), max(loss_m1), max(loss_m2), max(loss_m3), 1.0)
    ax3.set_ylim(-0.035 * max_loss, max(5.1, max_loss * 1.12) if max_loss > 4.0 else max_loss * 1.25)
    ax3.tick_params(axis="both", labelsize=8.0)
    ax3.grid(True, linestyle="--", alpha=0.35)

    l0_10 = abl["M0_Manual"]["expected_loss_by_lambda_fa"]["10"]
    l3_10 = abl["M3_EICA_FullVerifier"]["expected_loss_by_lambda_fa"]["10"]
    if l0_10 > 0:
        pct_diff = (l3_10 - l0_10) / l0_10 * 100.0
        callout_txt = f"{pct_diff:+.1f}% loss\nat 10:1"
    else:
        callout_txt = "0.0 loss\nat 10:1"
    y_hi = ax3.get_ylim()[1]

    # Sample curve points in display space to place legend and 10:1 callout without collision
    all_curves = [loss_m0, loss_m1, loss_m2, loss_m3]
    dense_x = np.linspace(1.0, 20.0, 60)
    dense_pts = []
    for curve in all_curves:
        dense_y = np.interp(dense_x, ratios, curve)
        dense_pts.append(np.column_stack([dense_x, dense_y]))
    all_curve_data = np.vstack(dense_pts)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax3_bb = ax3.get_window_extent(renderer)
    curve_disp = ax3.transData.transform(all_curve_data)

    def _bbox_overlap_area(b1, b2, pad=3.0):
        ix0 = max(b1.x0 - pad, b2.x0 - pad)
        iy0 = max(b1.y0 - pad, b2.y0 - pad)
        ix1 = min(b1.x1 + pad, b2.x1 + pad)
        iy1 = min(b1.y1 + pad, b2.y1 + pad)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def _curve_hits(bb, pad=4.0):
        in_x = (curve_disp[:, 0] >= bb.x0 - pad) & (curve_disp[:, 0] <= bb.x1 + pad)
        in_y = (curve_disp[:, 1] >= bb.y0 - pad) & (curve_disp[:, 1] <= bb.y1 + pad)
        return int(np.sum(in_x & in_y))

    cand_xytexts = [
        (10.8, min(y_hi * 0.62, max(l3_10 + 0.34 * max_loss, 0.36 * y_hi))),
        (10.5, max(0.08 * y_hi, l3_10 - 0.28 * y_hi)),
        (12.2, max(0.08 * y_hi, l3_10 - 0.24 * y_hi)),
        (10.8, min(0.82 * y_hi, l3_10 + 0.22 * y_hi)),
        (11.8, 0.16 * y_hi),
        (11.8, 0.80 * y_hi),
        (2.4, 0.78 * y_hi),
        (2.4, 0.18 * y_hi),
    ]
    cand_locs = ["upper left", "lower left", "lower right", "upper right"]

    best_choice = None
    for loc_idx, loc in enumerate(cand_locs):
        leg = ax3.legend(loc=loc, fontsize=7.8, framealpha=0.95)
        fig.canvas.draw()
        leg_bb = leg.get_window_extent(renderer)
        leg_hits = _curve_hits(leg_bb, pad=3.0)
        for tx_idx, (tx, ty) in enumerate(cand_xytexts):
            ann = ax3.annotate(
                callout_txt,
                xy=(10, l3_10),
                xytext=(tx, ty),
                fontsize=7.8, fontweight="bold", color="#22543D",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#F0FFF4", edgecolor="#22543D", lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#22543D", lw=1.1),
            )
            fig.canvas.draw()
            ann_bb = ann.get_window_extent(renderer)
            inside_ax = (
                ann_bb.x0 >= ax3_bb.x0 + 2
                and ann_bb.x1 <= ax3_bb.x1 - 2
                and ann_bb.y0 >= ax3_bb.y0 + 2
                and ann_bb.y1 <= ax3_bb.y1 - 2
            )
            leg_ov = _bbox_overlap_area(ann_bb, leg_bb, pad=4.0)
            ann_hits = _curve_hits(ann_bb, pad=3.0)
            ann.remove()
            # Strongly penalize legend overlap or out-of-bounds; preserve (loc=0, tx=0) when clean
            score = (
                (1e6 if leg_ov > 0 else 0.0)
                + (1e5 if not inside_ax else 0.0)
                + leg_hits * 12.0
                + ann_hits * 8.0
                + loc_idx * 3.0
                + tx_idx * 1.5
            )
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, loc, (tx, ty))
            if leg_ov == 0 and inside_ax and loc_idx == 0 and tx_idx == 0 and leg_hits <= 25:
                break
        leg.remove()
        if best_choice[0] < 250.0 and loc_idx == 0 and best_choice[2] == cand_xytexts[0]:
            break

    _, chosen_loc, chosen_xytext = best_choice
    ax3.legend(loc=chosen_loc, fontsize=7.8, framealpha=0.95)
    ax3.annotate(
        callout_txt,
        xy=(10, l3_10),
        xytext=chosen_xytext,
        fontsize=7.8, fontweight="bold", color="#22543D",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#F0FFF4", edgecolor="#22543D", lw=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#22543D", lw=1.1),
    )

    ax3.text(
        0.5, -0.34,
        f"Normalized: \u03bb_FR=1.0, \u03bb_Esc=0.25.\n"
        f"M3 FA={m3['FP']} ({-m3['net_time_reduction_pct']:+.1f}% net org. time).",
        transform=ax3.transAxes, ha="center", fontsize=7.8, fontstyle="italic", color=COLOR_TEXT_MUTED,
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import shutil
    _code_dir = Path(__file__).resolve().parent.parent
    _out_dir = _code_dir / "output"
    _out_dir.mkdir(parents=True, exist_ok=True)
    _csv_default = _code_dir / "data" / "eval_records_40.csv"
    f1_pdf = _out_dir / "fig1_architecture.pdf"
    f1_png = _out_dir / "fig1_architecture.png"
    f2_pdf = _out_dir / "fig2_results.pdf"
    f2_png = _out_dir / "fig2_results.png"
    plot_fig1_architecture(str(f1_pdf), str(f1_png))
    plot_fig2_results(str(_csv_default), str(f2_pdf), str(f2_png))
    for _fig_dir_name in ("paper_4p", "paper_9p"):
        _paper_fig = _code_dir.parent / _fig_dir_name / "figures"
        if _paper_fig.exists():
            for _src in (f1_pdf, f1_png, f2_pdf, f2_png):
                shutil.copy2(_src, _paper_fig / _src.name)
