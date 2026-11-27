# -*- coding: utf-8 -*-
"""
Shared Non-Destructive Helpers & Telemetry Analyzers for the 4-Tier E2E Suite.

Design guarantees:
  - Strictly read-only against code/data/, code/src/, paper_9p/, Templates/, docs.
  - Compiles paper_9p/main.tex via `tectonic --keep-logs -o <tmp_dir>` inside a temp dir.
  - Runs code/reproduce_all.py inside an isolated temp copy of code/ + paper_9p/figures/.
  - Uses `with fitz.open(...) as doc:` and Windows `onexc` write-bit clearing to prevent
    WinError 32 / WinError 5 cleanup issues.
  - Caches isolated tectonic and reproduce_all runs once per process via lru_cache.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import fitz  # PyMuPDF (required for PDF/manuscript checks; optional in standalone code clones)
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


_THIS_CODE_ROOT = Path(__file__).resolve().parents[2]
_PARENT_ROOT = Path(__file__).resolve().parents[3]
HAS_FULL_WORKSPACE = (_PARENT_ROOT / "paper_9p" / "main.tex").exists() and (_THIS_CODE_ROOT.name == "code")
STANDALONE_REPLICATION_MODE = not HAS_FULL_WORKSPACE

REPO_ROOT = _PARENT_ROOT if HAS_FULL_WORKSPACE else _THIS_CODE_ROOT
CODE_DIR = _THIS_CODE_ROOT
SRC_DIR = CODE_DIR / "src"
DATA_DIR = CODE_DIR / "data"
PAPER_DIR = REPO_ROOT / "paper_9p"
SECTIONS_DIR = PAPER_DIR / "sections"
PAPER_FIG_DIR = PAPER_DIR / "figures"
TEMPLATES_DIR = REPO_ROOT / "Templates"

EVAL_CSV_PATH = DATA_DIR / "eval_records_40.csv"
ARCHIVE_CSV_PATH = DATA_DIR / "recruitment_archive_18m.csv"

# Canonical LF-normalized SHA-256 digests (cross-platform invariant across Windows CRLF and Linux/macOS LF)
EXPECTED_EVAL_CSV_CANONICAL_SHA256 = "f096b353ed1bc1856cedc8761d7b627f8f05ad6d7043e0ca343c15bb5ff1abdc"
EXPECTED_ARCHIVE_CSV_CANONICAL_SHA256 = "070945d3d1b0b157eaac4170855d5c7f22e43b87caec875b71de4ad45b1da66f"
# Legacy Windows CRLF byte-exact digests retained for backward-compatible reference
EXPECTED_EVAL_CSV_CRLF_SHA256 = "15b02531243e5e5bfe4702c4c3fe2108798b3f2a4ff8897504048d142b2c3910"
EXPECTED_ARCHIVE_CSV_CRLF_SHA256 = "2a2c38be33a96221669d747f1414a4c3482b7ab8f468562e8ffc1414e4be3914"
EXPECTED_EVAL_CSV_SHA256 = EXPECTED_EVAL_CSV_CANONICAL_SHA256
EXPECTED_ARCHIVE_CSV_SHA256 = EXPECTED_ARCHIVE_CSV_CANONICAL_SHA256


def requires_full_workspace(test_item):
    """Skip manuscript/LaTeX/Templates-only E2E checks cleanly in standalone replication clones."""
    return unittest.skipUnless(
        HAS_FULL_WORKSPACE,
        "Skipped in standalone code replication clone (requires full author workspace with paper_9p/, Templates/, and tectonic).",
    )(test_item)


def requires_pymupdf(test_item):
    """Skip PyMuPDF-dependent PDF inspection when fitz is not installed in minimal standalone env."""
    return unittest.skipUnless(
        fitz is not None,
        "Skipped because PyMuPDF (fitz) is optional in standalone code-only environments.",
    )(test_item)

SECTION_FILES = [
    "01_intro.tex",
    "02_related.tex",
    "03_model.tex",
    "04_setup.tex",
    "05_results.tex",
    "06_discussion.tex",
    "07_conclusion.tex",
]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
PUA_RE = re.compile(r"[\ue000-\uf8ff]")


def ensure_src_on_path() -> None:
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def compute_sha256(path: Path) -> str:
    """Compute canonical LF-normalized SHA-256 for text/CSV files so CRLF (Windows) and LF (Linux/macOS) match."""
    raw = path.read_bytes()
    if path.suffix.lower() in {".csv", ".txt", ".md", ".py", ".tex", ".yaml", ".yml"}:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def compute_raw_sha256(path: Path) -> str:
    """Compute exact raw-byte SHA-256 without line-ending normalization."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_csv_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if any(v.strip() for v in r.values() if v is not None)]
    return fieldnames, rows


def build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _handle_remove_readonly(func, path, exc_info=None):
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    try:
        func(path)
    except OSError:
        pass


def safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_handle_remove_readonly)
    else:
        shutil.rmtree(path, onerror=_handle_remove_readonly)


@dataclass
class TectonicBuildResult:
    returncode: int
    stdout: str
    stderr: str
    log_text: str
    pdf_bytes: bytes
    elapsed_sec: float


@functools.lru_cache(maxsize=1)
def get_tectonic_build() -> TectonicBuildResult:
    """Compile paper_9p/main.tex non-destructively into a temp directory once per process."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="eica_tex_"))
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["tectonic", "--keep-logs", "-o", str(tmp_dir), str(PAPER_DIR / "main.tex")],
            cwd=str(PAPER_DIR),
            env=build_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        log_path = tmp_dir / "main.log"
        pdf_path = tmp_dir / "main.pdf"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else b""
        return TectonicBuildResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            log_text=log_text,
            pdf_bytes=pdf_bytes,
            elapsed_sec=elapsed,
        )
    finally:
        safe_rmtree(tmp_dir)


@dataclass
class ReproduceRunResult:
    returncode: int
    stdout: str
    stderr: str
    fig1_pdf_bytes: bytes
    fig1_png_bytes: bytes
    fig2_pdf_bytes: bytes
    fig2_png_bytes: bytes
    elapsed_sec: float


@functools.lru_cache(maxsize=1)
def get_reproduce_run() -> ReproduceRunResult:
    """Run code/reproduce_all.py inside an isolated temporary workspace copy once per process."""
    tmp_root = Path(tempfile.mkdtemp(prefix="eica_repro_"))
    try:
        tmp_code = tmp_root / "code"
        tmp_paper_fig = tmp_root / "paper_9p" / "figures"
        tmp_code.mkdir(parents=True, exist_ok=True)
        tmp_paper_fig.mkdir(parents=True, exist_ok=True)
        (tmp_code / "output").mkdir(parents=True, exist_ok=True)

        shutil.copytree(DATA_DIR, tmp_code / "data")
        shutil.copytree(SRC_DIR, tmp_code / "src")
        shutil.copy2(CODE_DIR / "reproduce_all.py", tmp_code / "reproduce_all.py")

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(tmp_code / "reproduce_all.py")],
            cwd=str(tmp_code),
            env=build_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        elapsed = time.perf_counter() - t0

        out_dir = tmp_code / "output"
        f1_pdf = (out_dir / "fig1_architecture.pdf").read_bytes() if (out_dir / "fig1_architecture.pdf").exists() else b""
        f1_png = (out_dir / "fig1_architecture.png").read_bytes() if (out_dir / "fig1_architecture.png").exists() else b""
        f2_pdf = (out_dir / "fig2_results.pdf").read_bytes() if (out_dir / "fig2_results.pdf").exists() else b""
        f2_png = (out_dir / "fig2_results.png").read_bytes() if (out_dir / "fig2_results.png").exists() else b""

        return ReproduceRunResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            fig1_pdf_bytes=f1_pdf,
            fig1_png_bytes=f1_png,
            fig2_pdf_bytes=f2_pdf,
            fig2_png_bytes=f2_png,
            elapsed_sec=elapsed,
        )
    finally:
        safe_rmtree(tmp_root)


def parse_reproduce_ledger(stdout: str) -> List[Tuple[str, str, str, str]]:
    """Parse the 1:1 audit ledger lines from reproduce_all.py stdout."""
    rows: List[Tuple[str, str, str, str]] = []
    for line in stdout.splitlines():
        m = re.match(r"^(.*?)\s{2,}(\S.*?)\s{2,}(\S.*?)\s+(\[(?:MATCH 100%|CHECK)\])\s*$", line)
        if m:
            rows.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()))
    return rows


# =============================================================================
# TeX Source Helpers
# =============================================================================

def strip_tex_comments(tex: str) -> str:
    """Strip unescaped % comments line-by-line."""
    cleaned_lines = []
    for line in tex.splitlines():
        out = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = (ch == "\\") and not escaped
            if ch != "\\":
                escaped = False
        cleaned_lines.append("".join(out))
    return "\n".join(cleaned_lines)


def read_paper_sources() -> Dict[str, str]:
    sources = {"main.tex": (PAPER_DIR / "main.tex").read_text(encoding="utf-8")}
    for sec in SECTION_FILES:
        sources[sec] = (SECTIONS_DIR / sec).read_text(encoding="utf-8")
    return sources


def combined_body_tex() -> str:
    sources = read_paper_sources()
    return "\n".join(strip_tex_comments(sources[s]) for s in SECTION_FILES)


def extract_bibitems(main_tex: str) -> List[Tuple[str, str]]:
    """Return [(cite_key, bib_body_text), ...] from main.tex."""
    clean = strip_tex_comments(main_tex)
    m = re.search(r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}", clean, re.DOTALL)
    if not m:
        return []
    bib_block = m.group(1)
    parts = re.split(r"\\bibitem\{([^}]+)\}", bib_block)
    items: List[Tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        items.append((parts[i].strip(), parts[i + 1].strip()))
    return items


def extract_ordered_citations(sources: Optional[Dict[str, str]] = None) -> List[str]:
    """Extract citation keys in order of first appearance in body sections (excluding title/abstract)."""
    if sources is None:
        sources = read_paper_sources()
    seen: List[str] = []
    seen_set: Set[str] = set()
    for sec in SECTION_FILES:
        txt = strip_tex_comments(sources[sec])
        for m in re.finditer(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", txt):
            for key in m.group(1).split(","):
                k = key.strip()
                if k and k not in seen_set:
                    seen.append(k)
                    seen_set.add(k)
    return seen


@functools.lru_cache(maxsize=1)
def get_cached_metrics() -> dict:
    ensure_src_on_path()
    from metrics_calculator import compute_metrics
    return compute_metrics(EVAL_CSV_PATH)


@functools.lru_cache(maxsize=1)
def get_cached_stats_20k() -> dict:
    ensure_src_on_path()
    from statistical_tests import run_statistical_suite
    return run_statistical_suite(EVAL_CSV_PATH, n_bootstraps=20000)


@functools.lru_cache(maxsize=1)
def get_cached_ablation_20k() -> dict:
    ensure_src_on_path()
    from ablation_and_benchmark import run_trace_driven_ablation
    return run_trace_driven_ablation(EVAL_CSV_PATH, n_bootstraps=20000)


@functools.lru_cache(maxsize=1)
def get_cached_synthetic_42() -> dict:
    ensure_src_on_path()
    from ablation_and_benchmark import run_synthetic_verification_benchmark
    return run_synthetic_verification_benchmark(seed=42)


def extract_tex_brace_arg(tex: str, cmd_prefix: str) -> List[str]:
    """Extract brace-balanced arguments following cmd_prefix (e.g., '\\caption{')."""
    results: List[str] = []
    idx = 0
    n = len(tex)
    while True:
        pos = tex.find(cmd_prefix, idx)
        if pos == -1:
            break
        start = pos + len(cmd_prefix)
        depth = 1
        cur = start
        while cur < n and depth > 0:
            ch = tex[cur]
            if ch == "\\" and cur + 1 < n:
                cur += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cur += 1
        if depth == 0:
            results.append(tex[start : cur - 1].strip())
        idx = cur
    return results


def extract_caption_args(tex: str) -> List[str]:
    """Extract brace-balanced \\caption{...} or \\caption[short]{...} arguments."""
    results: List[str] = []
    n = len(tex)
    for m in re.finditer(r"\\caption(?:\[[^\]]*\])?\{", tex):
        start = m.end()
        depth = 1
        cur = start
        while cur < n and depth > 0:
            ch = tex[cur]
            if ch == "\\" and cur + 1 < n:
                cur += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cur += 1
        if depth == 0:
            results.append(tex[start : cur - 1].strip())
    return results


# =============================================================================
# PyMuPDF Telemetry Analyzers (Baseline-Aware, False-Positive Free)
# =============================================================================

@dataclass
class PdfTelemetry:
    page_count: int
    page_sizes: List[Tuple[float, float]]
    full_text: str
    pages_text: List[str]
    fonts_base: Set[str]
    has_times_font: bool
    has_mono_font: bool
    has_lmroman_fallback: bool
    cjk_chars: List[Tuple[int, str]]
    pua_chars: List[Tuple[int, str]]
    unresolved_refs: List[str]
    orphaned_punct: List[Tuple[int, str, float, float]]
    wrapped_eq_numbers: List[Tuple[int, str, Tuple[float, float, float, float]]]
    table_small_spans: List[Tuple[int, float, str]]
    figure_spans: List[Tuple[int, float, str]]
    last_page_col_diff_pt: Optional[float]
    has_page_numbers_header_footer: bool


def _clean_font_name(font_name: str) -> str:
    return font_name.split("+", 1)[-1] if "+" in font_name else font_name


def _collect_page_lines(page: fitz.Page) -> List[dict]:
    lines: List[dict] = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            spans = [s for s in l.get("spans", []) if s.get("text", "")]
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            lines.append({
                "bbox": tuple(l["bbox"]),
                "spans": spans,
                "text": text,
                "strip": text.strip(),
            })
    return lines


def analyze_pdf_bytes(pdf_bytes: bytes) -> PdfTelemetry:
    if not pdf_bytes:
        raise ValueError("Empty PDF bytes provided to analyze_pdf_bytes.")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)
        page_sizes = [(float(p.rect.width), float(p.rect.height)) for p in doc]
        pages_text = [p.get_text("text") for p in doc]
        full_text = "\n".join(pages_text)

        fonts_base: Set[str] = set()
        cjk_chars: List[Tuple[int, str]] = []
        pua_chars: List[Tuple[int, str]] = []
        orphaned_punct: List[Tuple[int, str, float, float]] = []
        wrapped_eq_numbers: List[Tuple[int, str, Tuple[float, float, float, float]]] = []
        table_small_spans: List[Tuple[int, float, str]] = []
        figure_spans: List[Tuple[int, float, str]] = []
        has_header_footer_pagenum = False

        for pno, page in enumerate(doc):
            page_num = pno + 1
            for f in page.get_fonts(full=True):
                if len(f) > 3 and f[3]:
                    fonts_base.add(_clean_font_name(f[3]))

            lines = _collect_page_lines(page)

            # Check running headers/footers (IEEEtran conference mode must omit page numbers)
            for ln in lines:
                x0, y0, x1, y1 = ln["bbox"]
                if (y0 < 36.0 or y1 > page.rect.height - 36.0) and re.fullmatch(r"\d+", ln["strip"]):
                    first_font = _clean_font_name(ln["spans"][0]["font"]).lower() if ln["spans"] else ""
                    if not any(k in first_font for k in ("dejavu", "arial", "helvetica", "liberation")):
                        has_header_footer_pagenum = True

            # Scan CJK / PUA / figure spans
            for ln in lines:
                for sp in ln["spans"]:
                    t = sp["text"]
                    fn = _clean_font_name(sp["font"])
                    if not t.strip():
                        continue
                    fonts_base.add(fn)
                    for m in CJK_RE.finditer(t):
                        cjk_chars.append((page_num, m.group(0)))
                    for m in PUA_RE.finditer(t):
                        pua_chars.append((page_num, hex(ord(m.group(0)))))
                    if any(k in fn.lower() for k in ("dejavu", "arial", "helvetica", "liberation")):
                        figure_spans.append((page_num, float(sp["size"]), t.strip()))

            # Baseline-aware orphaned citation punctuation detector
            # Sort lines per column top-to-bottom to inspect true visual vertical predecessor
            mid_x = page.rect.width / 2.0
            for col_idx, col_lines in enumerate([
                sorted([l for l in lines if l["bbox"][2] <= mid_x + 8.0], key=lambda l: l["bbox"][1]),
                sorted([l for l in lines if l["bbox"][0] >= mid_x - 8.0], key=lambda l: l["bbox"][1]),
            ]):
                col_left = 49.0 if col_idx == 0 else 312.0
                for idx, ln in enumerate(col_lines):
                    x0, y0, x1, y1 = ln["bbox"]
                    st = ln["strip"]
                    if abs(x0 - col_left) <= 12.0 and st and st[0] in (".", ",") and not st.startswith("..."):
                        # Check if previous visual line in same column ended with citation bracket ']'
                        if idx > 0:
                            prev = col_lines[idx - 1]
                            if (y0 - prev["bbox"][1]) > 4.0 and (y0 - prev["bbox"][3]) < 20.0:
                                if prev["strip"].endswith("]"):
                                    orphaned_punct.append((page_num, st[:12], x0, y0))

            # Equation number line-wrapping detector:
            # Right-aligned equation number '^(N)$' in left or right column must vertically overlap
            # at least one non-eqnum line in the same column.
            for idx, ln in enumerate(lines):
                st = ln["strip"]
                if not re.fullmatch(r"\(\d+\)", st):
                    continue
                x0, y0, x1, y1 = ln["bbox"]
                in_left_col_edge = (270.0 <= x0 <= 305.0 and x1 <= 312.0)
                in_right_col_edge = (530.0 <= x0 <= 568.0 and x1 <= 575.0)
                if not (in_left_col_edge or in_right_col_edge):
                    continue
                overlaps = False
                for jdx, other in enumerate(lines):
                    if jdx == idx:
                        continue
                    ox0, oy0, ox1, oy1 = other["bbox"]
                    same_col = (ox1 <= 312.0) if in_left_col_edge else (ox0 >= 305.0)
                    if not same_col:
                        continue
                    if min(y1, oy1) - max(y0, oy0) > 2.0:
                        overlaps = True
                        break
                if not overlaps:
                    wrapped_eq_numbers.append((page_num, st, (x0, y0, x1, y1)))

            # Table body font size detector:
            # Locate Table captions ("Table N" or "TABLE N") and inspect tabular body rows below them.
            table_captions = [
                ln for ln in lines
                if re.match(r"^(?:Table|TABLE)\s+(?:\d+|[IVX]+)\b", ln["strip"])
                and not re.match(r"^(?:Table|TABLE)\s+(?:\d+|[IVX]+)\s+[a-z]", ln["strip"])
            ]
            for cap in table_captions:
                cx0, cy0, cx1, cy1 = cap["bbox"]
                is_wide = (cx0 < mid_x - 20.0 and cx1 > mid_x + 20.0) or (cx0 < mid_x and cx1 > mid_x)
                cand_lines = []
                for ln in lines:
                    lx0, ly0, lx1, ly1 = ln["bbox"]
                    if ly0 <= cy1 or ly0 > cy1 + 320.0:
                        continue
                    if not is_wide:
                        if cx1 <= mid_x + 10.0 and lx0 > mid_x + 5.0:
                            continue
                        if cx0 >= mid_x - 10.0 and lx1 < mid_x - 5.0:
                            continue
                    cand_lines.append(ln)

                # Group candidate table lines by baseline round(y0, 1) to check line-level max_sz
                rows_by_y: Dict[float, List[dict]] = {}
                for ln in cand_lines:
                    rows_by_y.setdefault(round(ln["bbox"][1], 1), []).append(ln)

                table_rows = []
                for y_key in sorted(rows_by_y.keys()):
                    row_lines = rows_by_y[y_key]
                    if any(re.match(r"^(?:\*?Note:|Figure\s+\d+|Table\s+\d+|TABLE\s+)", ln["strip"]) for ln in row_lines):
                        break
                    row_spans = [sp for ln in row_lines for sp in ln["spans"] if sp.get("text", "").strip()]
                    if not row_spans:
                        continue
                    row_max_sz = max(float(sp["size"]) for sp in row_spans)
                    if row_max_sz >= 9.0:
                        break
                    ry0 = min(ln["bbox"][1] for ln in row_lines)
                    ry1 = max(ln["bbox"][3] for ln in row_lines)
                    row_origins = [float(sp["origin"][1]) for sp in row_spans if "origin" in sp]
                    table_rows.append((y_key, ry0, ry1, row_origins, row_lines, row_spans, row_max_sz))

                ok_rows = [r for r in table_rows if r[6] >= 6.8]
                for y_key, ry0, ry1, row_origins, row_lines, row_spans, row_max_sz in table_rows:
                    if row_max_sz >= 6.8:
                        continue
                    row_txt = " ".join(ln["strip"] for ln in row_lines).strip()
                    # Exempt stray short math subscript/superscript lines overlapping a >= 6.8 pt row
                    if len(row_txt) <= 4:
                        overlaps_ok = any(
                            (min(ry1, oy1) - max(ry0, oy0) > 1.0)
                            or any(abs(o1 - o2) <= 3.0 for o1 in row_origins for o2 in o_origins)
                            for _, oy0, oy1, o_origins, _, _, _ in ok_rows
                        )
                        if overlaps_ok:
                            continue
                    for sp in row_spans:
                        fn = _clean_font_name(sp["font"])
                        if any(k in fn.lower() for k in ("dejavu", "arial", "helvetica")):
                            continue
                        txt = sp["text"].strip()
                        if len(txt) >= 3 and re.search(r"[A-Za-z0-9]", txt):
                            table_small_spans.append((page_num, float(row_max_sz), txt[:30]))

        # Last-page column balance delta
        last_page = doc[-1]
        last_lines = _collect_page_lines(last_page)
        mid_x = last_page.rect.width / 2.0
        left_ys = [l["bbox"][3] for l in last_lines if l["bbox"][2] <= mid_x + 8.0]
        right_ys = [l["bbox"][3] for l in last_lines if l["bbox"][0] >= mid_x - 8.0]
        col_diff = abs(max(left_ys) - max(right_ys)) if (left_ys and right_ys) else None

    unresolved_refs = re.findall(r"(?:\?\?|\[\?\])", full_text)
    lower_fonts = {f.lower() for f in fonts_base}
    has_times = any(
        any(tok in f for tok in ("times", "nimbusrom", "texgyretermes", "ptm"))
        for f in lower_fonts
    )
    has_mono = any(
        any(tok in f for tok in ("courier", "nimbusmon", "texgyrecursor", "lmmono", "pcr", "cmtt"))
        for f in lower_fonts
    )
    has_lmroman = any("lmroman" in f for f in lower_fonts)

    return PdfTelemetry(
        page_count=page_count,
        page_sizes=page_sizes,
        full_text=full_text,
        pages_text=pages_text,
        fonts_base=fonts_base,
        has_times_font=has_times,
        has_mono_font=has_mono,
        has_lmroman_fallback=has_lmroman,
        cjk_chars=cjk_chars,
        pua_chars=pua_chars,
        unresolved_refs=unresolved_refs,
        orphaned_punct=orphaned_punct,
        wrapped_eq_numbers=wrapped_eq_numbers,
        table_small_spans=table_small_spans,
        figure_spans=figure_spans,
        last_page_col_diff_pt=col_diff,
        has_page_numbers_header_footer=has_header_footer_pagenum,
    )


def png_pixel_size(png_bytes: bytes) -> Tuple[int, int]:
    if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG header")
    w, ht = struct.unpack(">II", png_bytes[16:24])
    return int(w), int(ht)


def analyze_pdf_file(path: Path) -> PdfTelemetry:
    return analyze_pdf_bytes(path.read_bytes())


def collect_all_pdf_spans(pdf_bytes: bytes) -> List[Tuple[int, float, str, str]]:
    """Return [(page_num, size_pt, clean_font, text_strip), ...] across all non-empty spans."""
    if not pdf_bytes:
        raise ValueError("Empty PDF bytes provided to collect_all_pdf_spans.")
    spans_out: List[Tuple[int, float, str, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for pno, page in enumerate(doc):
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                for l in b.get("lines", []):
                    for sp in l.get("spans", []):
                        txt = sp.get("text", "").strip()
                        if not txt:
                            continue
                        spans_out.append((pno + 1, float(sp["size"]), _clean_font_name(sp["font"]), txt))
    return spans_out


def find_tiny_figure_spans(
    spans: Sequence[Tuple[int, float, str]],
    min_body_pt: float = 4.8,
    min_sub_pt: Optional[float] = None,
) -> List[Tuple[int, float, str]]:
    """
    Check figure span legibility while exempting short math subscripts (len <= 3)
    from the full body-label threshold (or enforcing min_sub_pt when specified).
    """
    tiny: List[Tuple[int, float, str]] = []
    for pg, sz, txt in spans:
        st = txt.strip()
        if not st:
            continue
        if len(st) > 3:
            if sz < min_body_pt:
                tiny.append((pg, sz, st))
        elif min_sub_pt is not None and sz < min_sub_pt:
            tiny.append((pg, sz, st))
    return tiny


def locate_figure_pages(tel: PdfTelemetry) -> Tuple[Set[int], Set[int]]:
    """Dynamically locate Figure 1 and Figure 2 page numbers (1-indexed) in main.pdf."""
    spans_by_page: Dict[int, List[Tuple[float, str]]] = {}
    for pg, sz, txt in tel.figure_spans:
        spans_by_page.setdefault(pg, []).append((sz, txt))

    fig_pages_ordered = sorted(spans_by_page.keys())
    fig1_sig = ("stage 1:", "ontology", "deterministic verifier", "hard fsm")
    fig2_sig = ("(a) field agreement", "(b) architectural policy", "(c) compliance loss", "field agreement")

    fig1_pages: Set[int] = set()
    fig2_pages: Set[int] = set()

    for pg in fig_pages_ordered:
        joined = " ".join(t.lower() for _, t in spans_by_page[pg])
        if any(sig in joined for sig in fig1_sig):
            fig1_pages.add(pg)
        if any(sig in joined for sig in fig2_sig):
            fig2_pages.add(pg)

    if not fig1_pages:
        for pg in fig_pages_ordered:
            pt = tel.pages_text[pg - 1] if 1 <= pg <= len(tel.pages_text) else ""
            if re.search(r"(?m)^\s*Figure\s+1\s*[\.:]|\bFigure\s+1\.", pt):
                fig1_pages.add(pg)
    if not fig2_pages:
        for pg in fig_pages_ordered:
            pt = tel.pages_text[pg - 1] if 1 <= pg <= len(tel.pages_text) else ""
            if re.search(r"(?m)^\s*Figure\s+2\s*[\.:]|\bFigure\s+2\.", pt):
                fig2_pages.add(pg)

    if not fig1_pages and len(fig_pages_ordered) >= 1:
        fig1_pages.add(fig_pages_ordered[0])
    if not fig2_pages and len(fig_pages_ordered) >= 2:
        rem = [p for p in fig_pages_ordered if p not in fig1_pages]
        if rem:
            fig2_pages.add(rem[0])

    return fig1_pages, fig2_pages
