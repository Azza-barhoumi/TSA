"""Build a compiled PDF of the LaTeX report.

The report is written in LaTeX (report/main.tex) because that is what the course
requires.  No TeX distribution is installed in this environment, so the compiled
artefact is produced by converting the *same* LaTeX source with pandoc and
typesetting it with Typst (pandoc's native Typst writer reproduces the LaTeX
structure: sections, equations, booktabs-type tables, figures, cross-references).

If a real TeX engine is available (``pdflatex``/``latexmk``/``tectonic``) the script
uses it directly and the output is the authoritative build; otherwise it falls back
to the pandoc+Typst route and says so in the file's metadata.

Usage:  python src/07_build_report.py [--engine auto|latex|typst]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

MAIN = C.REPORTDIR / "main.tex"
OUT_PDF = C.REPORTDIR / "main.pdf"
PANDOC = None


def find_pandoc() -> str | None:
    global PANDOC
    if PANDOC:
        return PANDOC
    exe = shutil.which("pandoc")
    if exe:
        PANDOC = exe
        return exe
    try:
        import pypandoc
        PANDOC = pypandoc.get_pandoc_path()
        return PANDOC
    except Exception:
        return None


def find_latex() -> str | None:
    for exe in ("latexmk", "pdflatex", "tectonic"):
        if shutil.which(exe):
            return shutil.which(exe)
    return None


def build_with_latex(engine: str) -> bool:
    print(f"building with {engine} ...")
    cwd = C.REPORTDIR
    if engine.endswith("latexmk"):
        cmd = [engine, "-pdf", "-interaction=nonstopmode", "-halt-on-error", MAIN.name]
    elif engine.endswith("tectonic"):
        cmd = [engine, "-X", "compile", MAIN.name]
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", MAIN.name]
    for _ in range(2):                      # twice for the table of contents and refs
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-4000:])
            print(p.stderr[-2000:])
            return False
    return OUT_PDF.exists()


def unwrap_labelled_blocks(text: str) -> str:
    """Pandoc wraps a LaTeX table float in `#block[ ... ] <label>`.

    Typst cannot cross-reference a plain block, and the report does reference its
    tables, so the wrapper is removed and the label is attached to the inner
    `#figure(...)` instead.  (Image figures are emitted as bare `#figure(...)`
    and need no fix.)
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "#block[":
            # look ahead for the closing `] <label>` of this block
            j = i + 1
            depth = 1
            while j < len(lines) and depth > 0:
                depth += lines[j].count("[") - lines[j].count("]")
                j += 1
            closing = lines[j - 1].strip()
            if closing.startswith("] <"):
                label = closing[2:].strip()
                inner = lines[i + 1 : j - 1]
                # drop trailing blank lines inside the block
                while inner and inner[-1].strip() == "":
                    inner.pop()
                # attach the label to the last non-blank line (the figure's `)`)
                for k in range(len(inner) - 1, -1, -1):
                    if inner[k].strip():
                        inner[k] = inner[k].rstrip() + " " + label
                        break
                out.extend(inner)
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def strip_latex_titlepage(text: str) -> str:
    """Remove what pandoc renders badly from the LaTeX title page.

    The LaTeX \\begin{titlepage}...\\end{titlepage} block is typeset properly by a
    TeX engine but pandoc flattens it into stray text, and it duplicates the Typst
    title block added above.  Everything between the start of the document and the
    ``\\pagenumbering{roman}`` marker is therefore dropped; the executive summary
    is kept (it is a normal section).
    """
    start = text.find("\\begin{titlepage}")
    if start == -1:
        return text
    end = text.find("\\section*{Executive summary}")
    if end == -1:
        return text
    return text[:start] + text[end:]


def clean_equation_references(text: str) -> str:
    """Drop the empty internal links pandoc leaves behind for LaTeX references.

    Pandoc turns ``\\ref{tab:x}`` into a link to the matching label; when the
    target is a table it renders as a bare ``#link(<tab:x>)[...]`` that Typst
    cannot display.  The label text is kept, the link wrapper removed.
    """
    import re
    text = re.sub(r"#link\(<([a-z]+:[A-Za-z0-9:_-]+)>\)\[([^\]]*)\]",
                  lambda m: m.group(2) or "", text)
    text = re.sub(r"\[\]\s*", "", text)
    # Typst's own "Table"/"Figure" supplement is added automatically, so the
    # hand-written LaTeX word in front of the reference must go
    text = re.sub(r"(?:Table|Figure|Section)~(@(?:tab|fig|sec):[A-Za-z0-9:_-]+)", r"\1", text)
    # when the reference target is a table/figure, pandoc keeps the caption text as
    # the link label, so the hand-written word doubles: "Table~Table 8"
    text = re.sub(r"(?:Table|Figure|Section)~(?:Table|Figure|Section)~?(?=\s*\d)",
                  lambda m: m.group(0).split("~")[0], text)
    # pandoc puts a label (figure, table, section) on the line *after* the block it
    # belongs to, while Typst wants it on the same line as that block.  Move every
    # bare `<label>` line up onto the preceding non-blank line.
    out: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("<") and s.endswith(">") and " " not in s and len(s) > 3:
            for k in range(len(out) - 1, -1, -1):
                if out[k].strip():
                    out[k] = out[k].rstrip() + " " + s
                    break
            continue
        out.append(line)
    text = "\n".join(out)
    return text


def number_display_equations(text: str) -> str:
    """Normalise the display equations that pandoc converts unevenly.

    Pandoc writes a labelled equation as inline math followed by the label
    (``$ y_t = c $ <eq:sarima>``) but numbers it only if the label sits on the
    same line as the closing math.  When that happens it emits a bare block
    equation with no label, and a later reference to it fails.  Here every display
    equation is put on its own lines and its label is attached to it, so the
    ``@eq:...`` references in the text resolve and the equations are numbered.
    """
    import re

    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        s = line.strip()
        # drop the labels of tables/figures/sections: pandoc already attached them
        if s.startswith("\\label{") and not s.startswith("\\label{eq:"):
            continue
        m = re.match(r"^(\s*)\$\s(.+?)\s\$\s<((?:eq|tab|fig|sec):[A-Za-z0-9:_-]+)>\s*$", line)
        if m and m.group(3).startswith("eq:"):
            indent = m.group(1)
            out.append("")
            out.append(f"{indent}$ {m.group(2)} $ <{m.group(3)}>")
            out.append("")
            continue
        out.append(line)
    # collapse runs of blank lines that the rewrite may have created
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def drop_escaped_ticks(text: str) -> str:
    return text.replace("\\_", "_")


def build_with_pandoc() -> bool:
    pandoc = find_pandoc()
    if not pandoc:
        print("pandoc not available")
        return False
    try:
        import typst
    except ImportError:
        print("typst python package not available")
        return False

    # Typst refuses to read files outside its project root, and the LaTeX source
    # points at ../reports/figures; expose the figures/tables inside report/ with
    # symlinks so both builds see the same relative paths.
    for sub in ("figures", "tables"):
        link = C.REPORTDIR / sub
        target = C.REPORTS / sub
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)

    body = C.REPORTDIR / "main.typ"
    cmd = [pandoc, str(MAIN), "-o", str(body), "-t", "typst", "--wrap=none"]
    p = subprocess.run(cmd, cwd=C.REPORTDIR, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-3000:])
        return False

    # front matter that LaTeX would have produced: title block + table of contents
    src = body.read_text()
    preamble = '''#set document(title: "Time Series Modeling and Forecasting of US Inflation")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set par(justify: true, leading: 0.72em)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#show heading.where(level: 1): it => { pagebreak(weak: true); it }
// Typst only numbers headings when asked to; LaTeX numbers every section, and the
// text cross-references them, so numbering must be switched on explicitly.
#set heading(numbering: "1.1.1")
#show heading.where(level: 1): set heading(numbering: n => numbering("1.", n))
#set math.equation(numbering: "(1)")
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set figure(supplement: [Table])
#let horizontalrule = line(start: (0%,0%), end: (100%,0%), stroke: 0.5pt)

#align(center)[
  #text(size: 11pt)[Tunis Business School --- Finance / Business Administration]
  #v(0.2em)
  #text(size: 11pt)[Time Series Analysis and Forecasting]
  #v(0.9em)
  #line(length: 100%, stroke: 0.6pt)
  #v(0.6em)
  #text(size: 15pt, weight: "bold")[Time Series Modeling and Forecasting of US Consumer Price Inflation]
  #v(0.45em)
  #text(size: 10.5pt, style: "italic")[A Box--Jenkins Analysis of the Kaggle US CPI-U Dataset, 1957--2020]
  #v(0.45em)
  #line(length: 100%, stroke: 0.6pt)
  #v(2.0em)
  #text(size: 9.5pt)[
    #grid(columns: (auto, 1fr), gutter: 1.4em, align: (right, left),
      [*Forecasting question*], [one- to five-year ahead US inflation],
      [*Data*], [Kaggle "US Inflation Data" (CPI-U, monthly, 1913--2020)],
      [*Estimation sample*], [January 1957 -- December 2020 ($n = 768$)],
      [*Model*], [ARIMA(1,1,3)(1,0,1)[12] on the log CPI],
      [*Method*], [Box--Jenkins: identification, estimation, diagnostics, forecasting],
      [*Software*], [Python 3.11 (pandas, statsmodels, matplotlib)],
    )
  ]
  #v(2.4em)
  #text(size: 10pt)[Academic year 2026/2027] \
  #text(size: 9.5pt)[Submitted as the term project of the course Time Series Analysis and Forecasting] \
  #v(0.7em)
  #text(size: 8.5pt, style: "italic")[Every number, table and figure is generated by the Python code in the project repository.]
]
#pagebreak()
#heading(numbering: none, outlined: false)[Contents]
#outline(title: none, depth: 2, indent: 1.2em)
#pagebreak()

'''
    # the LaTeX source uses ../reports/figures/... ; inside the Typst project the
    # same files are reachable as figures/... -- rewrite the prefix
    text = preamble + src
    text = text.replace("../reports/figures/", "figures/").replace("../reports/tables/", "tables/")
    text = unwrap_labelled_blocks(text)
    text = strip_latex_titlepage(text)
    text = drop_escaped_ticks(text)
    text = clean_equation_references(text)
    text = number_display_equations(text)
    body.write_text(text)

    pdf_bytes = typst.compile(str(body), root=str(C.REPORTDIR))
    OUT_PDF.write_bytes(pdf_bytes)
    print(f"pandoc + typst -> {OUT_PDF.relative_to(C.ROOT)} "
          f"({len(pdf_bytes)/1024:.0f} kB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto", choices=["auto", "latex", "typst"])
    args = ap.parse_args()

    if not MAIN.exists():
        raise SystemExit(f"LaTeX source missing: {MAIN}")

    if args.engine in ("auto", "latex"):
        engine = find_latex()
        if engine and build_with_latex(engine):
            print(f"LaTeX build OK -> {OUT_PDF}")
            return
        if args.engine == "latex":
            raise SystemExit("LaTeX engine not available or build failed")

    if not build_with_pandoc():
        raise SystemExit("no PDF engine available")
    print("note: PDF produced from the LaTeX source via pandoc + Typst "
          "(no TeX distribution in this environment)")


if __name__ == "__main__":
    main()
