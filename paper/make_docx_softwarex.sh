#!/usr/bin/env bash
# Build a Word (.docx) from softwarex.tex (Paper B) via pandoc.
#   - figure PDFs -> figures/*.png; bake Figure N into captions; resolve \ref->num
#   - numeric [N] citations via numeric.csl + refs.bib
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

python3 - <<'PY'
import re, pathlib
src = pathlib.Path("softwarex.tex").read_text()

# custom macro -> texttt
src = re.sub(r"\\code\{([^}]*)\}", r"\\texttt{\1}", src)

# figure paths: vector PDF -> figures/<name>.png
src = re.sub(r"\{([a-z][a-z0-9_]+)\.pdf\}", r"{figures/\1.png}", src)
# avoid pandoc wrapping centered narrow figures in layout tables
src = src.replace("\\begin{figure}[t]\n\\centering", "\\begin{figure}[t]")
src = src.replace(r"width=0.96\linewidth", "width=16cm")
src = src.replace(r"width=0.74\linewidth", "width=12cm")

# bake canonical figure numbers into captions (pandoc does not auto-number)
caps = {
    "Closed-loop \\emph{reference} architecture implemented": "1",
    "Mechanism illustration (in-silico)": "2",
    "Validated re-implementations of two published": "3",
    "Built-in falsification (the package's central result)": "4",
}
for phrase, num in caps.items():
    src = src.replace(r"\caption{" + phrase, r"\caption{\textbf{Figure %s.} %s" % (num, phrase))

# resolve cross-references (pandoc does not process LaTeX \ref)
src = src.replace(r"\ref{tab:meta}", "1")
for key, num in {"arch": "1", "toy": "2", "osc": "3", "neg": "4"}.items():
    src = src.replace(r"\ref{fig:%s}" % key, num)
src = src.replace(r"\S\ref{sec:desc}", "Section 2").replace(r"\S\ref{sec:examples}", "Section 3")
src = src.replace(r"\ref{sec:desc}", "2").replace(r"\ref{sec:examples}", "3")

# hand bibliography to citeproc
src = re.sub(r"\\bibliographystyle\{[^}]*\}\n?", "", src)
src = src.replace(r"\bibliography{refs}", r"\section*{References}")

pathlib.Path("softwarex_docx.tex").write_text(src)
print("wrote softwarex_docx.tex")
PY

pandoc softwarex_docx.tex \
  --from=latex+raw_tex \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --resource-path=.:figures \
  -o softwarex.docx
echo "wrote softwarex.docx"
