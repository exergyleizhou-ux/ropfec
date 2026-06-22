#!/usr/bin/env bash
# Build a Word (.docx) from ropfec.tex: render-swap the inline TikZ architecture
# diagram for its rendered image (pandoc cannot convert TikZ), resolve numeric
# citations via citeproc. Figures embedded as PNG; vector copies are figures/*.pdf.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

python3 - <<'PY'
import re, pathlib
src = pathlib.Path("ropfec.tex").read_text()
# swap the inline TikZ diagram for the rendered image
src = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
             r"\\includegraphics[width=15cm]{figures/ropfec_architecture.png}",
             src, flags=re.S)
# avoid pandoc wrapping centered figures in layout tables: drop \centering in floats
src = re.sub(r"(\\begin\{figure\}[^\n]*)\n\\centering", r"\1", src)
src = src.replace(r"width=0.72\textwidth", "width=12cm")
src = src.replace(r"width=0.95\textwidth", "width=15cm")
# hand bibliography to citeproc
src = re.sub(r"\\bibliographystyle\{[^}]*\}\n?", "", src)
src = src.replace(r"\bibliography{refs}", r"\section*{References}")
pathlib.Path("ropfec_docx.tex").write_text(src)
print("wrote ropfec_docx.tex")
PY

pandoc ropfec_docx.tex \
  --from=latex+raw_tex \
  --citeproc \
  --bibliography=refs.bib \
  --csl=numeric.csl \
  --resource-path=.:figures \
  -o ropfec.docx
echo "wrote ropfec.docx"
