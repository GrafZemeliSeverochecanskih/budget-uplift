# Paper

Two-column LaTeX write-up of releases v1.0 to v1.5.

- `main.tex`: the paper (pdfLaTeX + BibTeX; `article`, two columns, APA author-year citations via `apacite`
  with the `natbibapa` option, so `\citet`/`\citep` work; style `apacite`).
- `references.bib`: bibliography. Page ranges are written `1148{-}1178` so `apacite` prints a plain hyphen.
- `figures/`: copies of the figures in `results/figures/`.

**Overleaf:** New Project → Upload Project → choose a zip of this folder. The defaults (pdfLaTeX, BibTeX) compile it as is.
