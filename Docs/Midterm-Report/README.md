# Midterm Report

- `latex/` holds the report source (`main.tex`, `references.bib`, ACL style files, `figures/`).
- `literature-review/` holds one evidence sheet per cited paper.

## Building the PDF

`latex/main.pdf` is committed, so you can read the report without building. `main.pdf` and `main.bbl` are build outputs; after editing `main.tex` or `references.bib`, regenerate them in `latex/` one of two ways and commit them with your change.

**VS Code.** Install a TeX distribution (MacTeX or TeX Live) and the LaTeX Workshop extension (`james-yu.latex-workshop`). Open `latex/main.tex` and save it, or run "LaTeX Workshop: Build LaTeX project" from the command palette. The default `latexmk` recipe runs pdfLaTeX and BibTeX and writes `main.pdf` and `main.bbl` next to `main.tex`.

**Terminal.** The same build, without VS Code:

```bash
cd Docs/Midterm-Report/latex && latexmk -pdf main.tex
```

To remove the intermediate files afterwards (keeps `main.pdf` and `main.bbl`):

```bash
cd Docs/Midterm-Report/latex && latexmk -c
```
