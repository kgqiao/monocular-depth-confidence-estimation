# Midterm Report

To read the report, open [midterm-report.pdf](midterm-report.pdf). No build needed.

## Layout

- `midterm-report.pdf`: the built report, refreshed after each merge (see below).
- `latex/main.tex`: preamble, title, teaser figure, and one `\input` per section. Rarely needs editing.
- `latex/sections/`: one file per section. Edit your own.
- `latex/references.bib`, `latex/figures/`: shared bibliography and figures.
- `literature-review/`: one evidence sheet per cited paper. Their line references to the section files were accurate when the files were split and drift as sections are edited.

| File in `latex/sections/` | Section | Owner |
|---|---|---|
| `abstract.tex` | Abstract | all |
| `introduction.tex` | 1 Introduction | Arteaga |
| `literature-review.tex` | 2 Literature Review | Arteaga |
| `progress.tex` | 3 Progress So Far | Aday, Bao (deviations table: Arteaga) |
| `results.tex` | 4 Initial Results | Tawde, Qiao, with Aday and Bao |
| `plan.tex` | 5 Updated Plan | Arteaga, each item's owner named |

## Workflow

All report work goes through the `midterm-report` branch. It merges into `main` once, when the report is final.

1. Branch from `midterm-report`, named `midterm/<your-name>`:

   ```bash
   git fetch --prune && git switch -c midterm/<your-name> origin/midterm-report
   ```

2. Edit your section file, plus `references.bib` or `figures/` if you need them.
3. Open a pull request with base `midterm-report`. GitHub picks `main` by default, so set the base:

   ```bash
   gh pr create --base midterm-report
   ```

Two rules:

- Do not name a branch `midterm-report/<anything>`. Git cannot hold both that and the `midterm-report` branch.
- Do not commit `midterm-report.pdf` or build outputs from a working branch. The outputs in `latex/` are ignored by git; the PDF is refreshed on `midterm-report` only, so PRs do not conflict on it.

## Building the PDF locally

Install a TeX distribution (MacTeX or TeX Live), then either:

- **VS Code:** install the LaTeX Workshop extension (`james-yu.latex-workshop`), open `latex/main.tex` or any section file, and save. Each section file points back to `main.tex`, so the whole report builds.
- **Terminal:**

  ```bash
  cd Docs/Midterm-Report/latex && latexmk -pdf main.tex
  ```

The result is `latex/main.pdf`, which git ignores.

## Refreshing the shared PDF (whoever merges a PR)

After merging into `midterm-report`, on an up-to-date `midterm-report`:

```bash
cd Docs/Midterm-Report/latex && latexmk -pdf main.tex && cp main.pdf ../midterm-report.pdf
```

Then commit `Docs/Midterm-Report/midterm-report.pdf` and push.
