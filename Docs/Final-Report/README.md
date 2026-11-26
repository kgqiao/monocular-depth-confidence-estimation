# Final Report

The final report of the project, due **Monday, December 7, 2026**. It is written as a research paper: 8 pages of body in the ACL template, then references, an appendix, and a one-page contribution statement.

- [rubric.md](rubric.md): the team's own rubric. The course gives none, so this is what the report is checked against before it is submitted.
- `final-report.pdf`: the built report. It does not exist yet; it is added on `main` once the first draft builds (see the last section).

## Layout

- `latex/main.tex`: preamble, title, teaser figure, and one `\input` per section. Rarely needs editing.
- `latex/sections/`: one file per section. Edit your own.
- `latex/references.bib`, `latex/figures/`: shared bibliography and figures.
- `figures-src/`: one script per figure, with the data it reads. See [figures-src/README.md](figures-src/README.md).

| File in `latex/sections/` | Section | Pages |
|---|---|---|
| `abstract.tex` | Abstract | – |
| `introduction.tex` | 1 Introduction | 0.9 |
| `related-work.tex` | 2 Related Work | 0.8 |
| `method.tex` | 3 Method | 1.5 |
| `setup.tex` | 4 Experimental Setup | 1.2 |
| `results.tex` | 5 Results | 2.3 |
| `discussion.tex` | 6 Discussion and Limitations | 0.6 |
| `conclusion.tex` | 7 Conclusion and Future Work | 0.3 |
| `appendix.tex` | A Deviations, B Run matrix, C Cache, D Additional results, E Reproducibility | not counted |
| `contributions.tex` | Contribution Statement | 1 |

The page budgets add up to 7.6 pages, which leaves room for the teaser figure. A section that runs over its budget moves material to the appendix; it does not take pages from another section.

## Where the Midterm Text Went

The skeleton was built from the midterm report. Its prose was moved, not rewritten, wherever it still holds.

| Midterm section | Now in |
|---|---|
| Introduction | `introduction.tex`, with the research questions and a contributions list added |
| Literature Review | `related-work.tex` |
| Progress: Method as Implemented | `method.tex` (3.2 to 3.4); the training settings are in `setup.tex` (4.2) |
| Progress: Data and Infrastructure | `appendix.tex` (C), with a summary in `setup.tex` (4.1) |
| Progress: Training Runs and Tracking | `appendix.tex` (B, E) |
| Progress: Evaluation Code | `setup.tex` (4.1, 4.3, 4.4) |
| Progress: Deviations from the Proposal | `appendix.tex` (A) |
| Results: Cache and Label Statistics | `appendix.tex` (C) |
| Results: Training Runs | `results.tex` (5.3), with the dev losses of the complete cache |
| Updated Plan | removed; its future-work items are in `conclusion.tex`, its risks that still hold in `discussion.tex` |

## Open Items in the Text

Three red markers show what is still missing. The submitted report has none of them.

- `\todo{what is missing}`: text or a number to write.
- `\tbd`: a table cell whose number is not measured yet. It prints a red question mark.
- `\placeholderfig{width}{height}{what goes here}`: a framed box in place of a figure.

To list them all, from `latex/`:

```bash
grep -n '\\todo\|\\tbd\|\\placeholderfig' sections/*.tex main.tex
```

Each results subsection also starts with a comment block that says what the subsection must show and which script gives the numbers. Delete the comment when the subsection is written.

Two rules for numbers:

- A number goes into the report only from the complete cache (204,894 entries, heads trained on 2026-11-17). A number of the partial cache is either refreshed or removed. The ones still in the text are wrapped in a `\todo` that says "refresh".
- All methods of one table are scored on the same validation images and the same labeled cells, and the caption gives the counts.

## Workflow

1. Branch from `main`, named `final-report/<your-name>`:

   ```bash
   git fetch --prune && git switch -c final-report/<your-name> origin/main
   ```

2. Edit your section file, plus `references.bib`, `figures/`, or `figures-src/` if you need them.
3. Open a pull request against `main`.

Two rules:

- Do not create a branch named `final-report`. Git cannot hold both that and the `final-report/<name>` branches.
- Do not commit `final-report.pdf` or build outputs from a working branch. The outputs in `latex/` are ignored by git; the PDF is refreshed on `main` only, so pull requests do not conflict on it.

## Building the PDF Locally

Install a TeX distribution (MacTeX or TeX Live). The VS Code extension alone is not enough: it calls `latexmk`, which the distribution provides. Then either:

- **VS Code:** install the LaTeX Workshop extension (`james-yu.latex-workshop`), open `latex/main.tex` or any section file, and save. Each section file points back to `main.tex`, so the whole report builds.
- **Terminal:**

  ```bash
  cd Docs/Final-Report/latex && latexmk -pdf main.tex
  ```

The result is `latex/main.pdf`, which git ignores.

LaTeX Workshop rebuilds `main.pdf` on its own whenever a source file changes, once it has built the report once (its `latex-workshop.latex.autoBuild.run` setting, `onFileChange` by default).

`main.tex` uses the `review` option of the template, which prints line numbers; the rubric review cites them. For the submitted copy, change `review` to `preprint`.

## Refreshing the Shared PDF (Whoever Merges a Pull Request)

After merging into `main`, on an up-to-date `main`:

```bash
cd Docs/Final-Report/latex && latexmk -pdf main.tex && cp main.pdf ../final-report.pdf
```

Then commit `Docs/Final-Report/final-report.pdf` and push.
