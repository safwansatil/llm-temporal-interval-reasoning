# Evaluating Temporal Reasoning in Large Language Models

**Event Temporal Relation Extraction with Allen's Interval Algebra, Logical Consistency, and Relative Time in Narratives**

Anonymous submission to **COLING 2027** (ACL style, long paper).

---

## Overview

Reading a narrative means knowing how every event relates in time to every other. Most
LLM evaluations test isolated event pairs or coarse labels. This paper asks five LLMs to
reconstruct the **complete network** of Allen's 13 interval relations for 360 English
narratives whose events are placed relative to one another rather than by dates.

Each model receives a passage and its event list and must return the full *n x n* relation
matrix. Outputs are read three complementary ways: the upper triangle is scored against
gold, the lower triangle gives an independent converse reading of every pair, and an
endpoint encoding of each network yields an exact satisfiability test and a consistent
repair.

## Main findings

- **Best model beats trivial baselines on macro-F1, but not on accuracy.** DeepSeek-V4.1-Flash
  reaches 0.270 macro-F1 (13-way, missing cells counted as wrong) and 0.452 accuracy, while
  always answering `before` reaches 0.607 accuracy and 0.058 macro-F1.
- **Output fidelity is a first-order bottleneck.** Models answer only 2.4%-60.0% of cells,
  mostly because matrix rows have one or two symbols too many or too few. Coverage collapses
  on passages with more than 40 events.
- **On answered cells, the picture reverses.** DeepSeek (0.753) and GPT-6-Luna (0.728) clearly
  beat `always before` (0.594 / 0.620) on the same cells, with roughly six times its macro-F1.
- **Which relations fail.** Models recover precedence and containment but rarely overlap
  (`o` / `oi` stay at or below 0.07 F1).
- **Flashbacks are the hard case.** Accuracy drops when both the narration order and the event
  list present the later event first.
- **Consistency is not correctness.** Converse violations span two orders of magnitude
  (0.3%-37.6%), yet enforcing consistency raises accuracy mainly by filling *empty* cells,
  changing answered-cell accuracy by at most 2.5 points.
- **Reading, not composing.** Only 4.4% of errors on distant pairs are composition-available,
  so failures come from extracting relations out of text rather than from symbolic composition.

## Repository contents

| Path | Description |
| --- | --- |
| `acl_latex.tex` | Full paper source (anonymous, `review` mode) |
| `acl_latex.pdf` | Compiled paper |
| `custom.bib` | Bibliography |
| `acl.sty`, `acl_natbib.bst` | ACL style files |
| `figures/` | All paper figures (PDF) |

## Compiling

```bash
pdflatex acl_latex
bibtex   acl_latex
pdflatex acl_latex
pdflatex acl_latex
```

Requires a standard TeX Live installation with the ACL style files, which are included here.

## Data and code

The annotated corpus, model outputs, parsing / scoring / consistency-analysis code and the
reproducibility scripts are released with the paper. See the Limitations section for known
caveats, including 32 passages whose gold networks are unsatisfiable (reported separately from
gold agreement) and a set of 665 model-gold disagreements that are candidate annotation issues
but have not been audited.

## Citation

```bibtex
@inproceedings{anonymous2027temporal,
  title     = {Evaluating Temporal Reasoning in Large Language Models:
               Event Temporal Relation Extraction with Allen's Interval Algebra,
               Logical Consistency, and Relative Time in Narratives},
  author    = {{Anonymous COLING 2027 submission}},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
  year      = {2027}
}
```

## License

Released under the MIT License (see `LICENSE`).