arXiv submission package
========================

Upload arxiv-submission.zip directly to arXiv. It contains:

  main.tex          paper source, de-anonymised (acl.sty "preprint" option)
  main.bbl          pre-compiled bibliography -- arXiv does NOT run BibTeX,
                    so this file is required
  acl.sty           ACL style file
  acl_natbib.bst    bibliography style (kept for regeneration; unused by arXiv)
  custom.bib        bibliography source (kept for reference; unused by arXiv)
  figures/          all 10 figures (PDF)

arXiv builds this with pdflatex. Verified locally: 0 errors, 0 overfull boxes,
22 pages.

Suggested arXiv metadata
------------------------
Primary category : cs.CL
Cross-list       : cs.AI
License          : choose one; the repository itself is MIT

Authors (this order, matching the paper):
  Adib Sakhawat, Fardeen Sadab, Tamjid Hasan Fahim, Multazam Mahmud,
  Tahsan Ferdous Lihan, Md Isfak Iqbal Chowdhury, Nafis Ahnaf Jamil,
  Md. Atik Shahriar, Sieam Shahriare, Wasi Omar Syed, M Safwan Hasan Khan,
  Tabib Hassan

Note: ARR double-blind policy permits a non-anonymous preprint, but check the
cycle's preprint deadline rules before posting.
