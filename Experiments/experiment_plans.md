# Experiment & Analysis Plan — Single Model Run

*Revised 2026-10-06 for the constraint: each model is run **once**. No data in `exports/` is changed.*

---

## 0. What "one run" means, and what it costs

**Definition used here.**
- One fixed prompt template for all models.
- One API call per (model, stem).
- Fixed decoding settings, e.g. temperature 0.
- Calls are never repeated for content reasons. Everything after the run is **offline analysis of the stored outputs**.

**What is lost compared with the earlier plan** (`experiment_plans.md`):
- run-to-run variance;
- comparisons between prompt formats (pairwise vs whole graph vs timeline);
- CoT vs no-CoT, and self-consistency voting;
- sparse-skeleton completion;
- asking directly for coarse labels.

These need extra calls, so they are out of scope; Section 5 lists them for future work.

**What makes up for it:**
1. Choose the output format so one answer supports many analyses. Most important: ask for the full n×n matrix, which gives a built-in converse test (§2).
2. Use **comparisons across models on identical items** as the main contrast.
3. Use **structure already in the data** (narrative order, distance, Allen algebra, gold redundancy) as analysis factors.
4. Use stem-level bootstrap for uncertainty instead of repeated runs.

---

## 1. Data facts used by this plan

From the primary (non-`_batch_`) files:
- 361 stems; 360 after excluding `stem_254`, which has two different events both labelled E18.
- 101,880 unordered event pairs over those 360 stems.
- Events per stem: 8–78 (median 17).

Computed over all relation files:

| Property | Value | Used in |
|---|---|---|
| Label mix (unordered) | before/after 78.9%, during/contains 8.3%, meets 4.4%, equal 3.5%, overlaps 2.5%, finishes 1.4%, starts 1.1% | A1, A4 |
| "Always <" baseline (i<j) | 60.7% acc, 0.058 macro-F1 | A1 |
| "Follow text order" baseline | 60.1% acc, 0.072 macro-F1 | A1, A6 |
| Gold "before" pairs narrated in reverse text order | 23.9% (90 stems fully linear; 111 stems >20% reversed) | A6 |
| Pairs derivable from two other gold relations in one composition step | mean 88.6% per stem | A5 |
| Path-inconsistent gold files | 32 files / 640 triples, all *A m B, B m C, A m C* | A9 |
| Stems with ≥1 duration/offset phrase (regex heuristic) | 431 / 462 | A13 |

**Output size.** A full ordered matrix is n(n−1) cells. For 63 stems that is over 1,000 cells, for 22 stems over 2,000, and for 5 stems over 3,000; the largest (78 events) has 6,006. The output format must stay compact (§2.2).

---

## 2. Getting the single run right (decide everything before the run)

Nothing can be redone, so these decisions are part of the method and should be written down before any call is made.

### 2.1 Input
- Stem text plus event list (ID + verbatim span), in the gold ID order.
- The 13 Allen relations, each with a one-line endpoint definition (e.g. *A meets B: A ends exactly when B starts*), in the same words for every model.
- Run all 360 primary stems. Optionally also run the 8 `_batch_` versions as separate items (useful for A9).
- **Exclude `stem_254`**: its duplicate E18 makes the event list ambiguous.

### 2.2 Output: ask for the full n×n matrix, compactly
- One line per event: `E3: < < = m d ...`, i.e. relation symbols in column order E1..En, with `-` on the diagonal. This is far cheaper than one JSON object per pair.
- **Why both triangles:** the lower triangle should equal the converse of the upper. Asking for both turns the single run into a self-consistency test (A2) without a second call. The cost is about twice the output.
- Set the output-token limit high enough for the largest stem. Before the run, check that every planned model's maximum output length can fit the largest matrices with a tokenizer. If a model cannot, record that, and either exclude the largest stems for every model or accept and report truncation. The same rule must apply to all models.

### 2.3 Fixed policies
- Temperature 0 (or the lowest the model allows); record every decoding parameter.
- **Retries:** only for transport failures (HTTP errors or empty responses), never because an answer looks wrong. Log every retry.
- **Unparseable or missing cells:** count as wrong in the main scores. Report coverage separately and also score "answered cells only".
- Log per call: model ID, the provider that served it, timestamp, prompt hash, raw output, finish reason (to detect truncation), input/output tokens, cost, latency. Store token log-probabilities if the provider returns them; they enable calibration (A3b).
- If a dry run is allowed, use ≤3 stems, only to check parsing, and **exclude them from all results**. If not allowed, validate the parser on synthetic outputs instead.

### 2.4 Pre-register before seeing outputs
Write down the hypotheses (H1–H8 below), the primary metric (**13-way macro-F1 on the upper triangle**), the exclusion rules and the statistical tests. With one run, this is the main protection against analysing the data until something looks significant.

### 2.5 Optional: one embedded manipulation (choose at most one, or none)
A single run can still contain a controlled contrast **between stems**. Randomly assign stems (stratified by event count and non-linearity, fixed seed, the same assignment for every model) to two arms:
- **M1 — relabelled IDs:** event IDs randomly permuted (E-numbers no longer follow gold order). This tests whether models exploit ID order: in gold, 70% of pairs with i<j point forward.
- **M2 — shuffled event list:** the list order is randomised, while IDs and spans stay the same.

Scoring maps back to the original IDs, and the gold files are untouched. Cost: each arm has about half the stems, so the main results should still be reported pooled, and the manipulation effect estimated with stem-level tests. If the main goal is a clean benchmark, skip this.

---

## 3. Offline analyses (all from the single run)

Each analysis states a hypothesis where one makes sense.

### A1 — Agreement with the human matrix (core result)
- **Upper triangle:** accuracy, macro-F1 (13 relations), per-relation P/R/F1, 13×13 confusion matrix, and Cohen's κ between model and human.
- **Collapsed views:** 7-way (direction ignored) and 4-way {before, after, equal, overlapping-family}.
- **Temporal awareness** (closure-based P/R; UzZaman & Allen, 2011).
- Always show the two trivial baselines (60.7% / 60.1% accuracy) next to model scores.
- **H1:** models beat the baselines on macro-F1 by a wide margin but much less on accuracy; the rare relations (s, f, o, m) score lowest.

### A2 — Converse self-consistency (enabled by §2.2)
- For every pair, check whether the lower-triangle cell equals the converse of the upper-triangle cell. Report the violation rate per model, per relation, and against distance.
- Score the lower triangle on its own as a second, dependent reading of the same item. Report the "both directions right" rate.
- **H2:** violations concentrate in the non-symmetric fine relations (s/si, f/fi, d/di). ChronoSense reports inconsistent handling even of symmetric relation pairs.

### A3 — Logical consistency and offline repair
- **Exact satisfiability check:** rewrite each atomic Allen relation as order constraints between the endpoints (start_i < end_i, plus the start/end comparisons the relation implies). The network is consistent iff the point graph, after merging "=" nodes, has no cycle through a strict "<". Report the % of consistent stems per model and the number of path-inconsistent triples (Allen composition table).
- **A3a — Repair without new calls:** find the nearest consistent network, i.e. the fewest label changes needed (ILP/MaxSAT). Two variants: (i) upper triangle only; (ii) both triangles as two votes per pair. Re-score.
  - **H3:** repair raises accuracy slightly, but consistency ≠ correctness (cf. Kougia et al., 2024; Eirew et al., 2025, who use an ILP closure step).
- **A3b — Only if log-probs exist:** confidence-weighted repair and calibration (reliability diagram, ECE). Without log-probs, skip; do not use self-reported confidence.

### A4 — Error anatomy at the endpoint level
- Each Allen relation corresponds to fixed comparisons (<, =, >) of four endpoint pairs: start–start, start–end, end–start, end–end. Translate both predicted and gold relations into these.
- Report which endpoint comparisons models get wrong, and **error severity** = number of the four comparisons that differ (0–4). Severity 1 means a near miss, such as *meets* vs *before*, which differ only in end_i vs start_j.
- **H4:** most errors are severity-1 near misses on "=" endpoints: meets, starts, finishes, equal.

### A5 — Reading vs reasoning (using Allen composition, no new calls)
For every wrong pair (i,j):
- **Composition-available error:** some event k exists where the model got r(i,k) and r(k,j) right, and their composition gives a single result equal to the gold r(i,j). The model "had" the premises but did not apply them. This is a reasoning failure.
- **Self-entailment error:** the model's own r(i,k) and r(k,j) entail a single relation that differs from its own r(i,j). This measures internal incoherence regardless of gold.
- **Anchoring:** compare error rates for pairs mentioned in the same or adjacent sentence ("local reading") with distant pairs that need inference.
- **H5:** a large share of errors on distant pairs are composition-available. Gold is ~89% one-step derivable, so the information is usually present in the model's own correct answers.

### A6 — Narrative order and heuristic reliance
- Split gold "<"/">" pairs into text order agreeing vs reversed (23.9% reversed).
- Compare accuracy with a mixed-effects logistic model:
  correct ~ reversed + sentence_distance + relation + (1 | stem), fitted per model.
- **Heuristic reliance rate:** on reversed pairs, the share of errors where the model predicted the text-order answer.
- Stem level: macro-F1 by non-linearity bin (0, 0–10%, 10–20%, >20% reversed).
- **H6:** accuracy drops on reversed pairs, and errors there mostly follow text order (cf. Qiu et al., 2024).

### A7 — Distance, position, length, event count, truncation
- Accuracy and per-relation F1 by sentence distance (same / 1 / 2–3 / 4–7 / 8+). Report per-relation results within bins, because the gold mix itself changes with distance.
- Position of each event in the passage (beginning / middle / end), following Lost in the Middle (Liu et al., 2024).
- Stem-level F1 against word count and event count (correlated, r = 0.71; model both).
- **Within-output drift:** accuracy of matrix rows generated early vs late in the output, and truncation rate against n.
- **H7:** F1 falls with event count, partly through output length (late rows, truncation) rather than only text length.

### A8 — Are errors spread out or concentrated in a few events?
- Per-event row accuracy; the share of a stem's errors explained by its worst 1–2 events.
- **Event misplacement test:** for each stem, check whether moving one event's interval in the model's implied timeline would fix most errors. A simpler proxy is whether one row accounts for over half the errors.
- This tells you whether models misunderstand single events (e.g. a flashback event) or make many scattered pair-level mistakes.

### A9 — Interaction with known gold issues
- Score separately on the 32 path-inconsistent gold files and the rest.
- For the 640 *meets*-chain triples: what do models predict for (A, C)? If models systematically answer "<" (the logically required relation), that is evidence for a gold error, not a model error.
- **Candidate gold errors:** pairs where all or nearly all models agree with each other but not with gold. Manually check a random sample and report the proportion that are actual gold errors. This turns the single run into a gold-audit tool without changing any data file.
- If the `_batch_` versions were run: model scores on the two human versions of the same stem.

### A10 — Cross-model analysis (the main contrast in a one-run design)
- Pairwise Cohen's κ between models and Fleiss' κ across all models, compared with model–human κ.
- **Offline ensemble:** a majority vote across models per cell (then A3 repair) as an extra "system", plus an oracle "any model correct" upper bound.
- **Item difficulty:** the number of models correct per pair. Relate difficulty to relation, distance, reversal and derivability. Optionally fit a simple item-response model (pair difficulty and model ability jointly, with a stem random effect).
- **Paired model comparisons** use the same items, so use a stem-level paired bootstrap or mixed-effects model, not pair-level McNemar (pairs within a stem are not independent).

### A11 — Model attributes (descriptive)
- Relate scores to model attributes: open vs closed, parameter count where published, reasoning vs non-reasoning.
- With one run and a handful of models these are **descriptive associations only**; do not claim effects of scale or reasoning from a few data points.

### A12 — Reliability and cost
- Parse-failure rate, truncation rate, coverage, tokens, cost and latency per stem.
- Macro-F1 per dollar; a Pareto plot of cost against F1.

### A13 — Explicit durations subset
- Flag pairs where either event's sentence contains a duration or offset phrase (regex; heuristic).
- Compare accuracy on flagged vs unflagged pairs.
- For a stronger claim, manually tag a sample of pairs whose relation **requires** arithmetic (e.g. "twelve days later") and report accuracy on those. Test of Time (Fatemi et al., 2024) finds arithmetic temporal reasoning harder.

### A14 — Passage type
- Label each stem by type: personal narrative, long non-linear literary piece, procedural/engineering timeline, short synthetic scene.
- Use manual labels or a documented rule. Report the labelling procedure and, if done by hand, agreement on a sample.
- Report F1 by type. Types are confounded with length and non-linearity, so include those as covariates.

---

## 4. Statistical plan

- **Primary metric:** 13-way macro-F1 on the upper triangle, all stems except 254, missing cells counted wrong.
- **Uncertainty:** 95% CIs from cluster bootstrap over stems (≥1,000 resamples). With one run, these CIs reflect sampling of stems, not run-to-run variance; state this explicitly.
- **Comparisons:** stem-level paired bootstrap, or mixed-effects logistic regression with a stem random effect. Holm correction within each analysis family.
- **Robustness of conclusions:** repeat the key results (A1, A6) on (a) the consistent-gold subset and (b) the 90 fully linear stems vs the rest.
- **Limitation to state in the paper:** single sample per model, so model rankings closer than run-to-run variance (unknown here) should not be over-interpreted. QSTRBench (Cohn & Blackwell, 2026) observed that the same model can answer identical questions differently across runs.

---

## 5. Not possible in one run (future work)

- Prompt and format comparisons (pairwise vs whole graph vs timeline; Narrative-of-Thought).
- CoT vs no-CoT, and different reasoning budgets for the same model. Comparing a reasoning model with a non-reasoning model is A11, not this.
- Self-consistency voting and run-to-run variance.
- Sparse-skeleton completion and nonce-label tests.
- Directly prompting with coarse label sets (only post-hoc collapsing is possible: A1).
- Spatial relations: the gold is temporal only. A spatio-temporal extension needs new annotation (event locations, containment between places).

---

## 6. Deliverable tables and figures from the single run

1. Main table: per model × {acc, macro-F1 13/7/4-way, κ, temporal-awareness F1, converse violation %, % consistent stems, coverage, cost}, with baselines.
2. Per-relation F1 heatmap (models × 13 relations).
3. Aggregated confusion matrix, plus the endpoint-severity histogram (A4).
4. Accuracy on reversed vs in-order pairs and the heuristic-reliance rate (A6).
5. F1 against sentence distance, event count and output position (A7).
6. Error decomposition: composition-available vs self-entailment vs other (A5).
7. Repair effect, before vs after (A3).
8. Cross-model κ matrix, ensemble/oracle results, and the gold-error audit summary (A9–A10).

---

## References

- Allen (1983), *Maintaining knowledge about temporal intervals*, CACM. https://doi.org/10.1145/182.358434
- UzZaman & Allen (2011), *Temporal Evaluation*, ACL. https://aclanthology.org/P11-2061
- Qiu et al. (2024), *Are Large Language Models Temporally Grounded?*, NAACL. https://aclanthology.org/2024.naacl-long.391
- Kougia et al. (2024), *Analysing zero-shot temporal relation extraction on clinical notes using temporal consistency*, BioNLP. https://aclanthology.org/2024.bionlp-1.6
- Eirew, Bar & Dagan (2025), *Beyond Pairwise: Global Zero-shot Temporal Graph Generation*. https://arxiv.org/abs/2502.11114
- Islakoglu & Kalo (2025), *ChronoSense*, ACL. https://aclanthology.org/2025.acl-short.46
- Fatemi et al. (2024), *Test of Time*, ICLR 2025. https://arxiv.org/abs/2406.09170
- Liu et al. (2024), *Lost in the Middle*, TACL. https://aclanthology.org/2024.tacl-1.9
- Cohn & Blackwell (2026), *QSTRBench*. https://arxiv.org/abs/2605.18380
- Yuan, Xie & Ananiadou (2023), *Zero-shot Temporal Relation Extraction with ChatGPT*. https://arxiv.org/abs/2304.05454
