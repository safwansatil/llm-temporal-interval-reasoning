# Pre-registration (written automatically before the first model call)

Created: 2026-10-06T18:14:30Z
Prompt template sha256: c4d2caddb6033ba9efdfcabe8304c220d2e21bc0d96aaedb98e36be51a36ea62
Gold data sha256: 6b5df09a459cd563cdb362e79e0d9c5e45906624373548b4b83d8636fd7304f8

## Design
- One call per (model, item); identical frozen prompt for all models; settings in run_manifest.json.
- Models receive only the passage and its event list (plus task instructions). No gold relations.
- Items: 360 primary stems (stem_254 excluded: duplicate event ID E18) + 8 '_batch_'
  annotation versions (analysed separately, A9 only).
- Retries only for transport/provider failures, never because of answer content.

## Primary metric
13-way macro-F1 on the upper triangle (event-list order), all primary stems, missing/unparseable
cells counted as wrong.

## Hypotheses (from experiment_plans_single_run.md)
- H1: models beat the trivial baselines by a wide margin on macro-F1 but much less on accuracy;
  the rare relations (s, f, o, m) score lowest.
- H2: converse violations concentrate in the non-symmetric fine relations (s/si, f/fi, d/di).
- H3: offline repair raises accuracy slightly, but consistency is not the same as correctness.
- H4: most errors are near misses differing in one endpoint comparison (meets, starts, finishes, equal).
- H5: a large share of errors on distant pairs are composition-available.
- H6: accuracy drops on pairs narrated in reverse order, and errors there mostly follow text order.
- H7: F1 falls with event count, partly through output length (late rows, truncation).

## Statistics
Cluster bootstrap over stems (1,000 resamples) for 95% CIs; stem-level paired bootstrap for model
comparisons with Holm correction; GEE logistic models clustered by stem. Single sample per model:
run-to-run variance is unknown and is stated as a limitation.
