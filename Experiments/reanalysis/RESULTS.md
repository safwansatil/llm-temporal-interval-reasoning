# Re-analysis of the stored outputs (no new model calls, no new annotation)

Every number below comes from `Experiments/results/` via `Experiments/scripts/common.py`
(the frozen parser) and `analysis_lib.py` (the frozen metrics). Strict-parse numbers
reproduce `A1_main_metrics.csv` to six decimals, so these scripts are a verified
layer on top of the original pipeline, not a reimplementation.

Run any script with `python3 <name>.py` (needs `statsmodels`, `ortools`).

## 1. Second annotation pass -> agreement estimate (`iaa.py`, `iaa2.py`)

Eight stems in `exports/stem_relations/` have a `*_batch_*` version covering a subset of the
same event spans. Matching by verbatim span text gives 190 doubly-annotated pairs.

| comparison | n | accuracy | Cohen kappa |
|---|---|---|---|
| second pass vs primary gold | 190 | 0.779 | 0.582  (95% CI [0.196, 0.803]) |
| deepseek vs gold, same pairs | 173 | 0.821 | 0.627 |
| gpt-luna vs gold, same pairs | 170 | 0.812 | 0.549 |
| gemini vs gold, same pairs | 139 | 0.727 | 0.282 |
| qwen vs gold, same pairs | 142 | 0.669 | 0.299 |

Top disagreements between passes: `<` vs `m` (9), `di` vs `<` (8), `<` vs `=` (4), `<` vs `o` (3).

CAVEAT: provenance of the batch files is unconfirmed. If the same annotator produced both
passes this is intra-annotator agreement (an upper bound on IAA), and if those 8 stems were
revisited because the first pass looked wrong the subset is selected for difficulty.
Do not put this in the paper until both are settled.

## 2. Recovery parsing (`recover.py`, `run_recovery.py`, `validate_recovery.py`, `boot_recovery.py`)

R2 rule: when a bad-length row carries its `-` at the true diagonal, read it left-anchored and
right-anchored and accept a cell only where the two readings agree. Then backfill missing cells
from the model's own mirror cell (converse).

| model | cov strict | cov rec | acc strict | acc rec | F1 strict | F1 rec |
|---|---|---|---|---|---|---|
| deepseek | 0.600 | 0.836 | 0.452 | 0.630 | 0.270 | 0.301 |
| gpt-luna | 0.481 | 0.870 | 0.350 | 0.580 | 0.215 | 0.243 |
| gemini   | 0.577 | 0.979 | 0.328 | 0.535 | 0.128 | 0.150 |
| qwen     | 0.345 | 0.636 | 0.202 | 0.361 | 0.119 | 0.155 |
| mistral  | 0.024 | 0.116 | 0.001 | 0.039 | 0.003 | 0.014 |

Validation: agreement with strict on shared cells = 1.000 for all models (pure superset).
Accuracy on newly recovered cells: 0.767 / 0.610 / 0.516 / 0.555 (strict cells: 0.753 / 0.728 /
0.569 / 0.585). Converse-backfilled cells: 0.676 / 0.516 / 0.462 / 0.501.

deepseek recovery vs `always <` (0.6068): +0.023, cluster bootstrap [-0.041, +0.086], p=0.454.
=> parity, NOT an advantage. The pre-registered -0.155 deficit is ~85% parsing policy.

## 3. Comparability and weighting (`comparability.py`)

- Common subset (cells all four models answer): 22,665 pairs (22.2%).
  deepseek 0.805 acc / 0.438 F1; gpt-luna 0.786 / 0.403; qwen 0.642 / 0.268;
  gemini 0.614 / 0.230; `always <` 0.615 / 0.059. Ranking unchanged, margins wider.
- Pair concentration: 33 passages (9.2%) with >40 events hold 39.5% of all pairs;
  top 10% of passages hold 41.7%; top 25% hold 68.5%.
- Coverage cell-weighted vs passage-averaged: 0.600/0.854, 0.481/0.829, 0.577/0.826, 0.345/0.667.
- macro-P vs macro-R: 0.387/0.214, 0.373/0.156, 0.291/0.104, 0.301/0.079.
- Passages carrying each rare relation: oi 67, f 110, o 125, si 139 (vs 360 for `<`).

## 4. Repair controls (`repair_control.py`)

Reproduces `A3a_repair_summary.csv`. Trivial control = fill every missing cell with `<`.

| model | raw acc | repaired (ii) | fill `<` | raw F1 | repaired (ii) F1 | fill `<` F1 |
|---|---|---|---|---|---|---|
| deepseek | 0.452 | 0.591 | **0.702** | 0.270 | 0.278 | **0.284** |
| gpt-luna | 0.350 | 0.476 | **0.659** | 0.215 | 0.220 | **0.232** |
| gemini   | 0.328 | 0.417 | **0.582** | 0.128 | 0.132 | **0.142** |
| qwen     | 0.202 | 0.295 | **0.594** | 0.119 | 0.129 | **0.147** |

The control beats CP-SAT everywhere. Edit precision on answered cells the solver changes
(share wrong->right): variant i 0.26/0.37/0.21/0.27, variant ii 0.42/0.55/0.22/0.39.
gemini variant i breaks 371 correct cells while fixing 351.

## 5. Composition control (`composition2.py`)

Reproduces A5 exactly (0.0414 / 0.0253 / 0.0130 / 0.0188 share of errors composition-available).
The missing control: availability on cells the model gets RIGHT.

| model | on errors | on correct | ratio |
|---|---|---|---|
| deepseek | 0.041 | 0.781 | 18.9x |
| gpt-luna | 0.025 | 0.732 | 28.9x |
| gemini   | 0.013 | 0.708 | 54.6x |
| qwen     | 0.019 | 0.662 | 35.2x |

Premises are abundant in general and absent precisely where the model errs. This REFUTES the
"premise scarcity explains the low rate" objection and strengthens the reading-not-composing claim.

## 6. Cue interaction (`cue_gee.py`)

The paper's null came from controlling for gold relation, which is collinear with the list-order
cue: scored pairs are ordered i<j, so gold `>` holds exactly when the list shows the later event
first. Model: correct ~ list_contra * narr_rev + sent_dist, logistic GEE, clustered by passage,
answered before/after cells. 2x2 cells reproduce Figure 5 exactly (deepseek n = 34,447 / 1,902 /
2,068 / 8,412).

| model | interaction OR | 95% CI | p |
|---|---|---|---|
| deepseek | 0.222 | [0.097, 0.507] | 0.0004 |
| gemini   | 0.096 | [0.029, 0.313] | 0.0001 |
| qwen     | 0.200 | [0.073, 0.549] | 0.0018 |
| gpt-luna | 0.473 | [0.152, 1.473] | 0.1965 |

## 7. `m`/`mi` merged into `<`/`>` (`mmerge_comp.py` logic, in-line)

gold m+mi = 4,417 pairs (4.3%). Macro-F1 after merge: 0.283 / 0.227 / 0.141 / 0.129
(baseline 0.071). Ranking identical. No conclusion depends on the `m` annotations.

## 8. Definitional discrepancies resolved (not errors)

- "complete" in `A3_consistency_summary` = upper triangle fully answered;
  "parse_complete" in `A12` = all n(n-1) cells of BOTH triangles. The former is weaker, hence
  higher counts (gpt-luna 222 vs 205). Both captions now state which is meant.
- mistral macro-F1 per USD was 0.065 in Table 5; 0.0026/0.040 = 0.075. Corrected.
