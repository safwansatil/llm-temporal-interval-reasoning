"""
Writes results/analysis/REPORT.md from the saved tables. The report presents numbers and
mechanical checks only; interpretation is left to the researcher.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_write_text


def _r(tab: Path, name: str):
    p = tab / name
    return pd.read_csv(p) if p.exists() else None


def _md(df: pd.DataFrame, cols=None, nd=3) -> str:
    if df is None or not len(df):
        return "_(not available)_\n"
    df = df[cols] if cols else df
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{nd}f}"))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _fig(figdir: Path, name: str, alt: str) -> str:
    return f"\n![{alt}](figures/{name})\n" if (figdir / name).exists() else ""


def write_report(out: Path, tab: Path, figdir: Path, ctx) -> None:
    man = ctx.manifest
    L = []
    a = L.append
    a("# Single-run experiment: results report\n")
    a(f"_Generated {datetime.now().isoformat(timespec='seconds')} by scripts/analyze.py. "
      f"All numbers come from `results/analysis/tables/*.csv`; this report adds no interpretation._\n")
    a("## Run\n")
    a(f"- Run started: {man['created_at']} · prompt template sha256 `{man['prompt_template_sha256'][:16]}…` · "
      f"gold data sha256 `{man['data_sha256'][:16]}…`")
    a(f"- Items: {len(ctx.primary)} primary stems analysed (+{len(ctx.batch)} `_batch_` versions, A9 only). "
      f"Excluded before the run: " + "; ".join(f"{k} ({v})" for k, v in man.get("excluded_items", {}).items()))
    dropped = [k for k in man["models"] if k not in ctx.models]
    if dropped:
        a("- Models started but removed from the cohort during the run (not analysed; their saved answers "
          "remain in results/raw/): " + ", ".join(f"{k} ({man['models'][k]['model_key']})" for k in dropped))
    a("- Models and request parameters (one call per model and item):\n")
    rows = []
    for mid in ctx.models:
        mm = man["models"][mid]
        p = {k: v for k, v in mm["request_params"].items() if k != "stream"}
        rows.append({"model": mid, "OpenRouter key": mm["model_key"], "parameters": str(p),
                     "notes": "; ".join(mm.get("notes", []))})
    a(_md(pd.DataFrame(rows)))
    a("## Method notes\n")
    a("- **Unit of scoring:** upper triangle (i < j in event-list order); the model's cell (i, j) is its answer. "
      "Missing or unparseable cells count as wrong (false negative for the gold label, never a false positive).")
    a("- **Macro-F1** is the unweighted mean of per-relation F1 over the 13 Allen relations present in the gold set. "
      "**Cohen's κ** treats 'missing' as its own predicted category. 7-way = direction ignored; 4-way = before / after / "
      "equal / overlap-or-contact family (m, o, s, d, f and inverses).")
    a("- **Temporal awareness** (UzZaman & Allen 2011): the gold matrices are complete, so the gold closure equals the "
      "gold; precision therefore reduces to accuracy on answered cells and recall to accuracy on all cells.")
    a("- **Uncertainty:** 95% CIs from a cluster bootstrap over stems "
      f"({ctx.args.bootstrap} resamples, seed 20261006). Model comparisons: paired stem-level bootstrap, Holm-corrected; "
      f"bootstrap p-values have resolution 1/(B+1), so the smallest possible value is {1 / (ctx.args.bootstrap + 1):.4f}. "
      "These CIs reflect sampling of stems only, not run-to-run variance (each model was run once).")
    a("- **Regressions (A6, A14):** logistic GEE clustered by stem with an independence working correlation "
      "(i.e. logistic regression with cluster-robust standard errors), used instead of a random-intercept model.")
    a("- **Repair (A3a):** nearest consistent network = fewest changed labels, solved with OR-Tools CP-SAT "
      "(endpoint model). The problem is NP-hard; the solver stops at a time limit or after a stall, so many solutions are "
      "the best found, not proven optimal. Settings: time limit {tl} s, stall {st} s, {w} search workers per network. "
      "Solver status, objective and upper bound per network are in `A3a_repair_solver_status.csv`.".format(
          tl=ctx.args.repair_time_limit, st=ctx.args.repair_stall, w=ctx.args.repair_workers))
    a("- **Ensemble (A10):** plurality vote over models per cell; ties broken by the overall frequency of each label across "
      "all model predictions (no gold information used).\n")

    # ---------------- A1
    a("## A1 — Agreement with the human matrix\n")
    m = _r(tab, "A1_main_metrics.csv")
    a(_md(m, ["system", "n_pairs", "accuracy", "acc_lo", "acc_hi", "macro_f1", "f1_lo", "f1_hi", "kappa",
              "coverage", "ta_f1", "macro_f1_7way", "macro_f1_4way"]))
    a(_fig(figdir, "F1_main_metrics.png", "main metrics"))
    pr = _r(tab, "A1_per_relation.csv")
    if pr is not None:
        piv = pr[~pr.system.str.startswith("baseline")].pivot(index="system", columns="relation", values="f1")
        piv = piv.reindex(index=ctx.models)[["<", ">", "=", "m", "mi", "o", "oi", "s", "si", "d", "di", "f", "fi"]].reset_index()
        a("Per-relation F1:\n")
        a(_md(piv, nd=2))
    a(_fig(figdir, "F2_per_relation_f1.png", "per-relation F1"))
    a(_fig(figdir, "F3_confusion_matrices.png", "confusion matrices"))
    c = _r(tab, "A1_pairwise_model_comparisons.csv")
    if c is not None:
        a("Pairwise model differences (macro-F1), paired stem bootstrap, Holm-adjusted:\n")
        a(_md(c[c.metric == "macro_f1"], ["model_a", "model_b", "diff", "diff_lo", "diff_hi", "p_boot", "p_holm"]))

    # ---------------- A2
    a("## A2 — Converse self-consistency\n")
    a(_md(_r(tab, "A2_converse_summary.csv")))
    d = _r(tab, "A2_converse_by_gold_relation.csv")
    if d is not None:
        a("Violation rate by gold relation:\n")
        a(_md(d.pivot(index="model", columns="gold_relation", values="violation_rate").reindex(ctx.models).reset_index(), nd=3))

    # ---------------- A3
    a("## A3 — Logical consistency\n")
    a(_md(_r(tab, "A3_consistency_summary.csv")))
    rs = _r(tab, "A3a_repair_summary.csv")
    a("### A3a — Offline repair to the nearest consistent network\n")
    a(_md(rs) if rs is not None else "_Not run yet (run analyze.py without --skip-repair)._\n")
    a(_fig(figdir, "F7_repair_effect.png", "repair effect"))

    # ---------------- A4
    a("## A4 — Error anatomy at the endpoint level\n")
    sv = _r(tab, "A4_error_severity.csv")
    if sv is not None:
        a(_md(sv.pivot(index="model", columns="severity", values="share").reindex(ctx.models).reset_index()
              .rename(columns=lambda x: f"severity {x}" if isinstance(x, (int, np.integer)) else x)))
    a("Share of errors in which each endpoint comparison differs:\n")
    a(_md(_r(tab, "A4_endpoint_comparison_error_share.csv")))
    a(_fig(figdir, "F3b_error_severity.png", "error severity"))

    # ---------------- A5
    a("## A5 — Reading vs reasoning\n")
    a(_md(_r(tab, "A5_reading_vs_reasoning.csv")))
    a(_fig(figdir, "F6_error_decomposition.png", "error decomposition"))

    # ---------------- A6
    a("## A6 — Narrative order\n")
    a(_md(_r(tab, "A6_narrative_order.csv"), ["model", "pairs_in_order", "pairs_reversed", "acc_in_order", "acc_reversed",
                                              "heuristic_reliance_rate",
                                              "odds_ratio", "or_lo", "or_hi", "p"]))
    a("_GEE: correct ~ reversed + sentence-distance bin + gold relation, clustered by stem; odds ratio for reversed._\n")
    nb = _r(tab, "A6_by_nonlinearity_bin.csv")
    if nb is not None:
        a("Macro-F1 by stem non-linearity (share of gold before/after pairs narrated in reverse):\n")
        a(_md(nb.pivot(index="system", columns="nonlinearity_bin", values="macro_f1")
              .reindex(index=ctx.models + ["baseline: text order"], columns=["0", "(0,10%]", "(10,20%]", ">20%"])
              .reset_index()))
    a(_fig(figdir, "F4_narrative_order.png", "narrative order"))

    # ---------------- A7
    a("## A7 — Distance, position, length, drift, truncation\n")
    dd = _r(tab, "A7_by_sentence_distance.csv")
    if dd is not None:
        a(_md(dd.pivot(index="model", columns="dist_bin", values="macro_f1").reindex(ctx.models)
              .reindex(columns=["same", "1", "2-3", "4-7", "8+"]).reset_index()))
    a(_fig(figdir, "F5a_by_sentence_distance.png", "by distance"))
    a("Event position (accuracy over pairs involving an event in that part of the passage):\n")
    ep = _r(tab, "A7_by_event_position.csv")
    if ep is not None:
        a(_md(ep.pivot(index="model", columns="position", values="accuracy").reindex(ctx.models).reset_index()))
    a("Stem-level regression (HC3 robust SE; predictors standardised):\n")
    a(_md(_r(tab, "A7_stem_level_regression.csv")))
    ec = _r(tab, "A7_by_event_count.csv")
    if ec is not None:
        a(_md(ec.pivot(index="model", columns="n_bin", values="macro_f1").reindex(ctx.models).reset_index()))
    a(_fig(figdir, "F5b_by_event_count.png", "by event count"))
    tr = _r(tab, "A7_truncation_by_event_count.csv")
    if tr is not None:
        a("Truncation rate by event count:\n")
        a(_md(tr.pivot(index="model", columns="n_bin", values="truncation_rate").reindex(ctx.models).reset_index()))
    a(_fig(figdir, "F5c_within_output_drift.png", "within-output drift"))

    # ---------------- A8
    a("## A8 — Error concentration\n")
    a("_Stems with at least 5 errors for that model._\n")
    a(_md(_r(tab, "A8_error_concentration_summary.csv")))

    # ---------------- A9
    a("## A9 — Interaction with known gold issues\n")
    a(_md(_r(tab, "A9_gold_consistency_subsets.csv")))
    a("Predictions on the (A, C) pair of gold *meets*-chains (gold: A m C; logically required: A < C):\n")
    a(_md(_r(tab, "A9_meets_chain_ac_predictions.csv")))
    cg = _r(tab, "A9_candidate_gold_errors.csv")
    a(f"Candidate gold errors (≥ {max(len(ctx.models) - 1, 2)} models agree on the same label, different from gold): "
      f"**{0 if cg is None else len(cg):,} pairs** (`A9_candidate_gold_errors.csv`). A random sample of 50 is in "
      "`results/analysis_inputs/A9_gold_error_audit_sample.csv` for manual checking; fill the column and re-run "
      "`python scripts/analyze.py --force --only A9 REPORT`.\n")
    au = _r(tab, "A9_gold_error_audit_result.csv")
    if au is not None:
        a(_md(au))
    a(_md(_r(tab, "A9_batch_versions.csv")))

    # ---------------- A10
    a("## A10 — Cross-model analysis\n")
    a(_md(_r(tab, "A10_summary.csv")))
    a(_fig(figdir, "F8a_kappa_matrix.png", "kappa matrix"))
    a(_fig(figdir, "F8b_ensemble_oracle.png", "ensemble and oracle"))
    df = _r(tab, "A10_difficulty_by_factor.csv")
    if df is not None:
        a("Mean number of models correct, by factor:\n")
        a(_md(df))

    # ---------------- A11-A14
    a("## A11 — Model attributes (descriptive only)\n")
    a("Fill `results/analysis_inputs/model_attributes.csv` (open weights, parameter count) and re-run "
      "`python scripts/analyze.py --force --only A11 REPORT`. With a handful of models these are descriptive "
      "associations only.\n")
    a(_md(_r(tab, "A11_model_attributes_descriptive.csv")))
    a("## A12 — Reliability and cost\n")
    a(_md(_r(tab, "A12_reliability_and_cost.csv"), ["model", "calls", "parse_complete", "parse_partial", "parse_failed",
                                                    "truncated", "cell_coverage_all_cells", "completion_tokens",
                                                    "reasoning_tokens_reported", "cost_usd", "latency_median_s",
                                                    "retried_calls", "macro_f1", "macro_f1_per_usd"]))
    a("_reasoning_tokens_reported is as returned by the provider; some providers report 0 even when reasoning text "
      "was returned, so reasoning_chars in the CSV is the more reliable length measure._\n")
    a(_fig(figdir, "F9_cost_vs_f1.png", "cost vs F1"))
    a("## A13 — Explicit durations\n")
    a("_Flag = either event's sentence contains a number + time-unit phrase (regex heuristic, not an annotation)._\n")
    a(_md(_r(tab, "A13_duration_flag.csv")))
    t13 = _r(tab, "A13_arithmetic_tagged_accuracy.csv")
    a(_md(t13) if t13 is not None else "Manual tagging sample: `results/analysis_inputs/A13_arithmetic_tagging_sample.csv` "
      "(fill `requires_arithmetic`, then `python scripts/analyze.py --force --only A13 REPORT`).\n")
    a("## A14 — Passage type\n")
    t14 = _r(tab, "A14_by_passage_type.csv")
    a(_md(t14) if t14 is not None else "Label each stem in `results/analysis_inputs/stem_types.csv` (column `type`), "
      "then run `python scripts/analyze.py --force --only A14 REPORT`.\n")

    # ---------------- robustness
    a("## Robustness\n")
    rb = _r(tab, "R_robustness_A1.csv")
    if rb is not None:
        a(_md(rb.pivot(index="model", columns="subset", values="macro_f1").reindex(ctx.models).reset_index()))
    r6 = _r(tab, "R_robustness_A6.csv")
    if r6 is not None:
        a("Reversed-pair odds ratio (A6) by subset:\n")
        a(_md(r6.pivot(index="model", columns="subset", values="odds_ratio").reindex(ctx.models).reset_index()))

    # ---------------- mechanical hypothesis checks
    a("## Mechanical hypothesis checks (pre-registered criteria, no interpretation)\n")
    chk = []
    if m is not None:
        mm_ = m.set_index("system")
        base_f1 = mm_.loc[[s for s in mm_.index if s.startswith("baseline")], "macro_f1"].max()
        base_acc = mm_.loc[[s for s in mm_.index if s.startswith("baseline")], "accuracy"].max()
        for mid in ctx.models:
            chk.append({"hypothesis": "H1", "model": mid,
                        "check": "macro-F1 gain over best baseline vs accuracy gain over best baseline",
                        "value": f"ΔF1 = {mm_.loc[mid, 'macro_f1'] - base_f1:+.3f}; ΔAcc = {mm_.loc[mid, 'accuracy'] - base_acc:+.3f}"})
    if pr is not None:
        for mid in ctx.models:
            x = pr[pr.system == mid].sort_values("f1")
            chk.append({"hypothesis": "H1", "model": mid, "check": "4 lowest-F1 relations",
                        "value": ", ".join(f"{r.relation} ({r.f1:.2f})" for r in x.head(4).itertuples())})
    a6 = _r(tab, "A6_narrative_order.csv")
    if a6 is not None:
        for r in a6.itertuples():
            chk.append({"hypothesis": "H6", "model": r.model, "check": "acc reversed < acc in order; heuristic reliance > 0.5",
                        "value": f"{r.acc_reversed:.3f} vs {r.acc_in_order:.3f}; reliance {r.heuristic_reliance_rate:.2f}"})
    sv = _r(tab, "A4_error_severity.csv")
    if sv is not None:
        for mid in ctx.models:
            x = sv[(sv.model == mid) & (sv.severity == 1)]
            chk.append({"hypothesis": "H4", "model": mid, "check": "share of errors with severity 1 (> 0.5?)",
                        "value": f"{x.share.iloc[0]:.3f}" if len(x) else "–"})
    a5 = _r(tab, "A5_reading_vs_reasoning.csv")
    if a5 is not None:
        for r in a5.itertuples():
            chk.append({"hypothesis": "H5", "model": r.model, "check": "share of distant-pair errors that are composition-available",
                        "value": f"{getattr(r, 'share_comp_available_among_distant_errors'):.3f}"})
    if rs is not None:
        for r in rs.itertuples():
            chk.append({"hypothesis": "H3", "model": f"{r.model} ({r.variant})", "check": "Δ macro-F1 after repair",
                        "value": f"{r.delta_macro_f1:+.4f}"})
    a(_md(pd.DataFrame(chk)) if chk else "_(not available)_\n")

    a("## Limitations\n")
    a("- One sample per model: rankings closer than (unknown) run-to-run variance should not be over-interpreted.")
    a("- Some models do not accept temperature (see the run table); their provider default was used.")
    a("- The gold standard contains known issues (32 path-inconsistent stems; duplicate event wording in stems 548 and 550); "
      "see A9 and the robustness table.")
    a("- The repair solutions are best-found within a time limit, not always proven optimal.")
    a("- Passage-type (A14) and arithmetic (A13) analyses depend on manual labels in `results/analysis_inputs/`.\n")
    atomic_write_text(out / "REPORT.md", "\n".join(L))
