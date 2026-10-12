"""
Static figures (PNG, 200 dpi) built from the CSV tables in results/analysis/tables/.

Design rules (dataviz method): one colour per model in a fixed order (validated categorical
palette, slots 1-6, adjacent CVD dE >= 9.1), marker shapes as secondary encoding, legends
for every multi-series chart, thin marks, hairline solid grid, single y-axis per panel,
sequential single-hue blue for magnitudes. Every figure has its numbers in a CSV table.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9893"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REL_ORDER = ["<", ">", "=", "m", "mi", "o", "oi", "s", "si", "d", "di", "f", "fi"]
DIST_ORDER = ["same", "1", "2-3", "4-7", "8+"]
NBIN_ORDER = ["8-15", "16-25", "26-40", "41+"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def _colors(models: List[str]):
    return {m: SLOTS[k % len(SLOTS)] for k, m in enumerate(models)}, \
           {m: MARKERS[k % len(MARKERS)] for k, m in enumerate(models)}


def _save(fig, path: Path):
    fig.tight_layout()
    tmp = path.with_name("." + path.name)
    fig.savefig(tmp, dpi=200, format="png")
    plt.close(fig)
    tmp.replace(path)


def _read(tab: Path, name: str):
    p = tab / name
    return pd.read_csv(p) if p.exists() else None


def fig_main(tab, fig, models, col, mk):
    d = _read(tab, "A1_main_metrics.csv")
    if d is None:
        return
    d = d.set_index("system")
    systems = models + [s for s in d.index if s.startswith("baseline")]
    f, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(systems) + 1.6), sharey=True)
    for ax, (metric, lo, hi, title) in zip(axes, [("macro_f1", "f1_lo", "f1_hi", "13-way macro-F1 (95% stem-bootstrap CI)"),
                                                  ("accuracy", "acc_lo", "acc_hi", "Accuracy (95% stem-bootstrap CI)")]):
        y = np.arange(len(systems))[::-1]
        for yy, s in zip(y, systems):
            c = col.get(s, NEUTRAL)
            v = d.loc[s, metric]
            ax.barh(yy, v, height=0.5, color=c)
            ax.errorbar(v, yy, xerr=[[v - d.loc[s, lo]], [d.loc[s, hi] - v]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
            ax.text(d.loc[s, hi] + 0.01, yy, f"{v:.3f}", va="center", color=INK2, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(systems)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    _save(f, fig / "F1_main_metrics.png")


def fig_per_relation(tab, fig, models):
    d = _read(tab, "A1_per_relation.csv")
    if d is None:
        return
    d = d[d.system.isin(models)]
    M = d.pivot(index="system", columns="relation", values="f1").reindex(index=models, columns=REL_ORDER)
    sup = d.drop_duplicates("relation").set_index("relation").support.reindex(REL_ORDER)
    f, ax = plt.subplots(figsize=(11, 0.5 * len(models) + 1.8))
    im = ax.imshow(M.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(REL_ORDER)))
    ax.set_xticklabels([f"{r}\n(n={int(sup[r]):,})" for r in REL_ORDER], fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.grid(False)
    ax.set_title("F1 per Allen relation (gold support in brackets)", loc="left")
    cb = f.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    _save(f, fig / "F2_per_relation_f1.png")


def fig_confusion(tab, fig, models):
    d = _read(tab, "A1_confusion_long.csv")
    if d is None:
        return
    cols = REL_ORDER + ["missing"]
    k = len(models)
    nc = 3 if k > 4 else k
    nr = int(np.ceil(k / nc))
    f, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.2 * nr), squeeze=False)
    for a, m in enumerate(models):
        ax = axes[a // nc][a % nc]
        x = d[d.model == m].pivot(index="gold", columns="pred", values="count").reindex(index=REL_ORDER, columns=cols).fillna(0)
        R = x.values / np.maximum(x.values.sum(1, keepdims=True), 1)
        ax.imshow(R, cmap=BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c if c != "missing" else "∅" for c in cols], fontsize=7)
        ax.set_yticks(range(len(REL_ORDER)))
        ax.set_yticklabels(REL_ORDER, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("gold", fontsize=8)
        ax.grid(False)
        ax.set_title(m, loc="left")
    for a in range(k, nr * nc):
        axes[a // nc][a % nc].axis("off")
    f.suptitle("Confusion matrices (row-normalised; ∅ = missing/unparseable)", x=0.01, ha="left", fontweight="bold")
    _save(f, fig / "F3_confusion_matrices.png")


def fig_severity(tab, fig, models, col, mk):
    d = _read(tab, "A4_error_severity.csv")
    if d is None:
        return
    f, ax = plt.subplots(figsize=(8, 3.8))
    sev = [1, 2, 3, 4]
    w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        x = d[d.model == m].set_index("severity").share.reindex(sev).fillna(0)
        ax.bar(np.arange(4) + (k - (len(models) - 1) / 2) * w, x.values, width=w * 0.85, color=col[m], label=m)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s} endpoint\ncomparison{'s' if s > 1 else ''} differ" for s in sev])
    ax.set_ylabel("share of a model's errors")
    ax.set_title("Error severity: how far wrong answers are from the gold relation", loc="left")
    ax.legend(ncol=min(len(models), 6), loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(f, fig / "F3b_error_severity.png")


def fig_narrative(tab, fig, models, col, mk):
    d = _read(tab, "A6_narrative_order.csv")
    if d is None:
        return
    d = d.set_index("model").reindex(models)
    f, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(models) + 1.8), sharey=True)
    y = np.arange(len(models))[::-1]
    ax = axes[0]
    for yy, m in zip(y, models):
        a, b = d.loc[m, "acc_in_order"], d.loc[m, "acc_reversed"]
        ax.plot([a, a], [yy + 0.13, yy - 0.13], color=GRID, lw=1, zorder=1)
        ax.scatter([a], [yy + 0.13], s=60, color=SLOTS[0], zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="text order = chronological" if yy == y[0] else None)
        ax.scatter([b], [yy - 0.13], s=60, color=SLOTS[1], marker="s", zorder=2, edgecolor=SURFACE, linewidth=2,
                   label="narrated in reverse" if yy == y[0] else None)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy on gold before/after pairs")
    ax.set_title("Narrative order vs chronology", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    ax.barh(y, d.heuristic_reliance_rate.values, height=0.5, color=SLOTS[0])
    for yy, v in zip(y, d.heuristic_reliance_rate.values):
        if not np.isnan(v):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("share of errors on reversed pairs that follow text order")
    ax.set_title("Heuristic reliance", loc="left")
    ax.grid(axis="y", visible=False)
    _save(f, fig / "F4_narrative_order.png")


def _lines(ax, d, models, col, mk, xcol, order, ycol):
    for m in models:
        x = d[d.model == m].set_index(xcol)[ycol].reindex(order)
        ax.plot(range(len(order)), x.values, color=col[m], marker=mk[m], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)


def fig_distance_count_drift(tab, fig, models, col, mk):
    d = _read(tab, "A7_by_sentence_distance.csv")
    if d is not None:
        f, ax = plt.subplots(figsize=(7.5, 4))
        d["dist_bin"] = d.dist_bin.astype(str)
        _lines(ax, d, models, col, mk, "dist_bin", DIST_ORDER, "macro_f1")
        ax.set_xlabel("sentence distance between the two events")
        ax.set_ylabel("13-way macro-F1")
        ax.set_title("Performance by distance in the text", loc="left")
        ax.legend(ncol=3, fontsize=8)
        _save(f, fig / "F5a_by_sentence_distance.png")
    d = _read(tab, "A7_by_event_count.csv")
    if d is not None:
        f, ax = plt.subplots(figsize=(7.5, 4))
        d["n_bin"] = d.n_bin.astype(str)
        _lines(ax, d, models, col, mk, "n_bin", NBIN_ORDER, "macro_f1")
        ax.set_xlabel("events in the stem")
        ax.set_ylabel("13-way macro-F1")
        ax.set_title("Performance by number of events", loc="left")
        ax.legend(ncol=3, fontsize=8)
        _save(f, fig / "F5b_by_event_count.png")
    d = _read(tab, "A7_within_output_drift.csv")
    if d is not None:
        f, ax = plt.subplots(figsize=(7.5, 4))
        order = list(range(1, 11))
        _lines(ax, d, models, col, mk, "row_position_decile", order, "accuracy")
        ax.set_xlabel("position of the row in the generated matrix (decile)")
        ax.set_ylabel("cell accuracy (all cells)")
        ax.set_title("Within-output drift", loc="left")
        ax.legend(ncol=3, fontsize=8)
        _save(f, fig / "F5c_within_output_drift.png")


def fig_decomposition(tab, fig, models):
    d = _read(tab, "A5_error_decomposition.csv")
    if d is None:
        return
    d = d.set_index("model").reindex(models)
    parts = [("composition_available", "composition-available (premises right, conclusion wrong)", SLOTS[0]),
             ("self_entailment_only", "model contradicts its own premises", SLOTS[1]),
             ("other", "other errors", NEUTRAL)]
    f, ax = plt.subplots(figsize=(10, 0.5 * len(models) + 1.8))
    y = np.arange(len(models))[::-1]
    left = np.zeros(len(models))
    tot = d.errors.values.astype(float)
    tot[tot == 0] = np.nan
    for colname, lab, c in parts:
        v = d[colname].values / tot
        ax.barh(y, v, left=left, height=0.5, color=c, label=lab, edgecolor=SURFACE, linewidth=2)
        left = left + np.nan_to_num(v)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the model's errors")
    ax.set_title("Error decomposition (A5)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
    ax.grid(axis="y", visible=False)
    _save(f, fig / "F6_error_decomposition.png")


def fig_repair(tab, fig, models, col, mk):
    d = _read(tab, "A3a_repair_summary.csv")
    if d is None:
        return
    variants = [v for v in ["i", "ii"] if v in set(d.variant)]
    f, axes = plt.subplots(1, len(variants), figsize=(5.5 * len(variants), 0.5 * (len(models) + 1) + 1.8), sharey=True, squeeze=False)
    systems = models + ["ensemble"]
    for ax, v in zip(axes[0], variants):
        x = d[d.variant == v].set_index("model")
        y = np.arange(len(systems))[::-1]
        for yy, m in zip(y, systems):
            if m not in x.index:
                continue
            a, b = x.loc[m, "raw_macro_f1"], x.loc[m, "repaired_macro_f1"]
            ax.plot([a, b], [yy, yy], color=GRID, lw=2, zorder=1)
            ax.scatter([a], [yy], s=60, color=NEUTRAL, zorder=2, edgecolor=SURFACE, linewidth=2, label="raw" if yy == y[0] else None)
            ax.scatter([b], [yy], s=60, color=col.get(m, INK2), marker="D", zorder=3, edgecolor=SURFACE, linewidth=2,
                       label="repaired (consistent)" if yy == y[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels(systems)
        ax.set_xlabel("13-way macro-F1")
        ax.set_title(f"Repair variant {v}" + (" (upper triangle)" if v == "i" else " (both triangles as votes)"), loc="left")
        ax.legend(fontsize=8, loc="lower left")
    _save(f, fig / "F7_repair_effect.png")


def fig_kappa_ensemble(tab, fig, models, col):
    d = _read(tab, "A10_pairwise_kappa.csv")
    if d is not None:
        names = models + ["human gold"]
        K = np.full((len(models), len(names)), np.nan)
        for _, r in d.iterrows():
            if r.model_a in models and r.model_b in names:
                K[models.index(r.model_a), names.index(r.model_b)] = r.kappa
        f, ax = plt.subplots(figsize=(1.0 * len(names) + 2.5, 0.6 * len(models) + 1.6))
        im = ax.imshow(K, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if not np.isnan(K[i, j]):
                    ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if K[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.grid(False)
        ax.set_title("Cohen's kappa between models and with the human matrix", loc="left")
        cb = f.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.outline.set_visible(False)
        _save(f, fig / "F8a_kappa_matrix.png")
    s = _read(tab, "A10_summary.csv")
    m = _read(tab, "A1_main_metrics.csv")
    if s is not None and m is not None:
        m = m.set_index("system")
        labels = models + ["majority-vote ensemble", "oracle (any model right)"]
        vals = [m.loc[x, "accuracy"] for x in models] + [s.ensemble_accuracy.iloc[0], s.oracle_any_model_correct.iloc[0]]
        colors = [col[x] for x in models] + [INK2, NEUTRAL]
        f, ax = plt.subplots(figsize=(8, 0.45 * len(labels) + 1.4))
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, height=0.5, color=colors)
        for yy, v in zip(y, vals):
            ax.text(v + 0.01, yy, f"{v:.3f}", va="center", color=INK2, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("accuracy (13-way)")
        ax.set_title("Single models, ensemble and oracle", loc="left")
        ax.grid(axis="y", visible=False)
        _save(f, fig / "F8b_ensemble_oracle.png")


def fig_cost(tab, fig, models, col, mk):
    d = _read(tab, "A12_reliability_and_cost.csv")
    if d is None or d.cost_usd.isna().all():
        return
    f, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in d.iterrows():
        ax.scatter(r.cost_usd, r.macro_f1, s=70, color=col[r.model], marker=mk[r.model], edgecolor=SURFACE, linewidth=2, label=r.model)
        ax.annotate(r.model, (r.cost_usd, r.macro_f1), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("total cost of the run (USD, log scale)")
    ax.set_ylabel("13-way macro-F1")
    ax.set_title("Cost vs quality", loc="left")
    ax.legend(fontsize=8, ncol=2)
    _save(f, fig / "F9_cost_vs_f1.png")


def make_all(tab: Path, fig: Path, models: List[str]) -> None:
    fig.mkdir(parents=True, exist_ok=True)
    col, mk = _colors(models)
    fig_main(tab, fig, models, col, mk)
    fig_per_relation(tab, fig, models)
    fig_confusion(tab, fig, models)
    fig_severity(tab, fig, models, col, mk)
    fig_narrative(tab, fig, models, col, mk)
    fig_distance_count_drift(tab, fig, models, col, mk)
    fig_decomposition(tab, fig, models)
    fig_repair(tab, fig, models, col, mk)
    fig_kappa_ensemble(tab, fig, models, col)
    fig_cost(tab, fig, models, col, mk)
