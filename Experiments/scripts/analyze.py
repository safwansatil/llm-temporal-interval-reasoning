#!/usr/bin/env python
"""
Offline analysis of the single model run (experiment_plans_single_run.md, section 3: A1-A14).

Reads results/raw/** (never calls any model) and writes everything to results/analysis/.
Each step saves its outputs as soon as it finishes; finished steps are skipped on the next
start (unless the raw results changed or --force is given). The slow consistency-repair step
caches every solved network, so Ctrl+C at any time loses at most the network being solved.

Usage:  python scripts/analyze.py [--bootstrap 1000] [--repair-time-limit 30] [--repair-stall 3]
                                  [--force] [--allow-incomplete] [--skip-repair]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,  # noqa: E402
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table  # noqa: E402

import analysis_lib as L  # noqa: E402
from allen import (CODE2NAME, CODE2SYM, COMP_SINGLE, ENDPOINT_NAMES, ENDPOINTS, MISSING,  # noqa: E402
                   full_matrix_from_upper, inconsistent_triples, is_satisfiable,
                   repair_nearest_consistent)
from common import RESULTS, atomic_write_json, atomic_write_text, load_items, load_models, read_json  # noqa: E402

ANALYSIS_VERSION = "1.0"
OUT = RESULTS / "analysis"
TAB = OUT / "tables"
FIG = OUT / "figures"
CACHE = OUT / "cache"
INPUTS = RESULTS / "analysis_inputs"   # files the researcher may fill in (never overwritten)
PROGRESS = OUT / "_progress.json"
console = Console()
log = logging.getLogger("stnlp.analysis")


# ======================================================================= helpers
def save_csv(df: pd.DataFrame, name: str) -> Path:
    p = TAB / name
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    atomic_write_text(p, buf.getvalue())
    return p


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, nd=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], nd) if isinstance(r[c], (float, np.floating, int, np.integer)) and not isinstance(r[c], bool) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ======================================================================= context
class Ctx:
    """Everything loaded once: items, features, predictions, the long pair table."""

    def __init__(self, args):
        self.args = args
        man_path = RESULTS / "run_manifest.json"
        if not man_path.exists():
            raise SystemExit("results/run_manifest.json not found - run scripts/run_models.py first.")
        self.manifest = read_json(man_path)
        mj = load_models()
        self.models = [m["model_id"] for m in mj if m["model_id"] in self.manifest["models"]]
        self.model_keys = {m["model_id"]: m["model_key"] for m in mj}
        self.items = {it.item_id: it for it in load_items() if it.item_id in self.manifest["items"]}
        self.primary = [i for i, it in self.items.items() if it.is_primary]
        self.batch = [i for i, it in self.items.items() if not it.is_primary]

    def fingerprint(self) -> str:
        h = hashlib.sha256(ANALYSIS_VERSION.encode())
        for mid in self.models:
            d = RESULTS / "raw" / mid
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                st = p.stat()
                h.update(f"{mid}/{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()

    def load(self):
        missing = []
        self.pred: Dict[Tuple[str, str], np.ndarray] = {}
        self.parse: Dict[Tuple[str, str], Dict] = {}
        self.info: Dict[Tuple[str, str], Dict] = {}
        with Progress(SpinnerColumn(), TextColumn("loading model outputs"), BarColumn(),
                      MofNCompleteColumn(), console=console, transient=True) as pg:
            t = pg.add_task("load", total=len(self.models) * len(self.items))
            for mid in self.models:
                for iid, it in self.items.items():
                    p = RESULTS / "raw" / mid / f"{iid}.json"
                    if not p.exists():
                        missing.append((mid, iid))
                    else:
                        M, st, info = L.load_prediction(p, it)
                        self.pred[(mid, iid)] = M
                        self.parse[(mid, iid)] = st
                        self.info[(mid, iid)] = info
                    pg.advance(t)
        if missing:
            msg = f"{len(missing)} model call(s) not done yet (e.g. {missing[:3]})."
            if not self.args.allow_incomplete:
                raise SystemExit(msg + " Finish the run first, or use --allow-incomplete "
                                       "(items without all models are then left out of every comparison).")
            console.print(f"[yellow]{msg} Items missing for any model are excluded.[/yellow]")
            bad_items = {iid for _, iid in missing}
            self.primary = [i for i in self.primary if i not in bad_items]
            self.batch = [i for i in self.batch if i not in bad_items]
        self.feat = {iid: L.item_features(self.items[iid]) for iid in self.primary + self.batch}
        self.deriv = {iid: L.gold_one_hop_derivable(self.items[iid].gold) for iid in self.primary + self.batch}
        self.build_pairs()

    def build_pairs(self):
        rows = []
        for iid in self.primary + self.batch:
            it = self.items[iid]
            f = self.feat[iid]
            n = it.n
            iu, ju = np.triu_indices(n, 1)
            G = it.gold
            gold = G[iu, ju].astype(int)
            pos_i, pos_j = f.pos[iu], f.pos[ju]
            sd = np.abs(f.sent[iu] - f.sent[ju])
            text_order = np.where(pos_i <= pos_j, 1, -1)
            rev = np.full(len(iu), np.nan)
            ok = np.isin(gold, [1, -1]) & ~np.isnan(pos_i) & ~np.isnan(pos_j)
            first_is_i = gold == 1
            pf = np.where(first_is_i, pos_i, pos_j)
            ps = np.where(first_is_i, pos_j, pos_i)
            rev[ok] = (pf[ok] > ps[ok]).astype(float)
            base = pd.DataFrame({
                "item_id": iid, "stem_id": it.stem_id, "is_primary": it.is_primary, "n": n,
                "i": iu, "j": ju, "ev_i": [it.event_ids[k] for k in iu], "ev_j": [it.event_ids[k] for k in ju],
                "gold": gold, "sent_dist": sd, "char_dist": np.abs(pos_i - pos_j),
                "relpos_i": f.rel_pos[iu], "relpos_j": f.rel_pos[ju],
                "reversed": rev, "text_order_pred": text_order,
                "derivable": self.deriv[iid][iu, ju],
                "dur_flag": f.dur_event[iu] | f.dur_event[ju],
                "gold_item_consistent": f.gold_consistent,
                "in_bad_gold_triple": [(a, b) in f.bad_triple_pairs for a, b in zip(iu, ju)],
                "meets_chain_ac": [(a, b) in f.meets_chain_ac for a, b in zip(iu, ju)],
                "words": f.words, "nonlin_share": f.nonlin_share,
            })
            for mid in self.models:
                M = self.pred[(mid, iid)]
                d = base.copy()
                d["model"] = mid
                d["pred"] = M[iu, ju].astype(int)
                d["pred_low"] = M[ju, iu].astype(int)
                rows.append(d)
        P = pd.concat(rows, ignore_index=True)
        P["dist_bin"] = L.dist_bin(P.sent_dist.values)
        P["correct"] = (P.pred == P.gold)
        self.pairs = P
        self.pp = P[P.is_primary].copy()          # primary pairs, all models
        self.stems_primary = sorted(self.pp.stem_id.unique())

    # ---------------------------------------------------------------- matrices
    def upper_matrix(self, mid: str, iid: str) -> np.ndarray:
        M = self.pred[(mid, iid)]
        n = M.shape[0]
        U = np.full((n, n), MISSING, dtype=np.int16)
        iu = np.triu_indices(n, 1)
        U[iu] = M[iu]
        return full_matrix_from_upper(U)


# ======================================================================= A1
def step_A1(ctx: Ctx):
    P = ctx.pp
    B = ctx.args.bootstrap
    stems = ctx.stems_primary
    sidx = {s: k for k, s in enumerate(stems)}
    W = L.bootstrap_weights(len(stems), B, seed=20261006)
    systems = {}
    for mid in ctx.models:
        d = P[P.model == mid]
        systems[mid] = (d.gold.values, d.pred.values, d.stem_id.values)
    d0 = P[P.model == ctx.models[0]]
    systems["baseline: always <"] = (d0.gold.values, np.ones(len(d0), dtype=int), d0.stem_id.values)
    systems["baseline: text order"] = (d0.gold.values, d0.text_order_pred.values, d0.stem_id.values)
    main_rows, lab_rows, boot_store, conf_rows = [], [], {}, []
    for name, (g, p, st) in systems.items():
        C = L.confusion(g, p)
        m = L.metrics_from_conf(C)
        m7 = L.metrics_simple(L.collapse7(g), L.collapse7(p), [0, 1, 2, 3, 4, 5, 6])
        m4 = L.metrics_simple(L.collapse4(g), L.collapse4(p), [-1, 0, 1, 7])
        # per-stem confusion stack for bootstrap
        S = np.zeros((len(stems), L.NL, L.NL + 1))
        gi = np.searchsorted(L.LABELS, g)
        pi = np.where(p == MISSING, L.MISS_COL, np.searchsorted(L.LABELS, np.where(p == MISSING, 0, p)))
        np.add.at(S, (np.vectorize(sidx.get)(st), gi, pi), 1)
        Cb = (W @ S.reshape(len(stems), -1)).reshape(B, L.NL, L.NL + 1)
        mb = L.metrics_from_conf(Cb)
        boot_store[name] = mb
        row = {"system": name, "n_pairs": int(m["n_pairs"]),
               "accuracy": m["accuracy"], "acc_lo": np.percentile(mb["accuracy"], 2.5), "acc_hi": np.percentile(mb["accuracy"], 97.5),
               "macro_f1": m["macro_f1"], "f1_lo": np.percentile(mb["macro_f1"], 2.5), "f1_hi": np.percentile(mb["macro_f1"], 97.5),
               "kappa": m["kappa"], "coverage": m["coverage"],
               "ta_precision": m["ta_precision"], "ta_recall": m["ta_recall"], "ta_f1": m["ta_f1"],
               "macro_f1_7way": m7["macro_f1"], "acc_7way": m7["accuracy"],
               "macro_f1_4way": m4["macro_f1"], "acc_4way": m4["accuracy"]}
        main_rows.append(row)
        for k, c in enumerate(L.LABELS):
            lab_rows.append({"system": name, "relation": CODE2SYM[c], "code": c, "name": CODE2NAME[c],
                             "support": int(m["support"][k]), "precision": m["per_label_p"][k],
                             "recall": m["per_label_r"][k], "f1": m["per_label_f1"][k]})
        if not name.startswith("baseline"):
            for a, ca in enumerate(L.LABELS):
                for b in range(L.NL + 1):
                    conf_rows.append({"model": name, "gold": CODE2SYM[ca],
                                      "pred": CODE2SYM[L.LABELS[b]] if b < L.NL else "missing",
                                      "count": int(C[a, b])})
    main = pd.DataFrame(main_rows)
    save_csv(main, "A1_main_metrics.csv")
    save_csv(pd.DataFrame(lab_rows), "A1_per_relation.csv")
    save_csv(pd.DataFrame(conf_rows), "A1_confusion_long.csv")
    # paired model comparisons (macro-F1 and accuracy), Holm-corrected
    comp = []
    names = ctx.models
    for metric in ("macro_f1", "accuracy"):
        rows = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                da = boot_store[names[a]][metric] - boot_store[names[b]][metric]
                obs = main.loc[main.system == names[a], metric].item() - main.loc[main.system == names[b], metric].item()
                # two-sided bootstrap p-value with the usual +1 correction (resolution 1/(B+1))
                pv = min(1.0, 2 * min(((da <= 0).sum() + 1) / (B + 1), ((da >= 0).sum() + 1) / (B + 1)))
                rows.append({"metric": metric, "model_a": names[a], "model_b": names[b], "diff": obs,
                             "diff_lo": np.percentile(da, 2.5), "diff_hi": np.percentile(da, 97.5), "p_boot": pv})
        if rows:
            adj = L.holm([r["p_boot"] for r in rows])
            for r, pa in zip(rows, adj):
                r["p_holm"] = pa
        comp.extend(rows)
    save_csv(pd.DataFrame(comp), "A1_pairwise_model_comparisons.csv")
    return {"main": main}


# ======================================================================= A2
def step_A2(ctx: Ctx):
    P = ctx.pp
    rows, rel_rows, dist_rows = [], [], []
    for mid in ctx.models:
        d = P[P.model == mid]
        both = (d.pred != MISSING) & (d.pred_low != MISSING)
        viol = both & (d.pred_low != -d.pred)
        low_as_upper = np.where(d.pred_low == MISSING, MISSING, -d.pred_low)
        mlow = L.metrics_from_conf(L.confusion(d.gold.values, low_as_upper))
        both_right = ((d.pred == d.gold) & (low_as_upper == d.gold)).mean()
        rows.append({"model": mid, "pairs_both_valid": int(both.sum()),
                     "converse_violation_rate": viol.sum() / max(both.sum(), 1),
                     "upper_accuracy": (d.pred == d.gold).mean(),
                     "lower_accuracy": mlow["accuracy"], "lower_macro_f1": mlow["macro_f1"],
                     "both_directions_right": both_right})
        for c in L.LABELS:
            s = both & (d.gold == c)
            rel_rows.append({"model": mid, "gold_relation": CODE2SYM[c], "pairs": int(s.sum()),
                             "violation_rate": (viol & s).sum() / max(s.sum(), 1)})
        for _, _, b in L.DIST_BINS:
            s = both & (d.dist_bin == b)
            dist_rows.append({"model": mid, "dist_bin": b, "pairs": int(s.sum()),
                              "violation_rate": (viol & s).sum() / max(s.sum(), 1)})
    save_csv(pd.DataFrame(rows), "A2_converse_summary.csv")
    save_csv(pd.DataFrame(rel_rows), "A2_converse_by_gold_relation.csv")
    save_csv(pd.DataFrame(dist_rows), "A2_converse_by_distance.csv")


# ======================================================================= A3 (checks)
def step_A3(ctx: Ctx):
    rows = []
    for iid in ctx.primary:
        it = ctx.items[iid]
        gb, gc = inconsistent_triples(it.gold)
        rows.append({"system": "gold", "item_id": iid, "n": it.n, "complete": True,
                     "satisfiable": is_satisfiable(it.gold), "bad_triples": gb, "checked_triples": gc})
        for mid in ctx.models:
            M = ctx.upper_matrix(mid, iid)
            b, c = inconsistent_triples(M)
            rows.append({"system": mid, "item_id": iid, "n": it.n,
                         "complete": bool((M[~np.eye(it.n, dtype=bool)] != MISSING).all()),
                         "satisfiable": is_satisfiable(M), "bad_triples": b, "checked_triples": c})
    D = pd.DataFrame(rows)
    save_csv(D, "A3_consistency_per_item.csv")
    S = D.groupby("system").agg(stems=("item_id", "count"), consistent_stems=("satisfiable", "sum"),
                                complete_stems=("complete", "sum"), bad_triples=("bad_triples", "sum"),
                                checked_triples=("checked_triples", "sum")).reset_index()
    S["pct_consistent"] = S.consistent_stems / S.stems
    S["triple_violation_rate"] = S.bad_triples / S.checked_triples.clip(lower=1)
    comp = D[D.complete].groupby("system").satisfiable.mean().rename("pct_consistent_among_complete")
    S = S.merge(comp, on="system", how="left")
    order = {m: k for k, m in enumerate(["gold"] + ctx.models)}
    S = S.sort_values("system", key=lambda s: s.map(order))
    save_csv(S, "A3_consistency_summary.csv")


# ======================================================================= A4
def step_A4(ctx: Ctx):
    P = ctx.pp
    w = P[(P.pred != MISSING) & (P.pred != P.gold)]
    sev = np.zeros(len(w), dtype=int)
    which = np.zeros((len(w), 4), dtype=bool)
    sig = {c: np.array(v) for c, v in ENDPOINTS.items()}
    gp = np.stack([sig[c] for c in w.gold.values]) if len(w) else np.zeros((0, 4))
    pp = np.stack([sig[c] for c in w.pred.values]) if len(w) else np.zeros((0, 4))
    if len(w):
        which = gp != pp
        sev = which.sum(1)
    w = w.assign(severity=sev, **{f"diff_{n.replace('-', '_')}": which[:, k] for k, n in enumerate(ENDPOINT_NAMES)})
    s1 = w.groupby(["model", "severity"]).size().rename("errors").reset_index()
    s1["share"] = s1.errors / s1.groupby("model").errors.transform("sum")
    save_csv(s1, "A4_error_severity.csv")
    cols = [f"diff_{n.replace('-', '_')}" for n in ENDPOINT_NAMES]
    s2 = w.groupby("model")[cols].mean().reindex(ctx.models).reset_index()
    save_csv(s2, "A4_endpoint_comparison_error_share.csv")
    s3 = w.assign(gold_rel=w.gold.map(CODE2SYM)).groupby(["model", "gold_rel"]).agg(
        errors=("severity", "size"), mean_severity=("severity", "mean"),
        share_severity1=("severity", lambda x: (x == 1).mean())).reset_index()
    save_csv(s3, "A4_severity_by_gold_relation.csv")


# ======================================================================= A5
def step_A5(ctx: Ctx):
    rows = []
    per_pair = []
    for iid in ctx.primary:
        it = ctx.items[iid]
        G = it.gold.astype(int)
        n = it.n
        idx = np.arange(n)
        f = ctx.feat[iid]
        for mid in ctx.models:
            Pm = ctx.upper_matrix(mid, iid).astype(int)
            iu, ju = np.triu_indices(n, 1)
            wrong = Pm[iu, ju] != G[iu, ju]
            # composition-available: exists k with P[i,k]==G[i,k], P[k,j]==G[k,j], unique G-comp == G[i,j]
            okik = (Pm == G) & (G != MISSING)                 # (i,k)
            Gv = np.where(G == MISSING, 0, G + 6)
            Sg = COMP_SINGLE[Gv[:, :, None], Gv[None, :, :]]   # [i,k,j]
            Sg = np.where((G[:, :, None] != MISSING) & (G[None, :, :] != MISSING), Sg, MISSING)
            avail = okik[:, :, None] & okik[None, :, :] & (Sg == G[:, None, :])
            avail[idx, idx, :] = False
            avail[:, idx, idx] = False
            comp_av = avail.any(axis=1)[iu, ju]
            # self-entailment: model's own premises entail a unique relation different from its answer
            Pv = np.where(Pm == MISSING, 0, Pm + 6)
            Sp = COMP_SINGLE[Pv[:, :, None], Pv[None, :, :]]
            Sp = np.where((Pm[:, :, None] != MISSING) & (Pm[None, :, :] != MISSING), Sp, MISSING)
            ent = (Sp != MISSING) & (Sp != Pm[:, None, :]) & (Pm[:, None, :] != MISSING)
            ent[idx, idx, :] = False
            ent[:, idx, idx] = False
            self_ent = ent.any(axis=1)[iu, ju]
            sd = np.abs(f.sent[iu] - f.sent[ju])
            per_pair.append(pd.DataFrame({"model": mid, "item_id": iid, "wrong": wrong,
                                          "comp_available": comp_av & wrong,
                                          "self_entailment_violation": self_ent,
                                          "pred_valid": Pm[iu, ju] != MISSING,
                                          "local": sd <= 1}))
    D = pd.concat(per_pair, ignore_index=True)
    for mid, d in D.groupby("model", sort=False):
        w = d[d.wrong]
        rows.append({"model": mid, "pairs": len(d), "errors": len(w),
                     "share_errors_composition_available": w.comp_available.mean() if len(w) else np.nan,
                     "self_entailment_violation_rate_all_valid": d[d.pred_valid].self_entailment_violation.mean(),
                     "error_rate_local_pairs(sent_dist<=1)": d[d.local].wrong.mean(),
                     "error_rate_distant_pairs(sent_dist>=2)": d[~d.local].wrong.mean(),
                     "share_comp_available_among_local_errors": w[w.local].comp_available.mean() if w.local.any() else np.nan,
                     "share_comp_available_among_distant_errors": w[~w.local].comp_available.mean() if (~w.local).any() else np.nan})
    S = pd.DataFrame(rows)
    save_csv(S, "A5_reading_vs_reasoning.csv")
    # error decomposition for the figure
    dec = []
    for mid, d in D.groupby("model", sort=False):
        w = d[d.wrong]
        n_err = len(w)
        ca = int(w.comp_available.sum())
        se = int((w.self_entailment_violation & ~w.comp_available).sum())
        dec.append({"model": mid, "composition_available": ca, "self_entailment_only": se,
                    "other": n_err - ca - se, "errors": n_err})
    save_csv(pd.DataFrame(dec), "A5_error_decomposition.csv")


# ======================================================================= A6
def gee_reversed(d: pd.DataFrame):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = d.copy()
    d["correct_i"] = d.correct.astype(int)
    d["rev_i"] = d.reversed.astype(int)
    d["gold_c"] = d.gold.astype(str)
    if d.correct_i.nunique() < 2 or d.rev_i.nunique() < 2:
        return None
    mod = smf.gee("correct_i ~ rev_i + C(dist_bin) + C(gold_c)", groups="stem_id", data=d,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    r = mod.fit()
    b = r.params["rev_i"]
    se = r.bse["rev_i"]
    return {"coef_reversed": b, "se": se, "odds_ratio": np.exp(b), "or_lo": np.exp(b - 1.96 * se),
            "or_hi": np.exp(b + 1.96 * se), "p": r.pvalues["rev_i"], "n": len(d), "clusters": d.stem_id.nunique()}


def nonlin_bin(x: np.ndarray) -> np.ndarray:
    return np.select([x == 0, x <= 0.10, x <= 0.20], ["0", "(0,10%]", "(10,20%]"], ">20%")


def step_A6(ctx: Ctx, subset_name: str = "all", P: pd.DataFrame = None, save: bool = True):
    P = ctx.pp if P is None else P
    rows, bins = [], []
    tb = P[P.reversed.notna()]
    for mid in ctx.models:
        d = tb[tb.model == mid]
        rv = d[d.reversed == 1]
        io = d[d.reversed == 0]
        wrong_rev = rv[(rv.pred != rv.gold) & (rv.pred != MISSING)]
        row = {"subset": subset_name, "model": mid, "pairs_in_order": len(io), "pairs_reversed": len(rv),
               "acc_in_order": io.correct.mean(), "acc_reversed": rv.correct.mean(),
               "heuristic_reliance_rate": (wrong_rev.pred == wrong_rev.text_order_pred).mean() if len(wrong_rev) else np.nan}
        try:
            g = gee_reversed(d)
        except Exception as e:  # pragma: no cover
            g = None
            log.warning("GEE failed for %s: %s", mid, e)
        if g:
            row.update(g)
        rows.append(row)
    S = pd.DataFrame(rows)
    # stem-level non-linearity bins: macro-F1 (pooled confusion) per bin
    Pb = P.assign(nl_bin=nonlin_bin(P.nonlin_share.values))
    for mid in ctx.models + ["baseline: text order"]:
        src = Pb[Pb.model == (mid if not mid.startswith("baseline") else ctx.models[0])]
        for b in ["0", "(0,10%]", "(10,20%]", ">20%"]:
            d = src[src.nl_bin == b]
            if not len(d):
                continue
            pred = d.pred.values if not mid.startswith("baseline") else d.text_order_pred.values
            m = L.metrics_from_conf(L.confusion(d.gold.values, pred))
            bins.append({"subset": subset_name, "system": mid, "nonlinearity_bin": b, "stems": d.stem_id.nunique(),
                         "pairs": len(d), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
    if save:
        save_csv(S, "A6_narrative_order.csv")
        save_csv(pd.DataFrame(bins), "A6_by_nonlinearity_bin.csv")
    return S, pd.DataFrame(bins)


# ======================================================================= A7
def step_A7(ctx: Ctx):
    P = ctx.pp
    rows = []
    for mid in ctx.models:
        d = P[P.model == mid]
        for _, _, b in L.DIST_BINS:
            x = d[d.dist_bin == b]
            if not len(x):
                continue
            m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
            rows.append({"model": mid, "dist_bin": b, "pairs": len(x), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
            for k, c in enumerate(L.LABELS):
                rows[-1][f"f1_{CODE2SYM[c]}"] = m["per_label_f1"][k]
                rows[-1][f"support_{CODE2SYM[c]}"] = int(m["support"][k])
    save_csv(pd.DataFrame(rows), "A7_by_sentence_distance.csv")
    # event position (beginning / middle / end): accuracy over pairs involving the event
    ev = pd.concat([P[["model", "correct", "relpos_i"]].rename(columns={"relpos_i": "relpos"}),
                    P[["model", "correct", "relpos_j"]].rename(columns={"relpos_j": "relpos"})])
    ev["position"] = pd.cut(ev.relpos, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["beginning", "middle", "end"])
    pos = ev.groupby(["model", "position"], observed=True).correct.agg(["mean", "size"]).reset_index()
    pos.columns = ["model", "position", "accuracy", "pair_slots"]
    save_csv(pos, "A7_by_event_position.csv")
    # stem level
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    st_rows, reg_rows = [], []
    for mid in ctx.models:
        d = P[P.model == mid]
        g = d.groupby("item_id").agg(acc=("correct", "mean"), n=("n", "first"), words=("words", "first")).reset_index()
        g["macro_f1"] = [L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))["macro_f1"]
                         for _, x in d.groupby("item_id")]
        g["model"] = mid
        st_rows.append(g)
        for y in ("acc", "macro_f1"):
            r_w = spearmanr(g[y], g.words)
            r_n = spearmanr(g[y], g.n)
            gz = g.assign(zw=(g.words - g.words.mean()) / g.words.std(), zn=(g.n - g.n.mean()) / g.n.std())
            ols = smf.ols(f"{y} ~ zw + zn", data=gz).fit(cov_type="HC3")
            reg_rows.append({"model": mid, "outcome": y, "spearman_words": r_w[0], "p_words": r_w[1],
                             "spearman_events": r_n[0], "p_events": r_n[1],
                             "ols_coef_z_words": ols.params["zw"], "ols_p_words": ols.pvalues["zw"],
                             "ols_coef_z_events": ols.params["zn"], "ols_p_events": ols.pvalues["zn"],
                             "r2": ols.rsquared})
    save_csv(pd.concat(st_rows), "A7_stem_level_scores.csv")
    save_csv(pd.DataFrame(reg_rows), "A7_stem_level_regression.csv")
    # within-output drift: raw cell (i,j) accuracy by row position (generation order), all cells
    drift = []
    for mid in ctx.models:
        acc_by = {k: [0, 0] for k in range(10)}
        for iid in ctx.primary:
            M = ctx.pred[(mid, iid)]
            G = ctx.items[iid].gold
            n = G.shape[0]
            for i in range(n):
                dec = min(int(10 * i / n), 9)
                row_ok = (M[i] == G[i])
                row_ok[i] = False
                acc_by[dec][0] += int(row_ok.sum())
                acc_by[dec][1] += n - 1
        for k, (c, t) in acc_by.items():
            drift.append({"model": mid, "row_position_decile": k + 1, "cells": t, "accuracy": c / t if t else np.nan})
    save_csv(pd.DataFrame(drift), "A7_within_output_drift.csv")
    # truncation by event count
    tr = []
    for mid in ctx.models:
        for iid in ctx.primary:
            tr.append({"model": mid, "n": ctx.items[iid].n, "truncated": bool(ctx.info[(mid, iid)]["truncated"])})
    T = pd.DataFrame(tr)
    T["n_bin"] = pd.cut(T.n, [0, 15, 25, 40, 1000], labels=["8-15", "16-25", "26-40", "41+"])
    save_csv(T.groupby(["model", "n_bin"], observed=True).truncated.agg(["mean", "size"]).reset_index()
             .rename(columns={"mean": "truncation_rate", "size": "stems"}), "A7_truncation_by_event_count.csv")
    # F1 by event-count bin (pair level, pooled)
    nb = []
    Pn = P.assign(n_bin=pd.cut(P.n, [0, 15, 25, 40, 1000], labels=["8-15", "16-25", "26-40", "41+"]))
    for mid in ctx.models:
        for b, x in Pn[Pn.model == mid].groupby("n_bin", observed=True):
            m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
            nb.append({"model": mid, "n_bin": b, "stems": x.item_id.nunique(), "pairs": len(x),
                       "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    save_csv(pd.DataFrame(nb), "A7_by_event_count.csv")


# ======================================================================= A8
def step_A8(ctx: Ctx, min_errors: int = 5):
    rows = []
    for mid in ctx.models:
        for iid in ctx.primary:
            n = ctx.items[iid].n
            M = ctx.upper_matrix(mid, iid)
            G = ctx.items[iid].gold
            iu, ju = np.triu_indices(n, 1)
            wrong = M[iu, ju] != G[iu, ju]
            ne = int(wrong.sum())
            if ne < min_errors:
                rows.append({"model": mid, "item_id": iid, "n": n, "errors": ne, "eligible": False})
                continue
            cnt = np.bincount(np.concatenate([iu[wrong], ju[wrong]]), minlength=n)
            order = np.argsort(-cnt)
            top1 = order[0]
            top2 = order[:2]
            inv1 = (iu[wrong] == top1) | (ju[wrong] == top1)
            inv2 = np.isin(iu[wrong], top2) | np.isin(ju[wrong], top2)
            rows.append({"model": mid, "item_id": iid, "n": n, "errors": ne, "eligible": True,
                         "top_event": ctx.items[iid].event_ids[top1],
                         "share_errors_top1_event": inv1.mean(), "share_errors_top2_events": inv2.mean(),
                         "error_rate_without_top1": (wrong & ~((iu == top1) | (ju == top1))).sum()
                         / max(((iu != top1) & (ju != top1)).sum(), 1),
                         "error_rate": ne / len(iu)})
    D = pd.DataFrame(rows)
    save_csv(D, "A8_error_concentration_per_item.csv")
    E = D[D.eligible]
    S = E.groupby("model", sort=False).agg(stems_with_5plus_errors=("item_id", "count"),
                                           mean_share_top1=("share_errors_top1_event", "mean"),
                                           mean_share_top2=("share_errors_top2_events", "mean"),
                                           pct_stems_top1_over_half=("share_errors_top1_event", lambda x: (x > 0.5).mean()),
                                           mean_error_rate=("error_rate", "mean"),
                                           mean_error_rate_without_top1=("error_rate_without_top1", "mean")).reset_index()
    save_csv(S, "A8_error_concentration_summary.csv")


# ======================================================================= A9
def step_A9(ctx: Ctx):
    P = ctx.pp
    rows = []
    for mid in ctx.models:
        d = P[P.model == mid]
        for name, s in [("gold-consistent stems", d.gold_item_consistent), ("gold-inconsistent stems", ~d.gold_item_consistent),
                        ("pairs in a bad gold triple", d.in_bad_gold_triple), ("all other pairs", ~d.in_bad_gold_triple)]:
            x = d[s]
            if len(x):
                m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
                rows.append({"model": mid, "subset": name, "stems": x.item_id.nunique(), "pairs": len(x),
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    save_csv(pd.DataFrame(rows), "A9_gold_consistency_subsets.csv")
    # meets-chain (A, C) pairs: what do models predict?
    mc = P[P.meets_chain_ac]
    if len(mc):
        dist = []
        for mid in ctx.models:
            x = mc[mc.model == mid]
            # orientation: gold is m (2) or mi (-2); "logically required" is < or > with the same sign
            req = np.sign(x.gold.values)
            pred = x.pred.values
            dist.append({"model": mid, "ac_pairs": len(x),
                         "pred_same_as_gold_meets": (pred == x.gold.values).mean(),
                         "pred_logically_required_before_after": (pred == req).mean(),
                         "pred_other": 1 - (pred == x.gold.values).mean() - (pred == req).mean()})
        save_csv(pd.DataFrame(dist), "A9_meets_chain_ac_predictions.csv")
    # candidate gold errors: >= M-1 models agree on the same label, different from gold
    M = len(ctx.models)
    wide = P.pivot_table(index=["item_id", "i", "j"], columns="model", values="pred", aggfunc="first")[ctx.models]
    gold_al = P[P.model == ctx.models[0]].set_index(["item_id", "i", "j"]).gold.reindex(wide.index).values
    vals = wide.values
    counts = np.stack([(vals == c).sum(1) for c in L.LABELS], axis=1)   # (N, 13), missing never counted
    top = np.array(L.LABELS)[counts.argmax(1)]
    cnt = counts.max(1)
    sel = (cnt >= max(M - 1, 2)) & (top != gold_al)
    cand = [(idx, int(t), int(c), int(g)) for idx, t, c, g, s_ in zip(wide.index, top, cnt, gold_al, sel) if s_]
    rows = []
    for (iid, i, j), top, cnt, g in cand:
        it = ctx.items[iid]
        rows.append({"item_id": iid, "ev_i": it.event_ids[i], "text_i": it.event_texts[i],
                     "ev_j": it.event_ids[j], "text_j": it.event_texts[j],
                     "gold": CODE2SYM[g], "model_consensus": CODE2SYM[top], "n_models_agreeing": cnt,
                     "gold_item_consistent": ctx.feat[iid].gold_consistent})
    C = pd.DataFrame(rows)
    save_csv(C, "A9_candidate_gold_errors.csv")
    INPUTS.mkdir(parents=True, exist_ok=True)
    samp_path = INPUTS / "A9_gold_error_audit_sample.csv"
    if len(C) and not samp_path.exists():
        S = C.sample(n=min(50, len(C)), random_state=20261006).copy()
        S["passage"] = [ctx.items[i].stem_text for i in S.item_id]
        S["is_gold_error (yes/no/unsure)"] = ""
        S["notes"] = ""
        buf = io.StringIO()
        S.to_csv(buf, index=False, lineterminator="\n")
        atomic_write_text(samp_path, buf.getvalue())
    if samp_path.exists():
        try:
            S = pd.read_csv(samp_path)
            col = "is_gold_error (yes/no/unsure)"
            filled = S[col].dropna().astype(str).str.strip().str.lower()
            filled = filled[filled != ""]
            if len(filled):
                save_csv(pd.DataFrame([{"audited": len(filled), "yes": (filled == "yes").sum(),
                                        "no": (filled == "no").sum(), "unsure": (filled == "unsure").sum(),
                                        "share_yes": (filled == "yes").mean()}]), "A9_gold_error_audit_result.csv")
        except Exception as e:
            log.warning("could not read audit sample: %s", e)
    # batch versions
    if ctx.batch:
        B = ctx.pairs[~ctx.pairs.is_primary]
        rows = []
        for mid in ctx.models:
            for iid in ctx.batch:
                x = B[(B.model == mid) & (B.item_id == iid)]
                prim = ctx.pp[(ctx.pp.model == mid) & (ctx.pp.stem_id == ctx.items[iid].stem_id)]
                rows.append({"model": mid, "batch_item": iid, "stem_id": ctx.items[iid].stem_id,
                             "batch_pairs": len(x), "batch_accuracy": x.correct.mean(),
                             "primary_pairs": len(prim), "primary_accuracy": prim.correct.mean()})
        save_csv(pd.DataFrame(rows), "A9_batch_versions.csv")


# ======================================================================= A10
def ensemble_predictions(ctx: Ctx, P: pd.DataFrame) -> pd.DataFrame:
    """Plurality vote per cell over valid model answers. Ties are broken by the overall
    frequency of each label across all model predictions (gold-free); a cell with no valid
    answer from any model stays MISSING."""
    wide = P.pivot_table(index=["item_id", "i", "j"], columns="model", values="pred", aggfunc="first")[ctx.models]
    vals = wide.values
    allp = P.pred[P.pred != MISSING]
    freq = allp.value_counts()
    labs = np.array(L.LABELS)
    fr = np.array([freq.get(c, 0) for c in labs], dtype=float)
    # rank-based tie-breaker strictly below 1 vote: higher overall frequency wins a tie
    order = np.argsort(np.argsort(fr))            # 0 = rarest ... 12 = most frequent
    tieb = order / (len(labs) * 10.0)
    counts = np.stack([(vals == c).sum(1) for c in labs], axis=1).astype(float)
    score = counts + tieb[None, :]
    pred = labs[score.argmax(1)]
    pred = np.where(counts.sum(1) == 0, MISSING, pred)
    return pd.DataFrame({"item_id": [k[0] for k in wide.index], "i": [k[1] for k in wide.index],
                         "j": [k[2] for k in wide.index], "pred": pred.astype(int)})


def step_A10(ctx: Ctx):
    P = ctx.pp
    wide = P.pivot_table(index=["item_id", "i", "j"], columns="model", values="pred", aggfunc="first")[ctx.models]
    gold = P[P.model == ctx.models[0]].set_index(["item_id", "i", "j"]).loc[wide.index, "gold"].values
    rows = []
    for a in ctx.models:
        for b in ctx.models:
            va, vb = wide[a].values, wide[b].values
            ok = (va != MISSING) & (vb != MISSING)
            k = L.cohen_kappa_codes(va[ok], vb[ok]) if a != b else 1.0
            rows.append({"model_a": a, "model_b": b, "kappa": k, "pairs": int(ok.sum())})
        ok = wide[a].values != MISSING
        rows.append({"model_a": a, "model_b": "human gold", "kappa": L.cohen_kappa_codes(wide[a].values[ok], gold[ok]),
                     "pairs": int(ok.sum())})
    save_csv(pd.DataFrame(rows), "A10_pairwise_kappa.csv")
    allv = (wide.values != MISSING).all(1)
    fk = L.fleiss_kappa(wide.values[allv].astype(int)) if allv.sum() and len(ctx.models) > 1 else np.nan
    E = ensemble_predictions(ctx, P)
    save_csv(E, "A10_ensemble_predictions.csv")
    me = L.metrics_from_conf(L.confusion(gold, E.pred.values))
    oracle = (wide.values == gold[:, None]).any(1).mean()
    ncorrect = (wide.values == gold[:, None]).sum(1)
    S = pd.DataFrame([{"fleiss_kappa_all_models": fk, "pairs_all_valid": int(allv.sum()),
                       "ensemble_accuracy": me["accuracy"], "ensemble_macro_f1": me["macro_f1"],
                       "oracle_any_model_correct": oracle, "pairs": len(gold),
                       "pairs_no_model_correct": float((ncorrect == 0).mean()),
                       "pairs_all_models_correct": float((ncorrect == len(ctx.models)).mean())}])
    save_csv(S, "A10_summary.csv")
    # item difficulty
    meta = P[P.model == ctx.models[0]].set_index(["item_id", "i", "j"]).loc[wide.index]
    D = pd.DataFrame({"n_models_correct": ncorrect, "gold": meta.gold.map(CODE2SYM).values,
                      "dist_bin": meta.dist_bin.values, "reversed": meta.reversed.values,
                      "derivable": meta.derivable.values})
    out = []
    for fac in ("gold", "dist_bin", "reversed", "derivable"):
        g = D.groupby(fac, dropna=False).n_models_correct.agg(["mean", "size"]).reset_index()
        g.columns = ["level", "mean_models_correct", "pairs"]
        g.insert(0, "factor", fac)
        g["level"] = g.level.astype(str)
        out.append(g)
    save_csv(pd.concat(out), "A10_difficulty_by_factor.csv")
    save_csv(D.n_models_correct.value_counts().sort_index().rename_axis("n_models_correct")
             .reset_index(name="pairs"), "A10_difficulty_distribution.csv")


# ======================================================================= A11
def step_A11(ctx: Ctx):
    INPUTS.mkdir(parents=True, exist_ok=True)
    p = INPUTS / "model_attributes.csv"
    if not p.exists():
        rows = []
        for mid in ctx.models:
            mm = ctx.manifest["models"][mid]
            rows.append({"model_id": mid, "model_key": mm["model_key"],
                         "reasoning_param_sent": "reasoning" in mm["request_params"],
                         "open_weights (yes/no)": "", "parameters_billion": "", "notes": ""})
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False, lineterminator="\n")
        atomic_write_text(p, buf.getvalue())
    A = pd.read_csv(p)
    main = pd.read_csv(TAB / "A1_main_metrics.csv")
    D = A.merge(main.rename(columns={"system": "model_id"})[["model_id", "accuracy", "macro_f1", "kappa"]],
                on="model_id", how="left")
    save_csv(D, "A11_model_attributes_descriptive.csv")


# ======================================================================= A12
def step_A12(ctx: Ctx):
    main = pd.read_csv(TAB / "A1_main_metrics.csv").set_index("system")
    rows = []
    for mid in ctx.models:
        infos = [ctx.info[(mid, i)] for i in ctx.primary]
        parses = [ctx.parse[(mid, i)] for i in ctx.primary]

        def tot(k):
            v = [x[k] for x in infos if x.get(k) is not None]
            return float(np.sum(v)) if v else np.nan

        cost = tot("cost")
        rows.append({"model": mid, "calls": len(infos),
                     "parse_complete": np.mean([p["status"] == "complete" for p in parses]),
                     "parse_partial": np.mean([p["status"] == "partial" for p in parses]),
                     "parse_failed": np.mean([p["status"] == "failed" for p in parses]),
                     "truncated": np.mean([bool(x["truncated"]) for x in infos]),
                     "cell_coverage_all_cells": np.sum([p["cells_valid"] for p in parses]) / np.sum([p["cells_total"] for p in parses]),
                     "prompt_tokens": tot("prompt_tokens"), "completion_tokens": tot("completion_tokens"),
                     "reasoning_tokens_reported": tot("reasoning_tokens"), "reasoning_chars": tot("reasoning_chars"),
                     "cost_usd": cost, "latency_mean_s": np.nanmean([x["latency_s"] or np.nan for x in infos]),
                     "latency_median_s": np.nanmedian([x["latency_s"] or np.nan for x in infos]),
                     "retried_calls": int(np.sum([(x.get("attempts_total") or 1) > 1 for x in infos])),
                     "macro_f1": main.loc[mid, "macro_f1"],
                     "macro_f1_per_usd": main.loc[mid, "macro_f1"] / cost if cost and cost > 0 else np.nan})
        prov = pd.Series([x["provider"] for x in infos]).value_counts()
        rows[-1]["providers"] = "; ".join(f"{k}:{v}" for k, v in prov.items())
    save_csv(pd.DataFrame(rows), "A12_reliability_and_cost.csv")


# ======================================================================= A13
def step_A13(ctx: Ctx):
    P = ctx.pp
    rows = []
    for mid in ctx.models:
        d = P[P.model == mid]
        for flag in (True, False):
            x = d[d.dur_flag == flag]
            if len(x):
                m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
                rows.append({"model": mid, "duration_phrase_in_either_sentence": flag, "pairs": len(x),
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    save_csv(pd.DataFrame(rows), "A13_duration_flag.csv")
    INPUTS.mkdir(parents=True, exist_ok=True)
    p = INPUTS / "A13_arithmetic_tagging_sample.csv"
    if not p.exists():
        x = P[(P.model == ctx.models[0]) & P.dur_flag].sample(n=min(100, int(P[(P.model == ctx.models[0]) & P.dur_flag].shape[0])),
                                                                random_state=20261006)
        S = pd.DataFrame({"item_id": x.item_id.values, "i": x.i.values, "j": x.j.values,
                          "ev_i": x.ev_i.values, "ev_j": x.ev_j.values,
                          "text_i": [ctx.items[a].event_texts[b] for a, b in zip(x.item_id, x.i)],
                          "text_j": [ctx.items[a].event_texts[b] for a, b in zip(x.item_id, x.j)],
                          "gold": x.gold.map(CODE2SYM).values,
                          "passage": [ctx.items[a].stem_text for a in x.item_id],
                          "requires_arithmetic (yes/no)": ""})
        buf = io.StringIO()
        S.to_csv(buf, index=False, lineterminator="\n")
        atomic_write_text(p, buf.getvalue())
    S = pd.read_csv(p)
    col = "requires_arithmetic (yes/no)"
    tagged = S[S[col].astype(str).str.strip().str.lower().isin(["yes", "no"])]
    if len(tagged):
        tagged = tagged.assign(req=tagged[col].str.strip().str.lower() == "yes")
        m = P.merge(tagged[["item_id", "i", "j", "req"]], on=["item_id", "i", "j"])
        out = m.groupby(["model", "req"]).correct.agg(["mean", "size"]).reset_index()
        out.columns = ["model", "requires_arithmetic", "accuracy", "pairs"]
        save_csv(out, "A13_arithmetic_tagged_accuracy.csv")


# ======================================================================= A14
def step_A14(ctx: Ctx):
    INPUTS.mkdir(parents=True, exist_ok=True)
    p = INPUTS / "stem_types.csv"
    if not p.exists():
        rows = [{"stem_id": ctx.items[i].stem_id, "start_of_passage": ctx.items[i].stem_text[:160].replace("\n", " "),
                 "type": ""} for i in ctx.primary]
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False, lineterminator="\n")
        atomic_write_text(p, buf.getvalue())
        return
    T = pd.read_csv(p)
    T = T[T.type.notna() & (T.type.astype(str).str.strip() != "")]
    if not len(T):
        return
    P = ctx.pp.merge(T[["stem_id", "type"]], on="stem_id")
    rows = []
    for (mid, ty), x in P.groupby(["model", "type"]):
        m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
        rows.append({"model": mid, "type": ty, "stems": x.stem_id.nunique(), "pairs": len(x),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    save_csv(pd.DataFrame(rows), "A14_by_passage_type.csv")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    reg = []
    for mid in ctx.models:
        d = P[P.model == mid].copy()
        d["correct_i"] = d.correct.astype(int)
        d["zw"] = (d.words - d.words.mean()) / d.words.std()
        d["zn"] = (d.n - d.n.mean()) / d.n.std()
        try:
            r = smf.gee("correct_i ~ C(type) + zw + zn + nonlin_share", groups="stem_id", data=d,
                        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
            for k in r.params.index:
                reg.append({"model": mid, "term": k, "coef": r.params[k], "se": r.bse[k], "p": r.pvalues[k]})
        except Exception as e:
            log.warning("A14 GEE failed for %s: %s", mid, e)
    save_csv(pd.DataFrame(reg), "A14_gee_type_with_covariates.csv")


# ======================================================================= robustness
def step_robustness(ctx: Ctx):
    P = ctx.pp
    subsets = {
        "all stems": P,
        "gold-consistent stems only": P[P.gold_item_consistent],
        "fully linear stems (no reversed pairs)": P[P.nonlin_share == 0],
        "non-linear stems": P[P.nonlin_share > 0],
        "excluding duplicate-span stems (548, 550)": P[~P.stem_id.isin([548, 550])],
    }
    rows, a6 = [], []
    for name, d in subsets.items():
        for mid in ctx.models:
            x = d[d.model == mid]
            m = L.metrics_from_conf(L.confusion(x.gold.values, x.pred.values))
            rows.append({"subset": name, "model": mid, "stems": x.item_id.nunique(), "pairs": len(x),
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "kappa": m["kappa"]})
        if name in ("all stems", "gold-consistent stems only", "excluding duplicate-span stems (548, 550)"):
            S, _ = step_A6(ctx, subset_name=name, P=d, save=False)
            a6.append(S)
    save_csv(pd.DataFrame(rows), "R_robustness_A1.csv")
    save_csv(pd.concat(a6), "R_robustness_A6.csv")


# ======================================================================= A3a repair (slow)
def _repair_cache_path(key: str) -> Path:
    return CACHE / "repair" / f"{key}.json"


def _child_init() -> None:
    """Worker processes ignore Ctrl+C; the main process stops them gracefully."""
    import signal as _sig
    _sig.signal(_sig.SIGINT, _sig.SIG_IGN)


def _repair_job(key: str, n: int, votes, time_limit: float, stall: float, workers: int) -> Dict:
    """Solve one network and save it to the cache (runs in a worker process)."""
    r = repair_nearest_consistent(n, votes, time_limit_s=time_limit, stall_s=stall, workers=workers)
    out = {k: v for k, v in r.items() if k != "matrix"}
    out["upper"] = None if r["matrix"] is None else r["matrix"][np.triu_indices(n, 1)].astype(int).tolist()
    atomic_write_json(_repair_cache_path(key), out)
    return out


def step_A3a(ctx: Ctx):
    jobs = []
    E = pd.read_csv(TAB / "A10_ensemble_predictions.csv")
    ens = {iid: g for iid, g in E.groupby("item_id")}
    for iid in ctx.primary:
        n = ctx.items[iid].n
        iu, ju = np.triu_indices(n, 1)
        for mid in ctx.models + ["ensemble"]:
            if mid == "ensemble":
                g = ens[iid].sort_values(["i", "j"])
                U = np.full((n, n), MISSING, dtype=np.int16)
                U[g.i.values, g.j.values] = g.pred.values
                up = U[iu, ju]
                lowc = None
            else:
                M = ctx.pred[(mid, iid)]
                up = M[iu, ju]
                low = M[ju, iu]
                lowc = np.where(low == MISSING, MISSING, -low)
            for variant in (["i"] if mid == "ensemble" else ["i", "ii"]):
                votes = [(int(a), int(b), int(c), 1) for a, b, c in zip(iu, ju, up) if c != MISSING]
                if variant == "ii":
                    votes += [(int(a), int(b), int(c), 1) for a, b, c in zip(iu, ju, lowc) if c != MISSING]
                # skip the solver when the network is already complete and consistent
                # (and, for variant ii, both triangles agree): the repair is then the identity
                complete = bool((up != MISSING).all())
                same = variant == "i" or (lowc is not None and np.array_equal(lowc, up))
                trivial = complete and same and is_satisfiable(full_matrix_from_upper(_to_full_upper(n, up)))
                jobs.append((mid, iid, variant, n, votes, up, trivial))
    # ---- solve (parallel processes; every solved network is cached immediately)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    done_r: Dict[str, Dict] = {}
    todo = []
    for mid, iid, variant, n, votes, up, trivial in jobs:
        key = f"{mid}__{iid}__{variant}"
        if trivial:
            done_r[key] = {"status": "ALREADY_CONSISTENT", "upper": up.astype(int).tolist(), "objective": len(votes),
                           "bound": len(votes), "wall_time": 0.0, "max_objective": len(votes), "stopped_by_stall": False}
            continue
        cp = _repair_cache_path(key)
        if cp.exists():
            try:
                done_r[key] = read_json(cp)
                continue
            except Exception:
                pass
        todo.append((key, n, votes))
    todo.sort(key=lambda x: -x[1])  # largest networks first for better load balance
    nproc = ctx.args.repair_processes or max(1, (os.cpu_count() or 4) // max(ctx.args.repair_workers, 1))
    console.print(f"A3a: {len(jobs)} networks, {len(jobs) - len(todo)} already done/trivial, "
                  f"{len(todo)} to solve with {nproc} parallel process(es). Ctrl+C stops after the running ones.")
    if todo:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                      TimeElapsedColumn(), TimeRemainingColumn(), console=console) as pg:
            t = pg.add_task("A3a nearest-consistent repair", total=len(todo))
            ex = ProcessPoolExecutor(max_workers=nproc, initializer=_child_init)
            futs = {}
            try:
                for key, n, votes in todo:
                    futs[ex.submit(_repair_job, key, n, votes, ctx.args.repair_time_limit,
                                   ctx.args.repair_stall, max(ctx.args.repair_workers, 1))] = key
                pending = set(futs)
                while pending:
                    fin, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for f in fin:
                        done_r[futs[f]] = f.result()
                        pg.advance(t)
                        pg.update(t, description=f"A3a repair  last: {futs[f]}")
            except KeyboardInterrupt:
                pg.update(t, description="A3a: Ctrl+C - waiting for the running networks to finish and save ...")
                ex.shutdown(wait=True, cancel_futures=True)
                raise
            ex.shutdown(wait=True)
    results = [(mid, iid, variant, done_r[f"{mid}__{iid}__{variant}"]) for mid, iid, variant, *_ in jobs]
    rows, stat_rows = [], []
    for mid, iid, variant, r in results:
        it = ctx.items[iid]
        iu, ju = np.triu_indices(it.n, 1)
        gold = it.gold[iu, ju].astype(int)
        rep = np.array(r["upper"]) if r["upper"] is not None else np.full(len(iu), MISSING)
        rows.append(pd.DataFrame({"model": mid, "item_id": iid, "variant": variant, "gold": gold, "repaired": rep}))
        stat_rows.append({"model": mid, "item_id": iid, "variant": variant, "status": r["status"],
                          "objective": r["objective"], "bound": r["bound"], "max_objective": r["max_objective"],
                          "wall_time": r["wall_time"], "stopped_by_stall": r.get("stopped_by_stall")})
    R = pd.concat(rows, ignore_index=True)
    ST = pd.DataFrame(stat_rows)
    save_csv(ST, "A3a_repair_solver_status.csv")
    E2 = E.copy()
    summ = []
    for (mid, variant), x in R.groupby(["model", "variant"], sort=False):
        if mid == "ensemble":
            raw = ctx.pp[ctx.pp.model == ctx.models[0]][["item_id", "i", "j", "gold"]].merge(
                E2, on=["item_id", "i", "j"], how="left")
            mraw = L.metrics_from_conf(L.confusion(raw.gold.values, raw.pred.fillna(MISSING).astype(int).values))
        else:
            d = ctx.pp[ctx.pp.model == mid]
            mraw = L.metrics_from_conf(L.confusion(d.gold.values, d.pred.values))
        mrep = L.metrics_from_conf(L.confusion(x.gold.values, x.repaired.values))
        st = ST[(ST.model == mid) & (ST.variant == variant)]
        summ.append({"model": mid, "variant": variant,
                     "raw_accuracy": mraw["accuracy"], "raw_macro_f1": mraw["macro_f1"],
                     "repaired_accuracy": mrep["accuracy"], "repaired_macro_f1": mrep["macro_f1"],
                     "delta_accuracy": mrep["accuracy"] - mraw["accuracy"],
                     "delta_macro_f1": mrep["macro_f1"] - mraw["macro_f1"],
                     "stems_already_consistent": int((st.status == "ALREADY_CONSISTENT").sum()),
                     "stems_optimal": int((st.status == "OPTIMAL").sum()),
                     "stems_feasible_not_proven": int((st.status == "FEASIBLE").sum()),
                     "stems_no_solution": int((~st.status.isin(["ALREADY_CONSISTENT", "OPTIMAL", "FEASIBLE"])).sum())})
    save_csv(pd.DataFrame(summ), "A3a_repair_summary.csv")


def _to_full_upper(n: int, up: np.ndarray) -> np.ndarray:
    U = np.full((n, n), MISSING, dtype=np.int16)
    U[np.triu_indices(n, 1)] = up
    return U


# ======================================================================= steps registry
STEPS = [
    ("A1", "agreement with the human matrix", step_A1),
    ("A2", "converse self-consistency", step_A2),
    ("A3", "logical consistency checks", step_A3),
    ("A4", "endpoint-level error anatomy", step_A4),
    ("A5", "reading vs reasoning", step_A5),
    ("A6", "narrative order", step_A6),
    ("A7", "distance, position, length, drift", step_A7),
    ("A8", "error concentration", step_A8),
    ("A9", "gold issues", step_A9),
    ("A10", "cross-model analysis", step_A10),
    ("A11", "model attributes", step_A11),
    ("A12", "reliability and cost", step_A12),
    ("A13", "durations", step_A13),
    ("A14", "passage type", step_A14),
    ("R", "robustness subsets", step_robustness),
    ("A3a", "offline repair (slow)", step_A3a),
    ("FIG", "figures", None),
    ("REPORT", "report", None),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--repair-time-limit", type=float, default=30.0, help="max seconds per network")
    ap.add_argument("--repair-stall", type=float, default=3.0, help="stop a network after N s without improvement")
    ap.add_argument("--repair-processes", type=int, default=0,
                    help="parallel solver processes (0 = CPU cores / 4, at least 1)")
    ap.add_argument("--repair-workers", type=int, default=4,
                    help="CP-SAT search workers per process (default 4; fewer gives clearly worse solutions)")
    ap.add_argument("--skip-repair", action="store_true", help="skip the slow A3a step for now")
    ap.add_argument("--force", action="store_true", help="redo every step (or only the --only steps)")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these steps (e.g. A1 A6 FIG REPORT)")
    args = ap.parse_args()

    for d in (OUT, TAB, FIG, CACHE, INPUTS, RESULTS / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "logs" / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)
    log.propagate = False
    import warnings
    warnings.filterwarnings("ignore")

    console.print(Panel("Spatio-Temporal NLP - offline analysis (no model calls)\n"
                        f"results: {RESULTS}\nlog: {log_path}\nCtrl+C stops after saving; run again to resume.",
                        box=box.ROUNDED))
    try:
        ctx = Ctx(args)
        fp = ctx.fingerprint()
        prog = read_json(PROGRESS) if PROGRESS.exists() else {}
        if prog.get("fingerprint") != fp:
            if prog:
                console.print("[yellow]Raw results or analysis version changed since the last analysis: redoing all steps.[/yellow]")
            prog = {"fingerprint": fp, "done": {}}
        if args.force:
            for code in (args.only or list(prog["done"])):
                prog["done"].pop(code, None)
        atomic_write_json(PROGRESS, prog)
        ctx.load()
        log.info("loaded %d models x %d primary items (+%d batch)", len(ctx.models), len(ctx.primary), len(ctx.batch))
        import figures
        import report
        for code, desc, fn in STEPS:
            if args.only and code not in args.only:
                continue
            if code == "A3a" and args.skip_repair:
                console.print("[dim]A3a skipped (--skip-repair)[/dim]")
                continue
            if prog["done"].get(code) and code not in ("FIG", "REPORT"):
                console.print(f"[dim]{code:<6} {desc} - already done[/dim]")
                continue
            t0 = time.time()
            if code == "A3a":
                fn(ctx)  # has its own progress bar
            else:
                with console.status(f"{code}: {desc} ..."):
                    if code == "FIG":
                        figures.make_all(TAB, FIG, ctx.models)
                    elif code == "REPORT":
                        report.write_report(OUT, TAB, FIG, ctx)
                    else:
                        fn(ctx)
            prog["done"][code] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(PROGRESS, prog)
            msg = f"{code:<6} {desc} - done in {time.time() - t0:.1f}s"
            console.print(f"[green]{msg}[/green]")
            log.info(msg)
    except KeyboardInterrupt:
        console.print("[yellow]Stopped by Ctrl+C. Finished steps are saved; run the same command to resume.[/yellow]")
        log.warning("stopped by Ctrl+C")
        return 130
    except SystemExit as e:
        if e.code not in (0, None):
            console.print(f"[red]{e.code}[/red]" if isinstance(e.code, str) else "")
            log.error(str(e.code))
            return 2
        raise
    except Exception:
        tb = traceback.format_exc()
        log.error(tb)
        console.print(Panel(tb, title="Analysis failed", style="red"))
        return 1
    console.print(Panel(f"Done. Report: {OUT / 'REPORT.md'}\nTables: {TAB}\nFigures: {FIG}", style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
