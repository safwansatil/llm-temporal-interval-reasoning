"""
Data assembly and metric functions used by analyze.py.

Conventions
- "Upper triangle": pairs (i, j) with i < j in event-list order (the gold ID order). The
  primary prediction for a pair is the model's cell (i, j); its cell (j, i) is the
  "lower-triangle" answer, used only in A2 (converse self-consistency) and the A3 variant ii.
- MISSING (99) = no valid symbol for that cell. In every main score a missing cell counts
  as wrong (it is a false negative for the gold label and never a false positive).
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from allen import CODE2SYM, CODES, COMP_SINGLE, MISSING, full_matrix_from_upper
from common import Item, parse_matrix, read_json

LABELS = CODES  # 13 Allen codes, fixed order -6..6
LAB_IDX = {c: k for k, c in enumerate(LABELS)}
NL = len(LABELS)
MISS_COL = NL  # extra prediction column for "missing"

# Duration / offset phrase heuristic (identical to the one used in the plan's data facts).
DUR_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)"
    r"\s+(minutes?|hours?|days?|weeks?|months?|years?)\b", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DIST_BINS = [(0, 0, "same"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10 ** 9, "8+")]


# ======================================================================= collapsing
def collapse7(c: np.ndarray) -> np.ndarray:
    """Direction ignored: |code| (0..6). MISSING stays MISSING."""
    return np.where(c == MISSING, MISSING, np.abs(c))


def collapse4(c: np.ndarray) -> np.ndarray:
    """4-way: before (1), after (-1), equal (0), overlap/contact family (7). MISSING stays."""
    out = np.where(np.isin(c, [1, -1, 0]), c, 7)
    return np.where(c == MISSING, MISSING, out)


# ======================================================================= item features
@dataclass
class ItemFeatures:
    item_id: str
    pos: np.ndarray            # char position of each event span (first occurrence)
    rel_pos: np.ndarray        # pos / len(text)
    sent: np.ndarray           # sentence index of each event
    n_sent: int
    dur_event: np.ndarray      # event's sentence contains a duration phrase
    words: int
    nonlin_share: float        # share of gold before/after pairs narrated in reverse order
    gold_full: np.ndarray
    gold_consistent: bool
    bad_triple_pairs: set      # upper pairs (i,j) involved in a gold-inconsistent triple
    meets_chain_ac: set        # upper pairs that are the (A,C) pair of a gold m-m-m chain


def item_features(it: Item) -> ItemFeatures:
    from allen import COMP_OK
    text = it.stem_text
    low = text.lower()
    pos = np.array([low.find(t.lower()) for t in it.event_texts], dtype=float)
    pos[pos < 0] = np.nan
    starts = [0] + [m.end() for m in SENT_SPLIT_RE.finditer(text)]
    sent_texts = []
    for k, s0 in enumerate(starts):
        s1 = starts[k + 1] if k + 1 < len(starts) else len(text)
        sent_texts.append(text[s0:s1])
    sent = np.array([bisect.bisect_right(starts, p) - 1 if not np.isnan(p) else -1 for p in pos])
    dur_sent = [bool(DUR_RE.search(s)) for s in sent_texts]
    dur_event = np.array([dur_sent[s] if s >= 0 else False for s in sent])
    G = it.gold
    n = it.n
    rev = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            g = int(G[i, j])
            if g in (1, -1) and not (np.isnan(pos[i]) or np.isnan(pos[j])):
                tot += 1
                first, second = (i, j) if g == 1 else (j, i)
                if pos[first] > pos[second]:
                    rev += 1
    bad = set()
    chain = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = int(G[i, j]), int(G[j, k]), int(G[i, k])
                if not COMP_OK[a + 6, b + 6, c + 6]:
                    bad.update({(i, j), (j, k), (i, k)})
                    # m-m-m chain (A m B, B m C, A m C, in any orientation): record the
                    # (A, C) pair = the pair between the node with two outgoing 'meets'
                    # edges and the node with two incoming ones.
                    if abs(a) == 2 and abs(b) == 2 and abs(c) == 2:
                        outd = {i: 0, j: 0, k: 0}
                        ind = {i: 0, j: 0, k: 0}
                        for (x, y), r in (((i, j), a), ((j, k), b), ((i, k), c)):
                            src, dst = (x, y) if r == 2 else (y, x)
                            outd[src] += 1
                            ind[dst] += 1
                        A_ = [v for v in outd if outd[v] == 2]
                        C_ = [v for v in ind if ind[v] == 2]
                        if A_ and C_:
                            chain.add((min(A_[0], C_[0]), max(A_[0], C_[0])))
    from allen import is_satisfiable
    return ItemFeatures(
        item_id=it.item_id, pos=pos, rel_pos=pos / max(len(text), 1), sent=sent,
        n_sent=len(starts), dur_event=dur_event, words=len(text.split()),
        nonlin_share=(rev / tot) if tot else 0.0, gold_full=G,
        gold_consistent=is_satisfiable(G), bad_triple_pairs=bad, meets_chain_ac=chain)


def gold_one_hop_derivable(G: np.ndarray) -> np.ndarray:
    """D[i,j] = True if some k gives a unique composition G[i,k]∘G[k,j] equal to G[i,j]."""
    n = G.shape[0]
    A = G[:, :, None]   # (i,k)
    B = G[None, :, :]   # (k,j)
    valid = (A != MISSING) & (B != MISSING)
    S = np.where(valid, COMP_SINGLE[np.where(valid, A + 6, 0), np.where(valid, B + 6, 0)], MISSING)
    # S[i,k,j] = unique composition through k
    eq = (S == G[:, None, :]) & (G[:, None, :] != MISSING)
    idx = np.arange(n)
    eq[idx, idx, :] = False   # k == i
    eq[:, idx, idx] = False   # k == j
    return eq.any(axis=1)


# ======================================================================= loading predictions
def load_prediction(raw_path: Path, it: Item) -> Tuple[np.ndarray, Dict, Dict]:
    rec = read_json(raw_path)
    parsed = parse_matrix(rec["response"].get("content"), it.event_ids)
    resp = rec["response"]
    u = resp.get("usage") or {}
    ctd = u.get("completion_tokens_details") or {}
    info = {
        "finish_reason": resp.get("finish_reason"),
        "truncated": resp.get("finish_reason") == "length",
        "provider": resp.get("provider"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": ctd.get("reasoning_tokens", u.get("reasoning_tokens")),
        "reasoning_chars": len(resp.get("reasoning") or ""),
        "cost": u.get("cost"),
        "latency_s": resp.get("latency_s"),
        "attempts_total": rec.get("attempts_total"),
    }
    return parsed["matrix"], parsed["stats"], info


# ======================================================================= metrics
def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """13 x 14 confusion counts (rows gold label, cols predicted label + missing)."""
    gi = np.searchsorted(LABELS, gold)
    pi = np.where(pred == MISSING, MISS_COL, np.searchsorted(LABELS, np.where(pred == MISSING, 0, pred)))
    C = np.zeros((NL, NL + 1), dtype=np.int64)
    np.add.at(C, (gi, pi), 1)
    return C


def metrics_from_conf(C: np.ndarray) -> Dict[str, float]:
    """Works on a 13x14 matrix or a stack (..., 13, 14)."""
    C = np.asarray(C, dtype=float)
    tp = np.diagonal(C[..., :NL], axis1=-2, axis2=-1)
    gold_n = C.sum(axis=-1)
    pred_n = C[..., :NL].sum(axis=-2)
    total = gold_n.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / pred_n, 0.0)
        rec = np.where(gold_n > 0, tp / gold_n, np.nan)
        f1 = np.where((prec + np.nan_to_num(rec)) > 0, 2 * prec * np.nan_to_num(rec) / (prec + np.nan_to_num(rec)), 0.0)
        f1 = np.where(gold_n > 0, f1, np.nan)
        acc = tp.sum(axis=-1) / total
        macro = np.nanmean(f1, axis=-1)
        answered = total - C[..., MISS_COL].sum(axis=-1)
        pe = ((gold_n / total[..., None]) * (pred_n / total[..., None])).sum(axis=-1)
        kappa = (acc - pe) / (1 - pe)
        ta_p = tp.sum(axis=-1) / answered
        ta_f = 2 * ta_p * acc / (ta_p + acc)
    return {"accuracy": acc, "macro_f1": macro, "kappa": kappa, "coverage": answered / total,
            "ta_precision": ta_p, "ta_recall": acc, "ta_f1": ta_f, "n_pairs": total,
            "per_label_f1": f1, "per_label_p": prec, "per_label_r": rec, "support": gold_n}


def simple_conf(gold: np.ndarray, pred: np.ndarray, labels: List[int]) -> np.ndarray:
    lab = np.array(labels)
    gi = np.searchsorted(lab, gold)
    pi = np.where(pred == MISSING, len(lab), np.searchsorted(lab, np.where(pred == MISSING, lab[0], pred)))
    C = np.zeros((len(lab), len(lab) + 1), dtype=np.int64)
    np.add.at(C, (gi, pi), 1)
    return C


def metrics_simple(gold: np.ndarray, pred: np.ndarray, labels: List[int]) -> Dict[str, float]:
    C = simple_conf(gold, pred, labels).astype(float)
    k = len(labels)
    tp = np.diag(C[:, :k])
    gn = C.sum(1)
    pn = C[:, :k].sum(0)
    prec = np.where(pn > 0, tp / np.where(pn > 0, pn, 1), 0.0)
    rec = np.where(gn > 0, tp / np.where(gn > 0, gn, 1), np.nan)
    f1 = np.where(gn > 0, np.where(prec + np.nan_to_num(rec) > 0,
                                   2 * prec * np.nan_to_num(rec) / np.maximum(prec + np.nan_to_num(rec), 1e-12), 0.0), np.nan)
    acc = tp.sum() / C.sum()
    pe = ((gn / C.sum()) * (pn / C.sum())).sum()
    return {"accuracy": acc, "macro_f1": float(np.nanmean(f1)), "kappa": (acc - pe) / (1 - pe)}


def bootstrap_weights(n_clusters: int, B: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(B, n_clusters))
    W = np.zeros((B, n_clusters), dtype=np.float64)
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n_clusters)
    return W


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        val = min(1.0, (m - rank) * pvals[k])
        running = max(running, val)
        adj[k] = running
    return adj.tolist()


def cohen_kappa_codes(a: np.ndarray, b: np.ndarray) -> float:
    labs = np.array(LABELS)
    ia = np.searchsorted(labs, a)
    ib = np.searchsorted(labs, b)
    C = np.zeros((NL, NL))
    np.add.at(C, (ia, ib), 1)
    N = C.sum()
    po = np.trace(C) / N
    pe = (C.sum(1) / N * C.sum(0) / N).sum()
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fleiss_kappa(P: np.ndarray) -> float:
    """P: (N items, R raters) codes, all valid."""
    N, R = P.shape
    counts = np.zeros((N, NL))
    for r in range(R):
        np.add.at(counts, (np.arange(N), np.searchsorted(LABELS, P[:, r])), 1)
    p_j = counts.sum(0) / (N * R)
    P_i = ((counts * (counts - 1)).sum(1)) / (R * (R - 1))
    Pbar = P_i.mean()
    Pe = (p_j ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe))


def dist_bin(d: np.ndarray) -> np.ndarray:
    out = np.empty(d.shape, dtype=object)
    for lo, hi, name in DIST_BINS:
        out[(d >= lo) & (d <= hi)] = name
    return out
