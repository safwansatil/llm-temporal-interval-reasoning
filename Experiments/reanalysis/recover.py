"""Recovery re-parse layered over common.py. Strict path is untouched."""
import sys, re, json, numpy as np
from pathlib import Path
SCRIPTS = Path('/home/claude/llm-temporal-interval-reasoning/Experiments/scripts')
sys.path.insert(0, str(SCRIPTS))
import common as C
import allen as A
import analysis_lib as L

MISSING = C.MISSING
SYM2CODE = C.SYM2CODE

def _rows_from(content, event_ids):
    """Yield (row_index, token_list) for every row line, using common.py's own regex/normaliser."""
    n = len(event_ids); pos = {e:i for i,e in enumerate(event_ids)}
    if not content: return []
    block, how = C._extract_block(content)
    out = []; seen=set()
    for line in block.splitlines():
        m = C._ROW_RE.match(line)
        if not m: continue
        rid, rest = m.group(1), m.group(2)
        if rid not in pos or rid in seen: continue
        seen.add(rid)
        toks = [C._norm_token(t) for t in re.split(r"[\s,|]+", rest.strip()) if t.strip(" ,|")]
        toks = [t for t in toks if t != ""]
        out.append((pos[rid], toks))
    return out

def _assign(M, i, j, tok):
    if j == i: return
    c = SYM2CODE.get(tok)
    if c is not None and M[i, j] == MISSING:
        M[i, j] = c

def recover_matrix(content, event_ids, use_R1=True, use_R2=True):
    """Returns (matrix, stats). R1 = drop unknown tokens if that fixes length.
       R2 = dash-anchored two-sided agreement for off-by-k rows."""
    n = len(event_ids)
    M = np.full((n, n), MISSING, dtype=np.int16)
    st = dict(rows_strict=0, rows_R1=0, rows_R2=0, cells_R2=0)
    for i, toks in _rows_from(content, event_ids):
        # strict path (same rules as common.parse_matrix)
        t = list(toks)
        if len(t) == n-1 and "-" not in t:
            t = t[:i] + ["-"] + t[i:]
        if len(t) == n:
            st['rows_strict'] += 1
            for j, tk in enumerate(t): _assign(M, i, j, tk)
            continue
        # R1: drop tokens that are not valid symbols at all
        if use_R1:
            known = [x for x in toks if x in SYM2CODE or x == "-"]
            k = list(known)
            if len(k) == n-1 and "-" not in k:
                k = k[:i] + ["-"] + k[i:]
            if len(k) == n and len(known) != len(toks):
                st['rows_R1'] += 1
                for j, tk in enumerate(k): _assign(M, i, j, tk)
                continue
        # R2: dash-anchored, assign only where left- and right-anchoring agree
        if use_R2 and toks.count("-") == 1:
            p = toks.index("-")
            if p == i:                       # dash sits at the true diagonal
                shift = len(toks) - n        # +k = too long, -k = too short
                used = 0
                for j in range(n):
                    if j == i: continue
                    left = toks[j] if 0 <= j < len(toks) else None
                    right = toks[j+shift] if 0 <= j+shift < len(toks) else None
                    if left is not None and left == right:
                        _assign(M, i, j, left); used += 1
                if used:
                    st['rows_R2'] += 1; st['cells_R2'] += used
    return M, st

def converse_backfill(M):
    """Fill a missing cell from the model's own answer in the mirror cell."""
    n = M.shape[0]
    out = M.copy(); filled = 0
    for i in range(n):
        for j in range(n):
            if i != j and out[i, j] == MISSING and M[j, i] != MISSING:
                out[i, j] = A.converse(int(M[j, i])); filled += 1
    return out, filled
