"""
Allen interval algebra toolkit.

Relation codes follow exports/Relation_code.txt exactly:
     0 : =   (equal)
    +1 : <   (precedes / before)        -1 : >   (preceded by / after)
    +2 : m   (meets)                    -2 : mi  (met by)
    +3 : o   (overlaps)                 -3 : oi  (overlapped by)
    +4 : s   (starts)                   -4 : si  (started by)
    +5 : d   (during)                   -5 : di  (contains)
    +6 : f   (finishes)                 -6 : fi  (finished by)
A relation r(X, Y) is read "X r Y". The converse is always -r.

Everything in this module is derived programmatically from a brute-force
enumeration of concrete intervals, so the endpoint signatures and the
composition table cannot contain transcription errors.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

CODES: List[int] = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
SYM2CODE: Dict[str, int] = {
    "<": 1, ">": -1, "=": 0,
    "m": 2, "mi": -2,
    "o": 3, "oi": -3,
    "s": 4, "si": -4,
    "d": 5, "di": -5,
    "f": 6, "fi": -6,
}
CODE2SYM: Dict[int, str] = {v: k for k, v in SYM2CODE.items()}
CODE2NAME: Dict[int, str] = {
    0: "equal", 1: "before", -1: "after", 2: "meets", -2: "met-by",
    3: "overlaps", -3: "overlapped-by", 4: "starts", -4: "started-by",
    5: "during", -5: "contains", 6: "finishes", -6: "finished-by",
}
MISSING = 99  # sentinel for "no valid prediction"


def converse(code: int) -> int:
    return MISSING if code == MISSING else -code


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def relation_of(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Allen relation of interval a to interval b (both (start, end), start < end)."""
    (a1, a2), (b1, b2) = a, b
    if a2 < b1:
        return 1
    if b2 < a1:
        return -1
    if a1 == b1 and a2 == b2:
        return 0
    if a2 == b1:
        return 2
    if b2 == a1:
        return -2
    if a1 == b1:
        return 4 if a2 < b2 else -4
    if a2 == b2:
        return 6 if a1 > b1 else -6
    if b1 < a1 and a2 < b2:
        return 5
    if a1 < b1 and b2 < a2:
        return -5
    if a1 < b1 < a2 < b2:
        return 3
    if b1 < a1 < b2 < a2:
        return -3
    raise AssertionError(f"unclassified intervals {a} {b}")


# ---------------------------------------------------------------- derived tables
_IV = [(x, y) for x in range(7) for y in range(7) if x < y]

# Endpoint signature of each relation, for intervals X=(sx,ex), Y=(sy,ey):
# (sign(sx-sy), sign(sx-ey), sign(ex-sy), sign(ex-ey)); each in {-1,0,1}.
ENDPOINTS: Dict[int, Tuple[int, int, int, int]] = {}
for _a in _IV:
    for _b in _IV:
        _r = relation_of(_a, _b)
        _sig = (_sign(_a[0] - _b[0]), _sign(_a[0] - _b[1]), _sign(_a[1] - _b[0]), _sign(_a[1] - _b[1]))
        if _r in ENDPOINTS:
            assert ENDPOINTS[_r] == _sig
        ENDPOINTS[_r] = _sig
assert sorted(ENDPOINTS) == CODES
assert len(set(ENDPOINTS.values())) == 13
ENDPOINT_NAMES = ("start-start", "start-end", "end-start", "end-end")

# Composition table: COMP[(r1, r2)] = set of possible r(X,Z) given r(X,Y)=r1, r(Y,Z)=r2.
COMP: Dict[Tuple[int, int], frozenset] = {}
_tmp: Dict[Tuple[int, int], set] = {}
for _a in _IV:
    for _b in _IV:
        for _c in _IV:
            _tmp.setdefault((relation_of(_a, _b), relation_of(_b, _c)), set()).add(relation_of(_a, _c))
COMP = {k: frozenset(v) for k, v in _tmp.items()}
assert len(COMP) == 169
# Sanity checks against textbook entries.
assert COMP[(1, 1)] == {1}
assert COMP[(2, 2)] == {1}
assert COMP[(0, 5)] == {5}
assert len(COMP[(1, -1)]) == 13

# Dense lookup arrays (index = code + 6).
COMP_OK = np.zeros((13, 13, 13), dtype=bool)          # COMP_OK[r1, r2, r3] -> r3 in r1∘r2
COMP_SINGLE = np.full((13, 13), MISSING, dtype=np.int16)  # unique result or MISSING
for (r1, r2), s in COMP.items():
    for r3 in s:
        COMP_OK[r1 + 6, r2 + 6, r3 + 6] = True
    if len(s) == 1:
        COMP_SINGLE[r1 + 6, r2 + 6] = next(iter(s))


def code_from_endpoints(sx: int, ex: int, sy: int, ey: int) -> int:
    return relation_of((sx, ex), (sy, ey))


# ---------------------------------------------------------------- matrices
def full_matrix_from_upper(upper: np.ndarray) -> np.ndarray:
    """Given an n x n array whose strict upper triangle holds codes (MISSING allowed),
    return a full matrix with the lower triangle filled by converses and MISSING on the diagonal."""
    n = upper.shape[0]
    M = np.full((n, n), MISSING, dtype=np.int16)
    iu = np.triu_indices(n, 1)
    M[iu] = upper[iu]
    low = upper[iu]
    conv = np.where(low == MISSING, MISSING, -low)
    M[(iu[1], iu[0])] = conv
    return M


def inconsistent_triples(M: np.ndarray) -> Tuple[int, int]:
    """Count unordered triples i<j<k (all three relations present) that violate the
    composition table, using the full matrix M (M[i,j] = r(i,j)).
    For three intervals, r(i,k) in r(i,j)∘r(j,k) is necessary and sufficient for the
    3-node atomic network to be satisfiable.
    Returns (n_bad, n_checked)."""
    n = M.shape[0]
    if n < 3:
        return 0, 0
    A = M[:, :, None]   # (i,j)
    B = M[None, :, :]   # (j,k)
    C = M[:, None, :]   # (i,k)
    i, j, k = np.ogrid[:n, :n, :n]
    mask = (i < j) & (j < k)
    valid = mask & (A != MISSING) & (B != MISSING) & (C != MISSING)
    Ai = np.where(valid, A + 6, 0)
    Bi = np.where(valid, B + 6, 0)
    Ci = np.where(valid, C + 6, 0)
    ok = COMP_OK[Ai, Bi, Ci]
    bad = valid & ~ok
    return int(bad.sum()), int(valid.sum())


def is_satisfiable(M: np.ndarray) -> bool:
    """Exact satisfiability of an Allen network in which every present pair carries one
    atomic relation (absent pairs = MISSING = unconstrained).
    Each atomic relation is a conjunction of endpoint constraints in {<, =, >}; together with
    start_i < end_i the network is satisfiable iff, after merging '=' endpoints, the strict
    '<' graph has no edge inside a merged class and no directed cycle."""
    n = M.shape[0]
    parent = list(range(2 * n))  # node 2i = start_i, 2i+1 = end_i

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    strict: List[Tuple[int, int]] = [(2 * i, 2 * i + 1) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(M[i, j])
            if c == MISSING:
                continue
            si, ei, sj, ej = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
            for (x, y), sg in zip(((si, sj), (si, ej), (ei, sj), (ei, ej)), ENDPOINTS[c]):
                if sg == 0:
                    union(x, y)
                elif sg < 0:
                    strict.append((x, y))
                else:
                    strict.append((y, x))
    adj: Dict[int, set] = {}
    indeg: Dict[int, int] = {}
    nodes = {find(x) for x in range(2 * n)}
    for v in nodes:
        adj[v] = set()
        indeg[v] = 0
    for x, y in strict:
        a, b = find(x), find(y)
        if a == b:
            return False
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1
    queue = [v for v in nodes if indeg[v] == 0]
    seen = 0
    while queue:
        v = queue.pop()
        seen += 1
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    return seen == len(nodes)


def repair_nearest_consistent(n: int,
                              votes: Sequence[Tuple[int, int, int, int]],
                              time_limit_s: float = 60.0,
                              stall_s: float = 5.0,
                              workers: int = 0) -> Dict:
    """Nearest consistent network (offline repair, plan A3a).

    votes: list of (i, j, code, weight) meaning "r(i,j) = code" is preferred with the given
    integer weight. Several votes may concern the same pair.
    Model (OR-Tools CP-SAT): integer positions for the 2n endpoints, one Boolean per vote
    that, when true, enforces the vote's four endpoint comparisons; maximise the total weight
    of satisfied votes. Any solution is a consistent network by construction.
    The problem is NP-hard and its LP bound is weak, so for larger networks optimality is
    usually not proven: the search stops after `time_limit_s` or when no better solution has
    been found for `stall_s` seconds. The returned 'status' (OPTIMAL / FEASIBLE), 'objective'
    and 'bound' must be reported alongside any result that uses the repaired networks.
    Returns dict with 'matrix' (full n x n codes) and solver diagnostics."""
    import threading
    import time as _time

    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    H = 2 * n
    s = [model.NewIntVar(0, H, f"s{i}") for i in range(n)]
    e = [model.NewIntVar(0, H, f"e{i}") for i in range(n)]
    for i in range(n):
        model.Add(s[i] < e[i])
    lits = []
    weights = []
    for idx, (i, j, code, w) in enumerate(votes):
        if code == MISSING or i == j:
            continue
        b = model.NewBoolVar(f"v{idx}")
        for (x, y), sg in zip(((s[i], s[j]), (s[i], e[j]), (e[i], s[j]), (e[i], e[j])), ENDPOINTS[code]):
            if sg == 0:
                model.Add(x == y).OnlyEnforceIf(b)
            elif sg < 0:
                model.Add(x < y).OnlyEnforceIf(b)
            else:
                model.Add(x > y).OnlyEnforceIf(b)
        lits.append(b)
        weights.append(int(w))
    model.Maximize(sum(w * b for w, b in zip(weights, lits)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = 0
    if workers:
        solver.parameters.num_search_workers = int(workers)

    class _CB(cp_model.CpSolverSolutionCallback):
        def __init__(self) -> None:
            super().__init__()
            self.last = _time.monotonic()
            self.n_solutions = 0

        def on_solution_callback(self) -> None:  # noqa: N802
            self.last = _time.monotonic()
            self.n_solutions += 1

    cb = _CB()
    done = threading.Event()
    stalled = {"flag": False}

    def _watchdog() -> None:
        while not done.wait(0.2):
            if cb.n_solutions > 0 and _time.monotonic() - cb.last > stall_s:
                stalled["flag"] = True
                solver.stop_search()
                return

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()
    try:
        status = solver.Solve(model, cb)
    finally:
        done.set()
        wd.join(timeout=2)
    status_name = solver.StatusName(status)
    out = {"status": status_name, "objective": None, "bound": None,
           "wall_time": solver.WallTime(), "matrix": None,
           "stopped_by_stall": stalled["flag"],
           "max_objective": int(sum(weights))}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sv = [solver.Value(x) for x in s]
        ev = [solver.Value(x) for x in e]
        M = np.full((n, n), MISSING, dtype=np.int16)
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = code_from_endpoints(sv[i], ev[i], sv[j], ev[j])
        out.update(matrix=M, objective=int(round(solver.ObjectiveValue())),
                   bound=int(round(solver.BestObjectiveBound())))
    return out


def endpoint_diff(pred: int, gold: int) -> Tuple[int, Tuple[bool, bool, bool, bool]]:
    """Number of the four endpoint comparisons that differ between two relations, and which."""
    a, b = ENDPOINTS[pred], ENDPOINTS[gold]
    diff = tuple(x != y for x, y in zip(a, b))
    return sum(diff), diff  # type: ignore[return-value]


def self_test() -> None:
    # every converse is the converse in the table
    for c in CODES:
        # find an interval pair with relation c and check converse
        for a in _IV:
            for b in _IV:
                if relation_of(a, b) == c:
                    assert relation_of(b, a) == -c
    # satisfiability on trivial cases
    M = full_matrix_from_upper(np.array([[MISSING, 1, 1], [MISSING, MISSING, 1], [MISSING] * 3], dtype=np.int16))
    assert is_satisfiable(M)
    assert inconsistent_triples(M) == (0, 1)
    M2 = full_matrix_from_upper(np.array([[MISSING, 2, 2], [MISSING, MISSING, 2], [MISSING] * 3], dtype=np.int16))
    assert not is_satisfiable(M2)
    assert inconsistent_triples(M2) == (1, 1)
    M3 = full_matrix_from_upper(np.array([[MISSING, 1, -1], [MISSING, MISSING, 1], [MISSING] * 3], dtype=np.int16))
    assert not is_satisfiable(M3)


if __name__ == "__main__":
    self_test()
    print("allen.py self-test passed")
