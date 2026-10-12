"""
Shared helpers: project paths, gold-data loading, the frozen prompt, the output parser,
and crash-safe file writing.

Nothing in this module (or anywhere in scripts/) writes to exports/: the gold data is
read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from allen import CODE2SYM, MISSING, SYM2CODE

# ------------------------------------------------------------------ paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
EXPORTS = ROOT / "exports"
STEMS_CSV = EXPORTS / "all_stems.csv"
REL_DIR = EXPORTS / "stem_relations"
# (environment overrides are used only by the offline self-test)
MODELS_JSON = Path(os.environ.get("STNLP_MODELS_JSON", str(ROOT / "models.json")))
ENV_FILE = Path(os.environ.get("STNLP_ENV_FILE", str(ROOT / ".env")))
# Results location can be overridden (used only by the self-test).
RESULTS = Path(os.environ.get("STNLP_RESULTS_DIR", str(ROOT / "results"))).resolve()

OPENROUTER_CHAT_URL = os.environ.get("STNLP_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODELS_URL = os.environ.get("STNLP_MODELS_URL", "https://openrouter.ai/api/v1/models")

# Items excluded from the run, decided before any model call (see experiment plan §2.1).
EXCLUDED_ITEMS = {
    "stem_254": "two different events share the ID E18, so the event list is ambiguous",
}

FILE_RE = re.compile(r"^stem_(\d+)(?:_batch_(\d+))?_relations\.csv$")


# ------------------------------------------------------------------ io helpers
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str, retries: int = 20) -> None:
    """Write via a temp file + os.replace so a crash never leaves a half-written file.
    Retries when Windows reports the target as locked (e.g. open in Excel)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    last_err: Optional[Exception] = None
    for k in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:  # file locked by another program on Windows
            last_err = e
            time.sleep(0.25 * (k + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise PermissionError(f"Could not write {path} (is it open in another program?): {last_err}")


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env(path: Path = ENV_FILE) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_models() -> List[Dict[str, str]]:
    data = read_json(MODELS_JSON)
    models = data["models"]
    ids = [m["model_id"] for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError("models.json: model_id values must be unique")
    for m in models:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", m["model_id"]):
            raise ValueError(f"models.json: model_id {m['model_id']!r} must be a simple folder-safe name")
    return models


# ------------------------------------------------------------------ gold data
@dataclass
class Item:
    item_id: str                 # e.g. "stem_31" or "stem_102_batch_270"
    stem_id: int
    batch: Optional[int]
    file: str
    stem_text: str
    event_ids: List[str]         # in gold ID order (numeric)
    event_texts: List[str]
    gold: np.ndarray             # n x n codes, MISSING on diagonal
    is_primary: bool = True
    excluded_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.event_ids)


def _read_stems() -> Dict[int, str]:
    import pandas as pd
    s = pd.read_csv(STEMS_CSV)
    return dict(zip(s.stem_id.astype(int), s.stem_text.astype(str)))


def load_items(include_excluded: bool = False) -> List[Item]:
    """Load every relation file as an Item. Gold matrices are built from the CSV rows.
    Items listed in EXCLUDED_ITEMS (and any file whose event IDs map to more than one text)
    are flagged with excluded_reason and dropped unless include_excluded=True."""
    import pandas as pd
    stems = _read_stems()
    items: List[Item] = []
    for f in sorted(REL_DIR.glob("*.csv")):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        sid = int(m.group(1))
        batch = int(m.group(2)) if m.group(2) else None
        item_id = f"stem_{sid}" + (f"_batch_{batch}" if batch is not None else "")
        d = pd.read_csv(f)
        texts: Dict[str, set] = {}
        first_text: Dict[str, str] = {}
        for a, t in zip(d.from_event, d.text_i):
            texts.setdefault(a, set()).add(t)
            first_text.setdefault(a, t)
        for a, t in zip(d.to_event, d.text_j):
            texts.setdefault(a, set()).add(t)
            first_text.setdefault(a, t)
        ev = sorted(texts, key=lambda x: int(x[1:]))
        reason = EXCLUDED_ITEMS.get(item_id)
        multi = [e for e in ev if len(texts[e]) > 1]
        if multi and not reason:
            reason = f"event IDs with more than one text: {multi}"
        ix = {e: i for i, e in enumerate(ev)}
        n = len(ev)
        G = np.full((n, n), MISSING, dtype=np.int16)
        for a, b, c in zip(d.from_event, d.to_event, d.relation_code):
            if a != b:
                G[ix[a], ix[b]] = int(c)
        if sid not in stems:
            raise ValueError(f"{f.name}: stem {sid} not found in all_stems.csv")
        it = Item(item_id=item_id, stem_id=sid, batch=batch, file=f.name,
                  stem_text=stems[sid], event_ids=ev,
                  event_texts=[first_text[e] for e in ev], gold=G,
                  is_primary=batch is None, excluded_reason=reason)
        if not reason:
            # gold integrity: complete and converse-consistent
            off = ~np.eye(n, dtype=bool)
            if (G[off] == MISSING).any():
                it.notes.append("gold matrix has missing pairs")
            if not np.all(G[off] == -G.T[off]):
                it.notes.append("gold matrix is not converse-consistent")
            if len(set(it.event_texts)) != n:
                it.notes.append("duplicate event span texts (different IDs, same words)")
        if include_excluded or not reason:
            items.append(it)
    # deterministic order: primary first by stem id, then batch files
    items.sort(key=lambda x: (not x.is_primary, x.stem_id, x.batch or 0))
    return items


def data_fingerprint() -> str:
    h = hashlib.sha256()
    for p in [STEMS_CSV] + sorted(REL_DIR.glob("*.csv")):
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ prompt (FROZEN)
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an expert annotator of temporal relations between events described in text."
)

RELATION_DEFINITIONS = """\
<   X before Y: X ends before Y starts (there is a gap between them).
>   X after Y: X starts after Y ends (there is a gap between them).
m   X meets Y: X ends exactly when Y starts.
mi  X met by Y: X starts exactly when Y ends.
o   X overlaps Y: X starts before Y starts, and X ends after Y starts but before Y ends.
oi  X overlapped by Y: Y starts before X starts, and Y ends after X starts but before X ends.
s   X starts Y: X and Y start together, and X ends before Y ends.
si  X started by Y: X and Y start together, and X ends after Y ends.
d   X during Y: X starts after Y starts and ends before Y ends.
di  X contains Y: Y starts after X starts and ends before X ends.
f   X finishes Y: X and Y end together, and X starts after Y starts.
fi  X finished by Y: X and Y end together, and X starts before Y starts.
=   X equals Y: X and Y start together and end together."""

USER_TEMPLATE = """\
Read the passage and the list of events taken from it. Each event is identified by an ID and by the words that describe it in the passage.

Your task: for every ordered pair of different events (X, Y), decide which ONE of Allen's 13 interval relations holds between the time interval of event X and the time interval of event Y, according to the passage. Every pair must receive exactly one relation; if the passage does not state it explicitly, choose the most plausible one.

The 13 relations (read as "X relation Y"):
{definitions}

PASSAGE:
\"\"\"
{passage}
\"\"\"

EVENTS ({n} events):
{event_lines}

OUTPUT FORMAT
Write the complete {n} x {n} relation matrix between the tags <matrix> and </matrix>.
- Write exactly one line per event, in the order of the event list above.
- Each line starts with the row event's ID followed by a colon, then {n} symbols separated by single spaces.
- The k-th symbol in the row of event X is the relation of X to the k-th event of the list (X relation that event). Column order: {column_order}
- Use "-" on the diagonal (the cell of an event with itself).
- Use only these symbols: < > m mi o oi s si d di f fi = -
- Write nothing else inside the tags.

Format illustration for a different, 3-event case (the symbols here are only an illustration):
<matrix>
E1: - < m
E2: > - d
E3: mi di -
</matrix>"""


def build_messages(item_like) -> List[Dict[str, str]]:
    """Return the chat messages for an item. item_like needs .stem_text, .event_ids, .event_texts.
    The model receives ONLY the passage and the events (plus the task instructions)."""
    n = len(item_like.event_ids)
    event_lines = "\n".join(f"{e}: {t}" for e, t in zip(item_like.event_ids, item_like.event_texts))
    user = USER_TEMPLATE.format(
        definitions=RELATION_DEFINITIONS,
        passage=item_like.stem_text.strip(),
        n=n,
        event_lines=event_lines,
        column_order=" ".join(item_like.event_ids),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def template_fingerprint() -> str:
    return sha256_text(PROMPT_VERSION + SYSTEM_PROMPT + RELATION_DEFINITIONS + USER_TEMPLATE)


def messages_fingerprint(messages: List[Dict[str, str]]) -> str:
    return sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))


# ------------------------------------------------------------------ parser (FROZEN)
_TOKEN_NORMALISE = {
    "−": "-", "–": "-", "—": "-", "‐": "-",
    "＜": "<", "＞": ">", "＝": "=",
}
_ROW_RE = re.compile(r"^\s*[*_`>#|\-]*\s*(E\d+)\s*[*_`]*\s*[:|]\s*(.*)$")


def _extract_block(text: str) -> Tuple[str, str]:
    """Return (block_text, how) where how in {'tags', 'open_tag_only', 'no_tags'}."""
    low = text.lower()
    closes = [m.start() for m in re.finditer(r"</matrix>", low)]
    if closes:
        end = closes[-1]
        start = low.rfind("<matrix>", 0, end)
        if start != -1:
            return text[start + len("<matrix>"):end], "tags"
    start = low.rfind("<matrix>")
    if start != -1:
        return text[start + len("<matrix>"):], "open_tag_only"
    return text, "no_tags"


def _norm_token(tok: str) -> str:
    tok = tok.strip().strip("`'\"*.,;[](){}")
    for a, b in _TOKEN_NORMALISE.items():
        tok = tok.replace(a, b)
    if tok in ("<", ">", "=", "-"):
        return tok
    return tok.lower()


def parse_matrix(content: Optional[str], event_ids: List[str]) -> Dict:
    """Parse a model's answer into an n x n code matrix (MISSING where no valid symbol).

    Rules (fixed before the run):
    - Use the last <matrix>...</matrix> block; if only an opening tag exists (truncated
      output) use everything after it; if no tag exists, scan the whole answer.
    - A row is a line 'E<k>: tok tok ...'. Only IDs from the event list are used; the first
      occurrence of each ID wins.
    - A row with exactly n tokens is used as is. A row with n-1 tokens and no '-' is
      accepted by inserting the diagonal (flag 'diag_omitted'). Any other length makes
      the whole row invalid (all cells MISSING).
    - Unknown symbols make only that cell MISSING. The diagonal is ignored.
    """
    n = len(event_ids)
    pos = {e: i for i, e in enumerate(event_ids)}
    M = np.full((n, n), MISSING, dtype=np.int16)
    stats = {
        "block": None, "rows_found": 0, "rows_used": 0, "rows_bad_length": 0,
        "rows_unknown_id": 0, "rows_duplicate": 0, "rows_diag_omitted": 0,
        "unknown_tokens": 0, "diag_not_dash": 0, "cells_valid": 0, "cells_total": n * (n - 1),
        "status": "failed",
    }
    if not content:
        stats["block"] = "empty"
        return {"matrix": M, "stats": stats}
    block, how = _extract_block(content)
    stats["block"] = how
    seen = set()
    for line in block.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        rid, rest = m.group(1), m.group(2)
        stats["rows_found"] += 1
        if rid not in pos:
            stats["rows_unknown_id"] += 1
            continue
        if rid in seen:
            stats["rows_duplicate"] += 1
            continue
        seen.add(rid)
        toks = [_norm_token(t) for t in re.split(r"[\s,|]+", rest.strip()) if t.strip(" ,|")]
        toks = [t for t in toks if t != ""]
        i = pos[rid]
        if len(toks) == n - 1 and "-" not in toks:
            toks = toks[:i] + ["-"] + toks[i:]
            stats["rows_diag_omitted"] += 1
        if len(toks) != n:
            stats["rows_bad_length"] += 1
            continue
        stats["rows_used"] += 1
        for j, t in enumerate(toks):
            if j == i:
                if t != "-":
                    stats["diag_not_dash"] += 1
                continue
            c = SYM2CODE.get(t)
            if c is None:
                stats["unknown_tokens"] += 1
            else:
                M[i, j] = c
    valid = int((M != MISSING).sum())
    stats["cells_valid"] = valid
    if valid == n * (n - 1):
        stats["status"] = "complete"
    elif valid > 0:
        stats["status"] = "partial"
    return {"matrix": M, "stats": stats}


def matrix_to_rows(M: np.ndarray, event_ids: List[str]) -> List[Dict]:
    rows = []
    n = len(event_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(M[i, j])
            rows.append({"from_event": event_ids[i], "to_event": event_ids[j],
                         "pred_symbol": CODE2SYM.get(c, ""), "pred_code": "" if c == MISSING else c})
    return rows


# ------------------------------------------------------------------ smoke-test item
@dataclass
class SyntheticItem:
    item_id: str
    stem_text: str
    event_ids: List[str]
    event_texts: List[str]


SMOKE_ITEM = SyntheticItem(
    item_id="smoke_test",
    stem_text=("Maria boiled water for tea. While the water was boiling, she read a short letter. "
               "After the kettle switched off, she poured the tea."),
    event_ids=["E1", "E2", "E3"],
    event_texts=["boiled water", "read a short letter", "poured the tea"],
)
