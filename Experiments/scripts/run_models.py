#!/usr/bin/env python
"""
Single-run model experiment (experiment_plans_single_run.md, section 2).

Every model in models.json is called ONCE per item (stem + its event list) with one frozen
prompt, and must return the full n x n Allen relation matrix. Results are saved after every
call; the run can be stopped with Ctrl+C and resumed by starting the same command again.

Usage (from the project folder):
    python scripts/run_models.py --check     # verify models on OpenRouter (free, no generation)
    python scripts/run_models.py --smoke     # 1 tiny made-up story per model (not from the data)
    python scripts/run_models.py --dry-run   # show what would run + cost estimate, no calls
    python scripts/run_models.py             # the real run (asks for confirmation the first time)
    python scripts/run_models.py --workers   # same, with 8 parallel calls per model (default is 4)
    python scripts/run_models.py --workers 12

All models always run at the same time; --workers sets how many calls run in parallel for
each model (total parallel calls = workers x number of models).

Ctrl+C once  -> graceful stop: no new calls; calls already running finish and are saved.
Ctrl+C twice -> immediate stop: running calls are abandoned and will be redone on resume.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import io
import json
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

try:
    import httpx
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text
except ImportError as _e:  # pragma: no cover
    print(f"Missing package: {_e}. Run:  pip install -r scripts/requirements.txt")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allen import MISSING  # noqa: E402
from common import (  # noqa: E402
    EXCLUDED_ITEMS, OPENROUTER_CHAT_URL, OPENROUTER_MODELS_URL, PROMPT_VERSION, RESULTS, ROOT,
    SMOKE_ITEM, Item, atomic_write_json, atomic_write_text, build_messages, data_fingerprint,
    load_env, load_items, load_models, matrix_to_rows, messages_fingerprint, parse_matrix,
    read_json, template_fingerprint,
)

# ======================================================================= frozen settings
# These values are written into results/run_manifest.json on the first start and are
# checked on every resume, so the run cannot silently change halfway through.
FROZEN_SETTINGS = {
    "temperature": 0.0,                 # sent only to models that support it
    "reasoning": {"effort": "medium"},  # sent only to models that support 'reasoning'
    "max_tokens_cap": 65536,            # max_tokens = min(cap, model's max completion tokens)
    "seed": 20261006,                   # sent only to models that support 'seed'
    "stream": True,
    "include_batch_items": True,        # the 8 '_batch_' annotation versions (analysed separately)
    "item_order_seed": 20261006,
    "retry_policy": "retry only transport/provider failures; never re-ask because of answer content",
}
# Operational settings (may differ between sessions; they do not change what is asked).
DEFAULT_CONCURRENCY = 4          # simultaneous calls per model (default without --workers)
WORKERS_CONST = 8                # simultaneous calls per model when "--workers" is given without a number
MAX_ATTEMPTS_PER_SESSION = 4     # per item, per session (1 try + 3 retries) for failures
MAX_RATE_LIMIT_RETRIES = 12      # HTTP 429 retries per item, per session (not counted as failures)
BACKOFF_S = [10, 30, 90]
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 600             # max silence between streamed chunks
CALL_HARD_LIMIT_S = 3600         # max duration of one call

if os.environ.get("STNLP_SELFTEST_FAST_BACKOFF"):  # offline self-test only
    BACKOFF_S = [0.5, 1, 2]

RAW_DIR = RESULTS / "raw"
PARSED_DIR = RESULTS / "parsed"
ERR_DIR = RESULTS / "errors"
LOG_DIR = RESULTS / "logs"
PROMPT_DIR = RESULTS / "prompts"
SMOKE_DIR = RESULTS / "smoke"
MANIFEST = RESULTS / "run_manifest.json"
STATUS_CSV = RESULTS / "status.csv"

STATUS_FIELDS = [
    "model_id", "item_id", "stem_id", "is_primary", "n_events", "state", "finish_reason",
    "truncated", "parse_status", "cells_valid", "cells_total", "prompt_tokens",
    "completion_tokens", "reasoning_tokens", "reasoning_chars", "cost_usd", "latency_s", "provider",
    "attempts_total",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ======================================================================= errors
class FatalRunError(Exception):
    """Stops the whole run (e.g. invalid API key, no credits)."""


class ItemError(Exception):
    """Deterministic error for this request (e.g. HTTP 400). Not retried in this session."""


class TransientError(Exception):
    """Network / provider failure. Retried with backoff."""

    def __init__(self, msg: str, retry_after: Optional[float] = None, rate_limited: bool = False,
                 budget: bool = False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.rate_limited = rate_limited
        self.budget = budget  # OpenRouter "in-flight budget exhausted" (HTTP 402, temporary)


# ======================================================================= OpenRouter metadata
def fetch_model_metadata(model_keys: List[str]) -> Dict[str, Dict]:
    r = httpx.get(OPENROUTER_MODELS_URL, timeout=60)
    r.raise_for_status()
    data = {m["id"]: m for m in r.json()["data"]}
    out = {}
    for k in model_keys:
        m = data.get(k)
        if m is None:
            out[k] = None
            continue
        top = m.get("top_provider") or {}
        out[k] = {
            "id": k,
            "name": m.get("name"),
            "context_length": m.get("context_length"),
            "max_completion_tokens": top.get("max_completion_tokens"),
            "supported_parameters": sorted(m.get("supported_parameters") or []),
            "pricing": m.get("pricing"),
            "is_moderated": top.get("is_moderated"),
            "fetched_at": utcnow(),
        }
    return out


def build_request_params(meta: Dict) -> Dict:
    """Request parameters for one model (everything except model/messages)."""
    sp = set(meta.get("supported_parameters") or [])
    cap = FROZEN_SETTINGS["max_tokens_cap"]
    mct = meta.get("max_completion_tokens")
    max_tokens = min(cap, int(mct)) if mct else cap
    p: Dict = {"stream": True, "max_tokens": max_tokens}
    notes = []
    if "temperature" in sp:
        p["temperature"] = FROZEN_SETTINGS["temperature"]
    else:
        notes.append("temperature not supported by this model; provider default used")
    if "reasoning" in sp:
        p["reasoning"] = dict(FROZEN_SETTINGS["reasoning"])
    else:
        notes.append("no reasoning parameter (model does not support it)")
    if "seed" in sp:
        p["seed"] = FROZEN_SETTINGS["seed"]
    return {"params": p, "notes": notes}


# ======================================================================= one API call
def merge_reasoning_details(parts: List) -> List[Dict]:
    """Streamed reasoning_details arrive as many small fragments; merge consecutive fragments
    of the same block (same type/index/format/id) by concatenating their string fields."""
    merged: List[Dict] = []
    for p in parts:
        if not isinstance(p, dict):
            merged.append({"raw": p})
            continue
        key = (p.get("type"), p.get("index"), p.get("format"), p.get("id"))
        if merged and merged[-1].get("_key") == key:
            last = merged[-1]
            for f in ("text", "summary", "data", "signature"):
                if isinstance(p.get(f), str):
                    last[f] = (last.get(f) or "") + p[f]
            continue
        q = dict(p)
        q["_key"] = key
        merged.append(q)
    for q in merged:
        q.pop("_key", None)
    return merged


async def call_openrouter(client: "httpx.AsyncClient", api_key: str, model_key: str,
                          messages: List[Dict], params: Dict, progress: Dict) -> Dict:
    payload = {"model": model_key, "messages": messages, **params}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Spatio-Temporal NLP single-run experiment",
    }
    content: List[str] = []
    reasoning: List[str] = []
    reasoning_details: List = []
    res = {"finish_reason": None, "native_finish_reason": None, "usage": None, "provider": None,
           "generation_id": None, "model_served": None, "error": None, "done_marker": False,
           "bad_json_lines": 0}
    t0 = time.monotonic()
    progress.update(phase="connecting", t0=t0)
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S, write=60, pool=60)
    last_ui = 0.0
    async with client.stream("POST", OPENROUTER_CHAT_URL, json=payload, headers=headers,
                             timeout=timeout) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", "replace")[:4000]
            code = resp.status_code
            ra = resp.headers.get("retry-after")
            try:
                ra_f = float(ra) if ra else None
            except ValueError:
                ra_f = None
            msg = f"HTTP {code}: {body}"
            if code == 401:
                raise FatalRunError(f"Invalid or missing API key ({msg})")
            if code == 402:
                if "in_flight" in body:
                    # Not out of credits: OpenRouter reserves the maximum possible cost of every
                    # running request, and too many requests are running at once for the balance.
                    raise TransientError(msg, ra_f or 120.0, budget=True)
                raise FatalRunError(f"OpenRouter reports insufficient credits ({msg})")
            if code in (408, 409, 425, 429) or code >= 500:
                raise TransientError(msg, ra_f, rate_limited=(code == 429))
            raise ItemError(msg)
        progress["phase"] = "waiting"
        async for line in resp.aiter_lines():
            if time.monotonic() - t0 > CALL_HARD_LIMIT_S:
                raise TransientError(f"call exceeded hard limit of {CALL_HARD_LIMIT_S}s")
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                res["done_marker"] = True
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                res["bad_json_lines"] += 1
                continue
            if obj.get("error"):
                res["error"] = obj["error"]
            res["generation_id"] = obj.get("id") or res["generation_id"]
            res["provider"] = obj.get("provider") or res["provider"]
            res["model_served"] = obj.get("model") or res["model_served"]
            if obj.get("usage"):
                res["usage"] = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    progress["content_chars"] = progress.get("content_chars", 0) + len(delta["content"])
                    progress["phase"] = "writing"
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                    progress["reasoning_chars"] = progress.get("reasoning_chars", 0) + len(delta["reasoning"])
                    if progress.get("phase") != "writing":
                        progress["phase"] = "thinking"
                if delta.get("reasoning_details"):
                    reasoning_details.extend(delta["reasoning_details"])
                    if progress.get("phase") not in ("writing",):
                        progress["phase"] = "thinking"
                if ch.get("finish_reason"):
                    res["finish_reason"] = ch["finish_reason"]
                    res["native_finish_reason"] = ch.get("native_finish_reason")
            now = time.monotonic()
            if now - last_ui > 0.5:
                last_ui = now
                progress["rows"] = len(re.findall(r"(?m)^\W*E\d+\W*[:|]", "".join(content)))
    elapsed = time.monotonic() - t0
    text = "".join(content)
    res.update(content=text, reasoning="".join(reasoning),
               reasoning_details=merge_reasoning_details(reasoning_details),
               latency_s=round(elapsed, 3))
    if res["error"] or res["finish_reason"] == "error":
        err = res["error"] if isinstance(res["error"], dict) else {}
        raise TransientError(f"provider error mid-stream: {json.dumps(res['error'])[:1000]}",
                             rate_limited=str(err.get("code")) == "429")
    if not res["done_marker"] and res["finish_reason"] is None:
        raise TransientError("stream ended without a finish reason (connection dropped)")
    return res


# ======================================================================= the runner
class Runner:
    def __init__(self, args, api_key: str, models: List[Dict], items: List[Item], manifest: Dict,
                 logger: logging.Logger, log_path: Path):
        self.args = args
        self.api_key = api_key
        self.models = models
        self.items = {it.item_id: it for it in items}
        self.manifest = manifest
        self.log = logger
        self.log_path = log_path
        self.session = stamp()
        self.stop_requested = False
        self.abort_requested = False
        self.fatal: Optional[str] = None
        self.events: Deque[str] = collections.deque(maxlen=14)
        self.inflight: Dict[Tuple[str, str], Dict] = {}
        self.t_start = time.monotonic()
        self.status: Dict[Tuple[str, str], Dict] = {}
        self.stats = {m["model_id"]: collections.Counter() for m in models}
        self.cost = {m["model_id"]: 0.0 for m in models}
        self.lat = {m["model_id"]: [] for m in models}
        self.queues: Dict[str, Deque[str]] = {}
        self.total_per_model: Dict[str, int] = {}
        self.attempts_hist = collections.Counter()
        self.pause_until: Dict[str, float] = {m["model_id"]: 0.0 for m in models}
        self.workers_limit = args.concurrency          # may be lowered automatically (credit budget)
        self.global_pause_until = 0.0
        self._budget_cooldown_until = 0.0
        self.budget_hits_at_min = 0
        self._last_status_write = 0.0
        self._last_snapshot = 0.0

    # ------------------------------------------------------------ logging helpers
    def event(self, msg: str, level: int = logging.INFO, style: str = "") -> None:
        self.log.log(level, msg)
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[dim]{ts}[/dim] " + (f"[{style}]{_esc(msg)}[/{style}]" if style else _esc(msg)))

    # ------------------------------------------------------------ state from disk
    def raw_path(self, mid: str, iid: str) -> Path:
        return RAW_DIR / mid / f"{iid}.json"

    def load_state(self) -> None:
        order_rng = random.Random(FROZEN_SETTINGS["item_order_seed"])
        ids = sorted(self.items, key=lambda x: (not self.items[x].is_primary, self.items[x].stem_id,
                                                 self.items[x].batch or 0))
        order = ids[:]
        order_rng.shuffle(order)
        for m in self.models:
            mid = m["model_id"]
            q: Deque[str] = collections.deque()
            for iid in order:
                it = self.items[iid]
                p = self.raw_path(mid, iid)
                rec = None
                if p.exists():
                    try:
                        rec = read_json(p)
                    except Exception:
                        # A raw file is only ever written atomically, so this means it was
                        # damaged outside this program. Keep it aside and redo the call.
                        bad = p.with_suffix(f".corrupt_{stamp()}.json")
                        p.rename(bad)
                        self.event(f"{mid}/{iid}: unreadable raw file moved to {bad.name}; will redo",
                                   logging.WARNING, "yellow")
                if rec and rec.get("final"):
                    ps = self.ensure_parsed(mid, it, rec)
                    self.record_done(mid, it, rec, ps, quiet=True)
                else:
                    q.append(iid)
                    self.status[(mid, iid)] = self.status_row(mid, it, state="pending")
            self.queues[mid] = q
            self.total_per_model[mid] = len(order)
        self.write_status(force=True)

    def ensure_parsed(self, mid: str, it: Item, rec: Dict) -> Dict:
        pj = PARSED_DIR / mid / f"{it.item_id}.parse.json"
        pc = PARSED_DIR / mid / f"{it.item_id}.csv"
        if pj.exists() and pc.exists():
            try:
                return read_json(pj)
            except Exception:
                pass
        return self.parse_and_save(mid, it, rec)

    def parse_and_save(self, mid: str, it: Item, rec: Dict) -> Dict:
        parsed = parse_matrix(rec["response"].get("content"), it.event_ids)
        rows = matrix_to_rows(parsed["matrix"], it.event_ids)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["from_event", "to_event", "pred_symbol", "pred_code"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        atomic_write_text(PARSED_DIR / mid / f"{it.item_id}.csv", buf.getvalue())
        summary = {"item_id": it.item_id, "model_id": mid, "event_ids": it.event_ids,
                   **parsed["stats"]}
        atomic_write_json(PARSED_DIR / mid / f"{it.item_id}.parse.json", summary)
        return summary

    def status_row(self, mid: str, it: Item, state: str, rec: Optional[Dict] = None,
                   ps: Optional[Dict] = None) -> Dict:
        row = {k: "" for k in STATUS_FIELDS}
        row.update(model_id=mid, item_id=it.item_id, stem_id=it.stem_id, is_primary=it.is_primary,
                   n_events=it.n, state=state, cells_total=it.n * (it.n - 1))
        if rec:
            r = rec["response"]
            u = r.get("usage") or {}
            ctd = u.get("completion_tokens_details") or {}
            row.update(finish_reason=r.get("finish_reason"),
                       truncated=r.get("finish_reason") == "length",
                       prompt_tokens=u.get("prompt_tokens", ""),
                       completion_tokens=u.get("completion_tokens", ""),
                       reasoning_tokens=ctd.get("reasoning_tokens", u.get("reasoning_tokens", "")),
                       reasoning_chars=len(r.get("reasoning") or ""),
                       cost_usd=u.get("cost", ""), latency_s=r.get("latency_s", ""),
                       provider=r.get("provider", ""),
                       attempts_total=rec.get("attempts_total", ""))
        if ps:
            row.update(parse_status=ps.get("status"), cells_valid=ps.get("cells_valid"))
        return row

    def record_done(self, mid: str, it: Item, rec: Dict, ps: Dict, quiet: bool = False) -> None:
        self.status[(mid, it.item_id)] = self.status_row(mid, it, "done", rec, ps)
        st = self.stats[mid]
        st["done"] += 1
        st[f"parse_{ps.get('status')}"] += 1
        if rec["response"].get("finish_reason") == "length":
            st["truncated"] += 1
        u = rec["response"].get("usage") or {}
        try:
            self.cost[mid] += float(u.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
        st["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        st["completion_tokens"] += int(u.get("completion_tokens") or 0)
        if rec["response"].get("latency_s"):
            self.lat[mid].append(float(rec["response"]["latency_s"]))
        if not quiet:
            style = "green" if ps.get("status") == "complete" else "yellow"
            extra = " (TRUNCATED)" if rec["response"].get("finish_reason") == "length" else ""
            self.event(f"{mid} done {it.item_id} ({it.n} ev) parse={ps.get('status')} "
                       f"{ps.get('cells_valid')}/{ps.get('cells_total')} cells, "
                       f"{rec['response'].get('latency_s', 0):.0f}s{extra}", style=style)

    def write_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_write < 2.0:
            return
        self._last_status_write = now
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=STATUS_FIELDS, lineterminator="\n")
        w.writeheader()
        for k in sorted(self.status):
            w.writerow(self.status[k])
        try:
            atomic_write_text(STATUS_CSV, buf.getvalue())
        except PermissionError as e:
            self.event(f"status.csv not updated ({e}); it is rebuilt from raw files on every start",
                       logging.WARNING, "yellow")

    # ------------------------------------------------------------ worker
    async def worker(self, client, m: Dict, wid: int) -> None:
        mid = m["model_id"]
        params = self.manifest["models"][mid]["request_params"]
        q = self.queues[mid]
        while not self.stop_requested and not self.fatal:
            if not q or wid >= self.workers_limit:
                return
            iid = q.popleft()
            it = self.items[iid]
            messages = build_messages(it)
            mhash = messages_fingerprint(messages)
            if mhash != self.manifest["items"][iid]["prompt_sha256"]:
                self.fatal = f"Prompt for {iid} differs from the frozen manifest; refusing to continue."
                return
            attempt = 0
            rl_retries = 0
            prev_attempts = len(list((ERR_DIR / mid).glob(f"{iid}__*.json"))) if (ERR_DIR / mid).exists() else 0
            while True:
                attempt += 1
                key = (mid, iid)
                # a rate limit (HTTP 429) on this model pauses all of its workers briefly
                if await self.wait_model_pause(mid):
                    q.appendleft(iid)
                    self.status[key]["state"] = "pending"
                    return
                prog = {"model": mid, "item": iid, "n": it.n, "attempt": attempt, "phase": "starting",
                        "t0": time.monotonic(), "content_chars": 0, "reasoning_chars": 0, "rows": 0}
                self.inflight[key] = prog
                self.status[key]["state"] = "running"
                started = utcnow()
                try:
                    res = await call_openrouter(client, self.api_key, m["model_key"], messages,
                                                params, prog)
                except asyncio.CancelledError:
                    self.save_error(mid, iid, attempt, started, "aborted", "abandoned by user (Ctrl+C twice)", prog)
                    self.status[key]["state"] = "pending"
                    self.inflight.pop(key, None)
                    raise
                except FatalRunError as e:
                    self.inflight.pop(key, None)
                    self.save_error(mid, iid, attempt, started, "fatal", str(e), prog)
                    self.fatal = str(e)
                    self.status[key]["state"] = "pending"
                    self.event(f"FATAL: {e}", logging.ERROR, "bold red")
                    return
                except ItemError as e:
                    self.inflight.pop(key, None)
                    self.save_error(mid, iid, attempt, started, "request_error", str(e), prog)
                    self.stats[mid]["item_errors"] += 1
                    self.status[key]["state"] = "error_retry_next_start"
                    self.event(f"{mid} REJECTED {iid}: {str(e)[:160]} "
                               f"(will be retried next start)", logging.ERROR, "red")
                    break
                except (TransientError, httpx.HTTPError, OSError) as e:
                    self.inflight.pop(key, None)
                    kind = type(e).__name__
                    self.save_error(mid, iid, attempt, started, "transient", f"{kind}: {e}", prog)
                    if getattr(e, "budget", False):
                        # HTTP 402 in-flight budget: pause ALL models, halve the workers per model
                        attempt -= 1
                        self.stats[mid]["budget_waits"] += 1
                        now = time.monotonic()
                        if now >= self._budget_cooldown_until:
                            old = self.workers_limit
                            if old == 1:
                                self.budget_hits_at_min += 1
                            self.workers_limit = max(1, old // 2)
                            self._budget_cooldown_until = now + 30
                            self.event(f"OpenRouter credit in-flight budget reached (HTTP 402): too many calls "
                                       f"running for your credit balance. Pausing all models for "
                                       f"{min(getattr(e, 'retry_after', None) or 120, 180):.0f}s and lowering workers/model "
                                       f"{old} -> {self.workers_limit}. Nothing is lost.", logging.WARNING, "bold yellow")
                        pause = min(getattr(e, "retry_after", None) or 120, 180)
                        self.global_pause_until = max(self.global_pause_until, now + pause)
                        if self.budget_hits_at_min >= 3:
                            self.status[key]["state"] = "pending"
                            q.appendleft(iid)
                            self.fatal = ("OpenRouter's in-flight credit budget is exhausted even with 1 worker per "
                                          "model. Add credits at https://openrouter.ai/settings/credits, then run "
                                          "the same command again (everything done so far is saved).")
                            self.event(f"FATAL: {self.fatal}", logging.ERROR, "bold red")
                            return
                        if wid >= self.workers_limit:
                            q.appendleft(iid)
                            self.status[key]["state"] = "pending"
                            return
                        self.status[key]["state"] = "waiting_retry"
                        continue
                    if getattr(e, "rate_limited", False):
                        # 429: not a failure of this item; pause the model's workers and retry
                        rl_retries += 1
                        attempt -= 1
                        self.stats[mid]["rate_limited"] += 1
                        ra = getattr(e, "retry_after", None)
                        pause = min(max(ra or 0, 5 * rl_retries), 120) * (1 + random.random() * 0.2)
                        self.pause_until[mid] = max(self.pause_until[mid], time.monotonic() + pause)
                        if rl_retries > MAX_RATE_LIMIT_RETRIES:
                            self.status[key]["state"] = "failed_retry_next_start"
                            self.event(f"{mid} FAILED {iid}: still rate-limited after {rl_retries} retries; "
                                       f"will be retried next start (try fewer --workers)", logging.ERROR, "red")
                            break
                        self.status[key]["state"] = "waiting_retry"
                        self.event(f"{mid} rate-limited (HTTP 429): pausing this model for {pause:.0f}s "
                                   f"(consider fewer --workers if this repeats)", logging.WARNING, "yellow")
                        continue
                    self.stats[mid]["retries"] += 1
                    if attempt >= MAX_ATTEMPTS_PER_SESSION:
                        self.status[key]["state"] = "failed_retry_next_start"
                        self.event(f"{mid} FAILED {iid}: gave up after {attempt} attempts this session "
                                   f"({kind}); will be retried next start", logging.ERROR, "red")
                        break
                    wait = BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)]
                    ra = getattr(e, "retry_after", None)
                    if ra:
                        wait = max(wait, min(ra, 300))
                    wait = wait * (1 + random.random() * 0.2)
                    self.status[key]["state"] = "waiting_retry"
                    self.event(f"{mid} retry {iid}: {kind} - {str(e)[:120]}; retry {attempt + 1} in {wait:.0f}s",
                               logging.WARNING, "yellow")
                    if await self.sleep_or_stop(wait):
                        self.status[key]["state"] = "pending"
                        break
                    continue
                except Exception as e:  # unexpected programming error: keep the run alive
                    self.inflight.pop(key, None)
                    self.save_error(mid, iid, attempt, started, "unexpected", traceback.format_exc(), prog)
                    self.status[key]["state"] = "failed_retry_next_start"
                    self.event(f"{mid} FAILED {iid}: unexpected error {type(e).__name__}: {e}",
                               logging.ERROR, "red")
                    break
                # ---- success: save raw, parse, update status (each saved immediately)
                self.inflight.pop(key, None)
                rec = {
                    "final": True,
                    "model_id": mid, "model_key": m["model_key"], "item_id": iid,
                    "stem_id": it.stem_id, "is_primary": it.is_primary, "event_ids": it.event_ids,
                    "prompt_sha256": mhash, "prompt_file": f"prompts/{iid}.txt",
                    "request_params": params, "session": self.session,
                    "started_at": started, "finished_at": utcnow(),
                    "attempt_in_session": attempt, "attempts_total": prev_attempts + attempt,
                    "response": res,
                }
                atomic_write_json(self.raw_path(mid, iid), rec)
                ps = self.parse_and_save(mid, it, rec)
                self.record_done(mid, it, rec, ps)
                self.write_status()
                break

    async def wait_model_pause(self, mid: str) -> bool:
        """Wait while this model is paused after a rate limit. Returns True if a stop was requested."""
        while time.monotonic() < max(self.pause_until.get(mid, 0.0), self.global_pause_until):
            if self.stop_requested or self.fatal:
                return True
            await asyncio.sleep(0.25)
        return self.stop_requested or bool(self.fatal)

    async def sleep_or_stop(self, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_requested or self.fatal:
                return True
            await asyncio.sleep(0.25)
        return False

    def save_error(self, mid, iid, attempt, started, kind, message, prog) -> None:
        try:
            atomic_write_json(ERR_DIR / mid / f"{iid}__{stamp()}{datetime.now().strftime('%f')[:3]}__a{attempt}.json", {
                "model_id": mid, "item_id": iid, "session": self.session, "attempt_in_session": attempt,
                "started_at": started, "failed_at": utcnow(), "kind": kind, "message": message,
                "partial_content_chars": prog.get("content_chars"),
                "partial_reasoning_chars": prog.get("reasoning_chars"),
            })
        except Exception as e:  # never let error logging kill the run
            self.log.error(f"could not save error record: {e}")

    # ------------------------------------------------------------ UI
    def render(self):
        elapsed = time.monotonic() - self.t_start
        total = sum(self.total_per_model.values())
        done = sum(s["done"] for s in self.stats.values())
        cost = sum(self.cost.values())
        if self.fatal:
            state = Text("STOPPED - FATAL ERROR", style="bold white on red")
        elif self.abort_requested:
            state = Text("ABORTING", style="bold white on red")
        elif self.stop_requested:
            state = Text(f"STOPPING - waiting for {len(self.inflight)} running call(s)  "
                         f"(Ctrl+C again = abandon them)", style="bold black on yellow")
        else:
            state = Text("RUNNING", style="bold black on green")
        head = Table.grid(expand=True)
        head.add_column(ratio=1)
        head.add_column(justify="right")
        head.add_row(Text.assemble(("Spatio-Temporal NLP - single model run  ", "bold"), state),
                     Text((f"workers/model {self.workers_limit}" +
                           (f" (started {self.args.concurrency}; lowered for credit budget)"
                            if self.workers_limit != self.args.concurrency else "") +
                           (f"  |  ALL PAUSED {self.global_pause_until - time.monotonic():.0f}s"
                            if self.global_pause_until > time.monotonic() else "")) +
                          f"  |  elapsed {_fmt_dur(elapsed)}  |  "
                          f"calls {done}/{total}  |  cost ${cost:.4f}"))

        t = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
        for col, kw in [("model", {}), ("progress", {"ratio": 2}), ("done", {"justify": "right"}),
                        ("complete", {"justify": "right"}), ("partial", {"justify": "right"}),
                        ("failed parse", {"justify": "right"}), ("truncated", {"justify": "right"}),
                        ("retries", {"justify": "right"}), ("429/402 waits", {"justify": "right"}),
                        ("rejected", {"justify": "right"}),
                        ("running", {"justify": "right"}), ("avg s", {"justify": "right"}),
                        ("cost $", {"justify": "right"})]:
            t.add_column(col, **kw)
        for m in self.models:
            mid = m["model_id"]
            s = self.stats[mid]
            tot = self.total_per_model.get(mid, 0) or 1
            running = sum(1 for k in self.inflight if k[0] == mid)
            paused = self.pause_until.get(mid, 0.0) - time.monotonic()
            running_txt = f"{running}" + (f" (paused {paused:.0f}s)" if paused > 0 else "")
            lat = self.lat[mid]
            t.add_row(
                Text(mid, style="bold"),
                ProgressBar(total=tot, completed=s["done"], width=None),
                f"{s['done']}/{tot}", str(s["parse_complete"]), str(s["parse_partial"]),
                str(s["parse_failed"]), str(s["truncated"]), str(s["retries"]), str(s["rate_limited"] + s["budget_waits"]),
                str(s["item_errors"]), running_txt, f"{sum(lat) / len(lat):.0f}" if lat else "-", f"{self.cost[mid]:.4f}")

        f = Table(box=box.MINIMAL, expand=True, pad_edge=False, show_edge=False)
        for col in ("model", "item", "events", "try", "phase", "thinking chars", "answer chars",
                    "rows", "time"):
            f.add_column(col)
        now = time.monotonic()
        for (mid, iid), p in sorted(self.inflight.items(), key=lambda kv: kv[1]["t0"])[:18]:
            ph = p.get("phase", "")
            ph_style = {"thinking": "magenta", "writing": "cyan", "waiting": "dim", "connecting": "dim"}.get(ph, "")
            f.add_row(mid, iid, str(p["n"]), str(p["attempt"]), Text(ph, style=ph_style),
                      f"{p.get('reasoning_chars', 0):,}", f"{p.get('content_chars', 0):,}",
                      f"{p.get('rows', 0)}/{p['n']}", _fmt_dur(now - p["t0"]))
        if len(self.inflight) > 18:
            f.add_row("...", f"+{len(self.inflight) - 18} more", "", "", "", "", "", "", "")
        ev = Text.from_markup("\n".join(self.events) if self.events else "[dim]no events yet[/dim]")
        foot = Text(f"Ctrl+C: stop gracefully (running calls finish and are saved)  |  Ctrl+C twice: "
                    f"stop now  |  re-run the same command to resume  |  log: {self.log_path}",
                    style="dim")
        return Group(
            Panel(head, box=box.ROUNDED),
            Panel(t, title="models", box=box.ROUNDED),
            Panel(f, title=f"running calls ({len(self.inflight)})", box=box.ROUNDED),
            Panel(ev, title="recent events", box=box.ROUNDED),
            foot,
        )

    def snapshot(self, force: bool = False) -> None:
        """Save a plain-text copy of the dashboard (results/logs/dashboard_latest.txt)."""
        now = time.monotonic()
        if not force and now - self._last_snapshot < 5:
            return
        self._last_snapshot = now
        try:
            c = Console(file=io.StringIO(), width=160, record=True, color_system=None)
            c.print(self.render())
            atomic_write_text(LOG_DIR / "dashboard_latest.txt",
                              f"snapshot {datetime.now().isoformat(timespec='seconds')}\n" + c.export_text())
        except Exception:
            pass

    # ------------------------------------------------------------ main loop
    async def run(self, console: Console) -> int:
        loop = asyncio.get_running_loop()
        tasks: List[asyncio.Task] = []

        def on_sigint(signum, frame):  # runs in the main thread
            if not self.stop_requested:
                self.stop_requested = True
                loop.call_soon_threadsafe(lambda: self.event(
                    "Ctrl+C received - finishing running calls, no new calls will start. "
                    "Press Ctrl+C again to stop immediately.", logging.WARNING, "bold yellow"))
            else:
                self.abort_requested = True

                def _cancel():
                    self.event("Second Ctrl+C - abandoning running calls (they will be redone on resume).",
                               logging.WARNING, "bold red")
                    for tk in tasks:
                        tk.cancel()
                loop.call_soon_threadsafe(_cancel)

        signal.signal(signal.SIGINT, on_sigint)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, on_sigint)  # Windows Ctrl+Break

        limits = httpx.Limits(max_connections=self.args.concurrency * len(self.models) + 8,
                              max_keepalive_connections=self.args.concurrency * len(self.models) + 8)
        async with httpx.AsyncClient(limits=limits, http2=False) as client:
            for m in self.models:
                n_workers = min(self.args.concurrency, len(self.queues[m["model_id"]]))
                for w in range(n_workers):
                    tasks.append(asyncio.create_task(self.worker(client, m, w)))
            pending_total = sum(len(q) for q in self.queues.values())
            self.event(f"session {self.session}: {pending_total} call(s) to do, "
                       f"{sum(s['done'] for s in self.stats.values())} already done", style="bold")
            with Live(self.render(), console=console, refresh_per_second=4, screen=False,
                      transient=False) as live:
                while True:
                    alive = [tk for tk in tasks if not tk.done()]
                    live.update(self.render())
                    self.snapshot()
                    self.write_status()
                    if not alive:
                        break
                    if self.fatal and not self.stop_requested:
                        self.stop_requested = True
                    await asyncio.sleep(0.25)
                live.update(self.render())
            for tk in tasks:
                if tk.done() and not tk.cancelled() and tk.exception():
                    exc = tk.exception()
                    self.log.error("worker crashed: %s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self.write_status(force=True)
        self.snapshot(force=True)
        return 0


# ======================================================================= helpers
def _esc(s: str) -> str:
    return s.replace("[", "\\[")


def _fmt_dur(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def setup_logging(kind: str) -> Tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{kind}_{stamp()}.log"
    lg = logging.getLogger(f"stnlp.{kind}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    lg.addHandler(fh)
    lg.propagate = False
    return lg, path


def estimate_cost(items: List[Item], manifest_models: Dict, reasoning_multiplier: float) -> Dict[str, float]:
    """Rough estimate: input ~ len(prompt)/4 tokens; visible output ~ 2.2 tokens per matrix
    cell + 4 per row; reasoning tokens = multiplier x visible output (assumption)."""
    out = {}
    for mid, mm in manifest_models.items():
        pr = mm["metadata"].get("pricing") or {}
        p_in = float(pr.get("prompt") or 0)
        p_out = float(pr.get("completion") or 0)
        tin = tout = 0.0
        for it in items:
            msgs = build_messages(it)
            tin += sum(len(x["content"]) for x in msgs) / 4
            vis = 2.2 * it.n * it.n + 4 * it.n
            reas = reasoning_multiplier * vis if "reasoning" in mm["request_params"] else 0
            tout += vis + reas
        out[mid] = tin * p_in + tout * p_out
    return out


# ======================================================================= manifest
def build_or_check_manifest(models: List[Dict], items: List[Item], console: Console,
                            logger: logging.Logger) -> Dict:
    tfp = template_fingerprint()
    dfp = data_fingerprint()
    item_hashes = {it.item_id: messages_fingerprint(build_messages(it)) for it in items}
    if MANIFEST.exists():
        man = read_json(MANIFEST)
        problems = []
        if man["data_sha256"] != dfp:
            problems.append("the gold data in exports/ changed since the run started")
        if man["prompt_template_sha256"] != tfp:
            problems.append("the prompt template (scripts/common.py) changed since the run started")
        if man["frozen_settings"] != FROZEN_SETTINGS:
            problems.append("FROZEN_SETTINGS in run_models.py changed since the run started")
        if set(man["items"]) != set(item_hashes):
            problems.append("the item list changed since the run started")
        else:
            diff = [k for k, v in item_hashes.items() if man["items"][k]["prompt_sha256"] != v]
            if diff:
                problems.append(f"prompts changed for {len(diff)} item(s), e.g. {diff[:3]}")
        new_models = []
        for m in models:
            mm = man["models"].get(m["model_id"])
            if mm is None:
                new_models.append(m)
            elif mm["model_key"] != m["model_key"]:
                problems.append(f"model_id {m['model_id']} now points to {m['model_key']} "
                                f"(was {mm['model_key']})")
        removed = set(man["models"]) - {m["model_id"] for m in models}
        if removed:
            console.print(f"[yellow]Note: models in the manifest but no longer in models.json are "
                          f"skipped this session: {sorted(removed)}[/yellow]")
        if problems:
            console.print(Panel("\n".join(f"- {p}" for p in problems) +
                                "\n\nThe run was started with different inputs. To protect the single-run "
                                "design nothing will be sent. Restore the original files, or move the "
                                "'results' folder away to start a completely new run.",
                                title="Cannot resume", style="red"))
            sys.exit(2)
        if new_models:
            meta = fetch_model_metadata([m["model_key"] for m in new_models])
            for m in new_models:
                if meta[m["model_key"]] is None:
                    console.print(f"[red]Model {m['model_key']} not found on OpenRouter.[/red]")
                    sys.exit(2)
                rp = build_request_params(meta[m["model_key"]])
                man["models"][m["model_id"]] = {"model_key": m["model_key"], "metadata": meta[m["model_key"]],
                                                "request_params": rp["params"], "notes": rp["notes"],
                                                "added_at": utcnow()}
                logger.info("added new model to manifest: %s", m["model_id"])
            atomic_write_json(MANIFEST, man)
        return man
    # ---- first start: freeze everything
    meta = fetch_model_metadata([m["model_key"] for m in models])
    missing = [k for k, v in meta.items() if v is None]
    if missing:
        console.print(f"[red]These model keys were not found on OpenRouter: {missing}. Fix models.json.[/red]")
        sys.exit(2)
    man = {
        "created_at": utcnow(),
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": tfp,
        "data_sha256": dfp,
        "frozen_settings": FROZEN_SETTINGS,
        "excluded_items": EXCLUDED_ITEMS,
        "models": {},
        "items": {},
    }
    for m in models:
        rp = build_request_params(meta[m["model_key"]])
        man["models"][m["model_id"]] = {"model_key": m["model_key"], "metadata": meta[m["model_key"]],
                                        "request_params": rp["params"], "notes": rp["notes"]}
    for it in items:
        man["items"][it.item_id] = {"stem_id": it.stem_id, "batch": it.batch, "is_primary": it.is_primary,
                                    "n_events": it.n, "file": it.file, "prompt_sha256": item_hashes[it.item_id],
                                    "notes": it.notes}
    return man


def write_prompts(items: List[Item]) -> None:
    for it in items:
        p = PROMPT_DIR / f"{it.item_id}.txt"
        if p.exists():
            continue
        msgs = build_messages(it)
        atomic_write_text(p, "".join(f"===== {m['role'].upper()} =====\n{m['content']}\n\n" for m in msgs))


PREREG = """# Pre-registration (written automatically before the first model call)

Created: {created}
Prompt template sha256: {tfp}
Gold data sha256: {dfp}

## Design
- One call per (model, item); identical frozen prompt for all models; settings in run_manifest.json.
- Models receive only the passage and its event list (plus task instructions). No gold relations.
- Items: {n_primary} primary stems (stem_254 excluded: duplicate event ID E18) + {n_batch} '_batch_'
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
"""


# ======================================================================= modes
async def smoke(args, console: Console, logger) -> int:
    env = load_env()
    key = env.get("OPENROUTER_API_KEY")
    if not key:
        console.print("[red]OPENROUTER_API_KEY not found in .env[/red]")
        return 2
    models = load_models()
    meta = fetch_model_metadata([m["model_key"] for m in models])
    msgs = build_messages(SMOKE_ITEM)
    results = []

    async def one(client, m):
        mm = meta[m["model_key"]]
        if mm is None:
            return m, None, "model not found on OpenRouter"
        rp = build_request_params(mm)
        prog: Dict = {}
        try:
            res = await call_openrouter(client, key, m["model_key"], msgs, rp["params"], prog)
            return m, res, None
        except Exception as e:
            return m, None, f"{type(e).__name__}: {e}"

    with console.status("Sending the made-up 3-event story to every model..."):
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*[one(client, m) for m in models])
    t = Table(title="Smoke test (made-up story, not from the dataset)", box=box.SIMPLE_HEAVY)
    for c in ("model", "ok", "finish", "provider", "parse", "reasoning tok", "out tok", "cost $", "s", "answer"):
        t.add_column(c)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for m, res, err in results:
        if err:
            all_ok = False
            t.add_row(m["model_id"], "[red]no[/red]", "", "", "", "", "", "", "", err[:80])
            atomic_write_json(SMOKE_DIR / f"{m['model_id']}.json", {"error": err, "at": utcnow()})
            logger.error("smoke %s: %s", m["model_id"], err)
            continue
        ps = parse_matrix(res["content"], SMOKE_ITEM.event_ids)["stats"]
        u = res.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", "")
        ok = ps["status"] == "complete"
        all_ok &= ok
        t.add_row(m["model_id"], "[green]yes[/green]" if ok else "[yellow]check[/yellow]",
                  str(res["finish_reason"]), str(res["provider"]), ps["status"], str(rt),
                  str(u.get("completion_tokens", "")), str(u.get("cost", "")), f"{res['latency_s']:.1f}",
                  res["content"].strip().replace("\n", " | ")[:70])
        atomic_write_json(SMOKE_DIR / f"{m['model_id']}.json",
                          {"at": utcnow(), "request_params": build_request_params(meta[m['model_key']]),
                           "response": res, "parse": ps})
        logger.info("smoke %s: finish=%s parse=%s", m["model_id"], res["finish_reason"], ps["status"])
    console.print(t)
    console.print(f"Full responses saved in {SMOKE_DIR}" + ("" if all_ok else
                  "  [yellow](check rows marked above before the real run)[/yellow]"))
    return 0 if all_ok else 1


def check(console: Console) -> int:
    models = load_models()
    meta = fetch_model_metadata([m["model_key"] for m in models])
    t = Table(title="Models on OpenRouter (no generation, no cost)", box=box.SIMPLE_HEAVY)
    for c in ("model_id", "model_key", "found", "context", "max out", "params sent", "notes"):
        t.add_column(c)
    bad = False
    for m in models:
        mm = meta[m["model_key"]]
        if mm is None:
            bad = True
            t.add_row(m["model_id"], m["model_key"], "[red]NO[/red]", "", "", "", "")
            continue
        rp = build_request_params(mm)
        shown = {k: v for k, v in rp["params"].items() if k != "stream"}
        t.add_row(m["model_id"], m["model_key"], "[green]yes[/green]", str(mm["context_length"]),
                  str(mm["max_completion_tokens"]), json.dumps(shown), "; ".join(rp["notes"]))
    console.print(t)
    return 2 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify models on OpenRouter (no cost)")
    ap.add_argument("--smoke", action="store_true", help="send one made-up 3-event story to each model")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and a cost estimate; no calls")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before the first call")
    ap.add_argument("--workers", type=int, nargs="?", const=WORKERS_CONST, default=None, metavar="N",
                    help=f"parallel calls per model; all models always run at the same time. "
                         f"'--workers' alone = {WORKERS_CONST}; default without the flag = {DEFAULT_CONCURRENCY}")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="older name for --workers N")
    ap.add_argument("--models", nargs="*", default=None,
                    help="optional subset of model_id values to run this session")
    args = ap.parse_args()
    if args.workers is not None:
        args.concurrency = args.workers
    if args.concurrency < 1:
        print("--workers must be at least 1")
        return 2
    console = Console()
    if sys.version_info < (3, 9):
        console.print("[red]Python 3.9 or newer is required.[/red]")
        return 2
    if args.check:
        return check(console)
    if args.smoke:
        lg, lp = setup_logging("smoke")
        return asyncio.run(smoke(args, console, lg))

    logger, log_path = setup_logging("run")
    env = load_env()
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        console.print(f"[red]OPENROUTER_API_KEY not found in {ROOT / '.env'}[/red]")
        return 2
    models = load_models()
    if args.models:
        unknown = set(args.models) - {m["model_id"] for m in models}
        if unknown:
            console.print(f"[red]Unknown model_id(s): {sorted(unknown)}[/red]")
            return 2
    all_items = load_items(include_excluded=True)
    excluded = [it for it in all_items if it.excluded_reason]
    items = [it for it in all_items if not it.excluded_reason]
    if not FROZEN_SETTINGS["include_batch_items"]:
        items = [it for it in items if it.is_primary]
    if os.environ.get("STNLP_SELFTEST_LIMIT"):  # offline self-test only
        items = items[: int(os.environ["STNLP_SELFTEST_LIMIT"])]
    first_start = not MANIFEST.exists()
    man = build_or_check_manifest(models, items, console, logger)

    n_primary = sum(it.is_primary for it in items)
    n_batch = len(items) - n_primary
    info = Table.grid(padding=(0, 2))
    info.add_row("items", f"{n_primary} primary stems + {n_batch} batch versions = {len(items)} per model")
    info.add_row("excluded", "; ".join(f"{it.item_id} ({it.excluded_reason})" for it in excluded) or "none")
    info.add_row("models", ", ".join(m["model_id"] for m in models))
    for m in models:
        mm = man["models"][m["model_id"]]
        shown = {k: v for k, v in mm["request_params"].items() if k != "stream"}
        info.add_row(f"  {m['model_id']}", f"{m['model_key']}  {json.dumps(shown)}"
                     + (f"  [dim]({'; '.join(mm['notes'])})[/dim]" if mm["notes"] else ""))
    info.add_row("results", str(RESULTS))
    console.print(Panel(info, title="Run plan", box=box.ROUNDED))

    if args.dry_run or first_start:
        lo = estimate_cost(items, {m["model_id"]: man["models"][m["model_id"]] for m in models}, 1.0)
        hi = estimate_cost(items, {m["model_id"]: man["models"][m["model_id"]] for m in models}, 8.0)
        t = Table(title="Rough cost estimate (reasoning length is unknown; assumption: 1x-8x the answer)",
                  box=box.SIMPLE)
        t.add_column("model")
        t.add_column("low $", justify="right")
        t.add_column("high $", justify="right")
        for k in lo:
            t.add_row(k, f"{lo[k]:.2f}", f"{hi[k]:.2f}")
        t.add_row("[bold]total[/bold]", f"[bold]{sum(lo.values()):.2f}[/bold]", f"[bold]{sum(hi.values()):.2f}[/bold]")
        console.print(t)
    if args.dry_run:
        console.print("[dim]Dry run: nothing was sent to any model (only a log file was written).[/dim]")
        return 0

    if first_start:
        if not args.yes:
            console.print("[bold]This starts the one-time model run. Type YES to continue:[/bold] ", end="")
            try:
                ans = input().strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans != "YES":
                console.print("Not started. Nothing was sent.")
                return 1
        RESULTS.mkdir(parents=True, exist_ok=True)
        atomic_write_json(MANIFEST, man)
        atomic_write_text(RESULTS / "PREREGISTRATION.md", PREREG.format(
            created=man["created_at"], tfp=man["prompt_template_sha256"], dfp=man["data_sha256"],
            n_primary=n_primary, n_batch=n_batch))
        logger.info("manifest and pre-registration written")
    write_prompts(items)

    run_models = [m for m in models if (not args.models or m["model_id"] in args.models)]
    runner = Runner(args, api_key, run_models, items, man, logger, log_path)
    runner.load_state()
    logger.info("session %s start: models=%s workers per model=%d", runner.session,
                [m["model_id"] for m in run_models], args.concurrency)
    if sum(len(q) for q in runner.queues.values()) == 0:
        console.print("[green]All calls are already done. Next step: python scripts/analyze.py[/green]")
        return 0
    rc = asyncio.run(runner.run(console))
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # ---- final summary (printed and logged)
    done = sum(s["done"] for s in runner.stats.values())
    total = sum(runner.total_per_model.values())
    remaining = total - done
    if runner.fatal:
        msg = f"Stopped because of a fatal error: {runner.fatal}"
        console.print(Panel(msg, style="red"))
        logger.error(msg)
        rc = 3
    if remaining == 0:
        msg = "All calls are done. Next step:  python scripts/analyze.py"
        console.print(Panel(msg, style="green"))
    else:
        msg = (f"{done}/{total} calls done, {remaining} remaining. "
               f"Run the same command again to resume.")
        console.print(Panel(msg, style="yellow"))
    logger.info("session end: %s", msg)
    console.print(f"[dim]Log: {log_path}[/dim]")
    if runner.abort_requested:
        return 130
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C before the dashboard started (while loading/confirming): nothing in flight.
        print("\nStopped by Ctrl+C. Everything finished so far is saved; run the same command to resume.")
        sys.exit(130)
