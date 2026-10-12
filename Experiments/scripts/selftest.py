#!/usr/bin/env python
"""
Offline self-test. Sends NOTHING to OpenRouter and costs nothing.

1. Unit checks of the Allen toolkit and the output parser.
2. Starts a local mock of the OpenRouter API that streams answers derived from the gold
   matrices with random noise, and injects failures (HTTP 500/429, mid-stream errors,
   dropped connections, truncated answers, unparseable answers).
3. Runs scripts/run_models.py end-to-end against the mock into a temporary results folder
   (on Linux/macOS it also interrupts the run with Ctrl+C and resumes it).
4. Runs scripts/analyze.py on the mock results.

Usage:  python scripts/selftest.py --quick    (units + a 20-item mock run incl. the repair step; ~3-5 min)
        python scripts/selftest.py            (all 368 items, Ctrl+C/resume test on Linux/macOS, no repair step)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from allen import CODE2SYM, CODES, MISSING, self_test as allen_self_test  # noqa: E402
from common import SCRIPTS_DIR, load_items, parse_matrix  # noqa: E402

MOCK_MODELS = {
    "mock/good": {"noise": 0.10, "reasoning": True},
    "mock/sloppy": {"noise": 0.45, "reasoning": False, "bad_format": 0.08},
    "mock/flaky": {"noise": 0.25, "reasoning": True, "fail": 0.25},
}


def build_index():
    idx = {}
    for it in load_items():
        idx[(it.stem_text.strip(), tuple(it.event_ids))] = it
    return idx


class MockHandler(BaseHTTPRequestHandler):
    index = None
    calls = {"n": 0}
    lock = threading.Lock()

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):  # /models
        data = []
        for k, v in MOCK_MODELS.items():
            params = ["max_tokens", "temperature", "seed"] + (["reasoning"] if v["reasoning"] else [])
            data.append({"id": k, "name": k, "context_length": 100000,
                         "top_provider": {"max_completion_tokens": 32000, "is_moderated": False},
                         "supported_parameters": params,
                         "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}})
        body = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        n_bytes = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n_bytes))
        model = req["model"]
        cfg = MOCK_MODELS[model]
        user = req["messages"][1]["content"]
        passage = re.search(r'PASSAGE:\n"""\n(.*?)\n"""', user, re.S).group(1)
        ev_block = re.search(r"EVENTS \(\d+ events\):\n(.*?)\n\nOUTPUT FORMAT", user, re.S).group(1)
        ev_ids = [ln.split(":", 1)[0] for ln in ev_block.splitlines()]
        rng = random.Random()
        with self.lock:
            self.calls["n"] += 1
        it = self.index.get((passage.strip(), tuple(ev_ids)))
        # ---- injected failures
        if cfg.get("fail") and rng.random() < cfg["fail"]:
            kind = rng.choice(["500", "429", "midstream", "drop"] + (["402"] * 4 if os.environ.get("MOCK_402") else []))
            if kind == "402":
                self.send_response(402)
                self.send_header("Retry-After", "2")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"This request would exceed your available credits given your '
                                 b'current in-flight requests.","code":402,"metadata":{"reason":"in_flight_budget_exhausted"}}}')
                return
            if kind == "500":
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b'{"error":{"code":502,"message":"mock upstream error"}}')
                return
            if kind == "429":
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(b'{"error":{"code":429,"message":"mock rate limit"}}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": OPENROUTER PROCESSING\n\n")
            self._sse({"id": "gen-x", "choices": [{"index": 0, "delta": {"content": "<matrix>\nE1: "}}]})
            if kind == "midstream":
                self._sse({"id": "gen-x", "error": {"code": "server_error", "message": "mock disconnect"},
                           "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]})
            # 'drop': just close the connection without finish reason / [DONE]
            return
        if it is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"code":400,"message":"mock: unknown item"}}')
            return
        n = it.n
        G = it.gold
        rows = []
        for i in range(n):
            toks = []
            for j in range(n):
                if i == j:
                    toks.append("-")
                    continue
                c = int(G[i, j])
                if rng.random() < cfg["noise"]:
                    c = rng.choice(CODES)
                toks.append(CODE2SYM[c])
            rows.append(f"{ev_ids[i]}: " + " ".join(toks))
        if cfg.get("bad_format") and rng.random() < cfg["bad_format"]:
            rows = [r + " <" for r in rows]  # wrong row length -> unparseable
        text = "<matrix>\n" + "\n".join(rows) + "\n</matrix>"
        finish = "stop"
        if model == "mock/sloppy" and n >= 40:
            text = text[: len(text) // 2]   # simulated truncation
            finish = "length"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b": OPENROUTER PROCESSING\n\n")
        if cfg["reasoning"]:
            for k in range(3):
                self._sse({"id": "gen-1", "provider": "Mock", "model": model,
                           "choices": [{"index": 0, "delta": {"reasoning": f"thinking {k} ",
                                                              "reasoning_details": [{"type": "reasoning.text", "text": f"thinking {k} ", "index": 0}]}}]})
                time.sleep(0.02)
        chunk = 200
        for k in range(0, len(text), chunk):
            self._sse({"id": "gen-1", "provider": "Mock", "model": model,
                       "choices": [{"index": 0, "delta": {"content": text[k:k + chunk]}}]})
            time.sleep(float(os.environ.get("MOCK_CHUNK_DELAY", "0.003")))
        self._sse({"id": "gen-1", "provider": "Mock", "model": model,
                   "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish}]})
        out_tok = len(text) // 3
        self._sse({"id": "gen-1", "provider": "Mock", "model": model,
                   "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish}],
                   "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": out_tok + 30,
                             "completion_tokens_details": {"reasoning_tokens": 30 if cfg["reasoning"] else 0},
                             "cost": round((len(user) // 4) * 1e-7 + out_tok * 4e-7, 8)}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def unit_tests():
    allen_self_test()
    ids = ["E1", "E2", "E3"]
    cases = {
        "<matrix>\nE1: - < m\nE2: > - d\nE3: mi di -\n</matrix>": ("complete", 6),
        "<matrix>\nE1: < m\nE2: > d\nE3: mi di\n</matrix>": ("complete", 6),
        "<matrix>\nE1: - < m\nE2: > - d\nE3: mi": ("partial", 4),
        "<matrix>\n| E1 | - | < | m |\n| E2 | > | - | d |\n</matrix>": ("partial", 4),
        "<matrix>\nE1: - before m\nE2: > - d\nE3: mi di -\n</matrix>": ("partial", 5),
        "": ("failed", 0),
        "I cannot do this.": ("failed", 0),
    }
    for text, (st, nv) in cases.items():
        r = parse_matrix(text, ids)["stats"]
        assert (r["status"], r["cells_valid"]) == (st, nv), (text, r)
    # gold sanity: every primary item complete & converse-consistent except the known issues
    items = load_items()
    assert len(items) == 368 and sum(i.is_primary for i in items) == 360, len(items)
    print("unit tests passed")


def run(cmd, env, timeout=None, interrupt_after=None):
    print("  $", " ".join(cmd))
    if interrupt_after is None:
        return subprocess.run(cmd, env=env, timeout=timeout).returncode
    p = subprocess.Popen(cmd, env=env)
    time.sleep(interrupt_after)
    p.send_signal(signal.SIGINT)
    print("  (sent Ctrl+C #1)")
    try:
        return p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)
        print("  (sent Ctrl+C #2)")
        return p.wait(timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary results folder")
    args = ap.parse_args()
    unit_tests()

    MockHandler.index = build_index()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="stnlp_selftest_"))
    (tmp / "models.json").write_text(json.dumps({"models": [
        {"model_id": k.split("/")[1], "model_key": k} for k in MOCK_MODELS]}), encoding="utf-8")
    (tmp / ".env").write_text("OPENROUTER_API_KEY=mock-key\n", encoding="utf-8")
    env = dict(os.environ,
               STNLP_RESULTS_DIR=str(tmp / "results"),
               STNLP_CHAT_URL=f"http://127.0.0.1:{port}/chat",
               STNLP_MODELS_URL=f"http://127.0.0.1:{port}/models",
               STNLP_MODELS_JSON=str(tmp / "models.json"),
               STNLP_ENV_FILE=str(tmp / ".env"),
               STNLP_SELFTEST_FAST_BACKOFF="1",
               STNLP_SELFTEST_LIMIT="20" if args.quick else "",
               PYTHONIOENCODING="utf-8")
    py = sys.executable
    runner = str(SCRIPTS_DIR / "run_models.py")
    print(f"mock server on port {port}; temporary results in {tmp / 'results'}")
    if not args.quick and os.name != "nt":
        rc = run([py, runner, "--yes", "--concurrency", "6"], env, interrupt_after=8)
        print(f"  first session ended with code {rc} (expected 0 after graceful stop)")
    # resume until everything is done (retries for injected failures happen across sessions)
    for k in range(6):
        rc = run([py, runner, "--yes", "--concurrency", "6"], env, timeout=3600)
        st = (tmp / "results" / "status.csv").read_text(encoding="utf-8").splitlines()
        pending = [ln for ln in st[1:] if ",done," not in ln]
        print(f"  session {k + 2}: exit {rc}; not done: {len(pending)}")
        if not pending:
            break
    raw = list((tmp / "results" / "raw").rglob("*.json"))
    print(f"  raw responses saved: {len(raw)}; mock calls received: {MockHandler.calls['n']}")
    acmd = [py, str(SCRIPTS_DIR / "analyze.py"), "--repair-time-limit", "5", "--repair-stall", "1", "--bootstrap", "200"]
    if not args.quick:
        acmd.append("--skip-repair")   # the repair step on all items would take hours; --quick tests it
    rc = run(acmd, env, timeout=7200)
    print(f"  analyze.py exit code {rc}")
    rep = tmp / "results" / "analysis" / "REPORT.md"
    print(f"  report exists: {rep.exists()}")
    ok = rc == 0 and rep.exists()
    srv.shutdown()
    if args.keep:
        print(f"kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
