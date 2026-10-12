# How to run the single-run experiment

Everything is run from the project folder `D:\Research\Spatio-Temporal NLP` in a Windows terminal
(Windows Terminal or Command Prompt; maximise the window so the live dashboard fits).

```
D:\Research\Spatio-Temporal NLP
├─ exports\        gold data (read only – never modified by any script)
├─ models.json     model cohort
├─ .env            OPENROUTER_API_KEY=...   (keep private)
├─ scripts\        all code (this folder)
└─ results\        everything the scripts produce (created on the first run)
```

## 1. One-time setup (Python 3.10–3.13)

```bat
cd /d "D:\Research\Spatio-Temporal NLP"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r scripts\requirements.txt
```

If `py` is not found, use `python -m venv .venv`. In every new terminal you only need:
`cd /d "D:\Research\Spatio-Temporal NLP"` and `.venv\Scripts\activate`.

## 2. Free checks (no model calls, no cost)

```bat
python scripts\selftest.py --quick
python scripts\run_models.py --check
python scripts\run_models.py --dry-run
```

- `selftest.py --quick` runs the whole pipeline against a fake local server (≈3–5 min) and must end with `SELF-TEST PASSED`.
- `--check` confirms every model in `models.json` exists on OpenRouter and shows the exact parameters that will be sent.
- `--dry-run` shows the item list and a rough cost estimate.

## 3. The real run (one call per model and item)

```bat
python scripts\run_models.py
```

The first time, it shows the plan and cost estimate and asks you to type `YES`. It then freezes the prompt,
settings and model metadata in `results\run_manifest.json` and writes `results\PREREGISTRATION.md` before
the first call.

**The dashboard shows:**
- per model: progress, parsed answers (complete / partial / failed), truncated answers, retries, rejected requests, running calls, average seconds and cost;
- every running call: model, stem, number of events, phase (`waiting` / `thinking` / `writing`), characters of thinking and of answer, matrix rows written so far, elapsed time;
- the latest events.

**Stopping and resuming**
- **Ctrl+C once:** no new calls start; calls already running finish and are saved, then the program exits.
- **Ctrl+C twice:** stop immediately. Running calls are abandoned and redone next time.
- **Resume:** run the same command again (`python scripts\run_models.py`). Finished calls are never repeated.
  Calls that failed (network or provider errors) or were abandoned are started again.
- Each finished call is saved the moment it ends, so a crash or power cut loses at most the calls that were running.

**What it does on errors:** a network or provider failure is retried up to 4 times per session, with waits of 10, 30 and 90 s.
After that the item is left for the next start. A call is never repeated because the answer looks wrong.
An invalid key (401) or no credits (402) stops the run cleanly.

**Parallel workers.** All models always run at the same time. `--workers` sets how many calls run in parallel
*per model* (total = workers × number of models):

- `python scripts\run_models.py --workers`: 8 per model (48 calls at once for 6 models)
- `python scripts\run_models.py --workers 12`: 12 per model
- `python scripts\run_models.py`: the default, 4 per model

The number of workers can differ between sessions (stop with Ctrl+C, restart with another value); it does not
change what is asked, so it is not part of the frozen settings. If a model hits OpenRouter's rate limit (HTTP 429),
only that model's workers pause briefly, and the "429s" column on the dashboard counts it. These pauses are not
counted as failures. If 429s keep appearing, restart with fewer workers.

`--models gemini qwen` runs only some models this session.

**Do not edit** `scripts\common.py` (the prompt and parser), the `FROZEN_SETTINGS` in `run_models.py`, `exports\`
or the existing entries in `models.json` once the run has started. The runner detects such changes and refuses to
continue rather than mixing two different experiments. Adding a new model to `models.json` is allowed: it is
added to the manifest and run with the same frozen prompt.

## 4. Analysis (offline; no model calls)

```bat
python scripts\analyze.py
```

- All analyses A1–A14 from `experiment_plans_single_run.md`, plus robustness subsets, take a few minutes.
  Figures and `results\analysis\REPORT.md` are produced at the end.
- **A3a (repair to the nearest logically consistent network) is slow:** it can take 1–4 hours depending on your CPU.
  It runs last, shows its own progress bar and ETA, and saves every solved network. Ctrl+C then the same command
  resumes it. To get everything else first: `python scripts\analyze.py --skip-repair`, and later run
  `python scripts\analyze.py` again.
- Finished steps are skipped on the next start. `--force` redoes everything; `--force --only A9 REPORT` redoes only those steps.

**Optional manual inputs** (created in `results\analysis_inputs\`, never overwritten):

| File | What to fill in | Then run |
|---|---|---|
| `A9_gold_error_audit_sample.csv` | column `is_gold_error` (yes / no / unsure) | `--force --only A9 REPORT` |
| `A13_arithmetic_tagging_sample.csv` | column `requires_arithmetic` (yes / no) | `--force --only A13 REPORT` |
| `stem_types.csv` | column `type` for each stem | `--force --only A14 REPORT` |
| `model_attributes.csv` | open weights, parameter count | `--force --only A11 REPORT` |

## 5. Where everything is saved

| Path | Contents |
|---|---|
| `results\run_manifest.json` | frozen prompt hash, data hash, settings, per-model request parameters, item list |
| `results\PREREGISTRATION.md` | hypotheses and primary metric, written before the first call |
| `results\prompts\<item>.txt` | the exact prompt sent for each item (identical for all models) |
| `results\raw\<model>\<item>.json` | full response: answer, reasoning text, finish reason, usage, cost, provider, timings |
| `results\parsed\<model>\<item>.csv` | parsed matrix (from_event, to_event, symbol, code) + `.parse.json` summary |
| `results\errors\<model>\` | one file per failed or abandoned attempt |
| `results\status.csv` | one row per model × item (state, parse status, tokens, cost, latency) |
| `results\logs\run_*.log`, `analysis_*.log` | full terminal log of every session |
| `results\logs\dashboard_latest.txt` | plain-text copy of the dashboard, refreshed every 5 s |
| `results\smoke\` | the smoke test (made-up 3-event story, not from the data) |
| `results\analysis\tables\*.csv` | every number in the report |
| `results\analysis\figures\*.png` | figures |
| `results\analysis\REPORT.md` | the generated report |

## 6. Script overview

| File | Role |
|---|---|
| `run_models.py` | the model run: dashboard, saving, retries, Ctrl+C handling, resume |
| `analyze.py` | analyses A1–A14, robustness checks, A3a repair, figures, report |
| `common.py` | paths, gold loading, **the frozen prompt**, output parser, crash-safe writing |
| `allen.py` | Allen algebra: composition table, consistency check, nearest-consistent repair |
| `analysis_lib.py` | pair table, features, metrics, bootstrap, kappa |
| `figures.py`, `report.py` | figures and REPORT.md |
| `selftest.py` | offline end-to-end test with a fake OpenRouter server |
