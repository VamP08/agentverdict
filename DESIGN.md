# AgentVerdict — Design (source of truth)

AgentVerdict is a CI-native evaluation platform for tool-calling AI agents. It treats the LLM
judge itself as a measured ML artifact: golden datasets of agent trajectories are labeled by
humans, LLM judges are calibrated against those labels (Cohen's kappa, bias analysis), a small
open model is distilled to match the frontier judge, and a GitHub Action gates merges on
statistically sound regression evals.

This document is the authoritative spec. If code and this document disagree, this document wins;
update it deliberately when a decision changes.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Golden-dataset workbench: tasks, trajectories, human labels; JSONL import/export; browser labeling UI | done |
| M2 | Eval runner: replay tasks against agents, LLM-as-judge scoring via Groq | **in progress** |
| M3 | Calibration lab: judge-vs-human agreement (kappa), position/verbosity/self-preference bias stats | **in progress** |
| M4 | Distilled 4B judge: fine-tune on accumulated judgments, serve locally, publish to HF Hub | planned |
| M5 | CI integration: GitHub Action, cost-bounded suites, bootstrapped confidence intervals, merge gating | **in progress** |
| M6 | Public leaderboard + Langfuse online loop (production traces → new eval cases) | planned |

## M1 scope

Everything needed to build and label a golden dataset, with zero LLM calls:

1. Store **tasks** (scenarios), **trajectories** (recorded multi-step agent runs), and **human labels**.
2. Import/export trajectories as JSONL bundles; ship realistic sample data.
3. A minimal server-rendered **labeling UI** (list → view trajectory → submit label → next).
4. CLI for init/import/export/stats/serve.
5. Tests (SQLite in-memory, no network), ruff-clean, CI workflow, docker-compose for Postgres.

Explicitly deferred: auth, Alembic migrations (M1 uses `Base.metadata.create_all`), pgvector and
embeddings (M2+), judge tables (M2), async SQLAlchemy (sync is fine at this scale).

## M2 scope

Two halves; the judging half ships first, the replay half second.

**Part 1 — LLM-as-judge (shipped):**

1. **Tables** — `judges` (a named model + prompting recipe), `eval_runs` (one batch of judging),
   `judge_verdicts` (one decision or recorded failure per trajectory per run). A judge may score
   the same trajectory across multiple runs — repeat sampling feeds M3 consistency analysis.
2. **Groq client** (`judging/client.py`) — hand-rolled `httpx` client against Groq's
   OpenAI-compatible `/chat/completions` with JSON response format, bounded retries on 429/5xx
   (honoring `Retry-After`), latency and token-usage capture. `httpx` is therefore promoted from
   a dev-only to a runtime dependency (deliberate change, recorded here).
3. **Prompting** (`judging/prompts.py`) — renders the task (prompt, expected outcome, tools) and
   the full step transcript into judge messages; the judge must answer with strict JSON
   `{"verdict": "pass|fail|borderline", "rationale": str, "rubric_scores": {}}`, validated by the
   `JudgeDecision` schema, with one corrective retry on malformed output.
4. **Runner** (`judging/runner.py`) — iterates trajectories (optional task filter/limit),
   persists a `JudgeVerdict` per trajectory (error rows on failure — one bad call never aborts a
   run), aggregates verdict counts and token totals onto the `EvalRun`, and computes a naive
   human-agreement figure (judge vs. per-trajectory human majority) stored in `EvalRun.meta`.
   Cohen's kappa and bias analysis stay in M3.
5. **Surfaces** — CLI `agentverdict judge add|list|run`; `GET /api/judges` and
   `GET /api/trajectories/{id}/verdicts`; judge verdicts rendered on the labeling UI trajectory
   page (humans label first, then compare — the UI shows judge verdicts below the label form).
6. **Config** — `AGENTVERDICT_GROQ_API_KEY` (falls back to plain `GROQ_API_KEY`),
   `AGENTVERDICT_JUDGE_MODEL` (default `llama-3.3-70b-versatile`), base URL and timeout.
   Tests never hit the network: the client accepts an injected `httpx` transport.

**Part 2 — task replay (in progress):**

1. **Agent adapter contract** (`agents/base.py`, frozen) — `AgentAdapter` is a Protocol with a
   `name` and `run(task) -> AgentRunResult`; the result carries `steps` (list[StepIn]), `status`,
   `agent_config`, `meta`, timestamps, and token counts. Adapters never touch the database.
   `opening_message(task)` returns the first user turn.
2. **Task `user_message`** (new nullable column) — `Task.prompt` describes the scenario for
   graders and states what the agent *should* do, so replaying against it leaks the answer.
   Tasks therefore carry a first-person customer opening line; `opening_message()` falls back to
   `prompt` when it is absent. The JSONL bundle format gains an optional `task.user_message`.
3. **Mock tools** (`agents/mock_tools.py` + `examples/mock_tools.json`) — a fixture-driven
   registry so new scenarios need data, not code. Fixture shape:
   ```json
   {"tools": {"lookup_order": {
      "rules": [{"when": {"order_id": "NW-1001"},
                 "result": {"status": "delivered"}, "is_error": false}],
      "default": {"result": {"error": "Order not found"}, "is_error": true}}}}
   ```
   First rule whose `when` entries all equal the call's arguments (compared as trimmed strings)
   wins; otherwise `default`; otherwise a generic `is_error` result naming the unknown tool.
   Matching is deterministic — no randomness, so replays are reproducible.
4. **Groq reference agent** (`agents/groq_agent.py`) — a tool-calling loop over `chat`: send the
   system persona + opening message with the task's `tools_spec` wrapped as
   `{"type": "function", "function": spec}`; on `tool_calls`, record one `tool_call` step per
   call, execute each through the registry, record `tool_result` steps, append the assistant and
   `role: "tool"` messages, and iterate up to `max_turns` (default 8). A final text answer ends
   the run (`completed`); exhausting the budget yields `truncated`; a client error or malformed
   tool arguments records what happened and yields `error`. The system prompt is a generic
   support persona and must never mention the expected outcome.
5. **Replay runner** (`agents/replay.py`) — `run_replay(session, adapter, *, task_key, limit,
   repeats, on_progress) -> ReplayReport` iterates tasks (optionally one key), calls the adapter
   `repeats` times per task, persists each result as a `Trajectory` with `source="replay"` and
   the adapter's `agent_config`, and returns created ids plus per-task errors. One failing task
   never aborts the sweep.
6. **CLI** — `agentverdict replay [--adapter --task-key --limit --repeats --model]` records new
   trajectories; `agentverdict eval --judge NAME [...]` replays and then judges exactly the
   trajectories it just created (via `run_eval(trajectory_ids=...)`), printing a combined
   summary. `--skip-replay` judges existing trajectories instead.
7. **Migrations** (Alembic, `migrations/`) — `tasks.user_message` is the first schema change
   after M1, and `create_all` cannot add columns to existing tables. `agentverdict init-db` and
   `serve` therefore run `upgrade head`; `db.init_db()` (create_all) remains the fast path for
   tests. A test asserts `compare_metadata` finds no drift between migrations and the models.

## M5 scope

Part 1 (in progress) is the statistics and the gate; part 2 wires it into a packaged GitHub
Action with cost ceilings.

1. **Scoring a suite** — each verdict is worth `fail 0.0 / borderline 0.5 / pass 1.0`
   (`stats.VERDICT_SCORES`), linear on the ordinal scale. A task's score is the mean over its
   trajectories in that run, so `--repeats` sampling averages instead of double-counting; a run's
   score is the mean over tasks, so a task with more replays does not get more say.
2. **Pairing is by task, never by trajectory.** Replay creates fresh trajectories every time, so
   the two runs share no trajectory ids. Tasks present in only one run are excluded from the
   statistics and reported separately — silently dropping them would let a suite change look like
   a quality change.
3. **The test** (`gating/compare.py`) — take the per-task deltas, bootstrap their mean by
   resampling **tasks** (`stats.bootstrap_mean_ci`), and read the interval:
   entirely below zero → `regression`; entirely above → `improvement`; otherwise
   `inconclusive`. Only `regression` sets `blocks_merge`. A suite too small to resample is
   `inconclusive` by construction, which is the correct default: a gate that blocks on noise gets
   switched off within a week, and a gate nobody trusts is worse than no gate.
4. **The yardstick has to hold still.** The comparison refuses two runs from different judges,
   and equally two runs graded by different versions of the judge prompt — `prompts.PROMPT_VERSION`,
   a hash of the prompt text, stamped onto every `EvalRun`. Same judge, edited rubric is the same
   error wearing the judge's name, and it arrives on exactly the pull request that edits the
   rubric. Nullable, never backfilled: a run recorded before the column existed has an unknown
   rubric, and inventing one would be a lie the gate then acts on.
5. **Surfaces** — `agentverdict compare BASE CANDIDATE [--fail-on-regression --json]`;
   `GET /api/comparisons?base=&candidate=`; and a workflow that replays, judges, and compares
   against a stored baseline, posting the summary to the pull request.

   Three exit codes, because CI has to tell them apart: **1** a regression was measured and the
   merge should stop, **2** the comparison could not be made at all (unresolvable id, judge or
   rubric mismatch), **0** everything else. Collapsing 2 into 1 makes the gate report a
   configuration fault as a quality fault, in a message the contributor cannot act on. The
   workflow branches on all three, and separately on `tasks_compared`, so a green check that
   compared nothing does not read as a clean bill of health.

   The candidate run id comes from `eval --run-id-file`, not from "the newest run for this
   judge": on the shared database this design recommends, two concurrent pull requests would each
   gate the other's run.

Judge quality bounds all of this: a gate is only as trustworthy as the judge behind it, which is
why calibration (M3) comes first and why the comparison records the judge it used.

## Rubric rules

Decisions the verdict vocabulary alone does not settle, recorded here because judge and human
must apply the *same* rule or the measured disagreement is rubric noise rather than judge error.
Every rule here is also stated in the judge's system prompt (`judging/prompts.py`).

- **An unfinished conversation is `borderline`.** When a transcript ends with the agent waiting on
  a customer reply that never comes: the agent is not at fault, so it is not a `fail`; the task
  was not completed, so it is not a `pass`. The verdict grades what the run *achieved*, not who
  is to blame. Asking for confirmation before an irreversible action and then stopping is correct
  conduct that nonetheless leaves the job unfinished. Stalling, or asking for something the
  transcript already supplied, is a `fail` rather than a `borderline`.

  This rule was not obvious: in the first annotation round one annotator read it as `pass`
  (correct conduct) and another as `borderline` (incomplete outcome), which produced all three of
  the replay disagreements and dragged inter-annotator agreement to kappa 0.523. It is written
  down here, and in the judge's prompt, precisely so the two never diverge again.

  (It also matters because the M2 replay harness has no simulated customer, so every multi-turn
  task ends at the agent's first question. Fixing that harness is tracked separately; the rule
  stands on its own merits either way.)

- **An irreversible action without the customer's agreement is at best `borderline`.** Verifying
  the facts is not the same as obtaining consent. A refund that the scenario says must be
  confirmed, but which the agent issued unprompted, is never a `pass` — even when the refund was
  the right call and the verification was sound.

## M3 scope

Part 1 (in progress) answers *how much does the judge agree with us*; part 2 will answer
*where is it systematically wrong* (position, verbosity, and self-preference probes, plus
rubric versioning).

**Part 1 — the agreement core:**

1. **Statistics** (`calibration/stats.py`, frozen) — pure standard library. `agreement_stats`
   returns the confusion matrix, observed agreement, Cohen's kappa, an ordinal linear-weighted
   kappa (`fail < borderline < pass`, so a pass/fail mix-up costs more than pass/borderline), a
   seeded percentile-bootstrap CI over trajectory resamples, and per-verdict
   precision/recall/F1. Every statistic returns `None` rather than a misleading number when
   undefined (no items, a rater who never varies).
2. **Report builder** (`calibration/report.py`) — `build_calibration_report(session, judge, *,
   eval_run_id=None, task_key=None) -> CalibrationReport`. Pairs each trajectory's **human
   majority verdict** with that judge's **most recent** verdict for it (scoped to one eval run
   when given). Rows where the judge errored are excluded and counted separately; labeled
   trajectories with no human majority (a tie) are excluded and counted as `ties_excluded`.
3. **Human agreement, two ways** — the judge is scored against the human *majority*, and a
   majority is quieter than any individual annotator, so raw annotator-vs-annotator kappa is not
   the bar it appears to be: a judge with exactly average human accuracy beats it and looks
   superhuman. The report therefore leads with a **held-out baseline** (each annotator scored
   against the majority of the others — the same treatment the judge gets) and shows pairwise
   agreement underneath as a rubric-health signal. The baseline is scoped to judged
   trajectories, deliberately *not* to `compared`: with two annotators every disagreement is a
   tie and is dropped from `compared`, which would score humans only where they already agreed
   and always report a perfect 1.0.
4. **Disagreement drill-down** — the trajectories where judge and humans differ, worst first by
   ordinal distance (pass vs fail before pass vs borderline), each carrying the judge's
   rationale so a human can see *why* it went wrong. This is the actionable output.
5. **Surfaces** — `agentverdict calibrate JUDGE [--eval-run --task-key --json]`;
   `GET /api/judges/{judge_id}/calibration`; a `/calibration` web page (index of judges with
   their kappa) and `/calibration/{judge_id}` (matrix, ceiling, drill-down linking to each
   trajectory).

Calibration is computed on demand — there is no report table. Nothing here calls a model: it is
pure analysis over verdicts already stored, so it is free to run and safe in CI.

**Empty-state matters.** Until trajectories carry both a human label and a judge verdict, every
statistic is undefined. The report says so explicitly and names the two commands that fix it,
rather than rendering zeros that look like findings.

## Repository layout

```
agentverdict/
├── pyproject.toml            # deps are FIXED here; components must not add dependencies
├── DESIGN.md                 # this file
├── README.md, ROADMAP.md, LICENSE (MIT)
├── docker-compose.yml        # Postgres (pgvector/pgvector:pg16) for prod-like runs
├── Dockerfile
├── .env.example
├── .github/workflows/ci.yml  # ruff + pytest on push/PR
├── src/agentverdict/
│   ├── __init__.py
│   ├── config.py             # pydantic-settings; AGENTVERDICT_* env vars
│   ├── db.py                 # engine/session management, init_db(), get_session dependency
│   ├── models.py             # SQLAlchemy 2.0 models (frozen core)
│   ├── schemas.py            # Pydantic v2 schemas (frozen core)
│   ├── importer.py           # JSONL import/export + stats
│   ├── cli.py                # Typer app: init-db, import, export, stats, serve, judge
│   ├── judging/
│   │   ├── __init__.py
│   │   ├── client.py         # Groq chat client (httpx, retries, token usage)
│   │   ├── prompts.py        # transcript rendering + judge message construction
│   │   └── runner.py         # eval-run orchestration + verdict persistence
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── stats.py          # kappa, weighted kappa, bootstrap CI, per-class metrics
│   │   └── report.py         # judge-vs-human report over the golden dataset
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py            # create_app() factory + module-level `app = create_app()`
│   │   ├── routes_tasks.py
│   │   ├── routes_trajectories.py
│   │   ├── routes_labels.py
│   │   └── routes_judges.py
│   └── web/
│       ├── __init__.py
│       ├── routes.py         # exposes `router: APIRouter` (labeling UI)
│       └── templates/        # Jinja2: base.html, queue.html, trajectory.html
├── examples/
│   └── sample_trajectories.jsonl
└── tests/
    ├── conftest.py
    └── test_*.py
```

## Data model (M1 tables)

- **tasks** — `id` (hex uuid pk), `key` (unique slug), `prompt`, `user_message` (nullable text;
  the first-person opening line replay sends to the agent), `tools_spec` (JSON list),
  `expected_outcome` (nullable text), `tags` (JSON list), `created_at`.
- **trajectories** — `id`, `task_id` (FK), `agent_config` (JSON), `source`
  (`api|import|manual|langfuse`), `status` (`completed|error|truncated`), `meta` (JSON),
  `started_at`/`completed_at` (nullable), `created_at`. Steps cascade-delete.
- **trajectory_steps** — `id`, `trajectory_id` (FK), `index` (0-based, unique per trajectory),
  `type` (`user_message|assistant_message|tool_call|tool_result|system`), `content` (JSON).
- **human_labels** — `id`, `trajectory_id` (FK), `annotator`, `verdict`
  (`pass|fail|borderline`), `rubric_id` (nullable FK), `rubric_scores` (JSON dict),
  `rationale` (nullable), `time_spent_s` (nullable float), `created_at`.
  Unique `(trajectory_id, annotator)`; **re-submitting for the same annotator replaces the
  existing label** (update in place).
- **rubrics** — dormant in M1 (no routes); versioned criteria for M3 calibration.
- **judges** — `id`, `name` (unique), `model`, `description` (nullable), `config` (JSON, e.g.
  `{"temperature": 0.0}`), `created_at`.
- **eval_runs** — `id`, `judge_id` (FK), `task_key` (nullable filter), `status`
  (`running|completed|failed`), `trajectory_count`, `error_count`, `verdict_counts` (JSON),
  `input_tokens`/`output_tokens`, `meta` (JSON; holds the naive human-agreement figures),
  `started_at`/`completed_at`, `created_at`.
- **judge_verdicts** — `id`, `eval_run_id` (FK), `judge_id` (FK), `trajectory_id` (FK),
  `verdict` (nullable — null when the call errored), `rubric_scores` (JSON), `rationale`
  (nullable), `error` (nullable), `raw_response` (JSON, nullable), `latency_ms`,
  `input_tokens`/`output_tokens`, `created_at`. No uniqueness across runs by design.

Step `content` conventions:
- `user_message` / `assistant_message` / `system`: `{"text": str}`
- `tool_call`: `{"name": str, "arguments": dict}`
- `tool_result`: `{"name": str, "result": any, "is_error": bool}`

All timestamps are UTC. IDs are `uuid4().hex` strings (portable across SQLite/Postgres).

## JSONL trajectory bundle format

One JSON object per line = one trajectory. The embedded task is **upserted by `key`** (created if
missing; if it exists, the existing task is used and its fields are NOT overwritten).

```json
{"task": {"key": "refund-simple-01", "prompt": "...", "tools_spec": [], "expected_outcome": "...", "tags": ["refunds"]},
 "trajectory": {"agent_config": {"model": "llama-3.3-70b", "prompt_version": "v1"}, "status": "completed",
                "source": "import", "meta": {}, "steps": [
    {"type": "user_message", "content": {"text": "Hi, I want a refund for order A123"}},
    {"type": "tool_call", "content": {"name": "lookup_order", "arguments": {"order_id": "A123"}}},
    {"type": "tool_result", "content": {"name": "lookup_order", "result": {"status": "delivered"}, "is_error": false}},
    {"type": "assistant_message", "content": {"text": "Your refund has been initiated."}}
 ]}}
```

Export writes the same format. Import is line-tolerant: a malformed line is recorded in
`ImportReport.errors` and skipped; valid lines still import.

## HTTP API contract

All JSON routes under `/api`; the labeling UI is served at `/label`. `app.py` exposes
`create_app()` and a module-level `app` (used by `uvicorn agentverdict.api.app:app` and tests).

- `GET  /health` → `{"status": "ok"}`
- `POST /api/tasks` (TaskCreate → TaskRead, 409 on duplicate key)
- `GET  /api/tasks` (list), `GET /api/tasks/{id}` (404 if missing)
- `POST /api/trajectories` (TrajectoryCreate: nested steps; task referenced by `task_id` OR
  `task_key`, 400 with a clear message if neither or both are provided, 404 if the task doesn't
  exist; steps indexed by position) → TrajectoryRead
- `GET  /api/trajectories?task_id=&labeled=` (summaries incl. `label_count`),
  `GET /api/trajectories/{id}` (full, with steps)
- `POST /api/trajectories/{id}/labels` (LabelCreate; replaces same-annotator label) → LabelRead
- `GET  /api/trajectories/{id}/labels`
- `GET  /api/judges` (list) — judges are created via the CLI in M2
- `GET  /api/trajectories/{id}/verdicts` (judge verdicts, newest first; 404 unknown trajectory)

The web router (`agentverdict/web/routes.py`) exposes `router` and is mounted by `create_app()`:
- `GET  /label` — queue: unlabeled trajectories first, then labeled; links to detail
- `GET  /label/{trajectory_id}` — rendered steps (tool calls/results as code blocks) + label form
- `POST /label/{trajectory_id}` — form fields `annotator`, `verdict`, `rationale`; saves label,
  redirects to the next unlabeled trajectory (or back to the queue when none remain)

## CLI contract (`agentverdict ...`)

- `init-db` — create tables
- `import PATH` — import a JSONL bundle, print an ImportReport summary
- `export PATH [--task-key KEY]` — export trajectories to JSONL
- `stats` — counts: tasks, trajectories, labels, label coverage, verdict histogram
- `serve [--host --port --reload]` — run uvicorn on `agentverdict.api.app:app`
- `judge add NAME [--model --description]` — register a judge (model defaults from settings)
- `judge list` — judges with verdict counts
- `judge run NAME [--task-key --limit]` — judge trajectories, print per-trajectory progress and
  a summary (verdict histogram, tokens, error count, naive human agreement when labels exist)

## Conventions

- Python ≥3.11, SQLAlchemy 2.0 style (`Mapped`/`mapped_column`), Pydantic v2.
- Sync SQLAlchemy; sessions come from `agentverdict.db.get_session` (FastAPI dependency that
  commits on success, rolls back on exception).
- Ruff rules `E,F,I,UP,B`, line length 100. Tests run on SQLite in-memory (StaticPool); no
  network, no Docker required for the test suite.
- Dependency set is fixed in pyproject.toml — new deps require a deliberate DESIGN.md change.
