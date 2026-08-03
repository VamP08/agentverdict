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
| M3 | Calibration lab: judge-vs-human agreement (kappa), position/verbosity/self-preference bias stats | planned |
| M4 | Distilled 4B judge: fine-tune on accumulated judgments, serve locally, publish to HF Hub | planned |
| M5 | CI integration: GitHub Action, cost-bounded suites, bootstrapped confidence intervals, merge gating | planned |
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

**Part 2 — task replay (next):** an agent adapter interface that executes an agent against
stored tasks (starting with a built-in Groq tool-calling reference agent using each task's
`tools_spec` with mocked tools) and records fresh trajectories, plus `agentverdict eval`
combining replay + judging.

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

- **tasks** — `id` (hex uuid pk), `key` (unique slug), `prompt`, `tools_spec` (JSON list),
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
