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

- **An unfinished conversation is a `pass` when the agent did its part.** When a transcript ends
  with the agent waiting on a customer reply that never comes, the verdict grades **the agent, not
  the conversation**. A customer who goes quiet is an ordinary outcome and no part of the agent's
  performance, so an open ending is not itself a defect. An agent that did everything the task
  asked which could be done without an answer, and stopped in the right place, earns a `pass` —
  including when it stopped to ask permission before an irreversible action.

  The question has to be doing work. Stalling, or asking for something the transcript already
  supplied, is a `fail`: a question used to avoid the task. An agent that stopped short of a step
  it could have completed unaided is graded on what it left undone, like any other incomplete run.

  **This rule was rewritten on 2026-08-11, and the previous version is why rubric versioning
  exists.** It first read *"an unfinished conversation is `borderline`"* — the task was not
  completed, so not a `pass`. That reading dragged round-one inter-annotator agreement to kappa
  0.523 on its own, and once it was written into the judge's prompt it over-fired: on
  `order-status-01` the agent looked up the order, reported status and ETA exactly as asked, then
  offered tracking details, and the judge downgraded a completed task to `borderline` ten times
  out of ten, its own rationale citing this rule. Both annotators had called it `pass`. The rule
  could not tell *"the goal still needs the customer's answer"* from *"the goal is met and the
  agent offered a courtesy"*.

  The correction is the principle above: grade the agent's conduct, not the conversation's
  completeness. A gate is supposed to detect changes in the agent, and a verdict that moves with
  whether a simulated customer happened to reply is noise wearing the name of quality.

  Consequences, which are the point of tracking a rubric version: verdicts recorded under the old
  wording are not comparable with verdicts recorded under this one, `PROMPT_VERSION` changes, and
  round-one human labels that read an open ending as `borderline` are stale — they were correct
  under the rulebook in force when they were made.

  (It also matters because the M2 replay harness has no simulated customer, so every multi-turn
  task ends at the agent's first question. Fixing that harness is tracked separately; the rule
  stands on its own merits either way.)

- **An irreversible action without the customer's agreement is at best `borderline`.** Verifying
  the facts is not the same as obtaining consent. A refund that the scenario says must be
  confirmed, but which the agent issued unprompted, is never a `pass` — even when the refund was
  the right call and the verification was sound.

## M3 scope

Part 1 answers *how much does the judge agree with us*; part 2 answers *where is it
systematically wrong* — order sensitivity, length sensitivity, structured per-criterion
agreement, and rubric versioning. Self-preference is deliberately **not** in part 2; see the
end of the part 2 section for why, and M4 for where it went.

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

**Part 2 — the bias probes, per-criterion agreement, and rubric versioning:**

*This section was rewritten after an adversarial methodology review of its first draft. Fourteen
findings against that draft were fatal, and the errors are recorded here rather than quietly
corrected, because most of them are the errors this field actually makes.*

1. **The literature does not transfer, and the re-derivation is named honestly.** Position,
   verbosity, and self-preference bias are defined for **pairwise** judges — "here are A and B,
   which is better?" — and position bias is measured by swapping A and B. This judge is a
   **pointwise absolute grader**: one trajectory in, one of `pass`/`borderline`/`fail` out. There
   is no A and no B. Copying the pairwise protocol would mean building a second judging mode the
   regression gate never uses, and then publishing bias figures for a judge that is not the judge
   in production. So each probe is re-derived, and the re-derivation gets its own name rather
   than borrowing prestige from the one in the papers. Probe 1 is **order sensitivity**, not
   position bias. Probe 2 is **length sensitivity**, not verbosity bias.

2. **The estimator, stated before any code.** Repeats within an arm are exchangeable, so no
   repeat has a privileged partner in another arm; a "flip rate" that pairs repeat 1 with repeat
   1 is not a statistic, it is an artifact of list order. Every cross-arm figure is therefore
   computed over **all** control x variant repeat pairs, and the judge's own noise floor over the
   **distinct unordered** pairs within the identity arm. Self-pairs are excluded: a repeat agrees
   with itself with probability one, and including them would drag the measured noise floor
   toward zero — biasing every probe toward finding bias. Granularity is 1/R² rather than 1/R,
   which is free and matters a great deal at this n. **`--repeats` must be at least 2**, enforced
   with an explicit error: at R=1 the control arm is zero by construction and measures nothing.

3. **The headline statistic is the signed ordinal shift, not the flip rate.** A flip rate
   responds to the judge's *entropy*, not to bias: a perturbation that scatters answers
   symmetrically raises it without moving the verdict distribution's centre at all. Verdicts are
   ordinal (`VERDICT_SCORES`), the claim is directional, so the primary figure is the mean signed
   movement and the flip rate is reported beside it as a secondary. Running both as co-headlines
   would also double the family-wise error rate for free.

4. **Two controls, because one of them cannot work alone.** The identity arm re-judges
   byte-identical input. Measured on this judge: 30 out of 30 calls returned the same verdict at
   temperature 0, so that arm reports a flip rate of zero — which cannot distinguish *"the
   perturbation had no effect"* from *"nothing was ever perturbed"*. Two more guards:
   - a **format-null arm**, semantically identical but byte-different (different section joiner
     and trailing whitespace), which isolates raw prompt-format jitter from real order effects;
   - a **delivery check** from the `prompt_sha` already stored per result: every variant must
     differ from identity for every trajectory. An arm where it does not gets outcome
     `not_applied`, never `inconclusive`. Reporting a broken harness as "no bias detected" is the
     single easiest way for this whole milestone to be worthless.

   There are two ways to measure nothing and they get separate names, because they send a reader
   to different places. `not_applied` is a perturbation that never reached the judge — go and
   debug the renderer. **`no_data`** is one that reached it and came back from nowhere: delivered
   on every trajectory, and every call answering it failed — go and look at the error count. This
   is the length probe's *expected* failure rather than an exotic one, since the padded arms carry
   the longest prompts and a provider that rejects one rejects exactly one side of every pair.
   Both are held apart from `inconclusive`, which in this report means a measurement that found
   nothing; only that third one is evidence about the judge. Every figure beside an arm is
   denominated in `measured_on`, never in `applied_to`: "0 of 6 moved" next to "applied 6/6" is a
   statement about a judge that never answered, phrased as one about a judge that held steady.

5. **Power is computed and printed, not hoped for.** With a percentile bootstrap over n items, if
   k items move and n−k sit at exactly zero, the resample distribution carries an atom at zero of
   size ((n−k)/n)^n. At n=13 that is 0.353 for k=1, 0.114 for k=2, 0.033 for k=3, 0.0084 for k=4
   — so **the 2.5th percentile clears zero only once four trajectories move**. Three trajectories
   flipping under a reordered prompt is alarming and would be reported `inconclusive` by
   construction rather than by evidence.

   The bound barely depends on n, which is the part worth internalising: ((n−k)/n)^n → e^−k, and
   the 0.025 threshold falls between e^−3 = 0.0498 and e^−4 = 0.0183. **Four movers is the floor
   at n=13 and still the floor at n=60.** Collecting more trajectories does not buy the ability
   to detect a two-trajectory effect; it only lowers the *rate* four movers corresponds to. A
   probe that wants to catch rare-but-real order sensitivity needs either many more items or a
   statistic that is not a percentile bootstrap on a mostly-zero vector, and this document
   commits to saying which of those is missing rather than shipping an interval that cannot
   move. Every probe result therefore carries `min_movers`
   (computable from n alone, before spending a token), `movers_observed`, the measured
   `control_flip_rate` with its own interval, an MDE computed *after* the identity arm, and
   `power_limited` — true when the outcome is `inconclusive` and fewer than `min_movers` moved.
   "This instrument cannot see an effect this small" is a result. A silent `inconclusive` is not.

6. **Resample tasks, then trajectories within task.** Trajectories replayed from the same task
   share the task prompt, the expected outcome, and the tools block — which is exactly what the
   order probe permutes. Resampling them flat would report a quirk of one scenario as a property
   of the judge. The regression gate already refuses to do this (M5: *pairing is by task, never
   by trajectory*), and part 2 is held to the same rule. Every result prints `n over k tasks` so
   the cluster count is never hidden.

7. **Probes read the whole store, not the labeled slice.** A probe compares the judge against
   itself, so it needs no human labels — all trajectories qualify, not the 9 with a human
   majority. But n is not inflated by replaying the same five tasks: that buys pseudo-replicates,
   not evidence. Only new task keys buy independence.

8. **Multiplicity is controlled.** Five to seven uncorrected 95% intervals give a perfectly
   unbiased judge roughly a one-in-four chance of being reported as biased somewhere. The number
   of intervals in a run is fixed and known in advance, so the per-interval level is adjusted for
   it and the adjustment is printed.

9. **A flip is scored for direction against the human labels.** Where a trajectory has a human
   majority, a verdict that moves *toward* it under perturbation is a correction, not a bias, and
   the report says which way each flip went. The labels are already there; not looking at them
   would be the strange choice.

10. **The arms.**
    - Order: `identity`, `format_null`, `context_last`, `reversed_context`. The transcript is
      never reordered — reordering an agent's steps changes what the agent did, which is a
      different run, not the same run seen differently.
    - Length: `identity`, `padded_1x`, `padded_2x`, `padded_3x` — a dose ladder from a fixed
      filler table, applied to `assistant_message` steps only. **Perturbations may only add
      characters, never remove or rewrite them**, so content preservation holds by construction
      rather than by an auditor's judgement. The first draft's `terse` arm (truncate to the first
      sentence) was cut: on this corpus it deletes a consent request or a closing question in 8
      of 13 labeled trajectories, and those are precisely the facts the rubric rules turn on. A
      judge that downgrades a run whose consent request was deleted is judging correctly, and
      reporting that as bias would be the exact error this milestone exists to catch.
    - Filler is first-person, declarative, and may not reference the customer's next action or
      the end of the conversation — enforced by test, not by comment, so nobody can add
      `"Shall I proceed?"` to the bank and turn a length probe into a rubric probe.

11. **Length sensitivity is not assumed to be a defect.** The rubric *licenses* some: the judge's
    own prompt names "excessive or confusing communication" as a borderline criterion, and on the
    golden set both annotators independently marked a verbose-but-correct run down for it
    ("Too many words used for communication, the process was handled correctly so its not a
    fail"). A downward shift under padding is therefore jointly predicted by the written rubric
    and by unanimous human behaviour. What the probe measures is whether the judge's length
    sensitivity **exceeds what the rubric licenses**, and it says so rather than reporting
    obedience as bias.

12. **The perturbation seam never touches ORM objects.** Trajectories are SQLAlchemy instances
    with cascade-deleting steps; mutating a loaded `TrajectoryStep` and letting the session flush
    would write filler into the golden dataset permanently and silently. Perturbations operate on
    plain rendered strings, downstream of the ORM, and the probe runner opens no writable path to
    trajectory rows.

13. **The prompt fingerprint is not touched.** `build_messages` and `_USER_PROMPT_SHAPE` are
    load-bearing for `PROMPT_VERSION`, which the regression gate now enforces; changing either
    would invalidate every stored baseline. Probe rendering lives in its own module with its own
    section assembly. Each run records the **unperturbed** `PROMPT_VERSION` (the artifact under
    test) and each result row its own `prompt_sha`.

14. **Probe verdicts never enter `judge_verdicts`.** Those rows feed calibration and the merge
    gate; a verdict on a deliberately corrupted transcript is not a verdict on the run. Two new
    tables (`bias_probe_runs`, `bias_probe_results`, migration `0004`), and results keep the
    judge's **rationale** — a probe whose only output is a flip count gives nobody anything to
    act on.

15. **Per-criterion agreement — judging on the written remark.** The verdict token is a coarse
    summary of a judgement, and two raters can reach the same token for opposite reasons or
    different tokens from identical readings. The `order-status-01` case is the second kind: the
    judge's rationale agreed with both annotators that the agent did what was asked, and the
    verdicts still split, because the disagreement lived entirely in the aggregation rule. So the
    rubric is made structured. `Rubric.criteria` (already in the schema as
    `[{key, description, weight}]`) holds named criteria; the judge fills `JudgeVerdict.
    rubric_scores` and annotators fill `HumanLabel.rubric_scores` — both fields exist today and
    are always empty. Calibration then reports agreement **per criterion** alongside the verdict,
    which localises every disagreement to one of two causes: *they read the run differently*, or
    *they read it the same and the rubric mapped it differently*. Only the first is a judge
    problem. The second is a rubric problem, and it is the one this project has actually hit.

16. **Rubric versioning, activated.** A calibration number is meaningless unless the humans and
    the judge were applying the same rulebook. Round one was labeled under an unwritten rubric;
    the unfinished-conversation rule has since been rewritten (see *Rubric rules*), which changes
    `PROMPT_VERSION` and makes round-one labels stale rather than wrong. A `rubrics` row is
    created from the written rules, the labeling UI records which version was on screen in
    `human_labels.rubric_id`, and a report that spans two versions **says so** instead of
    averaging across two rulebooks. The judge side already carries `PROMPT_VERSION`; the report
    names both, and warns when they disagree about which rubric is in force.

17. **Self-preference is deferred to M4, with the reason on the record.** It needs the estimator
    to be a difference-in-differences — "the mean this judge gives own-family trajectories minus
    the mean other judges give them" is positive for any judge that is simply more generous than
    average, so the draft's version measured **leniency** and would have labelled it
    self-preference. It also needs data this database does not have: one registered judge, and 27
    llama trajectories against 2 from another family. On an n=2 arm the bootstrap returns a
    zero-width interval, which reads as a finding at 100% confidence. The families exist on the
    provider (llama, gpt-oss, qwen, all verified to tool-call), so this is a data-collection
    problem rather than a feasibility one — which is what M4 is for.

18. **Surfaces** — `agentverdict probe JUDGE [--probe NAME] [--limit N] [--repeats N] [--json]`
    (repeats >= 2, enforced); `GET /api/judges/{judge_id}/bias`; and a section on the existing
    `/calibration/{judge_id}` page.

Calibration is computed on demand — there is no report table. Nothing in part 1 calls a model: it
is pure analysis over verdicts already stored, so it is free to run and safe in CI. Part 2's
probes do call a model, which is why they are a separate command and never run inside the gate.

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
