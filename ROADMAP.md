# Roadmap

Six milestones, each shippable on its own. Statuses mirror [DESIGN.md](DESIGN.md), which is the
authoritative spec.

## M1 — Golden-dataset workbench (done)

Store tasks, recorded agent trajectories, and human labels; build golden datasets with zero LLM
calls.

- [x] SQLAlchemy data model for tasks, trajectories, steps, and human labels, on SQLite or
      Postgres, with `init-db` table creation
- [x] JSONL bundle import/export (task upsert by key, line-tolerant import with an error
      report) plus realistic sample data in `examples/`
- [x] Server-rendered labeling UI at `/label`: queue of unlabeled trajectories, step-by-step
      trajectory view, verdict form, auto-advance to the next unlabeled run
- [x] REST API under `/api` for tasks, trajectories, and labels, with one-label-per-annotator
      replace semantics
- [x] CLI (`init-db`, `import`, `export`, `stats`, `serve`), pytest suite on in-memory SQLite,
      ruff-clean, CI workflow, docker-compose for Postgres

## M2 — Eval runner (done)

Replay tasks against agents and score the results with an LLM judge.

- [x] LLM-as-judge scoring via Groq with structured verdicts persisted alongside trajectories
- [x] Judge, eval-run, and judge-verdict tables added to the data model
- [x] `agentverdict judge` CLI (add / list / run) printing a scored summary with a naive
      human-agreement readout; judge verdicts surfaced in the labeling UI and API
- [x] Task replay harness that executes an agent against stored tasks and records new
      trajectories automatically
- [x] `agentverdict eval` CLI combining replay + judging into one end-to-end suite run

## M3 — Calibration lab (in progress)

Measure the judge against the humans before trusting it.

- [x] Judge-vs-human agreement report: Cohen's kappa and per-verdict confusion matrix over the
      golden dataset
- [x] Per-annotator comparison (the judge scored against each rater on exactly the items both
      rated, so judge and human numbers sit on the same sample) and `--annotator` scoping, so a
      re-labeled round is reported on its own while the first round is kept as-is
- [ ] Position-bias analysis (does verdict flip when candidate order is swapped?)
- [ ] Verbosity-bias analysis (does the judge favor longer outputs at equal quality?)
- [ ] Self-preference analysis (does the judge favor outputs from its own model family?)
- [ ] Rubric versioning activated, with calibration reports tied to a rubric version

## M4 — Distilled judge (planned)

A small local judge, cheap enough to run on every PR.

- [ ] Training-set builder that exports accumulated judgments in a fine-tuning format
- [ ] Fine-tuned ~4B open model that matches the frontier judge within a stated kappa margin on
      a held-out golden set
- [ ] Local serving path so evals run without external API calls
- [ ] Model card and weights published to the Hugging Face Hub

## M5 — CI integration (in progress)

Merge gating on statistically sound evidence.

- [x] GitHub Action that runs an eval suite on pull requests and posts a results summary
- [ ] Cost-bounded suites: hard token/latency budgets per run
- [x] Bootstrapped confidence intervals on score deltas between base and PR
- [x] Merge gate that blocks only on statistically significant regressions, with a clear
      pass/fail annotation on the PR

## M6 — Leaderboard and online loop (planned)

Close the loop from production back into the golden dataset.

- [ ] Public leaderboard comparing agent configurations on shared eval suites
- [ ] Langfuse integration that pulls production traces into the trajectory store
- [ ] Triage flow that promotes interesting production traces into new labeled eval cases
- [ ] Drift monitoring between offline eval scores and online outcomes
