"""Render a task + trajectory into judge messages with a strict JSON answer contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agentverdict.models import Task, Trajectory

SYSTEM_PROMPT = """\
You are an expert evaluator of tool-calling AI agents. You are shown the task an agent was \
given and the full transcript of one recorded run (messages, tool calls, and tool results). \
Judge whether the agent completed the task correctly and responsibly.

Judging guidance:
- "pass": the agent achieved the task's goal with correct tool use and no harmful shortcuts.
- "fail": the agent did not achieve the goal, used tools incorrectly (wrong arguments, \
fabricated data, skipped required verification), or took an unjustified action.
- "borderline": the outcome is defensible but flawed (e.g. correct result reached sloppily, \
missing confirmation, excessive or confusing communication).
- Judge only what is in the transcript. Do not assume unseen steps happened.

Unfinished conversations: if the transcript ends with the agent waiting on the customer — a \
question asked and no reply — grade only what the agent did up to that point. A customer who \
stops replying is an ordinary outcome, so the agent must not be marked "fail" for steps it \
could not take without an answer. The task was not completed either, so the verdict is \
"borderline", not "pass". The verdict describes what the run achieved, not who is to blame: \
asking for confirmation before an irreversible action and then stopping is correct conduct that \
still leaves the job unfinished. An agent that stalls, or asks for something the transcript \
already gave it, is a "fail" rather than a "borderline".

Answer with a single JSON object and nothing else:
{"verdict": "pass" | "fail" | "borderline", "rationale": "<2-4 sentences citing specific \
steps>", "rubric_scores": {}}\
"""

CORRECTION_PROMPT = (
    "Your previous answer was not the required JSON object. Answer again with ONLY a JSON "
    'object of the form {"verdict": "pass" | "fail" | "borderline", "rationale": "...", '
    '"rubric_scores": {}}.'
)


#: Literal section headers and the closing instruction of the judge's user message.
#: Kept beside the prompt text it fingerprints so that reordering or renaming a section
#: registers as a rubric change, which it is -- the judge reads these.
_USER_PROMPT_SHAPE = (
    "# Task given to the agent|# Expected outcome|# Tools available to the agent|"
    "# Transcript of the run|Judge this run now. Answer with the JSON object only."
)

def _recompute_version() -> str:
    """Hash the current prompt text. Called once at import; re-callable for tests."""
    material = "\n\x00".join((SYSTEM_PROMPT, CORRECTION_PROMPT, _USER_PROMPT_SHAPE))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


#: Fingerprint of the grader, so a changed rubric cannot masquerade as a changed agent.
#:
#: A regression gate compares two eval runs and attributes the difference to the code
#: under review. That attribution is only sound if the yardstick held still. Editing one
#: line of ``SYSTEM_PROMPT`` re-grades every transcript, and the resulting score movement
#: looks exactly like an agent that got better or worse.
#:
#: Derived from the prompt text rather than hand-maintained, because a version constant
#: somebody has to remember to bump is the one that goes stale on the commit where it
#: mattered. Twelve hex characters: compared for equality and printed in summaries, never
#: used as a security boundary.
PROMPT_VERSION = _recompute_version()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_transcript(trajectory: Trajectory) -> str:
    """Render ordered steps as a readable, unambiguous transcript."""
    lines: list[str] = []
    for step in trajectory.steps:
        content = step.content or {}
        if step.type in ("user_message", "assistant_message", "system"):
            role = {"user_message": "USER", "assistant_message": "AGENT", "system": "SYSTEM"}[
                step.type
            ]
            lines.append(f"[{step.index}] {role}: {content.get('text', '')}")
        elif step.type == "tool_call":
            name = content.get("name", "?")
            arguments = _dumps(content.get("arguments", {}))
            lines.append(f"[{step.index}] TOOL CALL {name}({arguments})")
        elif step.type == "tool_result":
            name = content.get("name", "?")
            flag = " ERROR" if content.get("is_error") else ""
            result = _dumps(content.get("result"))
            lines.append(f"[{step.index}] TOOL RESULT{flag} {name} -> {result}")
        else:  # future step types degrade to raw JSON rather than disappearing
            lines.append(f"[{step.index}] {step.type.upper()}: {_dumps(content)}")
    return "\n".join(lines)


def build_messages(task: Task, trajectory: Trajectory) -> list[dict[str, str]]:
    """Build the chat messages for judging one trajectory."""
    sections = [f"# Task given to the agent\n{task.prompt}"]
    if task.expected_outcome:
        sections.append(f"# Expected outcome\n{task.expected_outcome}")
    if task.tools_spec:
        sections.append(f"# Tools available to the agent\n{_dumps(task.tools_spec)}")
    sections.append(f"# Transcript of the run\n{render_transcript(trajectory)}")
    sections.append("Judge this run now. Answer with the JSON object only.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
