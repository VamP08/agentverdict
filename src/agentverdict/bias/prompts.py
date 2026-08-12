"""Probe prompt rendering: the production judge message, assembled from a plan.

The order probe has to send the judge the same information in a different order.
That cannot be done by parameterising ``judging.prompts``: ``SYSTEM_PROMPT``,
``_USER_PROMPT_SHAPE`` and ``build_messages`` are what ``PROMPT_VERSION`` hashes,
and the merge gate refuses to compare two eval runs whose fingerprints differ. A
probe that reached into the production renderer would invalidate every stored
baseline to measure something the baselines were never part of.

So the section assembly is written out a second time here, parameterised by a
:class:`PromptPlan`. The duplication is deliberate and it carries exactly one rule:
under ``CANONICAL_PLAN`` this module must emit bytes indistinguishable from
``build_messages``. A test asserts that byte for byte, because the identity arm is
the control the entire probe is measured against — if it drifts, the control stops
describing the judge that calibration and the gate describe, and every number the
probe prints is about a judge that does not exist.

Everything the judge actually reads is imported rather than copied: the system
prompt, the transcript renderer, and the JSON encoder for the tools block.
"""

from __future__ import annotations

from dataclasses import dataclass

# ``_dumps`` is private to the judging package and imported anyway: the tools block
# has to serialise to the same bytes production sends, and a second json.dumps call
# with hand-copied keyword arguments is exactly the kind of near-copy that drifts.
from agentverdict.judging.prompts import SYSTEM_PROMPT, _dumps, render_transcript
from agentverdict.models import Task

__all__ = [
    "CANONICAL_PLAN",
    "PromptPlan",
    "build_probe_messages",
    "render_transcript",
    "render_user_message",
]

#: Section headers, byte for byte the ones ``build_messages`` emits. They are literals
#: rather than a parse of ``_USER_PROMPT_SHAPE`` because that constant encodes order
#: as well as text, and a probe that silently inherited a reordering would have no
#: control arm left. Keeping them here means a production rename shows up as a failing
#: byte-identity test — a five-second fix, and the only place the drift can hide.
_SECTION_HEADERS: dict[str, str] = {
    "task": "# Task given to the agent",
    "expected": "# Expected outcome",
    "tools": "# Tools available to the agent",
    "transcript": "# Transcript of the run",
}

#: The closing line of the user message, in every arm.
_CLOSING_INSTRUCTION = "Judge this run now. Answer with the JSON object only."


@dataclass(frozen=True)
class PromptPlan:
    """How a probe arm assembles the judge's user message.

    ``section_order`` is a subset or permutation of the four content sections.
    ``joiner`` and ``section_suffix`` exist for the format-null arm, which needs a
    rendering that differs in bytes while differing in nothing else — that arm is what
    separates real order sensitivity from the judge twitching at whitespace.

    The closing instruction is not a section and cannot be ordered. It states the
    answer contract rather than describing the run, so moving it would change what the
    judge is asked to do as well as what it reads, and the probe would be measuring two
    things at once.
    """

    section_order: tuple[str, ...]
    joiner: str = "\n\n"
    section_suffix: str = ""

    def __post_init__(self) -> None:
        # A typo drops a section and a repeat duplicates one, in both cases silently:
        # the arm still renders, still judges, and still reports a number. Fail here,
        # where it is a plain programming error, rather than in a published finding.
        unknown = tuple(key for key in self.section_order if key not in _SECTION_HEADERS)
        if unknown:
            raise ValueError(f"unknown prompt section(s): {unknown}")
        if len(set(self.section_order)) != len(self.section_order):
            raise ValueError(f"duplicate prompt section in {self.section_order}")


#: The production layout. The identity and length arms use it unchanged; the order
#: arms permute it. Its rendering is the one that must match ``build_messages``.
CANONICAL_PLAN = PromptPlan(("task", "expected", "tools", "transcript"))


def render_user_message(task: Task, transcript: str, plan: PromptPlan) -> str:
    """Assemble the judge's user message for one arm.

    ``expected`` and ``tools`` are omitted when the task lacks them, exactly as
    production does — a task with no expected outcome must not gain an empty heading
    just because a probe rendered it, since that heading is itself a prompt change.

    The transcript arrives already rendered (and, for the length probe, already
    padded). Nothing here reads the trajectory, which is what keeps the perturbation
    seam clear of the ORM.
    """
    bodies: dict[str, str] = {"task": task.prompt, "transcript": transcript}
    if task.expected_outcome:
        bodies["expected"] = task.expected_outcome
    if task.tools_spec:
        bodies["tools"] = _dumps(task.tools_spec)
    sections = [
        f"{_SECTION_HEADERS[key]}\n{bodies[key]}{plan.section_suffix}"
        for key in plan.section_order
        if key in bodies
    ]
    sections.append(_CLOSING_INSTRUCTION)
    return plan.joiner.join(sections)


def build_probe_messages(task: Task, transcript: str, plan: PromptPlan) -> list[dict[str, str]]:
    """Build the chat messages for one probe arm.

    With ``CANONICAL_PLAN`` and an unperturbed transcript the result is byte-identical
    to ``judging.prompts.build_messages(task, trajectory)``. That equivalence is the
    probe's entire claim to be measuring the production judge, so it is asserted by a
    test rather than trusted to review.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_message(task, transcript, plan)},
    ]
