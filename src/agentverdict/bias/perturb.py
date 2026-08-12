"""The perturbations themselves: pure functions over rendered strings.

Two rules shape everything in this module.

*Nothing here touches the database.* A trajectory is a SQLAlchemy instance whose
steps cascade-delete; padding a loaded ``TrajectoryStep`` and letting the session
flush would write filler into the golden dataset permanently, silently, and in a way
no export would flag. Perturbation therefore happens downstream of the ORM, on the
plain string ``render_transcript`` produced, where there is nothing to flush.

*Perturbations may only add characters, never remove or rewrite them.* An earlier
design had a ``terse`` arm that truncated each agent turn to its first sentence; on
this corpus that deletes a consent request or a closing question in most labeled
trajectories, which are precisely the facts the rubric rules turn on. A judge that
downgrades a run whose consent request was deleted is judging correctly, and
reporting that as length sensitivity would be the exact error this milestone exists
to catch. Append-only makes content preservation structural: the original line is a
prefix of the perturbed line, which a test can check on every trajectory, rather
than a claim an author makes about their own strings.
"""

from __future__ import annotations

import re

from agentverdict.bias.prompts import CANONICAL_PLAN, PromptPlan

#: Shared by both probes, so the two descriptions cannot drift apart.
_IDENTITY_DESCRIPTION = "Byte-identical re-judge: the judge's own noise floor."

#: Variant key -> human-readable description, for reports. ASCII only; these strings
#: reach the console, and Windows terminals mojibake anything else.
ORDER_VARIANTS: dict[str, str] = {
    "identity": _IDENTITY_DESCRIPTION,
    "format_null": (
        "Same sections in the same order, re-rendered with a wider joiner and a "
        "trailing newline per section: byte-different, semantically identical."
    ),
    "context_last": "Transcript first, task and expected outcome and tools after it.",
    "reversed_context": "Context sections in reverse order, transcript still last.",
}

#: Variant key -> human-readable description, for reports.
LENGTH_VARIANTS: dict[str, str] = {
    "identity": _IDENTITY_DESCRIPTION,
    "padded_1x": "One filler sentence appended to every agent turn.",
    "padded_2x": "Two filler sentences appended to every agent turn.",
    "padded_3x": "Three filler sentences appended to every agent turn.",
}

#: Filler appended by the length probe, one sentence per dose.
#:
#: Every sentence is first person and declarative, contains no question mark, and
#: refers neither to the customer's next action nor to the end of the conversation.
#: That is not stylistic: the rubric turns on exactly whether the agent ended by
#: asking something and stopped in the right place ("an unfinished conversation is a
#: pass when the agent did its part"). A bank containing "Shall I proceed?" or "Let
#: me know if there is anything else" would move the run's terminal state, the judge
#: would be right to score it differently, and the length probe would quietly become
#: a rubric probe reporting correct judging as bias.
#:
#: The sentences also assert no facts and claim no actions, so nothing the judge is
#: asked to verify changes. A test enforces all of this against an explicit banned
#: list, because a comment cannot stop the next person from adding a closing line.
FILLER: tuple[str, ...] = (
    "I completely understand how frustrating a situation like this can be.",
    "I really do appreciate your patience while I work through this.",
    "I want to be sure this is handled properly and carefully.",
    "I am treating this with the care that it deserves.",
    "I know that issues like this one are never convenient.",
    "I take this sort of thing seriously on my side.",
)

#: Section layout per order-probe arm. ``format_null`` differs from ``identity`` in
#: whitespace only — it is the arm that tells raw prompt-format jitter apart from a
#: real order effect, which the identity arm cannot do because it also cannot tell
#: "the perturbation had no effect" from "nothing was ever perturbed".
_ORDER_PLANS: dict[str, PromptPlan] = {
    "identity": CANONICAL_PLAN,
    "format_null": PromptPlan(CANONICAL_PLAN.section_order, joiner="\n\n\n", section_suffix="\n"),
    "context_last": PromptPlan(("transcript", "task", "expected", "tools")),
    "reversed_context": PromptPlan(("tools", "expected", "task", "transcript")),
}

#: A rendered step begins with its index in brackets. Agent turns can contain newlines
#: and ``render_transcript`` writes them through unchanged, so a physical line without
#: this prefix is a continuation of the step above it rather than a step of its own.
#: Text that itself begins "[7] AGENT: " would fool this, which is the price of
#: perturbing the rendered string; it is also the price of never touching the ORM.
_STEP_HEADER = re.compile(r"^\[(\d+)\] ")
_AGENT_HEADER = re.compile(r"^\[(\d+)\] AGENT: ")


def plan_for(variant: str) -> PromptPlan:
    """The section layout an arm renders under.

    Length arms perturb the transcript and not the layout, so they render under the
    canonical plan; the difference between them is the dose, not the shape.
    """
    if variant in _ORDER_PLANS:
        return _ORDER_PLANS[variant]
    if variant in LENGTH_VARIANTS:
        return CANONICAL_PLAN
    raise ValueError(f"unknown probe variant {variant!r}")


def doses_for(variant: str) -> int:
    """Filler sentences per agent turn for a length arm; ``identity`` is zero."""
    if variant not in LENGTH_VARIANTS:
        raise ValueError(f"unknown length variant {variant!r}")
    if variant == "identity":
        return 0
    return int(variant.removeprefix("padded_").removesuffix("x"))


def _filler_for(step_index: int, doses: int) -> str:
    """Pick ``doses`` sentences for one agent turn, rotated by its step index.

    Rotating means consecutive turns do not all end with the same sentence. A
    transcript where every turn closes identically reads as a malfunctioning agent,
    and a judge marking that down would be reacting to the repetition rather than to
    the length — a second factor in a probe that is supposed to have one.
    """
    return " ".join(FILLER[(step_index + dose) % len(FILLER)] for dose in range(doses))


def pad_transcript(transcript: str, doses: int) -> str:
    """Append filler to AGENT lines only. ``doses=0`` returns the input unchanged.

    Append-only, per line: the input line is always a prefix of the output line, and
    ``TOOL CALL``, ``TOOL RESULT``, ``USER`` and ``SYSTEM`` lines come through byte
    for byte. Filler lands on the last physical line of each agent turn, so a turn
    whose text contains newlines is extended at its end rather than interrupted.

    A transcript with no agent turns comes back unchanged. The runner's delivery
    check sees the identical ``prompt_sha`` and reports that arm ``not_applied``,
    which is the honest answer — an unperturbed arm measured nothing.
    """
    if doses < 0:
        raise ValueError(f"doses must not be negative, got {doses}")
    if doses == 0:
        return transcript

    lines = transcript.split("\n")
    targets: list[tuple[int, int]] = []  # (physical line to extend, step index)
    open_step: int | None = None
    last_line = -1
    for position, line in enumerate(lines):
        header = _STEP_HEADER.match(line)
        if header is None:
            if open_step is not None:
                last_line = position
            continue
        if open_step is not None:
            targets.append((last_line, open_step))
            open_step = None
        if _AGENT_HEADER.match(line):
            open_step = int(header.group(1))
            last_line = position
    if open_step is not None:
        targets.append((last_line, open_step))

    for position, step_index in targets:
        line = lines[position]
        # Written as line + separator + filler, never as a replacement expression, so
        # that append-only is visible in the assignment rather than argued about.
        separator = "" if not line or line.endswith(" ") else " "
        lines[position] = f"{line}{separator}{_filler_for(step_index, doses)}"
    return "\n".join(lines)
