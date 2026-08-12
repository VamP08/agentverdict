"""The perturbations: append-only padding, the filler bank, and the arm registry.

Two properties here are structural rather than stylistic, and both are asserted
rather than argued.

*Append-only.* The length probe may only add characters. An earlier design truncated
each agent turn to its first sentence, which on this corpus deletes a consent request
or a closing question in most labeled trajectories — the facts the rubric rules turn
on. A judge that downgrades a run whose consent request was deleted is judging
correctly, and the probe would have published that as length sensitivity. The test is
therefore per line and mechanical: the input line is a prefix of the output line.

*The filler bank cannot ask a question.* The rubric grades how a transcript ends. A
sentence like "Shall I proceed?" moves the run's terminal state, the judge would be
right to score it differently, and a length probe would quietly become a rubric probe
reporting correct judging as bias. An explicit banned list is what stops the next
person from adding one, which is why the ban lives here and not in a comment.
"""

import re
from collections.abc import Callable

import pytest

from agentverdict.bias.perturb import (
    FILLER,
    LENGTH_VARIANTS,
    ORDER_VARIANTS,
    doses_for,
    pad_transcript,
    plan_for,
)
from agentverdict.bias.prompts import CANONICAL_PLAN, render_user_message
from agentverdict.judging.prompts import render_transcript
from agentverdict.models import Task, Trajectory

#: A rendered transcript in every shape the renderer emits, including an agent turn
#: whose text runs over two physical lines — the case where "append to the agent's
#: line" and "append to the agent's turn" are different instructions.
TRANSCRIPT = "\n".join(
    (
        "[0] USER: Hi, I want a refund for order A123",
        "[1] AGENT: Let me look that up for you.",
        '[2] TOOL CALL lookup_order({"order_id": "A123"})',
        '[3] TOOL RESULT lookup_order -> {"status": "delivered"}',
        "[4] SYSTEM: escalation policy attached",
        "[5] AGENT: Your refund has been initiated.",
        "It should land within three days.",
        '[6] TOOL RESULT ERROR refund_order -> {"error": "upstream timeout"}',
    )
)

#: Physical lines of TRANSCRIPT that end an agent turn, and therefore the only ones
#: padding may touch. Line 6 is the continuation of the turn opened on line 5.
AGENT_TURN_LINES = (1, 6)

#: A step line begins with its index in brackets; anything else is a continuation.
STEP_HEADER = re.compile(r"^\[\d+\] ")

#: Substrings no filler sentence may contain, with the reason each one is banned.
#: The length probe holds everything except length fixed, and each of these would
#: move something else: the customer's next action, the end of the conversation, or a
#: claim about what the agent did — which is the thing the judge is asked to verify.
BANNED: tuple[tuple[str, str], ...] = (
    # The customer's next action. The rubric says an open ending is a pass when the
    # agent did its part, so inventing a question changes the verdict legitimately.
    ("?", "a question moves the run's terminal state"),
    ("shall i", "asks the customer to decide"),
    ("would you like", "asks the customer to decide"),
    ("do you want", "asks the customer to decide"),
    ("can you", "asks the customer to act"),
    ("could you", "asks the customer to act"),
    ("let me know", "asks the customer to reply"),
    ("please confirm", "asks the customer to reply"),
    ("please reply", "asks the customer to reply"),
    ("your reply", "points at a reply that never came"),
    ("get back to", "points at a reply that never came"),
    ("reach out", "points at a reply that never came"),
    ("feel free", "invites the customer to act"),
    ("you can", "tells the customer what to do next"),
    # The end of the conversation, for the same reason.
    ("anything else", "closes the conversation"),
    ("anything further", "closes the conversation"),
    ("in the meantime", "implies a later turn"),
    ("have a great", "signs off"),
    ("have a good", "signs off"),
    ("goodbye", "signs off"),
    ("take care", "signs off"),
    ("thanks for contacting", "signs off"),
    ("thank you for contacting", "signs off"),
    # Facts and actions. Filler must assert nothing the judge has to check, or the
    # padded arm is testing the judge's reading of a different run.
    ("confirm", "claims or requests a confirmation"),
    ("proceed", "announces an action"),
    ("refund", "claims an action the transcript must show"),
    ("cancel", "claims an action the transcript must show"),
    ("i have processed", "claims an action the transcript must show"),
    ("i checked", "claims a verification the transcript must show"),
)


def _lines(transcript: str) -> list[str]:
    return transcript.split("\n")


def _step_headers(lines: list[str]) -> list[str]:
    """The ``[n] `` prefix of every line that opens a step, in order."""
    return [match.group(0) for match in map(STEP_HEADER.match, lines) if match is not None]


# --- pad_transcript ------------------------------------------------------------


def test_zero_doses_is_the_identity() -> None:
    """The control arm of the length probe, and the thing every dose is measured from."""
    assert pad_transcript(TRANSCRIPT, 0) == TRANSCRIPT


@pytest.mark.parametrize("doses", [1, 2, 3])
def test_padding_only_appends(doses: int) -> None:
    """Line by line, the original is a prefix of the padded version.

    This is what makes content preservation structural instead of a claim the author
    of the filler bank makes about their own strings. Nothing was deleted, nothing was
    reworded, and no line was moved: same count, same order, each one extended or
    left alone.
    """
    padded = _lines(pad_transcript(TRANSCRIPT, doses))
    original = _lines(TRANSCRIPT)

    assert len(padded) == len(original)
    for before, after in zip(original, padded, strict=True):
        assert after.startswith(before)


@pytest.mark.parametrize("doses", [1, 2, 3])
def test_only_the_agent_turns_grow(doses: int) -> None:
    """Tool calls, tool results, user turns and system lines come back byte for byte.

    Padding a tool result would change what the agent was told; padding a user turn
    would change what it was asked. Either one makes the arm a content probe wearing
    a length probe's name.
    """
    padded = _lines(pad_transcript(TRANSCRIPT, doses))
    original = _lines(TRANSCRIPT)

    grew = [
        position
        for position, (before, after) in enumerate(zip(original, padded, strict=True))
        if before != after
    ]
    assert tuple(grew) == AGENT_TURN_LINES
    for position, line in enumerate(original):
        if position not in AGENT_TURN_LINES:
            assert padded[position] == line


def test_a_multi_line_agent_turn_is_extended_at_its_end_not_interrupted() -> None:
    """Filler lands after the last physical line of the turn.

    Splicing it in after the turn's first line would put the agent's own words after
    the padding, which reads as a different message rather than a longer one.
    """
    padded = _lines(pad_transcript(TRANSCRIPT, 1))
    original = _lines(TRANSCRIPT)

    assert padded[5] == original[5]  # "[5] AGENT: Your refund has been initiated."
    assert padded[6].startswith(original[6])
    assert len(padded[6]) > len(original[6])


@pytest.mark.parametrize("doses", [1, 2, 3])
def test_padding_cannot_forge_a_step(doses: int) -> None:
    """The step headers are exactly the ones the renderer wrote.

    Filler is appended inside a line and contains no newline, so a padded transcript
    has the same steps in the same order. A bank entry that started a new bracketed
    line would invent a step the agent never took.
    """
    padded = _lines(pad_transcript(TRANSCRIPT, doses))
    original = _lines(TRANSCRIPT)

    assert [bool(STEP_HEADER.match(line)) for line in padded] == [
        bool(STEP_HEADER.match(line)) for line in original
    ]
    assert _step_headers(padded) == _step_headers(original)


def test_the_dose_ladder_is_nested() -> None:
    """Each dose extends the one below it, so the ladder varies only in amount.

    Three arms that padded with three different sentences would be three different
    perturbations, and a shift that grew across them would say nothing about length.
    """
    one, two, three = (_lines(pad_transcript(TRANSCRIPT, doses)) for doses in (1, 2, 3))

    for position in AGENT_TURN_LINES:
        assert two[position].startswith(one[position])
        assert three[position].startswith(two[position])
        assert len(one[position]) < len(two[position]) < len(three[position])


def test_a_transcript_with_no_agent_turn_comes_back_unchanged() -> None:
    """Not an error: the runner's delivery check reports that arm ``not_applied``.

    Padding a transcript that has nothing to pad is a real state of the corpus, and
    quietly returning something is how an arm ends up reported as "no bias detected"
    when it never perturbed anything.
    """
    toolsonly = '[0] TOOL CALL lookup_order({"order_id": "A123"})\n[1] USER: any news'

    assert pad_transcript(toolsonly, 3) == toolsonly


def test_a_negative_dose_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        pad_transcript(TRANSCRIPT, -1)


def test_padding_a_rendered_trajectory_preserves_every_recorded_step(
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The same append-only property, on a transcript the renderer actually produced.

    TRANSCRIPT above is hand-written to cover the awkward shapes; this one comes off
    the default factory, so the two together cover both the edge cases and the case.
    """
    transcript = render_transcript(make_trajectory())

    padded = pad_transcript(transcript, 2)

    for before, after in zip(_lines(transcript), _lines(padded), strict=True):
        assert after.startswith(before)
    assert "Your refund has been initiated." in padded
    assert "TOOL RESULT lookup_order" in padded


# --- the filler bank -----------------------------------------------------------


@pytest.mark.parametrize("sentence", FILLER)
def test_every_filler_sentence_is_one_first_person_declarative_line(sentence: str) -> None:
    """First person, one sentence, no line break, ASCII.

    One sentence per dose is what makes the dose a unit: two arms differing by "one
    filler sentence" have to differ by a comparable amount of text. The line break
    matters for the same reason padding cannot forge a step.
    """
    assert sentence.startswith("I ")
    assert sentence.endswith(".")
    assert sentence.count(".") == 1
    assert "\n" not in sentence
    assert sentence.isascii()


@pytest.mark.parametrize("sentence", FILLER)
def test_no_filler_sentence_can_turn_the_length_probe_into_a_rubric_probe(
    sentence: str,
) -> None:
    """The explicit ban, checked rather than described.

    Every entry in BANNED moves something the length probe is supposed to hold
    fixed — what the customer is asked to do, whether the conversation ended, or what
    the agent claims to have done. A judge that reacted to any of those would be
    judging correctly, and the probe would report it as length sensitivity.
    """
    lowered = sentence.lower()
    for banned, reason in BANNED:
        assert banned not in lowered, f"{sentence!r} contains {banned!r}: {reason}"


def test_the_bank_is_large_enough_that_the_largest_dose_never_repeats_itself() -> None:
    """Three sentences per turn at the top of the ladder, and they must be distinct.

    A turn that ends with the same sentence three times reads as a malfunctioning
    agent. A judge marking that down would be reacting to the repetition, not to the
    length, and a probe with two factors in it measures neither.
    """
    largest = max(doses_for(variant) for variant in LENGTH_VARIANTS)

    assert largest == 3
    assert len(FILLER) >= largest
    assert len(set(FILLER)) == len(FILLER)


def test_consecutive_agent_turns_do_not_all_end_with_the_same_sentence() -> None:
    """The rotation, which is what keeps repetition out of the padded transcript."""
    original = _lines(TRANSCRIPT)
    padded = _lines(pad_transcript(TRANSCRIPT, 1))

    appended = [padded[position][len(original[position]) :] for position in AGENT_TURN_LINES]
    assert all(text.strip() in FILLER for text in appended)
    assert len(set(appended)) == len(appended)


# --- the arm registry ----------------------------------------------------------


def test_both_probes_share_one_identity_control() -> None:
    """The control is the same arm in both probes, so the two floors are comparable."""
    assert ORDER_VARIANTS["identity"] == LENGTH_VARIANTS["identity"]
    assert plan_for("identity") == CANONICAL_PLAN
    assert doses_for("identity") == 0


def test_the_length_arms_change_the_dose_and_never_the_layout() -> None:
    """Otherwise the length probe would be permuting sections as well as padding."""
    for variant in LENGTH_VARIANTS:
        assert plan_for(variant) == CANONICAL_PLAN
    assert [doses_for(variant) for variant in LENGTH_VARIANTS] == [0, 1, 2, 3]


def test_every_order_arm_carries_the_same_sections_as_the_control() -> None:
    """An order arm is a permutation. Dropping a section would change the content."""
    for variant in ORDER_VARIANTS:
        sections = plan_for(variant).section_order
        assert set(sections) == set(CANONICAL_PLAN.section_order)
        assert len(sections) == len(CANONICAL_PLAN.section_order)


def test_the_format_null_arm_reorders_nothing_and_still_differs(
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """Same order, different bytes: the arm that isolates whitespace jitter."""
    plan = plan_for("format_null")
    assert plan.section_order == CANONICAL_PLAN.section_order
    assert (plan.joiner, plan.section_suffix) != (
        CANONICAL_PLAN.joiner,
        CANONICAL_PLAN.section_suffix,
    )

    task = make_task(key="format-null-task", expected_outcome="Refunded.", tools_spec=[{"n": 1}])
    transcript = render_transcript(make_trajectory(task=task))
    assert render_user_message(task, transcript, plan) != render_user_message(
        task, transcript, CANONICAL_PLAN
    )


def test_no_order_arm_reorders_the_transcript_itself(
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """Reordering an agent's steps changes what the agent did.

    That would be a different run, not the same run seen differently, and any verdict
    movement would be the judge correctly noticing the change. The order probe moves
    the prompt's sections and leaves the transcript block byte-identical.
    """
    task = make_task(key="order-arm-task", expected_outcome="Refunded.", tools_spec=[{"n": 1}])
    transcript = render_transcript(make_trajectory(task=task))
    heading = "# Transcript of the run\n"

    for variant in ORDER_VARIANTS:
        rendered = render_user_message(task, transcript, plan_for(variant))
        block = rendered.split(heading, 1)[1]
        assert block.startswith(transcript)


def test_an_unknown_variant_is_refused_by_both_lookups() -> None:
    """A typo would otherwise render the canonical prompt and be reported as an arm."""
    with pytest.raises(ValueError, match="unknown probe variant"):
        plan_for("padded_4x")
    with pytest.raises(ValueError, match="unknown length variant"):
        doses_for("context_last")


def test_every_arm_description_is_ascii() -> None:
    """These strings reach the console, and Windows terminals mojibake anything else."""
    for registry in (ORDER_VARIANTS, LENGTH_VARIANTS):
        for key, description in registry.items():
            assert key.isascii()
            assert description.isascii()
