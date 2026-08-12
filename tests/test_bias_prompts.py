"""Probe rendering, and the one equivalence the whole milestone rests on.

``bias/prompts.py`` writes the judge's user message out a second time so the order
probe can permute it, because parameterising ``judging/prompts.py`` would move
``PROMPT_VERSION`` and invalidate every stored baseline the merge gate reads. A
second copy of anything drifts, and this one drifts silently: the probe would keep
printing intervals, they would simply describe a judge nobody runs.

So the first test here is the important one. Under ``CANONICAL_PLAN`` the probe's
identity arm must be byte-identical to ``build_messages`` — not equivalent, not
semantically the same, identical — on every shape of task the corpus contains. If
production renaming a heading or moving a section breaks it, that is the intended
outcome: a five-second fix, made deliberately, instead of a control arm that quietly
stopped being a control.
"""

from collections.abc import Callable

import pytest

from agentverdict.bias import prompts as probe_prompts
from agentverdict.bias.prompts import (
    CANONICAL_PLAN,
    PromptPlan,
    build_probe_messages,
    render_user_message,
)
from agentverdict.judging import prompts as judge_prompts
from agentverdict.judging.prompts import build_messages, render_transcript
from agentverdict.models import Task, Trajectory

#: The four task shapes the golden bundle actually contains. ``expected_outcome`` and
#: ``tools_spec`` are both optional in the schema and both change how many sections the
#: user message has, so an equivalence proved on the full shape alone proves little.
TASK_SHAPES: dict[str, dict[str, object]] = {
    "full": {
        "expected_outcome": "The refund is issued after the order is verified.",
        "tools_spec": [{"name": "lookup_order"}, {"name": "refund_order"}],
    },
    "no_expected": {"tools_spec": [{"name": "lookup_order"}]},
    "no_tools": {"expected_outcome": "The refund is issued after the order is verified."},
    "bare": {},
}


def _plan(*sections: str) -> PromptPlan:
    """A plan over the named sections, with production's joiner and suffix."""
    return PromptPlan(sections)


@pytest.fixture
def make_shaped_task(make_task: Callable[..., Task]) -> Callable[[str], Task]:
    """Build one of the TASK_SHAPES, keyed so several can coexist in one test."""

    def _make(shape: str) -> Task:
        return make_task(
            key=f"probe-prompt-{shape}",
            prompt="Refund the customer for the damaged item once the order checks out.",
            **TASK_SHAPES[shape],
        )

    return _make


# --- the equivalence -----------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(TASK_SHAPES))
def test_the_canonical_plan_renders_the_production_prompt_byte_for_byte(
    shape: str,
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The identity arm must be the production judge, not a close relative of it.

    Every probe number is a comparison against this arm. If it drifts from
    ``build_messages`` by so much as a blank line, the control describes a prompt
    that grades nothing in production, and the shifts measured against it are
    differences between two prompts nobody ships.
    """
    task = make_shaped_task(shape)
    trajectory = make_trajectory(task=task)

    probe = build_probe_messages(task, render_transcript(trajectory), CANONICAL_PLAN)
    production = build_messages(task, trajectory)

    assert probe == production
    # Compared as bytes as well as as strings: two messages can compare equal in
    # Python and still be different documents to a tokenizer only if the encoding
    # differs, which is cheap to rule out and expensive to discover later.
    assert [message["content"].encode("utf-8") for message in probe] == [
        message["content"].encode("utf-8") for message in production
    ]


def test_the_probe_imports_the_system_prompt_instead_of_copying_it(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """A copied system prompt is a second rubric that no fingerprint covers.

    ``PROMPT_VERSION`` hashes ``judging.prompts.SYSTEM_PROMPT``. A probe holding its
    own transcription of that text would keep passing the equality test above on the
    day production edits one line, and would then measure the old rubric while
    reporting the new rubric's version.
    """
    assert probe_prompts.SYSTEM_PROMPT is judge_prompts.SYSTEM_PROMPT

    task = make_shaped_task("full")
    transcript = render_transcript(make_trajectory(task=task))
    messages = build_probe_messages(task, transcript, CANONICAL_PLAN)

    assert messages[0] == {"role": "system", "content": judge_prompts.SYSTEM_PROMPT}
    assert [message["role"] for message in messages] == ["system", "user"]


def test_every_heading_the_fingerprint_covers_appears_in_the_probe_rendering(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """``_USER_PROMPT_SHAPE`` is what the rubric fingerprint knows about the layout.

    The probe keeps its own copy of the headings, deliberately, so that a production
    rename shows up here rather than being inherited silently. This is the assertion
    that turns that copy from a liability into a tripwire.
    """
    task = make_shaped_task("full")
    user_message = build_probe_messages(
        task, render_transcript(make_trajectory(task=task)), CANONICAL_PLAN
    )[1]["content"]

    for section in judge_prompts._USER_PROMPT_SHAPE.split("|"):
        assert section in user_message


# --- what a plan may and may not move ------------------------------------------


def test_reordering_moves_the_sections_and_changes_nothing_else(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """An order arm is a permutation, or it is measuring content as well as order."""
    task = make_shaped_task("full")
    transcript = render_transcript(make_trajectory(task=task))

    canonical = render_user_message(task, transcript, CANONICAL_PLAN)
    reordered = render_user_message(
        task, transcript, _plan("transcript", "task", "expected", "tools")
    )

    assert reordered != canonical
    # Same blocks, different order: nothing was added, dropped or reworded.
    assert sorted(reordered.split("\n\n")) == sorted(canonical.split("\n\n"))
    assert reordered.split("\n\n")[0].startswith("# Transcript of the run")


def test_the_closing_instruction_stays_last_under_every_plan(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """It states the answer contract, so moving it would change the task, not the order.

    A probe whose arms asked for the answer in different places would be measuring
    two things at once, and the one it named would not be the one that moved.
    """
    task = make_shaped_task("full")
    transcript = render_transcript(make_trajectory(task=task))
    closing = "Judge this run now. Answer with the JSON object only."

    for order in (
        ("task", "expected", "tools", "transcript"),
        ("transcript", "task", "expected", "tools"),
        ("tools", "expected", "task", "transcript"),
    ):
        rendered = render_user_message(task, transcript, _plan(*order))
        assert rendered.endswith(closing)
        assert rendered.count(closing) == 1


def test_the_format_null_plan_changes_bytes_without_changing_sections(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The arm that separates order sensitivity from whitespace sensitivity.

    The identity arm cannot do that job: it is byte-identical, so a judge that is
    merely twitchy about formatting looks perfectly stable there and every real order
    arm inherits the twitch as though it were an order effect.
    """
    task = make_shaped_task("full")
    transcript = render_transcript(make_trajectory(task=task))

    canonical = render_user_message(task, transcript, CANONICAL_PLAN)
    formatted = render_user_message(
        task,
        transcript,
        PromptPlan(CANONICAL_PLAN.section_order, joiner="\n\n\n", section_suffix="\n"),
    )

    assert formatted != canonical
    # Every non-whitespace character, in order, is the same document.
    assert "".join(formatted.split()) == "".join(canonical.split())


def test_a_task_without_expected_or_tools_gets_no_empty_headings(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """An empty heading is a prompt change, so a probe must not invent one.

    Production omits the section; a probe that emitted "# Expected outcome" with
    nothing under it would be telling the judge something about a task that says
    nothing, on exactly the arm that is supposed to hold content fixed.
    """
    task = make_shaped_task("bare")
    rendered = render_user_message(
        task, render_transcript(make_trajectory(task=task)), CANONICAL_PLAN
    )

    assert "# Expected outcome" not in rendered
    assert "# Tools available to the agent" not in rendered
    assert rendered.startswith("# Task given to the agent\n")


def test_a_plan_naming_a_section_that_does_not_exist_is_a_programming_error() -> None:
    """A typo would otherwise drop a section quietly and still report a number."""
    with pytest.raises(ValueError, match="unknown prompt section"):
        PromptPlan(("task", "transcrpit"))


def test_a_plan_that_repeats_a_section_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate prompt section"):
        PromptPlan(("task", "transcript", "task"))


def test_the_renderer_reads_a_string_and_never_the_trajectory(
    make_shaped_task: Callable[[str], Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The perturbation seam: the transcript arrives already rendered.

    This is what keeps padding downstream of the ORM. If rendering took a
    ``Trajectory`` here, the only place left to perturb would be the loaded steps,
    and the session would eventually flush them into the golden dataset.
    """
    task = make_shaped_task("full")
    trajectory = make_trajectory(task=task)

    rendered = render_user_message(task, "[0] AGENT: substituted entirely.", CANONICAL_PLAN)

    assert "[0] AGENT: substituted entirely." in rendered
    assert render_transcript(trajectory) not in rendered
