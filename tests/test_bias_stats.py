"""Probe statistics, with every number worked out in the comment beside it.

Three of these functions have a wrong version that looks right, and each wrong
version fails in the same direction — toward finding bias:

* ``disagreement_within`` counted over ordered pairs *including* self-pairs is the
  plug-in ``1 - sum p**2``. A draw agrees with itself with probability one, so the
  measured noise floor comes out low, and since that floor is what every perturbed
  arm is subtracted from, every arm gains a spurious positive shift.
* ``disagreement_across`` paired by index would depend on the order rows came back
  in, which disqualifies it as a statistic before its bias is even discussed.
* ``bootstrap_clustered_mean_ci`` resampling trajectories flat would treat twenty
  replays of one scenario as twenty draws on the judge, and report a quirk of that
  scenario with an interval narrow enough to be believed.

So the assertions here are exact fractions rather than approximations, and each one
also pins the wrong answer it must not return.
"""

import random

import pytest

from agentverdict.bias.stats import (
    ArmSamples,
    adjusted_confidence,
    bootstrap_clustered_mean_ci,
    disagreement_across,
    disagreement_within,
    min_movers,
    signed_shift,
)
from agentverdict.calibration.stats import bootstrap_mean_ci

#: A judge that has become less decisive without becoming more generous: the same
#: mean score, a different spread. Used to show what the headline must not be.
SCATTERED_CONTROL = ["pass"] * 8 + ["borderline", "fail"]
SCATTERED_VARIANT = ["pass"] * 7 + ["borderline"] * 3


# --- disagreement_within -------------------------------------------------------


def test_within_arm_disagreement_excludes_self_pairs() -> None:
    """Three draws, two distinct values: 2/3, and specifically not 4/9.

    The arm is [pass, pass, fail]. Over the C(3,2) = 3 distinct unordered pairs:
    (pass, pass) agrees, (pass, fail) differs, (pass, fail) differs -> 2/3.

    Counting all 3 x 3 = 9 ordered pairs instead sweeps in the three self-pairs,
    which agree by definition: 9 - (2*2 + 1*1) = 4 differ, giving 4/9 = 0.444. That
    is the plug-in estimator, it understates the judge's noise floor by floor/n, and
    every perturbed arm is then measured against a floor that is too low.
    """
    assert disagreement_within(["pass", "pass", "fail"]) == pytest.approx(2 / 3)
    assert disagreement_within(["pass", "pass", "fail"]) != pytest.approx(4 / 9)


def test_within_arm_disagreement_of_three_distinct_draws_is_one() -> None:
    """All three of [pass, borderline, fail] differ, so the floor is 1.0, not 6/9.

    Same trap from the other side: the plug-in counts 9 ordered pairs, 3 of which are
    self-pairs, and reports 6/9 = 0.667 for an arm in which no two draws agreed at
    all. A judge that answers differently every time has a noise floor of one.
    """
    assert disagreement_within(["pass", "borderline", "fail"]) == 1.0
    assert disagreement_within(["pass", "borderline", "fail"]) != pytest.approx(6 / 9)


def test_within_arm_disagreement_of_an_even_split_is_two_thirds() -> None:
    """[pass, pass, borderline, borderline]: 4 of the 6 distinct pairs differ.

    The plug-in returns 0.5 here. Pinned because 0.5 is the answer somebody reaches
    for when they "simplify" the denominator back to probabilities.
    """
    assert disagreement_within(["pass", "pass", "borderline", "borderline"]) == pytest.approx(2 / 3)


def test_a_judge_that_never_wavers_has_a_floor_of_zero() -> None:
    assert disagreement_within(["pass", "pass", "pass"]) == 0.0


def test_one_draw_cannot_see_noise() -> None:
    """None, not 0.0. At R=1 the control measures nothing, which is why R >= 2."""
    assert disagreement_within(["pass"]) is None
    assert disagreement_within([]) is None


def test_failed_calls_are_dropped_rather_than_treated_as_a_verdict() -> None:
    """A call that never returned is not an opinion, and not a disagreement either."""
    assert disagreement_within(["pass", None, "fail"]) == 1.0  # two usable draws, both differ
    assert disagreement_within(["pass", None]) is None  # one usable draw left


# --- disagreement_across -------------------------------------------------------


def test_cross_arm_disagreement_counts_every_pair() -> None:
    """Control [pass, pass] against variant [fail, pass]: 2 of the 4 pairs differ.

    Every one of the 2 x 2 pairs joins two independent draws, so all four count.
    """
    assert disagreement_across(["pass", "pass"], ["fail", "pass"]) == 0.5


def test_cross_arm_disagreement_does_not_depend_on_the_order_rows_came_back_in() -> None:
    """Repeats are exchangeable, so no repeat has a privileged partner in the other arm.

    A statistic that paired repeat 1 with repeat 1 would move when the database
    returned the same rows in a different order — which is not a property of the
    judge, and would make the probe unreproducible for a reason nobody could see.
    """
    control = ["pass", "pass", "borderline", "fail"]
    variant = ["borderline", "pass", "fail", "fail"]
    expected = disagreement_across(control, variant)

    # 4 x 4 = 16 pairs; agreeing = 2*1 (pass) + 1*1 (borderline) + 1*2 (fail) = 5.
    assert expected == pytest.approx(11 / 16)

    rng = random.Random(20260812)
    for _ in range(20):
        shuffled_control = control[:]
        shuffled_variant = variant[:]
        rng.shuffle(shuffled_control)
        rng.shuffle(shuffled_variant)
        assert disagreement_across(shuffled_control, variant) == expected
        assert disagreement_across(control, shuffled_variant) == expected
        assert disagreement_across(shuffled_control, shuffled_variant) == expected


def test_cross_and_within_are_equal_in_expectation_and_not_on_realised_draws() -> None:
    """[p, p, p, b, f] gives 0.700 within and 0.560 across, and both are correct.

    Within counts 20 ordered non-self pairs and finds 14 differing; across counts all
    25 pairs, including the 5 self-pairs, and finds 14. The two match in expectation
    under the null, which is what makes their difference meaningful — a test that
    asserted they match on identical realised draws would be testing the wrong thing
    and would be "fixed" by reintroducing the self-pair bug.
    """
    draws = ["pass", "pass", "pass", "borderline", "fail"]

    assert disagreement_within(draws) == pytest.approx(0.7)
    assert disagreement_across(draws, draws) == pytest.approx(0.56)


def test_cross_arm_disagreement_is_undefined_when_an_arm_is_empty() -> None:
    assert disagreement_across([], ["pass"]) is None
    assert disagreement_across(["pass"], [None, None]) is None


# --- signed_shift --------------------------------------------------------------


def test_the_headline_is_signed_and_ordinal() -> None:
    """Control [pass, pass] = 1.0; variant [borderline, fail] = 0.25; shift -0.75."""
    assert signed_shift(["pass", "pass"], ["borderline", "fail"]) == -0.75
    assert signed_shift(["borderline", "fail"], ["pass", "pass"]) == 0.75


def test_the_headline_ignores_a_perturbation_that_only_scatters_the_answers() -> None:
    """The reason the headline is not a flip rate.

    Control is (8 pass, 1 borderline, 1 fail), mean 0.850. Variant is (7 pass, 3
    borderline), mean 0.850. Nothing has been graded up or down — the judge has only
    become less decisive — and the signed shift correctly reports 0.000.

    Cross-arm disagreement rises from 0.340 to 0.410 over the same draws, so a flip
    rate promoted to headline would announce seven points of "bias" where there is no
    movement at all. A longer prompt is a plausible way to make a judge less decisive,
    so this is the case the length probe meets rather than a contrived one.
    """
    assert signed_shift(SCATTERED_CONTROL, SCATTERED_VARIANT) == 0.0
    assert disagreement_across(SCATTERED_CONTROL, SCATTERED_CONTROL) == pytest.approx(0.34)
    assert disagreement_across(SCATTERED_CONTROL, SCATTERED_VARIANT) == pytest.approx(0.41)


def test_the_headline_is_undefined_when_either_arm_returned_nothing() -> None:
    """None rather than 0.0: "every call failed" is not "the judge did not move"."""
    assert signed_shift(["pass"], [None]) is None
    assert signed_shift([None, None], ["pass"]) is None


def test_arm_samples_carry_the_cluster_they_belong_to() -> None:
    """The task key travels with the draws because it is the resampling unit."""
    sample = ArmSamples(trajectory_id="t1", task_key="refund-simple-01", verdicts=("pass", "fail"))

    assert disagreement_within(sample.verdicts) == 1.0
    assert sample.task_key == "refund-simple-01"


# --- min_movers ----------------------------------------------------------------


def test_four_trajectories_must_move_at_thirteen_and_still_at_sixty() -> None:
    """The bound this milestone is built around, and it barely depends on n.

    With k movers and n-k exact zeros, a resample misses every mover with probability
    ((n-k)/n)**n, and those resamples have a mean of exactly zero. At n=13 that atom
    is 0.0330 for k=3 and 0.0084 for k=4, and the 2.5th percentile clears zero only
    once the atom is under 0.025 — so four. At n=60 it is 0.0460 for k=3 and 0.0160
    for k=4: still four. More trajectories lower the *rate* four movers represents;
    they do not buy the ability to see a two-trajectory effect.
    """
    assert min_movers(13) == 4
    assert min_movers(60) == 4
    # Not a coincidence at those two points: ((n-k)/n)**n tends to e**-k, and 0.025
    # falls between e**-3 = 0.0498 and e**-4 = 0.0183.
    assert {min_movers(n) for n in range(9, 200)} == {4}


def test_a_tiny_sample_needs_almost_everything_to_move() -> None:
    """At n=4 the answer is 3 of 4, which is a statement about the instrument."""
    assert [min_movers(n) for n in range(2, 9)] == [2, 3, 3, 3, 3, 3, 3]


def test_reading_at_a_stricter_level_demands_more_movers() -> None:
    """The report reads intervals at the Sidak-adjusted level, so the floor moves too.

    At 95% over four arms the per-interval level is 0.9873, and at n=13 that needs
    five movers rather than four. A floor computed at the nominal level and printed
    beside intervals read at the adjusted one would understate what the run can see.
    """
    adjusted = adjusted_confidence(0.95, 4)

    assert min_movers(13, adjusted) == 5
    assert min_movers(13, 0.95) == 4


def test_min_movers_is_undefined_where_no_interval_exists_at_all() -> None:
    """Below two items nothing is resampled, so no number of movers helps."""
    assert min_movers(1) is None
    assert min_movers(0) is None
    assert min_movers(13, confidence=1.0) is None
    assert min_movers(13, confidence=0.0) is None


# --- bootstrap_clustered_mean_ci -----------------------------------------------


def test_one_big_task_does_not_resample_as_twenty_independent_draws() -> None:
    """Twenty replays of one scenario are twenty views of a single draw on the judge.

    The groups here are one task that moved +1.0 on twenty trajectories and one task
    that moved -1.0 on a single trajectory. Flat resampling over the 21 values gives
    an interval entirely above zero — a confident finding resting on one scenario.

    The clustered bootstrap draws two task keys with replacement, so it draws the
    small task twice a quarter of the time and the resample mean is exactly -1.0
    there; the 2.5th percentile therefore sits at -1.0 and the interval spans zero.
    That is the honest answer at two clusters: the effect and the scenario cannot be
    told apart.
    """
    groups = {"crowded": [1.0] * 20, "lonely": [-1.0]}

    clustered = bootstrap_clustered_mean_ci(groups).interval
    flat = bootstrap_mean_ci([value for values in groups.values() for value in values]).interval

    assert clustered == (-1.0, 1.0)
    assert flat is not None
    assert flat[0] > 0.0  # what resampling trajectories flat would have claimed


def test_a_task_contributes_its_own_size_to_every_resample() -> None:
    """Each drawn task brings all of its trajectories, so the mean stays size-weighted.

    That is the quantity the report prints beside the interval. A macro-average over
    tasks here would build an interval for an estimator nobody quotes.
    """
    groups = {"crowded": [1.0] * 20, "lonely": [-1.0]}

    interval = bootstrap_clustered_mean_ci(groups, iterations=200, seed=7).interval

    assert interval is not None
    low, high = interval
    # The three reachable resample means are -1.0 (both draws lonely), 19/21 (one of
    # each) and 1.0 (both crowded). A macro-average would put 0.0 among them.
    assert low == -1.0
    assert high == 1.0


def test_one_cluster_yields_no_interval() -> None:
    """With one task there is no between-task variation left to resample.

    An interval built from it would describe one scenario while reading as a statement
    about the judge, which is the failure this whole function exists to avoid.
    """
    assert bootstrap_clustered_mean_ci({"only": [0.5, 0.5, 1.0]}).interval is None
    assert bootstrap_clustered_mean_ci({}).interval is None
    # An empty task is not a cluster: it contributes no value to resample.
    assert bootstrap_clustered_mean_ci({"only": [0.5], "empty": []}).interval is None


def test_a_constant_effect_collapses_the_interval_onto_itself() -> None:
    """Every task moved by exactly -0.5, so every resample does too."""
    groups = {f"task-{position}": [-0.5, -0.5] for position in range(5)}

    result = bootstrap_clustered_mean_ci(groups)

    assert result.interval == (-0.5, -0.5)
    # A mean is defined for every resample, unlike kappa, so this count carries no
    # quality signal here — it is reported for consistency with the kappa tables.
    assert result.usable == result.iterations == 1000


def test_the_same_groups_and_seed_always_give_the_same_interval() -> None:
    """A report whose interval moved between refreshes would be argued with, not read."""
    groups = {"a": [0.0, 0.5], "b": [-0.5], "c": [1.0, 0.5, 0.0], "d": [0.25]}

    first = bootstrap_clustered_mean_ci(groups, seed=3)
    second = bootstrap_clustered_mean_ci(dict(reversed(list(groups.items()))), seed=3)

    assert first.interval == second.interval
    assert bootstrap_clustered_mean_ci(groups, seed=4).interval != first.interval


def test_a_bootstrap_that_cannot_run_returns_no_interval() -> None:
    groups = {"a": [0.0], "b": [1.0]}

    assert bootstrap_clustered_mean_ci(groups, iterations=0).interval is None
    assert bootstrap_clustered_mean_ci(groups, confidence=1.0).interval is None


# --- adjusted_confidence -------------------------------------------------------


def test_sidak_raises_the_per_interval_level_rather_than_lowering_it() -> None:
    """0.95 over four arms is 0.9873 per interval, not 0.5273.

    Sidak is stated on the error rate: the adjusted alpha is 1 - (1 - alpha)**(1/m).
    Applying that alpha-shaped expression to a confidence returns 0.527 here — below
    the nominal level, printed beside 0.95 as though the run had grown *less* certain
    of each arm, which is the opposite of a correction.
    """
    adjusted = adjusted_confidence(0.95, 4)

    assert adjusted == pytest.approx(0.95**0.25)
    assert adjusted == pytest.approx(0.98726, abs=5e-6)
    assert adjusted > 0.95
    assert adjusted != pytest.approx(1 - (1 - 0.95) ** (1 / 4))


def test_one_interval_needs_no_correction() -> None:
    assert adjusted_confidence(0.95, 1) == 0.95
    assert adjusted_confidence(0.95, 0) == 0.95
    assert adjusted_confidence(1.0, 4) == 1.0


def test_more_intervals_demand_a_stricter_reading_of_each() -> None:
    """Seven uncorrected 95% intervals give a clean judge a 30% chance of looking biased."""
    levels = [adjusted_confidence(0.95, intervals) for intervals in (1, 2, 4, 7)]

    assert levels == sorted(levels)
    assert levels[-1] > 0.99
