"""Hand-computed checks on the agreement statistics.

Every expectation here is worked out arithmetically in a comment before it is
asserted. Kappa is exactly the kind of statistic that stays plausible while being
wrong — a transposed matrix, a missing chance correction, or a zero substituted
for an undefined value all still produce a number between -1 and 1 — so "is not
None" assertions would buy nothing. The tolerance is only there for float noise.
"""

import pytest

from agentverdict.calibration.stats import (
    VERDICT_ORDER,
    agreement_stats,
    bootstrap_kappa_ci,
    cohens_kappa,
    confusion_matrix,
    interpret_kappa,
    observed_agreement,
    per_class_metrics,
    weighted_kappa,
)


def _pairs(*groups: tuple[int, str, str]) -> list[tuple[str, str]]:
    """Expand ``(count, reference, comparison)`` triples into a list of pairs."""
    pairs: list[tuple[str, str]] = []
    for count, reference, comparison in groups:
        pairs.extend([(reference, comparison)] * count)
    return pairs


#: 8 pass/pass, 2 pass/fail, 7 fail/fail, 3 fail/pass — the worked example (n = 20).
MIXED = _pairs((8, "pass", "pass"), (2, "pass", "fail"), (7, "fail", "fail"), (3, "fail", "pass"))


# --- the vocabulary ----------------------------------------------------------


def test_verdict_order_is_ordinal_worst_to_best() -> None:
    # The weighted kappa's distances only mean anything if this order holds.
    assert VERDICT_ORDER == ("fail", "borderline", "pass")


# --- Cohen's kappa -----------------------------------------------------------


def test_perfect_agreement_gives_kappa_one() -> None:
    # 5 pass/pass + 5 fail/fail. p_o = 1.0; both marginals are 0.5/0.5, so
    # p_e = 0.5*0.5 + 0.5*0.5 = 0.5; kappa = (1.0 - 0.5) / (1.0 - 0.5) = 1.0.
    pairs = _pairs((5, "pass", "pass"), (5, "fail", "fail"))
    assert observed_agreement(pairs) == pytest.approx(1.0)
    assert cohens_kappa(pairs) == pytest.approx(1.0)
    assert weighted_kappa(pairs) == pytest.approx(1.0)


def test_worked_mixed_case() -> None:
    # n = 20. p_o = (8 + 7) / 20 = 0.75.
    # human marginals: pass 10/20 = 0.50, fail 10/20 = 0.50.
    # judge marginals: pass (8+3)/20 = 0.55, fail (2+7)/20 = 0.45.
    # p_e = 0.50*0.55 + 0.50*0.45 = 0.275 + 0.225 = 0.50.
    # kappa = (0.75 - 0.50) / (1.0 - 0.50) = 0.25 / 0.50 = 0.5.
    assert observed_agreement(MIXED) == pytest.approx(0.75)
    assert cohens_kappa(MIXED) == pytest.approx(0.5)
    # Only pass/fail flips here, and every off-diagonal cell carries the same
    # weight (1.0), so the linear weighting cannot separate them: still 0.5.
    assert weighted_kappa(MIXED) == pytest.approx(0.5)


def test_chance_level_agreement_gives_kappa_zero() -> None:
    # A perfectly independent 2x2: 4 in every cell. p_o = 8/16 = 0.5; both
    # marginals 0.5/0.5 so p_e = 0.5; kappa = (0.5 - 0.5) / 0.5 = 0.0 exactly.
    # 50% raw agreement that is worth nothing once chance is accounted for.
    pairs = _pairs(
        (4, "pass", "pass"), (4, "pass", "fail"), (4, "fail", "pass"), (4, "fail", "fail")
    )
    assert observed_agreement(pairs) == pytest.approx(0.5)
    assert cohens_kappa(pairs) == pytest.approx(0.0, abs=1e-12)
    assert interpret_kappa(cohens_kappa(pairs)) == "slight"


def test_systematic_disagreement_gives_negative_kappa() -> None:
    # Every verdict flipped: 5 pass/fail + 5 fail/pass. p_o = 0.0;
    # p_e = 0.5*0.5 + 0.5*0.5 = 0.5; kappa = (0.0 - 0.5) / 0.5 = -1.0.
    pairs = _pairs((5, "pass", "fail"), (5, "fail", "pass"))
    assert cohens_kappa(pairs) == pytest.approx(-1.0)
    assert weighted_kappa(pairs) == pytest.approx(-1.0)
    assert interpret_kappa(cohens_kappa(pairs)) == "worse than chance"


def test_constant_identical_raters_are_undefined_not_perfect() -> None:
    # Both raters said "pass" every time: p_e = 1.0*1.0 = 1.0, so the chance
    # correction divides by zero. Raw agreement is a real 1.0, but kappa must be
    # None — reporting 1.0 would claim a perfectly calibrated judge from a rater
    # that never varies, which is the single most misleading number available.
    pairs = _pairs((5, "pass", "pass"))
    assert observed_agreement(pairs) == pytest.approx(1.0)
    assert cohens_kappa(pairs) is None
    assert weighted_kappa(pairs) is None
    assert interpret_kappa(cohens_kappa(pairs)) == "not enough data"


def test_empty_input_is_none_everywhere() -> None:
    assert observed_agreement([]) is None
    assert cohens_kappa([]) is None
    assert weighted_kappa([]) is None
    assert bootstrap_kappa_ci([]) is None

    stats = agreement_stats([])
    assert stats.n == 0
    assert stats.labels == VERDICT_ORDER
    assert stats.observed_agreement is None
    assert stats.cohens_kappa is None
    assert stats.weighted_kappa is None
    assert stats.kappa_ci is None
    assert stats.interpretation == "not enough data"
    assert stats.matrix == {row: dict.fromkeys(VERDICT_ORDER, 0) for row in VERDICT_ORDER}
    assert [metrics.label for metrics in stats.per_class] == list(VERDICT_ORDER)
    assert all(metrics.support == 0 and metrics.predicted == 0 for metrics in stats.per_class)
    assert all(
        metrics.precision is None and metrics.recall is None and metrics.f1 is None
        for metrics in stats.per_class
    )


def test_unknown_verdicts_are_dropped_not_fatal() -> None:
    # Only ("pass", "pass") and ("fail", "fail") use the known vocabulary; the
    # rest — a stray label, an empty string, a case mismatch — are dropped on
    # both sides, so n is 2 and the kappa is computed on those two alone.
    dirty = [
        ("pass", "pass"),
        ("pass", "excellent"),  # judge invented a verdict
        ("unsure", "fail"),  # annotator invented a verdict
        ("pass", ""),  # empty string, not a verdict
        ("PASS", "pass"),  # matching is case sensitive
        ("fail", "fail"),
    ]
    stats = agreement_stats(dirty)
    assert stats.n == 2
    assert stats.matrix["pass"]["pass"] == 1
    assert stats.matrix["fail"]["fail"] == 1
    assert sum(sum(row.values()) for row in stats.matrix.values()) == 2
    assert observed_agreement(dirty) == pytest.approx(1.0)
    # p_o = 1.0, p_e = 0.5*0.5 + 0.5*0.5 = 0.5, kappa = 1.0 on the surviving two.
    assert cohens_kappa(dirty) == pytest.approx(1.0)


# --- confusion matrix orientation --------------------------------------------


def test_matrix_is_indexed_reference_then_comparison() -> None:
    # Deliberately asymmetric: 3 trajectories the humans passed and the judge
    # failed, 1 the humans failed and the judge passed. A transposed
    # implementation would swap these two counts and still look reasonable.
    pairs = _pairs((3, "pass", "fail"), (1, "fail", "pass"))
    matrix = confusion_matrix(pairs, VERDICT_ORDER)
    assert matrix["pass"]["fail"] == 3
    assert matrix["fail"]["pass"] == 1
    assert matrix["pass"]["pass"] == 0
    assert matrix["borderline"] == {"fail": 0, "borderline": 0, "pass": 0}
    # agreement_stats must expose the same orientation it computed on.
    assert agreement_stats(pairs).matrix["pass"]["fail"] == 3


# --- weighted kappa ----------------------------------------------------------


ADJACENT = _pairs(
    (5, "pass", "pass"),
    (5, "borderline", "borderline"),
    (5, "fail", "fail"),
    (5, "pass", "borderline"),  # the only disagreement, and it is one step wide
)


def test_weighted_kappa_beats_plain_kappa_on_adjacent_disagreements() -> None:
    # n = 20, p_o = 15/20 = 0.75.
    # human marginals: fail 0.25, borderline 0.25, pass 0.50.
    # judge marginals: fail 0.25, borderline 0.50, pass 0.25.
    # p_e = 0.25*0.25 + 0.25*0.50 + 0.50*0.25 = 0.0625 + 0.125 + 0.125 = 0.3125.
    # kappa = (0.75 - 0.3125) / 0.6875 = 0.4375 / 0.6875 = 7/11 = 0.636363...
    assert cohens_kappa(ADJACENT) == pytest.approx(7 / 11)

    # Linear weights (distance / 2): pass-vs-borderline costs 0.5, pass-vs-fail 1.0.
    # observed cost = 5 * 0.5 / 20 = 0.125.
    # expected cost = 0.25*(0.5*0.50 + 1.0*0.25)      fail row      = 0.1250
    #               + 0.25*(0.5*0.25 + 0.5*0.25)      borderline    = 0.0625
    #               + 0.50*(1.0*0.25 + 0.5*0.50)      pass row      = 0.2500
    #               = 0.4375
    # weighted kappa = 1 - 0.125 / 0.4375 = 5/7 = 0.714285...
    linear = weighted_kappa(ADJACENT, weighting="linear")
    assert linear == pytest.approx(5 / 7)
    # The whole point of the ordinal weighting: near-misses are forgiven, so the
    # weighted figure sits above the unweighted one.
    assert linear > cohens_kappa(ADJACENT)
    assert weighted_kappa(ADJACENT) == pytest.approx(linear)  # linear is the default


def test_quadratic_weighting_differs_from_linear() -> None:
    # Squared weights: pass-vs-borderline 0.25, pass-vs-fail 1.0.
    # observed cost = 5 * 0.25 / 20 = 0.0625.
    # expected cost = 0.25*(0.25*0.50 + 1.0*0.25)     fail row      = 0.09375
    #               + 0.25*(0.25*0.25 + 0.25*0.25)    borderline    = 0.03125
    #               + 0.50*(1.0*0.25 + 0.25*0.50)     pass row      = 0.18750
    #               = 0.3125
    # weighted kappa = 1 - 0.0625 / 0.3125 = 0.8.
    quadratic = weighted_kappa(ADJACENT, weighting="quadratic")
    assert quadratic == pytest.approx(0.8)
    assert quadratic != pytest.approx(weighted_kappa(ADJACENT, weighting="linear"))


def test_unknown_weighting_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown weighting"):
        weighted_kappa(ADJACENT, weighting="cubic")


# --- per-class metrics -------------------------------------------------------


def test_per_class_metrics_hand_computed() -> None:
    # 3 borderline called pass, 4 pass called pass, 3 fail called fail (n = 10).
    pairs = _pairs((3, "borderline", "pass"), (4, "pass", "pass"), (3, "fail", "fail"))
    metrics = {row.label: row for row in per_class_metrics(pairs)}
    assert set(metrics) == set(VERDICT_ORDER)

    # fail: 3 hits, support 3, predicted 3 -> precision 1.0, recall 1.0, F1 1.0.
    assert metrics["fail"].support == 3
    assert metrics["fail"].predicted == 3
    assert metrics["fail"].precision == pytest.approx(1.0)
    assert metrics["fail"].recall == pytest.approx(1.0)
    assert metrics["fail"].f1 == pytest.approx(1.0)

    # borderline: the judge never once said "borderline". Precision is 0/0, which
    # is undefined, not 0.0 — a judge that never uses a label has not earned a
    # bad precision score for it, and F1 cannot be formed either. Recall is a
    # genuine 0.0: three borderline trajectories existed and none were caught.
    assert metrics["borderline"].support == 3
    assert metrics["borderline"].predicted == 0
    assert metrics["borderline"].precision is None
    assert metrics["borderline"].recall == pytest.approx(0.0)
    assert metrics["borderline"].f1 is None

    # pass: 4 hits, support 4, predicted 7 -> precision 4/7, recall 1.0,
    # F1 = 2 * (4/7) * 1 / (4/7 + 1) = (8/7) / (11/7) = 8/11.
    assert metrics["pass"].support == 4
    assert metrics["pass"].predicted == 7
    assert metrics["pass"].precision == pytest.approx(4 / 7)
    assert metrics["pass"].recall == pytest.approx(1.0)
    assert metrics["pass"].f1 == pytest.approx(8 / 11)


def test_per_class_metrics_on_the_mixed_case() -> None:
    metrics = {row.label: row for row in per_class_metrics(MIXED)}
    # fail: 7 hits, support 10, predicted 9 -> precision 7/9, recall 0.7,
    # F1 = 2*(7/9)*(7/10) / (7/9 + 7/10) = (98/90) / (133/90) = 98/133.
    assert metrics["fail"].precision == pytest.approx(7 / 9)
    assert metrics["fail"].recall == pytest.approx(0.7)
    assert metrics["fail"].f1 == pytest.approx(98 / 133)
    # pass: 8 hits, support 10, predicted 11 -> precision 8/11, recall 0.8,
    # F1 = 2*(8/11)*(8/10) / (8/11 + 8/10) = (128/110) / (168/110) = 128/168.
    assert metrics["pass"].precision == pytest.approx(8 / 11)
    assert metrics["pass"].recall == pytest.approx(0.8)
    assert metrics["pass"].f1 == pytest.approx(128 / 168)
    # borderline never appeared on either side: nothing to score at all.
    assert metrics["borderline"].support == 0
    assert metrics["borderline"].predicted == 0
    assert metrics["borderline"].precision is None
    assert metrics["borderline"].recall is None
    assert metrics["borderline"].f1 is None


# --- bootstrap interval ------------------------------------------------------


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed() -> None:
    first = bootstrap_kappa_ci(MIXED, seed=7)
    second = bootstrap_kappa_ci(MIXED, seed=7)
    assert first is not None
    assert first == second  # exact equality: a report must not drift between runs
    assert agreement_stats(MIXED, seed=7).kappa_ci == first


def test_bootstrap_ci_depends_on_the_seed() -> None:
    # Different resamples must actually produce a different interval; an
    # implementation that ignored the seed would return the same tuple twice.
    assert bootstrap_kappa_ci(MIXED, seed=7) != bootstrap_kappa_ci(MIXED, seed=99)


@pytest.mark.parametrize("seed", [7, 99])
def test_bootstrap_ci_brackets_the_point_estimate(seed: int) -> None:
    # The point estimate on MIXED is exactly 0.5 (worked above); a percentile
    # interval over resamples of a clean sample has to contain it.
    interval = bootstrap_kappa_ci(MIXED, seed=seed)
    assert interval is not None
    low, high = interval
    assert low < high
    assert low <= 0.5 <= high
    assert low >= -1.0
    assert high <= 1.0


def test_bootstrap_ci_is_none_when_there_is_nothing_to_resample() -> None:
    assert bootstrap_kappa_ci([("pass", "pass")]) is None  # n = 1
    assert bootstrap_kappa_ci([]) is None  # n = 0
    # Two items, but every resample of a constant rater has an undefined kappa.
    assert bootstrap_kappa_ci(_pairs((5, "pass", "pass"))) is None


# --- interpretation bands ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "not enough data"),
        (-1.0, "worse than chance"),
        (-0.001, "worse than chance"),
        (0.0, "slight"),  # exactly zero is chance-level, not "worse than chance"
        (0.20, "slight"),  # boundaries are inclusive at the top of each band
        (0.201, "fair"),
        (0.40, "fair"),
        (0.401, "moderate"),
        (0.60, "moderate"),
        (0.601, "substantial"),
        (0.80, "substantial"),
        (0.801, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_interpret_kappa_bands(value: float | None, expected: str) -> None:
    assert interpret_kappa(value) == expected


def test_agreement_stats_interpretation_matches_its_own_kappa() -> None:
    stats = agreement_stats(MIXED)
    assert stats.n == 20
    assert stats.cohens_kappa == pytest.approx(0.5)
    assert stats.interpretation == "moderate"  # 0.5 sits in (0.40, 0.60]
