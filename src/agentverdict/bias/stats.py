"""Statistics for the bias probes, on the standard library alone.

A probe compares the judge against *itself*: one trajectory is judged several times
under an unperturbed prompt and several times under a perturbed one, and the
question is whether the perturbation moved the verdict further than the judge's own
noise already moves it. That framing is why these functions live beside the probe
instead of in ``calibration.stats`` — that module compares two raters over items,
this one compares two arms over repeats of a single item, and the undefined cases
are different ones.

Three things are shared with ``calibration.stats`` rather than restated in slightly
different words: the verdict vocabulary (``VERDICT_ORDER``), the ordinal scores
(``VERDICT_SCORES``), and the habit of returning ``None`` instead of a
plausible-looking number when a statistic has no value. A judge call that errored
arrives here as ``None`` and is dropped, so every denominator counts only verdicts
that exist; the caller reports the error count separately, because a probe that
failed on its hardest transcripts must not read as a quiet, tidy result.

Nothing here calls a model, and the bootstrap is seeded, so the same draws always
produce the same numbers.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agentverdict.calibration.stats import VERDICT_ORDER, VERDICT_SCORES, BootstrapResult

#: One arm's judgments of one trajectory. ``None`` is a call that errored.
Draws = Iterable[str | None]


@dataclass(frozen=True)
class ArmSamples:
    """One trajectory's judgments in one arm.

    ``task_key`` travels with the sample because it is the resampling unit:
    trajectories replayed from the same task are not independent draws on the judge,
    and ``bootstrap_clustered_mean_ci`` has to be told which is which.
    """

    trajectory_id: str
    task_key: str
    verdicts: tuple[str | None, ...]  # None = the judge call errored


def _usable(draws: Draws) -> list[str]:
    """Keep the draws that carry a known verdict, in the order they were made."""
    return [draw for draw in draws if draw in VERDICT_ORDER]


def disagreement_within(verdicts: Draws) -> float | None:
    """Probability that two distinct draws of this arm differ — the noise floor.

    Counted over the C(n, 2) *distinct unordered* pairs. Self-pairs are excluded
    because a draw agrees with itself with probability one, so including them is the
    plug-in estimator ``1 - sum p_v ** 2``, whose expectation is the true
    disagreement times ``(1 - 1/n)``. It understates the floor by floor/n at every
    arm size, and because that floor is the yardstick every perturbed arm is measured
    against, the error is not symmetric: it biases every probe toward finding bias.
    At n=5 that is a fifth of the floor — enough to manufacture a +0.08 shift on
    every perturbed arm of a judge whose floor is 0.4, which is larger than most
    effects worth reporting. On ``["pass", "pass", "borderline", "borderline"]`` this
    returns 2/3; the plug-in returns 0.5, and that gap is the whole point.

    ``None`` below two usable draws: one draw cannot see noise, and 0.0 would be a
    confident-looking number for a statistic that has no value.
    """
    counts = Counter(_usable(verdicts))
    total = sum(counts.values())
    if total < 2:
        return None
    # Ordered pairs above and below the line: halving both gives the unordered
    # C(n,2) form and the same number. The differing count is divided directly
    # rather than subtracted from 1.0, which keeps exact fractions exact -- the
    # pinned 2/3 above is 0.666...67 if this is written as ``1.0 - agreeing/pairs``.
    ordered_pairs = total * (total - 1)
    agreeing = sum(count * (count - 1) for count in counts.values())
    return (ordered_pairs - agreeing) / ordered_pairs


def disagreement_across(control: Draws, variant: Draws) -> float | None:
    """Probability that a control draw and a variant draw differ.

    Counted over ALL control x variant pairs. Repeats within an arm are
    exchangeable — independent samples of the same call, carrying no order beyond
    the order the rows came back in — so no repeat has a privileged partner in the
    other arm. Pairing repeat 1 with repeat 1 and averaging would make the answer
    depend on that arrival order, which disqualifies it as a statistic; it also
    discards most of the comparisons, leaving a granularity of 1/n where the full
    cross product gives 1/n**2 for free.

    Every cross pair joins two independent draws, so this is unbiased at any pair of
    arm sizes, and under the null — the perturbation changed nothing about the
    judge's answer distribution — its expectation is exactly
    ``disagreement_within``'s. That equality in expectation is what makes the
    difference between the two mean anything. In expectation only: on the same
    realised draws the two disagree (``[p, p, p, b, f]`` gives 0.700 within and 0.560
    across), so a test asserting they match on matched counts is testing the wrong
    thing.

    ``None`` when either arm has no usable draw.
    """
    control_counts = Counter(_usable(control))
    variant_counts = Counter(_usable(variant))
    n_control = sum(control_counts.values())
    n_variant = sum(variant_counts.values())
    if not n_control or not n_variant:
        return None
    cross_pairs = n_control * n_variant
    agreeing = sum(control_counts[label] * variant_counts[label] for label in VERDICT_ORDER)
    return (cross_pairs - agreeing) / cross_pairs


def signed_shift(control: Draws, variant: Draws) -> float | None:
    """Mean ordinal score under the variant minus mean under the control — the headline.

    Positive means the perturbation made the judge more generous.

    Verdicts are ordinal (``VERDICT_SCORES``) and every probe's claim is directional
    — "the judge grades a padded transcript higher" — so the headline has to carry a
    sign and a magnitude. A flip rate carries neither, and worse, it answers a
    different question: it responds to the judge's *entropy* rather than to bias.
    Take a control distribution of (pass .8, borderline .1, fail .1) against a
    variant of (pass .7, borderline .3, fail 0). The mode is ``pass`` in both, the
    mean score is 0.850 in both, and no verdict has been graded up or down — the
    judge has only become less decisive. Cross-arm disagreement nevertheless rises
    from 0.340 to 0.410, so a flip-rate headline announces seven points of "bias"
    exactly where the signed shift correctly reports 0.000. A longer prompt is a
    plausible way to make a judge less decisive, so this is the case the length
    probe would meet, not a contrived one.

    Deliberately not pair-based: the mean over all cross pairs of the score
    difference is algebraically the difference of the means, so pairing costs
    O(n**2) work and re-opens the pairing mistake for no gain.

    ``None`` when either arm has no usable draw.
    """
    control_scores = [VERDICT_SCORES[verdict] for verdict in _usable(control)]
    variant_scores = [VERDICT_SCORES[verdict] for verdict in _usable(variant)]
    if not control_scores or not variant_scores:
        return None
    return sum(variant_scores) / len(variant_scores) - sum(control_scores) / len(control_scores)


def min_movers(n: int, confidence: float = 0.95) -> int | None:
    """Smallest number of moved trajectories a percentile bootstrap over ``n`` can see.

    Computable from n alone, before a single token is spent, which is why it is worth
    having: it says in advance what the instrument can and cannot resolve.

    Suppose k trajectories move under the perturbation and the other n-k sit at
    exactly zero. A resample draws n items with replacement, so it misses every mover
    with probability ((n-k)/n)**n, and each of those resamples has a mean of exactly
    zero. The bootstrap distribution therefore carries an atom at zero of that size,
    and the lower tail clears zero only once the atom is smaller than
    (1-confidence)/2. At n=13 the atom is 0.353 for k=1, 0.114 for k=2, 0.033 for k=3
    and 0.0084 for k=4 — so four trajectories must move before any interval at this n
    can exclude zero. Three trajectories flipping under a reordered prompt is
    alarming, and it comes back ``inconclusive`` by construction rather than by
    evidence.

    The part worth internalising is that this barely depends on n. ((n-k)/n)**n tends
    to e**-k, and 0.025 falls between e**-3 = 0.0498 and e**-4 = 0.0183, so the answer
    is 4 at n=13 and still 4 at n=60. Collecting more trajectories lowers the *rate*
    that four movers represents; it does not buy the ability to see a two-trajectory
    effect. Catching a rare-but-real order sensitivity needs a different statistic,
    not a bigger sample, and a probe that cannot see one is expected to say so
    (``power_limited``) rather than print a silent ``inconclusive``.

    Stated for a flat bootstrap. The interval this project actually reports is
    clustered (``bootstrap_clustered_mean_ci``), and a cluster resample misses the
    movers *more* often when they share a task, so this k is a floor on the
    requirement rather than the requirement itself — it can understate how many
    movers are needed, never overstate.

    ``None`` when there is nothing to answer: below two items no interval is reported
    at all, so no number of movers makes one exclude zero.
    """
    if n < 2 or not 0.0 < confidence < 1.0:
        return None
    tail = (1.0 - confidence) / 2.0
    # k = n always satisfies this (the atom is then exactly 0.0), so the default is
    # unreachable for valid n; it is there so the function stays total.
    return next((k for k in range(1, n + 1) if ((n - k) / n) ** n < tail), None)


def bootstrap_clustered_mean_ci(
    groups: Mapping[str, Sequence[float]],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval for a mean, resampling task keys and then trajectories.

    ``groups`` maps a task key to that task's per-trajectory values — one signed
    shift each, typically. Every iteration draws as many task keys as there are
    tasks, with replacement, and then redraws each drawn task's trajectories with
    replacement: a two-stage cluster bootstrap.

    Flat resampling would be wrong here, not merely conservative. Trajectories
    replayed from one task share the task prompt, the expected outcome and the tools
    block — precisely what the order probe permutes — so twenty trajectories from one
    scenario are twenty views of a single draw on the judge, not twenty draws.
    Resampling them flat reports a quirk of one scenario as a property of the judge,
    and does it with an interval narrow enough to be believed. M5's regression gate
    already refuses this ("pairing is by task, never by trajectory"); the probes are
    held to the same rule, and the cluster count is printed beside every result so
    nobody has to guess how much independence is behind an interval.

    Each drawn task contributes its own number of values, so the quantity resampled
    is the same size-weighted mean over trajectories that the report prints. A
    macro-average here would build an interval for an estimator nobody quotes.

    Tasks are visited in sorted key order so a seed reproduces a run regardless of
    how the mapping was built. The interval is ``None`` below two usable clusters:
    with one task there is no between-task variation left to resample, and the result
    would describe one scenario while reading as a statement about the judge.
    ``usable`` equals ``iterations`` whenever an interval comes back — a mean, unlike
    kappa, is defined for every resample, so that field carries no quality signal
    here even though it does in the kappa tables.
    """
    clusters = [list(groups[key]) for key in sorted(groups) if groups[key]]
    cluster_count = len(clusters)
    if cluster_count < 2 or iterations < 1 or not 0.0 < confidence < 1.0:
        return BootstrapResult(interval=None, usable=0, iterations=iterations)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        pool: list[float] = []
        for _ in range(cluster_count):
            drawn = clusters[rng.randrange(cluster_count)]
            size = len(drawn)
            pool.extend(drawn[rng.randrange(size)] for _ in range(size))
        means.append(sum(pool) / len(pool))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    last = iterations - 1
    return BootstrapResult(
        interval=(means[int(tail * last)], means[min(last, int(round((1.0 - tail) * last)))]),
        usable=iterations,
        iterations=iterations,
    )


def adjusted_confidence(confidence: float, intervals: int) -> float:
    """Per-interval confidence after a Sidak correction for ``intervals`` of them.

    A probe run reports one interval per arm. Read at a nominal 95% each, seven of
    them give a judge with no bias whatsoever a 30.2% chance of being called biased
    somewhere. The number of intervals in a run is fixed before it starts, so the
    per-interval level is adjusted for it and each interval is then read at the
    adjusted level: at 95% over 4 arms, 0.9873 per interval, which restores the
    family-wise error to exactly 5%.

    Sidak is stated on the error rate, not on the confidence: the adjusted alpha is
    ``1 - (1 - alpha) ** (1 / intervals)`` for a family alpha of ``1 - confidence``,
    and its complement is the ``confidence ** (1 / intervals)`` returned here.
    Applying the alpha-shaped expression to a confidence instead returns 0.527 for
    the canonical 95%-over-4-arms case — a number below the nominal level, which is
    the opposite of a correction and would be printed beside 0.95 as though the run
    had grown less certain of each arm.

    Sidak rather than Bonferroni because it is exact under independence and slightly
    less punishing. Printed rather than applied silently, because the arms of one run
    are all measured against a shared identity control and are therefore positively
    dependent — which makes Sidak mildly anti-conservative here. A reader shown both
    the nominal and the adjusted level can discount that; a reader shown one number
    cannot.

    Returned unchanged when there is nothing to correct.
    """
    if intervals < 1 or not 0.0 < confidence < 1.0:
        return confidence
    return confidence ** (1.0 / intervals)
