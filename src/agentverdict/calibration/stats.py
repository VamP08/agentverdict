"""Inter-rater agreement statistics, implemented directly on the standard library.

The arithmetic here is small but easy to get subtly wrong, so it is written out
rather than imported: chance-corrected agreement needs explicit decisions about
undefined cases (a rater who never varies, an empty comparison), and the ordinal
structure of ``fail < borderline < pass`` calls for a weighted variant that treats
a pass/fail mix-up as worse than a pass/borderline one. Keeping it dependency-free
also keeps the install light enough to run inside a CI gate.

Every function takes ``pairs`` — ``(reference, comparison)`` verdict tuples, one
per item, conventionally (human, judge) — and returns ``None`` rather than a
misleading number when a statistic is undefined.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Verdict vocabulary, ordered worst to best. The ordering is what makes the
#: weighted kappa meaningful; the unweighted statistics ignore it.
VERDICT_ORDER: tuple[str, ...] = ("fail", "borderline", "pass")

Pair = tuple[str, str]


@dataclass(frozen=True)
class ClassMetrics:
    """Per-verdict breakdown, treating the reference rater as ground truth."""

    label: str
    support: int  # times the reference used this label
    predicted: int  # times the comparison used this label
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap interval plus how much of the bootstrap actually worked.

    Resamples that fail to draw a varying rater have an undefined kappa and are
    discarded. That discard is not random — it removes exactly the resamples that
    would have shown the estimate resting on one or two items — so the count is
    reported alongside the interval instead of being swallowed.
    """

    interval: tuple[float, float] | None
    usable: int
    iterations: int


@dataclass(frozen=True)
class AgreementStats:
    """Everything worth knowing about how two raters compare on the same items."""

    n: int
    labels: tuple[str, ...]
    matrix: dict[str, dict[str, int]]  # matrix[reference][comparison] = count
    observed_agreement: float | None
    cohens_kappa: float | None
    weighted_kappa: float | None
    kappa_ci: tuple[float, float] | None
    per_class: tuple[ClassMetrics, ...]
    bootstrap_usable: int = 0
    bootstrap_iterations: int = 0

    @property
    def interpretation(self) -> str:
        return interpret_kappa(self.cohens_kappa)


def _clean(pairs: Iterable[Pair], labels: Sequence[str]) -> list[Pair]:
    """Drop pairs whose verdicts fall outside the known vocabulary."""
    allowed = set(labels)
    return [(a, b) for a, b in pairs if a in allowed and b in allowed]


def confusion_matrix(pairs: Iterable[Pair], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Counts indexed ``matrix[reference_label][comparison_label]``."""
    matrix = {row: dict.fromkeys(labels, 0) for row in labels}
    for reference, comparison in _clean(pairs, labels):
        matrix[reference][comparison] += 1
    return matrix


def observed_agreement(
    pairs: Iterable[Pair], labels: Sequence[str] = VERDICT_ORDER
) -> float | None:
    """Raw fraction of items both raters labeled identically."""
    cleaned = _clean(pairs, labels)
    if not cleaned:
        return None
    return sum(1 for a, b in cleaned if a == b) / len(cleaned)


def _marginals(
    matrix: dict[str, dict[str, int]], labels: Sequence[str], total: int
) -> tuple[dict[str, float], dict[str, float]]:
    reference = {row: sum(matrix[row].values()) / total for row in labels}
    comparison = {col: sum(matrix[row][col] for row in labels) / total for col in labels}
    return reference, comparison


def _is_degenerate(
    reference: dict[str, float], comparison: dict[str, float]
) -> bool:
    """True when either rater used a single label for everything.

    Kappa asks how much two raters agree beyond chance, and a rater with no
    variance has no chance model: the arithmetic collapses to exactly 0.0 no
    matter how well the other rater did. A judge that matched a constant human on
    19 of 20 items would score 0.000 — "slight agreement" — which is worse than
    useless. Such a comparison has no answer, so it returns ``None``.
    """
    return max(reference.values(), default=0.0) >= 1.0 or (
        max(comparison.values(), default=0.0) >= 1.0
    )


def cohens_kappa(pairs: Iterable[Pair], labels: Sequence[str] = VERDICT_ORDER) -> float | None:
    """Chance-corrected agreement.

    ``None`` when undefined: no comparable items, or either rater never varied
    (see ``_is_degenerate``) — the case where the formula would otherwise return a
    confident-looking 0.0 that means nothing.
    """
    cleaned = _clean(pairs, labels)
    total = len(cleaned)
    if total == 0:
        return None
    matrix = confusion_matrix(cleaned, labels)
    reference, comparison = _marginals(matrix, labels, total)
    if _is_degenerate(reference, comparison):
        return None
    p_observed = sum(matrix[label][label] for label in labels) / total
    p_expected = sum(reference[label] * comparison[label] for label in labels)
    if p_expected >= 1.0:
        return None
    return (p_observed - p_expected) / (1.0 - p_expected)


def weighted_kappa(
    pairs: Iterable[Pair],
    labels: Sequence[str] = VERDICT_ORDER,
    weighting: str = "linear",
) -> float | None:
    """Kappa that penalises disagreements by how far apart the verdicts are.

    With ``fail < borderline < pass``, calling a human "pass" a "fail" is a worse
    error than calling it "borderline"; unweighted kappa treats both identically.
    ``weighting`` is ``"linear"`` or ``"quadratic"``.
    """
    if weighting not in ("linear", "quadratic"):
        raise ValueError(f"unknown weighting {weighting!r}; use 'linear' or 'quadratic'")
    cleaned = _clean(pairs, labels)
    total = len(cleaned)
    if total == 0 or len(labels) < 2:
        return None
    index = {label: position for position, label in enumerate(labels)}
    span = len(labels) - 1

    def distance(a: str, b: str) -> float:
        raw = abs(index[a] - index[b]) / span
        return raw**2 if weighting == "quadratic" else raw

    matrix = confusion_matrix(cleaned, labels)
    reference, comparison = _marginals(matrix, labels, total)
    if _is_degenerate(reference, comparison):
        return None
    observed_cost = 0.0
    expected_cost = 0.0
    for row in labels:
        for col in labels:
            weight = distance(row, col)
            observed_cost += weight * matrix[row][col] / total
            expected_cost += weight * reference[row] * comparison[col]
    if expected_cost == 0.0:
        return None
    return 1.0 - observed_cost / expected_cost


#: Share of resamples that must yield a defined kappa before an interval is
#: reported. Discards are concentrated on the resamples that miss a rare class, so
#: a high discard rate means the interval describes a filtered, flattering subset.
MIN_USABLE_FRACTION = 0.9


def bootstrap_kappa(
    pairs: Iterable[Pair],
    labels: Sequence[str] = VERDICT_ORDER,
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap interval for Cohen's kappa, with its discard count.

    Items are resampled with replacement, so the interval reflects how much the
    estimate depends on which trajectories happen to be labeled — the honest way
    to report kappa from a golden set of a few dozen items. Seeded, so a report is
    reproducible.

    A resample in which either rater never varies has no defined kappa and cannot
    be counted. Those failures cluster on samples dominated by one verdict, so
    dropping them quietly would turn "almost every resample was degenerate" into a
    tight, confident-looking interval. The interval is therefore withheld unless
    at least ``MIN_USABLE_FRACTION`` of resamples were usable, and the count is
    returned either way so the caller can say why.
    """
    cleaned = _clean(pairs, labels)
    total = len(cleaned)
    if total < 2 or not 0.0 < confidence < 1.0:
        return BootstrapResult(interval=None, usable=0, iterations=iterations)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        resample = [cleaned[rng.randrange(total)] for _ in range(total)]
        estimate = cohens_kappa(resample, labels)
        if estimate is not None:
            estimates.append(estimate)
    usable = len(estimates)
    if usable < MIN_USABLE_FRACTION * iterations:
        return BootstrapResult(interval=None, usable=usable, iterations=iterations)
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    last = usable - 1
    low = estimates[int(tail * last)]
    high = estimates[min(last, int(round((1.0 - tail) * last)))]
    return BootstrapResult(interval=(low, high), usable=usable, iterations=iterations)


def bootstrap_kappa_ci(
    pairs: Iterable[Pair],
    labels: Sequence[str] = VERDICT_ORDER,
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """The interval alone, for callers that do not need the discard count."""
    return bootstrap_kappa(
        pairs, labels, iterations=iterations, confidence=confidence, seed=seed
    ).interval


def per_class_metrics(
    pairs: Iterable[Pair], labels: Sequence[str] = VERDICT_ORDER
) -> tuple[ClassMetrics, ...]:
    """Precision, recall, and F1 per verdict, with the reference rater as truth."""
    cleaned = _clean(pairs, labels)
    matrix = confusion_matrix(cleaned, labels)
    metrics: list[ClassMetrics] = []
    for label in labels:
        hits = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[row][label] for row in labels)
        precision = hits / predicted if predicted else None
        recall = hits / support if support else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics.append(
            ClassMetrics(
                label=label,
                support=support,
                predicted=predicted,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return tuple(metrics)


def interpret_kappa(value: float | None) -> str:
    """Landis & Koch's conventional bands, for readers who do not live in kappa."""
    if value is None:
        return "not enough data"
    if value < 0.0:
        return "worse than chance"
    if value <= 0.20:
        return "slight"
    if value <= 0.40:
        return "fair"
    if value <= 0.60:
        return "moderate"
    if value <= 0.80:
        return "substantial"
    return "almost perfect"


def agreement_stats(
    pairs: Iterable[Pair],
    labels: Sequence[str] = VERDICT_ORDER,
    *,
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> AgreementStats:
    """Compute every agreement statistic for one pair of raters in a single pass."""
    cleaned = _clean(pairs, labels)
    bootstrap = bootstrap_kappa(cleaned, labels, iterations=bootstrap_iterations, seed=seed)
    return AgreementStats(
        n=len(cleaned),
        labels=tuple(labels),
        matrix=confusion_matrix(cleaned, labels),
        observed_agreement=observed_agreement(cleaned, labels),
        cohens_kappa=cohens_kappa(cleaned, labels),
        weighted_kappa=weighted_kappa(cleaned, labels),
        kappa_ci=bootstrap.interval,
        per_class=per_class_metrics(cleaned, labels),
        bootstrap_usable=bootstrap.usable,
        bootstrap_iterations=bootstrap.iterations,
    )
