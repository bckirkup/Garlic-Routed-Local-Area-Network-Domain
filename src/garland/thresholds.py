"""Degrees-of-freedom calibration for Mahalanobis anomaly thresholds.

A Mahalanobis distance threshold is only interpretable together with the width
of the observation vector it scores. Under the null, the squared distance of a
``d``-channel residual is approximately chi-square with ``d`` degrees of
freedom, so a fixed cut of 3.5 that flags 1.6% of quiet epochs at four channels
flags far more of them at fourteen. A fleet whose members own different sensor
bundles would then have a per-agent false-positive rate set by what they bought
rather than by how sick they are.

This module converts a threshold to the per-epoch false-positive rate it
implies and back again, so a configured threshold can be read as a *rate* and
re-expressed at whatever width an agent actually reports this epoch. The
chi-square tail is evaluated directly (regularized incomplete gamma) to avoid a
hard SciPy dependency; SciPy is optional in this project.

See ``docs/BIOMETRICS.md``.
"""

from __future__ import annotations

import math
from functools import lru_cache

# Width the historical threshold was chosen for: HR, HRV, RR, temperature.
REFERENCE_DOF = 4

# Iteration limits for the incomplete-gamma evaluations. Both series converge
# geometrically for the arguments used here; the caps only bound pathological
# input.
_MAX_ITERATIONS = 300
_RELATIVE_TOLERANCE = 1e-14
_TINY = 1e-300


def _lower_regularized_gamma(a: float, x: float) -> float:
    """P(a, x) by its power series, accurate for ``x < a + 1``."""
    if x <= 0.0:
        return 0.0
    term = 1.0 / a
    total = term
    for n in range(1, _MAX_ITERATIONS):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * _RELATIVE_TOLERANCE:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_regularized_gamma(a: float, x: float) -> float:
    """Q(a, x) by its continued fraction, accurate for ``x >= a + 1``."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for n in range(1, _MAX_ITERATIONS):
        an = -n * (n - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _RELATIVE_TOLERANCE:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_survival(statistic: float, dof: int) -> float:
    """Return P(chi-square with ``dof`` degrees of freedom > ``statistic``)."""
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if statistic <= 0.0:
        return 1.0
    a = 0.5 * dof
    x = 0.5 * statistic
    if x < a + 1.0:
        return 1.0 - _lower_regularized_gamma(a, x)
    return _upper_regularized_gamma(a, x)


def chi_square_upper_quantile(tail_probability: float, dof: int) -> float:
    """Return the statistic whose chi-square upper tail is ``tail_probability``."""
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly between 0 and 1")
    low = 0.0
    high = float(max(dof, 1))
    while chi_square_survival(high, dof) > tail_probability:
        high *= 2.0
        if high > 1e12:
            raise ValueError("tail_probability is too small to invert")
    # The survival function is strictly decreasing, so plain bisection is both
    # sufficient and immune to the overshoot a Newton step can suffer far out
    # in the tail.
    for _ in range(200):
        middle = 0.5 * (low + high)
        if chi_square_survival(middle, dof) > tail_probability:
            low = middle
        else:
            high = middle
        if high - low < 1e-12 * max(high, 1.0):
            break
    return 0.5 * (low + high)


def per_epoch_false_positive_rate(threshold: float, dof: int = REFERENCE_DOF) -> float:
    """Fraction of quiet epochs a distance ``threshold`` flags at ``dof`` channels."""
    return chi_square_survival(threshold * threshold, dof)


def calibrated_threshold(dof: int, per_epoch_fpr: float) -> float:
    """Distance threshold that flags ``per_epoch_fpr`` of quiet ``dof``-wide epochs."""
    return math.sqrt(chi_square_upper_quantile(per_epoch_fpr, dof))


@lru_cache(maxsize=None)
def chi_mean(dof: int) -> float:
    """Mean Mahalanobis *distance* under the null at ``dof`` channels.

    ``E[chi_dof] = sqrt(2) * Gamma((dof + 1) / 2) / Gamma(dof / 2)``, which grows
    like ``sqrt(dof)``. A CUSUM slack chosen for four channels is therefore below
    the resting mean of a wider vector, and the statistic would ramp without any
    hazard present.
    """
    if dof < 1:
        raise ValueError("dof must be at least 1")
    return math.sqrt(2.0) * math.exp(math.lgamma(0.5 * (dof + 1)) - math.lgamma(0.5 * dof))


@lru_cache(maxsize=None)
def reference_value_for_dof(
    reference_slack: float,
    dof: int,
    reference_dof: int = REFERENCE_DOF,
) -> float:
    """Re-express a CUSUM slack at ``dof`` channels, holding its null margin fixed."""
    if dof == reference_dof:
        return reference_slack
    return reference_slack * chi_mean(dof) / chi_mean(reference_dof)


@lru_cache(maxsize=None)
def threshold_for_dof(
    reference_threshold: float,
    dof: int,
    reference_dof: int = REFERENCE_DOF,
) -> float:
    """Re-express ``reference_threshold`` at ``dof`` channels, holding its rate fixed.

    ``reference_threshold`` is interpreted as a cut chosen for ``reference_dof``
    channels; the return value is the cut with the same null-tail probability at
    ``dof`` channels. Widening the vector therefore does not by itself change
    how often an agent alarms.
    """
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if dof == reference_dof:
        return reference_threshold
    if reference_threshold <= 0.0:
        # A non-positive cut flags every epoch at any width; there is no rate
        # to preserve.
        return reference_threshold
    return calibrated_threshold(
        dof, per_epoch_false_positive_rate(reference_threshold, reference_dof)
    )


__all__ = [
    "REFERENCE_DOF",
    "calibrated_threshold",
    "chi_mean",
    "chi_square_survival",
    "chi_square_upper_quantile",
    "per_epoch_false_positive_rate",
    "reference_value_for_dof",
    "threshold_for_dof",
]
