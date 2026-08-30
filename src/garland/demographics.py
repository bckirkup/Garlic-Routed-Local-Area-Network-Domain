"""Age structure of the population and its effect on device ownership.

The fleet used to be demographically flat: every wearable owner drew each
device kind independently at the configured adoption fraction, so ownership
counts were binomial and nobody was an enthusiast. Two things are wrong with
that as a picture of a real fleet. Ownership is *correlated* within a person --
people who buy one sensor buy several -- and it is *age-conditioned*: an infant
plausibly wears a motion band and a respiratory patch chosen by a caregiver but
never lace-up gait shoes, while an older adult is the likely owner of the
cardiac and thoracic hardware and the unlikely owner of a sleep headband.

This module supplies both without changing the fleet-wide marginals. Age bands
are assigned through household composition so infants co-reside with adults and
seniors cluster together, which matters because base-device ownership is
household-patchy. Device ownership then draws the *same* number of owners per
kind as before, but weights who those owners are by an age affinity and a
per-person enthusiasm factor, so the configured adoption fraction still means
what it meant while the distribution across people becomes long-tailed.

Both the affinities and the composition fractions are the author's own
sourcing, chosen for plausible shape rather than fitted to survey data. They
are simulation-testbed calibration, not an adoption forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

INFANT = "infant"
CHILD = "child"
ADULT = "adult"
OLDER_ADULT = "older_adult"
ELDERLY = "elderly"

AGE_BANDS: tuple[str, ...] = (INFANT, CHILD, ADULT, OLDER_ADULT, ELDERLY)

FAMILY = "family"
ADULT_ONLY = "adult_only"
SENIOR = "senior"

HOUSEHOLD_TYPES: tuple[str, ...] = (FAMILY, ADULT_ONLY, SENIOR)

# Relative likelihood that a wearable owner of a given age band owns each device
# kind, applied on top of the configured fleet-wide adoption fraction. A zero
# means the form factor does not exist for that band (no toddler gait shoes, no
# infant sleep headband); values above one mean the band is over-represented
# among that kind's owners. Kinds absent from a band's mapping default to 1.0.
AGE_DEVICE_AFFINITY: dict[str, dict[str, float]] = {
    INFANT: {
        "motion_actigraphy": 2.0,
        "respiratory_acoustic_patch": 2.5,
        "abdominal_acoustic_band": 1.5,
        "thoracic_eit_acoustic_band": 0.6,
        "chest_electrode_patch": 0.4,
        "instrumented_footwear": 0.0,
        "headband_eeg": 0.0,
        "wrist_eda_module": 0.2,
    },
    CHILD: {
        "motion_actigraphy": 1.5,
        "respiratory_acoustic_patch": 1.3,
        "abdominal_acoustic_band": 0.8,
        "thoracic_eit_acoustic_band": 0.4,
        "chest_electrode_patch": 0.3,
        "instrumented_footwear": 0.7,
        "headband_eeg": 0.2,
        "wrist_eda_module": 0.4,
    },
    ADULT: {
        "motion_actigraphy": 1.0,
        "respiratory_acoustic_patch": 0.8,
        "abdominal_acoustic_band": 0.8,
        "thoracic_eit_acoustic_band": 0.7,
        "chest_electrode_patch": 0.7,
        "instrumented_footwear": 1.0,
        "headband_eeg": 1.4,
        "wrist_eda_module": 1.3,
    },
    OLDER_ADULT: {
        "motion_actigraphy": 1.0,
        "respiratory_acoustic_patch": 1.2,
        "abdominal_acoustic_band": 1.2,
        "thoracic_eit_acoustic_band": 1.8,
        "chest_electrode_patch": 2.0,
        "instrumented_footwear": 1.6,
        "headband_eeg": 0.6,
        "wrist_eda_module": 0.8,
    },
    ELDERLY: {
        "motion_actigraphy": 1.1,
        "respiratory_acoustic_patch": 1.4,
        "abdominal_acoustic_band": 1.3,
        "thoracic_eit_acoustic_band": 2.2,
        "chest_electrode_patch": 2.5,
        "instrumented_footwear": 2.0,
        "headband_eeg": 0.3,
        "wrist_eda_module": 0.5,
    },
}


def _validate_fraction_map(
    values: dict[str, float], allowed: tuple[str, ...], label: str
) -> dict[str, float]:
    """Return ``values`` as floats, requiring known keys and a positive sum."""
    resolved: dict[str, float] = {}
    for name, value in values.items():
        if name not in allowed:
            raise ValueError(f"unknown {label} {name!r}; have {sorted(allowed)}")
        if value < 0.0:
            raise ValueError(f"{label} {name!r} must be non-negative, got {value}")
        resolved[name] = float(value)
    total = sum(resolved.values())
    if total <= 0.0:
        raise ValueError(f"{label} fractions must sum above zero, got {total}")
    return {name: value / total for name, value in resolved.items()}


@dataclass
class DemographicsConfig:
    """Age structure and its coupling to device ownership.

    ``enabled`` defaults to False, which keeps the demographically flat fleet:
    no age bands, independent per-kind ownership draws. When enabled, age bands
    come from household composition, ownership becomes correlated through
    ``enthusiasm_sigma``, and the per-band base-device retention lets an infant
    in a wearing household plausibly not carry the core device at all.
    """

    enabled: bool = False
    household_type_fractions: dict[str, float] = field(
        default_factory=lambda: {FAMILY: 0.34, ADULT_ONLY: 0.46, SENIOR: 0.20}
    )
    infant_share_of_children: float = 0.22
    elderly_share_of_seniors: float = 0.35
    base_device_retention: dict[str, float] = field(
        default_factory=lambda: {
            INFANT: 0.80,
            CHILD: 0.92,
            ADULT: 1.0,
            OLDER_ADULT: 1.0,
            ELDERLY: 0.90,
        }
    )
    enthusiasm_sigma: float = 0.8

    def resolved_household_type_fractions(self) -> dict[str, float]:
        """Household type mix, validated and normalised to sum to one."""
        return _validate_fraction_map(
            self.household_type_fractions, HOUSEHOLD_TYPES, "household type"
        )

    def resolved_base_device_retention(self) -> dict[str, float]:
        """Per-band probability of carrying the core device, validated."""
        resolved: dict[str, float] = {}
        for name, value in self.base_device_retention.items():
            if name not in AGE_BANDS:
                raise ValueError(f"unknown age band {name!r}; have {sorted(AGE_BANDS)}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"retention for {name!r} must be in [0, 1], got {value}")
            resolved[name] = float(value)
        for band in AGE_BANDS:
            resolved.setdefault(band, 1.0)
        return resolved

    def validate(self) -> None:
        """Raise when any field is out of range."""
        self.resolved_household_type_fractions()
        self.resolved_base_device_retention()
        for name, value in (
            ("infant_share_of_children", self.infant_share_of_children),
            ("elderly_share_of_seniors", self.elderly_share_of_seniors),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.enthusiasm_sigma < 0.0:
            raise ValueError(f"enthusiasm_sigma must be non-negative, got {self.enthusiasm_sigma}")


def assign_age_bands(
    household_ids: NDArray[np.int64],
    config: DemographicsConfig,
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """Assign an age band per agent through household composition.

    Households draw a type -- family, adult-only or senior -- and members are
    filled from that type's template, so infants always co-reside with at least
    one adult and seniors cluster together. Returns band indices into
    ``AGE_BANDS``.
    """
    config.validate()
    type_fractions = config.resolved_household_type_fractions()
    types = [name for name in HOUSEHOLD_TYPES if name in type_fractions]
    probabilities = np.array([type_fractions[name] for name in types], dtype=np.float64)

    bands = np.full(len(household_ids), AGE_BANDS.index(ADULT), dtype=np.int8)
    unique_households = np.unique(household_ids)
    drawn_types = rng.choice(len(types), size=len(unique_households), p=probabilities)
    infant_index = AGE_BANDS.index(INFANT)
    child_index = AGE_BANDS.index(CHILD)
    adult_index = AGE_BANDS.index(ADULT)
    older_index = AGE_BANDS.index(OLDER_ADULT)
    elderly_index = AGE_BANDS.index(ELDERLY)

    for position, household in enumerate(unique_households):
        members = np.nonzero(household_ids == household)[0]
        household_type = types[int(drawn_types[position])]
        size = len(members)
        if household_type == FAMILY:
            n_adults = min(size, 2)
            juveniles = size - n_adults
            member_bands = [adult_index] * n_adults
            infant_draws = rng.random(juveniles) < config.infant_share_of_children
            member_bands.extend(
                infant_index if is_infant else child_index for is_infant in infant_draws
            )
        elif household_type == SENIOR:
            n_seniors = min(size, 2)
            member_bands = [
                elderly_index if draw < config.elderly_share_of_seniors else older_index
                for draw in rng.random(n_seniors)
            ]
            member_bands.extend([adult_index] * (size - n_seniors))
        else:
            member_bands = [adult_index] * size
        bands[members] = member_bands
    return bands


def band_counts(bands: NDArray[np.int8]) -> dict[str, int]:
    """Number of agents per age band, keyed by band name."""
    return {band: int(np.count_nonzero(bands == index)) for index, band in enumerate(AGE_BANDS)}


def ownership_weights(
    bands: NDArray[np.int8],
    kind_name: str,
    enthusiasm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-person relative likelihood of owning ``kind_name``.

    The weight is the age affinity times the person's enthusiasm factor. A zero
    affinity is a hard exclusion: the band cannot own that form factor at all.
    """
    affinity = np.ones(len(bands), dtype=np.float64)
    for index, band in enumerate(AGE_BANDS):
        affinity[bands == index] = AGE_DEVICE_AFFINITY.get(band, {}).get(kind_name, 1.0)
    return affinity * enthusiasm


def draw_enthusiasm(
    n: int, config: DemographicsConfig, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Per-person multi-device enthusiasm, mean-one and long-tailed.

    A lognormal factor shared across every kind is what turns independent
    binomial ownership into a fleet where most people carry one or two sensors
    and a minority carry most of the catalogue.
    """
    if config.enthusiasm_sigma <= 0.0:
        return np.ones(n, dtype=np.float64)
    sigma = float(config.enthusiasm_sigma)
    return np.asarray(rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n), dtype=np.float64)


__all__ = [
    "ADULT",
    "ADULT_ONLY",
    "AGE_BANDS",
    "AGE_DEVICE_AFFINITY",
    "CHILD",
    "DemographicsConfig",
    "ELDERLY",
    "FAMILY",
    "HOUSEHOLD_TYPES",
    "INFANT",
    "OLDER_ADULT",
    "SENIOR",
    "assign_age_bands",
    "band_counts",
    "draw_enthusiasm",
    "ownership_weights",
]
