"""Device bundles: who owns which sensor, and when it actually reports.

A person does not adopt *channels*, they adopt *devices*. This module makes the
device the unit of ownership and the unit of missingness:

- A :class:`DeviceKind` is one wearable a person can own. It carries a bundle of
  :class:`DeviceChannel` bindings, each of which is a channel plus the yield
  ("usable duty cycle") that hardware achieves for it in the field.
- A :class:`DeviceFleet` assigns ownership across the wearable population and,
  each epoch, produces the observed-channel mask every agent's detector needs.

The fleet keeps **one** observation layout for the whole population: the union
of the channels of every enabled device kind. A person who does not own the
abdominal band simply has those channels permanently missing, which is exactly
the mask semantics :mod:`garland.biometrics` already implements — the channels
are marginalized out of the Mahalanobis score, learn nothing, and, because the
cut is re-calibrated to the reported width (:mod:`garland.thresholds`), do not
change that person's alarm rate. Ownership therefore costs nothing in false
positives, which is the property that makes a mixed-modality fleet measurable.

Yields, resting distributions, and illness effect sizes for the EIT/contact
acoustic channels come from the calibration table in
``docs/SENSOR_MODALITIES.md``. Two modelling notes are load-bearing:

- ``gastric_emptying_index`` is not a per-epoch quantity. A liquid-meal T-half
  needs a two-to-three hour postprandial window, so the band reports it once per
  digestion window rather than continuously: ``event_completion_hours``.
- Yield draws are independent per channel. Real artifact is correlated within a
  band (one torso twist spoils every frame at once); modelling that correlation
  is future work and is noted in the docs as a known simplification.

Each kind also carries its own :class:`SubsystemPowerProfile`: a band is a
separate piece of hardware with its own cell, its own draw, and its own habits
about being taken off, so it dies and is charged independently of the watch.
:class:`garland.device_lifecycle.SubsystemLifecycle` runs one lifecycle engine
per kind from these profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.channels import (
    BOWEL_SOUND_BURST_RATE,
    CORE_VITALS,
    GAIT_ASYMMETRY,
    GAIT_SPEED,
    GASTRIC_EMPTYING_INDEX,
    PEP_MS,
    PULSE_WAVE_VELOCITY,
    REGIONAL_VENTILATION_HETEROGENEITY,
    SLEEP_FRAGMENTATION_INDEX,
    STEP_COUNT,
    STRIDE_TIME_VARIABILITY,
    Channel,
    ChannelSet,
)
from garland.constants import STEPS_PER_DAY

_HOURS_PER_STEP = 24.0 / STEPS_PER_DAY


def _epoch_of_day(hour_of_day: float) -> int:
    """Index of the epoch containing ``hour_of_day`` within the day."""
    return int(hour_of_day / _HOURS_PER_STEP + 1e-9) % STEPS_PER_DAY


def _is_sleep_hour(hour_of_day: float) -> bool:
    """Whether ``hour_of_day`` falls in the sedentary 22:00-06:00 window."""
    return hour_of_day >= 22.0 or hour_of_day < 6.0


@dataclass(frozen=True)
class DeviceChannel:
    """One channel a device reports, and how often it manages to.

    Parameters
    ----------
    channel
        The channel definition from :mod:`garland.channels`.
    duty_cycle
        Fraction of eligible epochs the device returns a usable value for this
        channel while the wearer is sedentary and awake.
    sleep_yield_bonus
        Added to ``duty_cycle`` during sleep hours. Abdominal acoustics reach
        their highest yield when the wearer is still and quiet.
    activity_penalty
        Subtracted from ``duty_cycle``, scaled by activity level. Gross motion
        is what actually destroys these channels: electrode contact drift for
        impedance, fiducial-point loss for cardiac timing, garment friction and
        speech for contact acoustics.
    activity_bonus
        Added to ``duty_cycle``, scaled by activity level. The inverse case: a
        shoe cannot estimate a stride the wearer never takes, so ambulation is
        the *precondition* for these channels rather than their artifact.
    event_completion_hours
        Local hours at which an event-gated estimate completes. When non-empty
        the channel reports *only* in the epoch containing one of those hours,
        modelling a quantity that is integrated over a long window rather than
        sampled per epoch.
    """

    channel: Channel
    duty_cycle: float
    sleep_yield_bonus: float = 0.0
    activity_penalty: float = 0.0
    activity_bonus: float = 0.0
    event_completion_hours: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError(
                f"duty_cycle for {self.channel.name} must be in [0, 1], got {self.duty_cycle}"
            )

    @property
    def is_event_gated(self) -> bool:
        """Whether this channel reports only at event completion times."""
        return bool(self.event_completion_hours)

    def is_eligible(self, hour_of_day: float) -> bool:
        """Whether the current epoch is one this channel can report in."""
        if not self.is_event_gated:
            return True
        epoch = _epoch_of_day(hour_of_day)
        return any(_epoch_of_day(completion) == epoch for completion in self.event_completion_hours)

    def yield_probability(self, hour_of_day: float, activity_level: float) -> float:
        """Probability of a usable value this epoch, given state of the wearer."""
        activity = max(activity_level, 0.0)
        probability = self.duty_cycle + (self.activity_bonus - self.activity_penalty) * activity
        if _is_sleep_hour(hour_of_day):
            probability += self.sleep_yield_bonus
        return float(min(max(probability, 0.0), 1.0))


@dataclass(frozen=True)
class SubsystemPowerProfile:
    """How one device kind's own power budget differs from the wrist baseline.

    Multipliers scale the fleet-wide :class:`~garland.device_lifecycle.
    DeviceLifecycleConfig`, so tuning the fleet still moves every subsystem
    together while preserving the ordering between them.

    Parameters
    ----------
    drain_multiplier
        Scales per-step battery drain. Constant-current injection across 16-32
        electrodes plus multi-channel acoustic sampling costs far more than an
        intermittent optical pulse read.
    capacity_multiplier
        Scales battery capacity. A torso band has room for a larger cell than a
        watch, which partly offsets its draw.
    activity_drain_multiplier_scale
        Scales the *activity* component of drain: motion means more artifact
        rejection and retries on the impedance front end.
    removal_multiplier
        Scales both removal probabilities. A chest or abdominal band comes off
        for showers and sleep far more readily than a watch does.
    charge_multiplier
        Scales the home charge rate.
    """

    drain_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    activity_drain_multiplier_scale: float = 1.0
    removal_multiplier: float = 1.0
    charge_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("drain_multiplier", self.drain_multiplier),
            ("capacity_multiplier", self.capacity_multiplier),
            ("activity_drain_multiplier_scale", self.activity_drain_multiplier_scale),
            ("removal_multiplier", self.removal_multiplier),
            ("charge_multiplier", self.charge_multiplier),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")


WRIST_POWER = SubsystemPowerProfile()


@dataclass(frozen=True)
class DeviceKind:
    """One wearable a person can adopt, with the channel bundle it reports."""

    name: str
    description: str
    device_channels: tuple[DeviceChannel, ...]
    default_adoption: float = 0.0
    power: SubsystemPowerProfile = WRIST_POWER

    def __post_init__(self) -> None:
        if not self.device_channels:
            raise ValueError(f"device kind {self.name!r} reports no channels")
        if not 0.0 <= self.default_adoption <= 1.0:
            raise ValueError(
                f"default_adoption for {self.name!r} must be in [0, 1], got {self.default_adoption}"
            )

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Channel definitions this device reports, in bundle order."""
        return tuple(binding.channel for binding in self.device_channels)


WRIST_PPG = DeviceKind(
    name="wrist_ppg",
    description="Consumer wrist optical wearable: the historical GARLAND device.",
    device_channels=tuple(
        DeviceChannel(channel=channel, duty_cycle=1.0) for channel in CORE_VITALS.channels
    ),
    default_adoption=1.0,
)

THORACIC_EIT_ACOUSTIC_BAND = DeviceKind(
    name="thoracic_eit_acoustic_band",
    description=(
        "Conformal thoracic band combining multi-frequency electrical impedance "
        "tomography with multipoint contact acoustics. Reports regional "
        "ventilation heterogeneity and cardiac-timing intervals."
    ),
    device_channels=(
        DeviceChannel(
            channel=REGIONAL_VENTILATION_HETEROGENEITY,
            duty_cycle=0.75,
            activity_penalty=0.30,
        ),
        DeviceChannel(channel=PEP_MS, duty_cycle=0.72, activity_penalty=0.40),
        DeviceChannel(channel=PULSE_WAVE_VELOCITY, duty_cycle=0.68, activity_penalty=0.40),
    ),
    power=SubsystemPowerProfile(
        # Current injection cycled to 1 MHz plus synchronous multi-channel
        # acoustics, against a larger torso cell.
        drain_multiplier=3.0,
        capacity_multiplier=1.6,
        activity_drain_multiplier_scale=1.5,
        removal_multiplier=2.0,
    ),
)

ABDOMINAL_ACOUSTIC_BAND = DeviceKind(
    name="abdominal_acoustic_band",
    description=(
        "Abdominal contact-microphone and impedance band. Reports bowel-sound "
        "burst rate continuously and a gastric-emptying estimate once per "
        "postprandial window."
    ),
    device_channels=(
        DeviceChannel(
            channel=BOWEL_SOUND_BURST_RATE,
            duty_cycle=0.52,
            sleep_yield_bonus=0.15,
            activity_penalty=0.45,
        ),
        DeviceChannel(
            channel=GASTRIC_EMPTYING_INDEX,
            duty_cycle=0.575,
            activity_penalty=0.30,
            # Estimates complete roughly three hours after typical meal times.
            event_completion_hours=(11.0, 16.0, 22.0),
        ),
    ),
    power=SubsystemPowerProfile(
        # Acoustics without a full EIT drive: cheaper than the thoracic band,
        # dearer than a watch, and the first thing taken off.
        drain_multiplier=2.0,
        capacity_multiplier=1.3,
        removal_multiplier=2.5,
    ),
)

MOTION_ACTIGRAPHY = DeviceKind(
    name="motion_actigraphy",
    description=(
        "Accelerometer-only actigraph (wrist or waist). Reports a per-epoch "
        "pedometer count continuously and one overnight sleep-motion aggregate "
        "at wake."
    ),
    device_channels=(
        # Motion is this device's signal rather than its artifact, so there is
        # no activity penalty; the losses are off-body time, which the
        # subsystem lifecycle already models.
        DeviceChannel(channel=STEP_COUNT, duty_cycle=0.95),
        DeviceChannel(
            channel=SLEEP_FRAGMENTATION_INDEX,
            # Nights scored as valid: the wearer slept in the window and the
            # device stayed on.
            duty_cycle=0.85,
            event_completion_hours=(7.0,),
        ),
    ),
    power=SubsystemPowerProfile(
        # An accelerometer with no optical or impedance front end is the
        # cheapest thing here, and it is worn overnight because that is where
        # half its signal comes from.
        drain_multiplier=0.4,
        removal_multiplier=0.6,
    ),
)

INSTRUMENTED_FOOTWEAR = DeviceKind(
    name="instrumented_footwear",
    description=(
        "Shoe-borne inertial insole. Reports gait speed, stride-time "
        "variability and left-right asymmetry, but only while the wearer is "
        "actually walking."
    ),
    device_channels=tuple(
        # Ambulation-gated rather than artifact-limited: yield is near zero for a
        # sedentary epoch and high during a walking bout, and the negative sleep
        # bonus takes the overnight epochs to zero. Asymmetry needs the most
        # strides of the three, so it is the least often reported.
        DeviceChannel(
            channel=channel,
            duty_cycle=0.10,
            activity_bonus=bonus,
            sleep_yield_bonus=-0.10,
        )
        for channel, bonus in (
            (GAIT_SPEED, 0.80),
            (STRIDE_TIME_VARIABILITY, 0.70),
            (GAIT_ASYMMETRY, 0.55),
        )
    ),
    power=SubsystemPowerProfile(
        # An insole IMU is cheap to run but has a tiny cell, and shoes come off
        # every evening and are rarely put on a charger.
        drain_multiplier=0.6,
        capacity_multiplier=0.5,
        removal_multiplier=2.2,
        charge_multiplier=0.7,
    ),
)

BASE_DEVICE_KIND = WRIST_PPG

DEVICE_CATALOGUE: dict[str, DeviceKind] = {
    kind.name: kind
    for kind in (
        WRIST_PPG,
        THORACIC_EIT_ACOUSTIC_BAND,
        ABDOMINAL_ACOUSTIC_BAND,
        MOTION_ACTIGRAPHY,
        INSTRUMENTED_FOOTWEAR,
    )
}


@dataclass
class DeviceFleetConfig:
    """Per-modality adoption for the wearable population.

    ``enabled`` defaults to False, which keeps the historical single-device
    fleet: every wearable owner reports the four core vitals every epoch.
    ``adoption`` maps device kind name to the fraction of wearable owners who
    own that kind; kinds omitted from the mapping are not simulated at all and
    contribute no channels.
    """

    enabled: bool = False
    adoption: dict[str, float] = field(default_factory=dict)

    def resolved_adoption(self) -> dict[str, float]:
        """Adoption fractions per kind, validated against the catalogue."""
        resolved: dict[str, float] = {}
        for name, fraction in self.adoption.items():
            if name not in DEVICE_CATALOGUE:
                raise ValueError(f"unknown device kind {name!r}; have {sorted(DEVICE_CATALOGUE)}")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"adoption fraction for {name!r} must be in [0, 1], got {fraction}"
                )
            resolved[name] = float(fraction)
        return resolved


def build_channel_set(kinds: tuple[DeviceKind, ...]) -> ChannelSet:
    """Union of the channels of ``kinds``, core vitals first, order stable."""
    channels: list[Channel] = list(CORE_VITALS.channels)
    seen = {channel.name for channel in channels}
    for kind in kinds:
        for channel in kind.channels:
            if channel.name in seen:
                continue
            channels.append(channel)
            seen.add(channel.name)
    return ChannelSet(tuple(channels))


class DeviceFleet:
    """Ownership and per-epoch reporting for a mixed-modality wearable fleet."""

    def __init__(
        self,
        n_wearable: int,
        config: DeviceFleetConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.n_wearable = n_wearable
        adoption = config.resolved_adoption()
        # The base device is always present: it defines the core vitals every
        # owner reports, and it is what ``has_wearable`` has always meant.
        self.kinds: tuple[DeviceKind, ...] = (BASE_DEVICE_KIND,) + tuple(
            DEVICE_CATALOGUE[name] for name in sorted(adoption) if name != BASE_DEVICE_KIND.name
        )
        self.channel_set = build_channel_set(self.kinds)
        self.ownership = np.zeros((n_wearable, len(self.kinds)), dtype=np.bool_)
        self.ownership[:, 0] = True
        for position, kind in enumerate(self.kinds):
            if position == 0:
                continue
            fraction = adoption[kind.name]
            n_owners = int(np.floor(fraction * n_wearable))
            if n_owners <= 0:
                continue
            owners = rng.choice(n_wearable, size=n_owners, replace=False)
            self.ownership[owners, position] = True

    def columns_of(self, kind: DeviceKind) -> tuple[int, ...]:
        """Channel-set columns reported by ``kind``."""
        return tuple(self.channel_set.index(channel.name) for channel in kind.channels)

    @property
    def base_columns(self) -> tuple[int, ...]:
        """Columns of the base wrist device, i.e. the core vitals."""
        return self.columns_of(self.kinds[0])

    def owner_counts(self) -> dict[str, int]:
        """Number of wearable owners per device kind."""
        return {
            kind.name: int(np.count_nonzero(self.ownership[:, position]))
            for position, kind in enumerate(self.kinds)
        }

    def observed_matrix(
        self,
        hour_of_day: float,
        activity_level: float,
        rng: np.random.Generator,
        subsystem_active: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Observed-channel mask for every wearable owner this epoch.

        Rows are wearable-local indices, columns follow ``channel_set``. A
        channel is observed when its owner owns the device, that subsystem is
        powered and worn, the epoch is one the channel can report in, and the
        yield draw succeeds.

        ``subsystem_active`` is an optional ``(n_wearable, n_kinds)`` mask of
        per-subsystem operability, laid out like ``ownership``. Because each
        subsystem carries its own battery, a flat watch silences only the core
        vitals and leaves an owned band reporting.
        """
        if subsystem_active is not None and subsystem_active.shape != self.ownership.shape:
            raise ValueError(
                f"subsystem_active has shape {subsystem_active.shape}, "
                f"expected {self.ownership.shape}"
            )
        observed = np.zeros((self.n_wearable, len(self.channel_set)), dtype=np.bool_)
        for position, kind in enumerate(self.kinds):
            owned = self.ownership[:, position]
            if subsystem_active is not None:
                owned = owned & subsystem_active[:, position]
            for binding in kind.device_channels:
                column = self.channel_set.index(binding.channel.name)
                if not binding.is_eligible(hour_of_day):
                    continue
                probability = binding.yield_probability(hour_of_day, activity_level)
                if probability >= 1.0:
                    observed[:, column] |= owned
                elif probability > 0.0:
                    observed[:, column] |= owned & (rng.random(self.n_wearable) < probability)
        return observed


__all__ = [
    "ABDOMINAL_ACOUSTIC_BAND",
    "BASE_DEVICE_KIND",
    "DEVICE_CATALOGUE",
    "MOTION_ACTIGRAPHY",
    "THORACIC_EIT_ACOUSTIC_BAND",
    "WRIST_POWER",
    "WRIST_PPG",
    "DeviceChannel",
    "DeviceFleet",
    "DeviceFleetConfig",
    "DeviceKind",
    "SubsystemPowerProfile",
    "build_channel_set",
]
