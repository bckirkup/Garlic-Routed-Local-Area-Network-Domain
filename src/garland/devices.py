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

Ownership itself is uniform across the wearable population by default. When
:class:`garland.demographics.DemographicsConfig` is enabled it becomes
age-conditioned and correlated within a person, without moving the configured
per-kind adoption fractions -- see :mod:`garland.demographics`.

Each kind also carries its own :class:`SubsystemPowerProfile`: a band is a
separate piece of hardware with its own cell, its own draw, and its own habits
about being taken off, so it dies and is charged independently of the watch.
:class:`garland.device_lifecycle.SubsystemLifecycle` runs one lifecycle engine
per kind from these profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.channels import (
    ACOUSTIC_MOTILITY_INDEX,
    ALPHA_THETA_RATIO,
    BLADDER_FILLING_IMPEDANCE_SHIFT,
    BODY_TEMPERATURE,
    BOWEL_SOUND_BURST_RATE,
    CORE_VITALS,
    COUGH_RATE,
    CRACKLE_COUNT_PER_CYCLE,
    ECTOPY_BURDEN,
    EDA_SCL,
    EIT_PERFUSION_PULSATILITY_RATIO,
    GAIT_ASYMMETRY,
    GAIT_SPEED,
    GASTRIC_EMPTYING_INDEX,
    HEART_RATE,
    HEART_SOUND_S1_S2_RATIO,
    HRV_RMSSD,
    PEP_MS,
    PTT_SYSTOLIC_BP,
    PULSE_WAVE_VELOCITY,
    QTC_MS,
    REGIONAL_VENTILATION_HETEROGENEITY,
    REM_SLEEP_FRACTION,
    S3_ENERGY_FRACTION,
    SLEEP_FRAGMENTATION_INDEX,
    SLEEP_ONSET_LATENCY,
    SLOW_WAVE_ACTIVITY_FRACTION,
    SPEECH_PAUSE_RATIO,
    SPO2,
    STEP_COUNT,
    STRIDE_TIME_VARIABILITY,
    WAKE_AFTER_SLEEP_ONSET,
    WHEEZE_DURATION_FRACTION,
    WRIST_SKIN_TEMPERATURE,
    Channel,
    ChannelSet,
)
from garland.constants import STEPS_PER_DAY
from garland.demographics import (
    AGE_BANDS,
    DemographicsConfig,
    draw_enthusiasm,
    ownership_weights,
)

_HOURS_PER_STEP = 24.0 / STEPS_PER_DAY
logger = logging.getLogger(__name__)


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
    )
    + (
        DeviceChannel(
            channel=SPO2,
            duty_cycle=0.45,
            sleep_yield_bonus=0.35,
            activity_penalty=0.30,
        ),
        DeviceChannel(
            channel=WRIST_SKIN_TEMPERATURE,
            duty_cycle=0.90,
            activity_penalty=0.10,
        ),
    ),
    default_adoption=1.0,
)

_CORE_WRIST_PPG = DeviceKind(
    name="wrist_ppg",
    description="Consumer wrist optical wearable: the historical GARLAND device.",
    device_channels=tuple(
        DeviceChannel(channel=channel, duty_cycle=1.0) for channel in CORE_VITALS.channels
    ),
    default_adoption=1.0,
)

WRIST_EDA_MODULE = DeviceKind(
    name="wrist_eda_module",
    description=(
        "EDA-capable wrist device line: continuous skin-conductance "
        "level from a dorsal-wrist electrode pair."
    ),
    device_channels=(
        DeviceChannel(
            channel=EDA_SCL,
            duty_cycle=0.80,
            activity_penalty=0.25,
        ),
    ),
    power=WRIST_POWER,
)

HEARABLE = DeviceKind(
    name="hearable",
    description=(
        "In-ear biometric earbud: near-core temperature, SpO2, and heart rate "
        "at conversational-wear duty."
    ),
    device_channels=(
        DeviceChannel(channel=BODY_TEMPERATURE, duty_cycle=0.55),
        DeviceChannel(channel=SPO2, duty_cycle=0.55, activity_penalty=0.10),
        DeviceChannel(channel=HEART_RATE, duty_cycle=0.55),
    ),
)

THORACIC_EIT_ACOUSTIC_BAND = DeviceKind(
    name="thoracic_eit_acoustic_band",
    description=(
        "Conformal thoracic band combining multi-frequency electrical impedance "
        "tomography with multipoint contact acoustics. Reports regional "
        "ventilation heterogeneity, cardiac-synchronous perfusion pulsatility, "
        "and cardiac-timing intervals."
    ),
    device_channels=(
        DeviceChannel(
            channel=REGIONAL_VENTILATION_HETEROGENEITY,
            duty_cycle=0.75,
            activity_penalty=0.30,
        ),
        # Separating the cardiac-synchronous component of the impedance signal
        # from the tidal one needs R-peak gating, so this channel is a little
        # more fragile than the ventilation field it is divided by.
        DeviceChannel(
            channel=EIT_PERFUSION_PULSATILITY_RATIO,
            duty_cycle=0.75,
            activity_penalty=0.40,
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
        "burst rate and acoustic motility fraction continuously, a pelvic "
        "bladder-filling impedance shift, and a gastric-emptying estimate once "
        "per postprandial window."
    ),
    device_channels=(
        DeviceChannel(
            channel=BOWEL_SOUND_BURST_RATE,
            duty_cycle=0.52,
            sleep_yield_bonus=0.15,
            activity_penalty=0.45,
        ),
        # Same transducer, same artifact profile: the duration fraction and the
        # burst rate are two readings of one recording, so they succeed and fail
        # together in the field even though the yield draws here are independent.
        DeviceChannel(
            channel=ACOUSTIC_MOTILITY_INDEX,
            duty_cycle=0.525,
            sleep_yield_bonus=0.15,
            activity_penalty=0.45,
        ),
        # Pelvic impedance rather than acoustics: posture changes move visceral
        # geometry, so ambulation costs less here than garment friction costs
        # the microphones.
        DeviceChannel(
            channel=BLADDER_FILLING_IMPEDANCE_SHIFT,
            duty_cycle=0.625,
            activity_penalty=0.35,
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

RESPIRATORY_ACOUSTIC_PATCH = DeviceKind(
    name="respiratory_acoustic_patch",
    description=(
        "Adhesive suprasternal/chest contact-microphone patch. Reports cough "
        "rate, a natural-speech pause ratio, the wheeze duration fraction and "
        "crackle count per breath, and the first-to-second heart-sound "
        "amplitude ratio with the post-S2 energy fraction."
    ),
    device_channels=(
        # Coughs are loud, brief and rare: nearly every awake epoch yields a
        # rate, and motion masks few of them because a cough dwarfs friction.
        DeviceChannel(channel=COUGH_RATE, duty_cycle=0.88, activity_penalty=0.15),
        # Speech is the only channel here that needs the wearer to be doing
        # something. Conversation happens while awake and mostly still, so it
        # gains a little with light activity and vanishes overnight.
        DeviceChannel(
            channel=SPEECH_PAUSE_RATIO,
            duty_cycle=0.30,
            activity_bonus=0.20,
            sleep_yield_bonus=-0.30,
        ),
        # A wheeze is continuous and tonal, so it survives more masking than a
        # crackle, which is a millisecond transient that garment shear imitates
        # and which therefore needs cross-channel rejection.
        DeviceChannel(
            channel=WHEEZE_DURATION_FRACTION,
            duty_cycle=0.775,
            sleep_yield_bonus=0.08,
            activity_penalty=0.50,
        ),
        DeviceChannel(
            channel=CRACKLE_COUNT_PER_CYCLE,
            duty_cycle=0.725,
            sleep_yield_bonus=0.12,
            activity_penalty=0.50,
        ),
        # Heart sounds are the quietest thing a contact mic chases, so they are
        # the most motion-fragile channels on the patch. The third heart sound
        # sits lower in frequency than S1 and S2 and is scored from the same
        # windows, hence a similar but slightly better yield.
        DeviceChannel(
            channel=HEART_SOUND_S1_S2_RATIO,
            duty_cycle=0.60,
            sleep_yield_bonus=0.18,
            activity_penalty=0.55,
        ),
        DeviceChannel(
            channel=S3_ENERGY_FRACTION,
            duty_cycle=0.675,
            sleep_yield_bonus=0.18,
            activity_penalty=0.55,
        ),
    ),
    power=SubsystemPowerProfile(
        # Continuous multi-channel audio sampling with on-node event extraction,
        # in a thin adhesive patch with a correspondingly small cell.
        drain_multiplier=2.4,
        capacity_multiplier=0.7,
        removal_multiplier=1.4,
    ),
)

CHEST_ELECTRODE_PATCH = DeviceKind(
    name="chest_electrode_patch",
    description=(
        "Adhesive two-lead chest electrode patch with a co-located "
        "accelerometer. Reports rate and beat-to-beat variability from the ECG "
        "itself, a rate-corrected QT interval, premature-beat burden, and a "
        "cuffless systolic estimate from electrode-to-pulse-foot transit time."
    ),
    device_channels=(
        # The patch re-reports two channels the wrist already covers, at a
        # yield an optical sensor cannot match. Masks are OR-ed across owned
        # devices, so this is redundancy rather than duplication: when the
        # watch battery flattens, an owner of both keeps reporting rate and
        # variability from the electrodes.
        DeviceChannel(channel=HEART_RATE, duty_cycle=0.97, activity_penalty=0.10),
        DeviceChannel(channel=HRV_RMSSD, duty_cycle=0.90, activity_penalty=0.35),
        # Repolarisation needs a clean T-wave, which is the first thing motion
        # takes away.
        DeviceChannel(
            channel=QTC_MS,
            duty_cycle=0.80,
            sleep_yield_bonus=0.10,
            activity_penalty=0.55,
        ),
        DeviceChannel(channel=ECTOPY_BURDEN, duty_cycle=0.85, activity_penalty=0.30),
        # Transit time needs an R-peak *and* a distal pulse foot from the
        # patch's accelerometer, so it is the most fragile channel here.
        DeviceChannel(
            channel=PTT_SYSTOLIC_BP,
            duty_cycle=0.65,
            sleep_yield_bonus=0.15,
            activity_penalty=0.50,
        ),
    ),
    power=SubsystemPowerProfile(
        # Continuous two-lead acquisition plus beat detection: dearer than an
        # actigraph, cheaper than driving an EIT electrode array. Adhesive
        # patches are worn continuously for days and then thrown away rather
        # than taken off nightly, so removal is barely above the wrist.
        drain_multiplier=1.8,
        capacity_multiplier=0.8,
        activity_drain_multiplier_scale=1.2,
        removal_multiplier=1.2,
    ),
)

HEADBAND_EEG = DeviceKind(
    name="headband_eeg",
    description=(
        "Dry-electrode forehead EEG headband. Reports four overnight sleep "
        "aggregates scored from the electroencephalogram itself — onset "
        "latency, wake after sleep onset, REM fraction and slow-wave activity "
        "— plus a waking alpha-over-theta vigilance ratio while the wearer is "
        "still."
    ),
    device_channels=(
        # The four staging aggregates are one scoring pass over one night, so
        # they complete together at wake rather than per epoch. Onset latency
        # needs only the start of the night and survives a partial recording;
        # REM and slow-wave shares need the whole of it.
        DeviceChannel(
            channel=SLEEP_ONSET_LATENCY,
            duty_cycle=0.80,
            event_completion_hours=(7.0,),
        ),
        DeviceChannel(
            channel=WAKE_AFTER_SLEEP_ONSET,
            duty_cycle=0.75,
            event_completion_hours=(7.0,),
        ),
        DeviceChannel(
            channel=REM_SLEEP_FRACTION,
            duty_cycle=0.68,
            event_completion_hours=(7.0,),
        ),
        DeviceChannel(
            channel=SLOW_WAVE_ACTIVITY_FRACTION,
            duty_cycle=0.70,
            event_completion_hours=(7.0,),
        ),
        # The only channel here that needs the wearer awake, and the most
        # motion-fragile in the fleet: a forehead derivation is millivolts of
        # signal under a dry electrode, and any frown or step buries it. The
        # negative sleep bonus takes it to zero overnight, when the band is
        # actually being worn most, so a daytime yield this low is the point.
        DeviceChannel(
            channel=ALPHA_THETA_RATIO,
            duty_cycle=0.25,
            activity_penalty=0.70,
            sleep_yield_bonus=-0.25,
        ),
    ),
    power=SubsystemPowerProfile(
        # Multi-channel high-gain amplification with on-node spectral scoring,
        # in a band that is worn for one night and then left on a bedside table:
        # a large removal rate against a generous charging habit.
        drain_multiplier=1.5,
        capacity_multiplier=0.9,
        removal_multiplier=3.0,
        charge_multiplier=1.3,
    ),
)

BASE_DEVICE_KIND = WRIST_PPG

DEVICE_CATALOGUE: dict[str, DeviceKind] = {
    kind.name: kind
    for kind in (
        WRIST_PPG,
        WRIST_EDA_MODULE,
        HEARABLE,
        THORACIC_EIT_ACOUSTIC_BAND,
        ABDOMINAL_ACOUSTIC_BAND,
        MOTION_ACTIGRAPHY,
        INSTRUMENTED_FOOTWEAR,
        RESPIRATORY_ACOUSTIC_PATCH,
        CHEST_ELECTRODE_PATCH,
        HEADBAND_EEG,
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
        age_bands: NDArray[np.int8] | None = None,
        demographics: DemographicsConfig | None = None,
        eligibility_by_kind: dict[str, NDArray[np.bool_]] | None = None,
    ) -> None:
        self.config = config
        self.n_wearable = n_wearable
        adoption = config.resolved_adoption() if config.enabled else {}
        # The base device is always present: it defines the core vitals every
        # owner reports, and it is what ``has_wearable`` has always meant.
        base_kind = BASE_DEVICE_KIND if config.enabled else _CORE_WRIST_PPG
        self.kinds: tuple[DeviceKind, ...] = (base_kind,) + tuple(
            DEVICE_CATALOGUE[name] for name in sorted(adoption) if name != BASE_DEVICE_KIND.name
        )
        self.channel_set = build_channel_set(self.kinds)
        self.age_bands = age_bands
        self.eligibility_by_kind = eligibility_by_kind or {}
        structured = demographics is not None and demographics.enabled and age_bands is not None
        self.enthusiasm: NDArray[np.float64] | None = None
        if structured:
            assert demographics is not None  # narrowed by ``structured``
            self.enthusiasm = draw_enthusiasm(n_wearable, demographics, rng)
        self.ownership = np.zeros((n_wearable, len(self.kinds)), dtype=np.bool_)
        self.ownership[:, 0] = True
        for position, kind in enumerate(self.kinds):
            if position == 0:
                continue
            fraction = adoption[kind.name]
            n_owners = int(np.floor(fraction * n_wearable))
            if n_owners <= 0:
                continue
            owners = self._draw_owners(n_owners, kind, rng)
            self.ownership[owners, position] = True

    def _draw_owners(
        self,
        n_owners: int,
        kind: DeviceKind,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Pick ``n_owners`` owners of ``kind`` from the wearable population.

        Without demographics this is a uniform draw, so ownership of one kind
        says nothing about ownership of another. With demographics it is a
        weighted draw of the *same size*: the configured adoption fraction still
        fixes how many people own the kind, while age affinity and the shared
        enthusiasm factor decide which people, making ownership correlated
        within a person and skewed by band.
        """
        eligibility = self.eligibility_by_kind.get(kind.name)
        if (self.age_bands is None or self.enthusiasm is None) and eligibility is None:
            return np.asarray(rng.choice(self.n_wearable, size=n_owners, replace=False))
        if self.age_bands is None or self.enthusiasm is None:
            weights = np.ones(self.n_wearable, dtype=np.float64)
        else:
            weights = ownership_weights(self.age_bands, kind.name, self.enthusiasm)
        if eligibility is not None:
            weights = weights * np.asarray(eligibility, dtype=bool)
        eligible = np.nonzero(weights > 0.0)[0]
        if len(eligible) <= n_owners:
            if len(eligible) < n_owners:
                logger.warning(
                    "Device adoption capped by eligibility: kind=%s requested=%d eligible=%d",
                    kind.name,
                    n_owners,
                    len(eligible),
                )
            return eligible.astype(np.int64)
        probabilities = weights[eligible] / weights[eligible].sum()
        return np.asarray(
            rng.choice(eligible, size=n_owners, replace=False, p=probabilities), dtype=np.int64
        )

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

    def devices_per_owner(self) -> NDArray[np.int64]:
        """Number of device kinds owned by each wearable owner, base included."""
        return np.asarray(self.ownership.sum(axis=1), dtype=np.int64)

    def ownership_distribution(self) -> dict[str, float]:
        """Shape of the per-person device count across the fleet.

        The mean is fixed by the configured adoption fractions, so it is the
        spread and the tail that say whether the fleet is demographically flat
        (binomial, nobody wired up) or heterogeneous (most people carrying one
        or two sensors, a minority carrying most of the catalogue).
        """
        counts = self.devices_per_owner()
        if len(counts) == 0:
            return {}
        n_kinds = len(self.kinds)
        return {
            "mean_devices_per_owner": float(counts.mean()),
            "sd_devices_per_owner": float(counts.std()),
            "max_devices_per_owner": int(counts.max()),
            "share_base_device_only": float(np.count_nonzero(counts <= 1) / len(counts)),
            "share_four_or_more_devices": float(np.count_nonzero(counts >= 4) / len(counts)),
            "share_owning_all_kinds": float(np.count_nonzero(counts >= n_kinds) / len(counts)),
        }

    def owner_counts_by_band(self) -> dict[str, dict[str, int]]:
        """Owners per device kind, split by age band.

        Empty without demographics, since there are no bands to split by.
        """
        if self.age_bands is None:
            return {}
        return {
            band: {
                kind.name: int(np.count_nonzero(self.ownership[self.age_bands == index, position]))
                for position, kind in enumerate(self.kinds)
            }
            for index, band in enumerate(AGE_BANDS)
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
    "CHEST_ELECTRODE_PATCH",
    "DEVICE_CATALOGUE",
    "HEADBAND_EEG",
    "HEARABLE",
    "INSTRUMENTED_FOOTWEAR",
    "MOTION_ACTIGRAPHY",
    "RESPIRATORY_ACOUSTIC_PATCH",
    "THORACIC_EIT_ACOUSTIC_BAND",
    "WRIST_POWER",
    "WRIST_PPG",
    "WRIST_EDA_MODULE",
    "DeviceChannel",
    "DeviceFleet",
    "DeviceFleetConfig",
    "DeviceKind",
    "SubsystemPowerProfile",
    "build_channel_set",
]
