"""Sensor channel registry for GARLAND observation vectors.

An observation is a vector of *derived per-epoch features*, one entry per
:class:`Channel`, ordered by the agent's :class:`ChannelSet`. Devices never
export waveforms; a channel is whatever a device can compute on-body and
report for a five-minute epoch.

Every consumer of an observation (synthesis, baseline tracking, sequential
detection, anomaly classification, hazard and confounder deltas, Open
Wearables export) addresses entries by channel *name* through this registry
rather than by position, so a wider channel set requires no changes to those
consumers.

``CORE_VITALS`` is the historical four-channel set (HR, HRV RMSSD, RR, core
temperature) and remains the default. Its parameters reproduce the previous
hard-coded synthesis exactly, including RNG draw order: channel noise is drawn
in channel order, and a channel with a degenerate circadian-amplitude range
draws nothing at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# Number of exceeded channels at or above which an excursion is reported as
# multi-system rather than attributed to one physiological system.
MULTI_SYSTEM_MIN_CHANNELS = 3


class ChannelSystem(str, Enum):
    """Physiological system a channel measures.

    Classification rules are written against systems, not channel names, so
    adding a second cardiac or respiratory channel does not require new rules.
    """

    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    THERMAL = "thermal"
    VASCULAR = "vascular"
    GASTROINTESTINAL = "gastrointestinal"
    MOTOR = "motor"
    SLEEP = "sleep"
    GAIT = "gait"
    URINARY = "urinary"


@dataclass(frozen=True)
class Channel:
    """One named feature channel and the parameters its consumers need.

    Parameters
    ----------
    name, unit, system
        Identity of the channel. ``name`` is the key used by every
        name-addressed API in this package.
    resting_mean, resting_sd, resting_min, resting_max
        Population distribution of the per-person resting value.
    circadian_amp_min, circadian_amp_max
        Per-person circadian amplitude range. A degenerate range (min equal to
        max) means the channel has no circadian term and draws no random
        amplitude.
    circadian_scale, seasonal_coefficient, activity_coefficient
        Multipliers on the shared circadian factor, seasonal factor, and the
        agent's activity level.
    noise_sd
        Standard deviation of the per-epoch measurement and physiological
        noise.
    floor
        Optional lower clamp applied after synthesis (RMSSD cannot be zero).
    hard_floor
        Whether ``floor`` also binds *after* hazard and confounder deltas land.
        Set for counts that have no meaning below their floor at all, such as a
        pedometer; the older channels keep their synthesis-only clamp so seeded
        core-vitals runs are unchanged.
    deviation_threshold
        Absolute deviation from baseline at which this channel counts as
        excursion-positive during classification.
    quiet_fraction
        Fraction of ``deviation_threshold`` below which the channel counts as
        quiet. Used by rules that require a channel to be *un*-perturbed, such
        as respiratory distress without fever.
    prior_variance
        Diagonal entry of the cold-start covariance prior.
    openwearables_type
        Open Wearables timeseries type, or ``None`` for channels with no
        equivalent in that schema (those are omitted from exports).
    """

    name: str
    unit: str
    system: ChannelSystem
    resting_mean: float
    resting_sd: float
    resting_min: float
    resting_max: float
    noise_sd: float
    deviation_threshold: float
    circadian_amp_min: float = 0.0
    circadian_amp_max: float = 0.0
    circadian_scale: float = 1.0
    seasonal_coefficient: float = 0.0
    activity_coefficient: float = 0.0
    floor: float | None = None
    hard_floor: bool = False
    quiet_fraction: float = 1.0
    prior_variance: float = 10.0
    openwearables_type: str | None = None

    @property
    def has_circadian_amplitude(self) -> bool:
        """Whether this channel draws a per-person circadian amplitude."""
        return self.circadian_amp_min != self.circadian_amp_max


HEART_RATE = Channel(
    name="heart_rate",
    unit="bpm",
    system=ChannelSystem.CARDIAC,
    resting_mean=72.0,
    resting_sd=8.0,
    resting_min=50.0,
    resting_max=110.0,
    noise_sd=2.0,
    deviation_threshold=10.0,
    circadian_amp_min=3.0,
    circadian_amp_max=8.0,
    activity_coefficient=40.0,
    openwearables_type="heart_rate",
)

HRV_RMSSD = Channel(
    name="hrv_rmssd",
    unit="ms",
    system=ChannelSystem.CARDIAC,
    resting_mean=42.0,
    resting_sd=15.0,
    resting_min=10.0,
    resting_max=120.0,
    noise_sd=5.0,
    deviation_threshold=10.0,
    activity_coefficient=-20.0,
    floor=5.0,
    openwearables_type="heart_rate_variability_rmssd",
)

RESPIRATORY_RATE = Channel(
    name="respiratory_rate",
    unit="brpm",
    system=ChannelSystem.RESPIRATORY,
    resting_mean=15.0,
    resting_sd=2.0,
    resting_min=8.0,
    resting_max=25.0,
    noise_sd=1.0,
    deviation_threshold=4.0,
    circadian_amp_min=0.5,
    circadian_amp_max=2.0,
    activity_coefficient=8.0,
    openwearables_type="respiratory_rate",
)

BODY_TEMPERATURE = Channel(
    name="body_temperature",
    unit="°C",
    system=ChannelSystem.THERMAL,
    resting_mean=36.8,
    resting_sd=0.3,
    resting_min=35.5,
    resting_max=38.0,
    noise_sd=0.1,
    deviation_threshold=0.8,
    circadian_amp_min=0.2,
    circadian_amp_max=0.5,
    circadian_scale=0.3,
    seasonal_coefficient=0.1,
    activity_coefficient=0.5,
    # 0.5 °C over a 0.8 °C threshold: the "no fever" arm of respiratory rules.
    quiet_fraction=0.625,
    openwearables_type="body_temperature",
)


REGIONAL_VENTILATION_HETEROGENEITY = Channel(
    name="regional_ventilation_heterogeneity",
    unit="GI index",
    system=ChannelSystem.RESPIRATORY,
    resting_mean=0.40,
    resting_sd=0.06,
    resting_min=0.25,
    resting_max=0.60,
    noise_sd=0.025,
    # Illness shifts this by +0.20 to +0.40; three within-person SDs sits well
    # inside that and well outside quiet drift.
    deviation_threshold=0.075,
    floor=0.0,
    prior_variance=0.01,
)

PEP_MS = Channel(
    name="pep_ms",
    unit="ms",
    system=ChannelSystem.CARDIAC,
    resting_mean=102.0,
    resting_sd=14.0,
    resting_min=70.0,
    resting_max=140.0,
    noise_sd=5.0,
    # Febrile inotropy shortens PEP by 20-35 ms.
    deviation_threshold=15.0,
    floor=40.0,
    prior_variance=200.0,
)

PULSE_WAVE_VELOCITY = Channel(
    name="pwv_m_s",
    unit="m/s",
    system=ChannelSystem.VASCULAR,
    resting_mean=7.2,
    resting_sd=1.2,
    resting_min=4.5,
    resting_max=12.0,
    noise_sd=0.4,
    # Inflammatory stiffening is +1.5 to +2.8 m/s; distributive shock is the
    # same magnitude with the opposite sign.
    deviation_threshold=1.2,
    floor=2.0,
    prior_variance=1.5,
)

BOWEL_SOUND_BURST_RATE = Channel(
    name="bowel_sound_burst_rate",
    unit="bursts/min",
    system=ChannelSystem.GASTROINTESTINAL,
    resting_mean=5.5,
    resting_sd=2.0,
    resting_min=1.0,
    resting_max=12.0,
    noise_sd=1.8,
    # Enteritis adds 12-25 bursts/min; ileus removes about 4.5.
    deviation_threshold=5.0,
    floor=0.0,
    prior_variance=4.0,
)

GASTRIC_EMPTYING_INDEX = Channel(
    name="gastric_emptying_index",
    unit="min",
    system=ChannelSystem.GASTROINTESTINAL,
    resting_mean=45.0,
    resting_sd=12.0,
    resting_min=20.0,
    resting_max=90.0,
    noise_sd=6.0,
    # Systemic infection delays T-half by 35-65 min.
    deviation_threshold=18.0,
    floor=5.0,
    prior_variance=150.0,
)


STEP_COUNT = Channel(
    name="step_count",
    unit="steps",
    system=ChannelSystem.MOTOR,
    # Sedentary floor; the activity coefficient supplies the ambulatory part, so
    # a day integrates to roughly 7,000-8,000 steps.
    resting_mean=7.5,
    resting_sd=4.0,
    resting_min=1.0,
    resting_max=25.0,
    noise_sd=10.0,
    # Two within-epoch SDs. Sickness behaviour is smaller than this on any one
    # epoch on purpose: the pedometer earns its keep through the sequential
    # detector accumulating a sustained shortfall, not through single epochs.
    deviation_threshold=20.0,
    activity_coefficient=150.0,
    floor=0.0,
    hard_floor=True,
    prior_variance=150.0,
    openwearables_type="step_count",
)

SLEEP_FRAGMENTATION_INDEX = Channel(
    name="sleep_fragmentation_index",
    unit="%",
    system=ChannelSystem.SLEEP,
    resting_mean=24.0,
    resting_sd=8.0,
    resting_min=5.0,
    resting_max=60.0,
    noise_sd=5.0,
    # Febrile illness fragments sleep by about +12 points, so a fully
    # symptomatic night clears this two-SD cut.
    deviation_threshold=10.0,
    floor=0.0,
    hard_floor=True,
    prior_variance=64.0,
)


GAIT_SPEED = Channel(
    name="gait_speed_m_s",
    unit="m/s",
    system=ChannelSystem.GAIT,
    resting_mean=1.30,
    resting_sd=0.15,
    resting_min=0.80,
    resting_max=1.70,
    noise_sd=0.10,
    # Malaise costs about 0.15 m/s, which is also the widely used
    # minimal-clinically-important difference for habitual gait speed.
    deviation_threshold=0.20,
    floor=0.20,
    hard_floor=True,
    prior_variance=0.05,
)

STRIDE_TIME_VARIABILITY = Channel(
    name="stride_time_variability",
    unit="% CV",
    system=ChannelSystem.GAIT,
    resting_mean=2.4,
    resting_sd=0.8,
    resting_min=1.0,
    resting_max=6.0,
    noise_sd=0.6,
    # Fatigue roughly doubles stride-time CV; two within-person SDs sits inside
    # that and outside quiet drift.
    deviation_threshold=1.2,
    floor=0.3,
    hard_floor=True,
    prior_variance=1.0,
)

GAIT_ASYMMETRY = Channel(
    name="gait_asymmetry",
    unit="%",
    system=ChannelSystem.GAIT,
    resting_mean=2.0,
    resting_sd=1.0,
    resting_min=0.5,
    resting_max=6.0,
    noise_sd=0.8,
    deviation_threshold=1.6,
    floor=0.0,
    hard_floor=True,
    prior_variance=1.5,
)


COUGH_RATE = Channel(
    name="cough_rate",
    unit="coughs/h",
    system=ChannelSystem.RESPIRATORY,
    # Healthy adults cough a handful of times an hour while awake.
    resting_mean=0.8,
    resting_sd=0.6,
    resting_min=0.1,
    resting_max=3.0,
    noise_sd=1.0,
    # Acute cough illness runs an order of magnitude above baseline, so a cut
    # well outside quiet variation is still cleared easily.
    deviation_threshold=3.0,
    floor=0.0,
    hard_floor=True,
    prior_variance=2.0,
)

SPEECH_PAUSE_RATIO = Channel(
    name="speech_pause_ratio",
    unit="%",
    system=ChannelSystem.RESPIRATORY,
    # Percentage of speaking time spent in pauses. Breathlessness shortens
    # phrases, so the pause fraction rises without the voice changing pitch.
    resting_mean=22.0,
    resting_sd=6.0,
    resting_min=10.0,
    resting_max=45.0,
    noise_sd=4.0,
    deviation_threshold=8.0,
    floor=0.0,
    hard_floor=True,
    prior_variance=36.0,
)

WHEEZE_DURATION_FRACTION = Channel(
    name="wheeze_duration_fraction",
    unit="fraction of cycle",
    system=ChannelSystem.RESPIRATORY,
    # Fraction of the breath cycle carrying continuous musical adventitious
    # sound. A healthy airway produces essentially none, so the resting mean is
    # zero and the between-person SD is the whole distribution.
    resting_mean=0.0,
    resting_sd=0.01,
    resting_min=0.0,
    resting_max=0.05,
    noise_sd=0.005,
    # Bronchospasm spans 0.15-0.45 of the cycle, four times this cut.
    deviation_threshold=0.02,
    floor=0.0,
    hard_floor=True,
    prior_variance=1e-4,
)

CRACKLE_COUNT_PER_CYCLE = Channel(
    name="crackle_count_per_cycle",
    unit="crackles/breath",
    system=ChannelSystem.RESPIRATORY,
    # Discontinuous reopening transients per breath. Healthy parenchyma manages
    # the odd micro-atelectatic crackle, which is why the noise SD is as large
    # as the mean.
    resting_mean=0.2,
    resting_sd=0.3,
    resting_min=0.0,
    resting_max=1.5,
    noise_sd=0.25,
    # Consolidation and fibrotic alveolitis add 4-12 per breath.
    deviation_threshold=0.75,
    floor=0.0,
    hard_floor=True,
    prior_variance=0.09,
)

HEART_SOUND_S1_S2_RATIO = Channel(
    name="heart_sound_s1_s2_ratio",
    unit="ratio",
    system=ChannelSystem.CARDIAC,
    # Relative amplitude of the first to the second heart sound, and the one
    # bidirectional channel in the fleet: an inotropic surge raises it while
    # impaired ventricular contractility suppresses S1 and drops it further
    # than any fever raises it.
    resting_mean=1.15,
    resting_sd=0.22,
    resting_min=0.60,
    resting_max=2.00,
    noise_sd=0.08,
    deviation_threshold=0.25,
    floor=0.10,
    hard_floor=True,
    prior_variance=0.05,
)

S3_ENERGY_FRACTION = Channel(
    name="s3_energy_fraction",
    unit="%",
    system=ChannelSystem.CARDIAC,
    # Share of cardiac acoustic energy in the low-frequency window just after
    # S2. Adults are nearly silent there until filling pressures rise.
    resting_mean=1.2,
    resting_sd=0.8,
    resting_min=0.0,
    resting_max=5.0,
    noise_sd=0.4,
    # Volume overload adds 5-12 points.
    deviation_threshold=1.2,
    floor=0.0,
    hard_floor=True,
    prior_variance=0.64,
)

ACOUSTIC_MOTILITY_INDEX = Channel(
    name="acoustic_motility_index",
    unit="%",
    system=ChannelSystem.GASTROINTESTINAL,
    # Percentage of recording time occupied by bowel-sound events: the duration
    # view of the same motility the burst rate counts, so the two move together
    # and the empirical covariance learns that they do.
    resting_mean=4.2,
    resting_sd=1.8,
    resting_min=0.5,
    resting_max=10.0,
    noise_sd=1.2,
    # Enteritis adds 8-18 points; ileus removes about 3.8.
    deviation_threshold=3.0,
    floor=0.0,
    hard_floor=True,
    prior_variance=3.24,
)

EIT_PERFUSION_PULSATILITY_RATIO = Channel(
    name="eit_perfusion_pulsatility_ratio",
    unit="ratio",
    system=ChannelSystem.VASCULAR,
    # Cardiac-synchronous impedance pulsation as a share of tidal ventilation
    # impedance. The one channel that falls toward a noise floor under vascular
    # occlusion rather than rising under inflammation.
    resting_mean=0.12,
    resting_sd=0.03,
    resting_min=0.05,
    resting_max=0.22,
    noise_sd=0.015,
    # Capillary occlusion costs 0.06-0.09 of the ratio.
    deviation_threshold=0.045,
    floor=0.0,
    hard_floor=True,
    prior_variance=9e-4,
)

BLADDER_FILLING_IMPEDANCE_SHIFT = Channel(
    name="bladder_filling_impedance_shift",
    unit="Δσ/σ₀",
    system=ChannelSystem.URINARY,
    # Pelvic conductivity relative to a post-void baseline. Normal filling and
    # voiding is a large diurnal swing in its own right, which is modelled as a
    # circadian term rather than as noise: the channel is only interpretable
    # against a person's own daily cycle.
    resting_mean=0.0,
    resting_sd=0.02,
    resting_min=-0.05,
    resting_max=0.05,
    noise_sd=0.012,
    # Retention at 400-600 mL shifts it by +0.15 to +0.35.
    deviation_threshold=0.036,
    circadian_amp_min=0.02,
    circadian_amp_max=0.06,
    # Filling overnight, voided on waking: opposite in sign to the heart-rate
    # circadian term.
    circadian_scale=-1.0,
    prior_variance=4e-4,
)


QTC_MS = Channel(
    name="qtc_ms",
    unit="ms",
    system=ChannelSystem.CARDIAC,
    # Rate-corrected QT interval from a two-lead chest recording.
    resting_mean=410.0,
    resting_sd=20.0,
    resting_min=350.0,
    resting_max=470.0,
    noise_sd=8.0,
    # Systemic inflammation prolongs QTc by roughly 15-25 ms, so this cut sits
    # inside a symptomatic episode and outside quiet drift.
    deviation_threshold=25.0,
    # QTc runs longer overnight, opposite in sign to the heart-rate circadian
    # term it shares a driver with.
    circadian_amp_min=4.0,
    circadian_amp_max=10.0,
    circadian_scale=-1.0,
    floor=250.0,
    hard_floor=True,
    prior_variance=400.0,
)

ECTOPY_BURDEN = Channel(
    name="ectopy_burden",
    unit="% beats",
    system=ChannelSystem.CARDIAC,
    # Percentage of beats in the epoch that are premature. Near zero for most
    # people most of the time, with a long right tail.
    resting_mean=0.3,
    resting_sd=0.5,
    resting_min=0.0,
    resting_max=4.0,
    noise_sd=0.4,
    deviation_threshold=1.2,
    floor=0.0,
    hard_floor=True,
    prior_variance=0.4,
)

PTT_SYSTOLIC_BP = Channel(
    name="ptt_systolic_bp",
    unit="mmHg",
    system=ChannelSystem.VASCULAR,
    # Cuffless systolic estimate from the electrode-to-pulse-foot transit time.
    # Absolute accuracy is calibration-limited; the channel is useful for change
    # from a person's own baseline, which is all the detector ever uses.
    resting_mean=118.0,
    resting_sd=12.0,
    resting_min=90.0,
    resting_max=165.0,
    noise_sd=4.0,
    deviation_threshold=10.0,
    circadian_amp_min=3.0,
    circadian_amp_max=8.0,
    activity_coefficient=25.0,
    floor=50.0,
    hard_floor=True,
    prior_variance=144.0,
    openwearables_type="blood_pressure_systolic",
)


@dataclass(frozen=True)
class ChannelSet:
    """An ordered set of channels defining one observation vector layout."""

    channels: tuple[Channel, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a channel set must contain at least one channel")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate channel names in channel set: {names}")
        self._index.update({name: position for position, name in enumerate(names)})

    def __len__(self) -> int:
        return len(self.channels)

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names in vector order."""
        return tuple(channel.name for channel in self.channels)

    def index(self, name: str) -> int:
        """Vector position of ``name``."""
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(
                f"channel {name!r} is not in this channel set; have {self.names}"
            ) from exc

    def has(self, name: str) -> bool:
        """Whether ``name`` is present in this channel set."""
        return name in self._index

    def get(self, name: str) -> Channel:
        """Return the channel named ``name``."""
        return self.channels[self.index(name)]

    def zeros(self) -> NDArray[np.float64]:
        """An all-zero vector with this set's width."""
        return np.zeros(len(self.channels), dtype=np.float64)

    def delta(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Build an additive perturbation vector from named channel deltas.

        Channels absent from ``values`` are zero. Unknown names raise, so a
        signature written for a channel the fleet does not carry is a
        configuration error rather than a silently dropped effect.
        """
        vector = self.zeros()
        for name, value in values.items():
            vector[self.index(name)] = value
        return vector

    def system_indices(self, system: ChannelSystem) -> tuple[int, ...]:
        """Vector positions of every channel measuring ``system``."""
        return tuple(
            position for position, channel in enumerate(self.channels) if channel.system == system
        )

    @property
    def deviation_thresholds(self) -> NDArray[np.float64]:
        """Per-channel excursion thresholds in vector order."""
        return np.array(
            [channel.deviation_threshold for channel in self.channels], dtype=np.float64
        )

    @property
    def prior_variances(self) -> NDArray[np.float64]:
        """Per-channel cold-start covariance prior diagonal, in vector order."""
        return np.array([channel.prior_variance for channel in self.channels], dtype=np.float64)

    @property
    def hard_floors(self) -> NDArray[np.float64]:
        """Per-channel post-perturbation clamp, ``-inf`` where there is none."""
        return np.array(
            [
                channel.floor if channel.hard_floor and channel.floor is not None else -np.inf
                for channel in self.channels
            ],
            dtype=np.float64,
        )

    def clamp(self, observation: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the hard per-channel floors to a fully perturbed observation.

        Synthesis floors its own draw, but hazard and confounder deltas land on
        top of it and some channels have no meaning below their floor: a
        pedometer cannot report a negative count.
        """
        return np.maximum(observation, self.hard_floors)

    def with_channels(self, *channels: Channel) -> ChannelSet:
        """Return a wider set with ``channels`` appended."""
        return ChannelSet(self.channels + channels)


CORE_VITALS = ChannelSet((HEART_RATE, HRV_RMSSD, RESPIRATORY_RATE, BODY_TEMPERATURE))

DEFAULT_CHANNEL_SET = CORE_VITALS


__all__ = [
    "ACOUSTIC_MOTILITY_INDEX",
    "BLADDER_FILLING_IMPEDANCE_SHIFT",
    "BODY_TEMPERATURE",
    "BOWEL_SOUND_BURST_RATE",
    "CORE_VITALS",
    "COUGH_RATE",
    "CRACKLE_COUNT_PER_CYCLE",
    "DEFAULT_CHANNEL_SET",
    "ECTOPY_BURDEN",
    "EIT_PERFUSION_PULSATILITY_RATIO",
    "GAIT_ASYMMETRY",
    "GAIT_SPEED",
    "GASTRIC_EMPTYING_INDEX",
    "HEART_RATE",
    "HEART_SOUND_S1_S2_RATIO",
    "HRV_RMSSD",
    "MULTI_SYSTEM_MIN_CHANNELS",
    "PEP_MS",
    "PTT_SYSTOLIC_BP",
    "PULSE_WAVE_VELOCITY",
    "QTC_MS",
    "REGIONAL_VENTILATION_HETEROGENEITY",
    "RESPIRATORY_RATE",
    "S3_ENERGY_FRACTION",
    "SLEEP_FRAGMENTATION_INDEX",
    "SPEECH_PAUSE_RATIO",
    "STEP_COUNT",
    "STRIDE_TIME_VARIABILITY",
    "WHEEZE_DURATION_FRACTION",
    "Channel",
    "ChannelSet",
    "ChannelSystem",
]
