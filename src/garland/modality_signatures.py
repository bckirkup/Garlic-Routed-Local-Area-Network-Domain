"""Illness and confounder signatures for the EIT/acoustic band channels.

The four core wrist channels carry hand-written per-hazard deltas. The band
channels are driven instead by a few latent physiological axes, because several
of them share one cause: a febrile inflammatory surge shortens the pre-ejection
period, stiffens the aorta and delays gastric emptying at the same time, so
writing three independent deltas would let one fever count as three unrelated
detections in the Mahalanobis score. A future pulse-transit-time blood pressure
channel reads the same ``arterial_stiffening`` axis as ``pwv_m_s`` for the same
reason.

Per-axis magnitudes are the midpoints of the illness deviations tabulated in
``docs/SENSOR_MODALITIES.md``, reached at an axis value of 1.0. They are
simulation calibration for a testbed, not clinical claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet

# Below this, an axis is at rest: axis values are severity fractions, so any
# real signature is orders of magnitude larger.
_AXIS_REST_TOLERANCE = 1e-12

# Global inhomogeneity index units; illness range +0.20 to +0.40.
VENTILATION_HETEROGENEITY_PER_PULMONARY = 0.30
# Milliseconds; febrile inotropy shortens PEP by 20 to 35 ms.
PEP_MS_PER_INFLAMMATION = -27.5
# m/s; inflammatory stiffening is +1.5 to +2.8, distributive shock the reverse.
PWV_PER_STIFFENING = 2.15
# Bursts/min; enteritis adds 12 to 25.
BOWEL_BURSTS_PER_ENTERIC = 18.5
# Bursts/min; paralytic ileus removes about 4.5, i.e. at ``enteric_drive`` -1.
BOWEL_BURSTS_PER_ILEUS = 4.5
# Minutes of T-half; systemic infection delays emptying by 35 to 65.
GASTRIC_MINUTES_PER_INFLAMMATION = 50.0
# Steps per five-minute epoch; -10 per epoch is about -2,900 steps/day, the
# upper end of the reported 1,500-2,500 step/day shortfall during ILI.
STEPS_PER_WITHDRAWAL = -10.0
# Steps per epoch added by an exercise bout at full intensity. Ambulation is
# wildly asymmetric: a run puts more steps into five minutes than a day of
# sickness behaviour removes, which is why the pedometer is confounder-prone.
STEPS_PER_EXERTION_BOUT = 600.0
# Percentage points of sleep fragmentation; febrile illness adds about +12.
SLEEP_FRAGMENTATION_PER_DISTURBANCE = 12.0
# m/s of habitual gait speed lost to full sickness behaviour. This is the usual
# minimal-clinically-important difference, i.e. deliberately near the noise.
GAIT_SPEED_PER_WITHDRAWAL = -0.15
# m/s added while an exercise bout is in progress: the wearer is not walking
# slowly, they are running. Same asymmetry as the pedometer.
GAIT_SPEED_PER_EXERTION_BOUT = 0.45
# Percentage points of stride-time CV; fatigue roughly doubles a 2.4% baseline.
STRIDE_CV_PER_FATIGUE = 2.4
# Percentage points of left-right asymmetry from a mis-seated or loose insole.
GAIT_ASYMMETRY_PER_ARTIFACT = 2.5
# Coughs per hour at full airway irritation. Acute cough illness runs an order
# of magnitude above the sub-1/h healthy baseline.
COUGHS_PER_IRRITATION = 18.0
# Index units of ventilation heterogeneity produced by electrode contact loss
# rather than by any regional loss of ventilation.
VENTILATION_HETEROGENEITY_PER_ARTIFACT = 0.25
# Percentage points of speech spent pausing: breathlessness shortens phrases.
SPEECH_PAUSE_PER_PULMONARY = 9.0
# Percentage points of breaths carrying a wheeze or crackle at consolidation.
ADVENTITIOUS_PER_PULMONARY = 18.0
# Percentage points of the same channel produced by garment friction alone,
# which a contact microphone cannot tell from a real crackle.
ADVENTITIOUS_PER_ARTIFACT = 5.0
# Ratio units of first-to-second heart sound amplitude added by inotropy.
S1_S2_RATIO_PER_INFLAMMATION = 0.35


@dataclass(frozen=True)
class IllnessAxes:
    """Latent drivers shared by the band channels.

    Parameters
    ----------
    inflammatory_drive : float
        Systemic febrile/beta-adrenergic surge, ``0`` to ``1``. Shortens PEP and
        delays gastric emptying.
    pulmonary_involvement : float
        Regional loss of ventilation (consolidation, atelectasis, oedema,
        irritant bronchospasm), ``0`` to ``1``. Raises ventilation heterogeneity.
    enteric_drive : float
        Gut motility, ``-1`` (ileus) to ``1`` (acute enteritis).
    arterial_stiffening : float
        Arterial tone, ``-1`` (distributive shock) to ``1`` (sympathetic
        vasoconstriction). Drives PWV and any future pulse-transit-time channel.
    activity_withdrawal : float
        Ambulation relative to habit, ``-1`` (an exercise bout) to ``1`` (full
        sickness behaviour). Drives the pedometer.
    sleep_disturbance : float
        Night-time restlessness, ``-1`` (unusually settled) to ``1`` (febrile,
        badly fragmented). Drives the sleep-motion aggregate.
    neuromotor_fatigue : float
        Loss of stride-to-stride motor control, ``0`` to ``1``. Drives
        stride-time variability. Kept separate from ``activity_withdrawal``
        because how *much* someone walks and how *steadily* they walk come
        apart: a hard run raises both speed and variability, while malaise
        lowers speed and raises variability.
    instrument_artifact : float
        Mechanical sensor fault rather than physiology, ``0`` to ``1``. Drives
        gait asymmetry and false adventitious breath sounds: no infection
        touches gait asymmetry at all, which is what makes it a negative
        control.
    airway_irritation : float
        Tussive drive, ``0`` to ``1``. Drives cough rate. Deliberately separate
        from ``pulmonary_involvement``: an irritant plume coughs harder than a
        pneumonia while consolidating nothing, and a consolidated lobe can be
        quiet.
    """

    inflammatory_drive: float = 0.0
    pulmonary_involvement: float = 0.0
    enteric_drive: float = 0.0
    arterial_stiffening: float = 0.0
    activity_withdrawal: float = 0.0
    sleep_disturbance: float = 0.0
    neuromotor_fatigue: float = 0.0
    instrument_artifact: float = 0.0
    airway_irritation: float = 0.0

    def __post_init__(self) -> None:
        for name, value, low in (
            ("inflammatory_drive", self.inflammatory_drive, 0.0),
            ("pulmonary_involvement", self.pulmonary_involvement, 0.0),
            ("enteric_drive", self.enteric_drive, -1.0),
            ("arterial_stiffening", self.arterial_stiffening, -1.0),
            ("activity_withdrawal", self.activity_withdrawal, -1.0),
            ("sleep_disturbance", self.sleep_disturbance, -1.0),
            ("neuromotor_fatigue", self.neuromotor_fatigue, 0.0),
            ("instrument_artifact", self.instrument_artifact, 0.0),
            ("airway_irritation", self.airway_irritation, 0.0),
        ):
            if not low <= value <= 1.0:
                raise ValueError(f"{name} must lie in [{low}, 1.0], got {value}")

    @property
    def is_quiet(self) -> bool:
        """Whether every axis is at rest, i.e. the signature is all-zero."""
        largest = max(
            abs(self.inflammatory_drive),
            abs(self.pulmonary_involvement),
            abs(self.enteric_drive),
            abs(self.arterial_stiffening),
            abs(self.activity_withdrawal),
            abs(self.sleep_disturbance),
            abs(self.neuromotor_fatigue),
            abs(self.instrument_artifact),
            abs(self.airway_irritation),
        )
        return largest < _AXIS_REST_TOLERANCE

    @property
    def step_delta(self) -> float:
        """Steps added to a five-minute epoch by this axis state."""
        if self.activity_withdrawal >= 0.0:
            return STEPS_PER_WITHDRAWAL * self.activity_withdrawal
        return STEPS_PER_EXERTION_BOUT * -self.activity_withdrawal

    @property
    def gait_speed_delta(self) -> float:
        """Metres per second added to habitual gait speed by this axis state."""
        if self.activity_withdrawal >= 0.0:
            return GAIT_SPEED_PER_WITHDRAWAL * self.activity_withdrawal
        return GAIT_SPEED_PER_EXERTION_BOUT * -self.activity_withdrawal


def modality_delta(
    axes: IllnessAxes,
    channel_set: ChannelSet = DEFAULT_CHANNEL_SET,
) -> NDArray[np.float64]:
    """Additive band-channel perturbation implied by ``axes``.

    Channels the fleet has not adopted are skipped, so the same signature is
    valid for every width: a person wearing only a watch sees an all-zero vector
    here and their core-vitals deltas unchanged.
    """
    if axes.is_quiet:
        return channel_set.zeros()
    enteric_scale = (
        BOWEL_BURSTS_PER_ENTERIC if axes.enteric_drive >= 0.0 else BOWEL_BURSTS_PER_ILEUS
    )
    return delta_where_present(
        channel_set,
        {
            "regional_ventilation_heterogeneity": (
                VENTILATION_HETEROGENEITY_PER_PULMONARY * axes.pulmonary_involvement
                + VENTILATION_HETEROGENEITY_PER_ARTIFACT * axes.instrument_artifact
            ),
            "pep_ms": PEP_MS_PER_INFLAMMATION * axes.inflammatory_drive,
            "pwv_m_s": PWV_PER_STIFFENING * axes.arterial_stiffening,
            "bowel_sound_burst_rate": enteric_scale * axes.enteric_drive,
            "gastric_emptying_index": (GASTRIC_MINUTES_PER_INFLAMMATION * axes.inflammatory_drive),
            "step_count": axes.step_delta,
            "sleep_fragmentation_index": (
                SLEEP_FRAGMENTATION_PER_DISTURBANCE * axes.sleep_disturbance
            ),
            "gait_speed_m_s": axes.gait_speed_delta,
            "stride_time_variability": STRIDE_CV_PER_FATIGUE * axes.neuromotor_fatigue,
            "gait_asymmetry": GAIT_ASYMMETRY_PER_ARTIFACT * axes.instrument_artifact,
            "cough_rate": COUGHS_PER_IRRITATION * axes.airway_irritation,
            "speech_pause_ratio": SPEECH_PAUSE_PER_PULMONARY * axes.pulmonary_involvement,
            "adventitious_breath_fraction": (
                ADVENTITIOUS_PER_PULMONARY * axes.pulmonary_involvement
                + ADVENTITIOUS_PER_ARTIFACT * axes.instrument_artifact
            ),
            "heart_sound_s1_s2_ratio": (S1_S2_RATIO_PER_INFLAMMATION * axes.inflammatory_drive),
        },
    )


def delta_where_present(
    channel_set: ChannelSet,
    values: dict[str, float],
) -> NDArray[np.float64]:
    """Like ``ChannelSet.delta`` but ignoring channels the set does not carry."""
    return channel_set.delta(
        {name: value for name, value in values.items() if channel_set.has(name)}
    )


def incubation_axes(progress: float) -> IllnessAxes:
    """Pre-symptomatic axes; a faint autonomic shift, no consolidation yet.

    ``progress`` runs 0 to 1 across the incubation period.
    """
    ramp = _clamp_unit(progress)
    return IllnessAxes(
        inflammatory_drive=0.15 * ramp,
        arterial_stiffening=0.2 * ramp,
        # Prodromal malaise runs ahead of fever: people slow down and sleep
        # badly before they are measurably hot.
        activity_withdrawal=0.3 * ramp,
        sleep_disturbance=0.35 * ramp,
        neuromotor_fatigue=0.25 * ramp,
        # A dry tickly cough is one of the earliest reported symptoms, well
        # ahead of any measurable fever.
        airway_irritation=0.35 * ramp,
    )


def infection_axes(progress: float, enteric_involvement: float = 0.0) -> IllnessAxes:
    """Symptomatic infection axes.

    ``progress`` runs 0 to 1 as symptoms ramp. ``enteric_involvement`` is the
    pathogen's gut tropism, 0 for a purely respiratory virus and 1 for a
    gastroenteritis-dominant one; it trades pulmonary for enteric involvement
    rather than adding to it, so a norovirus does not also consolidate a lung.
    """
    ramp = _clamp_unit(progress)
    enteric = _clamp_unit(enteric_involvement)
    return IllnessAxes(
        inflammatory_drive=ramp,
        pulmonary_involvement=ramp * (1.0 - enteric),
        enteric_drive=ramp * enteric,
        arterial_stiffening=ramp,
        activity_withdrawal=ramp,
        sleep_disturbance=ramp,
        neuromotor_fatigue=ramp,
        # Respiratory tropism drives the cough, so a gastroenteritis-dominant
        # pathogen leaves the acoustic patch comparatively quiet. Held below
        # full scale because an inhaled chemical irritant coughs harder than a
        # systemic viral illness does.
        airway_irritation=0.7 * ramp * (1.0 - enteric),
    )


def irritant_axes(effect: float) -> IllnessAxes:
    """Toxin-exposure axes: airway involvement and sympathetic tone, no fever.

    ``effect`` is the dose-response fraction, 0 to 1. Leaving
    ``inflammatory_drive`` at zero keeps the band channels consistent with the
    core vitals' key differentiator, which is the absence of a temperature rise:
    PEP and gastric emptying stay put while ventilation heterogeneity moves.
    """
    dose = _clamp_unit(effect)
    return IllnessAxes(
        # No sickness behaviour either: an irritant plume causes acute airway
        # symptoms, not days of malaise, so the behavioural channels stay put
        # and remain a toxin-versus-disease discriminator.
        pulmonary_involvement=0.8 * dose,
        arterial_stiffening=0.5 * dose,
        # A chemical irritant coughs *harder* than a fever does. Cough is
        # therefore not a disease-versus-toxin discriminator on its own; the
        # absent inflammatory drive still is.
        airway_irritation=dose,
    )


def exertion_axes(intensity: float) -> IllnessAxes:
    """Benign exercise axes: real physiology that mimics part of a fever.

    Exertion shortens PEP, stiffens arteries and stirs the gut, so the band
    channels are not a free discriminator between illness and a hard run; the
    ventilation field stays comparatively even, which is what separates them.
    """
    level = _clamp_unit(intensity)
    return IllnessAxes(
        inflammatory_drive=0.5 * level,
        pulmonary_involvement=0.1 * level,
        enteric_drive=0.3 * level,
        arterial_stiffening=0.6 * level,
        # Negative withdrawal: the bout adds steps rather than removing them.
        activity_withdrawal=-0.6 * level,
        sleep_disturbance=-0.1 * level,
        # A hard bout degrades stride regularity while the wearer is moving
        # faster, so gait speed and stride variability disagree in sign here and
        # agree under illness. That disagreement is the discriminator.
        neuromotor_fatigue=0.5 * level,
        # Exertional dyspnoea, not a tussive stimulus: speech fragments through
        # pulmonary involvement while the cough channel barely moves.
        airway_irritation=0.1 * level,
    )


def sleep_disruption_axes(intensity: float) -> IllnessAxes:
    """Benign disrupted-night axes: a fragmented night and a slow day after.

    A bad night is the sleep channel's dominant confounder, and it drags the
    next day's step count down a little, so neither behavioural channel is a
    free illness detector on its own.
    """
    level = _clamp_unit(intensity)
    return IllnessAxes(
        sleep_disturbance=level,
        activity_withdrawal=0.3 * level,
        neuromotor_fatigue=0.4 * level,
    )


def contact_artifact_axes(intensity: float) -> IllnessAxes:
    """Sensor-artifact axes: contact loss and garment friction, not physiology.

    Everything here runs through ``instrument_artifact`` rather than through any
    physiological axis, so an artifact can only reach the channels whose
    transducer it actually corrupts: the impedance field, the insole's
    left-right balance, and a contact microphone's crackle count. It cannot
    invent a fever, a cough, or a heart-sound amplitude ratio.
    """
    return IllnessAxes(instrument_artifact=0.6 * _clamp_unit(intensity))


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "ADVENTITIOUS_PER_ARTIFACT",
    "ADVENTITIOUS_PER_PULMONARY",
    "BOWEL_BURSTS_PER_ENTERIC",
    "BOWEL_BURSTS_PER_ILEUS",
    "COUGHS_PER_IRRITATION",
    "GAIT_ASYMMETRY_PER_ARTIFACT",
    "GAIT_SPEED_PER_EXERTION_BOUT",
    "GAIT_SPEED_PER_WITHDRAWAL",
    "GASTRIC_MINUTES_PER_INFLAMMATION",
    "PEP_MS_PER_INFLAMMATION",
    "PWV_PER_STIFFENING",
    "S1_S2_RATIO_PER_INFLAMMATION",
    "SLEEP_FRAGMENTATION_PER_DISTURBANCE",
    "SPEECH_PAUSE_PER_PULMONARY",
    "STEPS_PER_EXERTION_BOUT",
    "STEPS_PER_WITHDRAWAL",
    "STRIDE_CV_PER_FATIGUE",
    "VENTILATION_HETEROGENEITY_PER_ARTIFACT",
    "VENTILATION_HETEROGENEITY_PER_PULMONARY",
    "IllnessAxes",
    "contact_artifact_axes",
    "delta_where_present",
    "exertion_axes",
    "incubation_axes",
    "infection_axes",
    "irritant_axes",
    "modality_delta",
    "sleep_disruption_axes",
]
