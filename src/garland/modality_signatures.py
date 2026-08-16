"""Illness and confounder signatures for the EIT/acoustic band channels.

The four core wrist channels carry hand-written per-hazard deltas. The band
channels are driven instead by a few latent physiological axes, because several
of them share one cause: a febrile inflammatory surge shortens the pre-ejection
period, stiffens the aorta and delays gastric emptying at the same time, so
writing three independent deltas would let one fever count as three unrelated
detections in the Mahalanobis score. ``ptt_systolic_bp`` reads the same
``arterial_stiffening`` axis as ``pwv_m_s`` for the same reason: they are two
views of one vascular state, so they move together and the empirical covariance
learns that they do.

Per-axis magnitudes are the midpoints of the illness deviations tabulated in
``docs/SENSOR_MODALITIES.md``, reached at an axis value of 1.0. They are
simulation calibration for a testbed, not clinical claims.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

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
# Fraction of the breath cycle spent wheezing at full bronchospasm; reported
# range 0.15 to 0.45.
WHEEZE_FRACTION_PER_OBSTRUCTION = 0.30
# Crackles per breath at full consolidation; reported range 4 to 12.
CRACKLES_PER_CONSOLIDATION = 8.0
# Crackles per breath produced by garment shear against the chest wall, which a
# contact microphone cannot tell from a real reopening transient.
CRACKLES_PER_ARTIFACT = 1.0
# Ratio units of first-to-second heart sound amplitude added by inotropy.
S1_S2_RATIO_PER_INOTROPY = 0.35
# Ratio units *lost* when impaired contractility suppresses S1: the reported
# 0.45-0.65 drop is larger than any inotropic rise, which is what makes this
# channel bidirectional rather than monotone in illness.
S1_S2_RATIO_PER_DYSFUNCTION = -0.55
# Percentage points of post-S2 acoustic energy at full volume overload;
# reported range +5 to +12.
S3_ENERGY_PER_OVERLOAD = 8.5
# Percentage points of bowel-sound duration fraction added by enteritis.
MOTILITY_INDEX_PER_ENTERIC = 13.0
# Percentage points of the same channel removed by a paralytic ileus, i.e. at
# ``enteric_drive`` -1.
MOTILITY_INDEX_PER_ILEUS = 3.8
# Ratio units of cardiac-synchronous impedance pulsatility lost to capillary
# occlusion; reported range -0.06 to -0.09.
PULSATILITY_PER_PERFUSION_DEFICIT = -0.075
# Ratio units of the same channel lost to the perfusion-ventilation mismatch of
# a consolidated lobe. Deliberately a third of the channel's own excursion cut:
# a pneumonia nudges pulsatility without ever tripping it alone.
PULSATILITY_PER_CONSOLIDATION = -0.02
# Relative pelvic conductivity shift at retention; reported range +0.15 to +0.35.
BLADDER_SHIFT_PER_RETENTION = 0.25
# Milliseconds of rate-corrected QT prolonged by systemic inflammation.
QTC_MS_PER_INFLAMMATION = 20.0
# Milliseconds of the same interval prolonged by the electrolyte losses of an
# enteric illness, which is why a two-lead patch sees a gastroenteritis at all.
QTC_MS_PER_ENTERIC_LOSS = 12.0
# Percentage points of premature beats added by adrenergic drive.
ECTOPY_PER_INFLAMMATION = 1.2
# Percentage points of the same channel produced by lead noise and motion
# transients that beat detection cannot tell from a premature complex.
ECTOPY_PER_ARTIFACT = 1.5
# mmHg of cuffless systolic estimate added at full sympathetic vasoconstriction;
# distributive shock is the same magnitude with the opposite sign.
SYSTOLIC_BP_PER_STIFFENING = 12.0


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
        gait asymmetry, false crackle transients and false premature
        beats: no infection touches gait asymmetry at all, which is what makes
        it a negative control.
    airway_irritation : float
        Tussive drive, ``0`` to ``1``. Drives cough rate. Deliberately separate
        from ``pulmonary_involvement``: an irritant plume coughs harder than a
        pneumonia while consolidating nothing, and a consolidated lobe can be
        quiet.
    airway_obstruction : float
        Bronchospasm and secretional narrowing of the conducting airway, ``0``
        to ``1``. Drives the wheeze fraction.
    parenchymal_consolidation : float
        Alveolar filling or collapse, ``0`` to ``1``. Drives the crackle count.
        Split from ``airway_obstruction`` because that is the discrimination a
        contact microphone actually buys: an irritant narrows the airway with a
        clean parenchyma, while a pneumonia fills alveoli, and only the second
        cracks.
    cardiac_contractility : float
        Ventricular contractile state, ``-1`` (impaired, S1 suppressed) to ``1``
        (inotropic surge). The only bidirectional axis with an asymmetric
        magnitude, since dysfunction moves the heart-sound ratio further than
        any fever does.
    volume_overload : float
        Elevated filling pressures, ``0`` to ``1``. Drives the third heart sound.
    pulmonary_perfusion_deficit : float
        Loss of cardiac-synchronous regional perfusion (embolism, microvascular
        thrombosis), ``0`` to ``1``.
    urinary_retention : float
        Bladder distension beyond normal filling, ``0`` to ``1``.

    The last three axes are signature hooks: nothing in the current hazard or
    confounder set drives them, so the channels they own move only through the
    weaker couplings above until an event that warrants them exists.
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
    airway_obstruction: float = 0.0
    parenchymal_consolidation: float = 0.0
    cardiac_contractility: float = 0.0
    volume_overload: float = 0.0
    pulmonary_perfusion_deficit: float = 0.0
    urinary_retention: float = 0.0

    # Axes not listed here run 0 to 1; these run -1 to 1 because they have a
    # meaningful opposite (an exercise bout, an ileus, a failing ventricle).
    _SIGNED_AXES = frozenset(
        {
            "enteric_drive",
            "arterial_stiffening",
            "activity_withdrawal",
            "sleep_disturbance",
            "cardiac_contractility",
        }
    )

    def _axis_values(self) -> tuple[tuple[str, float], ...]:
        """Every axis as a (name, value) pair, in declaration order."""
        return tuple((item.name, float(getattr(self, item.name))) for item in fields(self))

    def __post_init__(self) -> None:
        for name, value in self._axis_values():
            low = -1.0 if name in self._SIGNED_AXES else 0.0
            if not low <= value <= 1.0:
                raise ValueError(f"{name} must lie in [{low}, 1.0], got {value}")

    @property
    def is_quiet(self) -> bool:
        """Whether every axis is at rest, i.e. the signature is all-zero."""
        largest = max(abs(value) for _, value in self._axis_values())
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

    @property
    def s1_s2_ratio_delta(self) -> float:
        """Heart-sound amplitude ratio added by this contractile state.

        Asymmetric on purpose: a failing ventricle suppresses S1 further than a
        fever's inotropy lifts it, so the two directions are not one slope.
        """
        if self.cardiac_contractility >= 0.0:
            return S1_S2_RATIO_PER_INOTROPY * self.cardiac_contractility
        return S1_S2_RATIO_PER_DYSFUNCTION * -self.cardiac_contractility


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
    hypermotile = axes.enteric_drive >= 0.0
    enteric_scale = BOWEL_BURSTS_PER_ENTERIC if hypermotile else BOWEL_BURSTS_PER_ILEUS
    motility_scale = MOTILITY_INDEX_PER_ENTERIC if hypermotile else MOTILITY_INDEX_PER_ILEUS
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
            "acoustic_motility_index": motility_scale * axes.enteric_drive,
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
            "wheeze_duration_fraction": (WHEEZE_FRACTION_PER_OBSTRUCTION * axes.airway_obstruction),
            "crackle_count_per_cycle": (
                CRACKLES_PER_CONSOLIDATION * axes.parenchymal_consolidation
                + CRACKLES_PER_ARTIFACT * axes.instrument_artifact
            ),
            "heart_sound_s1_s2_ratio": axes.s1_s2_ratio_delta,
            "s3_energy_fraction": S3_ENERGY_PER_OVERLOAD * axes.volume_overload,
            "eit_perfusion_pulsatility_ratio": (
                PULSATILITY_PER_PERFUSION_DEFICIT * axes.pulmonary_perfusion_deficit
                + PULSATILITY_PER_CONSOLIDATION * axes.parenchymal_consolidation
            ),
            "bladder_filling_impedance_shift": (
                BLADDER_SHIFT_PER_RETENTION * axes.urinary_retention
            ),
            "qtc_ms": (
                QTC_MS_PER_INFLAMMATION * axes.inflammatory_drive
                + QTC_MS_PER_ENTERIC_LOSS * max(axes.enteric_drive, 0.0)
            ),
            "ectopy_burden": (
                ECTOPY_PER_INFLAMMATION * axes.inflammatory_drive
                + ECTOPY_PER_ARTIFACT * axes.instrument_artifact
            ),
            "ptt_systolic_bp": SYSTOLIC_BP_PER_STIFFENING * axes.arterial_stiffening,
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
        # An early inotropic shift, matching the faint inflammatory drive. No
        # consolidation and no wheeze yet, so the adventitious-sound channels
        # are still silent through the prodrome.
        cardiac_contractility=0.15 * ramp,
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
        # Viral reactive airway disease wheezes, but less than a bronchospastic
        # exacerbation: the parenchymal arm carries most of a pneumonia.
        airway_obstruction=0.3 * ramp * (1.0 - enteric),
        parenchymal_consolidation=0.8 * ramp * (1.0 - enteric),
        # Febrile inotropy, the same surge that shortens the pre-ejection
        # period.
        cardiac_contractility=ramp,
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
        # An irritant narrows the conducting airway hard, and leaves the
        # alveoli alone. ``parenchymal_consolidation`` stays at zero on purpose:
        # a wheeze with no crackles and an intact ventilation field is what
        # excludes pneumonia, which is the discrimination the split buys.
        airway_obstruction=0.7 * dose,
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
        # A hard bout is the benign inotropic case, so the heart-sound ratio is
        # not an illness detector on its own either.
        cardiac_contractility=0.5 * level,
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
    left-right balance, a contact microphone's crackle count, and an electrode's
    premature-beat count. It cannot invent a fever, a cough, a heart-sound
    amplitude ratio, or a blood pressure.
    """
    return IllnessAxes(instrument_artifact=0.6 * _clamp_unit(intensity))


def cardiac_decompensation_axes(severity: float) -> IllnessAxes:
    """Cardiac decompensation axes: the one state that *lowers* S1/S2.

    ``severity`` runs 0 to 1. Impaired contractility suppresses the first heart
    sound while raised filling pressures add a third one and push oedema fluid
    into dependent alveoli, so the same episode moves the heart-sound ratio
    downward and the crackle count upward. No fever, which is what stops this
    looking like an infection to the core vitals.

    Nothing in the hazard or confounder set drives this yet; it is the hook the
    chronic-cardiac population will use.
    """
    level = _clamp_unit(severity)
    return IllnessAxes(
        cardiac_contractility=-level,
        volume_overload=level,
        # Coarse crackles from dependent oedema, without the regional patchiness
        # of a consolidating pneumonia.
        parenchymal_consolidation=0.5 * level,
        pulmonary_involvement=0.3 * level,
    )


def perfusion_deficit_axes(severity: float) -> IllnessAxes:
    """Vascular-occlusion axes: perfusion lost with ventilation preserved.

    ``severity`` runs 0 to 1. This is the only state that drives a band channel
    *down* toward its noise floor without any inflammatory drive, which is what
    separates an embolic event from a consolidating one on the impedance field.
    Not driven by any current hazard.
    """
    level = _clamp_unit(severity)
    return IllnessAxes(pulmonary_perfusion_deficit=level, arterial_stiffening=0.3 * level)


def urinary_retention_axes(severity: float) -> IllnessAxes:
    """Urinary-retention axes: pelvic distension beyond normal filling.

    ``severity`` runs 0 to 1. Normal diurnal filling and voiding already moves
    this channel through its circadian term, so retention is only legible as a
    shift that fails to reset. Not driven by any current hazard.
    """
    return IllnessAxes(urinary_retention=_clamp_unit(severity))


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "BLADDER_SHIFT_PER_RETENTION",
    "BOWEL_BURSTS_PER_ENTERIC",
    "BOWEL_BURSTS_PER_ILEUS",
    "COUGHS_PER_IRRITATION",
    "CRACKLES_PER_ARTIFACT",
    "CRACKLES_PER_CONSOLIDATION",
    "ECTOPY_PER_ARTIFACT",
    "ECTOPY_PER_INFLAMMATION",
    "GAIT_ASYMMETRY_PER_ARTIFACT",
    "GAIT_SPEED_PER_EXERTION_BOUT",
    "GAIT_SPEED_PER_WITHDRAWAL",
    "GASTRIC_MINUTES_PER_INFLAMMATION",
    "MOTILITY_INDEX_PER_ENTERIC",
    "MOTILITY_INDEX_PER_ILEUS",
    "PEP_MS_PER_INFLAMMATION",
    "PULSATILITY_PER_CONSOLIDATION",
    "PULSATILITY_PER_PERFUSION_DEFICIT",
    "PWV_PER_STIFFENING",
    "QTC_MS_PER_ENTERIC_LOSS",
    "QTC_MS_PER_INFLAMMATION",
    "S1_S2_RATIO_PER_DYSFUNCTION",
    "S1_S2_RATIO_PER_INOTROPY",
    "S3_ENERGY_PER_OVERLOAD",
    "SLEEP_FRAGMENTATION_PER_DISTURBANCE",
    "SPEECH_PAUSE_PER_PULMONARY",
    "STEPS_PER_EXERTION_BOUT",
    "STEPS_PER_WITHDRAWAL",
    "STRIDE_CV_PER_FATIGUE",
    "SYSTOLIC_BP_PER_STIFFENING",
    "VENTILATION_HETEROGENEITY_PER_ARTIFACT",
    "VENTILATION_HETEROGENEITY_PER_PULMONARY",
    "WHEEZE_FRACTION_PER_OBSTRUCTION",
    "IllnessAxes",
    "cardiac_decompensation_axes",
    "contact_artifact_axes",
    "delta_where_present",
    "exertion_axes",
    "incubation_axes",
    "infection_axes",
    "irritant_axes",
    "modality_delta",
    "perfusion_deficit_axes",
    "sleep_disruption_axes",
    "urinary_retention_axes",
]
