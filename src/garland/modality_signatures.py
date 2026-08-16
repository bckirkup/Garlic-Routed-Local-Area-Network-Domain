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
    """

    inflammatory_drive: float = 0.0
    pulmonary_involvement: float = 0.0
    enteric_drive: float = 0.0
    arterial_stiffening: float = 0.0

    def __post_init__(self) -> None:
        for name, value, low in (
            ("inflammatory_drive", self.inflammatory_drive, 0.0),
            ("pulmonary_involvement", self.pulmonary_involvement, 0.0),
            ("enteric_drive", self.enteric_drive, -1.0),
            ("arterial_stiffening", self.arterial_stiffening, -1.0),
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
        )
        return largest < _AXIS_REST_TOLERANCE


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
            ),
            "pep_ms": PEP_MS_PER_INFLAMMATION * axes.inflammatory_drive,
            "pwv_m_s": PWV_PER_STIFFENING * axes.arterial_stiffening,
            "bowel_sound_burst_rate": enteric_scale * axes.enteric_drive,
            "gastric_emptying_index": (GASTRIC_MINUTES_PER_INFLAMMATION * axes.inflammatory_drive),
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
        pulmonary_involvement=0.8 * dose,
        arterial_stiffening=0.5 * dose,
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
    )


def contact_artifact_axes(intensity: float) -> IllnessAxes:
    """Sensor-artifact axes: electrode contact loss, not physiology.

    Motion and posture transitions corrupt the impedance field far more than
    they do the derived intervals, so this drives ventilation heterogeneity
    alone.
    """
    return IllnessAxes(pulmonary_involvement=0.5 * _clamp_unit(intensity))


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


__all__ = [
    "BOWEL_BURSTS_PER_ENTERIC",
    "BOWEL_BURSTS_PER_ILEUS",
    "GASTRIC_MINUTES_PER_INFLAMMATION",
    "PEP_MS_PER_INFLAMMATION",
    "PWV_PER_STIFFENING",
    "VENTILATION_HETEROGENEITY_PER_PULMONARY",
    "IllnessAxes",
    "contact_artifact_axes",
    "delta_where_present",
    "exertion_axes",
    "incubation_axes",
    "infection_axes",
    "irritant_axes",
    "modality_delta",
]
