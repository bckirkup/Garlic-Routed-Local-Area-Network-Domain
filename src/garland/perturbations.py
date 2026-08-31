"""Cause-labelled biometric perturbation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class PerturbationCause(str, Enum):
    """A model-side cause that can contribute to a biometric excursion."""

    DISEASE = "disease"
    TOXIN = "toxin"
    BACKGROUND_ILI = "background_ili"
    EXERCISE = "exercise"
    SLEEP_DISRUPTION = "sleep_disruption"
    SENSOR_ARTIFACT = "sensor_artifact"
    IRRITANT_EXPOSURE = "irritant_exposure"
    ONBOARDING = "onboarding"
    HEAT_WAVE = "heat_wave"
    VENUE_CROWDING = "venue_crowding"
    MEAL_EXCURSION = "meal_excursion"


BENIGN_CAUSES = frozenset(
    {
        PerturbationCause.EXERCISE,
        PerturbationCause.SLEEP_DISRUPTION,
        PerturbationCause.SENSOR_ARTIFACT,
        PerturbationCause.HEAT_WAVE,
        PerturbationCause.VENUE_CROWDING,
        PerturbationCause.BACKGROUND_ILI,
        PerturbationCause.IRRITANT_EXPOSURE,
        PerturbationCause.ONBOARDING,
        PerturbationCause.MEAL_EXCURSION,
    }
)


@dataclass(frozen=True)
class PerturbationContribution:
    """One ordered, additive cause contribution to a four-channel observation."""

    cause: PerturbationCause
    delta: NDArray[np.float64]
