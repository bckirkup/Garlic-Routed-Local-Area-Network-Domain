"""Host phenotypes that alter susceptibility, presentation, and device needs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from garland.confounders import ConfoundersConfig
from garland.demographics import ADULT, AGE_BANDS, ELDERLY, OLDER_ADULT
from garland.modality_signatures import IllnessAxes

_DIABETIC_AGE_WEIGHTS = {
    "infant": 0.0,
    "child": 0.1,
    ADULT: 1.0,
    OLDER_ADULT: 1.8,
    ELDERLY: 2.2,
}
_LAW_ENFORCEMENT_AGE_WEIGHTS = {ADULT: 1.0, OLDER_ADULT: 1.0}
_ASSISTIVE_AGE_WEIGHTS = {ADULT: 1.0, OLDER_ADULT: 1.0, ELDERLY: 2.0}


@dataclass(frozen=True)
class HostPhenotypeConfig:
    """Population fractions for host phenotypes."""

    enabled: bool = False
    diabetic_fraction: float = 0.10
    law_enforcement_fraction: float = 0.01
    assistive_need_fraction: float = 0.02

    def __post_init__(self) -> None:
        for name, value in (
            ("diabetic_fraction", self.diabetic_fraction),
            ("law_enforcement_fraction", self.law_enforcement_fraction),
            ("assistive_need_fraction", self.assistive_need_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def _weighted_sample(
    weights: NDArray[np.float64],
    fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Draw a fixed-size weighted sample without replacement."""
    target = min(int(round(fraction * len(weights))), int(np.count_nonzero(weights > 0.0)))
    selected = np.zeros(len(weights), dtype=bool)
    if target == 0:
        return selected
    eligible = np.flatnonzero(weights > 0.0)
    probabilities = weights[eligible] / float(weights[eligible].sum())
    selected[rng.choice(eligible, size=target, replace=False, p=probabilities)] = True
    return selected


def _age_weights(
    age_bands: NDArray[np.int8],
    values: dict[str, float],
) -> NDArray[np.float64]:
    return np.asarray(
        [values.get(AGE_BANDS[int(age)], 0.0) for age in age_bands],
        dtype=np.float64,
    )


class HostPhenotypes:
    """Seeded host masks shared by hazards, confounders, and device ownership."""

    def __init__(
        self,
        n_agents: int,
        config: HostPhenotypeConfig,
        rng: np.random.Generator,
        age_bands: NDArray[np.int8] | None,
        confounder_config: ConfoundersConfig,
        demographics_enabled: bool,
    ) -> None:
        if age_bands is None:
            bands = np.full(n_agents, AGE_BANDS.index(ADULT), dtype=np.int8)
        else:
            bands = np.asarray(age_bands)
        if len(bands) != n_agents:
            raise ValueError("age_bands must have one entry per agent")
        if config.enabled:
            diabetic_weights = _age_weights(bands, _DIABETIC_AGE_WEIGHTS)
            law_weights = _age_weights(bands, _LAW_ENFORCEMENT_AGE_WEIGHTS)
            assistive_weights = _age_weights(bands, _ASSISTIVE_AGE_WEIGHTS)
            self.diabetic = _weighted_sample(diabetic_weights, config.diabetic_fraction, rng)
            self.law_enforcement = _weighted_sample(
                law_weights, config.law_enforcement_fraction, rng
            )
            self.assistive_need = _weighted_sample(
                assistive_weights, config.assistive_need_fraction, rng
            )
            if demographics_enabled:
                self.frail_elderly = bands == AGE_BANDS.index(ELDERLY)
            else:
                self.frail_elderly = rng.random(n_agents) < confounder_config.elderly_fraction
        else:
            self.diabetic = np.zeros(n_agents, dtype=bool)
            self.law_enforcement = np.zeros(n_agents, dtype=bool)
            self.assistive_need = np.zeros(n_agents, dtype=bool)
            self.frail_elderly = np.zeros(n_agents, dtype=bool)

    @property
    def hearable_eligible(self) -> NDArray[np.bool_]:
        """Agents whose role or need makes an in-ear device plausible."""
        return self.frail_elderly | self.law_enforcement | self.assistive_need

    def susceptibility_multiplier(self) -> NDArray[np.float64]:
        """Return multiplicative infection susceptibility by agent."""
        multiplier = np.ones(len(self.diabetic), dtype=np.float64)
        multiplier *= np.where(self.diabetic, 1.4, 1.0)
        multiplier *= np.where(self.frail_elderly, 1.6, 1.0)
        return np.minimum(multiplier, 3.0)


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def host_presentation(
    axes: IllnessAxes,
    core: dict[str, float],
    diabetic: bool,
    frail_elderly: bool,
) -> tuple[IllnessAxes, dict[str, float]]:
    """Apply calibrated host-specific illness presentation multipliers."""
    updated_axes = axes
    updated_core = dict(core)
    if diabetic:
        # Testbed calibration: diabetes blunts the febrile inflammatory signal.
        updated_axes = replace(
            updated_axes,
            inflammatory_drive=updated_axes.inflammatory_drive * 0.7,
            hypovolemia=_clamp_unit(updated_axes.hypovolemia * 1.5),
        )
        # Testbed calibration: diabetic dysregulation raises resting tachycardia.
        updated_core["body_temperature"] = updated_core.get("body_temperature", 0.0) * 0.7
        updated_core["heart_rate"] = updated_core.get("heart_rate", 0.0) * 1.1
    if frail_elderly:
        # Testbed calibration: frail elderly hosts often show afebrile illness.
        updated_axes = replace(
            updated_axes,
            inflammatory_drive=updated_axes.inflammatory_drive * 0.5,
            activity_withdrawal=_clamp_unit(updated_axes.activity_withdrawal * 1.4),
            neuromotor_fatigue=_clamp_unit(updated_axes.neuromotor_fatigue * 1.4),
            sleep_disturbance=_clamp_unit(updated_axes.sleep_disturbance * 1.4),
            pulmonary_involvement=_clamp_unit(updated_axes.pulmonary_involvement * 1.3),
        )
        # Testbed calibration: frailty increases atypical respiratory presentation.
        updated_core["body_temperature"] = updated_core.get("body_temperature", 0.0) * 0.5
    return updated_axes, updated_core


__all__ = ["HostPhenotypeConfig", "HostPhenotypes", "host_presentation"]
