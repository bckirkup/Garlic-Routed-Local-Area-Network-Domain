"""Pathogen library for SEIR calibration and outbreak presets.

Loads benchmark parameter sets from ``data/pathogens.json`` so simulations can
select documented disease profiles via config (``seir.pathogen``) instead of
hand-tuned values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

from garland.hazards import OutbreakSeed, SEIRConfig

STEPS_PER_DAY = 288
REFERENCE_R0 = 2.5
REFERENCE_BETA = 0.015
REFERENCE_GAMMA = 0.000347
_R0_SCALE = REFERENCE_R0 * REFERENCE_GAMMA / REFERENCE_BETA
_LIBRARY_FILENAME = "pathogens.json"


@dataclass(frozen=True)
class PathogenEpidemiology:
    """Literature reference values for a pathogen profile."""

    r0: float
    incubation_days: float
    infectious_days: float


@dataclass(frozen=True)
class PathogenOutbreakDefaults:
    """Suggested seeding parameters when no outbreak block is provided."""

    initial_infected: int = 10
    seed_radius: float = 500.0
    outbreaks: tuple[OutbreakSeed, ...] = ()


@dataclass(frozen=True)
class PathogenProfile:
    """One entry from the bundled pathogen library."""

    pathogen_id: str
    display_name: str
    pathogen_family: str
    description: str
    provenance: str
    references: tuple[str, ...]
    seir: dict[str, float | int]
    epidemiology: PathogenEpidemiology
    default_outbreak: PathogenOutbreakDefaults


@dataclass
class PathogenLibrary:
    """Loaded pathogen catalog."""

    schema_version: int
    step_duration_minutes: int
    steps_per_day: int
    pathogens: dict[str, PathogenProfile] = field(default_factory=dict)


def _library_path() -> Any:
    return resources.files("garland.data").joinpath(_LIBRARY_FILENAME)


def _parse_outbreak_defaults(raw: dict[str, Any] | None) -> PathogenOutbreakDefaults:
    if not raw:
        return PathogenOutbreakDefaults()
    outbreaks_raw = raw.get("outbreaks")
    outbreaks: tuple[OutbreakSeed, ...] = ()
    if outbreaks_raw:
        outbreaks = tuple(OutbreakSeed(**item) for item in outbreaks_raw)
    return PathogenOutbreakDefaults(
        initial_infected=int(raw.get("initial_infected", 10)),
        seed_radius=float(raw.get("seed_radius", 500.0)),
        outbreaks=outbreaks,
    )


def _parse_profile(pathogen_id: str, raw: dict[str, Any]) -> PathogenProfile:
    epidemiology_raw = raw.get("epidemiology", {})
    return PathogenProfile(
        pathogen_id=pathogen_id,
        display_name=str(raw.get("display_name", pathogen_id)),
        pathogen_family=str(raw.get("pathogen_family", "unknown")),
        description=str(raw.get("description", "")),
        provenance=str(raw.get("provenance", "")),
        references=tuple(str(item) for item in raw.get("references", [])),
        seir=dict(raw.get("seir", {})),
        epidemiology=PathogenEpidemiology(
            r0=float(epidemiology_raw.get("r0", 0.0)),
            incubation_days=float(epidemiology_raw.get("incubation_days", 0.0)),
            infectious_days=float(epidemiology_raw.get("infectious_days", 0.0)),
        ),
        default_outbreak=_parse_outbreak_defaults(raw.get("default_outbreak")),
    )


def parse_pathogen_library(payload: dict[str, Any]) -> PathogenLibrary:
    """Parse a pathogen-library mapping (typically loaded from JSON)."""
    pathogens_raw = payload.get("pathogens", {})
    if not isinstance(pathogens_raw, dict):
        raise ValueError("pathogen library must contain a 'pathogens' mapping")

    profiles = {
        pathogen_id: _parse_profile(pathogen_id, profile)
        for pathogen_id, profile in pathogens_raw.items()
    }
    return PathogenLibrary(
        schema_version=int(payload.get("schema_version", 1)),
        step_duration_minutes=int(payload.get("step_duration_minutes", 5)),
        steps_per_day=int(payload.get("steps_per_day", STEPS_PER_DAY)),
        pathogens=profiles,
    )


@lru_cache(maxsize=1)
def load_pathogen_library() -> PathogenLibrary:
    """Load the bundled pathogen library shipped with GARLAND."""
    text = _library_path().read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid pathogen library format in {_LIBRARY_FILENAME}")
    return parse_pathogen_library(payload)


def list_pathogen_ids() -> list[str]:
    """Return sorted pathogen identifiers from the bundled library."""
    return sorted(load_pathogen_library().pathogens)


def get_pathogen(pathogen_id: str) -> PathogenProfile:
    """Return one pathogen profile, raising if the id is unknown."""
    library = load_pathogen_library()
    profile = library.pathogens.get(pathogen_id)
    if profile is None:
        valid = ", ".join(list_pathogen_ids())
        raise ValueError(f"Unknown pathogen {pathogen_id!r}. Expected one of: {valid}")
    return profile


def compartment_rates_from_days(
    incubation_days: float,
    infectious_days: float,
    *,
    steps_per_day: int = STEPS_PER_DAY,
) -> tuple[float, float]:
    """Convert day-scale periods to per-step sigma and gamma rates."""
    if incubation_days <= 0 or infectious_days <= 0:
        raise ValueError("incubation_days and infectious_days must be positive")
    sigma = 1.0 / (incubation_days * steps_per_day)
    gamma = 1.0 / (infectious_days * steps_per_day)
    return sigma, gamma


def beta_from_r0(
    target_r0: float,
    gamma: float,
    *,
    reference_r0: float = REFERENCE_R0,
    reference_beta: float = REFERENCE_BETA,
    reference_gamma: float = REFERENCE_GAMMA,
) -> float:
    """Derive ``beta`` for a target R0 given ``gamma`` and the wild-type scale factor."""
    if target_r0 <= 0 or gamma <= 0:
        raise ValueError("target_r0 and gamma must be positive")
    return target_r0 * gamma * reference_beta / (reference_r0 * reference_gamma)


def estimate_r0(
    beta: float,
    gamma: float,
    *,
    reference_r0: float = REFERENCE_R0,
    reference_beta: float = REFERENCE_BETA,
    reference_gamma: float = REFERENCE_GAMMA,
) -> float:
    """Estimate R0 using the wild-type contact-scaling calibration.

    ``R0 ≈ (reference_r0 * reference_gamma / reference_beta) * beta / gamma``
    """
    if beta < 0 or gamma <= 0:
        raise ValueError("beta must be non-negative and gamma must be positive")
    scale = reference_r0 * reference_gamma / reference_beta
    return scale * beta / gamma


def apply_pathogen_to_seir_data(
    data: dict[str, Any] | None,
    pathogen_id: str,
) -> dict[str, Any]:
    """Merge pathogen defaults into a SEIR config mapping.

    Explicit keys in ``data`` override pathogen defaults. When ``outbreaks`` is
    omitted and ``initial_infected`` is not set, the pathogen's default outbreak
    seed count is applied.
    """
    profile = get_pathogen(pathogen_id)
    merged: dict[str, Any] = dict(profile.seir)

    if data:
        merged.update(data)

    if "outbreaks" not in merged and "initial_infected" not in (data or {}):
        defaults = profile.default_outbreak
        if defaults.outbreaks:
            merged["outbreaks"] = [
                {
                    "outbreak_id": outbreak.outbreak_id,
                    "start_step": outbreak.start_step,
                    "initial_infected": outbreak.initial_infected,
                    "center_x": outbreak.center_x,
                    "center_y": outbreak.center_y,
                    "seed_radius": outbreak.seed_radius,
                }
                for outbreak in defaults.outbreaks
            ]
        else:
            merged["initial_infected"] = defaults.initial_infected

    merged["pathogen"] = pathogen_id
    return merged


def seir_config_from_pathogen(
    pathogen_id: str,
    overrides: dict[str, Any] | None = None,
) -> SEIRConfig:
    """Build an ``SEIRConfig`` from a library pathogen plus optional overrides."""
    payload = apply_pathogen_to_seir_data(overrides, pathogen_id)
    payload.pop("pathogen", None)

    outbreaks_raw = payload.pop("outbreaks", None)
    outbreaks: list[OutbreakSeed] = []
    if outbreaks_raw:
        outbreaks = [OutbreakSeed(**item) for item in outbreaks_raw]
    return SEIRConfig(outbreaks=outbreaks, **payload)
