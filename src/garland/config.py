"""Configuration loading for the GARLAND simulation.

Supports YAML and TOML files that map to ``SimulationConfig``, with optional
CLI overrides applied on top of file-based settings.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from garland.attacks import AttackConfig, AttackType
from garland.hazards import PlumeConfig, SEIRConfig
from garland.privacy import PrivacyConfig
from garland.simulation import SimulationConfig

_CONFIG_ALIASES: dict[str, str] = {
    "decay_lambda": "baseline_decay_lambda",
    "seasonal_decay": "baseline_seasonal_decay",
}

_ATTACK_ENABLE_FLAGS: dict[str, AttackType] = {
    "enable_sybil": AttackType.SYBIL_INJECTION,
    "enable_deanon": AttackType.TARGETED_QUERY,
    "enable_correlation": AttackType.CORRELATION,
    "enable_eclipse": AttackType.ECLIPSE,
    "enable_replay": AttackType.REPLAY,
}


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML or TOML mapping from disk."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib

            data = tomllib.loads(text)
        else:
            try:
                import tomli
            except ImportError as exc:
                raise ImportError(
                    "tomli is required to load TOML config files on Python < 3.11. "
                    "Install with: pip install tomli"
                ) from exc
            data = tomli.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix} (use .yaml, .yml, or .toml)")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return data


def _parse_attack_types(values: list[Any]) -> list[AttackType]:
    attacks: list[AttackType] = []
    for value in values:
        if isinstance(value, AttackType):
            attacks.append(value)
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid attack type entry: {value!r}")
        try:
            attacks.append(AttackType(value))
        except ValueError as exc:
            valid = ", ".join(item.value for item in AttackType)
            raise ValueError(f"Unknown attack type {value!r}. Expected one of: {valid}") from exc
    return attacks


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Invalid datetime value: {value!r}")


def _build_attack_config(data: dict[str, Any]) -> AttackConfig:
    payload = dict(data)
    active_attacks = payload.pop("active_attacks", None)
    if active_attacks is None:
        active_attacks = []
        for flag, attack_type in _ATTACK_ENABLE_FLAGS.items():
            if payload.pop(flag, False):
                active_attacks.append(attack_type)
    else:
        for flag in _ATTACK_ENABLE_FLAGS:
            payload.pop(flag, None)
        active_attacks = _parse_attack_types(list(active_attacks))

    eclipse_zones = payload.pop("eclipse_zones", payload.pop("eclipse_target_zones", None))
    if eclipse_zones is None:
        eclipse_zones = []
    elif isinstance(eclipse_zones, str):
        eclipse_zones = [int(item.strip()) for item in eclipse_zones.split(",") if item.strip()]
    else:
        eclipse_zones = [int(item) for item in eclipse_zones]

    return AttackConfig(
        active_attacks=active_attacks,
        eclipse_target_zones=eclipse_zones,
        **payload,
    )


def _build_subconfig(
    cls: type[SEIRConfig | PlumeConfig | PrivacyConfig | AttackConfig],
    data: dict[str, Any] | None,
) -> SEIRConfig | PlumeConfig | PrivacyConfig | AttackConfig:
    if not data:
        return cls()
    if cls is AttackConfig:
        return _build_attack_config(data)
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> SimulationConfig:
    """Build a ``SimulationConfig`` from a nested mapping."""
    payload = deepcopy(data)
    for alias, field_name in _CONFIG_ALIASES.items():
        if alias in payload:
            payload[field_name] = payload.pop(alias)

    seir = payload.pop("seir", None)
    plume = payload.pop("plume", None)
    privacy = payload.pop("privacy", None)
    attacks = payload.pop("attacks", None)

    if "start_datetime" in payload:
        payload["start_datetime"] = _parse_datetime(payload["start_datetime"])

    return SimulationConfig(
        seir=_build_subconfig(SEIRConfig, seir),  # type: ignore[arg-type]
        plume=_build_subconfig(PlumeConfig, plume),  # type: ignore[arg-type]
        privacy=_build_subconfig(PrivacyConfig, privacy),  # type: ignore[arg-type]
        attacks=_build_subconfig(AttackConfig, attacks),  # type: ignore[arg-type]
        **payload,
    )


def load_config_file(path: str | Path) -> SimulationConfig:
    """Load a simulation config from a YAML or TOML file."""
    return config_from_dict(_load_mapping(Path(path)))


def _set_nested_value(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested path {path!r}: {part} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override mappings, supporting dotted nested paths."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if "." in key:
            _set_nested_value(merged, key, value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = apply_overrides(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Serialize a simulation config to a JSON-compatible mapping."""
    return {
        "n_agents": config.n_agents,
        "wearable_fraction": config.wearable_fraction,
        "grid_width": config.grid_width,
        "grid_height": config.grid_height,
        "cell_size": config.cell_size,
        "spatial_backend": config.spatial_backend,
        "h3_resolution": config.h3_resolution,
        "origin_lat": config.origin_lat,
        "origin_lng": config.origin_lng,
        "mobility_model": config.mobility_model,
        "mobility_speed_m": config.mobility_speed_m,
        "n_steps": config.n_steps,
        "households_per_neighborhood": config.households_per_neighborhood,
        "household_size_mean": config.household_size_mean,
        "start_datetime": config.start_datetime.isoformat(),
        "seed": config.seed,
        "baseline_decay_lambda": config.baseline_decay_lambda,
        "baseline_seasonal_decay": config.baseline_seasonal_decay,
        "seir": {
            "beta": config.seir.beta,
            "sigma": config.seir.sigma,
            "gamma": config.seir.gamma,
            "contact_radius": config.seir.contact_radius,
            "initial_infected": config.seir.initial_infected,
            "max_infectious_checks": config.seir.max_infectious_checks,
        },
        "plume": {
            "source_x": config.plume.source_x,
            "source_y": config.plume.source_y,
            "release_rate": config.plume.release_rate,
            "wind_speed": config.plume.wind_speed,
            "wind_direction": config.plume.wind_direction,
            "stability_class": config.plume.stability_class,
            "start_step": config.plume.start_step,
            "duration_steps": config.plume.duration_steps,
        },
        "privacy": {
            "threshold_m": config.privacy.threshold_m,
            "k_min": config.privacy.k_min,
            "time_window_steps": config.privacy.time_window_steps,
            "epsilon_per_response": config.privacy.epsilon_per_response,
            "randomized_response_p": config.privacy.randomized_response_p,
            "laplace_scale": config.privacy.laplace_scale,
            "dummy_rate": config.privacy.dummy_rate,
        },
        "attacks": {
            "sybil_count": config.attacks.sybil_count,
            "sybil_target_zone": config.attacks.sybil_target_zone,
            "target_agent_idx": config.attacks.target_agent_idx,
            "correlation_window": config.attacks.correlation_window,
            "eclipse_target_zones": list(config.attacks.eclipse_target_zones),
            "eclipse_drop_fraction": config.attacks.eclipse_drop_fraction,
            "replay_interval_steps": config.attacks.replay_interval_steps,
            "replay_lag_bins": config.attacks.replay_lag_bins,
            "replay_count": config.attacks.replay_count,
            "replay_cache_max": config.attacks.replay_cache_max,
            "deanon_interval_steps": config.attacks.deanon_interval_steps,
            "deanon_success_threshold_m": config.attacks.deanon_success_threshold_m,
            "correlation_eval_interval": config.attacks.correlation_eval_interval,
            "correlation_distinguish_threshold_m": (
                config.attacks.correlation_distinguish_threshold_m
            ),
            "active_attacks": [attack.value for attack in config.attacks.active_attacks],
        },
    }
