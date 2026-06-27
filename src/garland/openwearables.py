"""Open Wearables-compatible export format for GARLAND observations.

Maps GARLAND's 4-dimensional aggregate vectors (HR, HRV RMSSD, RR, temp)
to the Open Wearables timeseries schema used by the unified data model.

Reference: https://openwearables.io/docs/architecture/data-types
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from garland.paths import resolve_under_base, resolve_user_path

if TYPE_CHECKING:
    from garland.agents import CitizenAgent

STEP_MINUTES = 5

# Index order matches generate_observation output: HR, HRV, RR, Temp
_OBSERVATION_TYPES: tuple[tuple[str, str], ...] = (
    ("heart_rate", "bpm"),
    ("heart_rate_variability_rmssd", "ms"),
    ("respiratory_rate", "brpm"),
    ("body_temperature", "°C"),
)


def observation_to_records(
    observation: NDArray[Any],
    timestamp: datetime,
    *,
    zone_offset: str = "+00:00",
    source: str = "garland",
) -> list[dict[str, Any]]:
    """Convert a 4-vector observation to Open Wearables timeseries records."""
    if len(observation) != 4:
        raise ValueError(f"Expected 4-dimensional observation, got {len(observation)}")

    ts = timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    if ts.endswith("+00:00"):
        ts = ts.replace("+00:00", "Z")

    records: list[dict[str, Any]] = []
    for idx, (metric_type, unit) in enumerate(_OBSERVATION_TYPES):
        records.append(
            {
                "timestamp": ts,
                "zone_offset": zone_offset,
                "type": metric_type,
                "value": float(observation[idx]),
                "unit": unit,
                "source": source,
            }
        )
    return records


def export_timeseries_payload(
    records: list[dict[str, Any]],
    *,
    resolution: str = "5min",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, Any]:
    """Wrap records in the Open Wearables timeseries API response shape."""
    def _fmt(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        text = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        return text.replace("+00:00", "Z")

    return {
        "data": records,
        "pagination": {
            "next_cursor": None,
            "previous_cursor": None,
            "has_more": False,
        },
        "metadata": {
            "resolution": resolution,
            "sample_count": len(records),
            "start_time": _fmt(start_time),
            "end_time": _fmt(end_time),
        },
    }


def step_timestamp(start_datetime: datetime, step: int) -> datetime:
    """Wall-clock timestamp for a 0-based simulation step."""
    return start_datetime + timedelta(minutes=STEP_MINUTES * step)


def select_export_agent_indices(
    citizen_agents: list[CitizenAgent],
    max_agents: int | None,
) -> set[int]:
    """Choose which wearable agents to include in an export."""
    if max_agents is None:
        return {agent.idx for agent in citizen_agents}
    return {agent.idx for agent in citizen_agents[:max_agents]}


def append_step_observations(
    records: list[dict[str, Any]],
    citizen_agents: list[CitizenAgent],
    *,
    step: int,
    start_datetime: datetime,
    export_agent_indices: set[int],
    source: str = "garland",
) -> None:
    """Append Open Wearables records for operational wearables at one step."""
    timestamp = step_timestamp(start_datetime, step)
    for agent in citizen_agents:
        if agent.idx not in export_agent_indices or not agent.is_operational:
            continue
        records.extend(
            observation_to_records(
                agent.last_observation,
                timestamp,
                source=source,
            )
        )


def build_simulation_timeseries(
    records: list[dict[str, Any]],
    *,
    start_datetime: datetime,
    n_steps: int,
    resolution: str = "5min",
) -> dict[str, Any]:
    """Wrap accumulated simulation records in the Open Wearables payload shape."""
    start_time = step_timestamp(start_datetime, 0)
    end_time = step_timestamp(start_datetime, max(n_steps - 1, 0))
    return export_timeseries_payload(
        records,
        resolution=resolution,
        start_time=start_time,
        end_time=end_time,
    )


def resolve_export_path(path: str, output_dir: Path) -> Path:
    """Resolve a relative export path under the run output directory."""
    export_path = Path(path)
    if export_path.is_absolute():
        return resolve_user_path(export_path)
    return resolve_under_base(output_dir, export_path)


def write_simulation_timeseries(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write an Open Wearables timeseries payload to disk."""
    safe_path = resolve_user_path(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
