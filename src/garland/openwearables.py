"""Open Wearables-compatible export format for GARLAND observations.

Maps GARLAND's 4-dimensional aggregate vectors (HR, HRV RMSSD, RR, temp)
to the Open Wearables timeseries schema used by the unified data model.

Reference: https://openwearables.io/docs/architecture/data-types
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from numpy.typing import NDArray

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
