"""Structured venues and activity-based mobility calibration.

Extends the spatial model with optional structured locations (schools,
workplaces, hospitals, third places, shopping, sporting events, extended
family, gatherings) whose schedules and contact rates can be calibrated to
cell-phone mobility and time-use survey data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class VenueType(str, Enum):
    """Structured location categories with distinct mixing intensity."""

    HOME = "home"
    WORKPLACE = "workplace"
    SCHOOL = "school"
    HOSPITAL = "hospital"
    THIRD_PLACE = "third_place"
    SHOPPING = "shopping"
    SPORTING = "sporting_event"
    EXTENDED_FAMILY = "extended_family"
    GATHERING = "gathering"


# Default contact multipliers relative to baseline proximity contacts.
# Hospital and gatherings reflect sustained indoor co-presence; home is lower.
DEFAULT_CONTACT_MULTIPLIERS: dict[VenueType, float] = {
    VenueType.HOME: 1.2,
    VenueType.WORKPLACE: 2.5,
    VenueType.SCHOOL: 3.0,
    VenueType.HOSPITAL: 4.0,
    VenueType.THIRD_PLACE: 1.8,
    VenueType.SHOPPING: 2.0,
    VenueType.SPORTING: 3.5,
    VenueType.EXTENDED_FAMILY: 2.2,
    VenueType.GATHERING: 4.5,
}


@dataclass
class VenueSchedule:
    """Time window when a venue is active.

    Parameters
    ----------
    weekdays : list[int]
        ISO weekdays when the schedule applies (0=Monday … 6=Sunday).
        Empty list means every day.
    start_hour : float
        Inclusive start hour (0–24, fractional allowed).
    end_hour : float
        Exclusive end hour. Values <= ``start_hour`` wrap past midnight.
    """

    weekdays: list[int] = field(default_factory=list)
    start_hour: float = 8.0
    end_hour: float = 17.0

    def is_active(self, weekday: int, hour_of_day: float) -> bool:
        """Return True when ``hour_of_day`` falls inside this schedule."""
        if self.weekdays and weekday not in self.weekdays:
            return False
        if self.end_hour > self.start_hour:
            return self.start_hour <= hour_of_day < self.end_hour
        return hour_of_day >= self.start_hour or hour_of_day < self.end_hour


@dataclass
class VenueConfig:
    """A structured venue node in the contact graph.

    Parameters
    ----------
    venue_id : str
        Unique identifier.
    venue_type : str
        One of :class:`VenueType` values.
    center_x, center_y : float
        Location in simulation meters.
    radius : float
        Co-presence radius for snapping agents and venue-local contacts.
    capacity : int | None
        Optional attendance cap (agents beyond capacity spread at the edge).
    contact_multiplier : float | None
        Transmission scaling vs. baseline ``beta``. ``None`` uses the type default.
    schedule : VenueSchedule | None
        When agents assigned to this venue may attend. ``None`` means always open.
    """

    venue_id: str
    venue_type: str
    center_x: float
    center_y: float
    radius: float = 150.0
    capacity: int | None = None
    contact_multiplier: float | None = None
    schedule: VenueSchedule | None = None

    @property
    def typed(self) -> VenueType:
        return VenueType(self.venue_type)

    def effective_contact_multiplier(self) -> float:
        if self.contact_multiplier is not None:
            return self.contact_multiplier
        return DEFAULT_CONTACT_MULTIPLIERS.get(self.typed, 1.0)


@dataclass
class ActivityDwellProfile:
    """Hourly dwell weights for one activity type on a day class.

    Weights are relative probabilities (need not sum to 1). They are
    intended to be calibrated from cell-phone stay-point studies or
    time-use surveys (ATUS, SafeGraph dwell fractions).
    """

    weekday_hours: list[float] = field(default_factory=lambda: [0.0] * 24)
    weekend_hours: list[float] = field(default_factory=lambda: [0.0] * 24)

    def __post_init__(self) -> None:
        if len(self.weekday_hours) != 24:
            raise ValueError("Weekday dwell profile must have exactly 24 hourly weights")
        if len(self.weekend_hours) != 24:
            raise ValueError("Weekend dwell profile must have exactly 24 hourly weights")

    def weight(self, hour: int, is_weekend: bool) -> float:
        table = self.weekend_hours if is_weekend else self.weekday_hours
        if len(table) != 24:
            raise ValueError("Dwell profile must have exactly 24 hourly weights")
        return float(table[hour % 24])


@dataclass
class ActivityCalibration:
    """Population fractions and hourly dwell curves for activity types.

    Population fractions describe what share of agents *can* be assigned to
    each structured role. Hourly dwell curves describe when those agents are
    likely to be at that activity, enabling calibration to mobility data.
    """

    workplace_fraction: float = 0.55
    school_fraction: float = 0.18
    hospital_worker_fraction: float = 0.08
    hospital_patient_fraction: float = 0.01
    third_place_fraction: float = 0.25
    shopping_fraction: float = 0.35
    sporting_event_fraction: float = 0.15
    extended_family_fraction: float = 0.40
    gathering_fraction: float = 0.10
    dwell_profiles: dict[str, ActivityDwellProfile] = field(default_factory=dict)

    def profile(self, venue_type: VenueType) -> ActivityDwellProfile:
        key = venue_type.value
        if key in self.dwell_profiles:
            return self.dwell_profiles[key]
        return _DEFAULT_DWELL_PROFILES.get(key, ActivityDwellProfile())


def _hourly(
    weekday: list[float],
    weekend: list[float] | None = None,
) -> ActivityDwellProfile:
    if len(weekday) != 24:
        raise ValueError("Weekday dwell profile must have exactly 24 hourly weights")
    weekend_hours = weekend if weekend is not None else weekday
    if len(weekend_hours) != 24:
        raise ValueError("Weekend dwell profile must have exactly 24 hourly weights")
    return ActivityDwellProfile(
        weekday_hours=list(weekday),
        weekend_hours=list(weekend_hours),
    )


# Calibrated dwell curves inspired by US urban cell-phone stay-point patterns
# (home dominance overnight, workplace/school mid-day, retail peaks late afternoon).
_DEFAULT_DWELL_PROFILES: dict[str, ActivityDwellProfile] = {
    VenueType.HOME.value: _hourly(
        [
            0.95,
            0.95,
            0.95,
            0.95,
            0.92,
            0.85,
            0.70,
            0.45,
            0.25,
            0.15,
            0.12,
            0.10,
            0.10,
            0.12,
            0.12,
            0.15,
            0.20,
            0.30,
            0.45,
            0.55,
            0.65,
            0.75,
            0.85,
            0.92,
        ],
        [
            0.92,
            0.92,
            0.90,
            0.90,
            0.88,
            0.85,
            0.80,
            0.70,
            0.55,
            0.45,
            0.40,
            0.38,
            0.38,
            0.40,
            0.42,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.72,
            0.80,
            0.88,
            0.90,
        ],
    ),
    VenueType.WORKPLACE.value: _hourly(
        [0.0] * 7 + [0.15, 0.55, 0.75, 0.80, 0.82, 0.80, 0.75, 0.70, 0.55, 0.30, 0.10] + [0.0] * 6,
        [0.02] * 24,
    ),
    VenueType.SCHOOL.value: _hourly(
        [0.0] * 7 + [0.20, 0.70, 0.85, 0.88, 0.85, 0.80, 0.60, 0.25] + [0.0] * 9,
        [0.01] * 24,
    ),
    VenueType.HOSPITAL.value: _hourly(
        [
            0.25,
            0.22,
            0.20,
            0.20,
            0.22,
            0.28,
            0.35,
            0.45,
            0.55,
            0.60,
            0.62,
            0.62,
            0.62,
            0.62,
            0.62,
            0.60,
            0.58,
            0.55,
            0.50,
            0.45,
            0.40,
            0.35,
            0.30,
            0.27,
        ],
    ),
    VenueType.THIRD_PLACE.value: _hourly(
        (
            [0.01] * 6
            + [0.05, 0.08, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15, 0.18, 0.15, 0.10, 0.06]
            + [0.02] * 5
        ),
        (
            [0.02] * 8
            + [0.08, 0.12, 0.15, 0.15, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
            + [0.02] * 4
        ),
    ),
    VenueType.SHOPPING.value: _hourly(
        (
            [0.0] * 9
            + [0.05, 0.08, 0.10, 0.12, 0.18, 0.25, 0.30, 0.28, 0.20, 0.12, 0.06, 0.02]
            + [0.0] * 3
        ),
        (
            [0.01] * 9
            + [0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.28, 0.22, 0.15, 0.08, 0.04, 0.02]
            + [0.01] * 3
        ),
    ),
    VenueType.SPORTING.value: _hourly(
        [0.0] * 24,
        (
            [0.0] * 10
            + [0.05, 0.08, 0.12, 0.18, 0.35, 0.45, 0.40, 0.25, 0.12, 0.05, 0.02, 0.01]
            + [0.01] * 2
        ),
    ),
    VenueType.EXTENDED_FAMILY.value: _hourly(
        [0.02] * 16 + [0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02],
        (
            [0.03] * 10
            + [0.10, 0.15, 0.20, 0.22, 0.20, 0.18, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03]
            + [0.03] * 2
        ),
    ),
    VenueType.GATHERING.value: _hourly(
        [0.0] * 17 + [0.05, 0.10, 0.15, 0.12, 0.08, 0.04, 0.02],
        (
            [0.01] * 11
            + [0.08, 0.12, 0.18, 0.22, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02, 0.02]
            + [0.01]
        ),
    ),
}

CALIBRATION_PRESETS: dict[str, ActivityCalibration] = {
    "us_urban_weekday": ActivityCalibration(
        workplace_fraction=0.58,
        school_fraction=0.17,
        hospital_worker_fraction=0.09,
        hospital_patient_fraction=0.012,
        third_place_fraction=0.28,
        shopping_fraction=0.38,
        sporting_event_fraction=0.12,
        extended_family_fraction=0.35,
        gathering_fraction=0.08,
    ),
    "us_suburban": ActivityCalibration(
        workplace_fraction=0.52,
        school_fraction=0.19,
        hospital_worker_fraction=0.07,
        hospital_patient_fraction=0.01,
        third_place_fraction=0.20,
        shopping_fraction=0.42,
        sporting_event_fraction=0.18,
        extended_family_fraction=0.45,
        gathering_fraction=0.12,
    ),
    "weekend_leisure": ActivityCalibration(
        workplace_fraction=0.15,
        school_fraction=0.05,
        hospital_worker_fraction=0.06,
        hospital_patient_fraction=0.01,
        third_place_fraction=0.35,
        shopping_fraction=0.55,
        sporting_event_fraction=0.30,
        extended_family_fraction=0.50,
        gathering_fraction=0.20,
    ),
    # College-town calibration: student-heavy, social, and sports-oriented.
    "college_town": ActivityCalibration(
        workplace_fraction=0.30,
        school_fraction=0.42,
        hospital_worker_fraction=0.06,
        hospital_patient_fraction=0.01,
        third_place_fraction=0.45,
        shopping_fraction=0.30,
        sporting_event_fraction=0.35,
        extended_family_fraction=0.10,
        gathering_fraction=0.22,
    ),
}


def _empty_int32() -> NDArray[np.int32]:
    return np.empty(0, dtype=np.int32)


def _empty_float32() -> NDArray[np.float32]:
    return np.empty(0, dtype=np.float32)


@dataclass
class VenueSystemConfig:
    """Top-level structured-venue settings."""

    enabled: bool = False
    venues: list[VenueConfig] = field(default_factory=list)
    calibration: ActivityCalibration = field(default_factory=ActivityCalibration)
    calibration_preset: str | None = None
    use_proximity_contacts: bool = True
    use_venue_contacts: bool = True
    position_jitter_fraction: float = 0.35

    def resolved_calibration(self) -> ActivityCalibration:
        if self.calibration_preset:
            preset = CALIBRATION_PRESETS.get(self.calibration_preset)
            if preset is None:
                valid = ", ".join(sorted(CALIBRATION_PRESETS))
                raise ValueError(
                    f"Unknown calibration_preset {self.calibration_preset!r}. "
                    f"Expected one of: {valid}"
                )
            return preset
        return self.calibration


@dataclass
class VenueEngine:
    """Assigns agents to venues and updates positions from activity schedules."""

    config: VenueSystemConfig
    venues: list[VenueConfig] = field(default_factory=list)
    venue_index_by_id: dict[str, int] = field(default_factory=dict)
    # Per-agent venue assignments (-1 = none) by activity role
    assigned_workplace: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_school: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_hospital: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_third_place: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_shopping: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_sporting: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_extended_family: NDArray[np.int32] = field(default_factory=_empty_int32)
    assigned_gathering: NDArray[np.int32] = field(default_factory=_empty_int32)
    current_venue_idx: NDArray[np.int32] = field(default_factory=_empty_int32)
    home_x: NDArray[np.float32] = field(default_factory=_empty_float32)
    home_y: NDArray[np.float32] = field(default_factory=_empty_float32)
    extended_family_home_x: NDArray[np.float32] = field(default_factory=_empty_float32)
    extended_family_home_y: NDArray[np.float32] = field(default_factory=_empty_float32)
    venue_center_x: NDArray[np.float32] = field(default_factory=_empty_float32)
    venue_center_y: NDArray[np.float32] = field(default_factory=_empty_float32)
    venue_radius: NDArray[np.float32] = field(default_factory=_empty_float32)
    _calibration: ActivityCalibration = field(default_factory=ActivityCalibration, repr=False)
    _venues_by_type: dict[VenueType, list[int]] = field(default_factory=dict, repr=False)

    def initialize(
        self,
        n_agents: int,
        rng: np.random.Generator,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        household_ids: NDArray[np.int64],
        household_centroid_x: NDArray[np.float32] | None = None,
        household_centroid_y: NDArray[np.float32] | None = None,
    ) -> None:
        """Build venue assignments and home anchors."""
        self.venues = list(self.config.venues)
        self.venue_index_by_id = {v.venue_id: i for i, v in enumerate(self.venues)}
        self.venue_center_x = np.asarray([v.center_x for v in self.venues], dtype=np.float32)
        self.venue_center_y = np.asarray([v.center_y for v in self.venues], dtype=np.float32)
        self.venue_radius = np.asarray([v.radius for v in self.venues], dtype=np.float32)
        self._calibration = self.config.resolved_calibration()
        self._venues_by_type = {}
        for idx, venue in enumerate(self.venues):
            self._venues_by_type.setdefault(venue.typed, []).append(idx)

        none = np.full(n_agents, -1, dtype=np.int32)
        self.assigned_workplace = none.copy()
        self.assigned_school = none.copy()
        self.assigned_hospital = none.copy()
        self.assigned_third_place = none.copy()
        self.assigned_shopping = none.copy()
        self.assigned_sporting = none.copy()
        self.assigned_extended_family = none.copy()
        self.assigned_gathering = none.copy()
        self.current_venue_idx = np.full(n_agents, -1, dtype=np.int32)

        if household_centroid_x is not None and household_centroid_y is not None:
            self.home_x = household_centroid_x[household_ids].astype(np.float32)
            self.home_y = household_centroid_y[household_ids].astype(np.float32)
        else:
            self.home_x = agent_x.copy()
            self.home_y = agent_y.copy()

        self.extended_family_home_x = np.zeros(n_agents, dtype=np.float32)
        self.extended_family_home_y = np.zeros(n_agents, dtype=np.float32)

        cal = self._calibration
        self._assign_role(self.assigned_workplace, cal.workplace_fraction, VenueType.WORKPLACE, rng)
        self._assign_role(self.assigned_school, cal.school_fraction, VenueType.SCHOOL, rng)
        hospital_fraction = cal.hospital_worker_fraction + cal.hospital_patient_fraction
        self._assign_role(self.assigned_hospital, hospital_fraction, VenueType.HOSPITAL, rng)
        self._assign_role(
            self.assigned_third_place, cal.third_place_fraction, VenueType.THIRD_PLACE, rng
        )
        self._assign_role(self.assigned_shopping, cal.shopping_fraction, VenueType.SHOPPING, rng)
        self._assign_role(
            self.assigned_sporting, cal.sporting_event_fraction, VenueType.SPORTING, rng
        )
        self._assign_role(
            self.assigned_extended_family,
            cal.extended_family_fraction,
            VenueType.EXTENDED_FAMILY,
            rng,
        )
        self._assign_role(self.assigned_gathering, cal.gathering_fraction, VenueType.GATHERING, rng)
        self._init_extended_family_homes(rng)

    def _assign_role(
        self,
        target: NDArray[np.int32],
        fraction: float,
        venue_type: VenueType,
        rng: np.random.Generator,
    ) -> None:
        venue_indices = self._venues_by_type.get(venue_type, [])
        if not venue_indices or fraction <= 0:
            return
        n_agents = len(target)
        mask = rng.random(n_agents) < fraction
        chosen_agents = np.nonzero(mask)[0]
        if len(chosen_agents) == 0:
            return
        venue_picks = rng.choice(venue_indices, len(chosen_agents))
        target[chosen_agents] = venue_picks.astype(np.int32)

    def _init_extended_family_homes(self, rng: np.random.Generator) -> None:
        """Pick a visit location for extended-family trips (another home cluster)."""
        n = len(self.extended_family_home_x)
        for i in range(n):
            if self.assigned_extended_family[i] < 0:
                continue
            self.extended_family_home_x[i] = float(
                rng.uniform(0.1, 0.9) * (self.home_x[i] + 1500.0)
            )
            self.extended_family_home_y[i] = float(
                rng.uniform(0.1, 0.9) * (self.home_y[i] + 1200.0)
            )

    def update_positions(
        self,
        agent_x: NDArray[np.float32],
        agent_y: NDArray[np.float32],
        hour_of_day: float,
        weekday: int,
        rng: np.random.Generator,
        grid_width: float,
        grid_height: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Snap agents to scheduled venues or home based on dwell calibration."""
        del agent_y  # Retained for compatibility with the mobility interface.
        n = len(agent_x)
        hour = int(hour_of_day) % 24
        is_weekend = weekday >= 5
        weights, venue_indices, centers_x, centers_y, radii = self._destination_matrices(
            hour, weekday, is_weekend, n
        )
        totals = weights.sum(axis=1)
        active = totals > 0.0
        cdf = np.zeros_like(weights)
        cdf[active] = np.cumsum(weights[active], axis=1) / totals[active, None]
        choices = np.zeros(n, dtype=np.intp)
        u = rng.random(n)
        choices[active] = (cdf[active] < u[active, None]).sum(axis=1)
        choices = np.minimum(choices, 8)

        selected = choices[:, None]
        self.current_venue_idx[:] = np.take_along_axis(venue_indices, selected, axis=1)[:, 0]
        cx = np.take_along_axis(centers_x, selected, axis=1)[:, 0]
        cy = np.take_along_axis(centers_y, selected, axis=1)[:, 0]
        radius = np.take_along_axis(radii, selected, axis=1)[:, 0]
        jitter = radius * self.config.position_jitter_fraction
        new_x = np.clip(cx + rng.normal(0.0, 1.0, n) * jitter, 0, grid_width).astype(np.float32)
        new_y = np.clip(cy + rng.normal(0.0, 1.0, n) * jitter, 0, grid_height).astype(np.float32)
        return new_x, new_y

    def _destination_matrices(
        self,
        hour: int,
        weekday: int,
        is_weekend: bool,
        n: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Build candidate weights and destinations for every agent."""
        weights = np.zeros((n, 9), dtype=np.float64)
        venue_indices = np.full((n, 9), -1, dtype=np.int32)
        centers_x = np.broadcast_to(self.home_x[:, None], (n, 9)).copy()
        centers_y = np.broadcast_to(self.home_y[:, None], (n, 9)).copy()
        radii = np.full((n, 9), 80.0, dtype=np.float32)
        cal = self._calibration
        weights[:, 0] = cal.profile(VenueType.HOME).weight(hour, is_weekend)

        venue_active = np.ones(len(self.venues), dtype=bool)
        for idx, venue in enumerate(self.venues):
            if venue.schedule is not None:
                venue_active[idx] = venue.schedule.is_active(weekday, float(hour))

        role_map = [
            (VenueType.WORKPLACE, self.assigned_workplace),
            (VenueType.SCHOOL, self.assigned_school),
            (VenueType.HOSPITAL, self.assigned_hospital),
            (VenueType.THIRD_PLACE, self.assigned_third_place),
            (VenueType.SHOPPING, self.assigned_shopping),
            (VenueType.SPORTING, self.assigned_sporting),
            (VenueType.GATHERING, self.assigned_gathering),
        ]
        if self.venues:
            for column, (venue_type, assignments) in enumerate(role_map, start=1):
                venue_indices_for_agents = assignments
                valid = (venue_indices_for_agents >= 0) & venue_active[
                    np.maximum(venue_indices_for_agents, 0)
                ]
                weight = cal.profile(venue_type).weight(hour, is_weekend)
                valid &= weight > 0.0
                safe_indices = np.maximum(venue_indices_for_agents, 0)
                weights[:, column] = np.where(valid, weight, 0.0)
                venue_indices[:, column] = np.where(valid, venue_indices_for_agents, -1)
                centers_x[:, column] = np.where(
                    valid, self.venue_center_x[safe_indices], self.home_x
                )
                centers_y[:, column] = np.where(
                    valid, self.venue_center_y[safe_indices], self.home_y
                )
                radii[:, column] = np.where(valid, self.venue_radius[safe_indices], 80.0)

        extended = self.assigned_extended_family
        valid_extended = (extended >= 0) & (
            cal.profile(VenueType.EXTENDED_FAMILY).weight(hour, is_weekend) > 0.0
        )
        weights[:, 8] = np.where(
            valid_extended,
            cal.profile(VenueType.EXTENDED_FAMILY).weight(hour, is_weekend),
            0.0,
        )
        venue_indices[:, 8] = np.where(valid_extended, extended, -1)
        centers_x[:, 8] = np.where(valid_extended, self.extended_family_home_x, self.home_x)
        centers_y[:, 8] = np.where(valid_extended, self.extended_family_home_y, self.home_y)
        radii[:, 8] = np.where(valid_extended, 100.0, 80.0)
        return weights, venue_indices, centers_x, centers_y, radii

    def agents_at_venue(self, venue_idx: int) -> NDArray[np.intp]:
        """Return agent indices currently co-located at a venue."""
        return np.nonzero(self.current_venue_idx == venue_idx)[0]

    def venue_contact_multiplier(self, venue_idx: int) -> float:
        if venue_idx < 0:
            return 1.0
        return self.venues[venue_idx].effective_contact_multiplier()


def parse_venue_schedule(data: dict[str, Any] | None) -> VenueSchedule | None:
    if data is None:
        return None
    return VenueSchedule(**data)


def parse_venue_config(data: dict[str, Any]) -> VenueConfig:
    payload = dict(data)
    schedule_raw = payload.pop("schedule", None)
    schedule = parse_venue_schedule(schedule_raw)
    return VenueConfig(schedule=schedule, **payload)


def parse_activity_calibration(data: dict[str, Any] | None) -> ActivityCalibration:
    if not data:
        return ActivityCalibration()
    payload = dict(data)
    dwell_raw = payload.pop("dwell_profiles", None)
    dwell_profiles: dict[str, ActivityDwellProfile] = {}
    if dwell_raw:
        for key, profile_data in dwell_raw.items():
            dwell_profiles[key] = ActivityDwellProfile(**profile_data)
    return ActivityCalibration(dwell_profiles=dwell_profiles, **payload)


def parse_venue_system_config(data: dict[str, Any] | None) -> VenueSystemConfig:
    if not data:
        return VenueSystemConfig()
    payload = dict(data)
    venues_raw = payload.pop("venues", [])
    calibration_raw = payload.pop("calibration", None)
    venues = [parse_venue_config(item) for item in venues_raw]
    calibration = parse_activity_calibration(calibration_raw)
    enabled = payload.pop("enabled", bool(venues))
    return VenueSystemConfig(
        enabled=enabled,
        venues=venues,
        calibration=calibration,
        **payload,
    )
