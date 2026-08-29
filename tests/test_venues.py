"""Tests for structured venues, activity calibration, and venue-aware SEIR."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from garland.config import config_from_dict, load_config_file
from garland.hazards import SEIRConfig, SEIREngine, SEIRState
from garland.simulation import GarlandModel, SimulationConfig
from garland.venues import (
    _DEFAULT_DWELL_PROFILES,
    ActivityCalibration,
    ActivityDwellProfile,
    VenueConfig,
    VenueEngine,
    VenueSchedule,
    VenueSystemConfig,
    VenueType,
    _hourly,
    parse_venue_system_config,
)


def _venue_system_config() -> VenueSystemConfig:
    return VenueSystemConfig(
        enabled=True,
        calibration_preset="us_urban_weekday",
        use_proximity_contacts=False,
        use_venue_contacts=True,
        venues=[
            VenueConfig(
                venue_id="school_a",
                venue_type=VenueType.SCHOOL.value,
                center_x=500.0,
                center_y=500.0,
                radius=100.0,
                contact_multiplier=4.0,
                schedule=VenueSchedule(weekdays=[0, 1, 2, 3, 4], start_hour=8, end_hour=15),
            ),
            VenueConfig(
                venue_id="hospital_a",
                venue_type=VenueType.HOSPITAL.value,
                center_x=1500.0,
                center_y=1500.0,
                radius=150.0,
                contact_multiplier=5.0,
            ),
        ],
    )


def _small_venue_sim_config(**kwargs) -> SimulationConfig:
    defaults = {
        "n_agents": 800,
        "wearable_fraction": 0.1,
        "grid_width": 2000.0,
        "grid_height": 2000.0,
        "n_steps": 20,
        "seed": 11,
        "mobility_model": "schedule",
        "seir": SEIRConfig(initial_infected=0, beta=0.5, max_infectious_checks=200),
        "venues": _venue_system_config(),
    }
    defaults.update(kwargs)
    return SimulationConfig(**defaults)


class TestVenueConfig:
    def test_parse_venue_system_from_dict(self):
        data = {
            "enabled": True,
            "calibration_preset": "us_suburban",
            "venues": [
                {
                    "venue_id": "school_1",
                    "venue_type": "school",
                    "center_x": 100.0,
                    "center_y": 200.0,
                    "schedule": {"weekdays": [0, 1], "start_hour": 9, "end_hour": 14},
                }
            ],
        }
        cfg = parse_venue_system_config(data)
        assert cfg.enabled
        assert cfg.calibration_preset == "us_suburban"
        assert len(cfg.venues) == 1
        assert cfg.venues[0].schedule is not None
        assert cfg.venues[0].schedule.weekdays == [0, 1]

    def test_load_examples_venues_yaml(self):
        cfg = load_config_file("examples/venues.yaml")
        assert cfg.venues.enabled
        assert len(cfg.venues.venues) >= 2
        assert cfg.mobility_model == "schedule"


class TestVenueAssignment:
    def test_agents_assigned_to_school_and_hospital(self):
        model = GarlandModel(_small_venue_sim_config())
        engine = model.venue_engine
        assert engine is not None
        school_assigned = int(np.sum(engine.assigned_school >= 0))
        hospital_assigned = int(np.sum(engine.assigned_hospital >= 0))
        assert school_assigned > 0
        assert hospital_assigned > 0

    def test_schedule_moves_agents_to_venue_centers(self):
        model = GarlandModel(_small_venue_sim_config())
        engine = model.venue_engine
        assert engine is not None
        model.current_step = 120  # midday on day 1
        model._update_mobility()
        at_school = int(np.sum(engine.current_venue_idx == 0))
        assert at_school > 0


class TestVenueAwareSEIR:
    def test_venue_transmission_exceeds_baseline(self):
        """Co-located infectious/susceptible pairs at a high-multiplier venue."""
        venues = _venue_system_config()
        engine = VenueEngine(config=venues)
        n = 200
        rng = np.random.default_rng(3)
        agent_x = rng.uniform(0, 2000, n).astype(np.float32)
        agent_y = rng.uniform(0, 2000, n).astype(np.float32)
        household_ids = np.arange(n, dtype=np.int64)
        engine.initialize(n, rng, agent_x, agent_y, household_ids)

        current_venue_idx = np.full(n, 0, dtype=np.int32)
        multipliers = [v.effective_contact_multiplier() for v in engine.venues]

        seir_venue = SEIREngine(config=SEIRConfig(beta=0.2, initial_infected=0))
        seir_venue.initialize(n, rng, agent_x, agent_y)
        seir_venue.states[:] = SEIRState.SUSCEPTIBLE
        seir_venue.states[:20] = SEIRState.INFECTIOUS

        for _ in range(5):
            seir_venue.step(
                0,
                agent_x,
                agent_y,
                rng,
                current_venue_idx=current_venue_idx,
                venue_contact_multipliers=multipliers,
                use_proximity_contacts=False,
                use_venue_contacts=True,
            )
        venue_exposed = int(np.sum(seir_venue.states == SEIRState.EXPOSED))

        seir_base = SEIREngine(config=SEIRConfig(beta=0.2, initial_infected=0))
        seir_base.initialize(n, rng, agent_x, agent_y)
        seir_base.states[:] = SEIRState.SUSCEPTIBLE
        seir_base.states[:20] = SEIRState.INFECTIOUS
        spread_x = agent_x.copy()
        spread_y = agent_y.copy()
        spread_x[:20] = 10.0
        spread_y[:20] = 10.0
        spread_x[20:] = 5000.0
        spread_y[20:] = 5000.0

        for _ in range(5):
            seir_base.step(
                0,
                spread_x,
                spread_y,
                rng,
                use_proximity_contacts=True,
                use_venue_contacts=False,
            )
        baseline_exposed = int(np.sum(seir_base.states == SEIRState.EXPOSED))

        assert venue_exposed > baseline_exposed

    def test_simulation_with_venues_runs(self):
        model = GarlandModel(_small_venue_sim_config())
        metrics = model.run(steps=10)
        assert model.current_step == 10
        assert len(metrics.step_records) == 10


class TestActivityCalibration:
    def test_default_dwell_profiles_have_valid_hourly_weights(self):
        for profile in _DEFAULT_DWELL_PROFILES.values():
            for hours in (profile.weekday_hours, profile.weekend_hours):
                weights = np.asarray(hours, dtype=float)
                assert len(weights) == 24
                assert np.all(np.isfinite(weights))
                assert np.all(weights >= 0.0)

    def test_hourly_rejects_short_weekend_profile(self):
        with pytest.raises(ValueError, match="Weekend.*24"):
            _hourly([0.0] * 24, [0.0] * 23)

    @pytest.mark.parametrize("field", ["weekday_hours", "weekend_hours"])
    def test_dwell_profile_rejects_short_table(self, field):
        values = {"weekday_hours": [0.0] * 24, "weekend_hours": [0.0] * 24}
        values[field] = [0.0] * 23
        day_class = "Weekday" if field == "weekday_hours" else "Weekend"
        with pytest.raises(ValueError, match=f"{day_class}.*24"):
            ActivityDwellProfile(**values)

    def test_schedule_mobility_crosses_into_weekend(self):
        venue = VenueConfig(
            venue_id="third_place",
            venue_type=VenueType.THIRD_PLACE.value,
            center_x=500.0,
            center_y=500.0,
            radius=100.0,
            schedule=VenueSchedule(),
        )
        config = _small_venue_sim_config(
            n_agents=32,
            n_steps=4,
            start_datetime=datetime(2024, 7, 19, 23, 50),
            venues=VenueSystemConfig(
                enabled=True,
                calibration_preset="college_town",
                use_proximity_contacts=False,
                use_venue_contacts=True,
                venues=[venue],
            ),
        )
        model = GarlandModel(config)
        model.run(steps=4)

        assert model.current_step == 4
        profile = _DEFAULT_DWELL_PROFILES[VenueType.THIRD_PLACE.value]
        assert profile.weight(23, is_weekend=True) > 0.0

    def test_preset_resolves_fractions(self):
        cfg = VenueSystemConfig(calibration_preset="weekend_leisure")
        cal = cfg.resolved_calibration()
        assert cal.shopping_fraction > 0.4

    def test_college_town_preset_resolves_authored_fractions(self):
        cal = VenueSystemConfig(calibration_preset="college_town").resolved_calibration()
        values = (
            cal.workplace_fraction,
            cal.school_fraction,
            cal.hospital_worker_fraction,
            cal.hospital_patient_fraction,
            cal.third_place_fraction,
            cal.shopping_fraction,
            cal.sporting_event_fraction,
            cal.extended_family_fraction,
            cal.gathering_fraction,
        )
        assert all(0.0 <= value <= 1.0 for value in values)
        assert values == pytest.approx((0.30, 0.42, 0.06, 0.01, 0.45, 0.30, 0.35, 0.10, 0.22))

    def test_custom_dwell_profile(self):
        profile = ActivityDwellProfile(weekday_hours=[0.0] * 10 + [1.0] * 8 + [0.0] * 6)
        cal = ActivityCalibration(dwell_profiles={VenueType.WORKPLACE.value: profile})
        assert cal.profile(VenueType.WORKPLACE).weight(12, False) == pytest.approx(1.0)

    def test_unknown_preset_raises(self):
        cfg = VenueSystemConfig(calibration_preset="invalid")
        with pytest.raises(ValueError, match="Unknown calibration_preset"):
            cfg.resolved_calibration()

    def test_config_round_trip_includes_venues(self):
        original = _small_venue_sim_config()
        from garland.config import config_to_dict

        restored = config_from_dict(config_to_dict(original))
        assert restored.venues.enabled == original.venues.enabled
        assert len(restored.venues.venues) == len(original.venues.venues)
