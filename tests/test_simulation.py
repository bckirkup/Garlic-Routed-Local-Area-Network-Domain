"""Tests for simulation model initialization and basic stepping.

Validates:
1. Model initializes without error at reduced scale
2. Wearable assignment is patchy by household
3. SEIR transitions occur correctly
4. Plume model produces expected concentration patterns
5. End-to-end simulation produces valid metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.biometrics import (
    BaselineTracker,
    BiometricProfile,
    circadian_factor,
    generate_observation,
    generate_profiles,
    seasonal_factor,
)
from garland.hazards import (
    PlumeConfig,
    SEIRConfig,
    SEIREngine,
    SEIRState,
    compute_plume_concentration,
    plume_biometric_perturbation,
)
from garland.simulation import GarlandModel, SimulationConfig


@pytest.fixture
def small_config():
    """Small-scale config for fast testing."""
    return SimulationConfig(
        n_agents=1000,
        wearable_fraction=0.15,
        grid_width=2000.0,
        grid_height=2000.0,
        cell_size=200.0,
        n_steps=50,
        seed=42,
        seir=SEIRConfig(initial_infected=3, beta=0.02),
        plume=PlumeConfig(start_step=10, duration_steps=20),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestModelInitialization:
    """Test model setup."""

    def test_model_creates_without_error(self, small_config):
        """Model should initialize at 1000 agents."""
        model = GarlandModel(small_config)
        assert model is not None
        assert model.current_step == 0

    def test_wearable_count_approximately_correct(self, small_config):
        """Wearable count should be close to wearable_fraction."""
        model = GarlandModel(small_config)
        actual_fraction = np.sum(model.has_wearable) / small_config.n_agents
        assert abs(actual_fraction - small_config.wearable_fraction) < 0.05

    def test_wearable_patchy_by_household(self, small_config):
        """Wearable agents should cluster by household."""
        model = GarlandModel(small_config)
        # Check that agents in same household tend to have same wearable status
        households = model.household_ids
        unique_hh = np.unique(households)
        mixed_count = 0
        for hh in unique_hh[:100]:  # Sample 100 households
            members = np.where(households == hh)[0]
            if len(members) > 1:
                statuses = model.has_wearable[members]
                if not (np.all(statuses) or np.all(~statuses)):
                    mixed_count += 1
        # Most households should be uniform (patchy)
        assert mixed_count < 50

    def test_seir_initial_states(self, small_config):
        """SEIR should start with correct number of infected."""
        model = GarlandModel(small_config)
        infectious = np.sum(model.seir.states == SEIRState.INFECTIOUS)
        assert infectious == small_config.seir.initial_infected

    def test_spatial_grid_has_all_agents(self, small_config):
        """All agents should be assigned to grid cells."""
        model = GarlandModel(small_config)
        total_in_grid = sum(
            len(agents)
            for agents in model.grid._cell_agents.values()
        )
        assert total_in_grid == small_config.n_agents


class TestSEIR:
    """Test SEIR engine."""

    def test_seir_transitions_occur(self, rng):
        """After many steps, some transitions should occur."""
        engine = SEIREngine(config=SEIRConfig(
            beta=0.05, sigma=0.01, gamma=0.005, initial_infected=5
        ))
        n = 500
        engine.initialize(n, rng)

        # Place agents in a cluster
        x = rng.normal(500, 50, n).astype(np.float32)
        y = rng.normal(500, 50, n).astype(np.float32)

        for step in range(100):
            engine.step(step, x, y, rng)

        # Should have some exposed/recovered by now
        exposed = np.sum(engine.states == SEIRState.EXPOSED)
        recovered = np.sum(engine.states == SEIRState.RECOVERED)
        assert exposed + recovered > 0

    def test_seir_biometric_perturbation(self):
        """Infectious agents should show biometric shifts."""
        engine = SEIREngine()
        engine.states = np.array([SEIRState.INFECTIOUS], dtype=np.int8)
        engine.infection_step = np.array([0], dtype=np.int32)

        perturb = engine.biometric_perturbation(0, 576)  # 2 days since infection
        # Should have elevated HR and temp
        assert perturb[0] > 5.0  # HR increase
        assert perturb[3] > 0.5  # Temp increase

    def test_susceptible_no_perturbation(self):
        """Susceptible agents should have no biometric shift."""
        engine = SEIREngine()
        engine.states = np.array([SEIRState.SUSCEPTIBLE], dtype=np.int8)
        engine.infection_step = np.array([-1], dtype=np.int32)

        perturb = engine.biometric_perturbation(0, 0)
        assert np.all(perturb == 0)


class TestPlume:
    """Test plume dispersion model."""

    def test_plume_before_start_is_zero(self):
        """No concentration before plume starts."""
        config = PlumeConfig(start_step=100)
        x = np.array([5100.0], dtype=np.float32)
        y = np.array([5000.0], dtype=np.float32)
        conc = compute_plume_concentration(x, y, config, current_step=50)
        assert conc[0] == 0.0

    def test_plume_after_end_is_zero(self):
        """No concentration after plume ends."""
        config = PlumeConfig(start_step=100, duration_steps=50)
        x = np.array([5100.0], dtype=np.float32)
        y = np.array([5000.0], dtype=np.float32)
        conc = compute_plume_concentration(x, y, config, current_step=200)
        assert conc[0] == 0.0

    def test_plume_downwind_has_concentration(self):
        """Agents downwind should have nonzero concentration."""
        config = PlumeConfig(
            source_x=0.0,
            source_y=500.0,
            wind_direction=0.0,  # East
            start_step=0,
            duration_steps=100,
        )
        # Agent downwind (east of source)
        x = np.array([500.0], dtype=np.float32)
        y = np.array([500.0], dtype=np.float32)
        conc = compute_plume_concentration(x, y, config, current_step=50)
        assert conc[0] > 0.0

    def test_plume_upwind_is_zero(self):
        """Agents upwind should have zero concentration."""
        config = PlumeConfig(
            source_x=500.0,
            source_y=500.0,
            wind_direction=0.0,  # East
            start_step=0,
            duration_steps=100,
        )
        # Agent upwind (west of source)
        x = np.array([100.0], dtype=np.float32)
        y = np.array([500.0], dtype=np.float32)
        conc = compute_plume_concentration(x, y, config, current_step=50)
        assert conc[0] == 0.0

    def test_plume_perturbation_no_fever(self):
        """Plume should cause RR spike but NOT fever."""
        perturb = plume_biometric_perturbation(1.0)
        assert perturb[2] > 5.0  # RR elevated
        assert perturb[3] == 0.0  # No fever


class TestBiometrics:
    """Test biometric generation and baselines."""

    def test_profile_generation(self, rng):
        """Profiles should have physiologically plausible values."""
        profiles = generate_profiles(100, rng)
        assert len(profiles) == 100
        for p in profiles:
            assert 50 <= p.resting_hr <= 110
            assert 10 <= p.resting_hrv <= 120
            assert 8 <= p.resting_rr <= 25
            assert 35.5 <= p.resting_temp <= 38.0

    def test_circadian_factor_range(self):
        """Circadian factor should be in [-1, 1]."""
        for h in range(24):
            assert -1.0 <= circadian_factor(float(h)) <= 1.0

    def test_seasonal_factor_range(self):
        """Seasonal factor should be in [-1, 1]."""
        for d in range(1, 366):
            assert -1.0 <= seasonal_factor(d) <= 1.0

    def test_observation_dimensions(self, rng):
        """Observations should be 4-dimensional."""
        profile = BiometricProfile(
            resting_hr=72, resting_hrv=42, resting_rr=15, resting_temp=36.8
        )
        obs = generate_observation(profile, 12.0, 180, rng)
        assert obs.shape == (4,)

    def test_baseline_mahalanobis_distance(self, rng):
        """Normal observations should have low Mahalanobis distance."""
        tracker = BaselineTracker()
        profile = BiometricProfile(
            resting_hr=72, resting_hrv=42, resting_rr=15, resting_temp=36.8
        )

        # Train baseline with normal data
        for _ in range(100):
            obs = generate_observation(profile, 12.0, 180, rng)
            tracker.update(obs, 12, 6)

        # A normal observation should have low distance
        normal_obs = generate_observation(profile, 12.0, 180, rng)
        dist = tracker.mahalanobis_distance(normal_obs, 12, 6)
        assert dist < 5.0

    def test_anomalous_observation_high_distance(self, rng):
        """Anomalous observations should have high Mahalanobis distance."""
        tracker = BaselineTracker()
        profile = BiometricProfile(
            resting_hr=72, resting_hrv=42, resting_rr=15, resting_temp=36.8
        )

        # Train baseline
        for _ in range(100):
            obs = generate_observation(profile, 12.0, 180, rng)
            tracker.update(obs, 12, 6)

        # Create anomalous observation (simulating fever)
        anomalous = np.array([95.0, 20.0, 22.0, 38.5], dtype=np.float64)
        dist = tracker.mahalanobis_distance(anomalous, 12, 6)
        assert dist > 3.0


class TestEndToEnd:
    """Test full simulation pipeline."""

    def test_simulation_runs_without_error(self, small_config):
        """Full simulation should complete without crashing."""
        model = GarlandModel(small_config)
        metrics = model.run()
        assert len(metrics.step_records) == small_config.n_steps

    def test_metrics_have_expected_fields(self, small_config):
        """Metrics should contain all required fields."""
        model = GarlandModel(small_config)
        metrics = model.run()
        df = metrics.to_dataframe()

        expected_cols = [
            "step",
            "time_hours",
            "susceptible",
            "exposed",
            "infectious",
            "recovered",
            "plume_exposed",
            "anomalies_detected",
            "tokens_submitted",
            "cumulative_epsilon",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_seir_conservation(self, small_config):
        """Total population should be conserved in SEIR model."""
        model = GarlandModel(small_config)
        model.run()
        df = model.metrics.to_dataframe()
        for _, row in df.iterrows():
            total = row["susceptible"] + row["exposed"] + row["infectious"] + row["recovered"]
            assert total == small_config.n_agents

    def test_summary_has_detection_metrics(self, small_config):
        """Summary should include detection and privacy metrics."""
        model = GarlandModel(small_config)
        model.run()
        summary = model.metrics.summary()

        assert "fpr_disease" in summary
        assert "fnr_disease" in summary
        assert "fpr_toxin" in summary
        assert "total_epsilon" in summary
        assert "discrimination_score" in summary
