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

from garland.attacks import AttackConfig, AttackType
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
from garland.metrics import DetectionEvent
from garland.privacy import AnomalyType, BroadcastQuery, PerturbedResponse, PrivacyConfig
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
def multi_neighborhood_config():
    """Config with multiple neighborhoods for spatial coherence tests."""
    return SimulationConfig(
        n_agents=1200,
        households_per_neighborhood=50,
        household_size_mean=3,
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
        households = model.household_ids
        unique_hh = np.unique(households)
        mixed_count = 0
        for hh in unique_hh:
            members = np.where(households == hh)[0]
            if len(members) > 1:
                statuses = model.has_wearable[members]
                if not (np.all(statuses) or np.all(~statuses)):
                    mixed_count += 1
        assert mixed_count == 0

    def test_household_members_share_neighborhood(self, multi_neighborhood_config):
        """All members of a household must belong to the same neighborhood."""
        model = GarlandModel(multi_neighborhood_config)
        for hh in np.unique(model.household_ids):
            members = np.where(model.household_ids == hh)[0]
            neighborhoods = model.neighborhood_ids[members]
            assert np.all(neighborhoods == neighborhoods[0])

    def test_wearable_households_within_neighborhood(self, multi_neighborhood_config):
        """Wearable agents must belong to households within a single neighborhood."""
        model = GarlandModel(multi_neighborhood_config)
        for hh in np.unique(model.household_ids):
            members = np.where(model.household_ids == hh)[0]
            if not np.any(model.has_wearable[members]):
                continue
            neighborhoods = model.neighborhood_ids[members]
            assert np.all(neighborhoods == neighborhoods[0])
            assert np.all(model.has_wearable[members])

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

    def test_fn_not_inflated_per_step(self, small_config):
        """False negatives should not multiply-count every undetected step."""
        model = GarlandModel(small_config)
        model.run(steps=small_config.n_steps)
        assert model.metrics.false_negatives_disease < small_config.n_steps
        assert model.metrics.false_negatives_toxin < small_config.n_steps

    def test_summary_broadcast_counts_populated(self, small_config):
        """Summary broadcast/response totals should reflect simulation activity."""
        model = GarlandModel(small_config)
        model.run(steps=small_config.n_steps)
        summary = model.metrics.summary()
        df = model.metrics.to_dataframe()
        assert summary["total_broadcasts"] == int(df["broadcasts_issued"].sum())
        assert summary["total_responses"] == int(df["responses_received"].sum())


class TestDetectionClassification:
    """Test zone-local plume classification in _classify_detection."""

    @staticmethod
    def _plume_config() -> PlumeConfig:
        return PlumeConfig(
            source_x=500.0,
            source_y=500.0,
            wind_direction=0.0,  # East
            start_step=10,
            duration_steps=50,
        )

    @staticmethod
    def _genuine_response() -> PerturbedResponse:
        return PerturbedResponse(
            query_id=0,
            reported_x=0.0,
            reported_y=0.0,
            anomaly_confirmed=True,
            is_dummy=False,
        )

    def _classify_respiratory_at(
        self,
        agent_x: float,
        agent_y: float,
        step: int,
    ) -> DetectionEvent | None:
        config = SimulationConfig(
            n_agents=1,
            grid_width=2000.0,
            grid_height=2000.0,
            cell_size=200.0,
            n_steps=1,
            seed=42,
            plume=self._plume_config(),
            seir=SEIRConfig(initial_infected=0),
        )
        model = GarlandModel(config)
        model.agent_x = np.array([agent_x], dtype=np.float32)
        model.agent_y = np.array([agent_y], dtype=np.float32)
        model.grid.assign_positions(model.agent_x, model.agent_y)
        model.current_step = step

        concentrations = compute_plume_concentration(
            model.agent_x, model.agent_y, model.plume_config, step
        )
        zone_cell = model.grid.cell_of(0)
        query = BroadcastQuery(
            zone_cells=[zone_cell],
            anomaly_type=AnomalyType.RESPIRATORY,
            time_window_start=0,
            time_window_end=1,
        )
        model._classify_detection(query, [self._genuine_response()], concentrations)
        if not model.metrics.detection_events:
            return None
        return model.metrics.detection_events[-1]

    def test_downwind_during_plume_is_toxin_true_positive(self):
        """Downwind agents in an exposed zone should count as toxin TPs."""
        event = self._classify_respiratory_at(agent_x=560.0, agent_y=500.0, step=20)
        assert event is not None
        assert event.hazard_type == "toxin"
        assert event.true_positive is True

    def test_upwind_during_plume_is_not_toxin_true_positive(self):
        """Upwind agents should not be toxin TPs even when the plume is globally active."""
        event = self._classify_respiratory_at(agent_x=100.0, agent_y=500.0, step=20)
        assert event is not None
        assert event.hazard_type == "disease"
        assert event.true_positive is False

    def test_downwind_before_plume_start_is_not_toxin_true_positive(self):
        """Spatial exposure should be zero before the plume begins."""
        event = self._classify_respiratory_at(agent_x=560.0, agent_y=500.0, step=5)
        assert event is not None
        assert event.hazard_type == "disease"
        assert event.true_positive is False

    def test_global_timing_would_misclassify_upwind(self):
        """During plume hours, global timing alone would wrongly label upwind as toxin TP."""
        plume = self._plume_config()
        step = 20
        global_plume_active = (
            step >= plume.start_step and step < plume.start_step + plume.duration_steps
        )
        assert global_plume_active

        event = self._classify_respiratory_at(agent_x=100.0, agent_y=500.0, step=step)
        assert event is not None
        assert event.hazard_type != "toxin" or not event.true_positive

    def _classify_cardiac_at(
        self,
        agent_x: float,
        agent_y: float,
        step: int,
        seir_state: SEIRState = SEIRState.SUSCEPTIBLE,
    ) -> DetectionEvent | None:
        config = SimulationConfig(
            n_agents=1,
            grid_width=2000.0,
            grid_height=2000.0,
            cell_size=200.0,
            n_steps=1,
            seed=42,
            plume=self._plume_config(),
            seir=SEIRConfig(initial_infected=0),
        )
        model = GarlandModel(config)
        model.agent_x = np.array([agent_x], dtype=np.float32)
        model.agent_y = np.array([agent_y], dtype=np.float32)
        model.grid.assign_positions(model.agent_x, model.agent_y)
        model.seir.states[0] = seir_state
        model.current_step = step

        concentrations = compute_plume_concentration(
            model.agent_x, model.agent_y, model.plume_config, step
        )
        zone_cell = model.grid.cell_of(0)
        query = BroadcastQuery(
            zone_cells=[zone_cell],
            anomaly_type=AnomalyType.CARDIAC,
            time_window_start=0,
            time_window_end=1,
        )
        model._classify_detection(query, [self._genuine_response()], concentrations)
        if not model.metrics.detection_events:
            return None
        return model.metrics.detection_events[-1]

    def test_cardiac_downwind_during_plume_is_toxin_true_positive(self):
        """Cardiac anomalies in a plume-exposed zone should count as toxin TPs."""
        event = self._classify_cardiac_at(agent_x=560.0, agent_y=500.0, step=20)
        assert event is not None
        assert event.anomaly_type == AnomalyType.CARDIAC
        assert event.hazard_type == "toxin"
        assert event.true_positive is True

    def test_cardiac_with_zone_disease_is_disease_true_positive(self):
        """Cardiac anomalies with local disease exposure should count as disease TPs."""
        event = self._classify_cardiac_at(
            agent_x=100.0,
            agent_y=500.0,
            step=20,
            seir_state=SEIRState.INFECTIOUS,
        )
        assert event is not None
        assert event.hazard_type == "disease"
        assert event.true_positive is True

    def test_cardiac_without_hazard_is_disease_false_positive(self):
        """Cardiac anomalies without local hazards should be recorded as disease FPs."""
        event = self._classify_cardiac_at(agent_x=100.0, agent_y=500.0, step=20)
        assert event is not None
        assert event.hazard_type == "disease"
        assert event.true_positive is False

    def test_cardiac_detection_appears_in_summary(self):
        """Summary metrics should include cardiac detection counts."""
        event = self._classify_cardiac_at(agent_x=560.0, agent_y=500.0, step=20)
        assert event is not None
        config = SimulationConfig(
            n_agents=1,
            grid_width=2000.0,
            grid_height=2000.0,
            cell_size=200.0,
            n_steps=1,
            seed=42,
            plume=self._plume_config(),
            seir=SEIRConfig(initial_infected=0),
        )
        model = GarlandModel(config)
        model.metrics.record_detection(event)
        assert model.metrics.summary()["cardiac_detections"] == 1


class TestAttackSummaryMetrics:
    """Summary attack metrics should reflect enabled attack activity."""

    def test_sybil_enabled_records_false_alerts(self):
        config = SimulationConfig(
            n_agents=500,
            n_steps=30,
            seed=42,
            seir=SEIRConfig(initial_infected=0, beta=0.0),
            plume=PlumeConfig(start_step=10_000, duration_steps=1),
            privacy=PrivacyConfig(threshold_m=5, k_min=10),
            attacks=AttackConfig(
                sybil_count=20,
                sybil_target_zone=0,
                active_attacks=[AttackType.SYBIL_INJECTION],
            ),
        )
        model = GarlandModel(config)
        summary = model.run()
        assert summary.sybil_false_alerts > 0
        assert summary.summary()["total_broadcasts"] > 0

    def test_deanon_enabled_records_attempts(self):
        config = SimulationConfig(
            n_agents=500,
            n_steps=300,
            seed=42,
            seir=SEIRConfig(initial_infected=0, beta=0.0),
            plume=PlumeConfig(start_step=10_000, duration_steps=1),
            attacks=AttackConfig(
                target_agent_idx=0,
                active_attacks=[AttackType.TARGETED_QUERY],
            ),
        )
        model = GarlandModel(config)
        summary = model.run()
        assert summary.deanon_attempts > 0
        assert 0.0 <= summary.summary()["deanon_success_rate"] <= 1.0

    def test_no_attacks_leave_attack_metrics_at_zero(self, small_config):
        model = GarlandModel(small_config)
        metrics = model.run()
        summary = metrics.summary()
        assert summary["sybil_false_alerts"] == 0
        assert summary["deanon_attempts"] == 0
        assert summary["deanon_successes"] == 0

    def test_total_broadcasts_matches_aggregator(self, small_config):
        model = GarlandModel(small_config)
        model.run()
        assert model.metrics.summary()["total_broadcasts"] == model.aggregator.broadcasts_issued


class TestProtocolSimulationIntegration:
    """Full GarlandModel step loop: token → broadcast → response → detection."""

    def test_infection_cluster_triggers_broadcasts_and_responses(self):
        """SEIR-driven anomalies should propagate through the privacy protocol."""
        config = SimulationConfig(
            n_agents=600,
            wearable_fraction=1.0,
            grid_width=2000.0,
            grid_height=2000.0,
            cell_size=200.0,
            n_steps=80,
            seed=42,
            seir=SEIRConfig(initial_infected=40, beta=0.04, sigma=0.01, gamma=0.001),
            plume=PlumeConfig(start_step=10_000, duration_steps=1),
            privacy=PrivacyConfig(threshold_m=3, k_min=10, time_window_steps=12),
        )
        model = GarlandModel(config)
        model.run()
        df = model.metrics.to_dataframe()

        assert int(df["tokens_submitted"].sum()) > 0
        assert int(df["broadcasts_issued"].sum()) > 0
        assert int(df["responses_received"].sum()) > 0
        assert model.aggregator.broadcasts_issued > 0
