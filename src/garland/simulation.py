"""Mesa-based simulation engine for the GARLAND epidemiological security testbed.

Orchestrates 250,000 agents at 5-minute resolution with:
- Vectorized biometric generation (only wearable-equipped agents)
- SEIR disease + plume hazard co-occurrence
- Privacy protocol execution (blind gating → aggregation → broadcast → response)
- Attack simulation layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import mesa
import numpy as np

from garland.agents import CitizenAgent, MaliciousAgent, NetworkAggregator
from garland.attacks import AttackConfig, AttackOrchestrator
from garland.biometrics import BaselineTracker, generate_profiles
from garland.hazards import (
    PlumeConfig,
    SEIRConfig,
    SEIREngine,
    SEIRState,
    compute_plume_concentration,
    plume_biometric_perturbation,
)
from garland.metrics import DetectionEvent, MetricsCollector
from garland.privacy import AnomalyType, EncryptedToken, PrivacyConfig
from garland.spatial import SpatialGrid


@dataclass
class SimulationConfig:
    """Top-level configuration for the GARLAND simulation.

    Parameters
    ----------
    n_agents : int
        Total population size.
    wearable_fraction : float
        Fraction of agents with wearable devices (patchy by household).
    grid_width : float
        Spatial domain width in meters.
    grid_height : float
        Spatial domain height in meters.
    cell_size : float
        Spatial grid cell size in meters.
    n_steps : int
        Total simulation steps (each = 5 minutes).
    households_per_neighborhood : int
        Number of households per neighborhood zone.
    household_size_mean : int
        Mean household size.
    start_datetime : datetime
        Simulation start time (for circadian/seasonal effects).
    seed : int
        Random seed for reproducibility.
    baseline_decay_lambda : float
        Forgetting rate for biometric baselines.
    baseline_seasonal_decay : float
        Seasonal learning rate for baselines.
    """

    n_agents: int = 250_000
    wearable_fraction: float = 0.15
    grid_width: float = 10_000.0
    grid_height: float = 10_000.0
    cell_size: float = 200.0
    n_steps: int = 2016  # 7 days at 5-min resolution
    households_per_neighborhood: int = 200
    household_size_mean: int = 3
    start_datetime: datetime = field(default_factory=lambda: datetime(2024, 1, 15, 0, 0))
    seed: int = 42
    baseline_decay_lambda: float = 0.01
    baseline_seasonal_decay: float = 0.001
    # Sub-configs
    seir: SEIRConfig = field(default_factory=SEIRConfig)
    plume: PlumeConfig = field(default_factory=PlumeConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)


class GarlandModel(mesa.Model):
    """Mesa ABM model for the GARLAND epidemiological security testbed.

    Architecture:
    - Agent state stored in flat numpy arrays for vectorized computation
    - Only wearable-equipped agents (wearable_fraction) run biometric logic
    - Spatial grid enables efficient proximity queries
    - Privacy protocol runs each step after biometric evaluation
    """

    def __init__(self, config: SimulationConfig | None = None):
        super().__init__()
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.current_step = 0

        # Initialize spatial grid
        self.grid = SpatialGrid(
            self.config.grid_width, self.config.grid_height, self.config.cell_size
        )

        # Generate agent positions (clustered by neighborhood)
        self._init_positions()

        # Assign wearables (patchy by household)
        self._init_wearables()

        # Initialize biometric profiles for wearable agents
        n_wearable = int(np.sum(self.has_wearable))
        self.profiles = generate_profiles(n_wearable, self.rng)

        # Initialize baseline trackers for wearable agents
        self.baselines: list[BaselineTracker] = [
            BaselineTracker(
                decay_lambda=self.config.baseline_decay_lambda,
                seasonal_decay=self.config.baseline_seasonal_decay,
            )
            for _ in range(n_wearable)
        ]

        # SEIR engine
        self.seir = SEIREngine(config=self.config.seir)
        self.seir.initialize(self.config.n_agents, self.rng)

        # Plume config
        self.plume_config = self.config.plume

        # Privacy protocol components
        self.aggregator = NetworkAggregator(config=self.config.privacy)

        # Agent objects (lightweight — heavy state in arrays)
        self.citizen_agents: list[CitizenAgent] = []
        self._init_citizen_agents()

        # Malicious agents
        self.malicious_agents: list[MaliciousAgent] = []
        if self.config.attacks.active_attacks:
            self._init_malicious_agents()

        # Attack orchestrator
        self.attack_orchestrator = AttackOrchestrator(config=self.config.attacks)

        # Metrics
        self.metrics = MetricsCollector()

    def _init_positions(self) -> None:
        """Generate clustered agent positions by neighborhood."""
        n = self.config.n_agents
        # Create neighborhood centers
        n_neighborhoods = max(
            1, n // (self.config.households_per_neighborhood * self.config.household_size_mean)
        )
        neighborhood_centers_x = self.rng.uniform(
            500, self.config.grid_width - 500, n_neighborhoods
        )
        neighborhood_centers_y = self.rng.uniform(
            500, self.config.grid_height - 500, n_neighborhoods
        )

        # Assign agents to neighborhoods, then cluster within
        self.neighborhood_ids = self.rng.integers(0, n_neighborhoods, n)

        # Households are nested within neighborhoods (not index-ordered globally)
        self.household_ids = np.empty(n, dtype=np.int64)
        next_household_id = 0
        chunk = self.config.household_size_mean
        for nb in range(n_neighborhoods):
            members = np.where(self.neighborhood_ids == nb)[0]
            self.rng.shuffle(members)
            for start in range(0, len(members), chunk):
                self.household_ids[members[start : start + chunk]] = next_household_id
                next_household_id += 1

        # Position = neighborhood center + Gaussian offset
        self.agent_x = np.empty(n, dtype=np.float32)
        self.agent_y = np.empty(n, dtype=np.float32)

        for i in range(n):
            nb = self.neighborhood_ids[i]
            self.agent_x[i] = np.clip(
                neighborhood_centers_x[nb] + self.rng.normal(0, 300),
                0,
                self.config.grid_width,
            )
            self.agent_y[i] = np.clip(
                neighborhood_centers_y[nb] + self.rng.normal(0, 300),
                0,
                self.config.grid_height,
            )

        self.grid.assign_positions(self.agent_x, self.agent_y)

    def _init_wearables(self) -> None:
        """Assign wearables with household-patchy penetration."""
        n = self.config.n_agents
        self.has_wearable = np.zeros(n, dtype=bool)
        target_count = int(n * self.config.wearable_fraction)

        unique_households = np.unique(self.household_ids)
        household_sizes = {
            int(hh): int(np.sum(self.household_ids == hh)) for hh in unique_households
        }
        shuffled = self.rng.permutation(unique_households)

        wearable_households: set[int] = set()
        cumulative = 0
        for hh in shuffled:
            size = household_sizes[int(hh)]
            if cumulative >= target_count:
                break
            if cumulative + size > target_count and cumulative > 0:
                break
            wearable_households.add(int(hh))
            cumulative += size

        if cumulative < target_count:
            for hh in shuffled:
                hh_int = int(hh)
                if hh_int in wearable_households:
                    continue
                size = household_sizes[hh_int]
                if cumulative + size > target_count * 1.05:
                    continue
                wearable_households.add(hh_int)
                cumulative += size
                if cumulative >= target_count:
                    break

        for i in range(n):
            if int(self.household_ids[i]) in wearable_households:
                self.has_wearable[i] = True

        # Map: wearable global index → local profile index
        self.wearable_indices = np.where(self.has_wearable)[0]
        self.wearable_local_map = {
            int(gidx): lidx for lidx, gidx in enumerate(self.wearable_indices)
        }

    def _init_citizen_agents(self) -> None:
        """Create CitizenAgent objects for wearable-equipped agents."""
        for lidx, gidx in enumerate(self.wearable_indices):
            agent = CitizenAgent(
                idx=int(gidx),
                has_wearable=True,
                profile=self.profiles[lidx],
                household_id=int(self.household_ids[gidx]),
                neighborhood_id=int(self.neighborhood_ids[gidx]),
                baseline=self.baselines[lidx],
            )
            self.citizen_agents.append(agent)

    def _init_malicious_agents(self) -> None:
        """Create malicious agents for attack simulation."""
        mal = MaliciousAgent(
            idx=-1,
            target_zone=self.config.attacks.sybil_target_zone,
            target_agent=self.config.attacks.target_agent_idx,
            sybil_identities=self.config.attacks.sybil_count,
        )
        self.malicious_agents.append(mal)

    def _current_time_info(self) -> tuple[float, int, int, int]:
        """Compute current time parameters from step count.

        Returns (hour_of_day, hour_int, month, day_of_year).
        """
        minutes_elapsed = self.current_step * 5
        total_minutes = (
            self.config.start_datetime.hour * 60
            + self.config.start_datetime.minute
            + minutes_elapsed
        )
        hour_of_day = (total_minutes % 1440) / 60.0
        hour_int = int(hour_of_day) % 24
        # Approximate day and month
        day_offset = minutes_elapsed // 1440
        start_day = self.config.start_datetime.timetuple().tm_yday
        day_of_year = ((start_day + day_offset - 1) % 365) + 1
        month = self.config.start_datetime.month  # Simplified
        return hour_of_day, hour_int, month, day_of_year

    def step(self) -> None:
        """Execute one 5-minute simulation step.

        Pipeline:
        1. Advance SEIR disease model
        2. Compute plume concentrations
        3. Generate biometric observations (wearable agents only)
        4. Run anomaly detection → emit encrypted tokens
        5. Aggregator threshold check → broadcast queries
        6. Agents respond with DP perturbation
        7. Attack layer execution
        8. Collect metrics
        """
        hour_of_day, hour_int, month, day_of_year = self._current_time_info()
        time_bin = self.current_step // self.config.privacy.time_window_steps

        # --- 1. SEIR Step ---
        self.seir.step(self.current_step, self.agent_x, self.agent_y, self.rng)

        # Track disease onset
        if self.metrics.disease_onset_step is None:
            infectious_count = np.sum(self.seir.states == SEIRState.INFECTIOUS)
            if infectious_count > self.config.seir.initial_infected:
                self.metrics.record_disease_onset(self.current_step)

        # --- 2. Plume Concentrations ---
        concentrations = compute_plume_concentration(
            self.agent_x, self.agent_y, self.plume_config, self.current_step
        )
        plume_exposed_count = int(np.sum(concentrations > 0.01))

        # Track toxin onset
        if plume_exposed_count > 0:
            self.metrics.record_toxin_onset(self.current_step)

        # --- 3 & 4. Biometric Observation + Anomaly Detection ---
        tokens: list[EncryptedToken] = []
        anomalies_detected = 0

        # Activity level: simple day/night model
        if 6 <= hour_of_day <= 22:
            activity = 0.3 * max(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        else:
            activity = 0.0

        for agent in self.citizen_agents:
            gidx = agent.idx
            cell_id = self.grid.cell_of(gidx)

            # Compute hazard perturbation
            perturbation = np.zeros(4, dtype=np.float64)

            # SEIR perturbation
            if self.seir.states[gidx] in (SEIRState.EXPOSED, SEIRState.INFECTIOUS):
                ref_step = (
                    self.seir.infection_step[gidx]
                    if self.seir.states[gidx] == SEIRState.INFECTIOUS
                    else self.seir.exposure_step[gidx]
                )
                if ref_step >= 0:
                    steps_since = self.current_step - ref_step
                    perturbation += self.seir.biometric_perturbation(gidx, steps_since)

            # Plume perturbation
            conc = concentrations[gidx]
            if conc > 0.01:
                perturbation += plume_biometric_perturbation(conc)

            # Observe and detect
            token = agent.observe_and_detect(
                hour=hour_int,
                month=month,
                day_of_year=day_of_year,
                hour_of_day=hour_of_day,
                rng=self.rng,
                cell_id=cell_id,
                hazard_perturbation=perturbation if np.any(perturbation != 0) else None,
                activity_level=activity + self.rng.normal(0, 0.05),
            )

            if token is not None:
                # Stamp time bin
                token = EncryptedToken(
                    zone_id=token.zone_id,
                    anomaly_type=token.anomaly_type,
                    timestamp_bin=time_bin,
                    agent_id_hash=token.agent_id_hash,
                    is_dummy=token.is_dummy,
                )
                tokens.append(token)
                anomalies_detected += 1

            # Dummy traffic
            dummy = agent.generate_dummy_traffic(
                float(self.agent_x[gidx]),
                float(self.agent_y[gidx]),
                cell_id,
                self.config.privacy,
                self.rng,
            )
            if dummy is not None:
                tokens.append(
                    EncryptedToken(
                        zone_id=dummy.zone_id,
                        anomaly_type=dummy.anomaly_type,
                        timestamp_bin=time_bin,
                        agent_id_hash=dummy.agent_id_hash,
                        is_dummy=True,
                    )
                )

        # --- 5. Attack Layer: Inject Sybil tokens ---
        sybil_injected = 0
        if self.config.attacks.active_attacks:
            fake_tokens = self.attack_orchestrator.step(
                self.current_step, time_bin, self.rng
            )
            tokens.extend(fake_tokens)
            sybil_injected = len(fake_tokens)

        # --- 6. Aggregator Threshold Check ---
        self.aggregator.ingest_tokens(tokens, time_bin)
        queries = self.aggregator.evaluate_and_broadcast(
            time_bin, self.grid.dilated_zone
        )

        # --- 7. Agents Respond to Queries ---
        responses_received = 0
        for query in queries:
            responses = []
            for agent in self.citizen_agents:
                agent_cell_id = self.grid.cell_of(agent.idx)
                if agent_cell_id in query.zone_cells:
                    resp = agent.respond_to_query(
                        query,
                        float(self.agent_x[agent.idx]),
                        float(self.agent_y[agent.idx]),
                        agent_cell_id,
                        self.config.privacy,
                        self.rng,
                    )
                    if resp is not None:
                        responses.append(resp)

            self.aggregator.collect_responses(responses)
            responses_received += len(responses)

            # Classify detection event
            self._classify_detection(query, responses)

        # --- 8. Update hazard episode metrics ---
        has_active_disease = np.sum(
            self.seir.states == SEIRState.INFECTIOUS
        ) > self.config.seir.initial_infected
        has_active_plume = plume_exposed_count > 0

        step_events = [
            e for e in self.metrics.detection_events if e.step == self.current_step
        ]
        disease_tp_this_step = any(
            e.hazard_type == "disease" and e.true_positive for e in step_events
        )
        disease_fp_this_step = any(
            e.hazard_type == "disease" and not e.true_positive for e in step_events
        )
        toxin_tp_this_step = any(
            e.hazard_type == "toxin" and e.true_positive for e in step_events
        )
        toxin_fp_this_step = any(
            e.hazard_type == "toxin" and not e.true_positive for e in step_events
        )

        self.metrics.update_hazard_episode(
            "disease", has_active_disease, disease_tp_this_step, disease_fp_this_step
        )
        self.metrics.update_hazard_episode(
            "toxin", has_active_plume, toxin_tp_this_step, toxin_fp_this_step
        )

        # --- 9. Record Metrics ---
        seir_counts = {
            "S": int(np.sum(self.seir.states == SEIRState.SUSCEPTIBLE)),
            "E": int(np.sum(self.seir.states == SEIRState.EXPOSED)),
            "I": int(np.sum(self.seir.states == SEIRState.INFECTIOUS)),
            "R": int(np.sum(self.seir.states == SEIRState.RECOVERED)),
        }
        self.metrics.record_step(
            step=self.current_step,
            seir_counts=seir_counts,
            plume_exposed=plume_exposed_count,
            anomalies_detected=anomalies_detected,
            tokens_submitted=len(tokens),
            broadcasts_issued=len(queries),
            responses_received=responses_received,
            cumulative_epsilon=self.aggregator.state.total_epsilon,
            sybil_tokens_injected=sybil_injected,
        )

        self.current_step += 1

    def _classify_detection(self, query, responses) -> None:
        """Classify a broadcast query as TP or FP for each hazard type."""
        genuine_responses = [r for r in responses if r.anomaly_confirmed and not r.is_dummy]

        if not genuine_responses:
            return

        # Determine if this corresponds to a real hazard
        if query.anomaly_type == AnomalyType.RESPIRATORY:
            # Could be toxin or disease
            # Check if plume is active in query zone
            is_toxin_tp = self.current_step >= self.plume_config.start_step and (
                self.current_step < self.plume_config.start_step + self.plume_config.duration_steps
            )
            event = DetectionEvent(
                step=self.current_step,
                hazard_type="toxin" if is_toxin_tp else "disease",
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=is_toxin_tp,
                agents_affected=len(genuine_responses),
            )
            self.metrics.record_detection(event)
        elif query.anomaly_type in (AnomalyType.FEBRILE, AnomalyType.MULTI_SYSTEM):
            # Likely disease
            is_disease_tp = np.sum(self.seir.states == SEIRState.INFECTIOUS) > (
                self.config.seir.initial_infected
            )
            event = DetectionEvent(
                step=self.current_step,
                hazard_type="disease",
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=is_disease_tp,
                agents_affected=len(genuine_responses),
            )
            self.metrics.record_detection(event)

    def run(self, steps: int | None = None) -> MetricsCollector:
        """Run the full simulation.

        Parameters
        ----------
        steps : int | None
            Override number of steps (default: config.n_steps).
        """
        n_steps = steps or self.config.n_steps
        for _ in range(n_steps):
            self.step()
        self.metrics.finalize_hazard_episodes()
        return self.metrics
