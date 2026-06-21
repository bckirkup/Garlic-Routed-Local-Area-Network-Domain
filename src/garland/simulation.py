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

from garland.agents import CitizenAgent, NetworkAggregator
from garland.attacks import AttackConfig, AttackOrchestrator, AttackType
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
from garland.spatial import SpatialIndex, create_spatial_grid


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
    spatial_backend : str
        Spatial index backend: ``hex`` (H3, default) or ``rect``.
    h3_resolution : int
        H3 resolution when ``spatial_backend`` is ``hex`` (9 ≈ 200 m cells).
    origin_lat : float
        Origin latitude for H3 meter ↔ lat/lng conversion.
    origin_lng : float
        Origin longitude for H3 meter ↔ lat/lng conversion.
    mobility_model : str
        Agent movement model: ``random_walk`` (default) or ``static``.
    mobility_speed_m : float
        Maximum random-walk displacement per step in meters.
    biometric_synthesis : str
        Observation backend: ``custom`` (default, fast) or ``neurokit`` (slow).
    neurokit_window_seconds : float
        ECG/RSP simulation window when using NeuroKit2 synthesis.
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
    spatial_backend: str = "hex"
    h3_resolution: int = 9
    origin_lat: float = 40.0
    origin_lng: float = -74.0
    mobility_model: str = "random_walk"
    mobility_speed_m: float = 50.0
    biometric_synthesis: str = "custom"
    neurokit_window_seconds: float = 60.0
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

        # Initialize spatial grid (H3 hex by default)
        self.grid: SpatialIndex = create_spatial_grid(
            width=self.config.grid_width,
            height=self.config.grid_height,
            cell_size=self.config.cell_size,
            backend=self.config.spatial_backend,  # type: ignore[arg-type]
            h3_resolution=self.config.h3_resolution,
            origin_lat=self.config.origin_lat,
            origin_lng=self.config.origin_lng,
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

        # Attack orchestrator
        self.attack_orchestrator = AttackOrchestrator(config=self.config.attacks)
        self._resolve_attack_defaults()

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

        # Position = neighborhood center + Gaussian offset (vectorized)
        offsets_x = self.rng.normal(0, 300, n)
        offsets_y = self.rng.normal(0, 300, n)
        self.agent_x = np.clip(
            neighborhood_centers_x[self.neighborhood_ids] + offsets_x,
            0,
            self.config.grid_width,
        ).astype(np.float32)
        self.agent_y = np.clip(
            neighborhood_centers_y[self.neighborhood_ids] + offsets_y,
            0,
            self.config.grid_height,
        ).astype(np.float32)

        self.grid.assign_positions(self.agent_x, self.agent_y)
        self.agent_cell_ids = self.grid.cell_ids.copy()

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

        self.has_wearable = np.isin(self.household_ids, list(wearable_households))

        # Map: wearable global index → local profile index
        self.wearable_indices = np.where(self.has_wearable)[0]
        self.wearable_local_map = {
            int(gidx): lidx for lidx, gidx in enumerate(self.wearable_indices)
        }

    def _init_citizen_agents(self) -> None:
        """Create CitizenAgent objects for wearable-equipped agents."""
        self.wearable_agents_by_cell: dict[int, list[CitizenAgent]] = {}
        for lidx, gidx in enumerate(self.wearable_indices):
            gidx_int = int(gidx)
            cell_id = int(self.agent_cell_ids[gidx_int])
            agent = CitizenAgent(
                idx=gidx_int,
                has_wearable=True,
                profile=self.profiles[lidx],
                household_id=int(self.household_ids[gidx_int]),
                neighborhood_id=int(self.neighborhood_ids[gidx_int]),
                baseline=self.baselines[lidx],
                cell_id=cell_id,
            )
            self.citizen_agents.append(agent)
            self.wearable_agents_by_cell.setdefault(cell_id, []).append(agent)

    def _resolve_attack_defaults(self) -> None:
        """Fill attack zone defaults from the target agent when unset."""
        attacks = self.config.attacks
        if not attacks.active_attacks:
            return

        target_idx = min(max(attacks.target_agent_idx, 0), self.config.n_agents - 1)
        target_cell = self.grid.cell_of(target_idx)

        if (
            AttackType.SYBIL_INJECTION in attacks.active_attacks
            and attacks.sybil_target_zone == 0
        ):
            attacks.sybil_target_zone = target_cell

        if AttackType.ECLIPSE in attacks.active_attacks and not attacks.eclipse_target_zones:
            attacks.eclipse_target_zones = [target_cell]

        self.attack_orchestrator.config = attacks
        self.attack_orchestrator._sync_sub_configs()

    def _update_mobility(self) -> None:
        """Advance agent positions and refresh spatial / wearable cell membership."""
        if self.config.mobility_model == "static":
            return

        n = self.config.n_agents
        angles = self.rng.uniform(0, 2 * np.pi, n)
        distance = self.rng.uniform(0, self.config.mobility_speed_m, n)
        self.agent_x = np.clip(
            self.agent_x + distance * np.cos(angles),
            0,
            self.config.grid_width,
        ).astype(np.float32)
        self.agent_y = np.clip(
            self.agent_y + distance * np.sin(angles),
            0,
            self.config.grid_height,
        ).astype(np.float32)
        self.grid.assign_positions(self.agent_x, self.agent_y)
        self._reconcile_wearable_cells()

    def _reconcile_wearable_cells(self) -> None:
        """Update cached cell IDs and zone index after agent movement."""
        new_cell_ids = self.grid.cell_ids
        for agent in self.citizen_agents:
            new_cell = int(new_cell_ids[agent.idx])
            old_cell = agent.cell_id
            if new_cell == old_cell:
                continue
            bucket = self.wearable_agents_by_cell.get(old_cell)
            if bucket is not None:
                bucket.remove(agent)
                if not bucket:
                    del self.wearable_agents_by_cell[old_cell]
            agent.cell_id = new_cell
            self.wearable_agents_by_cell.setdefault(new_cell, []).append(agent)
        self.agent_cell_ids = new_cell_ids.copy()

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

        # --- 0. Agent Mobility ---
        self._update_mobility()

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
            cell_id = agent.cell_id

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
                synthesis_backend=self.config.biometric_synthesis,  # type: ignore[arg-type]
                neurokit_window_seconds=self.config.neurokit_window_seconds,
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

        # --- 5. Attack Layer ---
        sybil_injected = 0
        replay_injected = 0
        eclipse_dropped = 0
        if self.config.attacks.active_attacks:
            tokens, eclipse_dropped = self.attack_orchestrator.filter_tokens(
                tokens, self.rng
            )
            fake_tokens, sybil_injected, replay_injected = (
                self.attack_orchestrator.step_injections(
                    self.current_step, time_bin, self.rng
                )
            )
            tokens.extend(fake_tokens)

        # --- 6. Aggregator Threshold Check ---
        self.aggregator.ingest_tokens(tokens, time_bin)
        queries = self.aggregator.evaluate_and_broadcast(
            time_bin, self.grid.dilated_zone
        )

        if self.config.attacks.active_attacks:
            self.attack_orchestrator.cache_tokens_for_replay(tokens)

        sybil_zone = self.config.attacks.sybil_target_zone
        for query in queries:
            if (
                sybil_injected > 0
                and AttackType.SYBIL_INJECTION in self.config.attacks.active_attacks
                and sybil_zone in query.zone_cells
            ):
                self.metrics.record_sybil_false_alert()
                self.attack_orchestrator.false_positives_triggered += 1
            if (
                replay_injected > 0
                and AttackType.REPLAY in self.config.attacks.active_attacks
            ):
                self.attack_orchestrator.record_replay_false_alerts(query.zone_cells)

        # --- 7. Agents Respond to Queries ---
        responses_received = 0
        time_window_steps = self.config.privacy.time_window_steps
        for query in queries:
            responses = []
            for cell_id in query.zone_cells:
                for agent in self.wearable_agents_by_cell.get(cell_id, ()):
                    resp = agent.respond_to_query(
                        query,
                        float(self.agent_x[agent.idx]),
                        float(self.agent_y[agent.idx]),
                        agent.cell_id,
                        self.config.privacy,
                        self.rng,
                    )
                    if resp is not None:
                        responses.append(resp)

            self.aggregator.collect_responses(responses)
            responses_received += len(responses)

            self.attack_orchestrator.observe_protocol_responses(
                time_bin, responses, time_window_steps
            )

            # Classify detection event
            self._classify_detection(query, responses, concentrations)

        self._run_deanon_attack(time_bin)
        self.attack_orchestrator.evaluate_periodic(
            self.current_step, self.agent_x, self.agent_y
        )
        self.metrics.sync_attack_metrics(self.attack_orchestrator)

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
            replay_tokens_injected=replay_injected,
            eclipse_tokens_dropped=eclipse_dropped,
        )

        self.current_step += 1

    def _classify_detection(self, query, responses, concentrations: np.ndarray) -> None:
        """Classify a broadcast query as TP or FP for each hazard type."""
        genuine_responses = [r for r in responses if r.anomaly_confirmed and not r.is_dummy]

        if not genuine_responses:
            return

        # Determine if this corresponds to a real hazard
        if query.anomaly_type == AnomalyType.RESPIRATORY:
            # Could be toxin or disease — use zone-local plume exposure, not global timing
            is_toxin_tp = self._zone_has_plume_exposure(query.zone_cells, concentrations)
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
            # Likely disease — use zone-local exposure, not global infectious count
            is_disease_tp = self._zone_has_active_disease(query.zone_cells)
            event = DetectionEvent(
                step=self.current_step,
                hazard_type="disease",
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=is_disease_tp,
                agents_affected=len(genuine_responses),
            )
            self.metrics.record_detection(event)
        elif query.anomaly_type == AnomalyType.CARDIAC:
            # Isolated HR/HRV — attribute to toxin if zone is plume-exposed, else disease
            is_toxin_tp = self._zone_has_plume_exposure(query.zone_cells, concentrations)
            if is_toxin_tp:
                hazard_type = "toxin"
                true_positive = True
            else:
                hazard_type = "disease"
                true_positive = self._zone_has_active_disease(query.zone_cells)
            event = DetectionEvent(
                step=self.current_step,
                hazard_type=hazard_type,
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=true_positive,
                agents_affected=len(genuine_responses),
            )
            self.metrics.record_detection(event)

    def _run_deanon_attack(self, time_bin: int) -> None:
        """Execute a periodic targeted-query deanonymization attempt."""
        if AttackType.TARGETED_QUERY not in self.config.attacks.active_attacks:
            return
        if self.current_step % self.config.attacks.deanon_interval_steps != 0:
            return

        target_idx = self.config.attacks.target_agent_idx
        if target_idx < 0 or target_idx >= self.config.n_agents:
            return

        target_cell = self.grid.cell_of(target_idx)
        query = self.attack_orchestrator.deanon.craft_targeted_query(
            target_cell=target_cell,
            time_start=time_bin - self.config.privacy.time_window_steps,
            time_end=time_bin,
            query_id=self.aggregator.broadcasts_issued,
        )

        for cell_id in query.zone_cells:
            for agent in self.wearable_agents_by_cell.get(cell_id, ()):
                resp = agent.respond_to_query(
                    query,
                    float(self.agent_x[agent.idx]),
                    float(self.agent_y[agent.idx]),
                    agent.cell_id,
                    self.config.privacy,
                    self.rng,
                )
                if resp is not None:
                    self.attack_orchestrator.deanon.collect_response(resp)

        self.attack_orchestrator.evaluate_deanonymization(
            float(self.agent_x[target_idx]),
            float(self.agent_y[target_idx]),
            success_threshold=self.config.attacks.deanon_success_threshold_m,
        )

    def _zone_has_plume_exposure(
        self, zone_cells: list[int], concentrations: np.ndarray, threshold: float = 0.01
    ) -> bool:
        """Return True if any agent in the query zone exceeds the plume threshold."""
        for cell_id in zone_cells:
            for agent_idx in self.grid.agents_in_cell(cell_id):
                if concentrations[agent_idx] > threshold:
                    return True
        return False

    def _zone_has_active_disease(self, zone_cells: list[int]) -> bool:
        """Return True if any agent in the query zone is exposed or infectious."""
        for cell_id in zone_cells:
            for agent_idx in self.grid.agents_in_cell(cell_id):
                if self.seir.states[agent_idx] in (
                    SEIRState.EXPOSED,
                    SEIRState.INFECTIOUS,
                ):
                    return True
        return False

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
