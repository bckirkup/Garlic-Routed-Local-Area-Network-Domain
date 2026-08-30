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
from functools import partial
from typing import Any

import mesa
import numpy as np
from numpy.typing import NDArray

from garland.adoption import AdoptionConfig
from garland.advisories import AdvisoryConfig, AdvisoryEngine
from garland.agents import CitizenAgent, NetworkAggregator
from garland.alarm_calibration import AlarmCalibrationConfig, AlarmRateCalibrator, calibrator_for
from garland.attacks import AttackConfig, AttackOrchestrator, AttackType
from garland.baseline_maturation import BaselineMaturationConfig
from garland.biometrics import (
    COVARIANCE_WARMUP_SAMPLES,
    BaselineTracker,
    generate_observation,
    generate_profiles,
)
from garland.channels import DEFAULT_CHANNEL_SET, ChannelSet
from garland.confounders import (
    BenignInstance,
    ConfounderEngine,
    ConfoundersConfig,
    ConfounderStep,
)
from garland.constants import STEPS_PER_DAY
from garland.demographics import (
    ADULT,
    AGE_BANDS,
    DemographicsConfig,
    assign_age_bands,
    band_counts,
)
from garland.detection import SequentialDetector
from garland.detection_power import AblationProbe, DetectionPowerConfig
from garland.device_lifecycle import (
    DeviceLifecycleConfig,
    DeviceLifecycleEngine,
    DeviceStatus,
    SubsystemLifecycle,
)
from garland.devices import DeviceFleet, DeviceFleetConfig
from garland.disambiguation import (
    DisambiguationConfig,
    DisambiguationHypothesis,
    DisambiguationScore,
    DisambiguationTriggerConfig,
)
from garland.hazards import (
    LEGACY_TOXIN_EXPOSURE_CONCENTRATION_GATE,
    TOXIN_PHYSIOLOGY_NEGLIGIBILITY_DELTA_BPM,
    PlumeConfig,
    SEIRConfig,
    SEIREngine,
    SEIRState,
    compute_plume_concentrations,
    concentration_for_respiratory_delta,
    plume_biometric_perturbation,
)
from garland.metrics import DetectionEvent, MetricsCollector
from garland.perturbations import (
    BENIGN_CAUSES,
    PerturbationCause,
    PerturbationContribution,
)
from garland.privacy import (
    AnomalyType,
    BroadcastQuery,
    DisambiguationQuery,
    EncryptedToken,
    PrivacyConfig,
)
from garland.spatial import SpatialIndex, create_spatial_grid
from garland.venues import VenueEngine, VenueSystemConfig
from garland.zone_threshold import (
    ZoneThresholdCalibrationConfig,
    ZoneThresholdCalibrator,
    zone_threshold_calibrator_for,
)


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
        Agent movement model: ``random_walk`` (default), ``static``, or
        ``schedule`` (structured venues with calibrated activity patterns).
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
    baseline_mean_prior_strength : float
        Pseudo-observation strength of the configured baseline mean prior.
    baseline_mean_prior_source : str
        Source of the baseline mean prior: ``population`` uses channel resting
        means, while ``zero`` preserves the historical zero-mean baseline.
    baseline_maturation : BaselineMaturationConfig
        Device-local prior history learned before the scenario starts.
    anomaly_threshold : float
        Mahalanobis distance above which a wearable emits an anomaly token.
    minimum_respiratory_delta_bpm : float
        Minimum dose-derived respiratory-rate shift used for model-side toxin
        exposure truth. This does not affect physiology or protocol decisions.
    toxin_exposure_gate_mode : str
        ``dose_derived`` uses ``minimum_respiratory_delta_bpm``. The explicit
        ``legacy_0_01`` mode reproduces pre-calibration results.
    detector_mode : str
        ``instant`` preserves the per-step gate; ``sequential`` uses CUSUM.
    sequential_reference_value : float
        CUSUM reference value.
    sequential_threshold : float
        CUSUM alarm threshold.
    sequential_clear_steps : int
        Consecutive zero-statistic steps required to clear an alarm.
    sequential_clear_fraction : float
        Fraction of the alarm threshold below which clearing can begin.
    sequential_residual_ewma_alpha : float
        EWMA weight for sustained residual classification.
    baseline_warmup_steps : int
        Steps at start during which baselines adapt but anomaly tokens are not
        emitted to the privacy protocol.
    warmup_on_device_adopt : bool
        When True, restore the legacy behavior of applying a fresh local
        warm-up window after device removal or power-off. The default preserves
        retained baselines without re-arming warm-up.
    demographics : DemographicsConfig
        Age structure of the population and its coupling to device ownership.
        The default is demographically flat: one adult band, uniform ownership.
    adoption : AdoptionConfig
        First-time wearable adoption schedule. The default adopts every
        wearable at step zero, preserving historical behavior.
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
    baseline_maturation: BaselineMaturationConfig = field(default_factory=BaselineMaturationConfig)
    anomaly_threshold: float = 3.5
    minimum_respiratory_delta_bpm: float = 2.0
    toxin_exposure_gate_mode: str = "dose_derived"
    detector_mode: str = "instant"
    sequential_reference_value: float = 2.0
    sequential_threshold: float = 10.0
    sequential_clear_steps: int = 3
    sequential_clear_fraction: float = 0.5
    sequential_residual_ewma_alpha: float = 0.2
    baseline_mean_prior_strength: float = 12.0
    baseline_mean_prior_source: str = "population"
    baseline_warmup_steps: int = 0
    world_settling_steps: int = field(default_factory=lambda: STEPS_PER_DAY)
    warmup_on_device_adopt: bool = False
    adoption: AdoptionConfig = field(default_factory=AdoptionConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    disambiguation: DisambiguationConfig = field(default_factory=DisambiguationConfig)
    advisories: AdvisoryConfig = field(default_factory=AdvisoryConfig)
    confounders: ConfoundersConfig = field(default_factory=ConfoundersConfig)
    # Sub-configs
    seir: SEIRConfig = field(default_factory=SEIRConfig)
    plumes: list[PlumeConfig] = field(default_factory=lambda: [PlumeConfig()])
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    device_lifecycle: DeviceLifecycleConfig = field(default_factory=DeviceLifecycleConfig)
    devices: DeviceFleetConfig = field(default_factory=DeviceFleetConfig)
    detection_power: DetectionPowerConfig = field(default_factory=DetectionPowerConfig)
    alarm_calibration: AlarmCalibrationConfig = field(default_factory=AlarmCalibrationConfig)
    zone_threshold_calibration: ZoneThresholdCalibrationConfig = field(
        default_factory=ZoneThresholdCalibrationConfig
    )
    venues: VenueSystemConfig = field(default_factory=VenueSystemConfig)

    def __post_init__(self) -> None:
        """Validate the evaluation-only toxin exposure calibration."""
        concentration_for_respiratory_delta(self.minimum_respiratory_delta_bpm)
        if self.toxin_exposure_gate_mode not in {"dose_derived", "legacy_0_01"}:
            raise ValueError("toxin_exposure_gate_mode must be 'dose_derived' or 'legacy_0_01'")
        if self.adoption.new_device_warmup_steps < 0:
            raise ValueError("adoption.new_device_warmup_steps must be non-negative")
        if self.baseline_mean_prior_source not in {"population", "zero"}:
            raise ValueError("baseline_mean_prior_source must be 'population' or 'zero'")
        if (
            not np.isfinite(self.baseline_mean_prior_strength)
            or self.baseline_mean_prior_strength < 0.0
        ):
            raise ValueError("baseline_mean_prior_strength must be finite and non-negative")

    def toxin_exposure_concentration_gate(self) -> float:
        """Return the model-side toxin truth gate in concentration units."""
        if self.toxin_exposure_gate_mode == "legacy_0_01":
            return LEGACY_TOXIN_EXPOSURE_CONCENTRATION_GATE
        return concentration_for_respiratory_delta(self.minimum_respiratory_delta_bpm)

    def toxin_physiology_concentration_floor(self) -> float:
        """Return the concentration below which toxin physiology is negligible."""
        if self.toxin_exposure_gate_mode == "legacy_0_01":
            return LEGACY_TOXIN_EXPOSURE_CONCENTRATION_GATE
        return concentration_for_respiratory_delta(TOXIN_PHYSIOLOGY_NEGLIGIBILITY_DELTA_BPM)

    @property
    def plume(self) -> PlumeConfig:
        """First plume source (backward compatibility)."""
        return self.plumes[0]


@dataclass(frozen=True)
class _TokenProvenance:
    """Model-side truth bookkeeping kept outside opaque protocol tokens."""

    zone_id: int
    anomaly_type: AnomalyType
    timestamp_bin: int
    toxin_affected: bool
    disease_affected: bool
    causes: frozenset[PerturbationCause]


@dataclass(frozen=True)
class _DisambiguationQueryOutcome:
    """Protocol counters and model-side score for one issued ask."""

    reached: int
    acks: int
    yes: int
    no: int
    pending: int
    ack_release: int
    epsilon_delta: float
    score: DisambiguationScore


@dataclass
class _DisambiguationResult:
    """Per-step disambiguation counters and reporting-only score buckets."""

    queries: int = 0
    suppressed_by_budget: int = 0
    acks: int = 0
    ack_releases: int = 0
    reached: int = 0
    yes: int = 0
    no: int = 0
    unanswered: int = 0
    unresolved: int = 0
    well_founded: int = 0
    unfounded: int = 0
    unscored: int = 0
    unfounded_epsilon: float = 0.0
    unscored_epsilon: float = 0.0
    max_ask_epsilon_delta: float = 0.0
    well_founded_by_hypothesis: dict[str, int] = field(default_factory=dict)
    unfounded_by_hypothesis: dict[str, int] = field(default_factory=dict)
    unscored_by_hypothesis: dict[str, int] = field(default_factory=dict)


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
        if (
            self.config.adoption.mode != "all_at_start"
            and self.config.adoption.initial_adopted_fraction >= 1.0
        ):
            raise ValueError(
                "Non-default adoption modes require "
                "initial_adopted_fraction < 1.0 so pending adopters exist"
            )
        self.rng = np.random.default_rng(self.config.seed)
        self.baseline_maturation_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0xBA5E])
        )
        self.disambiguation_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0xD15A])
        )
        self.advisory_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0xAD71])
        )
        self.confounder_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0xC0F0])
        )
        self.current_step = 0
        # Observation layout for the whole fleet. Widening this set widens
        # profiles, baselines, detectors, perturbations, and exports together.
        self.channel_set: ChannelSet = DEFAULT_CHANNEL_SET

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

        # Age structure first: it conditions who carries a core device at all
        # and which sensors they own on top of it.
        self._init_demographics()

        # Assign wearables (patchy by household)
        self._init_wearables()

        # Per-modality device ownership. When enabled the fleet widens the
        # observation layout to the union of every adopted device's channels;
        # non-owners simply never report the channels they have no sensor for.
        n_wearable = int(np.sum(self.has_wearable))
        self.device_fleet: DeviceFleet | None = None
        self.device_fleet_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 0xDEF1])
        )
        if self.config.devices.enabled:
            self.device_fleet = DeviceFleet(
                n_wearable,
                self.config.devices,
                np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xDEF0])),
                age_bands=(
                    self.age_bands[self.wearable_indices]
                    if self.config.demographics.enabled
                    else None
                ),
                demographics=self.config.demographics,
            )
            self.channel_set = self.device_fleet.channel_set

        # Initialize biometric profiles for wearable agents
        self.profiles = generate_profiles(n_wearable, self.rng, self.channel_set)

        # Initialize baseline trackers for wearable agents
        self.baselines: list[BaselineTracker] = [
            BaselineTracker(
                decay_lambda=self.config.baseline_decay_lambda,
                seasonal_decay=self.config.baseline_seasonal_decay,
                mean_prior_strength=(
                    self.config.baseline_mean_prior_strength
                    if self.config.baseline_mean_prior_source == "population"
                    else 0.0
                ),
                mean_prior=(
                    self.channel_set.resting_means
                    if self.config.baseline_mean_prior_source == "population"
                    else self.channel_set.zeros()
                ),
                channel_set=self.channel_set,
            )
            for _ in range(n_wearable)
        ]

        # SEIR engine
        self.seir = SEIREngine(config=self.config.seir)
        self.seir.initialize(self.config.n_agents, self.rng, self.agent_x, self.agent_y)
        self._baseline_infectious = self.seir.initial_infectious_count()

        self.plume_configs = self.config.plumes

        # Privacy protocol components
        self.zone_threshold_calibrator: ZoneThresholdCalibrator | None = None
        if self.config.zone_threshold_calibration.enabled:
            self.zone_threshold_calibrator = zone_threshold_calibrator_for(
                self.config.zone_threshold_calibration, self.config.privacy.threshold_m
            )
        self.aggregator = NetworkAggregator(
            config=self.config.privacy,
            threshold_calibrator=self.zone_threshold_calibrator,
        )
        self.advisory_engine = (
            AdvisoryEngine(self.config.advisories, self.advisory_rng)
            if self.config.advisories.enabled
            else None
        )
        self.metrics = MetricsCollector()
        self.metrics.configure_advisory_accounting(self.config.advisories.enabled)
        self.metrics.record_fleet_composition(self.fleet_composition())
        self.metrics.record_plume_configuration(
            {
                plume.plume_id: {
                    "start_step": plume.start_step,
                    "duration_steps": plume.duration_steps,
                }
                for plume in self.config.plumes
            }
        )
        self.metrics.record_outbreak_realized_seeds(
            {
                outbreak.outbreak_id: outbreak.initial_infected
                for outbreak in self.config.seir.outbreaks
            },
            self.seir.realized_seeds,
        )
        self.metrics.configure_privacy_accounting(
            response_mechanism=self.config.privacy.response_mechanism,
            response_basis=self.config.privacy.response_epsilon_basis,
            response_epsilon=self.config.privacy.response_epsilon(),
            unaffected_positive_probability=(
                0.5 * (1.0 - self.config.privacy.randomized_response_p)
                if self.config.privacy.response_mechanism == "randomized_response"
                else None
            ),
            aggregate_count_epsilon_per_release=self.config.privacy.aggregate_count_epsilon,
            aggregate_count_evidence_threshold=(
                self.config.privacy.aggregate_count_evidence_threshold()
            ),
            aggregate_count_minimum_releasable_count=(
                self.config.privacy.aggregate_count_minimum_releasable_count()
            ),
            geo_epsilon_per_metre=self.config.privacy.geo_epsilon_per_metre(),
            geo_basis=self.config.privacy.geo_epsilon_basis,
            ack_basis=self.config.disambiguation.ack_epsilon_basis,
        )
        self._detection_widths: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self._detection_emitted: NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
        self._detection_observed: NDArray[np.bool_] | None = None
        self.ablation_probe: AblationProbe | None = None
        if self.config.detection_power.channel_ablation_rate > 0.0:
            self.ablation_probe = AblationProbe(
                channel_set=self.channel_set,
                sample_rate=self.config.detection_power.channel_ablation_rate,
                rng=np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xAB1A])),
            )
            self.metrics.detection_power.ablation = self.ablation_probe
        self.alarm_calibrator: AlarmRateCalibrator | None = None
        if self.config.alarm_calibration.enabled and self.config.detector_mode == "instant":
            self.alarm_calibrator = calibrator_for(
                self.config.alarm_calibration, self.config.anomaly_threshold
            )

        # Agent objects (lightweight — heavy state in arrays)
        self.citizen_agents: list[CitizenAgent] = []
        self._pending_adoption_indices: set[int] = set()
        self._onboarding_cohorts: dict[str, set[int]] = {}
        self._init_citizen_agents()
        self._citizen_by_global_idx = {agent.idx: agent for agent in self.citizen_agents}

        # Device lifecycle (battery, removal, power-off)
        self.device_lifecycle_engine: DeviceLifecycleEngine | None = None
        self.household_centroid_x: np.ndarray | None = None
        self.household_centroid_y: np.ndarray | None = None
        needs_home_centroids = self.config.device_lifecycle.enabled or self.config.venues.enabled
        if needs_home_centroids:
            self._init_household_centroids()
        self.subsystem_lifecycle: SubsystemLifecycle | None = None
        if self.config.device_lifecycle.enabled:
            n_wearable = len(self.citizen_agents)
            self.device_lifecycle_engine = DeviceLifecycleEngine(
                n_wearable, self.config.device_lifecycle, self.rng
            )
            self._sync_citizen_device_state()
            if self.device_fleet is not None:
                # Every extra subsystem carries its own cell and its own habits,
                # so it runs down and comes off independently of the watch.
                self.subsystem_lifecycle = SubsystemLifecycle(
                    self.device_fleet,
                    self.config.device_lifecycle,
                    np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xDEF2])),
                )
        # Structured venues (optional activity-based mobility)
        self.venue_engine: VenueEngine | None = None
        if self.config.venues.enabled and self.config.venues.venues:
            self.venue_engine = VenueEngine(config=self.config.venues)
            self.venue_engine.initialize(
                self.config.n_agents,
                self.rng,
                self.agent_x,
                self.agent_y,
                self.household_ids,
                self.household_centroid_x,
                self.household_centroid_y,
            )
            if self.config.mobility_model == "random_walk":
                # Venues imply schedule-driven movement unless explicitly static.
                self.config.mobility_model = "schedule"

        self.confounder_engine = ConfounderEngine(
            self.config.n_agents,
            self.config.confounders,
            self.confounder_rng,
            tuple(int(zone_id) for zone_id in np.unique(self.grid.cell_ids)),
            self.household_ids,
            self.venue_engine,
            self.agent_x.astype(np.float64, copy=False),
            self.agent_y.astype(np.float64, copy=False),
            np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xE5])),
            channel_set=self.channel_set,
        )
        staged_benign_windows = self.confounder_engine.scheduled_benign_sources()
        self._configured_staged_benign_ids = frozenset(staged_benign_windows)
        self.metrics.record_staged_benign_configuration(staged_benign_windows)
        self._confounder_step = ConfounderStep({}, {})
        self._disambiguation_trigger_history: dict[int, list[int]] = {}
        self._disambiguation_breadth_baseline: float | None = None
        self._disambiguation_breadth_history: list[tuple[int, int, float | None]] = []
        self._disambiguation_breadth_time_bin: int | None = None

        self._initialize_adoption_state()
        self._mature_fleet_start_baselines()
        if self.device_lifecycle_engine is not None:
            engine = self.device_lifecycle_engine
            engine.status[:] = np.asarray(
                [int(agent.device_status) for agent in self.citizen_agents],
                dtype=np.int8,
            )
            self._sync_citizen_device_state()

        # Attack orchestrator
        self.attack_orchestrator = AttackOrchestrator(config=self.config.attacks)
        self._resolve_attack_defaults()

        # Metrics
        self._token_provenance_lookup: dict[
            tuple[int, AnomalyType, int, int], _TokenProvenance
        ] = {}
        self._provenance_group_counts: dict[tuple[int, AnomalyType], dict[int, list[int]]] = {}
        self._provenance_cause_counts: dict[
            tuple[int, AnomalyType], dict[int, dict[PerturbationCause, int]]
        ] = {}
        self.metrics.record_baseline_warmup_config(self.config.baseline_warmup_steps)
        fleet_start_mask = np.asarray(
            [agent.fleet_start_adopter for agent in self.citizen_agents],
            dtype=np.bool_,
        )
        self.metrics.record_baseline_maturation(
            minimum_history_days=self.config.baseline_maturation.minimum_history_days,
            maximum_history_days=self.config.baseline_maturation.maximum_history_days,
            cadence_steps=self.config.baseline_maturation.cadence_steps,
            fleet_start_count=sum(agent.fleet_start_adopter for agent in self.citizen_agents),
            history_days=self.baseline_maturation_history_days[fleet_start_mask],
            sample_counts=self.baseline_maturation_sample_counts[fleet_start_mask],
            circadian_bins=self.baseline_maturation_circadian_bins[fleet_start_mask],
            monthly_bins=self.baseline_maturation_monthly_bins[fleet_start_mask],
        )
        self.metrics.record_world_settling_config(self.config.world_settling_steps)
        self.metrics.record_population_config(self.config.n_agents)
        self.metrics.record_anomaly_threshold_config(self.config.anomaly_threshold)
        self.metrics.record_aggregation_threshold_config(self.config.privacy.threshold_m)
        self.metrics.record_aggregation_window_config(self.config.privacy.time_window_steps)
        self.metrics.record_detector_config(
            self.config.detector_mode,
            self.config.sequential_reference_value,
            self.config.sequential_threshold,
            self.config.sequential_clear_steps,
            self.config.sequential_clear_fraction,
            self.config.sequential_residual_ewma_alpha,
        )

    @property
    def plume_config(self) -> PlumeConfig:
        """First plume source (backward compatibility for tests)."""
        return self.plume_configs[0]

    def _init_positions(self) -> None:
        """Generate clustered agent positions by neighborhood."""
        n = self.config.n_agents
        # Create neighborhood centers
        n_neighborhoods = max(
            1, n // (self.config.households_per_neighborhood * self.config.household_size_mean)
        )

        def position_margin(length: float) -> float:
            return 500.0 if length > 1_000.0 else length * 0.25

        def position_spread(length: float) -> float:
            return 300.0 if length > 1_000.0 else length * 0.15

        margin_x = position_margin(self.config.grid_width)
        margin_y = position_margin(self.config.grid_height)
        neighborhood_centers_x = self.rng.uniform(
            margin_x, self.config.grid_width - margin_x, n_neighborhoods
        )
        neighborhood_centers_y = self.rng.uniform(
            margin_y, self.config.grid_height - margin_y, n_neighborhoods
        )

        # Assign agents to neighborhoods, then cluster within
        self.neighborhood_ids = self.rng.integers(0, n_neighborhoods, n)

        # Households are nested within neighborhoods (not index-ordered globally)
        self.household_ids = np.empty(n, dtype=np.int64)
        next_household_id = 0
        chunk = self.config.household_size_mean
        for nb in range(n_neighborhoods):
            members = np.nonzero(self.neighborhood_ids == nb)[0]
            self.rng.shuffle(members)
            for start in range(0, len(members), chunk):
                self.household_ids[members[start : start + chunk]] = next_household_id
                next_household_id += 1

        # Position = neighborhood center + Gaussian offset (vectorized)
        offsets_x = self.rng.normal(0, position_spread(self.config.grid_width), n)
        offsets_y = self.rng.normal(0, position_spread(self.config.grid_height), n)
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

    def fleet_composition(self) -> dict[str, Any]:
        """Who wears what: age bands, per-kind owners, and per-person spread.

        Reported so a run states its own heterogeneity rather than leaving it to
        be inferred from the config: two fleets with identical adoption
        fractions can differ entirely in how the devices are distributed across
        people, and that distribution is what the detector actually sees.
        """
        payload: dict[str, Any] = {
            "age_band_population": band_counts(self.age_bands),
            "age_band_wearers": band_counts(self.age_bands[self.wearable_indices]),
        }
        if self.device_fleet is None:
            return payload
        payload["owner_counts"] = self.device_fleet.owner_counts()
        payload.update(self.device_fleet.ownership_distribution())
        owners_by_band = self.device_fleet.owner_counts_by_band()
        if owners_by_band:
            payload["owner_counts_by_band"] = owners_by_band
        return payload

    def _init_demographics(self) -> None:
        """Assign an age band per agent from household composition.

        Bands are all-adult unless demographics are enabled, which is what the
        historical demographically flat fleet amounts to.
        """
        n = self.config.n_agents
        if not self.config.demographics.enabled:
            self.age_bands = np.full(n, AGE_BANDS.index(ADULT), dtype=np.int8)
            return
        self.age_bands = assign_age_bands(
            self.household_ids,
            self.config.demographics,
            np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xA6E5])),
        )

    def _apply_base_device_retention(
        self,
        has_wearable: NDArray[np.bool_],
        shuffled: NDArray[np.int64],
        selected: set[int],
        target_count: int,
    ) -> NDArray[np.bool_]:
        """Drop core devices per age band, then top up to the wearer target.

        An infant in a wearing household plausibly has no wrist device of its
        own even when its parents do, so base-device ownership is thinned by
        band inside the selected households. Thinning would otherwise undershoot
        ``wearable_fraction``, so further households are admitted until the
        wearer count is met and the configured penetration still holds.
        """
        retention = self.config.demographics.resolved_base_device_retention()
        keep_probability = np.array([retention[band] for band in AGE_BANDS], dtype=np.float64)[
            self.age_bands
        ]
        draws = self.rng.random(self.config.n_agents)
        retained = draws < keep_probability
        kept = has_wearable & retained
        for household in shuffled:
            if int(np.count_nonzero(kept)) >= target_count:
                break
            household_int = int(household)
            if household_int in selected:
                continue
            selected.add(household_int)
            kept |= (self.household_ids == household_int) & retained
        return kept

    def _init_wearables(self) -> None:
        """Assign wearables with household-patchy penetration."""
        n = self.config.n_agents
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

        has_wearable: NDArray[np.bool_] = np.isin(self.household_ids, list(wearable_households))
        if self.config.demographics.enabled:
            has_wearable = self._apply_base_device_retention(
                has_wearable, shuffled, wearable_households, target_count
            )
        self.has_wearable = has_wearable

        # Map: wearable global index → local profile index
        self.wearable_indices: NDArray[np.int64] = np.nonzero(self.has_wearable)[0]
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
                anomaly_threshold=self.config.anomaly_threshold,
                detector_mode=self.config.detector_mode,
                sequential_detector=(
                    SequentialDetector(
                        reference_value=self.config.sequential_reference_value,
                        threshold=self.config.sequential_threshold,
                        clear_steps=self.config.sequential_clear_steps,
                        clear_fraction=self.config.sequential_clear_fraction,
                        residual_ewma_alpha=self.config.sequential_residual_ewma_alpha,
                        channel_set=self.channel_set,
                    )
                    if self.config.detector_mode == "sequential"
                    else None
                ),
                cell_id=cell_id,
                baseline_warmup_remaining=(
                    self.config.baseline_warmup_steps
                    if self.config.adoption.mode == "all_at_start"
                    else 0
                ),
                fleet_start_adopter=self.config.adoption.mode == "all_at_start",
                ablation_probe=self.ablation_probe,
                alarm_calibrator=self.alarm_calibrator,
            )
            self.citizen_agents.append(agent)
            self.wearable_agents_by_cell.setdefault(cell_id, []).append(agent)

    def _initialize_adoption_state(self) -> None:
        """Set initial adoption status and retain pending adopters."""
        if self.config.adoption.mode == "all_at_start":
            self._register_onboarding_cohort(list(range(len(self.citizen_agents))), 0)
            return
        self._pending_adoption_indices = set(range(len(self.citizen_agents)))
        for agent in self.citizen_agents:
            agent.device_status = DeviceStatus.NOT_ADOPTED
            agent.adoption_step = None
            agent.steps_since_adoption = None
            agent.fleet_start_adopter = False
        fraction = min(max(self.config.adoption.initial_adopted_fraction, 0.0), 1.0)
        initial_count = int(np.floor(fraction * len(self.citizen_agents)))
        if initial_count == 0:
            return
        if initial_count == len(self.citizen_agents):
            initial = list(self._pending_adoption_indices)
        elif self.config.adoption.mode == "cohort":
            groups = self._adoption_groups(self._pending_adoption_indices)
            group_keys = list(groups)
            selected_indices = self.rng.permutation(len(group_keys))
            initial = []
            for group_index in selected_indices:
                initial.extend(groups[group_keys[int(group_index)]])
                if len(initial) >= initial_count:
                    break
        else:
            initial = self.rng.choice(
                len(self.citizen_agents), initial_count, replace=False
            ).tolist()
        for lidx in initial:
            agent = self.citizen_agents[int(lidx)]
            agent.device_status = DeviceStatus.ACTIVE
            agent.adoption_step = 0
            agent.steps_since_adoption = 0
            agent.fleet_start_adopter = True
            agent.baseline_warmup_remaining = max(
                self.config.baseline_warmup_steps,
                self.config.adoption.new_device_warmup_steps,
            )
            self._pending_adoption_indices.remove(int(lidx))
        self._register_onboarding_cohort(initial, 0)

    def _mature_fleet_start_baselines(self) -> None:
        """Learn device-local prior history without entering the main pipeline."""
        n_wearable = len(self.citizen_agents)
        self.baseline_maturation_history_days = np.zeros(n_wearable, dtype=np.int64)
        self.baseline_maturation_sample_counts = np.zeros(n_wearable, dtype=np.int64)
        self.baseline_maturation_circadian_bins = np.zeros(n_wearable, dtype=np.int64)
        self.baseline_maturation_monthly_bins = np.zeros(n_wearable, dtype=np.int64)
        config = self.config.baseline_maturation
        if not config.enabled:
            return

        for local_idx, agent in enumerate(self.citizen_agents):
            if not agent.fleet_start_adopter:
                continue
            history_days = int(
                self.baseline_maturation_rng.integers(
                    config.minimum_history_days,
                    config.maximum_history_days + 1,
                )
            )
            self.baseline_maturation_history_days[local_idx] = history_days
            history_steps = history_days * STEPS_PER_DAY
            last_offset = (history_steps // config.cadence_steps) * config.cadence_steps
            for offset in range(last_offset, 0, -config.cadence_steps):
                hour_of_day, hour_int, month, day_of_year = self._time_info_for_step(-offset)
                activity_level = self._compute_activity_level(hour_of_day)
                activity_level += self.baseline_maturation_rng.normal(0, 0.05)
                observation = generate_observation(
                    self.profiles[local_idx],
                    hour_of_day,
                    day_of_year,
                    self.baseline_maturation_rng,
                    activity_level=activity_level,
                    backend=self.config.biometric_synthesis,  # type: ignore[arg-type]
                    neurokit_window_seconds=self.config.neurokit_window_seconds,
                )
                observation = self.channel_set.clamp(observation)
                self.baselines[local_idx].update(observation, hour_int, month)

            tracker = self.baselines[local_idx]
            self.baseline_maturation_sample_counts[local_idx] = tracker.n_samples
            self.baseline_maturation_circadian_bins[local_idx] = int(
                np.count_nonzero(np.any(tracker.circadian_counts > 1.0, axis=1))
            )
            self.baseline_maturation_monthly_bins[local_idx] = int(
                np.count_nonzero(np.any(tracker.monthly_counts > 1.0, axis=1))
            )

    def _register_onboarding_cohort(self, indices: list[int], adoption_step: int) -> None:
        """Track stable model-side identities for newly adopted cohorts."""
        if not indices:
            return
        if self.config.adoption.mode == "cohort":
            groups = self._adoption_groups(indices)
            for key, members in groups.items():
                instance_id = f"onboarding_{key[0]}_{key[1]}"
                self._onboarding_cohorts.setdefault(instance_id, set()).update(
                    self.citizen_agents[lidx].idx for lidx in members
                )
        else:
            instance_id = f"onboarding_step_{adoption_step}"
            self._onboarding_cohorts.setdefault(instance_id, set()).update(
                self.citizen_agents[lidx].idx for lidx in indices
            )

    def _active_onboarding_cohorts(self) -> dict[str, set[int]]:
        """Return active onboarding cohorts and prune expired membership."""
        active: dict[str, set[int]] = {}
        for instance_id, indices in list(self._onboarding_cohorts.items()):
            current = {
                idx
                for idx in indices
                if self._citizen_by_global_idx[idx].is_operational
                and self._citizen_by_global_idx[idx].is_onboarding(
                    self.config.adoption.onboarding_window_steps
                )
            }
            if current:
                active[instance_id] = current
            else:
                del self._onboarding_cohorts[instance_id]
        return active

    def _init_household_centroids(self) -> None:
        """Compute per-household centroid positions for at-home detection."""
        unique_households = np.unique(self.household_ids)
        n_households = int(unique_households.max()) + 1 if len(unique_households) else 0
        self.household_centroid_x = np.zeros(n_households, dtype=np.float32)
        self.household_centroid_y = np.zeros(n_households, dtype=np.float32)
        for hh in unique_households:
            hh_int = int(hh)
            members = self.household_ids == hh_int
            self.household_centroid_x[hh_int] = float(np.mean(self.agent_x[members]))
            self.household_centroid_y[hh_int] = float(np.mean(self.agent_y[members]))

    def _wearable_at_home_mask(self) -> np.ndarray:
        """Return boolean mask (length W) for wearables within home radius."""
        cfg = self.config.device_lifecycle
        n_wearable = len(self.citizen_agents)
        at_home = np.zeros(n_wearable, dtype=bool)
        if self.household_centroid_x is None or self.household_centroid_y is None:
            return at_home

        for lidx, agent in enumerate(self.citizen_agents):
            hh = agent.household_id
            dx = float(self.agent_x[agent.idx]) - float(self.household_centroid_x[hh])
            dy = float(self.agent_y[agent.idx]) - float(self.household_centroid_y[hh])
            at_home[lidx] = (dx * dx + dy * dy) <= cfg.home_radius * cfg.home_radius
        return at_home

    def _sync_citizen_device_state(self) -> None:
        """Sync CitizenAgent fields from the lifecycle engine arrays."""
        if self.device_lifecycle_engine is None:
            return

        engine = self.device_lifecycle_engine
        warmup_steps = self.config.baseline_warmup_steps
        adopt_warmup = self.config.warmup_on_device_adopt
        for lidx, agent in enumerate(self.citizen_agents):
            new_status = DeviceStatus(int(engine.status[lidx]))
            re_adopted = (
                agent.device_status != DeviceStatus.ACTIVE and new_status == DeviceStatus.ACTIVE
            )
            if re_adopted:
                self.metrics.record_device_re_adoption(adopt_warmup and warmup_steps > 0)
                if adopt_warmup and warmup_steps > 0:
                    agent.baseline_warmup_remaining = warmup_steps
            if agent.device_status == DeviceStatus.ACTIVE and new_status != DeviceStatus.ACTIVE:
                agent.anomaly_active = False
                agent.anomaly_type = None
                agent.anomaly_onset_step = None
            agent.device_status = new_status
            agent.battery_level = float(engine.battery_levels[lidx])

    def _update_device_lifecycle(self, hour_of_day: float, activity_level: float) -> None:
        """Advance wearable battery, removal, power-off, and charging state."""
        if self.device_lifecycle_engine is None:
            return

        at_home = self._wearable_at_home_mask()
        self.device_lifecycle_engine.step(hour_of_day, activity_level, at_home, self.rng)
        if self.subsystem_lifecycle is not None:
            self.subsystem_lifecycle.step(hour_of_day, activity_level, at_home)
        self._sync_citizen_device_state()

    def _activate_subsystems(self, local_idx: int) -> None:
        """Bring an adopter's owned subsystems online with a full charge."""
        if self.subsystem_lifecycle is None or self.device_fleet is None:
            return
        for position in self.subsystem_lifecycle.positions:
            kind = self.device_fleet.kinds[position]
            if not self.device_fleet.ownership[local_idx, position]:
                continue
            engine = self.subsystem_lifecycle.engines[kind.name]
            engine.status[local_idx] = DeviceStatus.ACTIVE
            engine.battery_levels[local_idx] = engine.config.battery_capacity

    def _fleet_observed_matrix(
        self, hour_of_day: float, activity: float
    ) -> NDArray[np.bool_] | None:
        """Per-epoch observed-channel mask for the fleet, or None when disabled."""
        if self.device_fleet is None:
            return None
        subsystem_active = (
            None if self.subsystem_lifecycle is None else self.subsystem_lifecycle.active_matrix()
        )
        return self.device_fleet.observed_matrix(
            hour_of_day, activity, self.device_fleet_rng, subsystem_active
        )

    def _agent_observed_channels(
        self,
        agent: CitizenAgent,
        observed_matrix: NDArray[np.bool_] | None,
    ) -> NDArray[np.bool_] | None:
        """One agent's observed-channel mask, gated by wrist-device operability.

        The extra subsystems have already been gated by their own batteries. The
        wrist device keeps its per-person status, so a flat or removed watch
        drops the core vitals while leaving an owned band reporting; someone who
        has not adopted at all reports nothing.
        """
        if observed_matrix is None or self.device_fleet is None:
            return None
        row = observed_matrix[self.wearable_local_map[agent.idx]]
        if agent.device_status == DeviceStatus.NOT_ADOPTED:
            return np.zeros_like(row)
        if not agent.is_operational:
            row = row.copy()
            row[list(self.device_fleet.base_columns)] = False
        return row

    def _detection_power_buffers(
        self, with_masks: bool
    ) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_] | None]:
        """Reusable per-agent scratch arrays for the detection-power telemetry.

        Allocated once and cleared per step: at city scale a fresh mask matrix
        every step would be the largest allocation in the loop.
        """
        n_local = len(self.citizen_agents)
        widths = self._detection_widths
        emitted = self._detection_emitted
        if widths.size != n_local:
            widths = np.zeros(n_local, dtype=np.int64)
            emitted = np.zeros(n_local, dtype=np.bool_)
            self._detection_widths = widths
            self._detection_emitted = emitted
        widths[:] = 0
        emitted[:] = False
        if not with_masks:
            return widths, emitted, None
        masks = self._detection_observed
        if masks is None or masks.shape != (n_local, len(self.channel_set)):
            masks = np.zeros((n_local, len(self.channel_set)), dtype=np.bool_)
            self._detection_observed = masks
        masks[:] = False
        return widths, emitted, masks

    def _hazard_affected_mask(self, concentrations: np.ndarray) -> NDArray[np.bool_]:
        """Model-side truth that each wearable owner is exposed or infected.

        This is an oracle for measurement only: it is never available to the
        protocol, which sees opaque tokens.
        """
        local = self.wearable_indices
        exposed = concentrations[local] > self.config.toxin_exposure_concentration_gate()
        infected = np.isin(
            self.seir.states[local],
            (SEIRState.EXPOSED, SEIRState.INFECTIOUS),
        )
        return np.asarray(exposed | infected, dtype=np.bool_)

    def _record_scored_epoch(
        self,
        *,
        local_idx: int,
        observed_channels: NDArray[np.bool_] | None,
        widths: NDArray[np.int64],
        masks: NDArray[np.bool_] | None,
    ) -> None:
        """Note the width one warmed-up reporting agent was scored at.

        A warmed-up reporting epoch is one the detector could have alarmed on,
        which is the denominator the width-stratified rates need.
        """
        widths[local_idx] = (
            len(self.channel_set) if observed_channels is None else int(observed_channels.sum())
        )
        if masks is not None and observed_channels is not None:
            masks[local_idx] = observed_channels

    def _record_detection_power(
        self,
        *,
        widths: NDArray[np.int64],
        emitted: NDArray[np.bool_],
        masks: NDArray[np.bool_] | None,
        concentrations: np.ndarray,
    ) -> None:
        """Fold this step's width- and device-stratified outcomes into metrics."""
        hazard_affected = self._hazard_affected_mask(concentrations)
        if self.alarm_calibrator is not None:
            self.metrics.detection_power.alarm_calibration = self.alarm_calibrator.summary()
        if self.zone_threshold_calibrator is not None:
            self.metrics.detection_power.zone_threshold_calibration = (
                self.zone_threshold_calibrator.summary()
            )
        self.metrics.detection_power.record_epochs(
            step=self.current_step,
            widths=widths,
            emitted=emitted,
            hazard=hazard_affected,
        )
        if self.device_fleet is not None and masks is not None:
            self.metrics.detection_power.record_device_epochs(
                fleet=self.device_fleet,
                observed=masks,
                emitted=emitted,
                hazard=hazard_affected,
            )

    def _adoption_group_key(self, agent: CitizenAgent) -> tuple[str, int]:
        """Return the configured cohort grouping key for one wearable.

        ``venue_kind='any'`` checks workplace, school, hospital, third place,
        shopping, sporting, extended-family, then gathering assignments.
        """
        if self.config.adoption.group_by == "venue" and self.venue_engine is not None:
            assignments = {
                "workplace": self.venue_engine.assigned_workplace,
                "school": self.venue_engine.assigned_school,
                "hospital": self.venue_engine.assigned_hospital,
                "third_place": self.venue_engine.assigned_third_place,
                "shopping": self.venue_engine.assigned_shopping,
                "sporting": self.venue_engine.assigned_sporting,
                "extended_family": self.venue_engine.assigned_extended_family,
                "gathering": self.venue_engine.assigned_gathering,
            }
            venue_kind = self.config.adoption.venue_kind
            if venue_kind == "any":
                venue_kinds = tuple(assignments)
            else:
                venue_kinds = (venue_kind,)
            for kind in venue_kinds:
                assignment = assignments.get(kind)
                if assignment is not None:
                    venue_id = int(assignment[agent.idx])
                    if venue_id >= 0:
                        return (f"venue:{kind}", venue_id)
        return ("household", agent.household_id)

    def _adoption_groups(self, indices: set[int] | list[int]) -> dict[tuple[str, int], list[int]]:
        """Group pending wearable indices for cohort selection."""
        groups: dict[tuple[str, int], list[int]] = {}
        for lidx in indices:
            groups.setdefault(self._adoption_group_key(self.citizen_agents[lidx]), []).append(lidx)
        return groups

    def _select_adoptions(self) -> list[int]:
        """Select local wearable indices adopting at the current step."""
        cfg = self.config.adoption
        if cfg.mode == "all_at_start" or self.current_step < cfg.start_step:
            return []
        candidates = list(self._pending_adoption_indices)
        if not candidates:
            return []
        if cfg.mode == "rollout":
            if cfg.rate <= 0:
                return []
            count = min(len(candidates), int(np.ceil(cfg.rate * len(candidates))))
            if count == 0:
                return []
            return self.rng.choice(candidates, count, replace=False).tolist()
        if cfg.mode == "trickle":
            if cfg.rate <= 0:
                return []
            return [lidx for lidx in candidates if self.rng.random() < min(cfg.rate, 1.0)]
        if cfg.mode == "cohort":
            if cfg.interval_steps <= 0 or (self.current_step - cfg.start_step) % cfg.interval_steps:
                return []
            groups = self._adoption_groups(candidates)
            group_keys = list(groups)
            count = min(len(group_keys), max(1, cfg.cohort_size))
            selected_indices = self.rng.permutation(len(group_keys))[:count]
            return [
                lidx
                for group_index in selected_indices
                for lidx in groups[group_keys[int(group_index)]]
            ]
        raise ValueError(
            f"Unknown adoption mode {cfg.mode!r}; expected all_at_start, rollout, "
            "trickle, or cohort"
        )

    def _update_device_adoption(self) -> None:
        """Apply first-time adoption events before lifecycle transitions."""
        selected = self._select_adoptions()
        self._register_onboarding_cohort(selected, self.current_step)
        for lidx in selected:
            agent = self.citizen_agents[lidx]
            agent.device_status = DeviceStatus.ACTIVE
            agent.adoption_step = self.current_step
            agent.steps_since_adoption = 0
            agent.fleet_start_adopter = False
            agent.baseline_warmup_remaining = max(
                self.config.baseline_warmup_steps,
                self.config.adoption.new_device_warmup_steps,
            )
            if self.device_lifecycle_engine is not None:
                self.device_lifecycle_engine.status[lidx] = DeviceStatus.ACTIVE
                self.device_lifecycle_engine.battery_levels[lidx] = (
                    self.config.device_lifecycle.battery_capacity
                )
            self._activate_subsystems(lidx)
            self.metrics.record_device_adoption(self.current_step, agent.cell_id)
            self._pending_adoption_indices.discard(lidx)

    def _device_lifecycle_metrics(self) -> dict[str, float | int]:
        """Collect per-step device lifecycle metrics for CSV output."""
        n_wearable = len(self.citizen_agents)
        if self.device_lifecycle_engine is None:
            not_adopted = sum(
                agent.device_status == DeviceStatus.NOT_ADOPTED for agent in self.citizen_agents
            )
            return {
                "wearables_active": n_wearable - not_adopted,
                "wearables_offline": 0,
                "wearables_not_worn": 0,
                "wearables_powered_off": 0,
                "wearables_depleted": 0,
                "wearables_not_adopted": not_adopted,
                "wearables_adopted": n_wearable - not_adopted,
                "mean_battery_level": 1.0,
            }

        counts = self.device_lifecycle_engine.count_by_status()
        active = counts["active"]
        subsystem = {} if self.subsystem_lifecycle is None else self.subsystem_lifecycle.metrics()
        return {
            **subsystem,
            "wearables_active": active,
            "wearables_offline": n_wearable - active,
            "wearables_not_worn": counts["not_worn"],
            "wearables_powered_off": counts["powered_off"],
            "wearables_depleted": counts["depleted"],
            "wearables_not_adopted": counts["not_adopted"],
            "wearables_adopted": n_wearable - counts["not_adopted"],
            "mean_battery_level": float(np.mean(self.device_lifecycle_engine.battery_levels)),
        }

    def _resolve_attack_defaults(self) -> None:
        """Fill attack zone defaults from the target agent when unset."""
        attacks = self.config.attacks
        if not attacks.active_attacks:
            return

        target_idx = min(max(attacks.target_agent_idx, 0), self.config.n_agents - 1)
        target_cell = self.grid.cell_of(target_idx)

        if AttackType.SYBIL_INJECTION in attacks.active_attacks and attacks.sybil_target_zone == 0:
            attacks.sybil_target_zone = target_cell

        if AttackType.ECLIPSE in attacks.active_attacks and not attacks.eclipse_target_zones:
            attacks.eclipse_target_zones = [target_cell]

        self.attack_orchestrator.config = attacks
        self.attack_orchestrator._sync_sub_configs()

    def _update_mobility(self) -> None:
        """Advance agent positions and refresh spatial / wearable cell membership."""
        if self.config.mobility_model == "static":
            return

        if self.config.mobility_model == "schedule" and self.venue_engine is not None:
            hour_of_day, _, _, _ = self._current_time_info()
            weekday = self._current_weekday()
            self.agent_x, self.agent_y = self.venue_engine.update_positions(
                self.agent_x,
                self.agent_y,
                hour_of_day,
                weekday,
                self.rng,
                self.config.grid_width,
                self.config.grid_height,
            )
            self.grid.assign_positions(self.agent_x, self.agent_y)
            self._reconcile_wearable_cells()
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

    def _true_wearable_population(self, cell_id: int) -> int:
        """Return evaluation-only wearable count for a spatial cell."""
        agent_indices = self.grid.agents_in_cell(cell_id)
        return int(np.count_nonzero(self.has_wearable[agent_indices]))

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

    def _current_weekday(self) -> int:
        """Return ISO weekday (0=Monday) for the current simulation step."""
        minutes_elapsed = self.current_step * 5
        day_offset = minutes_elapsed // 1440
        start_weekday = self.config.start_datetime.weekday()
        return int((start_weekday + day_offset) % 7)

    def _time_info_for_step(self, step: int) -> tuple[float, int, int, int]:
        """Compute time parameters for a simulated step, including history."""
        minutes_elapsed = step * 5
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
        month_ends = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
        month = next(index + 1 for index, end in enumerate(month_ends) if day_of_year <= end)
        return hour_of_day, hour_int, month, day_of_year

    def _current_time_info(self) -> tuple[float, int, int, int]:
        """Compute current time parameters from the main-loop step count."""
        return self._time_info_for_step(self.current_step)

    def _track_disease_onset(self, infectious_count: int) -> None:
        if infectious_count <= self._baseline_infectious:
            return
        self.metrics.record_disease_onset(self.current_step)
        for outbreak_id in np.unique(self.seir.outbreak_origin):
            oid = str(outbreak_id)
            if not oid:
                continue
            seeded = (
                sum(
                    1
                    for outbreak in self.config.seir.outbreaks
                    if outbreak.outbreak_id == oid and outbreak.start_step <= self.current_step
                )
                or self.config.seir.initial_infected
            )
            outbreak_infectious = int(
                np.sum(
                    (self.seir.outbreak_origin == oid) & (self.seir.states == SEIRState.INFECTIOUS)
                )
            )
            if outbreak_infectious > seeded:
                self.metrics.record_disease_onset(self.current_step, oid)

    def _compute_activity_level(self, hour_of_day: float) -> float:
        if 6 <= hour_of_day <= 22:
            return float(0.3 * max(0, np.sin(np.pi * (hour_of_day - 6) / 12)))
        return 0.0

    def _agent_perturbation_contributions(
        self, gidx: int, concentrations: np.ndarray
    ) -> tuple[PerturbationContribution, ...]:
        contributions: list[PerturbationContribution] = []
        if self.seir.states[gidx] in (SEIRState.EXPOSED, SEIRState.INFECTIOUS):
            ref_step = (
                self.seir.infection_step[gidx]
                if self.seir.states[gidx] == SEIRState.INFECTIOUS
                else self.seir.exposure_step[gidx]
            )
            if ref_step >= 0:
                steps_since = self.current_step - ref_step
                delta = self.seir.biometric_perturbation(gidx, steps_since, self.channel_set)
                if np.any(~np.isclose(delta, 0.0)):
                    contributions.append(PerturbationContribution(PerturbationCause.DISEASE, delta))
        conc = concentrations[gidx]
        if conc > self.config.toxin_physiology_concentration_floor():
            contributions.append(
                PerturbationContribution(
                    PerturbationCause.TOXIN,
                    plume_biometric_perturbation(conc, self.channel_set),
                )
            )
        contributions.extend(self._confounder_step.contributions.get(gidx, ()))
        return tuple(contributions)

    def _agent_perturbation(self, gidx: int, concentrations: np.ndarray) -> np.ndarray:
        """Return the combined perturbation for compatibility with callers."""
        perturbation = self.channel_set.zeros()
        for contribution in self._agent_perturbation_contributions(gidx, concentrations):
            perturbation += contribution.delta
        return perturbation

    def _collect_step_tokens(
        self,
        *,
        hour_of_day: float,
        hour_int: int,
        month: int,
        day_of_year: int,
        time_bin: int,
        concentrations: np.ndarray,
        activity: float,
    ) -> tuple[
        list[EncryptedToken],
        int,
        dict[int, int],
        dict[tuple[int, AnomalyType], int],
        dict[int, int],
        int,
        int,
        int,
        int,
    ]:
        tokens: list[EncryptedToken] = []
        anomalies_detected = 0
        background_by_group: dict[tuple[int, AnomalyType], int] = {}
        background_by_agent: dict[int, int] = {}
        eligible_by_zone: dict[int, int] = {}
        operational_wearables = 0
        cold_baseline_wearables = 0
        cold_baseline_emission = False
        onboarding_cold_by_zone: dict[int, int] = {}
        onboarding_by_zone: dict[int, int] = {}
        # Provenance is consumed after emission in this step; hash collisions
        # between agents within a step remain a measurement approximation.
        self._token_provenance_lookup.clear()
        observed_matrix = self._fleet_observed_matrix(hour_of_day, activity)
        scored_widths, scored_emitted, scored_masks = self._detection_power_buffers(
            observed_matrix is not None
        )
        for local_idx, agent in enumerate(self.citizen_agents):
            gidx = agent.idx
            cell_id = agent.cell_id
            contributions = self._agent_perturbation_contributions(gidx, concentrations)
            perturbation = self.channel_set.zeros()
            for contribution in contributions:
                perturbation += contribution.delta
            suppress_tokens = agent.baseline_warmup_remaining > 0
            cold_baseline = agent.baseline.n_samples < COVARIANCE_WARMUP_SAMPLES
            if agent.adoption_step is not None:
                agent.steps_since_adoption = self.current_step - agent.adoption_step
            observed_channels = self._agent_observed_channels(agent, observed_matrix)
            reporting = (
                agent.is_operational if observed_channels is None else bool(observed_channels.any())
            )
            if reporting:
                operational_wearables += 1
                if agent.is_onboarding(self.config.adoption.onboarding_window_steps):
                    onboarding_by_zone[cell_id] = onboarding_by_zone.get(cell_id, 0) + 1
                if cold_baseline:
                    cold_baseline_wearables += 1
                    if agent.is_onboarding(self.config.adoption.onboarding_window_steps):
                        onboarding_cold_by_zone[cell_id] = (
                            onboarding_cold_by_zone.get(cell_id, 0) + 1
                        )
            if reporting and not suppress_tokens:
                eligible_by_zone[cell_id] = eligible_by_zone.get(cell_id, 0) + 1
                self._record_scored_epoch(
                    local_idx=local_idx,
                    observed_channels=observed_channels,
                    widths=scored_widths,
                    masks=scored_masks,
                )
            has_perturbation = bool(np.any(~np.isclose(perturbation, 0.0)))
            agent.observation_step = self.current_step
            token = agent.observe_and_detect(
                hour=hour_int,
                month=month,
                day_of_year=day_of_year,
                hour_of_day=hour_of_day,
                rng=self.rng,
                cell_id=cell_id,
                perturbations=contributions if has_perturbation else None,
                activity_level=activity + self.rng.normal(0, 0.05),
                synthesis_backend=self.config.biometric_synthesis,  # type: ignore[arg-type]
                neurokit_window_seconds=self.config.neurokit_window_seconds,
                suppress_token_emission=suppress_tokens,
                observed_channels=observed_channels,
            )
            if token is not None:
                if cold_baseline and agent.fleet_start_adopter:
                    cold_baseline_emission = True
                token = EncryptedToken(
                    zone_id=token.zone_id,
                    anomaly_type=token.anomaly_type,
                    timestamp_bin=time_bin,
                    agent_id_hash=token.agent_id_hash,
                    is_dummy=token.is_dummy,
                )
                tokens.append(token)
                self._token_provenance_lookup[
                    (
                        token.zone_id,
                        token.anomaly_type,
                        token.timestamp_bin,
                        token.agent_id_hash,
                    )
                ] = _TokenProvenance(
                    zone_id=token.zone_id,
                    anomaly_type=token.anomaly_type,
                    timestamp_bin=token.timestamp_bin,
                    toxin_affected=bool(
                        concentrations[gidx] > self.config.toxin_exposure_concentration_gate()
                    ),
                    disease_affected=self.seir.states[gidx]
                    in (SEIRState.EXPOSED, SEIRState.INFECTIOUS),
                    causes=frozenset(contribution.cause for contribution in contributions)
                    | (
                        {PerturbationCause.ONBOARDING}
                        if agent.is_onboarding(self.config.adoption.onboarding_window_steps)
                        else set()
                    ),
                )
                provenance = self._token_provenance_lookup[
                    (
                        token.zone_id,
                        token.anomaly_type,
                        token.timestamp_bin,
                        token.agent_id_hash,
                    )
                ]
                if not provenance.toxin_affected and not provenance.disease_affected:
                    key = (token.zone_id, token.anomaly_type)
                    background_by_group[key] = background_by_group.get(key, 0) + 1
                    background_by_agent[gidx] = background_by_agent.get(gidx, 0) + 1
                scored_emitted[local_idx] = True
                anomalies_detected += 1
            dummy = agent.generate_dummy_traffic(
                float(self.agent_x[gidx]),
                float(self.agent_y[gidx]),
                cell_id,
                self.config.privacy,
                self.rng,
                suppress_token_emission=suppress_tokens,
            )
            if agent.is_operational and agent.baseline_warmup_remaining > 0:
                agent.baseline_warmup_remaining -= 1
            if dummy is not None:
                if cold_baseline and agent.fleet_start_adopter:
                    cold_baseline_emission = True
                tokens.append(
                    EncryptedToken(
                        zone_id=dummy.zone_id,
                        anomaly_type=dummy.anomaly_type,
                        timestamp_bin=time_bin,
                        agent_id_hash=dummy.agent_id_hash,
                        is_dummy=True,
                    )
                )
        self.metrics.record_fleet_cold_start(cold_baseline_emission)
        self._record_detection_power(
            widths=scored_widths,
            emitted=scored_emitted,
            masks=scored_masks,
            concentrations=concentrations,
        )
        return (
            tokens,
            anomalies_detected,
            eligible_by_zone,
            background_by_group,
            background_by_agent,
            operational_wearables,
            cold_baseline_wearables,
            max(onboarding_cold_by_zone.values(), default=0),
            max(onboarding_by_zone.values(), default=0),
        )

    def _record_token_provenance(self, tokens: list[EncryptedToken]) -> None:
        """Track surviving token provenance outside the privacy protocol."""
        for token in tokens:
            if token.is_dummy:
                continue
            provenance = self._token_provenance_lookup.get(
                (
                    token.zone_id,
                    token.anomaly_type,
                    token.timestamp_bin,
                    token.agent_id_hash,
                )
            )
            if provenance is None:
                continue
            group_key = (provenance.zone_id, provenance.anomaly_type)
            group_counts = self._provenance_group_counts.setdefault(group_key, {})
            counts = group_counts.setdefault(provenance.timestamp_bin, [0, 0, 0])
            counts[0] += 1
            counts[1] += int(provenance.toxin_affected)
            counts[2] += int(provenance.disease_affected)
            cause_counts = self._provenance_cause_counts.setdefault(group_key, {})
            cause_bin_counts = cause_counts.setdefault(provenance.timestamp_bin, {})
            for cause in sorted(provenance.causes, key=lambda item: item.value):
                cause_bin_counts[cause] = cause_bin_counts.get(cause, 0) + 1
            window_start = provenance.timestamp_bin - self.config.privacy.time_window_steps
            group_size = [
                sum(
                    bin_counts[index]
                    for timestamp_bin, bin_counts in group_counts.items()
                    if timestamp_bin >= window_start
                )
                for index in (1, 2)
            ]
            if provenance.toxin_affected:
                self.metrics.record_affected_agent_token(
                    "toxin", provenance.anomaly_type, group_size[0]
                )
            if provenance.disease_affected:
                self.metrics.record_affected_agent_token(
                    "disease", provenance.anomaly_type, group_size[1]
                )

    def _prune_token_provenance(self, time_bin: int) -> None:
        window_start = time_bin - self.config.privacy.time_window_steps
        for group_key, group_counts in list(self._provenance_group_counts.items()):
            for timestamp_bin in list(group_counts):
                if timestamp_bin < window_start:
                    del group_counts[timestamp_bin]
            if not group_counts:
                del self._provenance_group_counts[group_key]
            cause_counts = self._provenance_cause_counts.get(group_key)
            if cause_counts is not None:
                for timestamp_bin in list(cause_counts):
                    if timestamp_bin < window_start:
                        del cause_counts[timestamp_bin]
                if not cause_counts:
                    del self._provenance_cause_counts[group_key]

    def _clear_query_provenance(self, query: BroadcastQuery, time_bin: int) -> None:
        """Mirror aggregator consumption of a threshold-crossing source group."""
        window_start = time_bin - self.config.privacy.time_window_steps
        source_zones = {
            zone_id
            for zone_id in query.zone_cells
            if sum(
                bin_counts[0]
                for timestamp_bin, bin_counts in self._provenance_group_counts.get(
                    (zone_id, query.anomaly_type), {}
                ).items()
                if timestamp_bin >= window_start
            )
            >= self.config.privacy.threshold_m
        }
        for zone_id in source_zones:
            group_key = (zone_id, query.anomaly_type)
            group_counts = self._provenance_group_counts.get(group_key)
            if group_counts is None:
                continue
            for timestamp_bin in list(group_counts):
                if timestamp_bin < time_bin:
                    del group_counts[timestamp_bin]
            if not group_counts:
                del self._provenance_group_counts[group_key]
            cause_counts = self._provenance_cause_counts.get(group_key)
            if cause_counts is not None:
                for timestamp_bin in list(cause_counts):
                    if timestamp_bin < time_bin:
                        del cause_counts[timestamp_bin]
                if not cause_counts:
                    del self._provenance_cause_counts[group_key]

    def _query_has_affected_support(
        self, query: BroadcastQuery, time_bin: int
    ) -> tuple[dict[str, bool], frozenset[PerturbationCause]]:
        """Return affected-agent support for a threshold-crossing source group."""
        window_start = time_bin - self.config.privacy.time_window_steps
        support = {"toxin": False, "disease": False}
        causes: set[PerturbationCause] = set()
        for zone_id in query.zone_cells:
            bin_counts = self._provenance_group_counts.get((zone_id, query.anomaly_type), {})
            group_totals = [
                sum(
                    counts[index]
                    for timestamp_bin, counts in bin_counts.items()
                    if timestamp_bin >= window_start
                )
                for index in range(3)
            ]
            if group_totals[0] < self.config.privacy.threshold_m:
                continue
            support["toxin"] = support["toxin"] or group_totals[1] > 0
            support["disease"] = support["disease"] or group_totals[2] > 0
            cause_counts = self._provenance_cause_counts.get((zone_id, query.anomaly_type), {})
            for timestamp_bin, counts in cause_counts.items():
                if timestamp_bin >= window_start:
                    causes.update(counts)
        support["benign"] = bool(causes & BENIGN_CAUSES)
        return support, frozenset(causes)

    def _apply_attack_layer(
        self,
        tokens: list[EncryptedToken],
        time_bin: int,
    ) -> tuple[list[EncryptedToken], int, int, int]:
        sybil_injected = 0
        replay_injected = 0
        eclipse_dropped = 0
        if not self.config.attacks.active_attacks:
            return tokens, sybil_injected, replay_injected, eclipse_dropped
        tokens, eclipse_dropped = self.attack_orchestrator.filter_tokens(tokens, self.rng)
        fake_tokens, sybil_injected, replay_injected = self.attack_orchestrator.step_injections(
            self.current_step, time_bin, self.rng
        )
        tokens.extend(fake_tokens)
        return tokens, sybil_injected, replay_injected, eclipse_dropped

    def _collect_zone_responses(self, query: BroadcastQuery) -> list:
        """Ask every wearable currently inside the query's zone to answer it."""
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
        return responses

    def _process_queries(
        self,
        queries: list,
        *,
        concentrations: np.ndarray,
        per_plume: dict[str, np.ndarray],
        time_bin: int,
    ) -> int:
        responses_received = 0
        time_window_steps = self.config.privacy.time_window_steps
        for query in queries:
            epsilon_before = self.aggregator.state.total_epsilon
            responses = self._collect_zone_responses(query)
            estimated_population = self.aggregator.dilation_estimates_by_query_id[query.query_id]
            if estimated_population is None:
                raise RuntimeError("issued dilation query is missing a population estimate")
            released_count = self.aggregator.collect_responses(
                responses,
                population=estimated_population,
                rng=self.rng,
                query_id=query.query_id,
            )
            response_epsilon_burned = self.aggregator.state.total_epsilon - epsilon_before
            genuine_count = sum(
                int(response.anomaly_confirmed and not response.is_dummy) for response in responses
            )
            under_k_release = not self.aggregator.release_broadcast_aggregate(estimated_population)
            release_suppressed = under_k_release and self.config.privacy.enforce_release_k_anonymity
            true_population = sum(
                self._true_wearable_population(cell_id) for cell_id in query.zone_cells
            )
            self.metrics.record_dilation(
                step=self.current_step,
                dilated_cell_count=len(query.zone_cells),
                resident_population=sum(
                    self.grid.zone_population(cell_id) for cell_id in query.zone_cells
                ),
                estimated_respondent_population=estimated_population,
                true_respondent_population=true_population,
                k_min=self.config.privacy.k_min,
                responding_devices=len(responses),
                release_suppressed=under_k_release,
                response_epsilon_burned=response_epsilon_burned,
            )
            if released_count is not None:
                self.metrics.record_aggregate_count_release(
                    query_id=query.query_id,
                    released_count=released_count,
                    true_count=genuine_count,
                    population=estimated_population,
                    epsilon=self.config.privacy.aggregate_count_epsilon,
                    composed_epsilon=self.aggregator.state.aggregate_count_epsilon,
                    evidence_threshold=self.config.privacy.aggregate_count_evidence_threshold(),
                )
            responses_received += len(responses)
            self.attack_orchestrator.observe_protocol_responses(
                time_bin, responses, time_window_steps
            )
            if not release_suppressed:
                self._classify_detection(
                    query,
                    responses,
                    concentrations,
                    per_plume,
                    released_count=released_count,
                )
            self._clear_query_provenance(query, time_bin)
        return responses_received

    def _update_disambiguation_history(
        self, queries: list[BroadcastQuery], time_bin: int
    ) -> set[int]:
        trigger_cells = {
            self._trigger_cell_for_query(query) for query in queries if query.zone_cells
        }
        for trigger_cell in trigger_cells:
            history = self._disambiguation_trigger_history.setdefault(trigger_cell, [])
            if not history or history[-1] != time_bin:
                history.append(time_bin)
        history_start = time_bin - max(self.config.disambiguation.trigger_history_steps - 1, 0)
        for trigger_cell, history in list(self._disambiguation_trigger_history.items()):
            retained = [value for value in history if value >= history_start]
            if retained:
                self._disambiguation_trigger_history[trigger_cell] = retained
            else:
                del self._disambiguation_trigger_history[trigger_cell]
        return trigger_cells

    def _update_disambiguation_breadth(self, breadth: int, time_bin: int) -> None:
        """Record a broadcast bin's breadth and prior channel baseline."""
        if self._disambiguation_breadth_time_bin == time_bin:
            return
        baseline = self._disambiguation_breadth_baseline
        self._disambiguation_breadth_history.append((time_bin, breadth, baseline))
        history_steps = self.config.disambiguation.trigger_history_steps
        self._disambiguation_breadth_history = [
            item
            for item in self._disambiguation_breadth_history
            if item[0] >= time_bin - max(history_steps - 1, 0)
        ]
        alpha = self.config.disambiguation.breadth_baseline_alpha
        self._disambiguation_breadth_baseline = (
            breadth if baseline is None else (1.0 - alpha) * baseline + alpha * breadth
        )
        self._disambiguation_breadth_time_bin = time_bin

    def _disambiguation_worthwhile(
        self,
        query: BroadcastQuery,
        hypothesis: DisambiguationHypothesis,
        threshold: DisambiguationTriggerConfig,
    ) -> bool:
        if hypothesis is DisambiguationHypothesis.AMBIENT_HEAT:
            return len(
                self._disambiguation_breadth_history
            ) >= threshold.min_breadth_windows and all(
                bin_breadth >= threshold.min_breadth
                and prior_baseline is not None
                and bin_breadth > threshold.breadth_ratio * prior_baseline
                for _, bin_breadth, prior_baseline in self._disambiguation_breadth_history[
                    -threshold.min_breadth_windows :
                ]
            )
        trigger_cell = self._trigger_cell_for_query(query)
        responses = [
            response
            for response in self.aggregator.state.responses
            if response.query_id == query.query_id and not response.is_dummy
        ]
        if self.config.privacy.response_mechanism == "aggregate_noisy_count":
            released_count = self.aggregator.state.aggregate_count_released_by_query_id.get(
                query.query_id, 0
            )
            estimated_population = self.aggregator.dilation_estimates_by_query_id.get(
                query.query_id
            )
            confirmed_fraction = (
                released_count / estimated_population
                if estimated_population
                and released_count > self.config.privacy.aggregate_count_evidence_threshold()
                else 0.0
            )
        else:
            confirmed_fraction = (
                sum(response.anomaly_confirmed for response in responses) / len(responses)
                if responses
                else 0.0
            )
        return (
            len(set(query.zone_cells)) <= threshold.max_zone_cells
            and len(self._disambiguation_trigger_history.get(trigger_cell, []))
            >= threshold.min_persistent_windows
            and confirmed_fraction <= threshold.max_confirmed_fraction
        )

    def _trigger_cell_for_query(self, query: BroadcastQuery) -> int:
        """Return the aggregator-private trigger identity for a broadcast."""
        return self.aggregator._trigger_cells_by_query_id.get(query.query_id, min(query.zone_cells))

    def _run_disambiguation_query(self, query: DisambiguationQuery) -> _DisambiguationQueryOutcome:
        config = self.config.disambiguation
        epsilon_before = (
            self.aggregator.state.disambiguation_answer_epsilon
            + self.aggregator.state.disambiguation_ack_epsilon
        )
        reached = 0
        acks = 0
        yes = 0
        no = 0
        pending = 0
        for cell_id in query.zone_cells:
            for agent in self.wearable_agents_by_cell.get(cell_id, ()):
                if not agent.is_operational:
                    continue
                reached += 1
                if self.attack_orchestrator.suppresses_zone(cell_id, self.disambiguation_rng):
                    continue
                acks += 1
                answer = agent.respond_to_disambiguation(
                    query,
                    cell_id,
                    config.answer_rate,
                    config.yes_rate,
                    self.config.privacy,
                    self.disambiguation_rng,
                )
                if answer is None:
                    pending += 1
                elif answer:
                    yes += 1
                else:
                    no += 1
        release = self.aggregator.release_disambiguation_ack(
            acks,
            reached,
            self.config.privacy.k_min,
            config.ack_noise_scale,
            self.disambiguation_rng,
            config.ack_epsilon,
        )
        approved = yes + no
        self.aggregator.record_disambiguation_answers(
            approved, self.config.privacy.response_epsilon()
        )
        self.aggregator.register_disambiguation_pending(
            query.query_id,
            self.current_step + max(config.expiry_steps, 0),
            pending,
            approved > 0,
        )
        epsilon_after = (
            self.aggregator.state.disambiguation_answer_epsilon
            + self.aggregator.state.disambiguation_ack_epsilon
        )
        expected_cause = {
            DisambiguationHypothesis.RECENT_ADOPTION: PerturbationCause.ONBOARDING,
            DisambiguationHypothesis.AMBIENT_HEAT: PerturbationCause.HEAT_WAVE,
        }[query.hypothesis]
        benign_instances = self._zone_benign_instances(query.zone_cells)
        if not benign_instances:
            score = DisambiguationScore.UNSCORED
        elif any(instance.cause is expected_cause for instance in benign_instances):
            score = DisambiguationScore.WELL_FOUNDED
        else:
            score = DisambiguationScore.UNFOUNDED
        return _DisambiguationQueryOutcome(
            reached,
            acks,
            yes,
            no,
            pending,
            release,
            epsilon_after - epsilon_before,
            score,
        )

    def _process_disambiguation_queries(
        self, queries: list[BroadcastQuery], time_bin: int
    ) -> _DisambiguationResult:
        """Run the optional contextual, human-approved second-round query."""
        config = self.config.disambiguation
        expired_unanswered, expired_unresolved = self.aggregator.expire_disambiguation(
            self.current_step
        )
        result = _DisambiguationResult(
            unanswered=expired_unanswered,
            unresolved=expired_unresolved,
        )
        if not config.enabled:
            return result

        current_footprints = self._update_disambiguation_history(queries, time_bin)
        breadth = len(current_footprints)
        if current_footprints and self.current_step >= self.config.world_settling_steps:
            self._update_disambiguation_breadth(breadth, time_bin)
        hypotheses = sorted(
            config.enabled_hypotheses,
            key=lambda hypothesis: hypothesis.value,
        )
        for hypothesis in hypotheses:
            threshold: DisambiguationTriggerConfig = getattr(config, hypothesis.value)
            issued = self.aggregator.issue_disambiguation_queries(
                queries,
                hypothesis,
                should_ask=partial(
                    self._disambiguation_worthwhile,
                    hypothesis=hypothesis,
                    threshold=threshold,
                ),
            )
            for query in issued:
                current_epsilon = (
                    self.aggregator.state.disambiguation_answer_epsilon
                    + self.aggregator.state.disambiguation_ack_epsilon
                )
                if config.ask_epsilon_budget > 0.0 and current_epsilon >= config.ask_epsilon_budget:
                    result.suppressed_by_budget += 1
                    continue
                result.queries += 1
                outcome = self._run_disambiguation_query(query)
                result.max_ask_epsilon_delta = max(
                    result.max_ask_epsilon_delta,
                    outcome.epsilon_delta,
                )
                result.reached += outcome.reached
                result.acks += outcome.acks
                result.ack_releases += outcome.ack_release
                result.yes += outcome.yes
                result.no += outcome.no
                hypothesis_key = query.hypothesis.value
                if outcome.score is DisambiguationScore.WELL_FOUNDED:
                    result.well_founded += 1
                    result.well_founded_by_hypothesis[hypothesis_key] = (
                        result.well_founded_by_hypothesis.get(hypothesis_key, 0) + 1
                    )
                elif outcome.score is DisambiguationScore.UNFOUNDED:
                    result.unfounded += 1
                    result.unfounded_epsilon += outcome.epsilon_delta
                    result.unfounded_by_hypothesis[hypothesis_key] = (
                        result.unfounded_by_hypothesis.get(hypothesis_key, 0) + 1
                    )
                else:
                    result.unscored += 1
                    result.unscored_epsilon += outcome.epsilon_delta
                    result.unscored_by_hypothesis[hypothesis_key] = (
                        result.unscored_by_hypothesis.get(hypothesis_key, 0) + 1
                    )
        return result

    def _record_attack_side_effects(
        self,
        queries: list,
        *,
        sybil_injected: int,
        replay_injected: int,
    ) -> None:
        sybil_zone = self.config.attacks.sybil_target_zone
        for query in queries:
            if (
                sybil_injected > 0
                and AttackType.SYBIL_INJECTION in self.config.attacks.active_attacks
                and sybil_zone in query.zone_cells
            ):
                self.metrics.record_sybil_false_alert()
                self.attack_orchestrator.false_positives_triggered += 1
            if replay_injected > 0 and AttackType.REPLAY in self.config.attacks.active_attacks:
                self.attack_orchestrator.record_replay_false_alerts(query.zone_cells)

    def _update_hazard_episodes(
        self,
        *,
        infectious_count: int,
        plume_exposed_count: int,
    ) -> None:
        has_active_disease = infectious_count > self._baseline_infectious
        has_active_plume = plume_exposed_count > 0
        step_events = [e for e in self.metrics.detection_events if e.step == self.current_step]
        disease_tp_this_step = any(
            e.hazard_type == "disease" and e.true_positive for e in step_events
        )
        disease_fp_this_step = any(
            e.hazard_type == "disease" and not e.true_positive for e in step_events
        )
        toxin_tp_this_step = any(e.hazard_type == "toxin" and e.true_positive for e in step_events)
        toxin_fp_this_step = any(
            e.hazard_type == "toxin" and not e.true_positive for e in step_events
        )
        self.metrics.update_hazard_episode(
            "disease", has_active_disease, disease_tp_this_step, disease_fp_this_step
        )
        self.metrics.update_hazard_episode(
            "toxin", has_active_plume, toxin_tp_this_step, toxin_fp_this_step
        )

    def _broadcasts_deferred_for_calibration(self) -> bool:
        """Whether this step's zone triggers are withheld as pre-calibration.

        Tokens minted before the alarm scales freeze come from cuts the run is
        in the middle of establishing are miscalibrated, so broadcasting on them
        spends response epsilon on known-inflated evidence. Ingestion continues,
        so a zone that was already accumulating tokens when the scales froze can
        trigger on its remaining in-window history.
        """
        calibrator = self.alarm_calibrator
        return (
            calibrator is not None
            and self.config.alarm_calibration.defer_broadcasts_until_frozen
            and not calibrator.frozen
        )

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
        if self.alarm_calibrator is not None:
            self.alarm_calibrator.advance(self.current_step)
        if self.zone_threshold_calibrator is not None:
            self.zone_threshold_calibrator.advance(self.current_step)

        # --- 0. Agent Mobility ---
        self._update_mobility()

        # --- 1. SEIR Step ---
        self.seir.maybe_seed_outbreaks(self.current_step, self.agent_x, self.agent_y, self.rng)
        self.metrics.record_outbreak_realized_seeds(
            {
                outbreak.outbreak_id: outbreak.initial_infected
                for outbreak in self.config.seir.outbreaks
            },
            self.seir.realized_seeds,
        )
        self.seir.step(
            self.current_step,
            self.agent_x,
            self.agent_y,
            self.rng,
            current_venue_idx=(self.venue_engine.current_venue_idx if self.venue_engine else None),
            venue_contact_multipliers=(
                [v.effective_contact_multiplier() for v in self.venue_engine.venues]
                if self.venue_engine
                else None
            ),
            use_proximity_contacts=(
                self.config.venues.use_proximity_contacts if self.venue_engine else True
            ),
            use_venue_contacts=(
                self.config.venues.use_venue_contacts if self.venue_engine else False
            ),
        )

        infectious_count = int(np.sum(self.seir.states == SEIRState.INFECTIOUS))
        self._track_disease_onset(infectious_count)

        concentrations, self._per_plume_concentrations = compute_plume_concentrations(
            self.agent_x, self.agent_y, self.plume_configs, self.current_step
        )
        exposure_gate = self.config.toxin_exposure_concentration_gate()
        plume_exposed_count = int(np.sum(concentrations > exposure_gate))
        for plume_id, plume_field in self._per_plume_concentrations.items():
            exposed = plume_field > exposure_gate
            exposed_agents = set(np.flatnonzero(exposed).tolist())
            self.metrics.record_plume_exposure(
                plume_id,
                self.current_step,
                exposed_agents,
                set(np.flatnonzero(exposed & self.has_wearable).tolist()),
            )
            if exposed_agents:
                self.metrics.record_toxin_onset(self.current_step, plume_id)

        activity = self._compute_activity_level(hour_of_day)
        confounders_enabled = self.config.confounders.enabled
        previously_operational = (
            {agent.idx for agent in self.citizen_agents if agent.is_operational}
            if confounders_enabled
            else set()
        )
        self._update_device_adoption()
        self._update_device_lifecycle(hour_of_day, activity)
        if confounders_enabled:
            operational_now = {agent.idx for agent in self.citizen_agents if agent.is_operational}
            self._confounder_step = self.confounder_engine.step(
                self.current_step,
                hour_of_day,
                self.has_wearable,
                operational_now - previously_operational,
                self.agent_x.astype(np.float64, copy=False),
                self.agent_y.astype(np.float64, copy=False),
            )
        else:
            self._confounder_step = ConfounderStep({}, {})
        if confounders_enabled:
            for instance_id, onboarding_agents in self._active_onboarding_cohorts().items():
                self._confounder_step.benign_instances[instance_id] = BenignInstance(
                    instance_id,
                    PerturbationCause.ONBOARDING,
                    self.current_step,
                    self.current_step + 1,
                    onboarding_agents,
                )

        (
            tokens,
            anomalies_detected,
            eligible_by_zone,
            background_by_group,
            background_by_agent,
            operational_wearables,
            cold_baseline_wearables,
            onboarding_cold_wearables_in_zone,
            onboarding_wearables_in_zone,
        ) = self._collect_step_tokens(
            hour_of_day=hour_of_day,
            hour_int=hour_int,
            month=month,
            day_of_year=day_of_year,
            time_bin=time_bin,
            concentrations=concentrations,
            activity=activity,
        )
        self.metrics.record_background_step(
            self.current_step,
            time_bin,
            eligible_by_zone,
            background_by_group,
            background_by_agent,
        )
        tokens, sybil_injected, replay_injected, eclipse_dropped = self._apply_attack_layer(
            tokens, time_bin
        )

        self.aggregator.ingest_tokens(tokens, time_bin)
        self._record_token_provenance(tokens)
        population_fn = self.aggregator.dilation_population_fn(
            time_bin,
            self.grid.zone_population,
            self._true_wearable_population
            if self.config.privacy.dilation_basis == "true_devices"
            else None,
            current_step=self.current_step,
        )
        queries = (
            []
            if self._broadcasts_deferred_for_calibration()
            else self.aggregator.evaluate_and_broadcast(
                time_bin,
                self.grid.dilated_zone,
                population_fn,
            )
        )
        advisory_issued = []
        if self.advisory_engine is not None:
            advisory_issued = self.advisory_engine.refresh(
                queries,
                self.wearable_agents_by_cell,
                self.current_step,
            )
        for estimate in self.aggregator.last_suppressed_dilation_estimates:
            self.metrics.record_dilation_suppressed(
                estimated_respondent_population=estimate,
                k_min=self.config.privacy.k_min,
                step=self.current_step,
            )
        if self.config.attacks.active_attacks:
            self.attack_orchestrator.cache_tokens_for_replay(tokens)
        self._record_attack_side_effects(
            queries,
            sybil_injected=sybil_injected,
            replay_injected=replay_injected,
        )

        responses_received = self._process_queries(
            queries,
            concentrations=concentrations,
            per_plume=self._per_plume_concentrations,
            time_bin=time_bin,
        )
        disambiguation = self._process_disambiguation_queries(queries, time_bin)
        advisory_result = None
        if self.advisory_engine is not None:
            disease_exposed = set(
                np.flatnonzero(
                    np.isin(self.seir.states, [SEIRState.EXPOSED, SEIRState.INFECTIOUS])
                ).tolist()
            )
            toxin_exposed = set(np.flatnonzero(concentrations > exposure_gate).tolist())
            advisory_result = self.advisory_engine.process_step(
                self.citizen_agents,
                self.current_step,
                disease_exposed,
                toxin_exposed,
            )
            for _ in advisory_result.released_counts:
                self.aggregator.state.record_advisory_confirmation_release(
                    self.config.advisories.advisory_confirmation_epsilon
                )
                self.metrics.record_advisory_epsilon(
                    self.config.advisories.advisory_confirmation_epsilon
                )
            self.metrics.record_advisory_step(
                step=self.current_step,
                issued=advisory_issued,
                clinic_visits=advisory_result.clinic_visits,
                confirmations_by_type=advisory_result.confirmations_by_type,
                released_counts=advisory_result.released_counts,
                agents=self.citizen_agents,
                disease_exposed=disease_exposed,
                toxin_exposed=toxin_exposed,
            )
        self._prune_token_provenance(time_bin)
        self._run_deanon_attack(time_bin)
        self.attack_orchestrator.evaluate_periodic(self.current_step, self.agent_x, self.agent_y)
        self.metrics.sync_attack_metrics(self.attack_orchestrator)
        self._update_hazard_episodes(
            infectious_count=infectious_count,
            plume_exposed_count=plume_exposed_count,
        )

        seir_counts = {
            "S": int(np.sum(self.seir.states == SEIRState.SUSCEPTIBLE)),
            "E": int(np.sum(self.seir.states == SEIRState.EXPOSED)),
            "I": int(np.sum(self.seir.states == SEIRState.INFECTIOUS)),
            "R": int(np.sum(self.seir.states == SEIRState.RECOVERED)),
        }
        lc = self._device_lifecycle_metrics()
        wearables_in_warmup = sum(
            1 for agent in self.citizen_agents if agent.baseline_warmup_remaining > 0
        )
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
            wearables_active=int(lc["wearables_active"]),
            wearables_offline=int(lc["wearables_offline"]),
            wearables_not_worn=int(lc["wearables_not_worn"]),
            wearables_powered_off=int(lc["wearables_powered_off"]),
            wearables_depleted=int(lc["wearables_depleted"]),
            not_adopted_wearables=int(lc["wearables_not_adopted"]),
            adopted_wearables=int(lc["wearables_adopted"]),
            mean_battery_level=float(lc["mean_battery_level"]),
            baseline_warmup_active=(
                self.config.baseline_warmup_steps > 0
                and self.current_step < self.config.baseline_warmup_steps
            ),
            wearables_in_warmup=wearables_in_warmup,
            background_tokens=sum(background_by_group.values()),
            background_eligible_wearables=sum(eligible_by_zone.values()),
            background_rate=(
                sum(background_by_group.values()) / sum(eligible_by_zone.values())
                if eligible_by_zone
                else None
            ),
            operational_wearables=operational_wearables,
            cold_baseline_wearables=cold_baseline_wearables,
            onboarding_cold_wearables_in_zone=onboarding_cold_wearables_in_zone,
            onboarding_wearables_in_zone=onboarding_wearables_in_zone,
            disambiguation_queries_issued=disambiguation.queries,
            disambiguation_asks_suppressed_by_budget=disambiguation.suppressed_by_budget,
            disambiguation_acks=disambiguation.acks,
            disambiguation_ack_release_count=disambiguation.ack_releases,
            disambiguation_devices_reached=disambiguation.reached,
            disambiguation_yes_answers=disambiguation.yes,
            disambiguation_no_answers=disambiguation.no,
            disambiguation_unanswered_expired=disambiguation.unanswered,
            disambiguation_unresolved_hypotheses=disambiguation.unresolved,
            disambiguation_answer_epsilon=(self.aggregator.state.disambiguation_answer_epsilon),
            disambiguation_ack_epsilon=self.aggregator.state.disambiguation_ack_epsilon,
            disambiguation_well_founded_queries=disambiguation.well_founded,
            disambiguation_unfounded_queries=disambiguation.unfounded,
            disambiguation_unscored_queries=disambiguation.unscored,
            disambiguation_unfounded_ask_epsilon=disambiguation.unfounded_epsilon,
            disambiguation_unscored_ask_epsilon=disambiguation.unscored_epsilon,
            disambiguation_max_ask_epsilon_delta=disambiguation.max_ask_epsilon_delta,
            disambiguation_well_founded_by_hypothesis=(disambiguation.well_founded_by_hypothesis),
            disambiguation_unfounded_by_hypothesis=(disambiguation.unfounded_by_hypothesis),
            disambiguation_unscored_by_hypothesis=(disambiguation.unscored_by_hypothesis),
            confounder_contributions={
                cause.value: len(
                    [
                        contribution
                        for contributions in self._confounder_step.contributions.values()
                        for contribution in contributions
                        if contribution.cause == cause
                    ]
                )
                for cause in self._confounder_step.affected_agents_by_cause
            },
            confounder_agents_affected={
                cause.value: agents
                for cause, agents in self._confounder_step.affected_agents_by_cause.items()
            },
            heat_wave_active=self._confounder_step.heat_wave_active,
            heat_wave_instance_id=self._confounder_step.heat_wave_instance_id,
            heat_wave_zone_ids=self.confounder_engine.zone_ids,
            heat_wave_start_step=self._confounder_step.heat_wave_start_step,
            heat_wave_end_step=self._confounder_step.heat_wave_end_step,
            staged_benign_agents={
                instance_id: set(instance.current_agents)
                for instance_id, instance in self._confounder_step.benign_instances.items()
                if instance_id in self._configured_staged_benign_ids
            },
            occupied_zone_ids=set(self.wearable_agents_by_cell),
            alarming_zone_ids={int(query.zone_cells[0]) for query in queries if query.zone_cells},
        )

        self.current_step += 1

    def _classify_detection(
        self,
        query,
        responses,
        concentrations: np.ndarray,
        per_plume: dict[str, np.ndarray] | None = None,
        released_count: int | None = None,
    ) -> None:
        """Classify a broadcast query as TP or FP for each hazard type."""
        genuine_responses = [r for r in responses if r.anomaly_confirmed and not r.is_dummy]

        if self.config.privacy.response_mechanism == "aggregate_noisy_count":
            if released_count is None:
                return
            affected_count = released_count
            if affected_count <= self.config.privacy.aggregate_count_evidence_threshold():
                return
        else:
            if not genuine_responses:
                return
            affected_count = len(genuine_responses)

        provenance_support, cause_support = self._query_has_affected_support(
            query, self.current_step // self.config.privacy.time_window_steps
        )
        benign_instances = self._zone_benign_instances(query.zone_cells)
        benign_instance = benign_instances[0] if benign_instances else None
        benign_instance_id = benign_instance.instance_id if benign_instance is not None else None
        benign_attributed = benign_instance is not None and benign_instance.cause in cause_support
        benign_cause = benign_instance.cause if benign_instance is not None else None
        benign_causes = frozenset(instance.cause for instance in benign_instances)

        per_plume = per_plume or getattr(self, "_per_plume_concentrations", {})
        if not per_plume:
            plume_id = self.plume_configs[0].plume_id if self.plume_configs else "plume_0"
            per_plume = {plume_id: concentrations}

        # Determine if this corresponds to a real hazard
        if query.anomaly_type == AnomalyType.RESPIRATORY:
            plume_instance = self._zone_plume_instance(query.zone_cells, per_plume)
            is_toxin_tp = plume_instance is not None
            dosed_agents = (
                self._count_dosed_query_agents(query, concentrations, affected_count)
                if is_toxin_tp
                else None
            )
            event = DetectionEvent(
                step=self.current_step,
                hazard_type="toxin" if is_toxin_tp else "disease",
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=is_toxin_tp,
                agents_affected=affected_count,
                dosed_agents=dosed_agents,
                hazard_instance_id=plume_instance,
                attributed=provenance_support["toxin"] if is_toxin_tp else False,
                causes=cause_support,
                benign_instance_id=benign_instance_id,
                benign_cause=benign_cause,
                benign_causes=benign_causes,
                benign_attributed=benign_attributed,
            )
            self.metrics.record_detection(event)
        elif query.anomaly_type in (AnomalyType.FEBRILE, AnomalyType.MULTI_SYSTEM):
            outbreak_instance = self._zone_outbreak_instance(query.zone_cells)
            is_disease_tp = outbreak_instance is not None
            event = DetectionEvent(
                step=self.current_step,
                hazard_type="disease",
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=is_disease_tp,
                agents_affected=affected_count,
                hazard_instance_id=outbreak_instance,
                attributed=provenance_support["disease"] if is_disease_tp else False,
                causes=cause_support,
                benign_instance_id=benign_instance_id,
                benign_cause=benign_cause,
                benign_causes=benign_causes,
                benign_attributed=benign_attributed,
            )
            self.metrics.record_detection(event)
        elif query.anomaly_type == AnomalyType.CARDIAC:
            plume_instance = self._zone_plume_instance(query.zone_cells, per_plume)
            is_toxin_tp = plume_instance is not None
            if is_toxin_tp:
                hazard_type = "toxin"
                true_positive = True
                instance_id = plume_instance
            else:
                hazard_type = "disease"
                instance_id = self._zone_outbreak_instance(query.zone_cells)
                true_positive = instance_id is not None
            dosed_agents = (
                self._count_dosed_query_agents(query, concentrations, affected_count)
                if is_toxin_tp
                else None
            )
            if is_toxin_tp:
                attributed = provenance_support["toxin"]
            elif instance_id is not None:
                attributed = provenance_support["disease"]
            else:
                attributed = False
            event = DetectionEvent(
                step=self.current_step,
                hazard_type=hazard_type,
                anomaly_type=query.anomaly_type,
                zone_id=query.zone_cells[0] if query.zone_cells else -1,
                true_positive=true_positive,
                agents_affected=affected_count,
                dosed_agents=dosed_agents,
                hazard_instance_id=instance_id,
                attributed=attributed,
                causes=cause_support,
                benign_instance_id=benign_instance_id,
                benign_cause=benign_cause,
                benign_causes=benign_causes,
                benign_attributed=benign_attributed,
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

        for resp in self._collect_zone_responses(query):
            self.attack_orchestrator.deanon.collect_response(resp)

        self.attack_orchestrator.evaluate_deanonymization(
            float(self.agent_x[target_idx]),
            float(self.agent_y[target_idx]),
            success_threshold=self.config.attacks.deanon_success_threshold_m,
        )

    def _zone_plume_instance(
        self,
        zone_cells: list[int],
        per_plume: dict[str, np.ndarray],
    ) -> str | None:
        """Return an evaluation toxin instance above the configured dose gate."""
        threshold = self.config.toxin_exposure_concentration_gate()
        for plume_id, plume_field in per_plume.items():
            for cell_id in zone_cells:
                for agent_idx in self.grid.agents_in_cell(cell_id):
                    if plume_field[agent_idx] > threshold:
                        return plume_id
        return None

    def _zone_outbreak_instance(self, zone_cells: list[int]) -> str | None:
        """Return the dominant outbreak_id for diseased agents in the query zone."""
        counts: dict[str, int] = {}
        untagged = 0
        for cell_id in zone_cells:
            for agent_idx in self.grid.agents_in_cell(cell_id):
                if self.seir.states[agent_idx] in (
                    SEIRState.EXPOSED,
                    SEIRState.INFECTIOUS,
                ):
                    oid = str(self.seir.outbreak_origin[agent_idx])
                    if oid:
                        counts[oid] = counts.get(oid, 0) + 1
                    else:
                        untagged += 1
        if counts:
            return max(counts, key=lambda k: counts[k])
        if untagged > 0:
            return "outbreak_0"
        return None

    def _zone_benign_instances(self, zone_cells: list[int]) -> list[BenignInstance]:
        """Return active benign instances overlapping the query zone."""
        candidates: list[tuple[int, str, BenignInstance]] = []
        zone_set = set(zone_cells)
        for instance_id, instance in sorted(self._confounder_step.benign_instances.items()):
            if instance.global_scope:
                count = int(
                    np.count_nonzero(
                        self.has_wearable & np.isin(self.agent_cell_ids, list(zone_set))
                    )
                )
            else:
                agent_indices = np.fromiter(instance.current_agents, dtype=np.intp)
                count = int(
                    np.count_nonzero(np.isin(self.agent_cell_ids[agent_indices], list(zone_set)))
                )
            if count:
                candidates.append((count, instance_id, instance))
        return [item[2] for item in sorted(candidates, key=lambda item: (-item[0], item[1]))]

    def _zone_benign_instance(self, zone_cells: list[int]) -> BenignInstance | None:
        """Return the dominant active benign instance in the query zone."""
        instances = self._zone_benign_instances(zone_cells)
        return instances[0] if instances else None

    def _zone_has_plume_exposure(
        self,
        zone_cells: list[int],
        concentrations: np.ndarray,
    ) -> bool:
        """Return whether any agent exceeds the configured evaluation dose gate."""
        threshold = self.config.toxin_exposure_concentration_gate()
        for cell_id in zone_cells:
            for agent_idx in self.grid.agents_in_cell(cell_id):
                if concentrations[agent_idx] > threshold:
                    return True
        return False

    def _count_dosed_query_agents(
        self,
        query: BroadcastQuery,
        concentrations: np.ndarray,
        affected_count: int,
    ) -> int:
        """Count dosed matching agents for evaluation-only toxin metrics."""
        gate = self.config.toxin_exposure_concentration_gate()
        dosed = 0
        for cell_id in query.zone_cells:
            for agent in self.wearable_agents_by_cell.get(cell_id, ()):
                if (
                    agent.is_operational
                    and agent.anomaly_active
                    and agent.anomaly_type == query.anomaly_type
                    and concentrations[agent.idx] > gate
                ):
                    dosed += 1
        return min(dosed, affected_count)

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
        if self.config.disambiguation.enabled:
            unanswered, unresolved = self.aggregator.expire_disambiguation(
                self.current_step + max(self.config.disambiguation.expiry_steps, 0)
            )
            self.metrics.disambiguation_unanswered_expired += unanswered
            self.metrics.disambiguation_unresolved_hypotheses += unresolved
        self.metrics.finalize_hazard_episodes()
        return self.metrics
