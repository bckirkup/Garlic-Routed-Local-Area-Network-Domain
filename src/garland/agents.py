"""Agent definitions for the GARLAND epidemiological security testbed.

Defines:
- CitizenAgent: The edge device (wearable BAN) with biometric monitoring
- NetworkAggregator: Secure threshold aggregation node
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from garland.biometric_synthesis import SynthesisBackend, generate_observation
from garland.biometrics import BaselineTracker, BiometricProfile
from garland.detection import SequentialDetector
from garland.device_lifecycle import DeviceStatus
from garland.disambiguation import DisambiguationHypothesis
from garland.perturbations import PerturbationContribution
from garland.privacy import (
    AggregatorState,
    AnomalyType,
    BroadcastQuery,
    DisambiguationQuery,
    EncryptedToken,
    PerturbedResponse,
    PrivacyConfig,
    classify_anomaly,
    noised_count_with_floor,
    planar_laplace_noise,
    randomized_response,
)

# Anomaly detection threshold (Mahalanobis distance)
ANOMALY_THRESHOLD = 3.5


@dataclass
class CitizenAgent:
    """Edge device agent representing a wearable body area network.

    Only agents with `has_wearable=True` execute the full biometric pipeline.
    Others participate in spatial dynamics only (for SEIR transmission).

    Parameters
    ----------
    idx : int
        Unique agent index.
    has_wearable : bool
        Whether this agent has a wearable device.
    profile : BiometricProfile | None
        Biometric resting profile (None if no wearable).
    household_id : int
        Household cluster (wearable penetration is household-patchy).
    neighborhood_id : int
        Neighborhood cluster for population layout (not used by privacy protocol).
    """

    idx: int
    has_wearable: bool = False
    profile: BiometricProfile | None = None
    household_id: int = 0
    neighborhood_id: int = 0
    cell_id: int = 0

    # State
    baseline: BaselineTracker = field(default_factory=BaselineTracker)
    anomaly_threshold: float = ANOMALY_THRESHOLD
    detector_mode: str = "instant"
    sequential_detector: SequentialDetector | None = None
    baseline_warmup_remaining: int = 0
    anomaly_active: bool = False
    anomaly_type: AnomalyType | None = None
    last_observation: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    queries_answered: int = 0
    local_epsilon: float = 0.0
    device_status: DeviceStatus = DeviceStatus.ACTIVE
    battery_level: float = 1.0
    adoption_step: int | None = 0
    steps_since_adoption: int | None = 0
    fleet_start_adopter: bool = True

    def __post_init__(self) -> None:
        """Initialize sequential state when that detector mode is selected."""
        if self.detector_mode == "sequential" and self.sequential_detector is None:
            self.sequential_detector = SequentialDetector()

    @property
    def is_operational(self) -> bool:
        """True when the device is worn, powered on, and has charge."""
        return self.has_wearable and self.device_status == DeviceStatus.ACTIVE

    def is_onboarding(self, window_steps: int) -> bool:
        """Whether this device is within its explicit onboarding window."""
        return (
            not self.fleet_start_adopter
            and self.steps_since_adoption is not None
            and 0 <= self.steps_since_adoption < window_steps
        )

    def observe_and_detect(
        self,
        hour: int,
        month: int,
        day_of_year: int,
        hour_of_day: float,
        rng: np.random.Generator,
        cell_id: int,
        hazard_perturbation: NDArray[np.float64] | None = None,
        perturbations: Sequence[PerturbationContribution] | None = None,
        activity_level: float = 0.0,
        synthesis_backend: SynthesisBackend = "custom",
        neurokit_window_seconds: float = 60.0,
        suppress_token_emission: bool = False,
    ) -> EncryptedToken | None:
        """Generate biometric observation, update baseline, detect anomalies.

        Returns an encrypted token if anomaly detected, else None. When
        ``suppress_token_emission`` is True (baseline warm-up), baselines still
        adapt but no tokens are emitted and anomaly state is not latched.
        ``hazard_perturbation`` is the unlabelled legacy perturbation path.
        """
        if hazard_perturbation is not None and perturbations is not None:
            raise ValueError(
                "hazard_perturbation and perturbations cannot both be provided"
            )
        if not self.is_operational or self.profile is None:
            return None

        # Generate observation with any hazard effects
        obs = generate_observation(
            self.profile,
            hour_of_day,
            day_of_year,
            rng,
            activity_level,
            backend=synthesis_backend,
            neurokit_window_seconds=neurokit_window_seconds,
        )
        if perturbations:
            total_perturbation = np.zeros(4, dtype=np.float64)
            for contribution in perturbations:
                total_perturbation += contribution.delta
            obs += total_perturbation
        if hazard_perturbation is not None:
            obs += hazard_perturbation

        self.last_observation = obs

        sequential_residual: NDArray[np.float64] | None = None
        if self.detector_mode == "sequential":
            expected_baseline = self.baseline.expected_baseline(hour, month)
            sequential_residual = obs - expected_baseline
        # Compute anomaly score
        maha_dist = self.baseline.mahalanobis_distance(obs, hour, month)

        # Update baseline (adaptive forgetting)
        self.baseline.update(obs, hour, month)

        sequential_warmup = (
            self.detector_mode == "sequential" and self.baseline.n_samples < 5
        )
        if suppress_token_emission or sequential_warmup:
            self.anomaly_active = False
            self.anomaly_type = None
            if self.sequential_detector is not None:
                self.sequential_detector.reset()
            return None

        if self.detector_mode == "sequential":
            if self.sequential_detector is None or sequential_residual is None:
                raise RuntimeError("Sequential detector state is not initialized")
            self.sequential_detector.update(maha_dist, sequential_residual)
            self.anomaly_active = self.sequential_detector.alarm_active
            if not self.anomaly_active:
                self.anomaly_type = None
                return None
            atype = classify_anomaly(
                self.sequential_detector.residual_ewma,
                np.zeros(4, dtype=np.float64),
            )
            self.anomaly_type = atype
            if atype is not None:
                return EncryptedToken(
                    zone_id=cell_id,
                    anomaly_type=atype,
                    timestamp_bin=0,
                    agent_id_hash=hash(self.idx) & 0x7FFFFFFF,
                )
            return None

        # Check anomaly predicate
        if maha_dist > self.anomaly_threshold:
            baseline_expected = self.baseline.expected_baseline(hour, month)
            atype = classify_anomaly(obs, baseline_expected)
            if atype is not None:
                self.anomaly_active = True
                self.anomaly_type = atype
                # Generate blind-gated encrypted token
                return EncryptedToken(
                    zone_id=cell_id,
                    anomaly_type=atype,
                    timestamp_bin=0,  # Set by caller
                    agent_id_hash=hash(self.idx) & 0x7FFFFFFF,
                )
        else:
            self.anomaly_active = False
            self.anomaly_type = None

        return None

    def respond_to_query(
        self,
        query: BroadcastQuery,
        true_x: float,
        true_y: float,
        cell_id: int,
        config: PrivacyConfig,
        rng: np.random.Generator,
    ) -> PerturbedResponse | None:
        """Evaluate and respond to a reverse-query broadcast.

        Applies:
        1. Randomized Response (coin-flip DP)
        2. Planar Laplace noise for geo-indistinguishability
        """
        if not self.is_operational:
            return None

        # Check if agent matches the query criteria
        matches = (
            self.anomaly_active
            and self.anomaly_type == query.anomaly_type
            and cell_id in query.zone_cells
        )

        # Randomized response
        reported_match = randomized_response(matches, config.randomized_response_p, rng)

        if not reported_match:
            # Non-matching: optionally emit dummy packet
            if rng.random() < config.dummy_rate:
                dx, dy = planar_laplace_noise(config.laplace_scale * 2, rng)
                return PerturbedResponse(
                    query_id=query.query_id,
                    reported_x=true_x + dx,
                    reported_y=true_y + dy,
                    anomaly_confirmed=False,
                    is_dummy=True,
                )
            return None

        # Apply Planar Laplace noise to location
        dx, dy = planar_laplace_noise(config.laplace_scale, rng)
        perturbed_x = true_x + dx
        perturbed_y = true_y + dy

        # Track privacy budget
        self.queries_answered += 1
        self.local_epsilon += config.epsilon_per_response

        return PerturbedResponse(
            query_id=query.query_id,
            reported_x=perturbed_x,
            reported_y=perturbed_y,
            anomaly_confirmed=True,
            is_dummy=False,
        )

    def generate_dummy_traffic(
        self,
        _true_x: float,
        _true_y: float,
        cell_id: int,
        config: PrivacyConfig,
        rng: np.random.Generator,
        suppress_token_emission: bool = False,
    ) -> EncryptedToken | None:
        """Periodically emit dummy noise packets for traffic obfuscation."""
        if suppress_token_emission:
            return None
        if not self.is_operational:
            return None
        if rng.random() < config.dummy_rate:
            _anomaly_types = list(AnomalyType)
            return EncryptedToken(
                zone_id=cell_id,
                anomaly_type=_anomaly_types[int(rng.integers(0, len(_anomaly_types)))],
                timestamp_bin=0,
                agent_id_hash=int(rng.integers(0, 2**31)),
                is_dummy=True,
            )
        return None

    def respond_to_disambiguation(
        self,
        query: DisambiguationQuery,
        cell_id: int,
        answer_rate: float,
        yes_rate: float,
        config: PrivacyConfig,
        rng: np.random.Generator,
    ) -> bool | None:
        """Return a randomized human-approved yes/no answer, or silence."""
        if not self.is_operational or cell_id not in query.zone_cells:
            return None
        if rng.random() >= min(max(answer_rate, 0.0), 1.0):
            return None
        approved_answer = rng.random() < min(max(yes_rate, 0.0), 1.0)
        return randomized_response(approved_answer, config.randomized_response_p, rng)


@dataclass
class PendingDisambiguation:
    """State for a pending second-round hypothesis prompt."""

    expires_at_step: int
    unanswered_count: int
    answered: bool


@dataclass
class NetworkAggregator:
    """Secure threshold aggregation node.

    Cannot decrypt individual tokens. Performs:
    1. Homomorphic token counting per zone
    2. Threshold detection
    3. K-anonymity spatial dilution
    4. Broadcast query generation
    """

    config: PrivacyConfig = field(default_factory=PrivacyConfig)
    state: AggregatorState = field(default_factory=AggregatorState)
    broadcasts_issued: int = 0
    total_responses_received: int = 0
    disambiguation_queries_issued: int = 0
    pending_disambiguation: dict[int, PendingDisambiguation] = field(
        default_factory=dict
    )

    def ingest_tokens(self, tokens: list[EncryptedToken], time_bin: int) -> None:
        """Receive batch of encrypted tokens for aggregation."""
        for token in tokens:
            token_with_time = EncryptedToken(
                zone_id=token.zone_id,
                anomaly_type=token.anomaly_type,
                timestamp_bin=time_bin,
                agent_id_hash=token.agent_id_hash,
                is_dummy=token.is_dummy,
            )
            self.state.receive_token(token_with_time)

    def evaluate_and_broadcast(
        self,
        current_time_bin: int,
        spatial_dilate_fn,
    ) -> list[BroadcastQuery]:
        """Check thresholds and generate dilated broadcast queries."""
        triggers = self.state.check_thresholds(current_time_bin, self.config)
        queries = []

        for zone_id, anomaly_type in triggers:
            # Apply K-anonymity spatial dilution
            dilated_cells = spatial_dilate_fn(zone_id, self.config.k_min)

            query = BroadcastQuery(
                zone_cells=dilated_cells,
                anomaly_type=anomaly_type,
                time_window_start=current_time_bin - self.config.time_window_steps,
                time_window_end=current_time_bin,
                query_id=self.broadcasts_issued,
            )
            queries.append(query)
            self.broadcasts_issued += 1

        return queries

    def collect_responses(self, responses: list[PerturbedResponse]) -> None:
        """Collect perturbed responses from broadcast."""
        self.state.responses.extend(responses)
        self.total_responses_received += len(responses)
        # Record privacy budget via adaptive composition over genuine responses
        genuine = sum(1 for r in responses if r.anomaly_confirmed and not r.is_dummy)
        self.state.record_genuine_responses(genuine, self.config.epsilon_per_response)

    def issue_disambiguation_queries(
        self,
        queries: list[BroadcastQuery],
        hypothesis: DisambiguationHypothesis,
        should_ask: Callable[[BroadcastQuery], bool],
    ) -> list[DisambiguationQuery]:
        """Issue follow-up hypothesis queries for qualifying broadcasts."""
        issued: list[DisambiguationQuery] = []
        for query in queries:
            if not should_ask(query):
                continue
            disambiguation = DisambiguationQuery(
                zone_cells=query.zone_cells,
                hypothesis=hypothesis,
                referenced_query_ids=(query.query_id,),
                time_window_start=query.time_window_start,
                time_window_end=query.time_window_end,
                query_id=self.disambiguation_queries_issued,
            )
            self.disambiguation_queries_issued += 1
            issued.append(disambiguation)
        return issued

    def register_disambiguation_pending(
        self,
        query_id: int,
        expires_at_step: int,
        unanswered_count: int,
        answered: bool,
    ) -> None:
        """Track one issued prompt until it is answered or expires."""
        self.pending_disambiguation[query_id] = PendingDisambiguation(
            expires_at_step=expires_at_step,
            unanswered_count=unanswered_count,
            answered=answered,
        )

    def release_disambiguation_ack(
        self,
        ack_count: int,
        population: int,
        k_min: int,
        noise_scale: float,
        rng: np.random.Generator,
        ack_epsilon: float,
    ) -> int:
        """Release a bounded noisy zone acknowledgement count."""
        released = noised_count_with_floor(
            ack_count, population, k_min, noise_scale, rng
        )
        if population >= k_min:
            self.state.record_disambiguation_ack(ack_epsilon)
        return released

    def record_disambiguation_answers(
        self, count: int, epsilon_per_response: float
    ) -> None:
        """Charge approved disambiguation answers as genuine releases."""
        self.state.record_disambiguation_answers(count, epsilon_per_response)

    def expire_disambiguation(self, current_step: int) -> tuple[int, int]:
        """Expire unanswered prompts and return (unanswered, unresolved)."""
        unanswered = 0
        unresolved = 0
        for query_id, pending in list(self.pending_disambiguation.items()):
            if current_step < pending.expires_at_step:
                continue
            unanswered += pending.unanswered_count
            unresolved += int(not pending.answered)
            del self.pending_disambiguation[query_id]
        return unanswered, unresolved
