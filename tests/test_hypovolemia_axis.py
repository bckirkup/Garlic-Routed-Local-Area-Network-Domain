"""Tests for the shared volume-depletion axis.

``hypovolemia`` is the first axis with no device of its own: fever, enteric fluid
loss, exertional sweat loss and heat strain all converge on it, and it then moves
four channels that already have inflammatory drivers. So the properties worth
asserting are not "does a channel move" but "do unrelated causes move the same
channels the same way" (one hidden state, several causes) and "does the
inflammatory arm survive being opposed" (PEP and PWV are blunted, not reversed).
"""

from __future__ import annotations

import numpy as np
import pytest

from garland.channels import (
    BLADDER_FILLING_IMPEDANCE_SHIFT,
    EIT_PERFUSION_PULSATILITY_RATIO,
    PEP_MS,
    PULSE_WAVE_VELOCITY,
    ChannelSet,
)
from garland.confounders import ConfounderEngine, ConfoundersConfig
from garland.devices import (
    ABDOMINAL_ACOUSTIC_BAND,
    BASE_DEVICE_KIND,
    THORACIC_EIT_ACOUSTIC_BAND,
    build_channel_set,
)
from garland.modality_signatures import (
    BLADDER_SHIFT_PER_HYPOVOLEMIA,
    PEP_MS_PER_HYPOVOLEMIA,
    PEP_MS_PER_INFLAMMATION,
    PULSATILITY_PER_HYPOVOLEMIA,
    PWV_PER_HYPOVOLEMIA,
    PWV_PER_STIFFENING,
    IllnessAxes,
    contact_artifact_axes,
    exertion_axes,
    heat_strain_axes,
    incubation_axes,
    infection_axes,
    irritant_axes,
    modality_delta,
    sleep_disruption_axes,
)
from garland.perturbations import PerturbationCause
from tests import modality_harness as harness

# Both impedance bands, so every channel the axis touches is in one vector.
VOLUME_SET: ChannelSet = build_channel_set(
    (BASE_DEVICE_KIND, THORACIC_EIT_ACOUSTIC_BAND, ABDOMINAL_ACOUSTIC_BAND)
)

PEP = PEP_MS.name
PWV = PULSE_WAVE_VELOCITY.name
PULSATILITY = EIT_PERFUSION_PULSATILITY_RATIO.name
BLADDER = BLADDER_FILLING_IMPEDANCE_SHIFT.name

VOLUME_CHANNELS = (PEP, PWV, PULSATILITY, BLADDER)
# The channels depletion moves *downward*, i.e. the ones with no other cause
# pushing them in this direction.
FALLING_CHANNELS = (PWV, PULSATILITY, BLADDER)

SEVERITIES = (0.0, 0.25, 0.5, 1.0)


def value(delta: np.ndarray, name: str, channel_set: ChannelSet = VOLUME_SET) -> float:
    return harness.channel_value(delta, channel_set, name)


def depletion_delta(level: float) -> np.ndarray:
    return modality_delta(IllnessAxes(hypovolemia=level), VOLUME_SET)


class TestAxisResponse:
    def test_depletion_moves_four_channels_in_calibrated_directions(self) -> None:
        delta = depletion_delta(1.0)
        # Preload loss lengthens the pre-ejection period; everything else falls.
        assert value(delta, PEP) == pytest.approx(PEP_MS_PER_HYPOVOLEMIA)
        assert value(delta, PWV) == pytest.approx(PWV_PER_HYPOVOLEMIA)
        assert value(delta, PULSATILITY) == pytest.approx(PULSATILITY_PER_HYPOVOLEMIA)
        assert value(delta, BLADDER) == pytest.approx(BLADDER_SHIFT_PER_HYPOVOLEMIA)
        for name in FALLING_CHANNELS:
            assert value(delta, name) < 0.0

    def test_response_is_graded_and_monotone_in_severity(self) -> None:
        deltas = [depletion_delta(level) for level in SEVERITIES]
        pep = [value(delta, PEP) for delta in deltas]
        assert pep == sorted(pep)
        assert pep[0] == pytest.approx(0.0)
        assert pep[-1] > pep[1] > 0.0
        for name in FALLING_CHANNELS:
            falling = [value(delta, name) for delta in deltas]
            assert falling == sorted(falling, reverse=True)
            assert falling[-1] < falling[1] < 0.0

    def test_axis_is_unsigned_and_bounded(self) -> None:
        # Fluid *overload* has its own axis, so this one has no negative arm.
        with pytest.raises(ValueError, match="hypovolemia"):
            IllnessAxes(hypovolemia=-0.5)
        with pytest.raises(ValueError, match="hypovolemia"):
            IllnessAxes(hypovolemia=1.5)

    def test_depletion_leaves_unrelated_channels_alone(self) -> None:
        delta = depletion_delta(1.0)
        untouched = [
            name
            for name in VOLUME_SET.names
            if name not in VOLUME_CHANNELS and delta[VOLUME_SET.index(name)] != 0.0
        ]
        assert not untouched
        assert np.all(np.isfinite(delta))


class TestSharedCauses:
    def test_every_cause_of_depletion_moves_the_same_channels_the_same_way(self) -> None:
        causes = {
            "respiratory_infection": infection_axes(1.0),
            "gastroenteritis": infection_axes(1.0, enteric_involvement=1.0),
            "exercise": exertion_axes(1.0),
            "heat": heat_strain_axes(1.0),
            "prodrome": incubation_axes(1.0),
        }
        for label, axes in causes.items():
            assert axes.hypovolemia > 0.0, label
            delta = modality_delta(axes, VOLUME_SET)
            # The two channels no other axis in these states touches: their sign
            # is the shared state showing through, whatever the cause.
            assert value(delta, PULSATILITY) < 0.0, label
            assert value(delta, BLADDER) < 0.0, label

    def test_enteric_tropism_grades_depletion(self) -> None:
        depletion = [
            infection_axes(1.0, enteric_involvement=enteric).hypovolemia
            for enteric in (0.0, 0.5, 1.0)
        ]
        assert depletion == sorted(depletion)
        assert depletion[0] < depletion[-1]
        assert depletion[-1] <= 1.0

    def test_depletion_ramps_with_illness_progress(self) -> None:
        ramped = [infection_axes(progress).hypovolemia for progress in SEVERITIES]
        assert ramped == sorted(ramped)
        assert ramped[0] == pytest.approx(0.0)

    def test_exercise_depletes_as_much_as_a_respiratory_fever(self) -> None:
        # Sweat loss is not a smaller effect than febrile insensible loss, so
        # the bladder and perfusion channels cannot separate them.
        assert exertion_axes(1.0).hypovolemia >= infection_axes(1.0).hypovolemia


class TestInflammatoryArmSurvives:
    def test_depletion_blunts_the_febrile_pep_shortening_without_reversing_it(self) -> None:
        febrile = infection_axes(1.0)
        combined = value(modality_delta(febrile, VOLUME_SET), PEP)
        pure_inflammatory = PEP_MS_PER_INFLAMMATION * febrile.inflammatory_drive
        assert pure_inflammatory < combined < 0.0
        assert abs(combined) < abs(pure_inflammatory)

    def test_depletion_blunts_febrile_stiffening_without_reversing_it(self) -> None:
        febrile = infection_axes(1.0)
        combined = value(modality_delta(febrile, VOLUME_SET), PWV)
        pure_stiffening = PWV_PER_STIFFENING * febrile.arterial_stiffening
        assert 0.0 < combined < pure_stiffening

    def test_heat_and_infection_disagree_on_the_vascular_channels(self) -> None:
        # A heat wave looks fever-shaped to the core vitals (hotter, faster), so
        # the band's contribution is that its vascular signs point the other way.
        heat = heat_strain_axes(1.0)
        assert heat.inflammatory_drive == pytest.approx(0.0)
        heat_delta = modality_delta(heat, VOLUME_SET)
        febrile_delta = modality_delta(infection_axes(1.0), VOLUME_SET)
        assert value(heat_delta, PWV) < 0.0 < value(febrile_delta, PWV)
        assert value(heat_delta, PEP) > 0.0 > value(febrile_delta, PEP)


class TestNegativeControls:
    def test_causes_without_fluid_loss_leave_the_axis_at_rest(self) -> None:
        for label, axes in {
            "irritant": irritant_axes(1.0),
            "sensor_artifact": contact_artifact_axes(1.0),
            "disrupted_night": sleep_disruption_axes(1.0),
        }.items():
            assert axes.hypovolemia == pytest.approx(0.0), label

    def test_sensor_artifact_cannot_fake_depletion(self) -> None:
        delta = modality_delta(contact_artifact_axes(1.0), VOLUME_SET)
        for name in VOLUME_CHANNELS:
            assert value(delta, name) == pytest.approx(0.0)

    def test_core_vitals_are_untouched_by_the_new_axis(self) -> None:
        harness.assert_core_vitals_unchanged(VOLUME_SET)


class TestConfounderIntegration:
    def heat_wave_delta(self) -> np.ndarray:
        engine = ConfounderEngine(
            harness.CONFOUNDER_AGENTS,
            ConfoundersConfig(
                enabled=True,
                heat_wave_start_step=0,
                heat_wave_duration_steps=48,
                # Everyone outdoors and no air conditioning, so the exposure
                # weights are large enough for the assertions to be about
                # physiology rather than about who happened to be indoors.
                has_air_conditioning_fraction=0.0,
                outdoor_worker_fraction=1.0,
            ),
            np.random.default_rng(11),
            channel_set=VOLUME_SET,
        )
        step = engine.step(
            current_step=0,
            hour_of_day=15.0,
            wearable_mask=np.ones(harness.CONFOUNDER_AGENTS, dtype=bool),
            transition_indices=set(),
        )
        deltas = [
            contribution.delta
            for contributions in step.contributions.values()
            for contribution in contributions
            if contribution.cause is PerturbationCause.HEAT_WAVE
        ]
        assert deltas
        return deltas[0]

    def test_a_heat_wave_now_reaches_the_impedance_channels(self) -> None:
        delta = self.heat_wave_delta()
        assert np.all(np.isfinite(delta))
        for name in FALLING_CHANNELS:
            assert value(delta, name) < 0.0
        assert value(delta, PEP) > 0.0

    def test_heat_wave_core_vital_arm_still_looks_febrile(self) -> None:
        # The point of the axis is a disagreement, which requires the core
        # vitals to keep pointing at illness.
        delta = self.heat_wave_delta()
        assert value(delta, "body_temperature") > 0.0
        assert value(delta, "heart_rate") > 0.0


class TestModelIntegration:
    @pytest.mark.parametrize("backend", ["rect", "hex"])
    def test_a_fleet_wearing_the_bands_runs_on_both_backends(self, backend: str) -> None:
        model = harness.modality_model(THORACIC_EIT_ACOUSTIC_BAND, backend=backend, seed=83)
        harness.assert_model_runs(model, PEP)

    def test_synthesised_values_stay_physical_while_a_fever_depletes(self) -> None:
        model = harness.modality_model(THORACIC_EIT_ACOUSTIC_BAND, seed=83)
        harness.step_model(model, 24)
        columns = {name: model.channel_set.index(name) for name in (PEP, PWV, PULSATILITY)}
        for agent in model.citizen_agents:
            observation = agent.last_observation
            assert np.all(np.isfinite(observation))
            # An interval, a velocity and a ratio of impedance amplitudes:
            # depletion pushes two of them toward their floors and none through.
            for column in columns.values():
                assert observation[column] > 0.0
