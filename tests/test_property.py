"""Property-based and fuzz tests for privacy primitives and config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from garland.config import apply_overrides, config_from_dict, config_to_dict, load_config_file
from garland.privacy import planar_laplace_noise, randomized_response
from garland.simulation import SimulationConfig

# Keep example counts low so CI stays fast while still exploring edge cases.
HYPOTHESIS_SETTINGS = settings(max_examples=30, deadline=None)

_POSITIVE_SCALE = st.floats(
    min_value=1e-2, max_value=5_000.0, allow_nan=False, allow_infinity=False
)
_TRUTH_PROB = st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False)


def _rng_from_draw(data: object) -> np.random.Generator:
    return np.random.default_rng(hash(data) % (2**32))


class TestPlanarLaplaceProperties:
    @given(scale=_POSITIVE_SCALE)
    @HYPOTHESIS_SETTINGS
    def test_outputs_are_finite(self, scale: float) -> None:
        rng = _rng_from_draw(scale)
        dx, dy = planar_laplace_noise(scale, rng)
        assert np.isfinite(dx)
        assert np.isfinite(dy)

    @given(scale=_POSITIVE_SCALE)
    @HYPOTHESIS_SETTINGS
    def test_zero_mean_over_many_samples(self, scale: float) -> None:
        rng = np.random.default_rng(42)
        samples = [planar_laplace_noise(scale, rng) for _ in range(2_000)]
        mean_x = float(np.mean([s[0] for s in samples]))
        mean_y = float(np.mean([s[1] for s in samples]))
        # Means should stay small relative to typical noise magnitude.
        assert abs(mean_x) < scale * 0.25
        assert abs(mean_y) < scale * 0.25

    @given(small_scale=_POSITIVE_SCALE, factor=st.floats(min_value=2.0, max_value=8.0))
    @HYPOTHESIS_SETTINGS
    def test_scale_monotonicity(self, small_scale: float, factor: float) -> None:
        large_scale = min(small_scale * factor, 5_000.0)
        if large_scale <= small_scale * 1.5:
            return

        rng = np.random.default_rng(99)
        small_distances = [
            np.hypot(*planar_laplace_noise(small_scale, rng)) for _ in range(400)
        ]
        large_distances = [
            np.hypot(*planar_laplace_noise(large_scale, rng)) for _ in range(400)
        ]
        assert float(np.mean(large_distances)) > float(np.mean(small_distances))


class TestRandomizedResponseProperties:
    @given(true_value=st.booleans(), p=_TRUTH_PROB)
    @HYPOTHESIS_SETTINGS
    def test_output_is_bool(self, true_value: bool, p: float) -> None:
        rng = _rng_from_draw((true_value, p))
        result = randomized_response(true_value, p, rng)
        assert isinstance(result, bool)

    @given(p=_TRUTH_PROB)
    @HYPOTHESIS_SETTINGS
    def test_truthful_rate_matches_mechanism(self, p: float) -> None:
        """First branch is truthful with prob p; otherwise fair coin (50% truthful)."""
        rng = np.random.default_rng(7)
        trials = 5_000
        true_value = True
        truthful = sum(
            randomized_response(true_value, p, rng) == true_value for _ in range(trials)
        )
        expected_rate = p + (1.0 - p) * 0.5
        observed_rate = truthful / trials
        assert abs(observed_rate - expected_rate) < 0.05

    @given(p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @HYPOTHESIS_SETTINGS
    def test_certainty_endpoints(self, p: float) -> None:
        rng = np.random.default_rng(11)
        if p >= 1.0:
            for _ in range(20):
                assert randomized_response(True, p, rng) is True
                assert randomized_response(False, p, rng) is False
        elif p <= 0.0:
            for _ in range(50):
                assert isinstance(randomized_response(True, p, rng), bool)


_PRIVACY_FIELDS = st.fixed_dictionaries(
    {
        "k_min": st.integers(min_value=1, max_value=500),
        "epsilon_per_response": st.floats(
            min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        "laplace_scale": st.floats(
            min_value=1.0, max_value=2_000.0, allow_nan=False, allow_infinity=False
        ),
        "randomized_response_p": st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    }
)

_TOP_LEVEL_FIELDS = st.fixed_dictionaries(
    {
        "n_agents": st.integers(min_value=10, max_value=10_000),
        "n_steps": st.integers(min_value=1, max_value=500),
        "seed": st.integers(min_value=0, max_value=10_000),
        "privacy": _PRIVACY_FIELDS,
    }
)


class TestConfigProperties:
    @given(fields=_TOP_LEVEL_FIELDS)
    @HYPOTHESIS_SETTINGS
    def test_config_round_trip_from_dict(self, fields: dict) -> None:
        original = config_from_dict(fields)
        restored = config_from_dict(config_to_dict(original))
        assert restored.n_agents == original.n_agents
        assert restored.n_steps == original.n_steps
        assert restored.seed == original.seed
        assert restored.privacy.k_min == original.privacy.k_min
        assert restored.privacy.epsilon_per_response == original.privacy.epsilon_per_response
        assert restored.privacy.laplace_scale == original.privacy.laplace_scale
        assert restored.privacy.randomized_response_p == original.privacy.randomized_response_p

    @given(
        base_k_min=st.integers(min_value=1, max_value=200),
        override_eps=st.floats(
            min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @HYPOTHESIS_SETTINGS
    def test_dotted_override_merge(self, base_k_min: int, override_eps: float) -> None:
        base = {"privacy": {"k_min": base_k_min, "epsilon_per_response": 0.1}}
        merged = apply_overrides(base, {"privacy.epsilon_per_response": override_eps})
        assert merged["privacy"]["k_min"] == base_k_min
        assert merged["privacy"]["epsilon_per_response"] == override_eps

    def test_reject_invalid_nested_override_path(self) -> None:
        base = {"privacy": 42}
        with pytest.raises(ValueError, match="not a mapping"):
            apply_overrides(base, {"privacy.k_min": 10})

    @given(
        privacy_value=st.one_of(
            st.text(min_size=1, max_size=8),
            st.integers().filter(lambda value: value != 0),
            st.lists(st.integers(), min_size=1, max_size=4),
        )
    )
    @HYPOTHESIS_SETTINGS
    def test_reject_invalid_nested_privacy_type(self, privacy_value: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            config_from_dict({"privacy": privacy_value})

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sim.yaml"
        path.write_text(
            "n_agents: 250\nn_steps: 12\nprivacy:\n  k_min: 8\n",
            encoding="utf-8",
        )
        config = load_config_file(path)
        assert config.n_agents == 250
        assert config.n_steps == 12
        assert config.privacy.k_min == 8

    def test_simulation_config_round_trip(self) -> None:
        original = SimulationConfig(n_agents=321, n_steps=9, seed=17)
        restored = config_from_dict(config_to_dict(original))
        assert restored.n_agents == 321
        assert restored.n_steps == 9
        assert restored.seed == 17
