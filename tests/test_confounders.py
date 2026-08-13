"""Tests for opt-in ordinary-life confounder processes."""

from __future__ import annotations

import numpy as np

from garland.confounders import (
    BackgroundILIConfig,
    ConfounderEngine,
    CookingIrritantConfig,
)
from garland.perturbations import PerturbationCause


def _engine(
    *,
    ili: BackgroundILIConfig | None = None,
    cooking: CookingIrritantConfig | None = None,
    n_agents: int = 100,
) -> ConfounderEngine:
    household_ids = np.repeat(np.arange(n_agents // 2), 2).astype(np.int64)
    return ConfounderEngine(
        n_agents,
        household_ids,
        np.random.default_rng(42),
        ili or BackgroundILIConfig(),
        cooking or CookingIrritantConfig(),
    )


def test_disabled_confounders_do_not_consume_randomness():
    household_ids = np.repeat(np.arange(50), 2).astype(np.int64)
    first = np.random.default_rng(42)
    second = np.random.default_rng(42)
    ConfounderEngine(
        100,
        household_ids,
        first,
        BackgroundILIConfig(),
        CookingIrritantConfig(),
    )
    assert first.normal() == second.normal()


def test_background_ili_is_persistent_and_household_clustered():
    engine = _engine(
        ili=BackgroundILIConfig(
            enabled=True,
            onset_probability_per_step=1.0,
            duration_steps=3,
            household_secondary_multiplier=4.0,
        )
    )
    engine.step(12.0, 15)
    assert np.all(engine.ili_active)
    first = engine.contributions_for_agent(0, 0.0, 0.0, 0.0, 0.0)
    assert first[0].cause is PerturbationCause.BACKGROUND_ILI
    engine.background_ili.onset_probability_per_step = 0.0
    engine.step(12.0, 15)
    assert np.all(engine.ili_active)
    engine.step(12.0, 15)
    assert np.all(engine.ili_active)
    engine.step(12.0, 15)
    assert not np.any(engine.ili_active)


def test_ili_prevalence_parameter_grades_active_episode_count():
    low = _engine(
        ili=BackgroundILIConfig(enabled=True, onset_probability_per_step=0.0)
    )
    high = _engine(
        ili=BackgroundILIConfig(enabled=True, onset_probability_per_step=1.0)
    )
    low.step(12.0, 15)
    high.step(12.0, 15)
    assert np.sum(high.ili_active) > np.sum(low.ili_active)


def test_cooking_frequency_grades_household_event_count():
    low = _engine(
        cooking=CookingIrritantConfig(
            enabled=True,
            events_per_household_day=0.0,
        )
    )
    high = _engine(
        cooking=CookingIrritantConfig(
            enabled=True,
            events_per_household_day=36.0,
            frequency_log_sigma=0.0,
        )
    )
    low.step(18.0, 15)
    high.step(18.0, 15)
    assert np.sum(high.cooking_remaining > 0) >= np.sum(low.cooking_remaining)


def test_cooking_perturbation_has_no_temperature_component():
    engine = _engine(
        cooking=CookingIrritantConfig(
            enabled=True,
            events_per_household_day=36.0,
            frequency_log_sigma=0.0,
        )
    )
    engine.step(18.0, 15)
    contributions = engine.contributions_for_agent(0, 0.0, 0.0, 0.0, 0.0)
    irritant = next(
        contribution
        for contribution in contributions
        if contribution.cause is PerturbationCause.IRRITANT_EXPOSURE
    )
    assert irritant.delta[3] == 0.0


def test_cooking_susceptibility_spread_reaches_reactive_tail():
    narrow = _engine(
        cooking=CookingIrritantConfig(
            enabled=True,
            susceptibility_log_sigma=0.05,
        ),
        n_agents=1000,
    )
    wide = _engine(
        cooking=CookingIrritantConfig(
            enabled=True,
            susceptibility_log_sigma=1.15,
        ),
        n_agents=1000,
    )
    assert np.std(wide.susceptibility) > np.std(narrow.susceptibility)
    assert np.max(wide.susceptibility) / np.min(wide.susceptibility) > 10.0
