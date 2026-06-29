"""Tests for the bundled pathogen library and SEIR preset loading."""

from __future__ import annotations

import pytest

from garland.config import config_from_dict, load_config_file
from garland.hazards import SEIRConfig
from garland.pathogens import (
    compartment_rates_from_days,
    estimate_r0,
    get_pathogen,
    list_pathogen_ids,
    load_pathogen_library,
    seir_config_from_pathogen,
)


class TestPathogenLibrary:
    def test_library_loads(self):
        library = load_pathogen_library()
        assert library.schema_version == 1
        assert library.steps_per_day == 288
        assert "covid19_wildtype" in library.pathogens

    def test_list_pathogen_ids_sorted(self):
        ids = list_pathogen_ids()
        assert ids == sorted(ids)
        assert "influenza_seasonal" in ids

    def test_unknown_pathogen_raises(self):
        with pytest.raises(ValueError, match="Unknown pathogen"):
            get_pathogen("not_a_real_pathogen")

    def test_profile_metadata(self):
        profile = get_pathogen("measles")
        assert profile.pathogen_family == "paramyxovirus"
        assert profile.epidemiology.r0 == pytest.approx(15.0)
        assert profile.seir["beta"] > get_pathogen("influenza_seasonal").seir["beta"]


class TestPathogenRates:
    def test_compartment_rates_from_days(self):
        sigma, gamma = compartment_rates_from_days(5.0, 10.0)
        assert sigma == pytest.approx(1.0 / (5.0 * 288))
        assert gamma == pytest.approx(1.0 / (10.0 * 288))

    def test_covid_wildtype_r0_within_tolerance(self):
        profile = get_pathogen("covid19_wildtype")
        beta = float(profile.seir["beta"])
        gamma = float(profile.seir["gamma"])
        r0 = estimate_r0(beta, gamma)
        assert r0 == pytest.approx(profile.epidemiology.r0, rel=0.05)

    def test_omicron_more_transmissive_than_influenza(self):
        omicron = float(get_pathogen("covid19_omicron").seir["beta"])
        flu = float(get_pathogen("influenza_seasonal").seir["beta"])
        assert omicron > flu


class TestPathogenConfigLoading:
    def test_config_from_dict_applies_pathogen_defaults(self):
        config = config_from_dict({"seir": {"pathogen": "influenza_seasonal"}})
        flu = get_pathogen("influenza_seasonal")
        assert config.seir.beta == flu.seir["beta"]
        assert config.seir.sigma == flu.seir["sigma"]
        assert config.seir.gamma == flu.seir["gamma"]
        assert config.seir.initial_infected == flu.default_outbreak.initial_infected

    def test_explicit_override_wins(self):
        config = config_from_dict(
            {"seir": {"pathogen": "influenza_seasonal", "initial_infected": 99, "beta": 0.02}}
        )
        assert config.seir.initial_infected == 99
        assert config.seir.beta == pytest.approx(0.02)

    def test_seir_config_from_pathogen_helper(self):
        cfg = seir_config_from_pathogen("rsv", {"initial_infected": 4})
        assert isinstance(cfg, SEIRConfig)
        assert cfg.initial_infected == 4
        assert cfg.beta == get_pathogen("rsv").seir["beta"]

    def test_example_pathogen_yaml(self):
        config = load_config_file("examples/pathogen_influenza.yaml")
        flu = get_pathogen("influenza_seasonal")
        assert config.seir.beta == flu.seir["beta"]

    def test_wildtype_matches_seir_defaults(self):
        config = config_from_dict({"seir": {"pathogen": "covid19_wildtype"}})
        defaults = SEIRConfig()
        assert config.seir.beta == defaults.beta
        assert config.seir.sigma == defaults.sigma
        assert config.seir.gamma == defaults.gamma
