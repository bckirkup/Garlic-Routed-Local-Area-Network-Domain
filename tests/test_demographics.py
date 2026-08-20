"""Age structure and its coupling to who owns which device."""

from __future__ import annotations

import numpy as np
import pytest

from garland.config import config_from_dict
from garland.demographics import (
    ADULT,
    AGE_BANDS,
    CHILD,
    ELDERLY,
    INFANT,
    OLDER_ADULT,
    DemographicsConfig,
    assign_age_bands,
    band_counts,
    draw_enthusiasm,
)
from garland.simulation import GarlandModel

ADOPTION = {
    "motion_actigraphy": 0.55,
    "instrumented_footwear": 0.20,
    "respiratory_acoustic_patch": 0.12,
    "chest_electrode_patch": 0.08,
    "headband_eeg": 0.06,
    "thoracic_eit_acoustic_band": 0.03,
    "abdominal_acoustic_band": 0.03,
}


def build_model(
    *,
    demographics: dict | None = None,
    n_agents: int = 900,
    seed: int = 42,
    wearable_fraction: float = 0.85,
) -> GarlandModel:
    """A short-run model with the full device catalogue adopted."""
    payload: dict = {
        "n_agents": n_agents,
        "n_steps": 4,
        "seed": seed,
        "wearable_fraction": wearable_fraction,
        "devices": {"enabled": True, "adoption": dict(ADOPTION)},
    }
    if demographics is not None:
        payload["demographics"] = demographics
    return GarlandModel(config_from_dict(payload))


def owner_share_per_capita(model: GarlandModel, band: str, kind: str) -> float:
    """Fraction of that band's wearers who own ``kind``."""
    composition = model.fleet_composition()
    wearers = composition["age_band_wearers"][band]
    if wearers == 0:
        return 0.0
    return composition["owner_counts_by_band"][band][kind] / wearers


class TestSensitivity:
    """Graded response: a few different values in -> a few different values out."""

    def test_enthusiasm_sigma_grades_ownership_spread(self) -> None:
        """A wider enthusiasm tail concentrates devices on fewer people."""
        sigmas = [0.0, 0.4, 0.8, 1.6]
        spreads = []
        tails = []
        bare = []
        for sigma in sigmas:
            model = build_model(demographics={"enabled": True, "enthusiasm_sigma": sigma})
            distribution = model.fleet_composition()
            spreads.append(distribution["sd_devices_per_owner"])
            tails.append(distribution["share_four_or_more_devices"])
            bare.append(distribution["share_base_device_only"])

        assert spreads == sorted(spreads), f"spread not monotone in sigma: {spreads}"
        assert tails == sorted(tails), f"tail not monotone in sigma: {tails}"
        assert bare == sorted(bare), f"core-device-only share not monotone: {bare}"
        # Live knob, measured as concentration rather than variance: the count
        # distribution is bounded above by the catalogue size, so the standard
        # deviation moves modestly while the tail and the bare-wristband share
        # move a lot. Those two are the fleet property that matters -- devices
        # piling onto a minority instead of spreading evenly.
        assert tails[-1] > 2.0 * tails[0], f"multi-device tail looks dead: {tails}"
        assert bare[-1] > 1.4 * bare[0], f"core-only share looks dead: {bare}"
        assert spreads[-1] > 1.15 * spreads[0], f"spread looks dead: {spreads}"

    def test_mean_devices_per_owner_is_unmoved_by_enthusiasm(self) -> None:
        """Correlation redistributes ownership; it does not create devices.

        This is the property that keeps every adoption fraction in the existing
        calibration meaningful: only the *shape* across people changes.
        """
        means = [
            build_model(
                demographics={"enabled": True, "enthusiasm_sigma": sigma}
            ).fleet_composition()["mean_devices_per_owner"]
            for sigma in (0.0, 0.8, 1.6)
        ]
        # Tolerance covers only the base-device retention top-up, which shifts
        # the wearer denominator by a few people.
        assert max(means) - min(means) < 0.05, f"means drifted with sigma: {means}"

    def test_age_affinity_orders_ownership_by_band(self) -> None:
        """Cardiac and gait hardware skews old; the sleep headband skews adult."""
        model = build_model(demographics={"enabled": True})

        for kind in ("chest_electrode_patch", "instrumented_footwear"):
            adult = owner_share_per_capita(model, ADULT, kind)
            older = owner_share_per_capita(model, OLDER_ADULT, kind)
            elderly = owner_share_per_capita(model, ELDERLY, kind)
            assert older > adult, f"{kind}: older adults {older} <= adults {adult}"
            assert elderly > adult, f"{kind}: elderly {elderly} <= adults {adult}"

        assert owner_share_per_capita(model, ADULT, "headband_eeg") > owner_share_per_capita(
            model, ELDERLY, "headband_eeg"
        )
        infant_patch = owner_share_per_capita(model, INFANT, "respiratory_acoustic_patch")
        adult_patch = owner_share_per_capita(model, ADULT, "respiratory_acoustic_patch")
        assert infant_patch > adult_patch, "caregiver-chosen respiratory patch should skew infant"

    def test_infant_base_retention_grades_infant_wearers(self) -> None:
        """Retention decides how many infants carry a core device at all."""
        counts = []
        for retention in (0.0, 0.5, 1.0):
            model = build_model(
                demographics={
                    "enabled": True,
                    "base_device_retention": {INFANT: retention},
                }
            )
            counts.append(model.fleet_composition()["age_band_wearers"][INFANT])
        assert counts == sorted(counts), f"infant wearers not monotone in retention: {counts}"
        assert counts[0] == 0, "zero retention must leave no infant wearing a core device"
        assert counts[-1] > counts[0] + 3, f"retention looks dead: {counts}"

    def test_household_type_mix_grades_the_age_pyramid(self) -> None:
        """Shifting households from families to seniors ages the population."""
        elderly_counts = []
        child_counts = []
        for senior_share in (0.1, 0.3, 0.6):
            model = build_model(
                demographics={
                    "enabled": True,
                    "household_type_fractions": {
                        "family": 0.9 - senior_share,
                        "adult_only": 0.1,
                        "senior": senior_share,
                    },
                }
            )
            population = model.fleet_composition()["age_band_population"]
            elderly_counts.append(population[ELDERLY] + population[OLDER_ADULT])
            child_counts.append(population[CHILD] + population[INFANT])
        assert elderly_counts == sorted(elderly_counts), f"seniors not graded: {elderly_counts}"
        assert child_counts == sorted(child_counts, reverse=True), (
            f"juveniles not inversely graded: {child_counts}"
        )
        assert elderly_counts[-1] > 2 * elderly_counts[0], (
            f"household mix looks dead: {elderly_counts}"
        )

    def test_unrelated_knob_leaves_composition_unchanged(self) -> None:
        """Negative control: a detector knob does not move who owns what."""
        base = build_model(demographics={"enabled": True}).fleet_composition()
        payload = {
            "n_agents": 900,
            "n_steps": 4,
            "seed": 42,
            "wearable_fraction": 0.85,
            "devices": {"enabled": True, "adoption": dict(ADOPTION)},
            "demographics": {"enabled": True},
            "anomaly_threshold": 5.0,
        }
        other = GarlandModel(config_from_dict(payload)).fleet_composition()
        assert other["owner_counts"] == base["owner_counts"]
        assert other["age_band_population"] == base["age_band_population"]


class TestInvariants:
    """Ranges, exclusions, conservation, and reproducibility."""

    def test_configured_adoption_fractions_survive_the_weighting(self) -> None:
        """Structured ownership moves *who* owns a kind, not *how many* do.

        Any drift here would silently recalibrate every adoption fraction in the
        committed configs, so this is the load-bearing invariant of the feature.
        """
        flat = build_model(demographics={"enabled": False}).fleet_composition()
        structured = build_model(demographics={"enabled": True}).fleet_composition()
        n_wearers = structured["owner_counts"]["wrist_ppg"]
        for kind, fraction in ADOPTION.items():
            expected = int(np.floor(fraction * n_wearers))
            assert structured["owner_counts"][kind] == expected, (
                f"{kind}: {structured['owner_counts'][kind]} owners, expected {expected}"
            )
            # And the flat fleet agrees on the same target, up to the handful of
            # wearers the retention top-up adds or removes.
            assert abs(structured["owner_counts"][kind] - flat["owner_counts"][kind]) <= max(
                2, int(0.05 * expected)
            )

    def test_zero_affinity_is_a_hard_exclusion(self) -> None:
        """No form factor exists for a band that cannot wear it, at any fraction.

        Swept up to an adoption fraction far above the non-infant share, which
        is where a naive top-up to the target owner count would leak.
        """
        for fraction in (0.2, 0.6, 0.95):
            adoption = dict(ADOPTION)
            adoption["instrumented_footwear"] = fraction
            adoption["headband_eeg"] = fraction
            payload = {
                "n_agents": 900,
                "n_steps": 4,
                "seed": 42,
                "wearable_fraction": 0.85,
                "devices": {"enabled": True, "adoption": adoption},
                "demographics": {"enabled": True},
            }
            by_band = GarlandModel(config_from_dict(payload)).fleet_composition()[
                "owner_counts_by_band"
            ]
            assert by_band[INFANT]["instrumented_footwear"] == 0
            assert by_band[INFANT]["headband_eeg"] == 0

    def test_wearable_fraction_still_holds_under_retention_thinning(self) -> None:
        """Thinning infants and elderly must not undershoot the wearer target."""
        for wearable_fraction in (0.15, 0.5, 0.85):
            model = build_model(
                demographics={
                    "enabled": True,
                    "base_device_retention": {INFANT: 0.0, ELDERLY: 0.5, CHILD: 0.6},
                },
                wearable_fraction=wearable_fraction,
            )
            wearers = int(np.count_nonzero(model.has_wearable))
            target = wearable_fraction * model.config.n_agents
            assert 0.9 * target <= wearers <= 1.1 * target, (
                f"fraction {wearable_fraction}: {wearers} wearers against target {target}"
            )

    def test_bands_partition_the_population_and_stay_in_range(self) -> None:
        """Every agent has exactly one band, and shares are probabilities."""
        model = build_model(demographics={"enabled": True})
        composition = model.fleet_composition()
        population = composition["age_band_population"]
        assert set(population) == set(AGE_BANDS)
        assert sum(population.values()) == model.config.n_agents
        assert all(count >= 0 for count in population.values())
        for band, wearers in composition["age_band_wearers"].items():
            assert 0 <= wearers <= population[band]
        for key in ("share_base_device_only", "share_four_or_more_devices"):
            assert 0.0 <= composition[key] <= 1.0
        assert 1.0 <= composition["mean_devices_per_owner"] <= len(model.device_fleet.kinds)

    def test_infants_and_children_always_live_with_an_adult(self) -> None:
        """Composition, not independent draws: a toddler is not a lone household."""
        rng = np.random.default_rng(7)
        household_ids = np.repeat(np.arange(300, dtype=np.int64), 3)
        bands = assign_age_bands(household_ids, DemographicsConfig(enabled=True), rng)
        adult_index = AGE_BANDS.index(ADULT)
        juvenile = {AGE_BANDS.index(INFANT), AGE_BANDS.index(CHILD)}
        for household in np.unique(household_ids):
            members = bands[household_ids == household]
            if juvenile.intersection(members.tolist()):
                assert adult_index in members.tolist(), (
                    f"household {household} has juveniles and no adult: {members}"
                )

    def test_disabled_demographics_leaves_one_adult_band(self) -> None:
        """Boundary input: the default fleet is demographically flat."""
        model = build_model(demographics=None)
        composition = model.fleet_composition()
        assert composition["age_band_population"][ADULT] == model.config.n_agents
        assert "owner_counts_by_band" not in composition
        assert composition["age_band_wearers"][INFANT] == 0

    def test_enthusiasm_is_positive_finite_and_mean_one(self) -> None:
        """The multiplier cannot silently scale the fleet's ownership rate."""
        rng = np.random.default_rng(3)
        values = draw_enthusiasm(20_000, DemographicsConfig(enabled=True), rng)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        assert values.mean() == pytest.approx(1.0, rel=0.05)

    def test_tiny_population_is_handled(self) -> None:
        """Boundary input: a hamlet of 40 people still builds a coherent fleet.

        Populations below a household or two are not supported by the model at
        large (outbreak seeding needs someone to seed), so the boundary tested
        here is the smallest population the rest of the pipeline accepts.
        """
        model = build_model(demographics={"enabled": True}, n_agents=40, wearable_fraction=1.0)
        composition = model.fleet_composition()
        assert sum(composition["age_band_population"].values()) == 40
        assert composition["mean_devices_per_owner"] >= 1.0

    def test_same_seed_reproduces_the_fleet(self) -> None:
        """Reproducibility: composition is a function of the seed, not of order."""
        first = build_model(demographics={"enabled": True}, seed=11).fleet_composition()
        second = build_model(demographics={"enabled": True}, seed=11).fleet_composition()
        assert first == second
        third = build_model(demographics={"enabled": True}, seed=12).fleet_composition()
        assert third["age_band_population"] != first["age_band_population"]

    def test_band_counts_matches_a_hand_counted_array(self) -> None:
        """Analytic check on the counting helper itself."""
        bands = np.array([0, 0, 2, 4, 2, 2], dtype=np.int8)
        assert band_counts(bands) == {
            INFANT: 2,
            CHILD: 0,
            ADULT: 3,
            OLDER_ADULT: 0,
            ELDERLY: 1,
        }


class TestConfigValidation:
    """Bad configuration fails loudly at load time, not mid-run."""

    def test_unknown_age_band_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown age band"):
            config_from_dict(
                {
                    "n_agents": 10,
                    "demographics": {"enabled": True, "base_device_retention": {"teen": 0.5}},
                }
            )

    def test_unknown_household_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown household type"):
            config_from_dict(
                {
                    "n_agents": 10,
                    "demographics": {"enabled": True, "household_type_fractions": {"commune": 1.0}},
                }
            )

    def test_out_of_range_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="retention"):
            DemographicsConfig(base_device_retention={ADULT: 1.5}).validate()
        with pytest.raises(ValueError, match="infant_share_of_children"):
            DemographicsConfig(infant_share_of_children=-0.1).validate()
        with pytest.raises(ValueError, match="enthusiasm_sigma"):
            DemographicsConfig(enthusiasm_sigma=-1.0).validate()
        with pytest.raises(ValueError, match="sum above zero"):
            DemographicsConfig(
                household_type_fractions={"family": 0.0, "adult_only": 0.0, "senior": 0.0}
            ).validate()

    def test_household_type_fractions_are_normalised(self) -> None:
        """Unnormalised weights are accepted and rescaled, not silently skewed."""
        config = DemographicsConfig(
            household_type_fractions={"family": 2.0, "adult_only": 1.0, "senior": 1.0}
        )
        resolved = config.resolved_household_type_fractions()
        assert sum(resolved.values()) == pytest.approx(1.0)
        assert resolved["family"] == pytest.approx(0.5)
