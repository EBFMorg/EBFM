# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np

from ebfm.core.config import TimeConfig, CouplingConfig, FieldValidationLevel

from ebfm.coupling.fields import GenericExchangeType
from ebfm.coupling.couplers import FakeCoupler
from ebfm.coupling.couplers.fakeCoupler import FakeFieldConfig

from argparse import Namespace

from ebfm.core.logger import setup_logging, getLogger

setup_logging(
    # stdout_log_level=log_levels_map["DEBUG"],
    reset_handlers=True,
)
logger = getLogger(__name__)

# TODO: Add tests for coupling components, e.g.:
# - ElmerIce component setup and field exchange
# - IconAtmo component setup and field exchange
#
# Example structure:
#
# class TestElmerIceComponent(unittest.TestCase):
#     def test_init(self):
#         ...


class TestIconAtmoComponent(unittest.TestCase):
    # def test_init(self):
    #     ...

    args = Namespace(
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-02T00:00:00Z",
        time_step="PT1H",
        calendar="proleptic_gregorian",
        component_name="ebfm",
        couple_to_icon_atmo=True,
        couple_to_icon_land=False,
        couple_to_elmer_ice=False,
        fake_coupling=True,
        field_validation_level=FieldValidationLevel("FATAL"),
        coupler_config=None,
    )

    time_config = TimeConfig(args=args)

    coupling_config = CouplingConfig(
        args=args,
        time_config=time_config,
    )

    # just fake values to get coupler._n_points set
    grid_dict = {"x": np.array([0])}

    def test_exchange(self):
        """
        Test that IconAtmo component can exchange data with a coupler.
        """
        # fake_fields will be set later.
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_atmo = coupler.get_component("icon_atmo")

        coupler._register_fake_values(
            pr_fake_field := FakeFieldConfig(
                coupled_component=icon_atmo, name="pr", value=1, exchange_type=GenericExchangeType.TARGET
            )
        )
        coupler._register_fake_values(
            pr_snow_fake_field := FakeFieldConfig(
                icon_atmo, name="pr_snow", value=2, exchange_type=GenericExchangeType.TARGET
            )
        )

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_to_icon = {}

        # Simulate data exchange
        data_from_icon = icon_atmo.exchange(data_to_icon)

        # Check that the data is received correctly
        self.assertIsNotNone(data_from_icon)
        expected_pr = icon_atmo._map_pr_to_ebfm(np.full(self.grid_dict["x"].shape, pr_fake_field.value))
        self.assertTrue(np.array_equal(data_from_icon["pr"], expected_pr))
        expected_pr_snow = icon_atmo._map_pr_to_ebfm(np.full(self.grid_dict["x"].shape, pr_snow_fake_field.value))
        self.assertTrue(np.array_equal(data_from_icon["pr_snow"], expected_pr_snow))

    def test_fallback_values(self):
        """
        Test that fallback values are used when no data is received.
        """
        # fake_fields will be set later.
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_atmo = coupler.get_component("icon_atmo")

        logger.debug(f"{icon_atmo=}")

        coupler._register_fake_values(
            pr_fake_field := FakeFieldConfig(
                coupled_component=icon_atmo, name="pr", value=1, exchange_type=GenericExchangeType.TARGET
            )
        )

        coupler.setup(
            grid=self.grid_dict,
            time=self.time_config,
        )

        data_to_icon = {}

        fallback_values = {
            "pr": [10],
            "pr_snow": [20],
        }

        # Simulate data exchange
        data_from_icon = icon_atmo.exchange(data_to_icon, fallback_values=fallback_values)

        # Check that the data is received correctly
        self.assertIsNotNone(data_from_icon)
        # For 'pr' data is given; so no fallback should be used.
        expected_pr = icon_atmo._map_pr_to_ebfm(np.full(self.grid_dict["x"].shape, pr_fake_field.value))
        self.assertTrue(np.array_equal(data_from_icon["pr"], expected_pr))
        # For 'pr_snow' data is not given; so fallback should be used.
        print(data_from_icon["pr_snow"])
        print(fallback_values["pr_snow"])
        self.assertTrue(np.array_equal(data_from_icon["pr_snow"], fallback_values["pr_snow"]))


if __name__ == "__main__":
    unittest.main()


class TestIconLandComponent(unittest.TestCase):
    args = Namespace(
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-02T00:00:00Z",
        time_step="PT1H",
        calendar="proleptic_gregorian",
        component_name="ebfm",
        couple_to_icon_atmo=False,
        couple_to_icon_land=True,
        couple_to_elmer_ice=False,
        fake_coupling=True,
        field_validation_level=FieldValidationLevel("FATAL"),
        coupler_config=None,
    )

    time_config = TimeConfig(args=args)

    coupling_config = CouplingConfig(
        args=args,
        time_config=time_config,
    )

    # just fake values to get coupler._n_points set
    grid_dict = {"x": np.array([0, 1, 2])}

    def test_field_definitions(self):
        """
        Test that the IconLand component defines icefract as a source field only.
        """
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        self.assertTrue(coupler.has_coupling_to("icon_land"))
        self.assertFalse(coupler.has_coupling_to("icon_atmo"))
        self.assertTrue(coupler.has_field("icon_land", "icefract", GenericExchangeType.SOURCE))
        self.assertTrue(coupler.has_field("icon_land", "albedo", GenericExchangeType.SOURCE))
        self.assertFalse(coupler.has_field("icon_land", "icefract", GenericExchangeType.TARGET))

        self.assertTrue(coupler.has_field("icon_land", "t_srf", GenericExchangeType.TARGET))
        self.assertTrue(coupler.has_field("icon_land", "melt", GenericExchangeType.TARGET))

        field_names = {field.name for field in icon_land.get_field_definitions(self.time_config)}
        self.assertEqual(
            field_names,
            {"icefract", "albedo", "t_srf", "melt", "evapotrans", "hfss", "hfls", "rsns", "rlns", "ghf"},
        )

    def test_exchange(self):
        """
        Test that the IconLand component sends icefract/albedo and receives the surface energy
        balance fields; mass fluxes are converted from kg m-2 s-1 to m w.e. per time step.
        """
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")

        coupler._register_fake_values(
            FakeFieldConfig(coupled_component=icon_land, name="t_srf", value=260.0, exchange_type=GenericExchangeType.TARGET)
        )
        coupler._register_fake_values(
            FakeFieldConfig(coupled_component=icon_land, name="melt", value=2.0, exchange_type=GenericExchangeType.TARGET)
        )
        coupler._register_fake_values(
            FakeFieldConfig(
                coupled_component=icon_land, name="evapotrans", value=-1.0, exchange_type=GenericExchangeType.TARGET
            )
        )
        coupler._register_fake_values(
            FakeFieldConfig(coupled_component=icon_land, name="hfss", value=-15.0, exchange_type=GenericExchangeType.TARGET)
        )

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_to_icon_land = {
            "icefract": np.array([1.0, 0.0, 1.0]),
            "albedo": np.array([0.8, 0.3, 0.6]),
        }

        data_from_icon_land = icon_land.exchange(data_to_icon_land)

        # only the fields with a fake source are received
        self.assertEqual(set(data_from_icon_land), {"t_srf", "melt", "evapotrans", "hfss"})
        self.assertTrue(np.allclose(data_from_icon_land["t_srf"], 260.0))
        self.assertTrue(np.allclose(data_from_icon_land["hfss"], -15.0))
        # 2 kg m-2 s-1 over a 1 h time step = 7.2 m w.e. per time step
        self.assertTrue(np.allclose(data_from_icon_land["melt"], 2.0 * 1e-3 * 3600.0))
        self.assertTrue(np.allclose(data_from_icon_land["evapotrans"], -1.0 * 1e-3 * 3600.0))

    def test_exchange_nothing_received(self):
        """
        Test that without coupled target fields nothing is received.
        """
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_from_icon_land = icon_land.exchange(
            {"icefract": np.array([1.0, 0.0, 1.0]), "albedo": np.array([0.8, 0.3, 0.6])}
        )

        self.assertEqual(data_from_icon_land, {})

    def test_exchange_missing_data(self):
        """
        Test that exchanging without providing icefract fails.
        """
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        with self.assertRaises(AssertionError):
            icon_land.exchange({})
        with self.assertRaises(AssertionError):
            icon_land.exchange({"icefract": np.ones(3)})
