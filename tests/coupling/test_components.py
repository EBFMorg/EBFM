# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from collections.abc import Mapping
import numpy as np

from ebfm.core.config import TimeConfig, CouplingConfig, FieldValidationLevel

from ebfm.coupling.components import Component, ExchangeKeySet
from ebfm.coupling.fields import Field, FieldSet, GenericExchangeType, Timestep
from ebfm.coupling.couplers import FakeCoupler
from ebfm.coupling.couplers.base import CouplerExitCode
from ebfm.coupling.couplers.fakeCoupler import FakeFieldConfig

from argparse import Namespace

from ebfm.core.logger import setup_logging, getLogger

setup_logging(
    # stdout_log_level=log_levels_map["DEBUG"],
    reset_handlers=True,
)
logger = getLogger(__name__)

# TODO: Add tests for coupling components, e.g.:
# - ElmerIce component setup
# - IconAtmo component setup


class RecordingFakeCoupler(FakeCoupler):
    """
    FakeCoupler that records which fields are actually communicated.

    Used to check that an exchange only puts and gets the keys it is supposed to communicate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.put_fields: list[str] = []
        self.get_fields: list[str] = []

    def put(self, component_name: str, field_name: str, data: np.ndarray) -> CouplerExitCode | None:
        self.put_fields.append(field_name)
        return super().put(component_name, field_name, data)

    def get(self, component_name: str, field_name: str) -> tuple[np.ndarray | None, CouplerExitCode | None]:
        self.get_fields.append(field_name)
        return super().get(component_name, field_name)


class TestExchangeKeySet(unittest.TestCase):
    def test_keys_compare_independently_of_set_type_and_name(self):
        """
        Test that key sets compare by their keys only, so that the key set a component declares matches the one
        requested by a caller of exchange, which is unnamed and built from frozensets.
        """
        accepted = ExchangeKeySet(name="exchange", source_keys={"smb", "runoff"}, target_keys={"surface_elevation"})
        requested = ExchangeKeySet(
            source_keys=frozenset({"runoff", "smb"}), target_keys=frozenset({"surface_elevation"})
        )

        self.assertEqual(requested, accepted)
        self.assertIn(requested, (accepted,))

    def test_keys_of_different_sets_differ(self):
        """
        Test that key sets with different keys are not equal, so that an unexpected key is not accepted.
        """
        accepted = ExchangeKeySet(name="exchange", source_keys={"smb"}, target_keys={"surface_elevation"})

        more_source_keys = ExchangeKeySet(source_keys=frozenset({"smb", "runoff"}), target_keys=accepted.target_keys)
        no_target_keys = ExchangeKeySet(source_keys=accepted.source_keys, target_keys=frozenset())

        self.assertNotEqual(more_source_keys, accepted)
        self.assertNotEqual(no_target_keys, accepted)


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

    # All TARGET fields defined by IconAtmo, i.e. everything EBFM receives from the ICON atmosphere.
    all_icon_atmo_fields = ["clt", "huss", "pr", "pr_snow", "rlds", "rsds", "sfcpres", "sfcwind", "tas"]

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
        expected_pr = icon_atmo._map_mass_flux_to_ebfm(np.full(self.grid_dict["x"].shape, pr_fake_field.value))
        self.assertTrue(np.array_equal(data_from_icon["pr"], expected_pr))
        expected_pr_snow = icon_atmo._map_mass_flux_to_ebfm(
            np.full(self.grid_dict["x"].shape, pr_snow_fake_field.value)
        )
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
        expected_pr = icon_atmo._map_mass_flux_to_ebfm(np.full(self.grid_dict["x"].shape, pr_fake_field.value))
        self.assertTrue(np.array_equal(data_from_icon["pr"], expected_pr))
        # For 'pr_snow' data is not given; so fallback should be used.
        print(data_from_icon["pr_snow"])
        print(fallback_values["pr_snow"])
        self.assertTrue(np.array_equal(data_from_icon["pr_snow"], fallback_values["pr_snow"]))

    def test_exchange_without_target_keys(self):
        """
        Test that omitting target_keys receives all coupled fields and that fallback_values stays the second
        positional argument of exchange (as used in main.py).
        """
        coupler = RecordingFakeCoupler(self.coupling_config)  # provides fake values for all TARGET fields
        icon_atmo = coupler.get_component("icon_atmo")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_from_icon = icon_atmo.exchange({}, {})

        self.assertEqual(sorted(data_from_icon), self.all_icon_atmo_fields)
        self.assertEqual(sorted(coupler.get_fields), self.all_icon_atmo_fields)

    def test_exchange_with_target_keys(self):
        """
        Test that requesting exactly the target keys of the key set IconAtmo accepts works.
        """
        coupler = RecordingFakeCoupler(self.coupling_config)
        icon_atmo = coupler.get_component("icon_atmo")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_from_icon = icon_atmo.exchange({}, target_keys=self.all_icon_atmo_fields)

        self.assertEqual(sorted(data_from_icon), self.all_icon_atmo_fields)
        self.assertEqual(sorted(coupler.get_fields), self.all_icon_atmo_fields)

    def test_exchange_with_unexpected_target_keys(self):
        """
        Test that requesting only a subset of the target keys of IconAtmo is rejected, since IconAtmo cannot
        exchange that subset.
        """
        coupler = RecordingFakeCoupler(self.coupling_config)
        icon_atmo = coupler.get_component("icon_atmo")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        with self.assertRaises(ValueError) as context:
            icon_atmo.exchange({}, target_keys={"rsds", "tas"})

        message = str(context.exception)
        # The error has to name the missing keys and the key set that would have been accepted.
        self.assertIn("missing target keys", message)
        self.assertIn("'huss'", message)
        self.assertIn("'exchange'", message)
        # Nothing is communicated if the requested keys are rejected.
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])

    def test_exchange_rejects_source_data(self):
        """
        Test that IconAtmo rejects any source data: surface fields like albedo are sent to IconLand
        instead, so IconAtmo itself sends nothing.
        """
        coupler = RecordingFakeCoupler(self.coupling_config)
        icon_atmo = coupler.get_component("icon_atmo")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        with self.assertRaises(ValueError) as context:
            icon_atmo.exchange({"albedo": 0.5})

        message = str(context.exception)
        self.assertIn("unexpected source keys: {'albedo'}", message)

    def test_exchange_with_unknown_field(self):
        """
        Test that a misspelled field name in data_to_exchange is rejected instead of being silently ignored.
        """
        coupler = RecordingFakeCoupler(self.coupling_config)
        icon_atmo = coupler.get_component("icon_atmo")

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        with self.assertRaises(ValueError) as context:
            icon_atmo.exchange({"albdeo": 0.5})

        message = str(context.exception)
        self.assertIn("unexpected source keys: {'albdeo'}", message)


class TestElmerIceComponent(unittest.TestCase):
    args = Namespace(
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-02T00:00:00Z",
        time_step="PT1H",
        calendar="proleptic_gregorian",
        component_name="ebfm",
        couple_to_icon_atmo=False,
        couple_to_icon_land=False,
        couple_to_elmer_ice=True,
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

    data_to_elmer = {
        "smb": np.array([1.0]),
        "T_ice": np.array([260.0]),
        "runoff": np.array([0.5]),
    }

    def _create_coupler(self) -> tuple[RecordingFakeCoupler, FakeFieldConfig]:
        """
        Create a coupler that provides a fake surface elevation for Elmer/Ice.
        """
        # fake_fields will be set later.
        coupler = RecordingFakeCoupler(self.coupling_config, fake_fields={})
        elmer_ice = coupler.get_component("elmer_ice")

        coupler._register_fake_values(
            surface_elevation_fake_field := FakeFieldConfig(
                coupled_component=elmer_ice, name="surface_elevation", value=1234.0
            )
        )

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        return coupler, surface_elevation_fake_field

    def test_exchange(self):
        """
        Test that ElmerIce sends and receives all its data in a single exchange.
        """
        coupler, surface_elevation_fake_field = self._create_coupler()
        elmer_ice = coupler.get_component("elmer_ice")

        data_from_elmer = elmer_ice.exchange(self.data_to_elmer)

        self.assertEqual(list(data_from_elmer), ["surface_elevation"])
        expected_surface_elevation = np.full(self.grid_dict["x"].shape, surface_elevation_fake_field.value)
        self.assertTrue(np.array_equal(data_from_elmer["surface_elevation"], expected_surface_elevation))
        self.assertEqual(sorted(coupler.put_fields), ["T_ice", "runoff", "smb"])
        self.assertEqual(coupler.get_fields, ["surface_elevation"])

    def test_split_exchange_is_rejected(self):
        """
        Test that ElmerIce rejects a split exchange, since Elmer/Ice expects all data in a single exchange.
        """
        coupler, _ = self._create_coupler()
        elmer_ice = coupler.get_component("elmer_ice")

        # Sending without receiving.
        with self.assertRaises(ValueError) as context:
            elmer_ice.exchange(self.data_to_elmer, target_keys={})
        self.assertIn("missing target keys: {'surface_elevation'}", str(context.exception))

        # Receiving without sending.
        with self.assertRaises(ValueError) as context:
            elmer_ice.exchange({}, target_keys={"surface_elevation"})
        self.assertIn("missing source keys: {'T_ice', 'runoff', 'smb'}", str(context.exception))

        # Nothing is communicated if the requested keys are rejected.
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])


class SurfaceEnergyBalanceComponent(Component):
    """
    Test component that cannot communicate all of its data in a single exchange.

    Mimics the coupling to a land model: EBFM sends its surface state, the component computes the surface energy
    balance from it and EBFM receives the results in a separate exchange. Sending and receiving in one exchange
    is not accepted, because the results are only available once the surface state has been sent.
    """

    SOURCE_FIELDS = {"icefract", "albedo"}
    TARGET_FIELDS = {"t_srf", "melt"}

    surface_state = ExchangeKeySet(name="surface state", source_keys=SOURCE_FIELDS)
    energy_balance = ExchangeKeySet(name="surface energy balance", target_keys=TARGET_FIELDS)

    accepted_exchange_key_sets = (surface_state, energy_balance)

    def get_field_definitions(self, time: TimeConfig) -> FieldSet:
        timestep = Timestep(value=time.time_step_iso8601())

        return FieldSet(
            {
                Field(name=name, coupled_component=self, timestep=timestep, exchange_type=GenericExchangeType.SOURCE)
                for name in self.SOURCE_FIELDS
            }
            | {
                Field(name=name, coupled_component=self, timestep=timestep, exchange_type=GenericExchangeType.TARGET)
                for name in self.TARGET_FIELDS
            }
        )

    def _exchange(
        self,
        data_to_exchange: Mapping[str, np.ndarray],
        fallback_values: Mapping[str, np.ndarray],
        requested_key_set: ExchangeKeySet,
    ) -> dict[str, np.ndarray]:
        """
        Send the surface state or receive the surface energy balance, depending on the requested key set.
        """
        if requested_key_set == self.surface_state:
            self._put_if_coupled("icefract", data_to_exchange)
            self._put_if_coupled("albedo", data_to_exchange)
            return {}

        # exchange() only calls this for an accepted key set, so the surface energy balance is requested.

        received_data: dict[str, np.ndarray] = {}
        for name in self.TARGET_FIELDS:
            data = self._get_if_coupled(name, fallback_values=fallback_values)
            if data is not None:
                received_data[name] = data
        return received_data


class TestSplitExchange(unittest.TestCase):
    """
    Test a component whose data has to be exchanged in more than one call.
    """

    args = Namespace(
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-02T00:00:00Z",
        time_step="PT1H",
        calendar="proleptic_gregorian",
        component_name="ebfm",
        couple_to_icon_atmo=False,
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

    fake_values = {"t_srf": 270.0, "melt": 1.0}

    surface_state_data = {
        "icefract": np.array([1.0]),
        "albedo": np.array([0.5]),
    }

    def _create_coupler(self) -> tuple[RecordingFakeCoupler, SurfaceEnergyBalanceComponent]:
        """
        Create a coupler with a SurfaceEnergyBalanceComponent attached to it.
        """
        coupler = RecordingFakeCoupler(self.coupling_config, fake_fields={})
        component = SurfaceEnergyBalanceComponent(coupler, name="surface_energy_balance")
        # The component is not part of CouplingConfig, therefore it has to be registered explicitly here.
        coupler._coupled_components[component.name] = component

        for name, value in self.fake_values.items():
            coupler._register_fake_values(FakeFieldConfig(component, name, value))

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        return coupler, component

    def test_exchange_of_source_keys_only(self):
        """
        Test that an exchange with empty target_keys sends the surface state and receives nothing.
        """
        coupler, component = self._create_coupler()

        received_data = component.exchange(self.surface_state_data, target_keys={})

        self.assertEqual(received_data, {})
        self.assertEqual(sorted(coupler.put_fields), ["albedo", "icefract"])
        self.assertEqual(coupler.get_fields, [])

    def test_exchange_of_target_keys_only(self):
        """
        Test that an exchange with empty data_to_exchange receives the surface energy balance and sends nothing.
        """
        coupler, component = self._create_coupler()

        received_data = component.exchange({}, target_keys={"t_srf", "melt"})

        self.assertEqual(sorted(received_data), ["melt", "t_srf"])
        for name, value in self.fake_values.items():
            self.assertTrue(np.array_equal(received_data[name], np.full(self.grid_dict["x"].shape, value)))
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(sorted(coupler.get_fields), ["melt", "t_srf"])

    def test_default_target_keys(self):
        """
        Test that omitting target_keys requests all target keys of the component. For this component that is the
        key set receiving the surface energy balance, so an exchange that only sends has to pass target_keys.
        """
        coupler, component = self._create_coupler()

        self.assertEqual(sorted(component.exchange({})), ["melt", "t_srf"])
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(sorted(coupler.get_fields), ["melt", "t_srf"])

        with self.assertRaises(ValueError) as context:
            component.exchange(self.surface_state_data)
        self.assertIn("unexpected target keys: {'melt', 't_srf'}", str(context.exception))

    def test_combined_exchange_is_rejected(self):
        """
        Test that sending and receiving in one exchange is rejected and that the error names both key sets.
        """
        coupler, component = self._create_coupler()

        with self.assertRaises(ValueError) as context:
            component.exchange(self.surface_state_data, target_keys={"t_srf", "melt"})

        message = str(context.exception)
        self.assertIn("'surface state'", message)
        self.assertIn("'surface energy balance'", message)
        self.assertIn("unexpected target keys: {'melt', 't_srf'}", message)
        self.assertIn("unexpected source keys: {'albedo', 'icefract'}", message)
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])

    def test_incomplete_exchange_is_rejected(self):
        """
        Test that providing only part of the keys of a valid exchange is rejected.
        """
        coupler, component = self._create_coupler()

        with self.assertRaises(ValueError) as context:
            component.exchange({"icefract": np.array([1.0])}, target_keys={})
        self.assertIn("missing source keys: {'albedo'}", str(context.exception))

        with self.assertRaises(ValueError) as context:
            component.exchange({}, target_keys={"melt"})
        self.assertIn("missing target keys: {'t_srf'}", str(context.exception))

        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])


class TestIconLandComponent(unittest.TestCase):
    """
    Test IconLand, whose state (surface_state) and surface energy balance results (energy_balance)
    have to be exchanged in two separate calls: the results are only available once ICON-Land has
    processed the state, so a single combined exchange (as EBFM used before the exchange API
    supported phases) is rejected.
    """

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

    data_to_icon_land = {
        "icefract": np.array([1.0, 0.0, 1.0]),
        "albedo": np.array([0.8, 0.3, 0.6]),
        "t_sub": np.array([250.0, 255.0, 260.0]),
        "ghf_cond": np.array([1.0, 2.0, 3.0]),
        "hcap_sub": np.array([5e4, 1e5, 1.5e5]),
        "runoff": np.array([0.0, 1e-3, 2e-3]),
        "smb": np.array([1e-3, -1e-3, 0.0]),
        "snowmass": np.array([0.5, 0.0, 2.0]),
    }

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
            {
                "icefract",
                "albedo",
                "t_sub",
                "ghf_cond",
                "hcap_sub",
                "runoff",
                "smb",
                "snowmass",
                "t_srf",
                "melt",
                "evapotrans",
            },
        )
        for name in ("t_sub", "ghf_cond", "hcap_sub", "runoff", "smb", "snowmass"):
            self.assertTrue(coupler.has_field("icon_land", name, GenericExchangeType.SOURCE))

    def _create_coupler(self) -> RecordingFakeCoupler:
        """
        Create a coupler that provides fake surface energy balance results for IconLand.
        """
        coupler = RecordingFakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")

        for name, value in {"t_srf": 260.0, "melt": 2.0, "evapotrans": -1.0}.items():
            coupler._register_fake_values(
                FakeFieldConfig(
                    coupled_component=icon_land, name=name, value=value, exchange_type=GenericExchangeType.TARGET
                )
            )

        coupler.setup(grid=self.grid_dict, time=self.time_config)

        return coupler

    def test_surface_state(self):
        """
        Test that surface_state sends icefract/albedo and the firn state, and receives nothing.
        """
        coupler = self._create_coupler()
        icon_land = coupler.get_component("icon_land")

        received_data = icon_land.exchange(self.data_to_icon_land, target_keys=set())

        self.assertEqual(received_data, {})
        self.assertEqual(
            sorted(coupler.put_fields),
            ["albedo", "ghf_cond", "hcap_sub", "icefract", "runoff", "smb", "snowmass", "t_sub"],
        )
        self.assertEqual(coupler.get_fields, [])

    def test_energy_balance(self):
        """
        Test that energy_balance receives the surface energy balance and sends nothing; mass fluxes
        are converted from kg m-2 s-1 to m w.e. per time step.
        """
        coupler = self._create_coupler()
        icon_land = coupler.get_component("icon_land")

        data_from_icon_land = icon_land.exchange({}, target_keys={"t_srf", "melt", "evapotrans"})

        self.assertEqual(set(data_from_icon_land), {"t_srf", "melt", "evapotrans"})
        self.assertTrue(np.allclose(data_from_icon_land["t_srf"], 260.0))
        # 2 kg m-2 s-1 over a 1 h time step = 7.2 m w.e. per time step
        self.assertTrue(np.allclose(data_from_icon_land["melt"], 2.0 * 1e-3 * 3600.0))
        self.assertTrue(np.allclose(data_from_icon_land["evapotrans"], -1.0 * 1e-3 * 3600.0))
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(sorted(coupler.get_fields), ["evapotrans", "melt", "t_srf"])

    def test_default_target_keys_is_energy_balance(self):
        """
        Test that omitting target_keys requests all target keys of IconLand, i.e. the surface energy
        balance, so a surface_state exchange has to pass target_keys=set() explicitly (as main.py
        does).
        """
        coupler = self._create_coupler()
        icon_land = coupler.get_component("icon_land")

        self.assertEqual(sorted(icon_land.exchange({})), ["evapotrans", "melt", "t_srf"])

        with self.assertRaises(ValueError) as context:
            icon_land.exchange(self.data_to_icon_land)
        self.assertIn("unexpected target keys", str(context.exception))

    def test_combined_exchange_is_rejected(self):
        """
        Test that sending the state and receiving the results in a single exchange is rejected,
        since the results are only available once ICON-Land has processed the state sent via
        surface_state.
        """
        coupler = self._create_coupler()
        icon_land = coupler.get_component("icon_land")

        with self.assertRaises(ValueError) as context:
            icon_land.exchange(self.data_to_icon_land, target_keys={"t_srf", "melt", "evapotrans"})

        message = str(context.exception)
        self.assertIn("'surface state'", message)
        self.assertIn("'surface energy balance'", message)
        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])

    def test_exchange_nothing_received(self):
        """
        Test that energy_balance returns nothing when no fake target values are registered.
        """
        coupler = RecordingFakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")
        coupler.setup(grid=self.grid_dict, time=self.time_config)

        data_from_icon_land = icon_land.exchange({}, target_keys={"t_srf", "melt", "evapotrans"})

        self.assertEqual(data_from_icon_land, {})

    def test_exchange_missing_data(self):
        """
        Test that surface_state without providing all the state fields is rejected before anything
        is communicated.
        """
        coupler = self._create_coupler()
        icon_land = coupler.get_component("icon_land")

        with self.assertRaises(ValueError) as context:
            icon_land.exchange({}, target_keys=set())
        self.assertIn("missing source keys", str(context.exception))

        with self.assertRaises(ValueError) as context:
            icon_land.exchange({"icefract": np.ones(3), "albedo": np.ones(3)}, target_keys=set())
        self.assertIn("missing source keys", str(context.exception))

        self.assertEqual(coupler.put_fields, [])
        self.assertEqual(coupler.get_fields, [])

    def test_mass_flux_conversions(self):
        """
        m w.e. per time step <-> kg m-2 s-1 (time step 1 h in this test configuration)
        """
        coupler = FakeCoupler(self.coupling_config, fake_fields={})
        icon_land = coupler.get_component("icon_land")
        coupler.setup(grid=self.grid_dict, time=self.time_config)

        mwe = np.array([3.6e-3])  # 3.6 mm per hour = 1e-3 kg m-2 s-1
        flux = icon_land._map_mass_flux_from_ebfm(mwe)
        np.testing.assert_allclose(flux, [1e-3])
        np.testing.assert_allclose(icon_land._map_mass_flux_to_ebfm(flux), mwe)


if __name__ == "__main__":
    unittest.main()
