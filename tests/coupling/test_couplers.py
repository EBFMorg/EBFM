# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from unittest import mock

from ebfm.coupling.couplers.helpers import coupling_supported
from ebfm.coupling.components.base import Component
from ebfm.coupling.fields import Field, FieldSet, GenericExchangeType, Timestep

if coupling_supported:
    import yac
    from ebfm.coupling.couplers.yacCoupler import YACCoupler


class _StubTime:
    """Stand-in for TimeConfig: get_field_definitions only ever calls time_step_iso8601()."""

    def time_step_iso8601(self) -> str:
        return "PT1H"


class _StubComponent(Component):
    """
    Component that only exists to give Field.coupled_component a name; it never exchanges data.
    """

    def get_field_definitions(self, time):
        return FieldSet()

    def _exchange(self, data_to_exchange, fallback_values, requested_key_set):
        return {}


def _field(name: str, coupled_component: Component, exchange_type=GenericExchangeType.TARGET) -> Field:
    return Field(
        name=name,
        coupled_component=coupled_component,
        timestep=Timestep(value="PT1H"),
        exchange_type=exchange_type,
    )


def _bare_yac_coupler(component_names: list[str]) -> tuple["YACCoupler", dict[str, Component]]:
    """
    Build a YACCoupler without running its real __init__, which needs a live YAC/MPI session.

    Only sets the state _construct_coupling_pre_sync() actually reads.
    """
    coupler = YACCoupler.__new__(YACCoupler)
    coupler.interface = None
    coupler.component = None
    coupler.cell_centers = None
    coupler._fields = FieldSet()
    components = {name: _StubComponent(coupler=coupler, name=name) for name in component_names}
    coupler._coupled_components = dict(components)
    return coupler, components


@unittest.skipUnless(coupling_supported, "requires yac (pip install 'ebfm[cpl]')")
class TestYACCouplerFieldRegistration(unittest.TestCase):
    """
    Test that _construct_coupling_pre_sync() registers each field name with YAC at most once. YAC itself rejects
    a second registration for the same (component, grid, name), but only deep inside its own error handling, so
    EBFM has to catch a duplicate field name earlier, before it ever reaches YAC.
    """

    def test_distinct_field_names_are_each_registered_once(self):
        coupler, components = _bare_yac_coupler(["partner_a", "partner_b"])
        field_definitions = FieldSet(
            {
                _field("field_a", components["partner_a"]),
                _field("field_b", components["partner_b"]),
            }
        )

        with mock.patch.object(yac.Field, "create", side_effect=lambda *a, **k: object()) as create:
            coupler._construct_coupling_pre_sync(field_definitions)

        self.assertEqual(create.call_count, 2)
        self.assertEqual({f.name for f in coupler._fields.all()}, {"field_a", "field_b"})

    def test_field_name_declared_by_two_components_is_rejected(self):
        coupler, components = _bare_yac_coupler(["partner_a", "partner_b"])
        field_definitions = FieldSet(
            {
                _field("shared", components["partner_a"]),
                _field("shared", components["partner_b"]),
            }
        )

        with mock.patch.object(yac.Field, "create", side_effect=lambda *a, **k: object()):
            with self.assertRaises(AssertionError) as context:
                coupler._construct_coupling_pre_sync(field_definitions)

        # The error must name the field and both components involved, regardless of which one is processed
        # first (field_definitions is a FieldSet, so iteration order is not guaranteed).
        message = str(context.exception)
        self.assertIn("shared", message)
        self.assertIn("partner_a", message)
        self.assertIn("partner_b", message)

    def test_elmer_ice_and_icon_land_reject_overlapping_field_definitions(self):
        """
        Regression test using the real component classes rather than _StubComponent: nothing stops
        --couple-to-elmer-ice and --couple-to-icon-land from both being enabled at once, and their
        field definitions currently overlap (e.g. both declare "smb"), so this must be rejected
        instead of silently registering the same field name with YAC twice.
        """
        from ebfm.coupling.components.elmer_ice import ElmerIce
        from ebfm.coupling.components.icon_land import IconLand

        coupler, _ = _bare_yac_coupler([])
        # Unlike _field()'s synthetic fields, ElmerIce's and IconLand's real fields carry metadata, which
        # construct_yac_field reports to self.interface; a Mock stands in for the live YAC interface bare_yac_coupler
        # doesn't have.
        coupler.interface = mock.Mock()
        elmer_ice = ElmerIce(coupler=coupler, name="elmer_ice")
        icon_land = IconLand(coupler=coupler, name="icon_land")
        coupler._coupled_components = {"elmer_ice": elmer_ice, "icon_land": icon_land}

        field_definitions = elmer_ice.get_field_definitions(_StubTime()) | icon_land.get_field_definitions(_StubTime())

        # Unlike the other tests here, these fields carry metadata, so construct_yac_field also reads
        # component_name/grid_name/name off the created field; a Mock() (rather than object()) answers those.
        with mock.patch.object(yac.Field, "create", side_effect=lambda *a, **k: mock.Mock()):
            with self.assertRaises(AssertionError) as context:
                coupler._construct_coupling_pre_sync(field_definitions)

        # Whichever field name collides first is up to FieldSet's iteration order, which isn't
        # guaranteed, so only check that both components are named, not which field it was.
        message = str(context.exception)
        self.assertIn("elmer_ice", message)
        self.assertIn("icon_land", message)


if __name__ == "__main__":
    unittest.main()
