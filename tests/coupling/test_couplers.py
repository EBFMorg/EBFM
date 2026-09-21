# SPDX-FileCopyrightText: 2025 EBFM Authors
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ebfm.coupling.couplers import FakeCoupler
from ebfm.coupling.couplers.helpers import coupling_supported
from ebfm.coupling.components.base import Component
from ebfm.coupling.fields import Field, FieldSet, GenericExchangeType, Timestep

if coupling_supported:
    import yac
    from ebfm.coupling.couplers.yacCoupler import YACCoupler


class _StubComponent(Component):
    """
    Component that only exists to give Field.coupled_component a name; it never exchanges data.
    """

    def get_field_definitions(self):
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


class TestFakeCouplerGridPoints(unittest.TestCase):
    """
    FakeCoupler._infer_n_points sizes the fake arrays returned by get(), so it has to report the number of
    columns EBFM exchanges, not the vertex count of the mesh a grid may have been built from.
    """

    def _bare_fake_coupler(self) -> FakeCoupler:
        """
        Build a FakeCoupler without running its __init__, which needs a CouplingConfig. _infer_n_points reads
        no state of its own.
        """
        return FakeCoupler.__new__(FakeCoupler)

    def test_mesh_does_not_override_the_column_count(self):
        """
        A grid built from an Elmer mesh carries that mesh next to its per-column fields. The mesh describes
        the same geometry by vertex, so its length is not what the coupled fields are sized by.
        """
        mesh = SimpleNamespace(vertex_ids=np.arange(11), x_vertices=np.zeros(11), y_vertices=np.zeros(11))
        grid = {"mask": np.ones(4), "x": np.zeros(4), "mesh": mesh}

        self.assertEqual(self._bare_fake_coupler()._infer_n_points(grid), 4)

    def test_grid_without_per_column_fields_is_rejected(self):
        mesh = SimpleNamespace(vertex_ids=np.arange(11))

        with self.assertRaises(ValueError) as context:
            self._bare_fake_coupler()._infer_n_points({"mesh": mesh})
        self.assertIn("Could not infer number of grid points", str(context.exception))

    def test_no_grid_has_no_points(self):
        self.assertEqual(self._bare_fake_coupler()._infer_n_points(None), 0)


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


if __name__ == "__main__":
    unittest.main()
