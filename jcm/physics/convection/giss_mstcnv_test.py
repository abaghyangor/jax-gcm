"""Tests for the GISS convection PhysicsTerm scaffold.

Milestone 1: the term returns zero tendencies. These tests verify the
composable-physics interface contract (shapes, ``provides`` diagnostics,
composition, differentiability) and -- explicitly as a *trivial* check -- that
zero output is consistent with the inactive moist convection in the DYCOMS-II
RF02 oracle. They do NOT claim a validated physics port (see giss_mstcnv.py).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.convection.giss_mstcnv import GissConvection
from jcm.physics.modele.params import GissConvectionParameters
from jcm.physics.modele.physics_data import GissConvectionData
from jcm.physics.modele import oracle, adapter


def _make_terrain(shape):
    zero = jnp.zeros(shape)
    return TerrainData(
        orog=zero, phis0=zero, fmask=zero, lfluxland=jnp.bool_(False),
        orostd=zero, orosig=zero, orogam=zero,
        orothe=zero, oropic=zero, oroval=zero,
    )


class TestGissConvectionTerm(unittest.TestCase):
    def setUp(self):
        self.nlev, self.ncols = 63, 1
        self.shape = (self.nlev, self.ncols)
        self.term = GissConvection()

    def test_metadata(self):
        self.assertEqual(GissConvection.category, "convection")
        self.assertEqual(GissConvection.provides, ("convection",))
        self.assertEqual(GissConvection.requires, ())

    def test_returns_zero_and_writes_convection(self):
        state = PhysicsState.ones(self.shape)
        tend, diag = self.term(state, {}, None, None)
        self.assertIsInstance(tend, PhysicsTendency)
        self.assertEqual(tend.temperature.shape, self.shape)
        self.assertTrue(jnp.all(tend.temperature == 0))
        self.assertTrue(jnp.all(tend.specific_humidity == 0))
        # diagnostics dict carries a zero GissConvectionData under "convection"
        self.assertIn("convection", diag)
        conv = diag["convection"]
        self.assertIsInstance(conv, GissConvectionData)
        self.assertEqual(conv.dq_mc.shape, self.shape)
        self.assertEqual(conv.mcp.shape, (self.ncols,))
        self.assertTrue(jnp.all(conv.dq_mc == 0))
        self.assertTrue(jnp.all(conv.dth_mc == 0))
        self.assertTrue(jnp.all(conv.mcp == 0))

    def test_grad_through_state_no_nan(self):
        # Differentiability smoke w.r.t. state inputs (gradients are zero but
        # must be defined and finite, not error/NaN).
        state = PhysicsState.ones(self.shape)

        def loss(s):
            tend, _ = self.term(s, {}, None, None)
            return jnp.sum(tend.temperature ** 2) + jnp.sum(
                tend.specific_humidity ** 2
            )

        g = jax.grad(loss)(state)
        self.assertTrue(jnp.all(jnp.isfinite(g.temperature)))
        self.assertTrue(jnp.all(jnp.isfinite(g.specific_humidity)))

    def test_custom_params(self):
        term = GissConvection(
            params=GissConvectionParameters(dtsrc=jnp.array(900.0)),
            allow_mc=True,
        )
        self.assertEqual(float(term.params.get_value().dtsrc), 900.0)
        self.assertTrue(term.allow_mc)


class TestGissConvectionComposition(unittest.TestCase):
    """The term must compose and run inside ComposablePhysics."""

    def setUp(self):
        self.shape3d = (4, 6, 8)   # (nlev, nlon, nlat)
        self.grid = self.shape3d[1:]

    def _state(self):
        return PhysicsState.ones(self.shape3d)

    def test_compose_and_run(self):
        physics = ComposablePhysics(
            terms=[GissConvection()], checkpoint_terms=False,
        )
        tend, diag = physics.compute_tendencies(
            self._state(), ForcingData.zeros(self.grid), _make_terrain(self.grid),
        )
        self.assertEqual(tend.temperature.shape, self.shape3d)
        self.assertTrue(jnp.all(tend.temperature == 0))
        self.assertIn("convection", diag)

    def test_nnx_grad_runs(self):
        # nnx.grad through composable physics must run and stay finite. The
        # scaffold's loss is identically zero, so parameter gradients are zero
        # (defined, finite) -- the point is the autodiff plumbing works.
        physics = ComposablePhysics(
            terms=[GissConvection()], checkpoint_terms=False,
        )
        state = self._state()
        forcing = ForcingData.zeros(self.grid)
        terrain = _make_terrain(self.grid)

        def loss(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain)
            return jnp.sum(tend.temperature ** 2)

        self.assertEqual(float(loss(physics)), 0.0)
        grads = nnx.grad(loss)(physics)
        # dtsrc gradient is defined and finite (zero here).
        dtsrc_grad = grads.terms[0].params.get_value().dtsrc
        self.assertTrue(jnp.all(jnp.isfinite(dtsrc_grad)))


class TestGissScaffoldVsOracle(unittest.TestCase):
    """Trivial consistency: scaffold zeros match the inactive-MC oracle.

    This is a *no-convection* check, NOT validation of physics. The DYCOMS
    oracle's dq_mc/dth_mc/mcp are identically zero (see oracle_test.py).
    """

    def test_scaffold_matches_zero_mc_oracle(self):
        path = oracle.fixture_path()
        period = 24
        state = adapter.oracle_to_physics_state(path, period)
        self.assertEqual(state.temperature.shape, (63, 1))  # (nlev, ncols)

        term = GissConvection()
        tend, diag = term(state, {}, None, None)
        conv = diag["convection"]

        # Oracle MC diagnostics for this column (kg/kg/day, K/day, mm/day).
        dq_mc = oracle.read_convection_field(path, "dq_mc", period=period)
        dth_mc = oracle.read_convection_field(path, "dth_mc", period=period)
        mcp = oracle.read_column_field(path, "mcp", period=period)

        self.assertTrue(np.allclose(np.asarray(conv.dq_mc).ravel(),
                                    np.asarray(dq_mc).ravel()))
        self.assertTrue(np.allclose(np.asarray(conv.dth_mc).ravel(),
                                    np.asarray(dth_mc).ravel()))
        self.assertTrue(np.allclose(np.asarray(conv.mcp).ravel(),
                                    np.asarray(mcp).ravel()))
        self.assertTrue(jnp.all(tend.temperature == 0))


if __name__ == "__main__":
    unittest.main()
