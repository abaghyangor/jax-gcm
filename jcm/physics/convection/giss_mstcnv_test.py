"""Tests for the GISS convection PhysicsTerm scaffold.

Milestone 1: the term returns zero tendencies. These tests verify the
composable-physics interface contract (shapes, ``provides`` diagnostics,
composition, differentiability) and -- explicitly as a *trivial* check -- that
zero output matches the inactive moist convection in the DYCOMS-II RF02 case.
They do NOT claim a validated physics port (see giss_mstcnv.py).

Reference arrays come from a committed, deterministic fixture
``jcm/data/test/modele/dycoms_period24.npz`` (one DYCOMS column + its
moist-convective targets), mirroring how the SPEEDY convection tests load
``.npy`` oracle arrays. The fixture is produced offline by the separate ModelE
reading/conversion tooling; this repo holds only the committed arrays, not the
NetCDF reader.
"""

import unittest
from importlib import resources

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

_G = 9.80665      # geopotential height (m) -> geopotential (m^2/s^2)
_P0_HPA = 1013.25


def _make_terrain(shape):
    zero = jnp.zeros(shape)
    return TerrainData(
        orog=zero, phis0=zero, fmask=zero, lfluxland=jnp.bool_(False),
        orostd=zero, orosig=zero, orogam=zero,
        orothe=zero, oropic=zero, oroval=zero,
    )


def _load_dycoms_fixture():
    path = resources.files("jcm.data.test") / "modele" / "dycoms_period24.npz"
    return np.load(str(path))


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

    def test_cloud_base_sentinel_without_pressure(self):
        # With no pressure_full diagnostic the term is a pure scaffold: the
        # cloud base is the "no cloud base" sentinel (nlev) and tendencies zero.
        state = PhysicsState.ones(self.shape)
        _, diag = self.term(state, {}, None, None)
        self.assertTrue(jnp.all(diag["convection"].cloud_base == self.nlev))

    def test_cloud_base_from_pressure_full(self):
        # Provide a realistic column (JCM order: index 0 = top, last = surface)
        # with a moist surface parcel; the term should find a cloud base.
        nlev = 12
        # pressure_full increases top -> surface (Pa).
        pfull = jnp.linspace(20000.0, 100000.0, nlev)[:, None]   # (nlev, 1)
        temperature = jnp.linspace(230.0, 298.0, nlev)[:, None]
        specific_humidity = jnp.full((nlev, 1), 14.0)            # g/kg, moist
        state = PhysicsState.zeros(
            (nlev, 1), temperature=temperature,
            specific_humidity=specific_humidity)

        _, diag = self.term(state, {"pressure_full": pfull}, None, None)
        cb = diag["convection"].cloud_base
        self.assertEqual(cb.shape, (1,))
        cb0 = int(cb[0])
        self.assertGreater(cb0, 0)        # surface parcel not saturated at surface
        self.assertLess(cb0, nlev)        # ...but condenses below model top

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
        dtsrc_grad = grads.terms[0].params.get_value().dtsrc
        self.assertTrue(jnp.all(jnp.isfinite(dtsrc_grad)))


class TestGissScaffoldVsDycomsFixture(unittest.TestCase):
    """Trivial consistency: scaffold zeros match the inactive-MC DYCOMS column.

    This is a *no-convection* regression check against committed oracle arrays,
    NOT validation of physics. The DYCOMS moist-convective targets
    (dq_mc/dth_mc/mcp) are identically zero -- a convectively active case
    (BOMEX/RICO) is required to validate real physics. When that lands, this
    fixture is replaced with nonzero targets and the same test shape applies.
    """

    def test_scaffold_matches_dycoms_mc_targets(self):
        data = _load_dycoms_fixture()
        nlev, ncols = data["temperature"].shape

        state = PhysicsState.zeros(
            (nlev, ncols),
            temperature=jnp.asarray(data["temperature"]),
            specific_humidity=jnp.asarray(data["specific_humidity"]),  # g/kg
            geopotential=jnp.asarray(data["geopotential_height"]) * _G,
            normalized_surface_pressure=jnp.asarray(data["pressure_hpa"][0] / _P0_HPA),
        )

        term = GissConvection()
        tend, diag = term(state, {}, None, None)
        conv = diag["convection"]

        # Sanity: the committed DYCOMS targets really are inactive convection.
        self.assertTrue(np.all(data["dq_mc"] == 0))

        self.assertTrue(np.allclose(np.asarray(conv.dq_mc).ravel(),
                                    data["dq_mc"].ravel()))
        self.assertTrue(np.allclose(np.asarray(conv.dth_mc).ravel(),
                                    data["dth_mc"].ravel()))
        self.assertTrue(np.allclose(np.asarray(conv.mcp).ravel(),
                                    data["mcp"].ravel()))
        self.assertTrue(jnp.all(tend.temperature == 0))


if __name__ == "__main__":
    unittest.main()
