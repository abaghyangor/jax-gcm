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

import contextlib
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
from jcm.physics.convection.giss_mstcnv import (
    GissConvection,
    cloud_base_closure_mass_flux,
)
from jcm.physics.convection.giss_mass_flux import cloud_base_mass_flux_column
from jcm.physics.convection.giss_thermodynamics import KAPA
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

    def _moist_column(self, nlev=12):
        # JCM order: index 0 = top, last = surface. Warm moist surface parcel
        # under a cooler troposphere -> a real cloud base.
        pfull = jnp.linspace(20000.0, 100000.0, nlev)[:, None]        # (nlev, 1) Pa
        temperature = jnp.linspace(230.0, 298.0, nlev)[:, None]
        specific_humidity = jnp.full((nlev, 1), 16.0)                 # g/kg
        state = PhysicsState.zeros(
            (nlev, 1), temperature=temperature,
            specific_humidity=specific_humidity)
        # air_density = p/(Rd T); layer_thickness chosen so density*thickness =
        # |dp|/g (a physical layer mass ~ tens-hundreds kg/m^2).
        density = pfull / (287.0 * temperature)
        dp = jnp.abs(jnp.gradient(pfull, axis=0))
        thickness = dp / (_G * density)
        diagnostics = {"pressure_full": pfull, "layer_thickness": thickness,
                       "air_density": density}
        return state, diagnostics

    def test_mass_flux_zero_without_layer_mass(self):
        # With pressure but no layer-mass diagnostics: cloud base is found but the
        # closure mass flux stays zero (degraded diagnostic).
        state, diag = self._moist_column()
        _, out = self.term(state, {"pressure_full": diag["pressure_full"]}, None, None)
        conv = out["convection"]
        self.assertLess(int(conv.cloud_base[0]), 12)                  # cloud base found
        self.assertTrue(jnp.all(conv.cloud_base_mass_flux == 0.0))    # but no mass flux

    def test_mass_flux_with_full_diagnostics(self):
        # With pressure + layer mass, the closure runs and returns a finite,
        # non-negative cloud-base mass flux of the right shape.
        state, diag = self._moist_column()
        _, out = self.term(state, diag, None, None)
        fmp2 = out["convection"].cloud_base_mass_flux
        self.assertEqual(fmp2.shape, (1,))
        self.assertTrue(jnp.all(jnp.isfinite(fmp2)))
        self.assertTrue(jnp.all(fmp2 >= 0.0))

    # --- allow_mc tendency path (structural; magnitudes not validated) ---
    def test_allow_mc_off_returns_zero_tendency(self):
        state, diag = self._moist_column()
        tend, _ = GissConvection(allow_mc=False)(state, diag, None, None)
        self.assertTrue(jnp.all(tend.temperature == 0))
        self.assertTrue(jnp.all(tend.specific_humidity == 0))

    def test_allow_mc_produces_finite_heating(self):
        # With convection on, the term produces finite tendencies and net
        # heating somewhere in the column (compensating subsidence warms).
        state, diag = self._moist_column()
        tend, out = GissConvection(allow_mc=True)(state, diag, None, None)
        self.assertTrue(jnp.all(jnp.isfinite(tend.temperature)))
        self.assertTrue(jnp.all(jnp.isfinite(tend.specific_humidity)))
        self.assertGreater(float(jnp.max(tend.temperature)), 0.0)
        self.assertTrue(jnp.any(out["convection"].dth_mc != 0))

    def test_allow_mc_zero_without_layer_mass(self):
        # The tendency path needs layer-mass diagnostics; without them it is zero
        # even though allow_mc is set (cloud base is still diagnosed).
        state, diag = self._moist_column()
        tend, out = GissConvection(allow_mc=True)(
            state, {"pressure_full": diag["pressure_full"]}, None, None)
        self.assertTrue(jnp.all(tend.temperature == 0))
        self.assertLess(int(out["convection"].cloud_base[0]), 12)

    def test_allow_mc_differentiable(self):
        state, diag = self._moist_column()
        term = GissConvection(allow_mc=True)

        def loss(temperature):
            s = state.copy(temperature=temperature)
            tend, _ = term(s, diag, None, None)
            return jnp.sum(tend.temperature ** 2)

        g = jax.grad(loss)(state.temperature)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


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


class TestCloudBaseClosureMassFlux(unittest.TestCase):
    """The (nlev,...) gather wrapper around the MASS_FLUX2 closure."""

    def setUp(self):
        # Surface-first column (index 0 = surface). A convectively unstable
        # profile: well-mixed moist boundary layer under a sharp inversion, so
        # the blended source parcel is supersaturated when lifted -- which is
        # what makes the MASS_FLUX2 closure return a real plume rather than
        # bottoming out at its lower bisection bound.
        self.nlev = 8
        self.t = jnp.array(
            [299., 298.4, 297.8, 297.2, 292., 290., 288., 286.])[:, None]
        self.q = jnp.array(
            [0.0180, 0.0178, 0.0176, 0.0173, 0.0100, 0.0080, 0.0060,
             0.0050])[:, None]
        self.p = jnp.linspace(101000.0, 80000.0, self.nlev)[:, None]
        self.air_mass = jnp.full((self.nlev, 1), 100.0)

    def test_matches_column_closure(self):
        # The wrapper delegates to the multi-source (nlpi>1) MASS_FLUX2 closure,
        # so it must agree with calling that directly on the same column.
        cloud_base = jnp.array([2])
        _, fmp2 = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base)

        exner = (self.p / 100000.0) ** KAPA
        theta = self.t / exner
        _, fmp2_expected, _ = cloud_base_mass_flux_column(
            theta, self.q, self.air_mass, exner, self.p, cloud_base)

        self.assertAlmostEqual(float(fmp2[0]), float(fmp2_expected[0]), places=5)

    def test_boundary_layer_top_limits_source_blend(self):
        # Restricting the source blend to the boundary layer excludes the drier
        # air just below cloud base, giving a moister (more unstable) parcel and
        # so a larger closure mass flux.
        cloud_base = jnp.array([3])
        _, fmp2_all = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base)
        _, fmp2_bl = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base,
            boundary_layer_top=jnp.array([2]))
        self.assertGreater(float(fmp2_bl[0]), float(fmp2_all[0]))

    def test_source_boost_increases_mass_flux(self):
        # The tstar/qstar source-parcel enhancement makes the parcel warmer and
        # moister, hence more unstable, hence a larger closure mass flux.
        cloud_base = jnp.array([3])
        _, plain = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base)
        _, boosted = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base,
            source_dtheta=0.05, source_dq=3.0e-4)
        self.assertGreater(float(boosted[0]), float(plain[0]))

    def test_no_cloud_base_is_zero(self):
        # Sentinel cloud base (== nlev) => no cloud => zero mass flux.
        cloud_base = jnp.array([self.nlev])
        fplume, fmp2 = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, cloud_base)
        self.assertEqual(float(fmp2[0]), 0.0)
        self.assertEqual(float(fplume[0]), 0.0)

    def test_broadcasting_matches_single_column(self):
        cloud_base = jnp.array([2, 3])
        t = jnp.concatenate([self.t, self.t + 1.0], axis=1)
        q = jnp.concatenate([self.q, self.q], axis=1)
        p = jnp.concatenate([self.p, self.p], axis=1)
        am = jnp.concatenate([self.air_mass, self.air_mass], axis=1)
        _, fmp2 = cloud_base_closure_mass_flux(t, q, p, am, cloud_base)
        self.assertEqual(fmp2.shape, (2,))
        # Column 0 must match running that single column alone.
        _, fmp2_col0 = cloud_base_closure_mass_flux(
            self.t, self.q, self.p, self.air_mass, jnp.array([2]))
        self.assertAlmostEqual(float(fmp2[0]), float(fmp2_col0[0]), places=5)


if __name__ == "__main__":
    unittest.main()


@contextlib.contextmanager
def _float64():
    """Run the block in float64, then restore whatever was set before.

    Scoped on purpose. ``giss_plume_driver_test`` enables x64 at *module
    import*, and leaking precision that way once made ``giss_bsort_test``'s
    vectorised-agreement check pass only when the whole directory ran in one
    process; that test now sets its own precision. A finite-difference check
    genuinely needs float64, but it should not silently change the precision
    every other test in the process runs at.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


class TestGissConvectionGradient(unittest.TestCase):
    """Gradients on a *convecting* column, against finite differences.

    The deliverable for this port is a differentiable ``PhysicsTerm``, and the
    other gradient tests here take ``PhysicsState.ones(...)`` -- a state where
    convection never fires, every tendency is zero, and so every gradient is
    legitimately zero too. Those check that the autodiff plumbing survives the
    trace; they cannot distinguish a correct gradient from no gradient at all.
    This one runs the column that actually convects.

    **What this does and does not establish.** The scheme is piecewise smooth:
    the buoyancy sort routes each blend by a threshold test roughly 7e-4 wide
    in relative buoyancy, and the downdraft takes a three-way branch. Those
    decisions are ``jnp.where`` on comparisons, so the derivative *through a
    branch choice* is zero -- moving a blend across the threshold changes the
    answer discontinuously and no gradient sees it coming. What is verified
    here is the derivative *within* a piece, which is what gradient-based
    fitting actually consumes, and which finite differences can confirm.
    """

    def _setup(self):
        term = GissConvection(allow_mc=True)
        holder = TestGissConvectionTerm("test_allow_mc_differentiable")
        holder.setUp()
        state, diag = holder._moist_column()
        return term, state, diag

    @staticmethod
    def _loss(term, state, diag):
        # Heating in K/day rather than K/s. The scaling is not cosmetic: in
        # K/s the loss is ~1e-10 and a central difference is below float
        # resolution long before the step is small enough to be a derivative.
        def loss(temperature):
            tendency, _ = term(state.copy(temperature=temperature), diag,
                               None, None)
            return jnp.sum((tendency.temperature * 86400.0) ** 2)
        return loss

    def test_gradient_is_nonzero_where_the_column_convects(self):
        term, state, diag = self._setup()
        tendency, _ = term(state, diag, None, None)
        self.assertGreater(float(jnp.max(jnp.abs(tendency.temperature))), 0.0,
                           "fixture must actually convect for this to mean anything")
        grad = jax.grad(self._loss(term, state, diag))(state.temperature)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        # A zero gradient here would pass the older finiteness-only tests.
        self.assertGreater(int(jnp.sum(grad != 0.0)), 3)

    def test_gradient_matches_finite_differences(self):
        with _float64():
            term = GissConvection(allow_mc=True)
            holder = TestGissConvectionTerm("test_allow_mc_differentiable")
            holder.setUp()
            state, diag = holder._moist_column()
            state = state.copy(
                temperature=jnp.asarray(state.temperature, jnp.float64))
            loss = self._loss(term, state, diag)
            grad = np.asarray(jax.grad(loss)(state.temperature)).squeeze()
            base = np.asarray(state.temperature).squeeze()

            checked = 0
            for level in np.argsort(-np.abs(grad))[:4]:
                if abs(grad[level]) < 1e-8:
                    continue
                step = 1e-5
                up, down = base.copy(), base.copy()
                up[level] += step
                down[level] -= step
                shape = state.temperature.shape
                forward = float(loss(jnp.asarray(up).reshape(shape)))
                backward = float(loss(jnp.asarray(down).reshape(shape)))
                secant = (forward - backward) / (2.0 * step)
                self.assertAlmostEqual(
                    grad[level] / secant, 1.0, places=3,
                    msg=f"autodiff vs finite difference at level {level}")
                checked += 1
            self.assertGreaterEqual(checked, 3)


class TestEnvironmentCondensate(unittest.TestCase):
    """The term forwards the environment's cloud condensate from the tracers.

    ModelE loads the environment's virtual temperature with `qcl + qci` when it
    decides whether the downdraft detrains (`svm1`, MSTCNV.F90:4573). BOMEX
    cannot test this at all: `qcl + qci` is identically zero on every oracle
    level and every step, so a port ignoring the term would match the oracle
    exactly -- and this one did, until the wiring was added.

    **The physics is verified in `giss_downdraft_test`, not here.** The
    idealised column in this module seeds no downdraft, so the condensate has
    no consumer and cannot change the answer whatever value it takes; asserting
    otherwise here would be a test that can only ever pass. What this class
    checks is the wiring: tracers of either name are accepted, combined as a
    sum, and treated as zero when absent.
    """

    def setUp(self):
        holder = TestGissConvectionTerm("test_allow_mc_differentiable")
        holder.setUp()
        self.state, self.diag = holder._moist_column()
        self.term = GissConvection(allow_mc=True)

    def _heating(self, state):
        tendency, _ = self.term(state, self.diag, None, None)
        return tendency.temperature

    def test_runs_with_condensate_tracers(self):
        cloudy = self._heating(self.state.copy(tracers={
            "qc": jnp.full_like(self.state.temperature, 5.0e-4),
            "qi": jnp.full_like(self.state.temperature, 2.0e-4),
        }))
        self.assertTrue(bool(jnp.all(jnp.isfinite(cloudy))))
        self.assertEqual(cloudy.shape, self.state.temperature.shape)

    def test_absent_tracers_behave_as_zero(self):
        bare = self._heating(self.state)
        explicit_zero = self._heating(self.state.copy(tracers={
            "qc": jnp.zeros_like(self.state.temperature),
            "qi": jnp.zeros_like(self.state.temperature),
        }))
        np.testing.assert_allclose(np.asarray(bare),
                                   np.asarray(explicit_zero),
                                   rtol=1e-6, atol=1e-12)


class TestPhaseAssumption(unittest.TestCase):
    """This port is all-liquid, and that is a stated limit rather than an oversight.

    ModelE picks the latent heat per level from the *parcel* temperature against
    `tfmc` and latches it once ice forms (MSTCNV.F90:1716-1721), tracks a second
    phase for the precipitation, and redistributes the melting heat where the two
    differ. None of that is ported.

    BOMEX cannot show the difference -- it is all-liquid on every level of every
    step -- so these tests pin the assumption instead of testing physics they
    cannot reach: the scheme must stay finite on a sub-freezing column rather
    than producing nonsense, and the flag must stay honest about what is
    supported.
    """

    def setUp(self):
        holder = TestGissConvectionTerm("test_allow_mc_differentiable")
        holder.setUp()
        self.state, self.diag = holder._moist_column()
        self.term = GissConvection(allow_mc=True)

    def test_flag_states_the_limitation(self):
        from jcm.physics.convection import giss_mstcnv
        self.assertFalse(giss_mstcnv.MIXED_PHASE_SUPPORTED)
        # `tf - 15`, the threshold ModelE would switch on.
        self.assertAlmostEqual(giss_mstcnv._FREEZING_THRESHOLD, 258.15, places=6)

    def test_sub_freezing_column_stays_finite(self):
        """Below `tfmc` the answer is liquid-phase, which ModelE's would not be.

        The requirement here is only that the scheme degrades quietly rather
        than returning NaN, so that a global run cannot be poisoned by a cold
        column. Its *accuracy* there is not claimed and is not tested, because
        nothing available can check it.
        """
        from jcm.physics.convection import giss_mstcnv
        cold = self.state.copy(
            temperature=self.state.temperature - 40.0)
        self.assertLess(float(jnp.min(cold.temperature)),
                        giss_mstcnv._FREEZING_THRESHOLD)
        tendency, _ = self.term(cold, self.diag, None, None)
        self.assertTrue(bool(jnp.all(jnp.isfinite(tendency.temperature))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(tendency.specific_humidity))))

    def test_sub_freezing_column_is_differentiable(self):
        """A cold column must not break the gradient either."""
        cold = self.state.copy(temperature=self.state.temperature - 40.0)

        def loss(temperature):
            tendency, _ = self.term(cold.copy(temperature=temperature),
                                    self.diag, None, None)
            return jnp.sum((tendency.temperature * 86400.0) ** 2)

        grad = jax.grad(loss)(cold.temperature)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
