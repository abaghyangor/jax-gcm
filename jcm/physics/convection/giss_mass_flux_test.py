"""Tests for the GISS cloud-base mass-flux closure (nlpi=1 port).

Structural tests only -- the closure *magnitude* is not a level-by-level oracle
match. They assert the defining behaviour: the bisection drives the residual
cloud-base instability ``DMSE1`` toward neutral, a more unstable column needs a
larger mass flux, and the routine is finite/differentiable and
broadcasting-native (three-level stencil on axis 0).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.convection.giss_cloud_base import cloud_base_instability
import jcm.physics.convection.giss_mass_flux as mf
from jcm.physics.convection.giss_mass_flux import (
    cloud_base_closure,
    cloud_base_mass_flux,
)


# A convectively unstable cloud-base stencil (vertical on axis 0):
# [lmin, lmin+1, lmin+2]. Source layer warm + moist; cooler/drier above.
_THETA = jnp.array([320.0, 318.0, 316.0])
_EXNER3 = jnp.array([0.97, 0.96, 0.95])
_PRESSURE3 = jnp.array([90000.0, 87000.0, 85000.0])
_Q = jnp.array([0.035, 0.012, 0.008])
_MASS = jnp.array([600.0, 600.0, 600.0])
_EXNER = jnp.array([0.96, 0.95])
_PRESSURE = jnp.array([87000.0, 85000.0])


class TestCloudBaseMassFlux(unittest.TestCase):
    def test_closure_neutralizes_instability(self):
        # Instability with no mass flux (fplume=0) == cloud_base_instability.
        dmse_initial = float(cloud_base_instability(
            _THETA[0], _Q[0], jnp.array(0.0),
            _THETA[1], _Q[1], jnp.array(0.0),
            _EXNER[1], _PRESSURE[1]))
        self.assertLess(dmse_initial, 0.0)               # genuinely unstable

        fplume, fmp2, dmse_final = cloud_base_mass_flux(
            _THETA, _Q, _MASS, _EXNER, _PRESSURE)
        # The closure moves the cloud base toward neutral.
        self.assertLess(abs(float(dmse_final)), abs(dmse_initial))
        self.assertGreater(float(fplume), 0.0)           # nonzero mass flux
        self.assertAlmostEqual(float(fmp2), float(fplume) * 600.0, places=4)

    def test_more_unstable_needs_more_mass_flux(self):
        fplume_a, _, _ = cloud_base_mass_flux(_THETA, _Q, _MASS, _EXNER, _PRESSURE)
        stronger = _THETA.at[0].set(323.0)               # warmer, more buoyant source
        fplume_b, _, _ = cloud_base_mass_flux(stronger, _Q, _MASS, _EXNER, _PRESSURE)
        self.assertGreater(float(fplume_b), float(fplume_a))

    def test_finite_and_differentiable(self):
        def loss(theta_dn):
            theta = _THETA.at[0].set(theta_dn)
            fplume, _, _ = cloud_base_mass_flux(theta, _Q, _MASS, _EXNER, _PRESSURE)
            return fplume
        g = jax.grad(loss)(jnp.array(320.0))
        self.assertTrue(jnp.isfinite(g))
        self.assertTrue(jnp.isfinite(loss(jnp.array(320.0))))

    def test_broadcasting(self):
        ncols = 2
        theta = jnp.broadcast_to(_THETA[:, None], (3, ncols))
        q = jnp.broadcast_to(_Q[:, None], (3, ncols))
        mass = jnp.broadcast_to(_MASS[:, None], (3, ncols))
        exner = jnp.broadcast_to(_EXNER[:, None], (2, ncols))
        pressure = jnp.broadcast_to(_PRESSURE[:, None], (2, ncols))
        fplume, fmp2, dmse = cloud_base_mass_flux(theta, q, mass, exner, pressure)
        self.assertEqual(fplume.shape, (ncols,))
        # Both columns identical here -> identical result, and matches the column.
        self.assertAlmostEqual(float(fplume[0]), float(fplume[1]), places=6)
        fpl_col, _, _ = cloud_base_mass_flux(_THETA, _Q, _MASS, _EXNER, _PRESSURE)
        self.assertAlmostEqual(float(fplume[0]), float(fpl_col), places=6)


if __name__ == "__main__":
    unittest.main()


class CloudBaseClosureTest(unittest.TestCase):
    """The multi-source closure, `nlpi > 1`.

    BOMEX runs `nlpi` between 6 and 9 in every oracle closure call, so this is
    the form the model actually takes; the three-level function above is its
    `nlpi = 1` special case.
    """

    NLEV = 12

    def _column(self):
        level = np.arange(self.NLEV, dtype=float)
        pressure = jnp.asarray((1000.0 - 25.0 * level) * 100.0)
        exner = (pressure / 100.0) ** 0.28622        # ModelE `plk`
        # Tuned so the bisection lands *inside* its range rather than riding a
        # bound: a column that stays unstable however much mass is removed
        # saturates `fplume` at the same lattice point whatever the source
        # block looks like, and then nothing here discriminates.
        theta = jnp.asarray(
            (298.0 + 2.0 * np.maximum(0.0, level - 5.0)) / 1000.0 ** 0.28622)
        q = jnp.asarray(np.maximum(0.016 - 0.0012 * level, 1.0e-5))
        layer_mass = jnp.full(self.NLEV, 25.0 * 100.0 / 9.80665)
        return theta, q, layer_mass, exner, pressure

    def _weights(self, bottom, top):
        level = jnp.arange(self.NLEV)
        inside = (level >= bottom) & (level <= top)
        return jnp.where(inside, 1.0, 0.0) / jnp.sum(jnp.where(inside, 1.0, 0.0))

    def test_reduces_to_the_single_source_case(self):
        # With all the weight on one layer the general form must reproduce the
        # three-level closure exactly: no cascade, and the blend is that layer.
        theta, q, layer_mass, exner, pressure = self._column()
        top = 4
        general = cloud_base_closure(
            theta, q, layer_mass, exner, pressure, jnp.array(top),
            jnp.array(top), self._weights(top, top))
        stencil = mf.cloud_base_mass_flux(
            theta[top:top + 3], q[top:top + 3], layer_mass[top:top + 3],
            exner[top:top + 2], pressure[top:top + 2])
        # `fplume` and `fmp2` must agree exactly -- they are lattice points and
        # a mass. `dmse1` is a small difference of large saturation terms, so it
        # carries single-precision noise at the 1e-5 level.
        self.assertEqual(float(general[0]), float(stencil[0]))
        self.assertAlmostEqual(float(general[1]), float(stencil[1]), places=4)
        self.assertAlmostEqual(float(general[2]), float(stencil[2]), places=3)

    def test_plume_mass_scales_the_top_source_layer(self):
        # `FMP2 = FPLUME*AML(NLPI)` -- the trial fraction scales the top source
        # layer however many layers the block spans. Verified against the oracle
        # bisection trace to 5e-7.
        theta, q, layer_mass, exner, pressure = self._column()
        fplume, fmp2, _ = cloud_base_closure(
            theta, q, layer_mass, exner, pressure, jnp.array(0), jnp.array(5),
            self._weights(0, 5))
        self.assertAlmostEqual(float(fmp2), float(fplume * layer_mass[5]),
                               places=10)

    def test_spreading_the_source_changes_the_answer(self):
        # If it did not, the multi-source generalisation would be doing nothing:
        # the removal, the cascade and the blended `SDN` all depend on how the
        # weight is distributed.
        theta, q, layer_mass, exner, pressure = self._column()
        deep = cloud_base_closure(theta, q, layer_mass, exner, pressure,
                                  jnp.array(0), jnp.array(5),
                                  self._weights(0, 5))[1]
        shallow = cloud_base_closure(theta, q, layer_mass, exner, pressure,
                                     jnp.array(5), jnp.array(5),
                                     self._weights(5, 5))[1]
        self.assertGreater(abs(float(deep) - float(shallow)), 1.0)

    def test_a_more_unstable_column_convects_harder(self):
        theta, q, layer_mass, exner, pressure = self._column()
        base = cloud_base_closure(theta, q, layer_mass, exner, pressure,
                                  jnp.array(0), jnp.array(5),
                                  self._weights(0, 5))[1]
        wetter = cloud_base_closure(theta, q * 1.05, layer_mass, exner,
                                    pressure, jnp.array(0), jnp.array(5),
                                    self._weights(0, 5))[1]
        self.assertGreater(float(wetter), float(base))

    def test_gradient_is_finite(self):
        theta, q, layer_mass, exner, pressure = self._column()
        weights = self._weights(0, 5)
        grad = jax.grad(lambda x: cloud_base_closure(
            theta, x, layer_mass, exner, pressure, jnp.array(0), jnp.array(5),
            weights)[1])(q)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_broadcasting(self):
        theta, q, layer_mass, exner, pressure = self._column()
        weights = self._weights(0, 5)
        single = cloud_base_closure(theta, q, layer_mass, exner, pressure,
                                    jnp.array(0), jnp.array(5), weights)[1]
        tile = lambda a: jnp.tile(a[:, None], (1, 3))
        block = cloud_base_closure(
            tile(theta), tile(q), tile(layer_mass), tile(exner), tile(pressure),
            jnp.array(0), jnp.array(5), tile(weights))[1]
        self.assertEqual(block.shape, (3,))
        np.testing.assert_allclose(np.asarray(block), float(single), rtol=1e-10)


class TestClosureDeclinesToConvect(unittest.TestCase):
    """ModelE refuses to convect at four points; all four must be ported.

    Found by running the port on DYCOMS-II. ModelE's convective tendencies
    there are identically zero at every level of every one of the 48 periods --
    it is stratocumulus, handled entirely by large-scale condensation -- and the
    port convected in **48 periods out of 48**, at a small but non-zero
    0.25 K/day. Only the saturation guard had been ported; the instability check
    and the two `MINFRAC` floors had not.

    BOMEX cannot catch this. Every one of its columns is meant to convect, so a
    missing veto is invisible there: a scheme that never declines agrees with
    the oracle on every case that convects, and is wrong only on the cases that
    do not.
    """

    def _column(self):
        """The driver's trade-cumulus column, in ModelE's `th`/`plk` pair.

        Borrowed rather than rebuilt because this closure reads `exner` as
        ModelE's `(p in mb)**kappa` -- about 7, not the normalised 0.97 -- and a
        fixture in the wrong convention silently looks subsaturated and gets
        vetoed for the wrong reason.
        """
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "_driver_fixture",
            __file__.replace("giss_mass_flux_test", "giss_plume_driver_test"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["_driver_fixture"] = module
        spec.loader.exec_module(module)
        return module._column()

    def _closure(self, column, base=5, return_gates=False):
        from jcm.physics.convection.giss_plume_driver import (
            source_bottom, source_weights)
        source_low = source_bottom(column["pressure"], jnp.array(base))
        weights = source_weights(column["layer_mass"], source_low,
                                 jnp.array(base), jnp.array(base - 1))
        return cloud_base_closure(
            column["potential_temperature"], column["specific_humidity"],
            column["layer_mass"], column["exner"], column["pressure"],
            source_low, jnp.array(base), weights,
            timestep=jnp.array(1800.0), return_gates=return_gates)

    def test_unstable_column_still_convects(self):
        """The vetoes must not suppress a column that should convect."""
        _, fmp2, _ = self._closure(self._column())
        self.assertGreater(float(fmp2), 1.0)

    def test_stable_column_gets_no_mass_flux(self):
        """Saturation alone is not a licence to convect.

        Same column, but with the potential temperature above the base raised
        steeply so a lifted parcel is always colder than its surroundings.
        `DMSE` never goes negative and ModelE returns before the bisection.
        """
        column = dict(self._column())
        theta = np.asarray(column["potential_temperature"]).copy()
        theta[5:] += np.arange(len(theta) - 5) * 2.0 + 2.0
        column["potential_temperature"] = jnp.asarray(theta)
        _, fmp2, _ = self._closure(column)
        self.assertEqual(float(fmp2), 0.0)

    def test_gates_say_which_veto_closed(self):
        """`return_gates` must identify the veto, not just report zero.

        Four vetoes share one output, so `fmp2 == 0` is not diagnosable after
        the fact: DYCOMS (three vetoes missing, the port convecting 48/48) and
        TWP-ICE (the saturation gate closing on a quarter of a deep case)
        present identically. This is the diagnostic that separates them.
        """
        column = dict(self._column())
        column["potential_temperature"] = (
            column["potential_temperature"].at[5:].add(12.0))
        *_, gates = self._closure(column, return_gates=True)
        self.assertFalse(bool(jnp.all(gates["unstable"])))
        for key in ("saturated", "big_enough", "above_floor", "fplume",
                    "dmse0", "humidity_deficit"):
            self.assertIn(key, gates)

    def test_gates_do_not_change_the_answer(self):
        column = self._column()
        plain = self._closure(column)
        with_gates = self._closure(column, return_gates=True)
        self.assertEqual(len(plain), 3)
        self.assertEqual(len(with_gates), 4)
        for a, b in zip(plain, with_gates[:3]):
            self.assertEqual(float(jnp.max(jnp.abs(a - b))), 0.0)

    def test_minfrac_floor_is_the_modele_value(self):
        # MSTCNV.F90:1226, `.0005` whenever `cold_pool_on` -- its default.
        self.assertEqual(mf._MIN_PLUME_FRACTION, 5.0e-4)
