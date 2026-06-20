"""Tests for the GISS thermodynamics port.

Standalone-function + gradient tests in the style of the other JCM convection
schemes (e.g. ``tiedtke_nordeng``'s ``convection_simple_test``). Reference
values are physically grounded (not regenerated from the function itself):

* water saturation vapour pressure at 0 C is ~611 Pa,
* and at 100 C equals ~1 atm (101325 Pa) by definition of boiling.
"""

import unittest

import jax
import jax.numpy as jnp
from jax.test_util import check_vjp, check_jvp

from jcm.physics.convection import giss_thermodynamics as gt


class TestSaturationVaporPressure(unittest.TestCase):
    def test_reference_values(self):
        # Murphy & Koop (2005) over liquid water, independent physical anchors.
        # Note: ModelE clips the water fit to <= 332 K (~59 C), faithfully
        # reproduced here, so anchors must stay within the fit's valid range.
        es_0c = float(gt.saturation_vapor_pressure(jnp.array(273.15), "water"))
        self.assertAlmostEqual(es_0c, 611.2, delta=3.0)          # ~611 Pa at 0 C
        es_20c = float(gt.saturation_vapor_pressure(jnp.array(293.15), "water"))
        self.assertAlmostEqual(es_20c, 2339.0, delta=20.0)       # ~2339 Pa at 20 C

    def test_clipped_above_fit_range(self):
        # ModelE clips the water fit at 332 K; above that the value is constant.
        es_332 = gt.saturation_vapor_pressure(jnp.array(332.0), "water")
        es_360 = gt.saturation_vapor_pressure(jnp.array(360.0), "water")
        self.assertTrue(jnp.allclose(es_332, es_360))

    def test_monotonic_and_positive(self):
        t = jnp.linspace(220.0, 320.0, 50)
        es = gt.saturation_vapor_pressure(t, "water")
        self.assertTrue(jnp.all(es > 0))
        self.assertTrue(jnp.all(jnp.diff(es) > 0))               # increases with T

    def test_ice_below_water_at_subzero(self):
        # Over ice the saturation pressure is below that over supercooled water.
        t = jnp.array(253.15)  # -20 C
        self.assertLess(
            float(gt.saturation_vapor_pressure(t, "ice")),
            float(gt.saturation_vapor_pressure(t, "water")),
        )

    def test_bad_phase_raises(self):
        with self.assertRaises(ValueError):
            gt.saturation_vapor_pressure(jnp.array(300.0), "plasma")


class TestSaturationSpecificHumidity(unittest.TestCase):
    def test_surface_magnitude(self):
        # ~300 K near the surface: qsat is a few % (tens of g/kg).
        qs = float(gt.saturation_specific_humidity(
            jnp.array(300.0), jnp.array(101325.0), "water"))
        self.assertGreater(qs, 0.015)
        self.assertLess(qs, 0.030)

    def test_decreases_with_pressure(self):
        t = jnp.array(290.0)
        qs_low_alt = gt.saturation_specific_humidity(t, jnp.array(101325.0))
        qs_high_alt = gt.saturation_specific_humidity(t, jnp.array(50000.0))
        self.assertLess(float(qs_low_alt), float(qs_high_alt))

    def test_increases_with_temperature(self):
        p = jnp.array(101325.0)
        qs = gt.saturation_specific_humidity(jnp.array([280.0, 290.0, 300.0]), p)
        self.assertTrue(jnp.all(jnp.diff(qs) > 0))


class TestMoistStaticEnergy(unittest.TestCase):
    def test_formula(self):
        t, phi, q = jnp.array(300.0), jnp.array(5000.0), jnp.array(0.01)
        expected = gt.SHA * t + phi + gt.LHE * q
        self.assertAlmostEqual(
            float(gt.moist_static_energy(t, phi, q, "water")), float(expected), places=3)

    def test_ice_uses_sublimation_latent_heat(self):
        t, phi, q = jnp.array(250.0), jnp.array(0.0), jnp.array(0.001)
        mse_ice = gt.moist_static_energy(t, phi, q, "ice")
        self.assertAlmostEqual(float(mse_ice), float(gt.SHA * t + gt.LHS * q), places=3)

    def test_monotonic_in_each_argument(self):
        base = gt.moist_static_energy(jnp.array(300.0), jnp.array(0.0), jnp.array(0.01))
        self.assertGreater(  # warmer
            float(gt.moist_static_energy(jnp.array(301.0), jnp.array(0.0), jnp.array(0.01))),
            float(base))
        self.assertGreater(  # higher
            float(gt.moist_static_energy(jnp.array(300.0), jnp.array(100.0), jnp.array(0.01))),
            float(base))
        self.assertGreater(  # moister
            float(gt.moist_static_energy(jnp.array(300.0), jnp.array(0.0), jnp.array(0.011))),
            float(base))


class TestDifferentiability(unittest.TestCase):
    """JAX gives d(qsat)/dT for free -- no need to port ModelE's DLNQSATDT."""

    def test_dqsat_dt_positive_and_finite(self):
        p = jnp.array(85000.0)

        def total_qsat(t):
            return jnp.sum(gt.saturation_specific_humidity(t, p))

        t = jnp.array([260.0, 280.0, 300.0])
        dqs_dt = jax.grad(total_qsat)(t)
        self.assertTrue(jnp.all(jnp.isfinite(dqs_dt)))
        self.assertTrue(jnp.all(dqs_dt > 0))  # qsat rises with T (Clausius-Clapeyron)

    def test_vjp_jvp_smoke(self):
        t = jnp.array([265.0, 285.0, 305.0])
        p = jnp.array([70000.0, 85000.0, 101325.0])

        def f(t, p):
            return gt.saturation_specific_humidity(t, p)

        check_vjp(f, lambda t, p: jax.vjp(f, t, p), (t, p), atol=1e-3, rtol=1e-3, eps=1e-4)
        check_jvp(f, lambda t, p: jax.jvp(f, t, p), (t, p), atol=1e-3, rtol=1e-3, eps=1e-4)


class TestBroadcasting(unittest.TestCase):
    """Broadcasting-native: identical code on (nlev,) and (nlev, ncols)."""

    def test_column_matches_vectorized(self):
        t_col = jnp.linspace(300.0, 230.0, 10)
        p_col = jnp.linspace(101325.0, 25000.0, 10)
        qs_col = gt.saturation_specific_humidity(t_col, p_col)

        ncols = 4
        t_block = jnp.tile(t_col[:, None], (1, ncols))
        p_block = jnp.tile(p_col[:, None], (1, ncols))
        qs_block = gt.saturation_specific_humidity(t_block, p_block)

        self.assertEqual(qs_block.shape, (10, ncols))
        for j in range(ncols):
            self.assertTrue(jnp.allclose(qs_block[:, j], qs_col))


if __name__ == "__main__":
    unittest.main()
