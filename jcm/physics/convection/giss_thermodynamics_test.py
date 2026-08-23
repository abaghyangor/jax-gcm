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


class TestVirtualTemperature(unittest.TestCase):
    def test_dry_air_equals_temperature(self):
        t = jnp.array(290.0)
        self.assertAlmostEqual(float(gt.virtual_temperature(t, jnp.array(0.0))),
                               float(t), places=6)

    def test_moisture_raises_it(self):
        t = jnp.array(290.0)
        tv = gt.virtual_temperature(t, jnp.array(0.015))   # 15 g/kg vapour
        self.assertGreater(float(tv), float(t))
        # exact: T*(1 + DELTX*q)
        self.assertAlmostEqual(float(tv), float(t) * (1.0 + gt.DELTX * 0.015),
                               places=5)

    def test_condensate_loads_it_down(self):
        t, q = jnp.array(290.0), jnp.array(0.015)
        tv_clear = gt.virtual_temperature(t, q)
        tv_cloudy = gt.virtual_temperature(t, q, condensate=jnp.array(0.002))
        self.assertLess(float(tv_cloudy), float(tv_clear))

    def test_gradient_finite(self):
        g = jax.grad(lambda q: gt.virtual_temperature(jnp.array(290.0), q).sum())(
            jnp.array(0.01))
        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(g), 0.0)


class TestDLnQsatDt(unittest.TestCase):
    def test_clausius_clapeyron_magnitude(self):
        # ~6-7 %/K near room temperature (Clausius-Clapeyron).
        d = float(gt.d_ln_qsat_dt(jnp.array(300.0), "water"))
        self.assertGreater(d, 0.05)
        self.assertLess(d, 0.075)

    def test_decreases_with_temperature(self):
        d = gt.d_ln_qsat_dt(jnp.array([260.0, 280.0, 300.0]))
        self.assertTrue(jnp.all(jnp.diff(d) < 0))      # ~1/T^2 falloff


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


class TestCondensateEvaporation(unittest.TestCase):
    """``get_dq_evap`` (``CLOUDS_COM.F90:903``), the blend-evaporation step.

    The reference here is the Fortran loop transcribed literally into Python
    floats, so these check the port's arithmetic rather than re-deriving the
    scheme from itself. Cases are chosen to exercise the Newton interior
    (large ``condensate``, so the clip does not bind) as well as both clips.
    """

    PL = 95000.0                        # Pa
    PLK = (95000.0 / 100.0) ** 0.28622  # ModelE PLK = p[mb]**KAPA

    def _reference(self, sm, qm, cond, phase="water"):
        latent = gt.LHE if phase == "water" else gt.LHS
        slh = latent / gt.SHA
        qmt, tp, dqsum = qm, sm * self.PLK, 0.0
        for _ in range(3):
            qst = float(gt.saturation_specific_humidity(
                jnp.array(tp), jnp.array(self.PL), phase))
            dq = (qmt - qst) / (1.0 + slh * qst * latent / (gt.RVAP * tp ** 2))
            tp += slh * dq
            qmt -= dq
            dqsum -= dq
        dqsum = max(0.0, min(dqsum, cond))
        return dqsum, (dqsum / cond if cond > 0 else 0.0)

    def _call(self, t, q, cond, phase="water"):
        return gt.condensate_evaporation(
            jnp.array(t / self.PLK), jnp.array(q), self.PLK, 1.0, self.PL,
            cond, phase)

    def test_matches_fortran_loop_in_newton_interior(self):
        # condensate = 0.05 kg/kg is far more than any blend can evaporate, so
        # the result is the unclipped three-iteration solve.
        for t, q in [(295.0, 0.010), (300.0, 0.020), (288.0, 0.002),
                     (292.0, 0.014)]:
            got, fevp = self._call(t, q, 0.05)
            want, want_fevp = self._reference(t / self.PLK, q, 0.05)
            # Relative, not absolute: JAX runs float32 here while the reference
            # loop is float64, and `dq` is a difference of two similar numbers,
            # so the near-neutral cases lose several digits to cancellation.
            self.assertAlmostEqual(float(got) / want, 1.0, delta=1e-3)
            self.assertAlmostEqual(float(fevp) / want_fevp, 1.0, delta=1e-3)
            self.assertGreater(want_fevp, 0.0)   # genuinely interior
            self.assertLess(want_fevp, 1.0)

    def test_supersaturated_parcel_evaporates_nothing(self):
        # q above qsat makes every dq positive (condensation), so dqsum < 0 and
        # the lower clip fires: this routine never condenses.
        got, fevp = self._call(295.0, 0.030, 1e-3)
        self.assertEqual(float(got), 0.0)
        self.assertEqual(float(fevp), 0.0)

    def test_capped_by_available_condensate(self):
        # A very dry, warm parcel wants far more evaporation than it holds.
        cond = 1e-4
        got, fevp = self._call(300.0, 0.001, cond)
        self.assertAlmostEqual(float(got) / cond, 1.0, delta=1e-6)
        self.assertAlmostEqual(float(fevp), 1.0, delta=1e-6)

    def test_zero_condensate_is_safe(self):
        got, fevp = self._call(295.0, 0.010, 0.0)
        self.assertEqual(float(got), 0.0)
        self.assertEqual(float(fevp), 0.0)
        self.assertTrue(bool(jnp.isfinite(fevp)))

    def test_ice_phase_evaporates_more_than_water(self):
        # At the same subsaturation, qsat over ice is lower, so a parcel that is
        # subsaturated w.r.t. both evaporates less into the ice case.
        t, q, cond = 250.0, 1e-4, 0.05
        water, _ = self._call(t, q, cond, "water")
        ice, _ = self._call(t, q, cond, "ice")
        self.assertGreater(float(water), float(ice))
        self.assertAlmostEqual(
            float(ice), self._reference(t / self.PLK, q, cond, "ice")[0],
            delta=1e-8)

    def test_column_matches_vectorized(self):
        # Broadcasting-native: vertical on axis 0, trailing axes broadcast.
        t = jnp.array([295.0, 292.0, 288.0])
        q = jnp.array([0.010, 0.014, 0.002])
        cond = jnp.full((3,), 0.05)
        col = gt.condensate_evaporation(
            t / self.PLK, q, self.PLK, 1.0, self.PL, cond)[0]
        block = gt.condensate_evaporation(
            (t / self.PLK)[:, None] * jnp.ones((1, 4)),
            q[:, None] * jnp.ones((1, 4)), self.PLK, 1.0, self.PL,
            cond[:, None] * jnp.ones((1, 4)))[0]
        self.assertLess(float(jnp.max(jnp.abs(col[:, None] - block))), 1e-12)

    def test_gradient_finite_and_nonzero(self):
        def f(sm):
            return gt.condensate_evaporation(
                sm, jnp.array(0.010), self.PLK, 1.0, self.PL, 0.05)[0]
        sm = jnp.array(295.0 / self.PLK)
        grad = jax.grad(f)(sm)
        self.assertTrue(bool(jnp.isfinite(grad)))
        # Warmer parcel -> higher qsat -> more evaporation.
        self.assertGreater(float(grad), 0.0)
        check_vjp(f, lambda s: jax.vjp(f, s), (sm,), atol=1e-2, rtol=1e-2,
                  eps=1e-3)


class TestCondensation(unittest.TestCase):
    """``get_dq_cond`` (``CLOUDS_COM.F90:861``), the mirror of the evaporation.

    Shares the Newton loop with :class:`TestCondensateEvaporation`; what differs
    is the sign convention and the clip, so the tests concentrate on those.
    """

    PL = 90000.0
    PLK = (90000.0 / 100.0) ** 0.28622

    def _call(self, t, q, mass=1.0):
        return gt.condensation(jnp.array(t / self.PLK * mass), jnp.array(q),
                               self.PLK, mass, self.PL)

    def test_supersaturated_parcel_condenses(self):
        qsat = float(gt.saturation_specific_humidity(
            jnp.array(295.0), jnp.array(self.PL)))
        got, fcond = self._call(295.0, qsat * 1.5)
        self.assertGreater(float(got), 0.0)
        self.assertGreater(float(fcond), 0.0)
        self.assertLess(float(fcond), 1.0)

    def test_subsaturated_parcel_condenses_nothing(self):
        # The lower clip: this routine never evaporates.
        got, fcond = self._call(295.0, 1e-4)
        self.assertEqual(float(got), 0.0)
        self.assertEqual(float(fcond), 0.0)

    def test_cannot_condense_more_vapour_than_present(self):
        got, _ = self._call(220.0, 1e-3)   # very cold: wants to condense it all
        self.assertLessEqual(float(got), 1e-3 + 1e-12)

    def test_zero_vapour_is_safe(self):
        got, fcond = self._call(295.0, 0.0)
        self.assertEqual(float(got), 0.0)
        self.assertEqual(float(fcond), 0.0)
        self.assertTrue(bool(jnp.isfinite(fcond)))

    def test_condensation_and_evaporation_are_opposite_branches(self):
        # A parcel cannot both condense and evaporate: at any state at most one
        # of the two returns a non-zero amount.
        for t, q in [(295.0, 0.001), (295.0, 0.030), (280.0, 0.010)]:
            cond, _ = gt.condensation(
                jnp.array(t / self.PLK), jnp.array(q), self.PLK, 1.0, self.PL)
            evap, _ = gt.condensate_evaporation(
                jnp.array(t / self.PLK), jnp.array(q), self.PLK, 1.0, self.PL,
                0.05)
            self.assertEqual(float(cond) * float(evap), 0.0, msg=f"{t},{q}")

    def test_gradient_finite(self):
        qsat = float(gt.saturation_specific_humidity(
            jnp.array(295.0), jnp.array(self.PL)))

        def f(q):
            return gt.condensation(jnp.array(295.0 / self.PLK), q, self.PLK,
                                   1.0, self.PL)[0]
        grad = jax.grad(f)(jnp.array(qsat * 1.5))
        self.assertTrue(bool(jnp.isfinite(grad)))
        self.assertGreater(float(grad), 0.0)   # more vapour -> more condensate
