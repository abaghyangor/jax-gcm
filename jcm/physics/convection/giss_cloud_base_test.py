"""Tests for the GISS cloud-base (LCL) trigger port.

Standalone-function tests in the style of the other JCM convection schemes
(e.g. tiedtke_nordeng's ``test_cloud_base``). They assert the defining property
-- the cloud base is the *first* level where a dry-adiabatically lifted parcel
saturates -- plus physical monotonicity, the no-trigger case, broadcasting, and
gradient behaviour (continuous through the thermodynamics; the index itself is a
discontinuous trigger).
"""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection import giss_thermodynamics as gt
from jcm.physics.convection.giss_cloud_base import (
    cloud_base_instability,
    cloud_base_mass_flux,
    cloud_base_triggers,
    dry_adiabatic_temperature,
    lifting_condensation_level,
)


# A descending pressure column (Pa): index 0 = surface, increasing index = up.
_P = jnp.linspace(100000.0, 50000.0, 11)


class TestDryAdiabat(unittest.TestCase):
    def test_origin_unchanged(self):
        t = dry_adiabatic_temperature(jnp.array(300.0), jnp.array(100000.0),
                                      jnp.array(100000.0))
        self.assertAlmostEqual(float(t), 300.0, places=5)

    def test_cools_with_ascent(self):
        t = dry_adiabatic_temperature(jnp.array(300.0), jnp.array(100000.0), _P)
        self.assertTrue(jnp.all(jnp.diff(t) < 0))     # cooler at lower pressure
        self.assertLess(float(t[-1]), 300.0)

    def test_gradient_finite(self):
        g = jax.grad(lambda t0: dry_adiabatic_temperature(
            t0, jnp.array(1.0e5), jnp.array(7.0e4)).sum())(jnp.array(300.0))
        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(g), 0.0)             # warmer origin -> warmer aloft


class TestLiftingCondensationLevel(unittest.TestCase):
    def _case(self, q0):
        return lifting_condensation_level(
            jnp.array(298.0), jnp.array(100000.0), jnp.array(q0), _P)

    def test_moist_parcel_condenses_first_saturated_level(self):
        cloud_base, condenses = self._case(0.014)   # 14 g/kg, ~70% RH at surface
        self.assertTrue(bool(condenses))
        k = int(cloud_base)
        self.assertGreater(k, 0)                     # not saturated at the surface
        self.assertLess(k, _P.shape[0])

        # Defining property: saturated at k, not saturated just below k.
        parcel_t = dry_adiabatic_temperature(jnp.array(298.0), jnp.array(1.0e5), _P)
        qsat = gt.saturation_specific_humidity(parcel_t, _P)
        self.assertGreater(0.014, float(qsat[k]))        # parcel q exceeds qsat at k
        self.assertLessEqual(0.014, float(qsat[k - 1]))  # ...but not at k-1

    def test_dry_parcel_never_condenses(self):
        cloud_base, condenses = self._case(1.0e-4)   # 0.1 g/kg, very dry
        self.assertFalse(bool(condenses))
        self.assertEqual(int(cloud_base), _P.shape[0])   # sentinel = nlev

    def test_drier_parcel_has_higher_cloud_base(self):
        # Less humidity -> must rise (cool) further before saturating -> larger
        # level index (lower pressure).
        cb_moist, _ = self._case(0.016)
        cb_dry, _ = self._case(0.011)
        self.assertGreater(int(cb_dry), int(cb_moist))


class TestCloudBaseInstability(unittest.TestCase):
    """DMSE trigger. Structural tests only -- not yet oracle-validated.

    Interface state: exner ~0.95, p = 850 hPa. Potential temperatures (K).
    """

    _EXNER = jnp.array(0.95)
    _P = jnp.array(85000.0)

    def test_unstable_saturated_parcel_triggers(self):
        # Warm, saturated, buoyant source parcel under a cooler/drier layer.
        trig, dmse = cloud_base_triggers(
            jnp.array(320.0), jnp.array(0.035), jnp.array(0.0),   # parcel
            jnp.array(318.0), jnp.array(0.012), jnp.array(0.0),   # layer above
            self._EXNER, self._P)
        self.assertLess(float(dmse), 0.0)        # unstable
        self.assertTrue(bool(trig))

    def test_saturated_but_stable_does_not_trigger(self):
        # Saturated parcel, but the layer above is warmer/more buoyant -> stable.
        trig, dmse = cloud_base_triggers(
            jnp.array(300.0), jnp.array(0.011), jnp.array(0.0),
            jnp.array(305.0), jnp.array(0.011), jnp.array(0.0),
            self._EXNER, self._P)
        self.assertGreater(float(dmse), 0.0)     # stable
        self.assertFalse(bool(trig))

    def test_subsaturated_parcel_does_not_trigger(self):
        # Dry parcel: fails the saturation gate regardless of DMSE.
        trig, _ = cloud_base_triggers(
            jnp.array(315.0), jnp.array(0.005), jnp.array(0.0),
            jnp.array(318.0), jnp.array(0.012), jnp.array(0.0),
            self._EXNER, self._P)
        self.assertFalse(bool(trig))

    def test_more_buoyant_parcel_is_more_unstable(self):
        def dmse(theta_parcel):
            return float(cloud_base_instability(
                jnp.array(theta_parcel), jnp.array(0.035), jnp.array(0.0),
                jnp.array(318.0), jnp.array(0.012), jnp.array(0.0),
                self._EXNER, self._P))
        self.assertLess(dmse(322.0), dmse(318.0))   # warmer parcel -> more negative

    def test_dmse_gradient_finite(self):
        # DMSE is continuous (the trigger boolean is the discontinuity).
        g = jax.grad(lambda th: cloud_base_instability(
            th, jnp.array(0.035), jnp.array(0.0),
            jnp.array(318.0), jnp.array(0.012), jnp.array(0.0),
            self._EXNER, self._P))(jnp.array(320.0))
        self.assertTrue(jnp.isfinite(g))

    def test_broadcasting(self):
        ncols = 2
        trig, dmse = cloud_base_triggers(
            jnp.array([320.0, 300.0]), jnp.array([0.035, 0.011]), jnp.zeros(ncols),
            jnp.array([318.0, 305.0]), jnp.array([0.012, 0.011]), jnp.zeros(ncols),
            jnp.full(ncols, 0.95), jnp.full(ncols, 85000.0))
        self.assertEqual(dmse.shape, (ncols,))
        self.assertTrue(bool(trig[0]))           # unstable column
        self.assertFalse(bool(trig[1]))          # stable column


class TestBroadcasting(unittest.TestCase):
    def test_column_matches_vectorized(self):
        ncols = 3
        t0 = jnp.array([298.0, 296.0, 300.0])
        p0 = jnp.full((ncols,), 100000.0)
        q0 = jnp.array([0.014, 0.011, 0.016])
        p_block = jnp.tile(_P[:, None], (1, ncols))

        cb_block, cond_block = lifting_condensation_level(t0, p0, q0, p_block)
        self.assertEqual(cb_block.shape, (ncols,))

        for j in range(ncols):
            cb_col, cond_col = lifting_condensation_level(
                t0[j], p0[j], q0[j], _P)
            self.assertEqual(int(cb_block[j]), int(cb_col))
            self.assertEqual(bool(cond_block[j]), bool(cond_col))


class TestCloudBaseMassFlux(unittest.TestCase):
    # A BOMEX-like cloud base: moist boundary-layer parcel, stable environment
    # above. theta in K, q in kg/kg, air mass ~ dp/g in kg/m^2.
    def setUp(self):
        # Strongly stratified environment so the closure bisection lands in the
        # interior (a near-neutral base would pin FPLUME at its 0/1 bounds).
        self.theta = jnp.array([298.7, 301.5, 304.5])     # source parcel, +1, +2
        self.q = jnp.array([0.0150, 0.0090, 0.0060])      # moist parcel, drier above
        self.air_mass = jnp.array([104.0, 100.0, 96.0])
        self.exner = jnp.array([0.957, 0.948])
        self.pressure = jnp.array([93000.0, 91000.0])     # Pa

    def test_fplume_physical_range(self):
        fplume, fmp2 = cloud_base_mass_flux(
            self.theta, self.q, self.air_mass, self.exner, self.pressure)
        # The bisection lives in (0, 1); a moist unstable base gives a real plume.
        self.assertGreater(float(fplume), 0.0)
        self.assertLess(float(fplume), 1.0)
        self.assertAlmostEqual(float(fmp2), float(fplume) * float(self.air_mass[0]),
                               places=4)

    def test_more_unstable_gives_larger_plume(self):
        # A moister source parcel is more unstable -> a stronger closure plume.
        f_dry, _ = cloud_base_mass_flux(
            self.theta, self.q.at[0].set(0.0150), self.air_mass,
            self.exner, self.pressure)
        f_moist, _ = cloud_base_mass_flux(
            self.theta, self.q.at[0].set(0.0190), self.air_mass,
            self.exner, self.pressure)
        self.assertGreater(float(f_moist), float(f_dry))

    def test_gradient_finite(self):
        g = jax.grad(lambda th: cloud_base_mass_flux(
            th, self.q, self.air_mass, self.exner, self.pressure)[1])(self.theta)
        self.assertTrue(jnp.all(jnp.isfinite(g)))

    def test_broadcasting(self):
        ncols = 4
        theta = jnp.tile(self.theta[:, None], (1, ncols))
        q = jnp.tile(self.q[:, None], (1, ncols))
        am = jnp.tile(self.air_mass[:, None], (1, ncols))
        ex = jnp.tile(self.exner[:, None], (1, ncols))
        pr = jnp.tile(self.pressure[:, None], (1, ncols))
        fpl, fmp2 = cloud_base_mass_flux(theta, q, am, ex, pr)
        self.assertEqual(fpl.shape, (ncols,))
        fpl0, _ = cloud_base_mass_flux(
            self.theta, self.q, self.air_mass, self.exner, self.pressure)
        self.assertTrue(jnp.allclose(fpl[0], fpl0))


if __name__ == "__main__":
    unittest.main()
