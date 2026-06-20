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


if __name__ == "__main__":
    unittest.main()
