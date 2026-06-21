"""Tests for the GISS plume moist-adiabatic ascent (undilute foundation).

Self-consistency / physical tests (no oracle): the undilute saturated ascent
must cool with height but stay *warmer* than a dry adiabat from the same base
(latent heating), condense water monotonically, follow a moist-adiabatic lapse
rate, and be differentiable and broadcasting-native. Validation of the *dilute*
(entraining) plume against ModelE plume diagnostics is future work.
"""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection.giss_thermodynamics import RGAS, SHA, GRAV
from jcm.physics.convection.giss_cloud_base import dry_adiabatic_temperature
from jcm.physics.convection.giss_plume import (
    moist_adiabat_ascent,
    saturated_lapse_rate_dlnp,
)

# Cloud-base-like start (warm, moist marine cumulus) and levels above it (Pa).
_T_BASE = jnp.array(295.0)
_P_BASE = jnp.array(95000.0)
_P_ABOVE = jnp.linspace(90000.0, 50000.0, 12)


class TestSaturatedLapseRate(unittest.TestCase):
    def test_below_dry_value(self):
        # Latent heating makes |dT/dlnp| smaller than the dry value Rd*T/cp.
        dry = RGAS * _T_BASE / SHA
        moist = saturated_lapse_rate_dlnp(_T_BASE, _P_BASE)
        self.assertGreater(float(moist), 0.0)
        self.assertLess(float(moist), float(dry))


class TestMoistAdiabatAscent(unittest.TestCase):
    def setUp(self):
        self.t, self.cond = moist_adiabat_ascent(_T_BASE, _P_BASE, _P_ABOVE)

    def test_cools_with_ascent(self):
        self.assertTrue(jnp.all(jnp.diff(self.t) < 0))
        self.assertLess(float(self.t[-1]), float(_T_BASE))

    def test_warmer_than_dry_adiabat(self):
        # The defining feature: latent heat keeps the moist parcel warmer than a
        # dry-adiabatically lifted one (the difference grows with height).
        dry = dry_adiabatic_temperature(_T_BASE, _P_BASE, _P_ABOVE)
        self.assertTrue(jnp.all(self.t > dry))
        self.assertGreater(float(self.t[-1] - dry[-1]), 5.0)   # tens of K aloft

    def test_condensate_accumulates(self):
        self.assertTrue(jnp.all(jnp.diff(self.cond) >= 0))     # monotic up
        self.assertGreater(float(self.cond[-1]), 0.0)
        self.assertGreaterEqual(float(self.cond[0]), 0.0)

    def test_moist_adiabatic_lapse_rate(self):
        # Convert the first step to K/km via hydrostatic dz; expect ~4-7 K/km
        # (moist), and below the ~9.8 K/km dry value.
        t_mean = 0.5 * (float(_T_BASE) + float(self.t[0]))
        dz = (RGAS * t_mean / GRAV) * float(jnp.log(_P_BASE / _P_ABOVE[0]))
        lapse_km = (float(_T_BASE) - float(self.t[0])) / dz * 1000.0
        self.assertGreater(lapse_km, 3.0)
        self.assertLess(lapse_km, 8.0)

    def test_gradient_finite(self):
        def loss(t_base):
            t, _ = moist_adiabat_ascent(t_base, _P_BASE, _P_ABOVE)
            return jnp.sum(t)
        g = jax.grad(loss)(_T_BASE)
        self.assertTrue(jnp.isfinite(g))
        self.assertGreater(float(g), 0.0)        # warmer base -> warmer profile


class TestBroadcasting(unittest.TestCase):
    def test_column_matches_vectorized(self):
        ncols = 3
        t_base = jnp.array([295.0, 297.0, 293.0])
        p_base = jnp.full((ncols,), 95000.0)
        p_above = jnp.tile(_P_ABOVE[:, None], (1, ncols))      # (n_above, ncols)

        t_block, cond_block = moist_adiabat_ascent(t_base, p_base, p_above)
        self.assertEqual(t_block.shape, (_P_ABOVE.shape[0], ncols))
        for j in range(ncols):
            t_col, cond_col = moist_adiabat_ascent(t_base[j], p_base[j], _P_ABOVE)
            self.assertTrue(jnp.allclose(t_block[:, j], t_col, atol=1e-4))
            self.assertTrue(jnp.allclose(cond_block[:, j], cond_col, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
