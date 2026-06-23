"""Tests for the GISS compensating-subsidence tendency operator."""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection.giss_tendencies import subsidence_tendency


class TestSubsidenceTendency(unittest.TestCase):
    def setUp(self):
        # 6 layers, surface-first; property increases with height (like dry
        # static energy / potential temperature).
        self.prop = jnp.array([300.0, 305.0, 311.0, 318.0, 326.0, 335.0])
        self.layer_mass = jnp.full(6, 300.0)

    def test_conserves_mass_weighted_total(self):
        # Pure internal advection: the mass-weighted column total is unchanged.
        flux = jnp.array([20.0, -15.0, 25.0, -10.0, 5.0])     # (n-1,)
        d = subsidence_tendency(flux, self.prop, self.layer_mass)
        col = float(jnp.sum(d * self.layer_mass))
        self.assertAlmostEqual(col, 0.0, places=3)

    def test_downward_subsidence_warms_below(self):
        # Compensating subsidence (downward env motion, negative flux) brings
        # high-property air from above downward -> the layer below an interface
        # gains property (warming), the layer above loses it.
        flux = jnp.array([-30.0, 0.0, 0.0, 0.0, 0.0])         # down across interface 0
        d = subsidence_tendency(flux, self.prop, self.layer_mass)
        self.assertGreater(float(d[0]), 0.0)                  # layer 0 warms
        self.assertLess(float(d[1]), 0.0)                     # layer 1 cools

    def test_upwind_donor_direction(self):
        # Positive flux advects the lower (donor) layer upward: layer below the
        # interface loses, layer above gains.
        flux = jnp.array([0.0, 40.0, 0.0, 0.0, 0.0])          # up across interface 1
        d = subsidence_tendency(flux, self.prop, self.layer_mass)
        self.assertLess(float(d[1]), 0.0)                     # donor (below) loses
        self.assertGreater(float(d[2]), 0.0)                  # layer above gains

    def test_zero_flux_zero_tendency(self):
        d = subsidence_tendency(jnp.zeros(5), self.prop, self.layer_mass)
        self.assertTrue(jnp.allclose(d, 0.0))

    def test_gradient_finite(self):
        flux = jnp.array([20.0, -15.0, 25.0, -10.0, 5.0])
        g = jax.grad(lambda p: jnp.sum(
            subsidence_tendency(flux, p, self.layer_mass) ** 2))(self.prop)
        self.assertTrue(jnp.all(jnp.isfinite(g)))

    def test_broadcasting(self):
        ncols = 3
        flux = jnp.tile(jnp.array([20.0, -15.0, 25.0, -10.0, 5.0])[:, None], (1, ncols))
        prop = jnp.tile(self.prop[:, None], (1, ncols))
        lm = jnp.tile(self.layer_mass[:, None], (1, ncols))
        d = subsidence_tendency(flux, prop, lm)
        self.assertEqual(d.shape, (6, ncols))
        d_col = subsidence_tendency(flux[:, 0], self.prop, self.layer_mass)
        self.assertTrue(jnp.allclose(d[:, 0], d_col))


if __name__ == "__main__":
    unittest.main()
