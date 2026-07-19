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

from jcm.physics.convection.giss_cloud_base import cloud_base_instability
from jcm.physics.convection.giss_mass_flux import cloud_base_mass_flux


# A convectively unstable cloud-base stencil (vertical on axis 0):
# [lmin, lmin+1, lmin+2]. Source layer warm + moist; cooler/drier above.
_THETA = jnp.array([320.0, 318.0, 316.0])
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
