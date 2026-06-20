"""Tests for the GISS cloud-base mass-flux closure (nlpi=1 port).

Structural tests only -- the closure *magnitude* is not yet validated against a
convectively active oracle. They assert the defining behaviour: the bisection
drives the residual cloud-base instability ``DMSE1`` toward neutral, a more
unstable column needs a larger mass flux, and the routine is finite/
differentiable and broadcasting-native.
"""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection.giss_cloud_base import cloud_base_instability
from jcm.physics.convection.giss_mass_flux import cloud_base_mass_flux


# A convectively unstable cloud-base stack (potential temperatures in K).
# Source layer warm + saturated; cooler/drier layers above.
_UNSTABLE = dict(
    theta_dn=jnp.array(320.0), q_dn=jnp.array(0.035),
    theta_up=jnp.array(318.0), q_up=jnp.array(0.012),
    theta_up2=jnp.array(316.0), q_up2=jnp.array(0.008),
    mass1=jnp.array(600.0), mass2=jnp.array(600.0), mass3=jnp.array(600.0),
    exner1=jnp.array(0.96), exner2=jnp.array(0.95),
    pressure1=jnp.array(87000.0), pressure2=jnp.array(85000.0),
)


class TestCloudBaseMassFlux(unittest.TestCase):
    def test_closure_neutralizes_instability(self):
        # Instability with no mass flux (fplume=0) == cloud_base_instability.
        dmse_initial = float(cloud_base_instability(
            _UNSTABLE["theta_dn"], _UNSTABLE["q_dn"], jnp.array(0.0),
            _UNSTABLE["theta_up"], _UNSTABLE["q_up"], jnp.array(0.0),
            _UNSTABLE["exner2"], _UNSTABLE["pressure2"]))
        self.assertLess(dmse_initial, 0.0)               # genuinely unstable

        fplume, fmp2, dmse_final = cloud_base_mass_flux(**_UNSTABLE)
        # The closure moves the cloud base toward neutral.
        self.assertLess(abs(float(dmse_final)), abs(dmse_initial))
        self.assertGreater(float(fplume), 0.0)           # nonzero mass flux
        self.assertAlmostEqual(float(fmp2), float(fplume) * 600.0, places=4)

    def test_more_unstable_needs_more_mass_flux(self):
        fplume_a, _, _ = cloud_base_mass_flux(**_UNSTABLE)
        stronger = dict(_UNSTABLE)
        stronger["theta_dn"] = jnp.array(323.0)          # warmer, more buoyant
        fplume_b, _, _ = cloud_base_mass_flux(**stronger)
        self.assertGreater(float(fplume_b), float(fplume_a))

    def test_finite_and_differentiable(self):
        def loss(theta_dn):
            args = dict(_UNSTABLE)
            args["theta_dn"] = theta_dn
            fplume, _, _ = cloud_base_mass_flux(**args)
            return fplume
        g = jax.grad(loss)(jnp.array(320.0))
        self.assertTrue(jnp.isfinite(g))
        self.assertTrue(jnp.isfinite(loss(jnp.array(320.0))))

    def test_broadcasting(self):
        ncols = 2
        args = {k: jnp.broadcast_to(v, (ncols,)) for k, v in _UNSTABLE.items()}
        fplume, fmp2, dmse = cloud_base_mass_flux(**args)
        self.assertEqual(fplume.shape, (ncols,))
        # Both columns identical here -> identical result.
        self.assertAlmostEqual(float(fplume[0]), float(fplume[1]), places=6)


if __name__ == "__main__":
    unittest.main()
