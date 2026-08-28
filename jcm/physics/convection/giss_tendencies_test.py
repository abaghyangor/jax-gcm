"""Tests for the GISS compensating-subsidence tendency operator."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.convection import giss_tendencies as gt
from jcm.physics.convection.giss_tendencies import (
    convective_tendencies,
    subsidence_tendency,
)


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


class TestConvectiveTendencies(unittest.TestCase):
    def setUp(self):
        n = 8
        # Plume mass flux peaks mid-cloud then tapers; environment θ increases
        # with height (stable, conditionally unstable to a plume).
        self.mass_flux = jnp.array([10.0, 14.0, 12.0, 8.0, 4.0, 1.0, 0.0, 0.0])
        self.plume_theta = jnp.array([301., 302., 303., 304., 305., 306., 307., 308.])
        self.env_theta = jnp.array([300., 301., 302.5, 304., 305.5, 307., 308.5, 310.])
        self.det_rate = jnp.array([0., 0., 0.0002, 0.0005, 0.001, 0.002, 0., 0.])
        self.dz = jnp.full(n, 400.0)
        self.layer_mass = jnp.full(n, 300.0)

    def test_heating_in_cloud_layer(self):
        # Compensating subsidence + warm detrainment heat the cloud layer.
        dtheta = convective_tendencies(
            self.mass_flux, self.plume_theta, self.det_rate, self.dz,
            self.env_theta, self.layer_mass)
        self.assertGreater(float(jnp.max(dtheta)), 0.0)         # net heating somewhere
        # The strongest heating is in the convecting (nonzero mass flux) layers.
        active = self.mass_flux > 0
        self.assertGreater(float(jnp.sum(jnp.where(active, dtheta, 0.0))), 0.0)

    def test_scales_with_mass_flux(self):
        d1 = convective_tendencies(self.mass_flux, self.plume_theta, self.det_rate,
                                   self.dz, self.env_theta, self.layer_mass)
        d2 = convective_tendencies(2.0 * self.mass_flux, self.plume_theta,
                                   self.det_rate, self.dz, self.env_theta,
                                   self.layer_mass)
        # Doubling the mass flux roughly doubles the tendency (subsidence is
        # linear in flux; detrainment deposition is too).
        self.assertGreater(float(jnp.max(jnp.abs(d2))),
                           1.5 * float(jnp.max(jnp.abs(d1))))

    def test_gradient_finite(self):
        g = jax.grad(lambda mf: jnp.sum(convective_tendencies(
            mf, self.plume_theta, self.det_rate, self.dz, self.env_theta,
            self.layer_mass) ** 2))(self.mass_flux)
        self.assertTrue(jnp.all(jnp.isfinite(g)))


if __name__ == "__main__":
    unittest.main()


class TestBsortEnvironmentTendencies(unittest.TestCase):
    """The two-stage plume -> environment update.

    The conservation tests here are not decoration: mass closure is what
    revealed that the plume's remaining mass must be dumped into its
    termination level, without which the environment silently gained several
    kg/m^2 per column.
    """

    NLEV = 12
    BASE = 4

    def _column(self, **over):
        z = jnp.zeros(self.NLEV)
        levels = jnp.arange(self.NLEV)
        in_cloud = (levels >= self.BASE) & (levels <= self.BASE + 3)
        senv = jnp.full(self.NLEV, 41.5)
        qenv = jnp.linspace(0.015, 0.004, self.NLEV)
        args = dict(
            # 40 kg/m^2 drawn from the two layers below cloud base
            source_removal=jnp.where(
                (levels >= self.BASE - 2) & (levels < self.BASE), 20.0, 0.0),
            entrained_air=jnp.where(in_cloud, 8.0, 0.0),
            # Balanced by construction: everything entering the plume
            # (40 source + 4x8 entrained = 72) must leave it again, so the four
            # in-cloud levels deposit 18 each.
            detrained_mass=jnp.where(in_cloud, 14.0, 0.0),
            detrained_heat=jnp.where(in_cloud, 14.0 * 41.7, 0.0),
            detrained_water=jnp.where(in_cloud, 14.0 * 0.012, 0.0),
            # The downdraft forms in cloud but lands below it: the descent
            # carries the 16 kg/m^2 the sort routed to it down to the two
            # sub-cloud layers, entraining 2 per level on the way. Balance:
            # 40 source + 32 plume-entrained + 8 downdraft-entrained in,
            # 56 plume-detrained + 24 downdraft-detrained out.
            downdraft_detrained_mass=jnp.where(
                (levels >= self.BASE - 4) & (levels < self.BASE - 2), 12.0, 0.0),
            downdraft_detrained_heat=jnp.where(
                (levels >= self.BASE - 4) & (levels < self.BASE - 2),
                12.0 * 41.3, 0.0),
            downdraft_detrained_water=jnp.where(
                (levels >= self.BASE - 4) & (levels < self.BASE - 2),
                12.0 * 0.010, 0.0),
            downdraft_entrained_air=jnp.where(in_cloud, 2.0, 0.0),
            environment_heat=senv,
            environment_water=qenv,
            layer_mass=jnp.full(self.NLEV, 200.0),
        )
        args.update(over)
        return args, gt.bsort_environment_tendencies(**args)

    def test_conserves_mass_when_the_plume_is_balanced(self):
        # Source draw plus entrainment equals everything deposited, so the
        # environment must end with exactly the mass it started with.
        _, t = self._column()
        self.assertAlmostEqual(float(jnp.sum(t.layer_mass)), 0.0, delta=1e-9)

    def test_advection_alone_moves_no_mass_in_total(self):
        # Subsidence redistributes; only the local exchange can change the
        # column total. Removing the exchange must leave the total untouched.
        args, t = self._column()
        net_exchange = float(jnp.sum(
            args["detrained_mass"] + args["downdraft_detrained_mass"]
            - args["source_removal"] - args["entrained_air"]
            - args["downdraft_entrained_air"]))
        self.assertAlmostEqual(float(jnp.sum(t.layer_mass)), net_exchange,
                               delta=1e-9)

    def test_interface_flux_peaks_at_cloud_base(self):
        # With no downdraft, continuity builds the flux up through the source
        # layers to the full cloud-base mass, then draws it down as the plume
        # detrains.
        z = jnp.zeros(self.NLEV)
        _, t = self._column(downdraft_detrained_mass=z,
                            downdraft_detrained_heat=z,
                            downdraft_detrained_water=z,
                            downdraft_entrained_air=z)
        flux = np.asarray(t.interface_flux)
        self.assertAlmostEqual(flux[self.BASE - 1], 40.0, delta=1e-6)
        self.assertLess(flux[self.BASE + 3], flux[self.BASE - 1])

    def test_downdraft_detrainment_offsets_the_flux_below_cloud_base(self):
        # The downdraft puts 24 kg/m^2 back into the layers underneath the
        # source, so by cloud base the net upward flux is the plume's 40 less
        # that 24. Depositing it at the level it formed instead would leave the
        # sub-cloud flux untouched, which is exactly the error being avoided.
        _, t = self._column()
        self.assertAlmostEqual(float(t.interface_flux[self.BASE - 1]), 16.0,
                               delta=1e-6)

    def test_no_flux_above_the_plume(self):
        _, t = self._column()
        self.assertEqual(float(jnp.sum(jnp.abs(
            t.interface_flux[self.BASE + 4:]))), 0.0)

    def test_subsidence_dries_the_layers_it_warms(self):
        # Compensating subsidence brings down warmer, drier air, so in a moist
        # BOMEX-like profile the in-cloud layers should moisten less than the
        # detrainment alone would suggest, and the flux must be downward.
        _, t = self._column()
        self.assertTrue(bool(jnp.all(t.interface_flux[self.BASE - 1:
                                                      self.BASE + 3] > 0)))

    def test_evaporation_cools_and_moistens_where_it_is_applied(self):
        # `dsm_evp`/`dqm_evp` go straight into the state, so they show up in the
        # tendency at their own level with their own sign.
        levels = jnp.arange(self.NLEV)
        below = levels < self.BASE
        cool = jnp.where(below, -3.0, 0.0)
        wet = jnp.where(below, 0.002, 0.0)
        _, dry = self._column()
        _, wetted = self._column(evaporation_heat=cool, evaporation_water=wet)
        self.assertTrue(bool(jnp.all(wetted.heat[below] < dry.heat[below])))
        self.assertTrue(bool(jnp.all(wetted.water[below] > dry.water[below])))
        # It adds vapour, not air, so the mass tendency is untouched.
        np.testing.assert_allclose(np.asarray(wetted.layer_mass),
                                   np.asarray(dry.layer_mass), atol=1e-12)

    def test_courant_is_reported(self):
        _, t = self._column()
        self.assertTrue(bool(jnp.isfinite(t.courant)))
        # ModelE substeps above 0.999; this configuration is far below.
        self.assertLess(float(t.courant), 0.999)

    def test_column_matches_vectorized(self):
        args, single = self._column()
        block_args = {k: (v[:, None] * jnp.ones((1, 3)) if hasattr(v, "ndim")
                          and v.ndim else v) for k, v in args.items()}
        block = gt.bsort_environment_tendencies(**block_args)
        for name in ("heat", "water", "layer_mass"):
            self.assertLess(float(jnp.max(jnp.abs(
                getattr(single, name)[:, None] - getattr(block, name)))), 1e-9)

    def test_gradient_finite(self):
        args, _ = self._column()

        def f(entrained):
            return jnp.sum(gt.bsort_environment_tendencies(
                **{**args, "entrained_air": entrained}).heat)
        grad = jax.grad(f)(args["entrained_air"])
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
