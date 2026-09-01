"""Tests for the single-plume driver.

The pieces the driver composes have their own oracle-checked tests; what is
tested here is the wiring and the profiles the driver derives from the column,
which nothing else covers.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.convection import giss_plume_driver as drv
from jcm.physics.convection.giss_thermodynamics import DELTX, LHE, RGAS, SHA

jax.config.update("jax_enable_x64", True)


def _column(nlev=20, horiz=()):
    """A conditionally unstable trade-cumulus column, surface-first.

    Mixed layer up to level 4, a weak stable lapse above. A parcel lifted from
    the mixed layer is subsaturated at the surface and condenses around 920 mb,
    which is what gives the plume something to rise on -- a column without that
    structure simply kills the plume at its base and tests nothing downstream.
    """
    def profile(values):
        return jnp.broadcast_to(
            jnp.asarray(values).reshape((nlev,) + (1,) * len(horiz)),
            (nlev,) + horiz)

    level = np.arange(nlev, dtype=float)
    pressure = profile((1000.0 - 20.0 * level) * 100.0)
    # ModelE pairs `plk = (p in mb)**kappa` with `th = theta/1000**kappa`, so
    # that `th*plk` is the actual temperature. Building the column from a
    # conventional potential temperature means dividing it back out; leaving it
    # in gives temperatures near 2000 K, outside the Murphy & Koop saturation
    # fit's validity range, and the saturation vapour pressure goes non-finite.
    exner = (pressure / 100.0) ** 0.28622
    # Mixed layer, a conditionally unstable cloud layer, then a trade inversion
    # at level 11 that stops the plume. Without something to stop it the plume
    # rises to the top of the model, which is not what a real column does and
    # leaves the test measuring the top-boundary handling instead of the physics.
    theta = profile((298.5 + 0.30 * np.clip(level - 4.0, 0.0, 7.0)
                     + 3.0 * np.maximum(0.0, level - 11.0))
                    / 1000.0 ** 0.28622)
    q = profile(np.maximum(0.0165 - 0.0006 * level, 1.0e-5))
    layer_mass = profile(np.full(nlev, 20.0 * 100.0 / 9.80665))
    height = profile(190.0 * level + 100.0)
    return dict(potential_temperature=theta, specific_humidity=q,
                layer_mass=layer_mass, exner=exner, pressure=pressure,
                height=height)


class ColumnGeometryTest(unittest.TestCase):

    def test_layer_depth_is_a_centred_difference(self):
        col = _column()
        g = drv.column_geometry(**col)
        z = np.asarray(col["height"])
        depth = np.asarray(g.layer_depth)
        # Interior: half the span between the layer centres either side.
        for l in range(1, len(z) - 1):
            self.assertAlmostEqual(depth[l], 0.5 * (z[l + 1] - z[l - 1]),
                                   places=9)

    def test_layer_depth_ends_follow_the_fortran(self):
        # CLOUDS_DRV.F90:584-585 zeroes the bottom layer and repeats the top.
        g = drv.column_geometry(**_column())
        depth = np.asarray(g.layer_depth)
        self.assertEqual(depth[0], 0.0)
        self.assertAlmostEqual(depth[-1], depth[-2], places=9)

    def test_layer_thickness_is_hydrostatic(self):
        col = _column()
        g = drv.column_geometry(**col)
        density = np.asarray(col["pressure"]) / (
            RGAS * np.asarray(col["potential_temperature"] * col["exner"]))
        np.testing.assert_allclose(
            np.asarray(g.layer_thickness),
            np.asarray(col["layer_mass"]) / density, rtol=1e-12)

    def test_virtual_temperature_carries_the_humidity_term(self):
        col = _column()
        g = drv.column_geometry(**col)
        t = np.asarray(col["potential_temperature"] * col["exner"])
        np.testing.assert_allclose(
            np.asarray(g.virtual_temperature),
            t * (1.0 + DELTX * np.asarray(col["specific_humidity"])),
            rtol=1e-12)


class SourceWeightsTest(unittest.TestCase):

    NLEV = 20

    def _weights(self, source_bottom, source_top, boundary_layer_top,
                 layer_mass=None):
        if layer_mass is None:
            layer_mass = jnp.full(self.NLEV, 460.0)
        return np.asarray(drv.source_weights(
            layer_mass, jnp.array(source_bottom), jnp.array(source_top),
            jnp.array(boundary_layer_top)))

    def test_weights_sum_to_one_over_the_source_block(self):
        w = self._weights(0, 6, 8)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        self.assertEqual(np.count_nonzero(w), 7)

    def test_uniform_layers_give_uniform_weights(self):
        w = self._weights(0, 6, 8)
        np.testing.assert_allclose(w[:7], 1.0 / 7.0, rtol=1e-12)

    def test_weights_follow_layer_mass(self):
        mass = jnp.arange(1.0, self.NLEV + 1.0)
        w = self._weights(0, 3, 8, layer_mass=mass)
        # 1:2:3:4 by mass.
        np.testing.assert_allclose(w[:4], np.array([1., 2., 3., 4.]) / 10.0,
                                   rtol=1e-12)

    def test_layers_above_the_boundary_layer_are_dropped(self):
        # Source spans 0..8 but the mixed layer tops out at 6, so 7 and 8 go.
        w = self._weights(0, 8, 6)
        self.assertEqual(np.count_nonzero(w), 7)
        self.assertEqual(w[7], 0.0)
        self.assertEqual(w[8], 0.0)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)

    def test_no_drop_when_the_source_starts_above_the_boundary_layer(self):
        # `lmin0 > dcl`: the parcel was never in the mixed layer, so nothing is
        # dropped and the full block contributes.
        w = self._weights(8, 11, 6)
        self.assertEqual(np.count_nonzero(w), 4)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)


class SourceParcelTest(unittest.TestCase):

    def test_removal_sums_to_the_plume_mass(self):
        col = _column()
        removal, _, _, _ = drv.source_parcel(
            col["potential_temperature"], col["specific_humidity"],
            col["layer_mass"], jnp.array(50.0), jnp.array(0), jnp.array(6),
            jnp.array(8))
        self.assertAlmostEqual(float(jnp.sum(removal)), 50.0, places=10)

    def test_removal_is_confined_to_the_source_layers(self):
        col = _column()
        removal, _, _, _ = drv.source_parcel(
            col["potential_temperature"], col["specific_humidity"],
            col["layer_mass"], jnp.array(50.0), jnp.array(0), jnp.array(6),
            jnp.array(8))
        self.assertEqual(float(jnp.sum(jnp.abs(removal[7:]))), 0.0)

    def test_parcel_starts_with_no_condensate(self):
        # It condenses on arrival at cloud base, not in the sub-cloud layers.
        col = _column()
        _, _, _, condensate = drv.source_parcel(
            col["potential_temperature"], col["specific_humidity"],
            col["layer_mass"], jnp.array(50.0), jnp.array(0), jnp.array(6),
            jnp.array(8))
        self.assertEqual(float(condensate), 0.0)

    def test_parcel_is_the_mass_weighted_source_air(self):
        col = _column()
        removal, heat, water, _ = drv.source_parcel(
            col["potential_temperature"], col["specific_humidity"],
            col["layer_mass"], jnp.array(50.0), jnp.array(0), jnp.array(6),
            jnp.array(8))
        theta = np.asarray(col["potential_temperature"])
        self.assertAlmostEqual(
            float(heat), float(np.sum(np.asarray(removal) * theta)), places=9)
        # The parcel's mean properties sit inside the source layers' range.
        self.assertGreaterEqual(float(heat) / 50.0, theta[:7].min())
        self.assertLessEqual(float(heat) / 50.0, theta[:7].max())
        del water


class ConvectiveFractionTest(unittest.TestCase):

    NLEV = 20

    def _fraction(self, **over):
        level = jnp.arange(self.NLEV)
        args = dict(
            plume_mass=jnp.where((level >= 6) & (level <= 12), 40.0, 0.0),
            vertical_velocity=jnp.where((level >= 6) & (level <= 12), 2.0, 0.0),
            density=jnp.full(self.NLEV, 1.1),
            cloud_base=jnp.array(6),
            cloud_top=jnp.array(12),
            timestep=jnp.array(1800.0),
        )
        args.update(over)
        return np.asarray(drv.convective_fraction(**args))

    def test_below_cloud_base_holds_the_cloud_base_value(self):
        # Virga: the fraction does not fall to zero under the cloud.
        f = self._fraction()
        self.assertGreater(f[6], 0.0)
        np.testing.assert_allclose(f[:6], f[6], rtol=1e-12)

    def test_zero_above_cloud_top(self):
        f = self._fraction()
        self.assertEqual(float(np.sum(np.abs(f[12:]))), 0.0)

    def test_capped_at_one(self):
        # A tiny updraft speed would otherwise send the ratio far above 1.
        level = jnp.arange(self.NLEV)
        f = self._fraction(vertical_velocity=jnp.where(
            (level >= 6) & (level <= 12), 1.0e-4, 0.0))
        self.assertLessEqual(f.max(), 1.0)

    def test_no_nan_where_the_plume_is_absent(self):
        f = self._fraction()
        self.assertTrue(np.all(np.isfinite(f)))


class RunPlumeTest(unittest.TestCase):

    def _run(self, horiz=(), **over):
        col = _column(horiz=horiz)
        args = dict(
            cloud_base=jnp.array(4), source_bottom=jnp.array(0),
            source_top=jnp.array(3), boundary_layer_top=jnp.array(3),
            cloud_base_mass=jnp.array(50.0), timestep=jnp.array(1800.0), **col)
        args.update(over)
        return drv.run_plume(**args)

    def test_runs_and_returns_finite_fields(self):
        r = self._run()
        for field in (r.tendency.heat, r.tendency.water, r.tendency.layer_mass,
                      r.ascent.plume_mass, r.descent.mass,
                      r.convective_fraction):
            self.assertTrue(bool(jnp.all(jnp.isfinite(field))))

    def test_source_removal_matches_the_cloud_base_mass(self):
        r = self._run()
        self.assertAlmostEqual(float(jnp.sum(r.source_removal)), 50.0,
                               places=10)

    def test_conserves_column_mass(self):
        # Everything drawn out of the environment is handed back to it, so the
        # column's air mass is unchanged. Advection only moves it around.
        r = self._run()
        self.assertAlmostEqual(float(jnp.sum(r.tendency.layer_mass)), 0.0,
                               delta=1e-8)

    def test_plume_rises_above_its_base(self):
        r = self._run()
        self.assertGreater(int(r.cloud_top), 4)

    def test_gradient_is_finite(self):
        col = _column()

        def loss(theta):
            r = drv.run_plume(
                cloud_base=jnp.array(4), source_bottom=jnp.array(0),
                source_top=jnp.array(3), boundary_layer_top=jnp.array(3),
                cloud_base_mass=jnp.array(50.0), timestep=jnp.array(1800.0),
                potential_temperature=theta,
                specific_humidity=col["specific_humidity"],
                layer_mass=col["layer_mass"], exner=col["exner"],
                pressure=col["pressure"], height=col["height"])
            return jnp.sum(r.tendency.heat ** 2)

        grad = jax.grad(loss)(col["potential_temperature"])
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertGreater(float(jnp.sum(jnp.abs(grad))), 0.0)

    def test_column_matches_broadcast_block(self):
        # The same code must run unchanged on a (kx,) column and a (kx, ncols)
        # block, per the repo's broadcasting-native convention.
        single = self._run()
        block = self._run(horiz=(3,))
        for a, b in ((single.tendency.heat, block.tendency.heat),
                     (single.tendency.water, block.tendency.water),
                     (single.ascent.plume_mass, block.ascent.plume_mass),
                     (single.descent.mass, block.descent.mass)):
            # `atol` covers the entries that are exactly zero in the column and
            # land on XLA's vectorised rounding in the block; 1e-9 kg/m^2 is far
            # below anything the scheme resolves.
            np.testing.assert_allclose(
                np.asarray(b), np.broadcast_to(np.asarray(a)[:, None], b.shape),
                rtol=1e-10, atol=1e-9)


class SourceBottomTest(unittest.TestCase):
    """`lmin0`: how deep a slab of boundary layer a plume may draw from."""

    def _bottom(self, mb_per_layer, source_top, nlev=20):
        pressure = jnp.asarray(
            (1000.0 - mb_per_layer * np.arange(nlev, dtype=float)) * 100.0)
        return int(drv.source_bottom(pressure, jnp.array(source_top)))

    def test_a_shallow_column_draws_from_the_surface(self):
        # 6 layers of 20 mb is 120 mb, well inside the limit.
        self.assertEqual(self._bottom(20.0, 6), 0)

    def test_a_deep_column_is_cut_off_at_300_mb(self):
        # 60 mb layers: only levels within 300 mb of the base qualify.
        self.assertEqual(self._bottom(60.0, 6), 2)

    def test_the_base_layer_always_qualifies(self):
        # Its own span is zero, so the source can never come out empty.
        self.assertLessEqual(self._bottom(200.0, 4), 4)


class ConvectiveColumnTest(unittest.TestCase):
    """The sweep over candidate cloud bases, with the environment carried."""

    NLEV = 20

    def _run(self, bases=(4, 6), horiz=(), **over):
        col = _column(horiz=horiz)
        level = jnp.arange(self.NLEV).reshape(
            (self.NLEV,) + (1,) * len(horiz))
        mass = jnp.zeros_like(col["layer_mass"])
        for b in bases:
            mass = jnp.where(level == b, 40.0, mass)
        args = dict(cloud_base_mass=mass, boundary_layer_top=jnp.array(3),
                    highest_base=jnp.array(6), timestep=jnp.array(1800.0),
                    max_plumes=6, **col)
        args.update(over)
        return col, drv.convective_column(**args)

    def test_runs_and_is_finite(self):
        _, r = self._run()
        for field in (r.heat, r.water, r.mass_flux, r.precipitation,
                      r.potential_temperature, r.specific_humidity):
            self.assertTrue(bool(jnp.all(jnp.isfinite(field))))

    def test_counts_the_plumes_that_convect(self):
        _, r = self._run(bases=(4, 6))
        self.assertEqual(int(r.plume_count), 2)
        _, one = self._run(bases=(6,))
        self.assertEqual(int(one.plume_count), 1)

    def test_candidates_outside_the_sweep_do_not_fire(self):
        # Level 1 is below `dcl` and level 9 above `highest_base`, so neither
        # convects however much mass the closure hands them.
        _, r = self._run(bases=(1, 9))
        self.assertEqual(int(r.plume_count), 0)
        self.assertEqual(float(jnp.sum(jnp.abs(r.heat))), 0.0)

    def test_extra_candidates_are_no_ops(self):
        # The trip count is static and set by `max_plumes`; sweeping more
        # candidates than can convect must not change the answer.
        _, few = self._run(max_plumes=6)
        _, many = self._run(max_plumes=self.NLEV)
        np.testing.assert_allclose(np.asarray(many.heat), np.asarray(few.heat),
                                   rtol=1e-12, atol=1e-10)

    def test_plumes_are_coupled_through_the_environment(self):
        # This is the whole point of the sweep: each plume ascends through what
        # its predecessors left behind, so the sequence is not the sum of the
        # parts. If these ever agree, the environment is not being carried.
        _, both = self._run(bases=(4, 6))
        _, upper = self._run(bases=(6,))
        _, lower = self._run(bases=(4,))
        independent = np.asarray(upper.heat) + np.asarray(lower.heat)
        self.assertGreater(
            np.abs(independent - np.asarray(both.heat)).max(), 1.0)

    def test_energy_closes_against_the_water_removed(self):
        # The column's dry static energy gain is the latent heat of the water
        # convection took out of it. Any mismatch beyond the condensate still
        # aloft means heat or water is being created somewhere.
        col, r = self._run()
        sensible = float(jnp.sum(r.heat * col["exner"]))
        latent = (LHE / SHA) * -float(jnp.sum(r.water))
        self.assertAlmostEqual(sensible / latent, 1.0, delta=0.02)

    def test_convection_warms_aloft_and_cools_below_cloud_base(self):
        col, r = self._run()
        dtheta = np.asarray(r.heat / col["layer_mass"])
        self.assertLess(dtheta[:4].max(), 0.0)     # sub-cloud cooling
        self.assertGreater(dtheta[7:14].max(), 0.0)   # cloud-layer heating

    def test_gradient_is_finite(self):
        col = _column()
        level = jnp.arange(self.NLEV)
        mass = jnp.where((level == 4) | (level == 6), 40.0, 0.0)

        def loss(theta):
            r = drv.convective_column(
                potential_temperature=theta,
                specific_humidity=col["specific_humidity"],
                layer_mass=col["layer_mass"], exner=col["exner"],
                pressure=col["pressure"], height=col["height"],
                cloud_base_mass=mass, boundary_layer_top=jnp.array(3),
                highest_base=jnp.array(6), timestep=jnp.array(1800.0),
                max_plumes=6)
            return jnp.sum(r.heat ** 2)

        grad = jax.grad(loss)(col["potential_temperature"])
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertGreater(float(jnp.sum(jnp.abs(grad))), 0.0)

    def test_column_matches_broadcast_block(self):
        _, single = self._run()
        _, block = self._run(horiz=(3,))
        for a, b in ((single.heat, block.heat), (single.water, block.water),
                     (single.mass_flux, block.mass_flux)):
            np.testing.assert_allclose(
                np.asarray(b), np.broadcast_to(np.asarray(a)[:, None], b.shape),
                rtol=1e-10, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
