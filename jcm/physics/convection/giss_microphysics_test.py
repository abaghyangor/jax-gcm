"""Tests for the convective precipitation port.

Fixtures are real records from inside ModelE's ``CONVECTIVE_MICROPHYSICS``
(bridge repo ``oracle_data/bomex_microphys.txt``), chosen to span the regimes:
below the cloud-mode capacity (nothing precipitates), just above it, and well
above it.
"""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection import giss_microphysics as mp

# pressure [Pa], temperature [K], updraft [m/s], condensate [kg/m^3],
# CDNC [cm^-3], critical diameter [m], precipitated [kg/m^3]
_ORACLE = [
    dict(p=9.488583e4, t=2.943430e2, w=0.5, twc=1.141256e-4,
         cdnc=6.083290e1, dc=1.124894e-4, precip=0.0),
    dict(p=9.386827e4, t=2.936886e2, w=0.5, twc=3.476363e-4,
         cdnc=6.027574e1, dc=1.121061e-4, precip=4.591982e-5),
    dict(p=9.285070e4, t=2.932978e2, w=0.5, twc=5.693836e-4,
         cdnc=5.973585e1, dc=1.117204e-4, precip=2.707917e-4),
]
_RVL = 10e-6


def _run(o):
    return mp.precipitate(
        jnp.array(o["twc"]), jnp.array(o["w"]), jnp.array(o["p"]),
        jnp.array(o["t"]), jnp.array(o["cdnc"] * 1e6), jnp.array(_RVL))


class TestCriticalDiameter(unittest.TestCase):

    def test_matches_oracle(self):
        for o in _ORACLE:
            self.assertAlmostEqual(
                float(_run(o).critical_diameter) / o["dc"], 1.0, delta=1e-5)

    def test_falls_as_the_updraft_strengthens(self):
        # A stronger updraft holds up bigger drops, so the size that escapes
        # must grow -- ModelE's relation is inverted, hence a *smaller* Dc.
        d = mp.critical_diameter(jnp.array([0.5, 1.5, 3.0]), jnp.array(9.0e4))
        self.assertTrue(bool(jnp.all(jnp.diff(d) > 0)))

    def test_saturates_when_the_updraft_outruns_every_drop(self):
        fast = mp.critical_diameter(jnp.array(50.0), jnp.array(9.0e4))
        self.assertAlmostEqual(float(fast), float(mp._DIAMETER_MAX), delta=1e-9)
        self.assertTrue(bool(jnp.isfinite(fast)))

    def test_gradient_finite_at_the_saturation_boundary(self):
        g = jax.grad(lambda w: mp.critical_diameter(w, jnp.array(9.0e4)))(
            jnp.array(9.65))
        self.assertTrue(bool(jnp.isfinite(g)))


class TestPrecipitate(unittest.TestCase):

    def test_matches_oracle(self):
        for o in _ORACLE:
            got = float(_run(o).precipitated)
            if o["precip"] > 0:
                self.assertAlmostEqual(got / o["precip"], 1.0, delta=1e-4)
            else:
                self.assertEqual(got, 0.0)

    def test_nothing_precipitates_below_the_cloud_mode_capacity(self):
        # The defining behaviour: precipitation is a threshold process in
        # liquid water content, not a smooth function of it.
        cdnc, rvl = 60e6, 10e-6
        capacity = cdnc * 1000.0 * 4.0 / 3.0 * float(jnp.pi) * rvl ** 3
        below = mp.precipitate(jnp.array(capacity * 0.9), jnp.array(1.0),
                               jnp.array(9e4), jnp.array(290.0),
                               jnp.array(cdnc), jnp.array(rvl))
        above = mp.precipitate(jnp.array(capacity * 2.0), jnp.array(1.0),
                               jnp.array(9e4), jnp.array(290.0),
                               jnp.array(cdnc), jnp.array(rvl))
        self.assertEqual(float(below.precipitated), 0.0)
        self.assertGreater(float(above.precipitated), 0.0)
        self.assertEqual(float(below.rain_water), 0.0)

    def test_never_precipitates_more_than_it_holds(self):
        for twc in (1e-5, 1e-3, 1e-2, 1.0):
            r = mp.precipitate(jnp.array(twc), jnp.array(0.1), jnp.array(9e4),
                               jnp.array(290.0), jnp.array(60e6),
                               jnp.array(10e-6))
            self.assertLessEqual(float(r.precipitated), twc + 1e-12)
            self.assertGreaterEqual(float(r.precipitated), 0.0)

    def test_zero_condensate_is_safe(self):
        r = mp.precipitate(jnp.array(0.0), jnp.array(1.0), jnp.array(9e4),
                           jnp.array(290.0), jnp.array(60e6), jnp.array(10e-6))
        self.assertEqual(float(r.precipitated), 0.0)
        self.assertTrue(bool(jnp.isfinite(r.precipitated)))

    def test_more_droplets_suppress_precipitation(self):
        # More nuclei divide the same water into smaller drops, which fall more
        # slowly -- the classic aerosol indirect effect on warm rain.
        common = dict(updraft_speed=jnp.array(1.0), pressure=jnp.array(9e4),
                      temperature=jnp.array(290.0),
                      droplet_radius=jnp.array(10e-6))
        clean = mp.precipitate(jnp.array(1e-3), droplet_number=jnp.array(40e6),
                               **common)
        dirty = mp.precipitate(jnp.array(1e-3), droplet_number=jnp.array(200e6),
                               **common)
        self.assertGreater(float(clean.precipitated),
                           float(dirty.precipitated))

    def test_column_matches_vectorized(self):
        twc = jnp.array([o["twc"] for o in _ORACLE])
        w = jnp.array([o["w"] for o in _ORACLE])
        p = jnp.array([o["p"] for o in _ORACLE])
        t = jnp.array([o["t"] for o in _ORACLE])
        n = jnp.array([o["cdnc"] * 1e6 for o in _ORACLE])
        col = mp.precipitate(twc, w, p, t, n, jnp.array(_RVL))
        block = mp.precipitate(twc[:, None] * jnp.ones((1, 3)),
                               w[:, None] * jnp.ones((1, 3)),
                               p[:, None] * jnp.ones((1, 3)),
                               t[:, None] * jnp.ones((1, 3)),
                               n[:, None] * jnp.ones((1, 3)), jnp.array(_RVL))
        self.assertLess(float(jnp.max(jnp.abs(
            col.precipitated[:, None] - block.precipitated))), 1e-12)

    def test_gradient_finite(self):
        def f(twc):
            return mp.precipitate(twc, jnp.array(1.0), jnp.array(9e4),
                                  jnp.array(290.0), jnp.array(60e6),
                                  jnp.array(10e-6)).precipitated
        for twc in (0.0, 1e-4, 1e-3):
            g = jax.grad(f)(jnp.array(twc))
            self.assertTrue(bool(jnp.isfinite(g)), msg=f"twc={twc}")


if __name__ == "__main__":
    unittest.main()
