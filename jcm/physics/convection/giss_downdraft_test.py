"""Tests for the GISS convective downdraft descent.

The detrainment branch is pinned numerically. ModelE sets the detrained
fraction as ``detr = ddraft*min(1, dfac + dd_detbyent*etal_)``
(``MSTCNV.F90:4573-4585``), and in the preset this port targets
(``MSTCNV.F90:280-300``, selected because ``decks/bomex_scm.R`` sets no
``tuning_name``) ``dd_detbyent`` is zero. That collapses the expression to
three exact constants, and a dump taken from inside ModelE's own descent loop
shows exactly those three values on all 873 of its downdraft records:

===========================  ===  ===========================
branch                         n  ``detr/ddrup`` (min=med=max)
===========================  ===  ===========================
positively buoyant           377  0.75
boundary layer, non-buoyant  188  0.50
free air, non-buoyant        185  0.00
===========================  ===  ===========================

These are worth pinning because all three constants were originally *fitted*
from the oracle rather than read from the source, and because two of the
tuned presets sitting next to the active one give ``dd_detbyent`` a non-zero
value -- so a preset change would silently alter the branch.

The rest is structural: the shaft conserves what it is given, evaporation cools
and moistens it, and the routine is finite, differentiable and
broadcasting-native.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.physics.convection.giss_downdraft as dd
from jcm.physics.convection.giss_downdraft import downdraft_descent


# A four-level column, vertical on axis 0 with index 0 at the surface (ModelE's
# ``l`` convention, which is what the boundary-layer and cloud-top indices are
# expressed in). Warm/moist below, cooler/drier aloft.
_NLEV = 6
_THETA_ENV = jnp.array([300.0, 299.0, 298.0, 297.0, 296.0, 295.0])
_Q_ENV = jnp.array([0.016, 0.014, 0.011, 0.008, 0.006, 0.004])
_QCOND = jnp.zeros(_NLEV)
_MASS = jnp.full((_NLEV,), 600.0)
_DEPTH = jnp.full((_NLEV,), 500.0)
_MCFRAC = jnp.full((_NLEV,), 0.04)
_EXNER = jnp.array([0.98, 0.96, 0.94, 0.92, 0.90, 0.88])
_PRESSURE = jnp.array([95000.0, 90000.0, 85000.0, 80000.0, 75000.0, 70000.0])


def _descend(**overrides):
    """Run the descent on the reference column, with keyword overrides."""
    kwargs = dict(
        # The plume routes a blend into the shaft at level 4, near cloud top.
        source_mass=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
        source_heat=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 295.0, 0.0]),
        source_water=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 0.002, 0.0]),
        precipitation=jnp.zeros(_NLEV),
        produced_precipitation=jnp.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.0]),
        environment_heat=_THETA_ENV,
        environment_water=_Q_ENV,
        environment_condensate=_QCOND,
        layer_mass=_MASS,
        layer_depth=_DEPTH,
        convective_fraction=_MCFRAC,
        exner=_EXNER,
        pressure=_PRESSURE,
        cloud_top=jnp.array(5),
        boundary_layer_top=jnp.array(2),
        cloud_base=jnp.array(2),
    )
    kwargs.update(overrides)
    return downdraft_descent(**kwargs)


class TestDetrainmentBranch(unittest.TestCase):
    """The three branch constants, pinned against the ModelE preset."""

    def test_branch_constants_match_the_active_preset(self):
        # `1 - fddet` with FDDET = 0.25d0 (MSTCNV.F90:42).
        self.assertEqual(dd._BUOYANT_RETENTION, 0.25)
        # `1 - detfac(l)`; detfac is filled with .5d0 (MSTCNV.F90:4329).
        self.assertEqual(dd._BOUNDARY_LAYER_DETRAINMENT, 0.5)
        # Zero in the preset this port targets (MSTCNV.F90:295). Non-zero in
        # the tuned presets, hence carried explicitly rather than dropped.
        self.assertEqual(dd._DD_DETBYENT, 0.0)
        # `entcon_dd = .2d-3` (MSTCNV.F90:296), used as `etal = entcon_dd*gzl`.
        self.assertEqual(dd._DOWNDRAFT_ENTRAINMENT_RATE, 2.0e-4)

    def test_boundary_layer_sheds_exactly_half(self):
        """A cold, non-buoyant shaft halves through every boundary-layer level.

        This is the profile the oracle shows: ModelE carries the shaft down to
        the boundary layer intact, then `detr/ddin` = 0.5 at each level in it.
        """
        # A very cold blend stays negatively buoyant the whole way down.
        cold = _descend(
            source_mass=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
            source_heat=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 280.0, 0.0]),
            source_water=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 0.002, 0.0]),
            produced_precipitation=jnp.zeros(_NLEV),
        )
        entering = np.asarray(cold.mass)
        detrained = np.asarray(cold.detrained_mass)
        # Levels 1 and 2 are inside the boundary layer (`boundary_layer_top=2`)
        # and are exchanging levels (`0 < level < cloud_top`).
        for level in (1, 2):
            self.assertGreater(entering[level], 0.0)
            self.assertAlmostEqual(detrained[level] / entering[level], 0.5,
                                   places=6)

    def test_buoyant_shaft_sheds_three_quarters(self):
        """Where the shaft is warmer than its surroundings it sheds 75%."""
        # A blend warmer than the environment is buoyant immediately, and with
        # no rain to evaporate it stays that way.
        warm = _descend(
            source_mass=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
            source_heat=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 320.0, 0.0]),
            source_water=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 0.010, 0.0]),
            produced_precipitation=jnp.zeros(_NLEV),
        )
        entering = np.asarray(warm.mass)
        detrained = np.asarray(warm.detrained_mass)
        # Level 4 is where the blend enters, level 3 the first full level below;
        # both sit above the boundary layer, so 0.75 can only come from the
        # buoyant branch.
        self.assertGreater(entering[3], 0.0)
        self.assertAlmostEqual(detrained[3] / entering[3], 0.75, places=6)

    def test_free_air_non_buoyant_sheds_nothing(self):
        """Above the boundary layer a negatively buoyant shaft detrains 0."""
        cold = _descend(
            source_mass=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0]),
            source_heat=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 280.0, 0.0]),
            source_water=jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 0.002, 0.0]),
            produced_precipitation=jnp.zeros(_NLEV),
        )
        # Levels 3 and 4 are above `boundary_layer_top=2` and the shaft is cold,
        # so neither branch fires and the entering mass passes straight through.
        detrained = np.asarray(cold.detrained_mass)
        self.assertAlmostEqual(detrained[3], 0.0, places=10)


class TestDescentStructure(unittest.TestCase):
    def test_mass_is_conserved(self):
        """Everything put in leaves as detrainment, entrainment aside.

        The shaft is closed: source in, detrainment out, with entrained air
        passing through. The leftover dump at `ldmin` is what makes this exact
        rather than merely close.
        """
        out = _descend()
        supplied = float(jnp.sum(out.entrained_air)) + 10.0
        self.assertAlmostEqual(float(jnp.sum(out.detrained_mass)), supplied,
                               places=4)

    def test_evaporation_cools_and_moistens(self):
        """Rain falling into the shaft lowers its theta and raises its q."""
        dry = _descend(produced_precipitation=jnp.zeros(_NLEV))
        wet = _descend(produced_precipitation=jnp.full((_NLEV,), 0.5))
        self.assertGreater(float(jnp.sum(wet.evaporated)),
                           float(jnp.sum(dry.evaporated)))
        # Compare at a level both shafts reach, below where the blend enters.
        self.assertLess(float(wet.potential_temperature[3]),
                        float(dry.potential_temperature[3]))
        self.assertGreater(float(wet.specific_humidity[3]),
                           float(dry.specific_humidity[3]))

    def test_nothing_happens_above_cloud_top(self):
        out = _descend()
        top = np.asarray(out.mass)[5]
        self.assertEqual(top, 0.0)


class TestNumerics(unittest.TestCase):
    def test_outputs_are_finite(self):
        for field in _descend():
            self.assertTrue(bool(jnp.all(jnp.isfinite(field))))

    def test_gradient_is_finite(self):
        """The shaft is full of guarded divisions and fractional powers.

        `x/max(d,tiny)` and `x**0.6` are both finite in value at zero and
        non-finite in derivative, so a value-only check would pass while the
        gradient NaNs. Take the gradient through an empty *and* an active shaft:
        the rain-free levels are the ones that trip the fractional power.
        """
        def loss(source_heat):
            out = _descend(source_heat=source_heat)
            return jnp.sum(out.detrained_heat) + jnp.sum(out.evaporated)

        source_heat = jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 295.0, 0.0])
        grad = jax.grad(loss)(source_heat)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

        # An entirely absent downdraft: every level is degenerate at once.
        empty = jax.grad(loss)(jnp.zeros(_NLEV))
        self.assertTrue(bool(jnp.all(jnp.isfinite(empty))))

    def test_broadcasting_matches_single_column(self):
        """A (nlev, ncols) block must agree column-by-column with (nlev,)."""
        single = _descend()

        ncols = 3
        def tile(x):
            return jnp.broadcast_to(x[:, None], (_NLEV, ncols))

        block = downdraft_descent(
            source_mass=tile(jnp.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])),
            source_heat=tile(jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 295.0,
                                        0.0])),
            source_water=tile(jnp.array([0.0, 0.0, 0.0, 0.0, 10.0 * 0.002,
                                         0.0])),
            precipitation=jnp.zeros((_NLEV, ncols)),
            produced_precipitation=tile(
                jnp.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.0])),
            environment_heat=tile(_THETA_ENV),
            environment_water=tile(_Q_ENV),
            environment_condensate=tile(_QCOND),
            layer_mass=tile(_MASS),
            layer_depth=tile(_DEPTH),
            convective_fraction=tile(_MCFRAC),
            exner=tile(_EXNER),
            pressure=tile(_PRESSURE),
            cloud_top=jnp.full((ncols,), 5),
            boundary_layer_top=jnp.full((ncols,), 2),
            cloud_base=jnp.full((ncols,), 2),
        )
        for one, many in zip(single, block):
            for col in range(ncols):
                np.testing.assert_allclose(np.asarray(many)[:, col],
                                           np.asarray(one), rtol=1e-6,
                                           atol=1e-9)


if __name__ == "__main__":
    unittest.main()
