"""Tests for the buoyancy-sorting plume entrainment/detrainment port.

The regression fixtures are **taken from inside ModelE's own ascent loop** --
not regenerated from this module -- via the instrumentation in the bridge repo's
``modele_patches/mstcnv_scm_dumps.patch``. Each fixture is a real BOMEX plume
level, chosen to span the four regimes the closure can land in:

* ``CEILING``   -- entrainment pinned at the ``4e-3`` maximum,
* ``INTERIOR``  -- the ``sqrt(kew/(bdzsum*mplume)) - 1`` branch unclipped,
* ``FLOOR``     -- pinned at the ``5e-4`` buoyant floor,
* ``OVERSHOOT`` -- buoyancy integral negative, so the rate becomes detrainment
  and the blend spectrum collapses to a single pure-updraft blend.

Full-column agreement (536 levels, 1572 blends) is recorded in
``BSORT_PORT_PLAN.md``; these tests lock in a representative sample plus the
structural properties.
"""

import unittest

import jax
import jax.numpy as jnp

from jcm.physics.convection import giss_bsort as bs

# level, mplume, ent, det, ma, mplume_lag, mplume_b, gzl, delz, buoy, bdzsum,
# kew, frem
_ORACLE = {
    "ceiling": dict(
        mplume=51.78002, ent=0.004, det=0.0, ma=114.139,
        lag=51.78002, mpb=51.78002, gzl=104.5873, delz=104.5267,
        buoy=0.002612452, bdzsum=0.1472851, kew=23.11038, frem=0.418349,
        smp=2155.917, qmp=0.8344364, wmp=0.0306781,
        senv=41.55117, qenv=0.01419186, tvl=295.343,
        plk=7.047142, pres=91782.26,
        plume_out=40.94896, detrained=32.49318, downdraft=0.0,
        mixbuoy=[-0.0003713399, -4.448836e-05, 0.001291898]),
    "interior": dict(
        mplume=49.80765, ent=0.002373484, det=0.0, ma=145.2678,
        lag=51.78002, mpb=51.78002, gzl=138.7207, delz=136.0888,
        buoy=0.002362324, bdzsum=0.5112552, kew=44.57142, frem=0.3421566,
        smp=2079.185, qmp=0.749384, wmp=0.02899073,
        senv=41.67797, qenv=0.01280588, tvl=293.6729,
        plk=6.991814, pres=89289.23,
        plume_out=41.12595, detrained=25.08094, downdraft=0.0,
        mixbuoy=[-0.000454143, -0.0006783077, 0.0008232186]),
    "floor": dict(
        mplume=23.86885, ent=0.0005, det=0.0, ma=238.6542,
        lag=51.78002, mpb=51.78002, gzl=237.6995, delz=237.0469,
        buoy=0.001210612, bdzsum=1.355113, kew=39.80636, frem=0.2941929,
        smp=1005.529, qmp=0.3215634, wmp=0.01914901,
        senv=42.17594, qenv=0.008228802, tvl=290.8705,
        plk=6.862273, pres=83641.74,
        plume_out=16.8468, detrained=0.0, downdraft=9.858854,
        mixbuoy=[-0.002642248, -0.002767787, -0.001156702]),
    "overshoot": dict(
        mplume=16.8468, ent=0.0, det=0.001809418, ma=269.783,
        lag=51.78002, mpb=51.78002, gzl=276.6867, delz=276.3319,
        buoy=-0.001605852, bdzsum=-0.08094228, kew=-50.86821, frem=0.500642,
        smp=713.1072, qmp=0.2176886, wmp=0.01345424,
        senv=42.56336, qenv=0.005121686, tvl=290.4641,
        plk=6.803096, pres=81148.71,
        plume_out=8.412587, detrained=8.434217, downdraft=0.0,
        mixbuoy=[-0.001605912]),
}

_PRESSURE = 9.5e4  # Pa; only affects the (unreachable here) iplume==1 floor.


def _rates(case, iplume=2):
    o = _ORACLE[case]
    return bs.entrainment_rate(
        jnp.array(o["buoy"]), jnp.array(o["mplume"]), jnp.array(o["delz"]),
        jnp.array(o["kew"]), jnp.array(o["bdzsum"]), jnp.array(_PRESSURE),
        iplume=iplume)


class TestEntrainmentRate(unittest.TestCase):

    def test_matches_oracle_in_every_regime(self):
        for case, o in _ORACLE.items():
            ent, det = _rates(case)
            # Relative where non-zero: JAX is float32 here, the oracle dump
            # carries six significant digits.
            for got, want in ((ent, o["ent"]), (det, o["det"])):
                if want > 0.0:
                    self.assertAlmostEqual(float(got) / want, 1.0, delta=1e-5,
                                           msg=case)
                else:
                    self.assertEqual(float(got), 0.0, msg=case)

    def test_rates_are_mutually_exclusive_and_non_negative(self):
        for case in _ORACLE:
            ent, det = _rates(case)
            self.assertGreaterEqual(float(ent), 0.0)
            self.assertGreaterEqual(float(det), 0.0)
            self.assertEqual(float(ent) * float(det), 0.0, msg=case)

    def test_negative_buoyancy_integral_gives_detrainment(self):
        # The overshoot fixture has kew < 0, so the closure takes the fixed
        # -0.5/delz branch, which becomes detrainment of that magnitude.
        o = _ORACLE["overshoot"]
        ent, det = _rates("overshoot")
        self.assertEqual(float(ent), 0.0)
        self.assertAlmostEqual(float(det) / (0.5 / o["delz"]), 1.0,
                               delta=1e-6)

    def test_ceiling_is_enforced(self):
        # Drive the closure far past its ceiling with a huge energy ratio.
        ent, det = bs.entrainment_rate(
            jnp.array(0.01), jnp.array(10.0), jnp.array(100.0),
            jnp.array(1e6), jnp.array(0.01), jnp.array(_PRESSURE))
        self.assertAlmostEqual(float(ent) / bs._ENT_MAX, 1.0, delta=1e-6)
        self.assertEqual(float(det), 0.0)

    def test_floor_applies_only_where_buoyant(self):
        # Same marginal energy ratio, opposite buoyancy sign. Where buoyant the
        # 5e-4 floor lifts the rate; where not, the fixed detrainment applies.
        common = dict(plume_mass=jnp.array(30.0), layer_thickness=jnp.array(200.0),
                      pressure=jnp.array(_PRESSURE))
        ent_pos, _ = bs.entrainment_rate(
            buoyancy=jnp.array(1e-4), kew=jnp.array(1.0001),
            bdzsum=jnp.array(1.0 / 30.0), **common)
        self.assertAlmostEqual(
            float(ent_pos) / bs._ENT_FLOOR_MORE_ENTRAINING, 1.0, delta=1e-6)
        ent_neg, det_neg = bs.entrainment_rate(
            buoyancy=jnp.array(-1e-4), kew=jnp.array(-1.0),
            bdzsum=jnp.array(1.0 / 30.0), **common)
        self.assertEqual(float(ent_neg), 0.0)
        self.assertGreater(float(det_neg), 0.0)

    def test_less_entraining_plume_floor_is_pressure_gated(self):
        # iplume == 1 gets the 3e-4 floor, but only below 700 mb; above it there
        # is no floor at all, unlike iplume == 2.
        args = dict(buoyancy=jnp.array(1e-4), plume_mass=jnp.array(30.0),
                    layer_thickness=jnp.array(200.0), kew=jnp.array(1.0001),
                    bdzsum=jnp.array(1.0 / 30.0))
        low, _ = bs.entrainment_rate(pressure=jnp.array(9.5e4), iplume=1, **args)
        high, _ = bs.entrainment_rate(pressure=jnp.array(5.0e4), iplume=1, **args)
        self.assertAlmostEqual(
            float(low) / bs._ENT_FLOOR_LESS_ENTRAINING, 1.0, delta=1e-6)
        self.assertLess(float(high), bs._ENT_FLOOR_LESS_ENTRAINING)

    def test_gradients_finite_including_the_discarded_sqrt_branch(self):
        # The non-buoyant branch must not leak a NaN from sqrt of a negative
        # ratio, even though `jnp.where` discards that value.
        def f(kew):
            ent, det = bs.entrainment_rate(
                jnp.array(-1e-3), jnp.array(20.0), jnp.array(200.0), kew,
                jnp.array(-0.05), jnp.array(_PRESSURE))
            return ent + det
        grad = jax.grad(f)(jnp.array(-50.0))
        self.assertTrue(bool(jnp.isfinite(grad)))


class TestBlendAirMasses(unittest.TestCase):

    def _blend(self, case):
        o = _ORACLE[case]
        return bs.blend_air_masses(
            jnp.array(o["mplume"]), jnp.array(o["mpb"]), jnp.array(o["lag"]),
            jnp.array(o["ent"]), jnp.array(o["det"]), jnp.array(o["gzl"]),
            jnp.array(o["ma"]))

    def test_updraft_air_matches_oracle_frem(self):
        # ModelE sets aside `frem = updairm/mplume` of the plume for sorting.
        for case, o in _ORACLE.items():
            _, updraft_air, _, _, _ = self._blend(case)
            self.assertAlmostEqual(
                float(updraft_air) / (o["mplume"] * o["frem"]), 1.0,
                delta=1e-5, msg=case)

    def test_environment_air_formula(self):
        for case, o in _ORACLE.items():
            environment_air, _, _, _, _ = self._blend(case)
            expected = min(o["mplume"], 2.0 * o["mpb"]) * (o["ent"] * o["gzl"])
            expected = min(expected, o["ma"] * bs._REMRAT)
            if expected > 0.0:
                self.assertAlmostEqual(float(environment_air) / expected, 1.0,
                                       delta=1e-6, msg=case)
            else:
                self.assertEqual(float(environment_air), 0.0, msg=case)

    def test_weights_partition_the_set_aside_air(self):
        # The per-blend factors must redistribute exactly the totals, no more.
        for case in _ORACLE:
            _, _, updraft_factor, environment_factor, _ = self._blend(case)
            self.assertAlmostEqual(float(jnp.sum(updraft_factor)), 1.0,
                                   delta=1e-6, msg=case)
            env_sum = float(jnp.sum(environment_factor))
            expected = 0.0 if case == "overshoot" else 1.0
            self.assertAlmostEqual(env_sum, expected, delta=1e-6, msg=case)

    def test_overshoot_collapses_to_a_single_pure_updraft_blend(self):
        o = _ORACLE["overshoot"]
        env, upd, uf, ef, _ = self._blend("overshoot")
        self.assertEqual(float(env), 0.0)
        self.assertEqual(float(jnp.sum(ef)), 0.0)
        # All updraft weight on the first blend, none on the others.
        self.assertAlmostEqual(float(uf[0]), 1.0, delta=1e-6)
        self.assertEqual(float(jnp.sum(uf[1:])), 0.0)
        self.assertAlmostEqual(
            float(upd) / (o["mplume"] * min(0.95, o["det"] * o["gzl"])), 1.0,
            delta=1e-6)

    def test_updraft_air_never_exceeds_the_plume(self):
        # A thin plume in a deep layer would ask for more updraft air than it
        # has; ModelE clamps and scales the entrained air back by the same ratio.
        env, upd, _, _, _ = bs.blend_air_masses(
            jnp.array(0.5), jnp.array(50.0), jnp.array(50.0),
            jnp.array(4e-3), jnp.array(0.0), jnp.array(500.0), jnp.array(200.0))
        self.assertLessEqual(float(upd), 0.5)
        self.assertAlmostEqual(float(upd) / 0.5, bs._UPDRAFT_FRACTION_MAX,
                               delta=1e-6)

    def test_thinning_plume_shifts_blends_toward_complete_mixing(self):
        # refblendwt < 1 pulls fupd away from [1/4,1/2,3/4] toward fupd_fullmix,
        # which compresses the spread of the spectrum.
        common = dict(cloud_base_mass=jnp.array(50.0), entrainment=jnp.array(1e-3),
                      detrainment=jnp.array(0.0), layer_depth=jnp.array(200.0),
                      layer_mass=jnp.array(200.0))
        _, _, _, _, fresh = bs.blend_air_masses(
            plume_mass=jnp.array(50.0), plume_mass_lag=jnp.array(50.0), **common)
        _, _, _, _, thinned = bs.blend_air_masses(
            plume_mass=jnp.array(10.0), plume_mass_lag=jnp.array(50.0), **common)
        spread_fresh = float(fresh[2] - fresh[0])
        spread_thinned = float(thinned[2] - thinned[0])
        self.assertLess(spread_thinned, spread_fresh)

    def test_column_matches_vectorized(self):
        keys = ("mplume", "mpb", "lag", "ent", "det", "gzl", "ma")
        cols = {k: jnp.array([_ORACLE[c][k] for c in _ORACLE]) for k in keys}
        column = bs.blend_air_masses(*[cols[k] for k in keys])
        block = bs.blend_air_masses(
            *[cols[k][:, None] * jnp.ones((1, 3)) for k in keys])
        for a, b in zip(column, block):
            # Blend axis leads; the level axis is axis 0 of the remainder.
            self.assertLess(
                float(jnp.max(jnp.abs(a[..., None] - b))), 1e-9)


if __name__ == "__main__":
    unittest.main()


class TestSortBlends(unittest.TestCase):
    """The sorting itself, against the same real BOMEX levels.

    The four fixtures cover all three fates: ``ceiling``/``interior`` detrain
    part of the blend air, ``floor`` sends its negative blends to the downdraft,
    and ``overshoot`` has blends below the downdraft threshold that detrain
    anyway because the plume is overshooting.
    """

    def _sorted(self, case):
        o = _ORACLE[case]
        env, upd, uf, ef, _ = bs.blend_air_masses(
            jnp.array(o["mplume"]), jnp.array(o["mpb"]), jnp.array(o["lag"]),
            jnp.array(o["ent"]), jnp.array(o["det"]), jnp.array(o["gzl"]),
            jnp.array(o["ma"]))
        return o, bs.sort_blends(
            jnp.array(o["mplume"]), jnp.array(o["smp"]), jnp.array(o["qmp"]),
            jnp.array(o["wmp"]), jnp.array(o["senv"]), jnp.array(o["qenv"]),
            env, upd, uf, ef, jnp.array(o["plk"]), jnp.array(o["pres"]),
            jnp.array(o["tvl"]), jnp.array(o["buoy"])), env

    def test_mass_budget_matches_oracle(self):
        for case in _ORACLE:
            o, r, _ = self._sorted(case)
            for got, want, label in (
                    (r.plume_mass, o["plume_out"], "plume"),
                    (r.detrained_mass, o["detrained"], "detrained"),
                    (r.downdraft_mass, o["downdraft"], "downdraft")):
                if want > 0.0:
                    self.assertAlmostEqual(float(got) / want, 1.0, delta=1e-4,
                                           msg=f"{case}/{label}")
                else:
                    self.assertLess(float(got), 1e-6, msg=f"{case}/{label}")

    def test_mixture_buoyancy_matches_oracle(self):
        # Absolute, not relative: mixbuoy is a small difference of two ~290 K
        # virtual temperatures, and the decision threshold it is compared
        # against is ~1.7e-4, so absolute agreement is what matters.
        for case, o in _ORACLE.items():
            _, r, _ = self._sorted(case)
            for blend, want in enumerate(o["mixbuoy"]):
                self.assertAlmostEqual(
                    float(r.mixture_buoyancy[blend]), want, delta=5e-6,
                    msg=f"{case}/blend{blend}")

    def test_conserves_mass(self):
        # Everything entering the level -- the plume plus the entrained
        # environmental air -- must leave it via exactly one of the three fates.
        for case in _ORACLE:
            o, r, env = self._sorted(case)
            total_in = o["mplume"] + float(env)
            total_out = (float(r.plume_mass) + float(r.detrained_mass)
                         + float(r.downdraft_mass))
            self.assertAlmostEqual(total_out / total_in, 1.0, delta=1e-6,
                                   msg=case)

    def test_conserves_water_because_evaporation_is_only_a_test(self):
        # ModelE evaporates each blend purely to decide its buoyancy, then hands
        # back the untouched pre-evaporation properties. If the port applied the
        # evaporation to the returned blend instead, vapour would be created
        # here and this invariant would break.
        for case in _ORACLE:
            o, r, env = self._sorted(case)
            water_in = o["qmp"] + float(env) * o["qenv"]
            water_out = (float(r.plume_water) + float(r.detrained_water)
                         + float(r.downdraft_water))
            self.assertAlmostEqual(water_out / water_in, 1.0, delta=1e-6,
                                   msg=case)
            condensate_out = (float(r.plume_condensate)
                              + float(r.detrained_condensate)
                              + float(r.downdraft_condensate))
            # Environmental air brings no condensate.
            self.assertAlmostEqual(condensate_out / o["wmp"], 1.0, delta=1e-5,
                                   msg=case)

    def test_overshoot_guard_diverts_downdraft_air_to_detrainment(self):
        # This fixture's blend is below the downdraft threshold, so without the
        # guard it would seed a downdraft. Because the plume is overshooting it
        # must detrain instead.
        o, r, _ = self._sorted("overshoot")
        threshold = bs._NEGATIVE_BUOYANCY / o["tvl"]
        self.assertLess(o["mixbuoy"][0], threshold)   # would qualify
        self.assertEqual(float(r.downdraft_mass), 0.0)
        self.assertGreater(float(r.detrained_mass), 0.0)

    def test_buoyant_blends_rejoin_and_grow_the_plume(self):
        # The ceiling fixture has one clearly positive blend; the plume it
        # returns must exceed what was retained after the set-aside.
        o, r, _ = self._sorted("ceiling")
        retained = o["mplume"] * (1.0 - o["frem"])
        self.assertGreater(float(r.plume_mass), retained)

    def test_column_matches_vectorized(self):
        keys = ("mplume", "mpb", "lag", "ent", "det", "gzl", "ma", "smp", "qmp",
                "wmp", "senv", "qenv", "plk", "pres", "tvl", "buoy")
        v = {k: jnp.array([_ORACLE[c][k] for c in _ORACLE]) for k in keys}

        def run(get):
            env, upd, uf, ef, _ = bs.blend_air_masses(
                get("mplume"), get("mpb"), get("lag"), get("ent"), get("det"),
                get("gzl"), get("ma"))
            return bs.sort_blends(
                get("mplume"), get("smp"), get("qmp"), get("wmp"), get("senv"),
                get("qenv"), env, upd, uf, ef, get("plk"), get("pres"),
                get("tvl"), get("buoy"))

        column = run(lambda k: v[k])
        block = run(lambda k: v[k][:, None] * jnp.ones((1, 3)))
        self.assertLess(
            float(jnp.max(jnp.abs(column.plume_mass[:, None]
                                  - block.plume_mass))), 1e-4)
        self.assertLess(
            float(jnp.max(jnp.abs(column.detrained_mass[:, None]
                                  - block.detrained_mass))), 1e-4)

    def test_gradient_finite(self):
        o = _ORACLE["interior"]

        def f(plume_mass):
            env, upd, uf, ef, _ = bs.blend_air_masses(
                plume_mass, jnp.array(o["mpb"]), jnp.array(o["lag"]),
                jnp.array(o["ent"]), jnp.array(o["det"]), jnp.array(o["gzl"]),
                jnp.array(o["ma"]))
            r = bs.sort_blends(
                plume_mass, jnp.array(o["smp"]), jnp.array(o["qmp"]),
                jnp.array(o["wmp"]), jnp.array(o["senv"]), jnp.array(o["qenv"]),
                env, upd, uf, ef, jnp.array(o["plk"]), jnp.array(o["pres"]),
                jnp.array(o["tvl"]), jnp.array(o["buoy"]))
            return r.plume_mass

        grad = jax.grad(f)(jnp.array(o["mplume"]))
        self.assertTrue(bool(jnp.isfinite(grad)))
