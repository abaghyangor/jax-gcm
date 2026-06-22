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
from jcm.physics.convection.giss_thermodynamics import (
    LHE, SHA, saturation_specific_humidity,
)
from jcm.physics.convection.giss_plume import (
    _saturation_adjust,
    entraining_plume_ascent,
    entrainment_rate,
    moist_adiabat_ascent,
    saturated_lapse_rate_dlnp,
    updraft_velocity,
)


def _conditionally_unstable_sounding(n=16):
    """Build a conditionally unstable environment above cloud base with a cap.

    Returns env T/vapour, geopotential, pressure, dz, and the cloud-base parcel.
    """
    from jcm.physics.convection.giss_thermodynamics import RGAS, GRAV
    p_base = jnp.array(95000.0)
    p = jnp.linspace(93000.0, 68000.0, n)
    # Hydrostatic thicknesses / heights from the base (mean T ~285 K).
    edges = jnp.concatenate([p_base[None], p])
    dz = (RGAS * 285.0 / GRAV) * (-jnp.diff(jnp.log(edges)))
    height = jnp.cumsum(dz)                       # m above cloud base
    phi = GRAV * height                           # geopotential [m^2/s^2]
    # Env lapse 6.5 K/km (between moist and dry adiabats -> conditionally
    # unstable), with a strong warm inversion (cap) above ~2.5 km.
    env_t = 290.0 - 6.5e-3 * height
    env_t = jnp.where(height > 2500.0, env_t + 12.0, env_t)
    env_q = 0.6 * saturation_specific_humidity(env_t, p)
    # Cloud-base parcel: saturated, slightly warmer than the environment base.
    t_base = jnp.array(290.5)
    q_base = saturation_specific_humidity(t_base, p_base)
    return dict(t_base=t_base, q_base=q_base, geopotential_base=jnp.array(0.0),
                p_base=p_base, w_base=jnp.array(1.0),
                env_temperature=env_t, env_vapor=env_q, geopotential=phi,
                pressure=p, layer_thickness=dz)

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


class TestEntrainmentRate(unittest.TestCase):
    def test_positive_for_buoyant_plume(self):
        self.assertGreater(
            float(entrainment_rate(jnp.array(0.002), jnp.array(2.0))), 0.0)

    def test_decreases_with_updraft_speed(self):
        # Buoyancy-sorting: faster updrafts entrain less (~1/w^2).
        slow = float(entrainment_rate(jnp.array(0.002), jnp.array(1.0)))
        fast = float(entrainment_rate(jnp.array(0.002), jnp.array(4.0)))
        self.assertGreater(slow, fast)

    def test_scales_with_contce(self):
        less = float(entrainment_rate(jnp.array(0.002), jnp.array(2.0), 0.4))
        more = float(entrainment_rate(jnp.array(0.002), jnp.array(2.0), 0.6))
        self.assertGreater(more, less)


class TestUpdraftVelocity(unittest.TestCase):
    def setUp(self):
        # Buoyant cloud layer (8 levels) then a strong stable cap (4 levels)
        # deep enough to decelerate the updraft back to zero within the domain.
        self.buoy = jnp.concatenate([jnp.full(8, 0.003), jnp.full(4, -0.012)])
        self.dz = jnp.full(12, 400.0)
        self.w_base = jnp.array(1.0)

    def test_w2_rises_then_falls_and_caps(self):
        w2, w, top = updraft_velocity(self.buoy, self.dz, self.w_base)
        self.assertGreater(float(w2[5]), float(w2[0]))        # grows in cloud layer
        self.assertGreater(float(w[5]), 1.0)                  # several m/s updraft
        top = int(top)
        self.assertGreaterEqual(top, 8)                       # top is in the cap
        self.assertLess(top, 12)                              # plume stopped in range
        self.assertLessEqual(float(w2[top]), 0.0)             # plume stopped

    def test_entrainment_weakens_updraft(self):
        # No entrainment (contce=0) gives a stronger updraft than the
        # entraining plume (the buoyancy-sorting drag removes kinetic energy).
        w2_dry, _, _ = updraft_velocity(self.buoy, self.dz, self.w_base, contce=0.0)
        w2_ent, _, _ = updraft_velocity(self.buoy, self.dz, self.w_base, contce=0.6)
        self.assertGreater(float(w2_dry[5]), float(w2_ent[5]))

    def test_more_buoyant_stronger_updraft(self):
        w2_a, _, _ = updraft_velocity(self.buoy, self.dz, self.w_base)
        w2_b, _, _ = updraft_velocity(self.buoy * 1.5, self.dz, self.w_base)
        self.assertGreater(float(w2_b[5]), float(w2_a[5]))

    def test_gradient_finite(self):
        def loss(wb):
            w2, _, _ = updraft_velocity(self.buoy, self.dz, wb)
            return jnp.sum(jnp.maximum(w2, 0.0))
        g = jax.grad(loss)(self.w_base)
        self.assertTrue(jnp.isfinite(g))


class TestSaturationAdjust(unittest.TestCase):
    def test_unsaturated_recovers_inputs(self):
        t, phi, qt, p = 280.0, 5000.0, 0.004, 80000.0     # subsaturated
        h = SHA * t + phi + LHE * qt
        t_out, qv, qc = _saturation_adjust(
            jnp.array(h), jnp.array(phi), jnp.array(qt), jnp.array(p))
        self.assertAlmostEqual(float(t_out), t, places=2)
        self.assertAlmostEqual(float(qv), qt, places=6)
        self.assertAlmostEqual(float(qc), 0.0, places=6)

    def test_saturated_splits_and_conserves_mse(self):
        t0, phi, p = 290.0, 3000.0, 90000.0
        qv0 = float(saturation_specific_humidity(jnp.array(t0), jnp.array(p)))
        qt = qv0 + 0.003                                  # 3 g/kg of condensate
        h = SHA * t0 + phi + LHE * qv0                    # MSE of the saturated parcel
        t_out, qv, qc = _saturation_adjust(
            jnp.array(h), jnp.array(phi), jnp.array(qt), jnp.array(p))
        self.assertAlmostEqual(float(t_out), t0, places=1)
        self.assertGreater(float(qc), 0.0)
        # MSE is conserved by the adjustment (to a few ppm after Newton).
        h_out = SHA * float(t_out) + phi + LHE * float(qv)
        self.assertAlmostEqual(h_out / h, 1.0, places=5)


class TestEntrainingPlumeAscent(unittest.TestCase):
    def setUp(self):
        self.s = _conditionally_unstable_sounding()

    def _run(self, contce=0.4):
        return entraining_plume_ascent(contce=contce, **self.s)

    def test_buoyant_and_condenses_then_caps(self):
        t_p, cond, buoy, w2, top = self._run()
        self.assertGreater(float(buoy[0]), 0.0)           # buoyant above base
        self.assertGreater(float(jnp.max(cond)), 0.0)     # cloud condensate forms
        self.assertGreater(float(jnp.max(w2)), 1.0)       # a real updraft develops
        self.assertLess(int(top), self.s["pressure"].shape[0])  # plume terminates

    def test_more_entrainment_lowers_cloud_top(self):
        # The entrainment feedback: a more strongly entraining plume is diluted
        # faster, loses buoyancy sooner, and tops out lower.
        _, _, _, _, top_low = self._run(contce=0.2)
        _, _, _, _, top_high = self._run(contce=0.8)
        self.assertGreaterEqual(int(top_low), int(top_high))

    def test_gradient_finite(self):
        def loss(t_base):
            s = dict(self.s)
            s["t_base"] = t_base
            _, _, _, w2, _ = entraining_plume_ascent(**s)
            return jnp.sum(jnp.maximum(w2, 0.0))
        g = jax.grad(loss)(self.s["t_base"])
        self.assertTrue(jnp.isfinite(g))


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
