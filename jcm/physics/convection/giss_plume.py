"""GISS plume above cloud base: moist-adiabatic ascent (foundation).

The convective plume in ``MSTCNV`` rises from cloud base, condensing vapour
(releasing latent heat), entraining environmental air, and driving an updraft;
its compensating subsidence and detrainment are what ultimately produce the
environmental tendencies ``dq_mc``/``dth_mc``. This module ports the **first**
piece of that chain: the saturated parcel's thermodynamic ascent.

Scope: undilute (no entrainment)
--------------------------------
This is the **undilute** saturated ascent -- a parcel that stays exactly
saturated as it rises, with no mixing of environmental air. ``MSTCNV``'s plume is
*entraining* (it dilutes with environment, which cools it and lowers cloud top),
so this is a building block, not the full plume. Entrainment, updraft velocity,
detrainment, and the closure→tendency mapping are subsequent steps. The undilute
adiabat is also the natural upper bound on in-cloud buoyancy.

Method
------
We integrate the saturated pseudoadiabatic lapse rate in ``ln(p)`` (needs only
pressure and the ported thermodynamics -- no geopotential):

    dT/dln(p) = (Rd·T/cp) · (1 + L·qs/(Rd·T)) / (1 + L²·qs/(cp·Rv·T²))

with ``qs = qsat(T, p)``. Between successive levels we take a midpoint (RK2) step,
accurate because model layers are thin near cloud base. Condensed water over a
step is the drop in saturation specific humidity, ``qs(below) − qs(above)``.

Differentiable, broadcasting-native (vertical on axis 0); units K, Pa, kg/kg.
"""

import jax
import jax.numpy as jnp

from jcm.physics.convection.giss_thermodynamics import (
    GRAV,
    LHE,
    RGAS,
    RVAP,
    SHA,
    d_ln_qsat_dt,
    saturation_specific_humidity,
    virtual_temperature,
)


def saturated_lapse_rate_dlnp(temperature: jnp.ndarray,
                              pressure: jnp.ndarray,
                              phase: str = "water") -> jnp.ndarray:
    """Saturated pseudoadiabatic lapse rate ``dT/dln(p)`` [K].

    Positive: temperature decreases as pressure decreases (ascent). The latent
    heat of condensation (the ``qs`` terms) makes it smaller in magnitude than
    the dry value ``Rd·T/cp``.
    """
    qs = saturation_specific_humidity(temperature, pressure, phase)
    numerator = 1.0 + LHE * qs / (RGAS * temperature)
    denominator = 1.0 + LHE ** 2 * qs / (SHA * RVAP * temperature ** 2)
    return (RGAS * temperature / SHA) * numerator / denominator


def moist_adiabat_ascent(t_base: jnp.ndarray,
                         p_base: jnp.ndarray,
                         pressure_levels: jnp.ndarray,
                         phase: str = "water"):
    """Lift a saturated parcel up the moist adiabat through ``pressure_levels``.

    The parcel starts saturated at ``(t_base, p_base)`` -- i.e. at cloud base --
    and is lifted through ``pressure_levels`` (ordered **upward**, i.e.
    decreasing pressure; vertical on axis 0). Each step integrates the saturated
    pseudoadiabat with a midpoint (RK2) step in ``ln(p)``.

    Args:
        t_base: Cloud-base parcel temperature [K], per column ``(...)``.
        p_base: Cloud-base pressure [Pa], per column ``(...)``.
        pressure_levels: Pressures above cloud base [Pa], ``(n_above, ...)``,
            ordered upward (each entry lower than ``p_base``).
        phase: ``"water"`` or ``"ice"`` (static).

    Returns:
        ``(parcel_temperature, condensate)`` -- each ``(n_above, ...)``: the
        moist-adiabatic parcel temperature [K] at each level, and the cumulative
        condensed water [kg/kg] produced up to that level.
    """
    def step(carry, p_next):
        t_prev, p_prev, cond = carry
        dlnp = jnp.log(p_next) - jnp.log(p_prev)               # < 0 on ascent
        # Midpoint (RK2) in ln(p).
        t_mid = t_prev + 0.5 * dlnp * saturated_lapse_rate_dlnp(t_prev, p_prev, phase)
        p_mid = jnp.exp(0.5 * (jnp.log(p_prev) + jnp.log(p_next)))
        t_next = t_prev + dlnp * saturated_lapse_rate_dlnp(t_mid, p_mid, phase)
        # Condensation over the step: drop in saturation specific humidity.
        qs_prev = saturation_specific_humidity(t_prev, p_prev, phase)
        qs_next = saturation_specific_humidity(t_next, p_next, phase)
        cond = cond + jnp.maximum(qs_prev - qs_next, 0.0)
        return (t_next, p_next, cond), (t_next, cond)

    cond0 = jnp.zeros_like(t_base * 1.0)
    _, (parcel_t, condensate) = jax.lax.scan(
        step, (t_base, p_base, cond0), pressure_levels)
    return parcel_t, condensate


# --- Entraining updraft: Gregory (2001) cumulus velocity + entrainment --------
# MSTCNV uses contce = 0.4 (less-entraining plume, no cold pools) and 0.6 (more-
# entraining plume); the 1/6 coefficients are Gregory's. ``_TEENY`` guards the
# 1/w^2 entrainment rate at/below cloud base where w can be tiny.
_SIXTH = 1.0 / 6.0
_TWO_THIRDS = 2.0 / 3.0   # detrainment-drag coefficient in the Gregory w^2 law
_TEENY = 1.0e-20


def entrainment_rate(buoyancy: jnp.ndarray,
                     updraft_speed: jnp.ndarray,
                     contce: jnp.ndarray = 0.4) -> jnp.ndarray:
    """Buoyancy-sorting fractional entrainment rate [1/m], MSTCNV / Gregory 2001.

    ``ENT = (1/6)·contce·g·B / w²`` (``MSTCNV`` line 2916). More buoyant or slower
    plumes entrain more; faster plumes entrain less. ``contce`` is the
    entrainment-strength scaling (≈0.4 for the less-entraining plume, 0.6 for the
    more-entraining one).

    Args:
        buoyancy: Fractional buoyancy ``B = (Tv_p − Tv_e)/Tv_e − condensate``.
        updraft_speed: Updraft speed ``w`` [m/s] (from the level below).
        contce: Entrainment-strength scaling (static or array).

    Returns:
        Fractional entrainment rate [1/m].
    """
    return _SIXTH * contce * GRAV * buoyancy / (updraft_speed ** 2 + _TEENY)


def updraft_velocity(buoyancy: jnp.ndarray,
                     layer_thickness: jnp.ndarray,
                     w_base: jnp.ndarray,
                     contce: jnp.ndarray = 0.4,
                     detrainment: jnp.ndarray = 0.0):
    """Integrate the Gregory (2001) cumulus updraft ``w²`` from cloud base up.

    Per level (``MSTCNV`` lines 3112-3115)::

        W2TEM = (1/6)·g·B − w(L-1)²·((2/3)·det + ent)
        w²(L) = w²(L-1) + 2·dz·W2TEM

    where ``ent`` is the buoyancy-sorting :func:`entrainment_rate` (computed from
    the previous level's ``w``) and ``det`` is the detrainment rate (an input
    here; the detrainment closure is a separate port — default 0). Buoyancy
    production lifts ``w²``; entrainment/detrainment drag and negative buoyancy
    bring it down. **Cloud top is the first level where ``w² ≤ 0``.**

    Args:
        buoyancy: Fractional buoyancy profile above cloud base, ``(n, ...)``,
            ordered upward.
        layer_thickness: Layer thickness ``dz = MA/ρ`` [m], ``(n, ...)``.
        w_base: Cloud-base updraft speed [m/s], per column ``(...)``.
        contce: Entrainment-strength scaling.
        detrainment: Detrainment rate [1/m] (default 0; separate port).

    Returns:
        ``(w2, w, cloud_top)`` -- the updraft ``w²`` [m²/s²] and ``w`` [m/s]
        profiles ``(n, ...)``, and the ``cloud_top`` level index ``(...)`` (first
        level with ``w² ≤ 0``; sentinel ``n`` if the plume never stops in range).
    """
    det = jnp.broadcast_to(jnp.asarray(detrainment) * 1.0, buoyancy.shape)

    def step(carry, inputs):
        w2_prev, w_prev = carry
        b, dz, det_l = inputs
        ent = entrainment_rate(b, w_prev, contce)
        w2tem = _SIXTH * GRAV * b - w_prev ** 2 * (_TWO_THIRDS * det_l + ent)
        w2 = w2_prev + 2.0 * dz * w2tem
        # Safe sqrt: above cloud top w2 <= 0, and a bare sqrt(max(w2,0)) has an
        # infinite derivative at 0 (NaN gradients). The double-``where`` keeps the
        # sqrt argument >= 1 on the masked branch so the gradient stays finite.
        positive = w2 > 0.0
        w = jnp.where(positive, jnp.sqrt(jnp.where(positive, w2, 1.0)), 0.0)
        w = jnp.minimum(w, 50.0)                    # MSTCNV caps |WCU| at 50 m/s
        return (w2, w), (w2, w)

    w_base = w_base * 1.0
    _, (w2_prof, w_prof) = jax.lax.scan(
        step, (w_base ** 2, w_base), (buoyancy, layer_thickness, det))

    n = buoyancy.shape[0]
    stopped = w2_prof <= 0.0
    cloud_top = jnp.where(jnp.any(stopped, axis=0),
                          jnp.argmax(stopped, axis=0), n)
    return w2_prof, w_prof, cloud_top


# --- Coupled entraining plume -------------------------------------------------
def _saturation_adjust(mse, geopotential, q_total, pressure, phase="water",
                       n_newton=4):
    """Split moist static energy + total water into ``(T, q_vapor, q_cond)``.

    Given the plume's moist static energy ``h = cp·T + Φ + L·q_v`` and total water
    ``q_t`` at a level, recover temperature and the vapour/condensate partition.
    If ``q_t <= qsat`` the parcel is unsaturated (``q_v = q_t``, no condensate);
    otherwise it is saturated (``q_v = qsat(T,p)``) and ``T`` solves
    ``cp·T + Φ + L·qsat(T,p) = h`` by a few Newton steps (using the analytic
    ``d ln qsat/dT``).
    """
    t_unsat = (mse - geopotential - LHE * q_total) / SHA
    saturated = q_total > saturation_specific_humidity(t_unsat, pressure, phase)

    t = t_unsat
    for _ in range(n_newton):
        qs = saturation_specific_humidity(t, pressure, phase)
        f = SHA * t + geopotential + LHE * qs - mse
        fprime = SHA + LHE * qs * d_ln_qsat_dt(t, phase)
        t = t - f / fprime

    t = jnp.where(saturated, t, t_unsat)
    q_vapor = jnp.where(
        saturated, saturation_specific_humidity(t, pressure, phase), q_total)
    q_cond = jnp.maximum(q_total - q_vapor, 0.0)
    return t, q_vapor, q_cond


def entraining_plume_ascent(t_base, q_base, geopotential_base, p_base, w_base,
                            env_temperature, env_vapor, geopotential,
                            pressure, layer_thickness,
                            contce=0.4, phase="water",
                            m_base=1.0, layer_mass=None, minfrac=0.01):
    """March a self-consistent entraining plume from cloud base to cloud top.

    Couples the pieces ported separately: at each level the plume (carried as
    moist static energy + total water) is saturation-adjusted, its buoyancy vs
    the environment drives the Gregory updraft (:func:`updraft_velocity`'s
    per-level law) and the buoyancy-sorting :func:`entrainment_rate`, and that
    entrainment then mixes environmental air into the plume for the next level --
    so buoyancy → updraft → entrainment → dilution feed back. Lifting between
    levels conserves the plume's moist static energy; entrainment relaxes it
    toward the environment by the fractional entrained mass ``ε·dz``.

    The plume's **mass** is tracked through the ascent: entrainment grows it
    (``×(1+ε·dz)``), detrainment sheds it (``×(1−det·dz)``), and the plume
    terminates either kinematically (``w² ≤ 0``) or when its mass falls below
    ``minfrac·layer_mass`` (``MSTCNV`` line 1676) -- the mass cap that stops a
    plume which has detrained itself out of existence even while ``w²`` is still
    positive. The returned mass-flux profile is the quantity that drives the
    compensating subsidence (the eventual ``dq_mc``/``dth_mc`` tendencies).
    Single plume, no precipitation yet; condensate accumulates as suspended water.

    Args (cloud-base scalars per column ``(...)``; profiles ``(n, ...)`` above
    cloud base, ordered upward):
        t_base, q_base, geopotential_base, p_base: cloud-base parcel temperature
            [K], saturated vapour [kg/kg], geopotential [m²/s²], pressure [Pa].
        w_base: cloud-base updraft speed [m/s].
        env_temperature, env_vapor: environment T [K] / vapour [kg/kg].
        geopotential, pressure, layer_thickness: Φ [m²/s²], p [Pa], dz [m].
        contce: entrainment-strength scaling.
        phase: ``"water"`` or ``"ice"`` (static).
        m_base: cloud-base plume mass [kg/m²] (e.g. the closure ``fmp2``).
        layer_mass: layer air mass ``MA = dp/g`` [kg/m²], ``(n, ...)``; if given,
            the plume terminates when its mass ``≤ minfrac·layer_mass``.
        minfrac: minimum plume / layer mass fraction for the mass cap.

    Returns:
        ``(parcel_temperature, condensate, buoyancy, w2, mass_flux, cloud_top)``
        -- profiles ``(n, ...)`` and the ``cloud_top`` level index ``(...)``.
    """
    h_base = SHA * t_base + geopotential_base + LHE * q_base
    qt_base = q_base
    w_base = w_base * 1.0
    m_base = jnp.asarray(m_base) * jnp.ones_like(w_base)

    def step(carry, inp):
        h_p, qt_p, w2_prev, w_prev, m = carry
        t_env, q_env, phi, p, dz = inp

        # Saturation-adjust the (entrained, lifted) plume at this level.
        t_p, qv_p, qc_p = _saturation_adjust(h_p, phi, qt_p, p, phase)

        # Buoyancy: virtual temperature of plume vs environment (condensate
        # loads the plume; the environment is taken cloud-free).
        tv_p = virtual_temperature(t_p, qv_p, qc_p)
        tv_e = virtual_temperature(t_env, q_env, 0.0)
        buoyancy = (tv_p - tv_e) / tv_e

        # Buoyancy-sorting rate. When the plume is negatively buoyant the rate
        # goes negative and MSTCNV converts it to *detrainment* (DET = -ENT,
        # ENT = 0; lines 2921-2925): the plume stops entraining and instead sheds
        # mass, which adds strong drag to the updraft and caps the cloud.
        rate = entrainment_rate(buoyancy, w_prev, contce)
        ent = jnp.maximum(rate, 0.0)
        det = jnp.maximum(-rate, 0.0)

        # Gregory updraft with entrainment + detrainment drag.
        w2tem = _SIXTH * GRAV * buoyancy - w_prev ** 2 * (_TWO_THIRDS * det + ent)
        w2 = w2_prev + 2.0 * dz * w2tem
        positive = w2 > 0.0
        w = jnp.where(positive, jnp.sqrt(jnp.where(positive, w2, 1.0)), 0.0)
        w = jnp.minimum(w, 50.0)

        # Entrain environmental air for the next level. Implicit (bounded)
        # mixing fraction ``ε·dz/(1+ε·dz)`` -- MSTCNV's implicit entrainment
        # limiter (line 2928) in intensive form; smooth and always in [0,1), so
        # the explicit ``clip`` discontinuity that destabilized strong
        # entrainment is gone. Only the entraining (buoyant) part dilutes the
        # plume; detrainment removes mass without changing the remaining plume's
        # intensive properties.
        e = ent * dz
        frac = e / (1.0 + e)
        h_env = SHA * t_env + phi + LHE * q_env
        h_next = h_p + frac * (h_env - h_p)
        qt_next = qt_p + frac * (q_env - qt_p)

        # Plume mass: entrainment adds (1+ε·dz), detrainment sheds (1−det·dz)
        # (capped at 0.95/level, MSTCNV line 2967). The two are mutually
        # exclusive per level (ent>0 XOR det>0), so the product does the right
        # thing in both phases.
        delta = jnp.minimum(det * dz, 0.95)
        m_next = m * (1.0 + ent * dz) * (1.0 - delta)

        return (h_next, qt_next, w2, w, m_next), (t_p, qc_p, buoyancy, w2, m_next)

    _, (parcel_t, condensate, buoyancy, w2_prof, mass_flux) = jax.lax.scan(
        step, (h_base, qt_base, w_base ** 2, w_base, m_base),
        (env_temperature, env_vapor, geopotential, pressure, layer_thickness))

    n = env_temperature.shape[0]
    stopped = w2_prof <= 0.0
    if layer_mass is not None:
        stopped = stopped | (mass_flux <= minfrac * layer_mass)
    cloud_top = jnp.where(jnp.any(stopped, axis=0),
                          jnp.argmax(stopped, axis=0), n)
    return parcel_t, condensate, buoyancy, w2_prof, mass_flux, cloud_top
