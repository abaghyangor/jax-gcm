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
_REMRAT = 0.333          # MSTCNV cap: entrained mass <= remrat * layer air mass


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


def _plume_core(h_p, qt_p, w2_prev, w_prev, m, t_env, q_env, phi, p, dz,
                layer_mass, contce, phase):
    """One entraining-plume level: physics shared by the base-relative and
    full-column ascents.

    Given the plume carry ``(moist static energy h, total water qt, prior w², w,
    mass m)`` and the environment at this level, saturation-adjust the plume,
    form its buoyancy, advance the Gregory updraft ``w²``, entrain/detrain, and
    return the next carry plus this level's diagnostics. See
    :func:`entraining_plume_ascent` for the physics rationale.

    ``layer_mass`` [kg/m²] bounds the entrained mass to ``MSTCNV``'s limits (the
    implicit limiter and the ``remrat``-fraction cap, lines 2928-2929): the
    entrained mass is capped **relative to the layer air mass**, not the plume
    mass, which is what stops the plume mass running away in a deep buoyant
    column. Pass ``inf`` to disable the cap (recovering ``m·(1+ε·dz)``).
    """
    # Saturation-adjust the (entrained, lifted) plume at this level.
    t_p, qv_p, qc_p = _saturation_adjust(h_p, phi, qt_p, p, phase)

    # Buoyancy: virtual temperature of plume vs environment (condensate loads the
    # plume; the environment is taken cloud-free).
    tv_p = virtual_temperature(t_p, qv_p, qc_p)
    tv_e = virtual_temperature(t_env, q_env, 0.0)
    buoyancy = (tv_p - tv_e) / tv_e

    # Buoyancy-sorting rate: negative buoyancy -> detrainment (DET = -ENT, ENT = 0).
    rate = entrainment_rate(buoyancy, w_prev, contce)
    ent = jnp.maximum(rate, 0.0)
    det = jnp.maximum(-rate, 0.0)

    # Gregory updraft with entrainment + detrainment drag.
    w2tem = _SIXTH * GRAV * buoyancy - w_prev ** 2 * (_TWO_THIRDS * det + ent)
    w2 = w2_prev + 2.0 * dz * w2tem
    positive = w2 > 0.0
    w = jnp.where(positive, jnp.sqrt(jnp.where(positive, w2, 1.0)), 0.0)
    w = jnp.minimum(w, 50.0)

    # Dilution: the plume relaxes toward the environment by the implicit bounded
    # fraction ``ε·dz/(1+ε·dz)`` (always in [0,1), no small denominator -- keeps
    # the gradient finite). Only the entraining part dilutes.
    e = ent * dz
    frac = e / (1.0 + e)
    h_env = SHA * t_env + phi + LHE * q_env
    h_next = h_p + frac * (h_env - h_p)
    qt_next = qt_p + frac * (q_env - qt_p)

    # Plume mass: the entrained mass added per level is bounded like MSTCNV --
    # the implicit limiter then the ``remrat`` fraction cap, both **relative to
    # the layer air mass** (not the plume mass), which stops the plume mass
    # running away in a deep buoyant column. Detrainment then sheds ≤ 0.95/level.
    entrained = m * e
    entrained = entrained / (1.0 + entrained / layer_mass)
    entrained = jnp.minimum(entrained, layer_mass * _REMRAT)
    delta = jnp.minimum(det * dz, 0.95)
    m_next = (m + entrained) * (1.0 - delta)

    return (h_next, qt_next, w2, w, m_next), (t_p, qc_p, buoyancy, w2, m_next, det)


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

    # Per-level layer mass for the entrainment cap; ``inf`` (no ``layer_mass``
    # supplied) disables the cap, recovering the uncapped ``m·(1+ε·dz)`` growth.
    cap_mass = (jnp.full_like(env_temperature, jnp.inf)
                if layer_mass is None else layer_mass)

    def step(carry, inp):
        h_p, qt_p, w2_prev, w_prev, m = carry
        t_env, q_env, phi, p, dz, ma = inp
        return _plume_core(h_p, qt_p, w2_prev, w_prev, m,
                           t_env, q_env, phi, p, dz, ma, contce, phase)

    _, (parcel_t, condensate, buoyancy, w2_prof, mass_flux, _det) = jax.lax.scan(
        step, (h_base, qt_base, w_base ** 2, w_base, m_base),
        (env_temperature, env_vapor, geopotential, pressure, layer_thickness,
         cap_mass))

    n = env_temperature.shape[0]
    stopped = w2_prof <= 0.0
    if layer_mass is not None:
        stopped = stopped | (mass_flux <= minfrac * layer_mass)
    cloud_top = jnp.where(jnp.any(stopped, axis=0),
                          jnp.argmax(stopped, axis=0), n)
    return parcel_t, condensate, buoyancy, w2_prof, mass_flux, cloud_top


def plume_ascent_column(cloud_base, t_base, q_base, geopotential_base, w_base,
                        m_base, env_temperature, env_vapor, geopotential,
                        pressure, layer_thickness, layer_mass,
                        contce=0.4, phase="water"):
    """Full-column entraining plume ascent launched at a (traced) cloud base.

    The composable ``GissConvection`` term has a **per-column cloud-base index**
    that is a traced value, so it cannot statically slice the environment above
    cloud base the way :func:`entraining_plume_ascent` expects. This runs the
    same per-level physics (:func:`_plume_core`) over the **whole column** and
    *launches* the plume at ``cloud_base`` with a :func:`jnp.where` gate (the
    static-shape-friendly device the Tiedtke-Nordeng term uses):

    * levels ``≤ cloud_base``: the plume is dormant -- zero mass flux, no ascent;
    * level ``cloud_base + 1``: the plume is seeded from the boundary-layer source
      parcel (``h_base``, ``q_base``, ``w_base``, ``m_base = fmp2``);
    * levels above: the normal entraining ascent.

    Above the first level where ``w² ≤ 0`` the plume is declared dead and its
    mass flux is zeroed *cumulatively* -- so a spurious re-buoyant layer above the
    real cloud top (the documented over-penetration) cannot revive it, and the
    intensive rate ``ε ∝ B/w²`` cannot blow up as ``w² → 0``.

    All profile inputs are ``(nlev, ...)`` **surface-first** (index 0 = surface);
    the cloud-base scalars are ``(...)``. Broadcasting-native.

    Args:
        cloud_base: Cloud-base (LCL) level index per column, ``(...)``. The
            sentinel ``nlev`` (no cloud) yields an all-zero plume.
        t_base, q_base, geopotential_base: cloud-base parcel temperature [K],
            saturated vapour [kg/kg], geopotential [m²/s²].
        w_base: cloud-base updraft speed [m/s]. m_base: cloud-base plume mass
            ``fmp2`` [kg/m²].
        env_temperature, env_vapor, geopotential, pressure, layer_thickness:
            environment profiles ``(nlev, ...)``.
        layer_mass: layer air mass ``dp/g`` [kg/m²], ``(nlev, ...)`` -- bounds the
            entrained mass (``MSTCNV`` remrat cap) so the plume mass cannot run
            away in a deep buoyant column.
        contce: entrainment-strength scaling. phase: ``"water"``/``"ice"``.

    Returns:
        ``(parcel_temperature, condensate, buoyancy, mass_flux, detrainment_rate,
        cloud_top)`` -- profiles ``(nlev, ...)`` (zero outside the live cloud) and
        the ``cloud_top`` level index ``(...)``.
    """
    nlev = env_temperature.shape[0]
    h_base = SHA * t_base + geopotential_base + LHE * q_base
    m_base = jnp.asarray(m_base) * jnp.ones_like(w_base)
    launch_level = cloud_base + 1               # first in-cloud level

    def step(carry, inp):
        h_p, qt_p, w2_prev, w_prev, m = carry
        t_env, q_env, phi, p, dz, ma, level = inp
        # Seed the plume from the boundary-layer source at the launch level.
        launch = level == launch_level
        h_p = jnp.where(launch, h_base, h_p)
        qt_p = jnp.where(launch, q_base, qt_p)
        w2_prev = jnp.where(launch, w_base ** 2, w2_prev)
        w_prev = jnp.where(launch, w_base, w_prev)
        m = jnp.where(launch, m_base, m)

        next_carry, out = _plume_core(h_p, qt_p, w2_prev, w_prev, m,
                                      t_env, q_env, phi, p, dz, ma, contce, phase)

        # Below the launch level the plume does not exist: keep the carry and the
        # outputs at zero so nothing propagates up from the sub-cloud layer.
        active = level >= launch_level
        next_carry = tuple(jnp.where(active, c, 0.0) for c in next_carry)
        out = tuple(jnp.where(active, o, 0.0) for o in out)
        return next_carry, out

    zero = jnp.zeros_like(h_base)
    level_idx = jnp.arange(nlev)
    _, (parcel_t, condensate, buoyancy, w2_prof, mass_flux, det) = jax.lax.scan(
        step, (zero, zero, zero, zero, zero),
        (env_temperature, env_vapor, geopotential, pressure, layer_thickness,
         layer_mass, level_idx))

    # Cloud top = first in-cloud level where w² <= 0. Kill the plume there and
    # above (cumulative), so an over-penetrating re-buoyant layer cannot revive
    # it and the mass flux stays finite.
    lev = level_idx.reshape((nlev,) + (1,) * cloud_base.ndim)
    in_cloud = lev >= launch_level
    stop = (in_cloud & (w2_prof <= 0.0)).astype(mass_flux.dtype)
    stopped_below = jnp.cumsum(stop, axis=0) - stop        # stops strictly below
    alive = in_cloud & (stopped_below == 0.0)
    mass_flux = jnp.where(alive, mass_flux, 0.0)
    det = jnp.where(alive, det, 0.0)

    # The cloud-base layer itself carries the plume's cloud-base mass flux
    # ``m_base`` (= fmp2): the plume ascends out of its top, so the compensating
    # subsidence must warm/dry that layer. Without this the cloud-base level --
    # where ModelE's convective tendency is *largest* -- gets nothing, because
    # the ascent proper only starts one level up (``launch_level``).
    mass_flux = jnp.where(lev == cloud_base, m_base, mass_flux)

    cloud_top = jnp.where(jnp.any(stop > 0, axis=0),
                          jnp.argmax(stop, axis=0), nlev)
    return parcel_t, condensate, buoyancy, mass_flux, det, cloud_top
