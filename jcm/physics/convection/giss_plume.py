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
    saturation_specific_humidity,
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
        w2tem = _SIXTH * GRAV * b - w_prev ** 2 * (2.0 * _SIXTH * det_l + ent)
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
