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
