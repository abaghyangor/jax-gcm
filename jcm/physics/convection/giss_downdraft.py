"""Convective downdraft descent -- GISS ModelE ``dd_evap_precip_loop``.

Port of the descent core of ``dd_evap_precip_loop`` (``MSTCNV.F90:3985-4792``).
The buoyancy-sorting plume routes its most negatively buoyant blends into a
downdraft (:mod:`jcm.physics.convection.giss_bsort`); this is what becomes of
them. The downdraft descends from cloud top, precipitation evaporates into it
and cools it, and it detrains into the environment where it turns buoyant or
once it reaches the boundary layer.

**Why this matters rather than being a refinement.** In BOMEX the downdraft
carries 29.8% of all the mass leaving the plume, and it forms roughly 350 m
higher than the local detrainment. Depositing that air where it forms rather
than where it ends up misplaces both the cooling and the moistening, and drops
the evaporative cooling entirely -- an error in the *shape* of the convective
tendency profile, not merely its magnitude.

Ported and not ported
---------------------
The descent, the evaporation into the downdraft, the buoyancy test and the
detrainment are ported and checked against a dump taken from inside ModelE's
own descent loop (``oracle_data/bomex_downdraft.txt``).

Deliberately outside this module:

* **Melting and freezing.** ModelE redistributes melting heat when the
  precipitation phase changes across a level. BOMEX is all-liquid on every
  oracle level (``lhx = LHE``), so the whole block is inert; a mixed-phase case
  would need it.
* **Environmental precipitation evaporation.** ModelE splits the precipitation
  into a downdraft share and an environmental share (``fddrt``) and evaporates
  the latter directly into the environment. The split is honoured here so the
  downdraft sees the right amount of water, but the environmental share's own
  evaporation is a separate term.
* **Tracers**, which the ``PhysicsTerm`` interface does not carry.

The precipitation supply itself comes from ModelE's convective microphysics,
which is not ported; see ``giss_bsort.precipitation_fraction``.

Broadcasting-native: vertical on axis 0, and the scan runs downward.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from jcm.physics.convection.giss_bsort import _safe_divide
from jcm.physics.convection.giss_thermodynamics import (
    DELTX, LHE, LHS, SHA, condensate_evaporation)

# Share of the precipitation flux that falls through the downdraft rather than
# the surrounding environment (`mc_fddrt`).
_DOWNDRAFT_PRECIP_SHARE = 0.5

# Fraction of the downdraft *retained* where it has turned positively buoyant;
# the rest detrains. ModelE's `dfac = 1 - fddet`. Measured from the oracle at
# 0.25 (so 75% detrains), `fddet` having no assignment left in MSTCNV.F90.
_BUOYANT_RETENTION = 0.25
# In the boundary layer detrainment is forced regardless of buoyancy, at
# `1 - detfac` with `detfac = 0.5` (MSTCNV.F90:4230).
_BOUNDARY_LAYER_DETRAINMENT = 0.5

# Downdraft entrainment: a constant rate times the layer depth. ModelE passes
# `etal` in; over 561 oracle records `etal/gzl` is 2.0e-4 /m to within 0.8%.
_DOWNDRAFT_ENTRAINMENT_RATE = 2.0e-4

# Ceiling on entrained environmental air as a fraction of the layer's mass.
_REMRAT = 0.333

# Evaporation efficiency: the saturation deficit is scaled down by how much
# precipitation is actually falling through the layer (`dd_evpeff_qp_scale`),
# so a thin shaft of rain cannot evaporate as if it filled the layer.
_EVAP_PRECIP_SCALE = 1.0e-3
_EVAP_MASS_REFERENCE = 30.0     # mb; `ma(l)*kg2mb/30`
_EVAP_EXPONENT = 0.6
_PRECIP_MIXING_RATIO_MAX = 0.01
_PRECIP_DEPTH_MAX = 400.0       # mb, `min(dp_from_cldtop, 400)`

_KG_TO_MB = 9.80665 / 100.0     # ModelE `kg2mb`
_TEENY = 1e-20


class DowndraftDescent(NamedTuple):
    """Per-level downdraft results. All arrays ``(nlev, *horiz)``."""
    mass: jnp.ndarray               # ddm: downdraft mass entering each level
    detrained_mass: jnp.ndarray     # mass handed to the environment
    detrained_heat: jnp.ndarray
    detrained_water: jnp.ndarray
    entrained_air: jnp.ndarray      # environmental air drawn into the downdraft
    evaporated: jnp.ndarray         # precipitation evaporated into it
    potential_temperature: jnp.ndarray   # thdn
    specific_humidity: jnp.ndarray       # qldn


def downdraft_descent(source_mass: jnp.ndarray,
                      source_heat: jnp.ndarray,
                      source_water: jnp.ndarray,
                      precipitation: jnp.ndarray,
                      environment_heat: jnp.ndarray,
                      environment_water: jnp.ndarray,
                      environment_condensate: jnp.ndarray,
                      layer_mass: jnp.ndarray,
                      layer_depth: jnp.ndarray,
                      convective_fraction: jnp.ndarray,
                      exner: jnp.ndarray,
                      pressure: jnp.ndarray,
                      cloud_top: jnp.ndarray,
                      boundary_layer_top: jnp.ndarray,
                      phase: str = "water") -> DowndraftDescent:
    """Descend the downdraft from cloud top, evaporating precipitation into it.

    Args:
        source_mass: ``ddr(l)``, blend mass the plume routed to the downdraft
            at each level [kg/m^2].
        source_heat: ``smdnl(l)``, its extensive heat.
        source_water: ``qmdnl(l)``, its extensive water vapour.
        precipitation: Condensate rained out at each level [kg/m^2]. Only the
            ``fddrt`` share falls through the downdraft.
        environment_heat: Intensive environmental heat.
        environment_water: Intensive environmental humidity.
        environment_condensate: Environmental cloud condensate, which loads the
            environment's virtual temperature in the buoyancy comparison.
        layer_mass: ``ma`` [kg/m^2].
        layer_depth: ``gzl`` [m], setting the entrainment per layer.
        convective_fraction: ``mcfrac``, the convective area fraction, which
            sets how concentrated the falling precipitation is.
        exner: ``plk``.
        pressure: [**Pa**].
        cloud_top: Index of the highest level the downdraft exists at
            (``ldraft``); above it nothing happens.
        boundary_layer_top: Level at or below which detrainment is forced
            (``max(lcl-1, dcl)``).
        phase: ``"water"`` or ``"ice"``. Static.

    Returns:
        A :class:`DowndraftDescent`.
    """
    latent_heat = LHE if phase == "water" else LHS
    slh = latent_heat / SHA
    nlev = layer_mass.shape[0]
    horiz = layer_mass.shape[1:]
    level_axis = jnp.arange(nlev).reshape((nlev,) + (1,) * len(horiz))
    zeros = jnp.zeros(horiz)

    # `area_min` is half the column's peak convective fraction, so the rain
    # shaft is never assumed narrower than that.
    area_min = 0.5 * jnp.max(convective_fraction, axis=0)

    def step(carry, level_inputs):
        (mass, heat, water, precip_down, precip_env, depth_from_top) = carry
        (level, ddr, smdnl, qmdnl, precip, senv, qenv, qcond, ma, gzl,
         mcfrac, plk, pres) = level_inputs

        below_top = level <= cloud_top

        # Precipitation is re-split between the downdraft and the environment at
        # every level, then the condensate the plume shed here is added.
        total_precip = precip_down + precip_env
        precip_down = total_precip * _DOWNDRAFT_PRECIP_SHARE + precip
        precip_env = total_precip * (1.0 - _DOWNDRAFT_PRECIP_SHARE)

        depth_from_top = depth_from_top + ma * _KG_TO_MB
        precip_area = jnp.maximum(area_min, mcfrac)
        precip_mixing_ratio = jnp.minimum(
            _safe_divide(total_precip,
                         precip_area * jnp.minimum(depth_from_top,
                                                   _PRECIP_DEPTH_MAX)),
            _PRECIP_MIXING_RATIO_MAX)

        mass_in = mass
        mass = mass + jnp.where(below_top, ddr, 0.0)
        heat = heat + jnp.where(below_top, smdnl, 0.0)
        water = water + jnp.where(below_top, qmdnl, 0.0)

        alive = (mass > 0.0) & below_top

        # Evaporate precipitation into the downdraft. ModelE computes the full
        # saturation deficit first, then scales it by how much rain is actually
        # falling through the layer, and finally caps it at what is there.
        safe_mass = jnp.where(alive, mass, 1.0)
        evaporated, _ = condensate_evaporation(
            heat, water, plk, safe_mass, pres,
            jnp.full_like(mass, 1e30), phase)
        efficiency = jnp.minimum(
            1.0, (ma * _KG_TO_MB / _EVAP_MASS_REFERENCE)
            * (precip_mixing_ratio / _EVAP_PRECIP_SCALE) ** _EVAP_EXPONENT)
        evaporated = jnp.minimum(evaporated * efficiency, precip_down)
        evaporated = jnp.where(alive, evaporated, 0.0)

        heat = heat - slh * evaporated / plk
        water = water + evaporated
        precip_down = precip_down - evaporated

        theta = _safe_divide(heat, mass)
        humidity = _safe_divide(water, mass)

        # Buoyancy against the environment, both loaded by their condensate.
        downdraft_virtual = theta * plk * (1.0 + DELTX * humidity
                                           - precip_mixing_ratio)
        environment_virtual = senv * plk * (1.0 + DELTX * qenv - qcond)

        buoyant = downdraft_virtual >= environment_virtual
        # The boundary layer is at *low* level indices (index rises upward).
        in_boundary_layer = level <= boundary_layer_top
        can_exchange = alive & (level > 0) & (level < cloud_top)

        # Where it has turned buoyant the downdraft stops entraining and sheds
        # most of itself; in the boundary layer it sheds regardless.
        detrained_fraction = jnp.where(
            buoyant, 1.0 - _BUOYANT_RETENTION,
            jnp.where(in_boundary_layer, 1.0 - _BOUNDARY_LAYER_DETRAINMENT,
                      0.0))
        entrainment = jnp.where(buoyant, 0.0,
                                _DOWNDRAFT_ENTRAINMENT_RATE * gzl)

        entrained = jnp.where(can_exchange, mass * entrainment, 0.0)
        # ModelE's "implicit" form, then a ceiling on how much of the layer may
        # be drawn in.
        entrained = _safe_divide(entrained, 1.0 + _safe_divide(entrained, ma))
        entrained = jnp.minimum(entrained, _REMRAT * ma)
        detrained = jnp.where(can_exchange,
                              mass * jnp.minimum(1.0, detrained_fraction), 0.0)

        detrained_fraction_actual = _safe_divide(detrained, mass)
        detrained_heat = heat * detrained_fraction_actual
        detrained_water = water * detrained_fraction_actual

        # What leaves with the detrained air is gone; what is entrained arrives
        # carrying the environment's properties.
        heat = heat - detrained_heat + entrained * senv
        water = water - detrained_water + entrained * qenv
        mass = mass - detrained + entrained

        carry = (mass, heat, water, precip_down, precip_env, depth_from_top)
        outputs = (mass_in, detrained, detrained_heat, detrained_water,
                   entrained, evaporated, theta, humidity)
        return carry, outputs

    initial = (zeros, zeros, zeros, zeros, zeros, zeros)
    _, outputs = lax.scan(
        step, initial,
        (level_axis, source_mass, source_heat, source_water, precipitation,
         environment_heat, environment_water, environment_condensate,
         layer_mass, layer_depth, convective_fraction, exner, pressure),
        reverse=True)
    return DowndraftDescent(*outputs)
