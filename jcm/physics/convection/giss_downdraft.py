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
* **Tracers**, which the ``PhysicsTerm`` interface does not carry.

The precipitation supply comes from
:mod:`jcm.physics.convection.giss_microphysics`, a direct port of ModelE's
``PRECIPLIQ_GAMMA``.

Broadcasting-native: vertical on axis 0, and the scan runs downward.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from jcm.physics.convection.giss_thermodynamics import (
    DELTX, LHE, LHS, SHA, condensate_evaporation, safe_divide)

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

# Extra evaporation fraction from the geometry of a partially-covered layer.
# Zero in the active preset (`MSTCNV.F90:290`), so `fevap` reduces to the
# precipitation-weighted convective-fraction excess.
_GEOMETRIC_FEVAP_FACTOR = 0.0

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
    environment_heat: jnp.ndarray        # dsm_evp: cooling by evaporating rain
    environment_water: jnp.ndarray       # dqm_evp: the vapour it adds


def downdraft_descent(source_mass: jnp.ndarray,
                      source_heat: jnp.ndarray,
                      source_water: jnp.ndarray,
                      precipitation: jnp.ndarray,
                      produced_precipitation: jnp.ndarray,
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
                      cloud_base: jnp.ndarray = None,
                      phase: str = "water") -> DowndraftDescent:
    """Descend the downdraft from cloud top, evaporating precipitation into it.

    Args:
        source_mass: ``ddr(l)``, blend mass the plume routed to the downdraft
            at each level [kg/m^2].
        source_heat: ``smdnl(l)``, its extensive heat.
        source_water: ``qmdnl(l)``, its extensive water vapour.
        precipitation: Condensate the sorted blends carry into the downdraft
            (``wmdnl``) [kg/m^2]. This, and only this, feeds the descending
            precipitation flux -- ModelE adds nothing else to it in the descent.
        produced_precipitation: Precipitation generated by the plume
            (``condpr``, from :mod:`giss_microphysics`) [kg/m^2]. A *separate*
            stream: it never enters the downdraft's flux, and is used only to
            weight how much rain evaporates into the clear air.
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
        cloud_base: Index of cloud base (``lcl``). Only in-cloud levels
            contribute to the precipitation weighting that sets how much rain
            evaporates into the clear air. Optional; omitting it treats every
            level as in-cloud.
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

    # Layer-edge convective fraction, used both to weight the precipitation and
    # to set how much of the layer the rain actually falls through. Only in-cloud
    # levels contribute; below cloud base ModelE sets it to zero.
    below = jnp.concatenate(
        [jnp.zeros((1,) + horiz), convective_fraction[:-1]], axis=0)
    edge_fraction = 0.5 * (below + convective_fraction)
    if cloud_base is not None:
        edge_fraction = jnp.where(level_axis >= cloud_base, edge_fraction, 0.0)

    def step(carry, level_inputs):
        (mass, heat, water, precip_down, precip_env, depth_from_top,
         precip_total, precip_weighted) = carry
        (level, ddr, smdnl, qmdnl, precip, senv, qenv, qcond, ma, gzl,
         mcfrac, plk, pres, mcfc, produced) = level_inputs

        below_top = level <= cloud_top

        # Precipitation is re-split between the downdraft and the environment at
        # every level, then the condensate the plume shed here is added.
        total_precip = precip_down + precip_env
        precip_down = total_precip * _DOWNDRAFT_PRECIP_SHARE + precip
        precip_env = total_precip * (1.0 - _DOWNDRAFT_PRECIP_SHARE)

        depth_from_top = depth_from_top + ma * _KG_TO_MB
        precip_area = jnp.maximum(area_min, mcfrac)
        precip_mixing_ratio = jnp.minimum(
            safe_divide(total_precip,
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

        theta = safe_divide(heat, mass)
        humidity = safe_divide(water, mass)

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
        entrained = safe_divide(entrained, 1.0 + safe_divide(entrained, ma))
        entrained = jnp.minimum(entrained, _REMRAT * ma)
        detrained = jnp.where(can_exchange,
                              mass * jnp.minimum(1.0, detrained_fraction), 0.0)

        detrained_fraction_actual = safe_divide(detrained, mass)
        detrained_heat = heat * detrained_fraction_actual
        detrained_water = water * detrained_fraction_actual

        # What leaves with the detrained air is gone; what is entrained arrives
        # carrying the environment's properties.
        heat = heat - detrained_heat + entrained * senv
        water = water - detrained_water + entrained * qenv
        mass = mass - detrained + entrained

        # --- rain falling outside the downdraft, evaporating into the layer --
        # ModelE runs this in the same loop (`MSTCNV.F90:4587-4643`). `fevap` is
        # the precipitation-weighted excess of convective fraction accumulated
        # from cloud top: layers that see proportionally more rain than their
        # own cloud cover get more of it evaporating into the clear air.
        precip_total = precip_total + jnp.where(mcfc > 0.0, produced, 0.0)
        precip_weighted = precip_weighted + jnp.where(mcfc > 0.0,
                                                      produced * mcfc, 0.0)
        evaporating_fraction = jnp.clip(
            jnp.maximum(safe_divide(precip_weighted, precip_total) - mcfc, 0.0)
            + mcfc * (1.0 - mcfc) * _GEOMETRIC_FEVAP_FACTOR, 0.0, 1.0)
        environment_air = evaporating_fraction * ma

        wets = (precip_env > 0.0) & (environment_air > 0.0)
        safe_environment_air = jnp.where(wets, environment_air, 1.0)
        environment_evaporated, _ = condensate_evaporation(
            senv * safe_environment_air, qenv * safe_environment_air, plk,
            safe_environment_air, pres, precip_env, phase)
        environment_evaporated = jnp.where(wets, environment_evaporated, 0.0)
        precip_env = precip_env - environment_evaporated

        carry = (mass, heat, water, precip_down, precip_env, depth_from_top,
                 precip_total, precip_weighted)
        outputs = (mass_in, detrained, detrained_heat, detrained_water,
                   entrained, evaporated, theta, humidity,
                   -slh * environment_evaporated / plk,
                   environment_evaporated)
        return carry, outputs

    initial = (zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros)
    _, outputs = lax.scan(
        step, initial,
        (level_axis, source_mass, source_heat, source_water, precipitation,
         environment_heat, environment_water, environment_condensate,
         layer_mass, layer_depth, convective_fraction, exner, pressure,
         edge_fraction, produced_precipitation),
        reverse=True)
    return DowndraftDescent(*outputs)
