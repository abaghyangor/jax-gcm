"""GISS convective environmental tendencies from the plume mass flux.

The convective plume's effect on the *environment* -- the ``dq_mc``/``dth_mc``
tendencies -- comes from **compensating subsidence**: by mass continuity, the
mass the plume carries up must sink in the surrounding environment, advecting the
environmental profile and warming/drying it. ``MSTCNV`` implements this as an
**upwind advection** of the mass-weighted environmental quantities by the
convective mass flux (``apply_continuity_tendencies`` / the subsidence loop).

This module ports that upwind operator. It takes a convective mass flux at the
interior interfaces and an environmental profile, and returns the per-layer
tendency from compensating subsidence.

Scope / status
--------------
This is the tendency *operator* (the mechanism). Producing the actual
``dq_mc``/``dth_mc`` still needs the convective mass-flux profile assembled from
the plume(s) (cloud-base closure magnitude + entrainment/detrainment continuity)
and the detrainment deposition added -- those are the remaining ports. The
single-step upwind here omits ``MSTCNV``'s CFL substepping (valid while the mass
flux is below the layer mass each step, as in shallow convection).

Sign convention follows ``MSTCNV``'s ``cmn``: a **positive** interface flux
advects the layer *below* the interface upward; negative advects the layer above
downward. Broadcasting-native: vertical on axis 0.
"""

from typing import NamedTuple

import jax.numpy as jnp


def _safe_ratio(numerator, denominator):
    """``numerator/denominator`` where positive, else zero -- gradient-safe.

    ``x / maximum(d, tiny)`` is finite in value but has a ~1/tiny derivative,
    which becomes NaN under ``grad`` on layers the plume never reached.
    """
    usable = denominator > 0.0
    return jnp.where(usable, numerator / jnp.where(usable, denominator, 1.0),
                     0.0)


def subsidence_tendency(interface_flux: jnp.ndarray,
                        prop: jnp.ndarray,
                        layer_mass: jnp.ndarray) -> jnp.ndarray:
    """Per-layer tendency of ``prop`` from compensating-subsidence advection.

    Faithful single-step port of the ``MSTCNV`` subsidence upwind flux: across
    each interior interface the mass flux carries the *upwind* (donor) layer's
    property; the per-layer change is the flux convergence, divided by the layer
    mass to give an intensive tendency.

    Args:
        interface_flux: Convective mass flux [kg/m²] at the ``n-1`` interior
            interfaces, ``(n-1, ...)``, ordered like the layers (interface ``i``
            sits between layers ``i`` and ``i+1``). Positive advects the lower
            layer up (``MSTCNV`` ``cmn`` sign).
        prop: Intensive environmental property per layer, ``(n, ...)`` (e.g.
            potential temperature, dry static energy, or specific humidity).
        layer_mass: Layer air mass ``MA = dp/g`` [kg/m²], ``(n, ...)``.

    Returns:
        Per-layer change in ``prop`` over the step (same units as ``prop``),
        ``(n, ...)``. The mass-weighted column total is zero (pure internal
        advection conserves the property).
    """
    # Upwind donor across each interior interface: lower layer if the flux is
    # upward (>0), upper layer if downward.
    donor = jnp.where(interface_flux > 0.0, prop[:-1], prop[1:])
    flux = interface_flux * donor                       # (n-1, ...)

    zero = jnp.zeros((1,) + flux.shape[1:], dtype=flux.dtype)
    flux_below = jnp.concatenate([zero, flux], axis=0)  # interface below each layer
    flux_above = jnp.concatenate([flux, zero], axis=0)  # interface above each layer

    # Mass-weighted convergence -> intensive tendency.
    return (flux_below - flux_above) / layer_mass


def convective_tendencies(mass_flux, plume_property, detrainment_rate,
                          layer_thickness, env_property, layer_mass):
    """Environmental tendency of a property from one plume (per step).

    Combines the two environmental effects of a convective plume on a conserved
    property (potential temperature for ``dth_mc``, specific humidity for
    ``dq_mc``):

    1. **Compensating subsidence** -- the plume's upward mass flux forces an
       equal environmental subsidence that brings higher-``property`` air down
       from above, in **advective** form ``(M/MA)·(prop[L+1] − prop[L])``. This
       is used rather than a flux-divergence of ``M·prop`` because the convective
       mass flux *diverges* (the plume detrains, so ``M`` falls with height): the
       advective form conserves a uniform profile and so introduces no spurious
       source where ``M`` changes, whereas a fixed-mass flux-divergence of
       ``M·prop`` would (``prop ≈ 300 K`` makes the error enormous). ModelE gets
       the same result by updating the layer mass as ``CM`` diverges through its
       subsidence substeps.
    2. **Detrainment deposition** -- where the plume sheds mass it deposits its
       own (warmer/moister) property: ``DM·(plume − env)/MA``, with the detrained
       *fraction* bounded to ``[0, 1)`` by ModelE's implicit limiter
       ``δ = (det·dz)/(1 + det·dz)`` so it can never shed more than the plume's
       own mass (the raw ``det`` diverges as ``w² → 0`` near cloud top).

    Single plume; omits precipitation/evaporation and the entrainment-removal
    bookkeeping. The **absolute magnitude scales with the cloud-base mass flux**
    (the closure ``fmp2`` seeding ``mass_flux``). Broadcasting-native; the
    returned tendency is a change over the step the mass flux represents.

    Args:
        mass_flux: Plume mass flux leaving each level upward [kg/m²], ``(n, ...)``
            (``mass_flux[L]`` is the flux through the top of layer ``L``).
        plume_property: Plume property per level (θ [K] or q [kg/kg]), ``(n, ...)``.
        detrainment_rate: Detrainment rate [1/m], ``(n, ...)``.
        layer_thickness: Layer thickness ``dz`` [m], ``(n, ...)``.
        env_property: Environmental property per level, ``(n, ...)``.
        layer_mass: Layer air mass ``MA`` [kg/m²], ``(n, ...)``.

    Returns:
        Per-layer environmental tendency of the property, ``(n, ...)``.
    """
    # Advective compensating subsidence: layer L is warmed/dried by the higher-
    # property air subsiding from layer L+1, at rate (M/MA)*(prop[L+1]-prop[L]).
    # The top layer has no layer above (and the plume mass flux there is ~0).
    zero = jnp.zeros((1,) + env_property.shape[1:], dtype=env_property.dtype)
    prop_above_minus = jnp.concatenate(
        [env_property[1:] - env_property[:-1], zero], axis=0)
    subsidence = mass_flux * prop_above_minus / layer_mass

    # Detrainment deposition of the plume property into each layer, with the
    # detrained *fraction* bounded to [0, 1) by ModelE's implicit limiter (the
    # raw det*dz diverges as w^2 -> 0 near cloud top; the plume can shed at most
    # its own mass).
    detrained_depth = detrainment_rate * layer_thickness
    detrained_fraction = detrained_depth / (1.0 + detrained_depth)
    detrained_mass = mass_flux * detrained_fraction
    deposition = detrained_mass * (plume_property - env_property) / layer_mass

    return subsidence + deposition


# --- Buoyancy-sorting plume -> environment ------------------------------------
#
# Port of `apply_continuity_tendencies` (MSTCNV.F90:4794) for the bsort plume.
# ModelE applies the plume's effect on the environment in two distinct stages,
# and conflating them gets the answer wrong:
#
#   1. **Direct exchange.** The air the plume entrained is *removed* from its
#      layer (`dmr`, `dsmr`, `dqmr`, all negative) and the air it detrained is
#      *deposited* (`dm`, `dsm`, `dqm`). This is a local swap, not advection.
#   2. **Compensating subsidence.** The interface mass flux follows from
#      continuity over that exchange -- `cm(l) = cm(l-1) - dm(l) - dmr(l)` --
#      *not* from the plume's own mass. The environment is then upwind-advected
#      by it.
#
# The layer mass used for the advection is the post-exchange `ma + dmr + dm`,
# not the original.

class EnvironmentTendency(NamedTuple):
    """Per-layer change the convection imposes on the environment, extensive."""
    heat: jnp.ndarray            # change in SM (mass * potential-temperature)
    water: jnp.ndarray           # change in QM (mass * specific humidity)
    layer_mass: jnp.ndarray      # change in layer air mass
    interface_flux: jnp.ndarray  # cm, positive upward, at each layer top
    courant: jnp.ndarray         # max |cm|/ml; ModelE substeps above 0.999


def bsort_environment_tendencies(
        source_removal: jnp.ndarray,
        entrained_air: jnp.ndarray,
        detrained_mass: jnp.ndarray,
        detrained_heat: jnp.ndarray,
        detrained_water: jnp.ndarray,
        downdraft_mass: jnp.ndarray,
        downdraft_heat: jnp.ndarray,
        downdraft_water: jnp.ndarray,
        environment_heat: jnp.ndarray,
        environment_water: jnp.ndarray,
        layer_mass: jnp.ndarray,
        deposit_downdraft_locally: bool = True) -> EnvironmentTendency:
    """Environmental tendency from one buoyancy-sorting plume.

    Args:
        source_removal: Air drawn from the sub-cloud source layers to launch the
            plume, ``mplume*fpi(l)`` (``MSTCNV.F90:1573``), as a positive mass
            [kg/m^2]. Summing to the cloud-base mass flux, it is what makes the
            continuity integration balance -- without it the interface flux
            starts from zero at cloud base instead of carrying the plume's mass,
            and the environment gains mass from nowhere.
        entrained_air: ``envairm`` drawn out of each layer [kg/m^2].
        detrained_mass: Locally detrained blend mass [kg/m^2].
        detrained_heat: Its extensive heat.
        detrained_water: Its extensive water vapour.
        downdraft_mass: Blend mass routed to the downdraft [kg/m^2].
        downdraft_heat: Its extensive heat.
        downdraft_water: Its extensive water vapour.
        environment_heat: Intensive environmental heat (``senv``).
        environment_water: Intensive environmental humidity (``qenv``).
        layer_mass: ``ma`` [kg/m^2].
        deposit_downdraft_locally: See the note below. Static.

    Returns:
        An :class:`EnvironmentTendency`.

    Note:
        **The downdraft is an approximation here.** ModelE hands downdraft air to
        ``dd_evap_precip_loop`` (``MSTCNV.F90:3985-4792``, ~800 lines), which
        lets it descend, evaporate precipitation into it, and detrain lower
        down. That routine is not ported. Downdraft air is 29.8% of everything
        leaving the plume in BOMEX -- far too much to discard -- so by default it
        is deposited **at the level where it formed**, which conserves mass but
        places the associated cooling and moistening roughly 350 m too high and
        omits the evaporative cooling entirely. Set the flag to ``False`` to
        exclude it instead, which is *not* mass-conserving and is provided only
        for isolating its effect.
    """
    if deposit_downdraft_locally:
        deposited_mass = detrained_mass + downdraft_mass
        deposited_heat = detrained_heat + downdraft_heat
        deposited_water = detrained_water + downdraft_water
    else:
        deposited_mass = detrained_mass
        deposited_heat = detrained_heat
        deposited_water = detrained_water

    # Stage 1: the local swap. Removal is negative, deposition positive. The
    # source draw and the entrainment are both removals at the environment's own
    # properties, so they combine.
    removed_air = source_removal + entrained_air
    removed_heat = removed_air * environment_heat
    removed_water = removed_air * environment_water
    exchange_mass = deposited_mass - removed_air
    exchange_heat = deposited_heat - removed_heat
    exchange_water = deposited_water - removed_water

    # Stage 2: continuity gives the interface flux. `cm(l) = cm(l-1) - dm - dmr`
    # with `dmr = -entrained_air`, so the increment is `entrained_air - dm`.
    interface_flux = jnp.cumsum(removed_air - deposited_mass, axis=0)
    # No flux above the plume top. The integration must start below cloud base,
    # where `source_removal` lives, so it is not masked by `active`.
    carries_flux = (deposited_mass > 0.0) | (removed_air > 0.0)
    above_top = jnp.cumsum(carries_flux.astype(interface_flux.dtype)[::-1],
                           axis=0)[::-1] == 0
    interface_flux = jnp.where(above_top, 0.0, interface_flux)

    updated_mass = layer_mass + exchange_mass
    heat = environment_heat * layer_mass + exchange_heat
    water = environment_water * layer_mass + exchange_water

    # Upwind advection by `cmneg = -cm`: a negative (downward) flux carries the
    # layer above, a positive one the layer below.
    down_flux = -interface_flux
    heat_above = jnp.concatenate(
        [heat[1:], jnp.zeros((1,) + heat.shape[1:], dtype=heat.dtype)], axis=0)
    water_above = jnp.concatenate(
        [water[1:], jnp.zeros((1,) + water.shape[1:], dtype=water.dtype)],
        axis=0)
    mass_above = jnp.concatenate(
        [updated_mass[1:],
         jnp.ones((1,) + updated_mass.shape[1:], dtype=updated_mass.dtype)],
        axis=0)

    from_above = down_flux <= 0.0
    heat_flux = jnp.where(
        from_above, _safe_ratio(down_flux * heat_above, mass_above),
        _safe_ratio(down_flux * heat, updated_mass))
    water_flux = jnp.where(
        from_above, _safe_ratio(down_flux * water_above, mass_above),
        _safe_ratio(down_flux * water, updated_mass))

    def convergence(flux):
        below = jnp.concatenate(
            [jnp.zeros((1,) + flux.shape[1:], dtype=flux.dtype), flux[:-1]],
            axis=0)
        return below - flux

    advected_mass = convergence(down_flux)
    return EnvironmentTendency(
        heat=exchange_heat + convergence(heat_flux),
        water=exchange_water + convergence(water_flux),
        layer_mass=exchange_mass + advected_mass,
        interface_flux=interface_flux,
        courant=jnp.max(jnp.abs(_safe_ratio(interface_flux, updated_mass)),
                        axis=0),
    )
