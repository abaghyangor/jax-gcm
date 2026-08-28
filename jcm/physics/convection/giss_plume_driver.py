"""One buoyancy-sorting plume, from sub-cloud source to environmental tendency.

Composes the ported pieces of ``MSTCNV`` into the sequence the Fortran runs for
a single plume, and derives from the model state every profile those pieces
need. Until now each of them was driven from an oracle dump; this is what
supplies them from the column itself.

The chain, and where each stage lives:

1. :func:`column_geometry` -- ``gzl``, ``delz``, ``tvl``, ``rho0``, none of
   which the ``PhysicsState`` carries directly.
2. :func:`source_parcel` -- which sub-cloud layers the plume is drawn from
   (``fpi``), how much comes from each, and the parcel that results.
3. :func:`~jcm.physics.convection.giss_bsort.plume_ascent` -- the ascent.
4. :func:`convective_fraction` -- ``mcfrac``, which the descent needs and which
   is only knowable once the ascent has produced its mass flux and ``wcu``.
5. :func:`~jcm.physics.convection.giss_downdraft.downdraft_descent`.
6. :func:`~jcm.physics.convection.giss_tendencies.bsort_environment_tendencies`.

Scope: one plume. ``MSTCNV`` runs this inside two nested loops -- over cloud
base levels and over the plume spectrum -- updating the environment after each
plume, so the caller cannot simply sum what this returns. That sequencing is
deliberately left out; see ``BSORT_PORT_PLAN.md`` (W5).

Two inputs come from the turbulence scheme and have no jcm diagnostic yet:
``dcl`` (the dry convective layer top, ``smixlev`` in ModelE) and ``wturb``,
which sets ``cloud_base_velocity``. Both are arguments here rather than being
guessed at. ``wturb`` never lifts the cloud-base speed off its 0.5 floor
anywhere in BOMEX, so the default is exact for that case and only that case.

Units follow ``MSTCNV``, not the usual meteorological convention: ``exner`` is
``plk = (p in mb)**kappa`` and ``potential_temperature`` is ``th =
theta/1000**kappa``, so that ``th*plk`` is the temperature in kelvin. Passing a
conventional potential temperature against this Exner function overstates every
temperature by ``1000**kappa`` (about sevenfold), which lands outside the
saturation fits' validity range rather than merely being inaccurate.

Broadcasting-native: vertical on axis 0, trailing axes horizontal.
"""

from typing import NamedTuple

import jax.numpy as jnp

from jcm.physics.convection import giss_bsort as bsort
from jcm.physics.convection import giss_downdraft as downdraft
from jcm.physics.convection import giss_tendencies as tendencies
from jcm.physics.convection.giss_thermodynamics import (
    DELTX, RGAS, safe_divide)

# `bsort_enteff(2)`, the entraining plume's entrainment efficiency
# (MSTCNV.F90:282). The less-entraining plume never fires under
# `lessent_scheme = 2`, so its 0.1 counterpart is not needed here.
_ENTRAINMENT_EFFICIENCY = 0.67

# `wbases(2) = max(0.5, wturb)` (MSTCNV.F90:2838). Measured at exactly 0.5 in
# all 52 BOMEX plumes, where `max(wturb)` peaks at 0.372.
_CLOUD_BASE_VELOCITY = 0.5

# `ccmul`, the multiplier on the updraft mass flux in the convective fraction.
# 2.0 unless `see_debris` is set (MSTCNV.F90:588-593).
_CCMUL = 2.0

# `qboost` scales the source parcel's humidity. Both branches reduce to 1.0
# whenever `mc_tqstar_fac > 0` (MSTCNV.F90:2624-2625, 2675-2682), which every
# active preset sets, so the boost is inert and kept only to name the term.
_QBOOST = 1.0


class ColumnGeometry(NamedTuple):
    """Profiles ``MSTCNV`` derives from the column before any plume runs."""
    layer_depth: jnp.ndarray           # gzl [m], centred geopotential spacing
    layer_thickness: jnp.ndarray       # delz [m], hydrostatic
    virtual_temperature: jnp.ndarray   # tvl [K]
    density: jnp.ndarray               # rho0 [kg/m^3]
    temperature: jnp.ndarray           # tl [K]


class PlumeCycle(NamedTuple):
    """Everything one plume does to the column."""
    tendency: tendencies.EnvironmentTendency
    ascent: bsort.PlumeAscent
    descent: downdraft.DowndraftDescent
    geometry: ColumnGeometry
    source_removal: jnp.ndarray        # mplume*fpi(l), positive [kg/m^2]
    convective_fraction: jnp.ndarray   # mcfrac
    cloud_top: jnp.ndarray             # lmax, highest level the plume reached


def column_geometry(potential_temperature: jnp.ndarray,
                    specific_humidity: jnp.ndarray,
                    layer_mass: jnp.ndarray,
                    exner: jnp.ndarray,
                    pressure: jnp.ndarray,
                    height: jnp.ndarray) -> ColumnGeometry:
    """Derive the geometric and virtual-temperature profiles.

    Args:
        potential_temperature: ``th`` [K].
        specific_humidity: ``qv`` [kg/kg].
        layer_mass: ``ma`` [kg/m^2].
        exner: ``plk``.
        pressure: [**Pa**].
        height: ``zl``, geopotential height at layer centres [m]. ModelE forms
            it as ``gz*bygrav`` (``ATM_UTILS.f:874``).

    Returns:
        A :class:`ColumnGeometry`.

    Note:
        ``gzl`` and ``delz`` measure the same layer two different ways and
        ModelE keeps both -- the Fortran carries a standing "why different than
        gzl?" comment at ``MSTCNV.F90:3459``. ``gzl`` is a centred difference of
        geopotential height across the *neighbouring* layer centres
        (``CLOUDS_DRV.F90:582``), so it spans half of each adjacent layer;
        ``delz`` is this layer's own hydrostatic thickness. The entrainment
        rates are per ``gzl`` and the buoyancy work integral is per ``delz``, so
        conflating them mis-scales one or the other.
    """
    temperature = potential_temperature * exner
    density = pressure / (RGAS * temperature)
    virtual_temperature = temperature * (1.0 + DELTX * specific_humidity)
    layer_thickness = safe_divide(layer_mass, density)

    # `GZL(L+1) = .5*(GZ(L+2)-GZ(L))*BYGRAV`, i.e. a centred difference over the
    # layer centres either side. The bottom layer gets zero and the top repeats
    # the level below it, as in CLOUDS_DRV.F90:584-585.
    above = jnp.concatenate([height[1:], height[-1:]], axis=0)
    below = jnp.concatenate([height[:1], height[:-1]], axis=0)
    layer_depth = 0.5 * (above - below)
    nlev = height.shape[0]
    level = jnp.arange(nlev).reshape((nlev,) + (1,) * (height.ndim - 1))
    layer_depth = jnp.where(level == 0, 0.0, layer_depth)
    layer_depth = jnp.where(level == nlev - 1,
                            jnp.take(layer_depth, nlev - 2, axis=0),
                            layer_depth)
    return ColumnGeometry(layer_depth=layer_depth,
                          layer_thickness=layer_thickness,
                          virtual_temperature=virtual_temperature,
                          density=density,
                          temperature=temperature)


def source_weights(layer_mass: jnp.ndarray,
                   source_bottom: jnp.ndarray,
                   source_top: jnp.ndarray,
                   boundary_layer_top: jnp.ndarray) -> jnp.ndarray:
    """``fpi``: the fraction of the plume drawn from each sub-cloud layer.

    Mass-weighted over ``lmin0..lmin`` (``MSTCNV.F90:2672``). When the parcel is
    lifted from below the boundary-layer top to a base above it, the layers
    above that top are dropped and the rest renormalised, so a plume rooted in
    the mixed layer is not diluted with free-tropospheric air on the way up
    (``MSTCNV.F90:2673-2675``).
    """
    nlev = layer_mass.shape[0]
    level = jnp.arange(nlev).reshape((nlev,) + (1,) * (layer_mass.ndim - 1))
    in_source = (level >= source_bottom) & (level <= source_top)
    displaced = ((source_top > boundary_layer_top)
                 & (source_bottom <= boundary_layer_top))
    above_boundary_layer = displaced & (level > boundary_layer_top)
    weight = jnp.where(in_source & ~above_boundary_layer, layer_mass, 0.0)
    return safe_divide(weight, jnp.sum(weight, axis=0))


def source_parcel(potential_temperature: jnp.ndarray,
                  specific_humidity: jnp.ndarray,
                  layer_mass: jnp.ndarray,
                  plume_mass: jnp.ndarray,
                  source_bottom: jnp.ndarray,
                  source_top: jnp.ndarray,
                  boundary_layer_top: jnp.ndarray):
    """Draw the plume out of the sub-cloud layers.

    ModelE removes ``mplume*fpi(l)`` from each source layer and gives the plume
    what that air carried (``MSTCNV.F90:1583-1601``). The parcel starts with no
    condensate: it condenses on arrival at cloud base, not before.

    Returns:
        ``(source_removal, heat, water, condensate)`` -- the per-layer removal
        as a positive mass, and the parcel's extensive heat, vapour and
        condensate.
    """
    fpi = source_weights(layer_mass, source_bottom, source_top,
                         boundary_layer_top)
    source_removal = plume_mass * fpi
    heat = jnp.sum(source_removal * potential_temperature, axis=0)
    water = jnp.sum(source_removal * specific_humidity, axis=0) * _QBOOST
    return source_removal, heat, water, jnp.zeros_like(heat)


def convective_fraction(plume_mass: jnp.ndarray,
                        vertical_velocity: jnp.ndarray,
                        density: jnp.ndarray,
                        cloud_base: jnp.ndarray,
                        cloud_top: jnp.ndarray,
                        timestep: jnp.ndarray,
                        ccmul: jnp.ndarray = _CCMUL) -> jnp.ndarray:
    """``mcfrac``: the convective area fraction at each layer edge.

    The ratio of the mass entering a layer to how fast the updraft is carrying
    it (``MSTCNV.F90:5136-5146``). ``ccm(l)``, the mass flux at the edge *below*
    level ``l+1``, is the plume mass entering that level (``MSTCNV.F90:1749``),
    which is what :class:`~jcm.physics.convection.giss_bsort.PlumeAscent` calls
    ``plume_mass``. Below cloud base ModelE holds the cloud-base value rather
    than letting it fall to zero, representing virga.
    """
    nlev = plume_mass.shape[0]
    level = jnp.arange(nlev).reshape((nlev,) + (1,) * (plume_mass.ndim - 1))
    mass_flux = jnp.concatenate(
        [plume_mass[1:], jnp.zeros((1,) + plume_mass.shape[1:])], axis=0)
    fraction = jnp.minimum(
        1.0, safe_divide(ccmul * mass_flux,
                         density * vertical_velocity * timestep))
    fraction = jnp.where((level >= cloud_base) & (level < cloud_top),
                         fraction, 0.0)
    # Virga: every level below cloud base takes the cloud-base value, which the
    # Fortran reaches by copying downward one layer at a time.
    at_base = jnp.sum(jnp.where(level == cloud_base, fraction, 0.0), axis=0)
    return jnp.where(level < cloud_base, at_base, fraction)


def run_plume(cloud_base: jnp.ndarray,
              source_bottom: jnp.ndarray,
              source_top: jnp.ndarray,
              boundary_layer_top: jnp.ndarray,
              cloud_base_mass: jnp.ndarray,
              potential_temperature: jnp.ndarray,
              specific_humidity: jnp.ndarray,
              layer_mass: jnp.ndarray,
              exner: jnp.ndarray,
              pressure: jnp.ndarray,
              height: jnp.ndarray,
              timestep: jnp.ndarray,
              environment_condensate: jnp.ndarray = None,
              cloud_base_velocity: jnp.ndarray = _CLOUD_BASE_VELOCITY,
              entrainment_efficiency: jnp.ndarray = _ENTRAINMENT_EFFICIENCY,
              phase: str = "water") -> PlumeCycle:
    """Run one plume and return what it does to the environment.

    Args:
        cloud_base: ``lcl``, the level the ascent starts at. Normally
            ``source_top + 1``.
        source_bottom: ``lmin0``, lowest layer the plume draws from.
        source_top: ``lmin``, highest layer it draws from.
        boundary_layer_top: ``dcl``. Bounds the source draw, and is where the
            downdraft is forced to detrain. Comes from the turbulence scheme
            (``smixlev``); there is no jcm diagnostic for it yet.
        cloud_base_mass: ``mplume``, from the cloud-base closure [kg/m^2].
        potential_temperature: ``th`` [K].
        specific_humidity: ``qv`` [kg/kg].
        layer_mass: ``ma`` [kg/m^2].
        exner: ``plk``.
        pressure: [**Pa**].
        height: ``zl`` [m].
        timestep: ``dtsrc`` [s], used only for ``mcfrac``.
        environment_condensate: ``qcl + qci``, which loads the environment in
            the downdraft's buoyancy test. Defaults to zero.
        cloud_base_velocity: ``wbases(iplume)`` [m/s].
        entrainment_efficiency: ``enteff``.
        phase: ``"water"`` or ``"ice"``. Static.

    Returns:
        A :class:`PlumeCycle`.
    """
    geometry = column_geometry(potential_temperature, specific_humidity,
                               layer_mass, exner, pressure, height)

    removal, parcel_heat, parcel_water, parcel_condensate = source_parcel(
        potential_temperature, specific_humidity, layer_mass, cloud_base_mass,
        source_bottom, source_top, boundary_layer_top)

    # `plume_ascent` takes its seed already condensed -- it skips the arrival
    # processing at the base level, because that is where ModelE's ascent loop
    # begins rather than something it does again. The sub-cloud parcel is still
    # unsaturated, so the condensation it undergoes on reaching cloud base has
    # to happen here.
    def at_base(profile):
        nlev = profile.shape[0]
        level = jnp.arange(nlev).reshape((nlev,) + (1,) * (profile.ndim - 1))
        return jnp.sum(jnp.where(level == cloud_base, profile, 0.0), axis=0)

    seed_heat, seed_water, seed_condensate = bsort.resaturate_plume(
        cloud_base_mass, parcel_heat, parcel_water, parcel_condensate,
        at_base(exner), at_base(pressure), phase)

    ascent = bsort.plume_ascent(
        cloud_base=cloud_base,
        cloud_base_mass=cloud_base_mass,
        cloud_base_heat=seed_heat,
        cloud_base_water=seed_water,
        cloud_base_condensate=seed_condensate,
        environment_heat=potential_temperature,
        environment_water=specific_humidity,
        environment_virtual_temperature=geometry.virtual_temperature,
        layer_mass=layer_mass,
        layer_depth=geometry.layer_depth,
        layer_thickness=geometry.layer_thickness,
        height=height,
        exner=exner,
        pressure=pressure,
        entrainment_efficiency=entrainment_efficiency,
        cloud_base_velocity=cloud_base_velocity,
        phase=phase)

    nlev = layer_mass.shape[0]
    level = jnp.arange(nlev).reshape((nlev,) + (1,) * (layer_mass.ndim - 1))
    plume_top = jnp.max(jnp.where(ascent.active, level, 0), axis=0)
    # `ldraft`, the highest level the downdraft was seeded at.
    downdraft_top = jnp.max(jnp.where(ascent.downdraft_mass > 0.0, level, 0),
                            axis=0)

    mcfrac = convective_fraction(ascent.plume_mass, ascent.vertical_velocity,
                                 geometry.density, cloud_base, plume_top,
                                 timestep)

    if environment_condensate is None:
        environment_condensate = jnp.zeros_like(layer_mass)

    descent = downdraft.downdraft_descent(
        source_mass=ascent.downdraft_mass,
        source_heat=ascent.downdraft_heat,
        source_water=ascent.downdraft_water,
        precipitation=ascent.downdraft_condensate,
        produced_precipitation=ascent.precipitation,
        environment_heat=potential_temperature,
        environment_water=specific_humidity,
        environment_condensate=environment_condensate,
        layer_mass=layer_mass,
        layer_depth=geometry.layer_depth,
        convective_fraction=mcfrac,
        exner=exner,
        pressure=pressure,
        cloud_top=downdraft_top,
        # `max(lcl-1, dcl)`: the downdraft sheds itself once it is back in the
        # mixed layer, or below the cloud it came from, whichever is higher.
        boundary_layer_top=jnp.maximum(cloud_base - 1, boundary_layer_top),
        cloud_base=cloud_base,
        phase=phase)

    tendency = tendencies.bsort_environment_tendencies(
        source_removal=removal,
        entrained_air=ascent.entrained_air,
        detrained_mass=ascent.detrained_mass,
        detrained_heat=ascent.detrained_heat,
        detrained_water=ascent.detrained_water,
        downdraft_detrained_mass=descent.detrained_mass,
        downdraft_detrained_heat=descent.detrained_heat,
        downdraft_detrained_water=descent.detrained_water,
        downdraft_entrained_air=descent.entrained_air,
        environment_heat=potential_temperature,
        environment_water=specific_humidity,
        layer_mass=layer_mass,
        evaporation_heat=descent.environment_heat,
        evaporation_water=descent.environment_water)

    return PlumeCycle(tendency=tendency, ascent=ascent, descent=descent,
                      geometry=geometry, source_removal=removal,
                      convective_fraction=mcfrac, cloud_top=plume_top)
