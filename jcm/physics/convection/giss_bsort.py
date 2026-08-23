"""Buoyancy-sorting plume entrainment and detrainment -- GISS ModelE.

Port of ``plume_ent_det_w2_bsort`` (``modelE/model/MSTCNV.F90``). This is the
routine ModelE actually runs: ``MSTCNV`` selects between two plume ascent
schemes on ``bsort_entdet``, which ``sync_param`` defaults to ``.true.`` and
which the SCM rundecks do not override. (The alternative,
``plume_ent_det_w2`` -- Gregory (2001) entrainment with an explicit downdraft
mass diversion -- is unreachable in this configuration.)

**How it differs from a conventional mass-flux plume.** There is no single
entrainment rate that mixes environmental air into the whole updraft. Instead,
per level:

1. an entrainment *rate* is computed from an energy closure (not from the local
   buoyancy over ``w^2``), and used only to *size* how much updraft and
   environmental air is set aside;
2. that air is split into a small spectrum of blends spanning a range of
   updraft/environment mixing ratios;
3. each blend evaporates its condensate and is then tested for buoyancy;
4. buoyant blends rejoin the updraft, strongly negative ones seed the downdraft,
   and the rest detrain locally.

Entrainment and detrainment are therefore *emergent from the sorting*: the
plume's mass can grow or decay smoothly with height, rather than growing until
it terminates. The ``ENT``/``DET`` values are step-1 inputs, not the realized
exchange -- reading them as the exchange is why the plume mass budget appeared
not to close during this port.

Every formula here was verified against a dump taken from inside ModelE's own
ascent loop (bridge repo ``oracle_data/bomex_mass_budget.txt`` and
``bomex_blend_diag.txt``); see ``BSORT_PORT_PLAN.md`` for the agreement table.

Broadcasting-native, per the repository convention: these are *per-level*
helpers intended for use inside the ascent scan, so their inputs have shape
``(*horiz)`` and the blend spectrum occupies a **leading** axis of size
``nmix``.
"""

from typing import NamedTuple

import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.convection.giss_thermodynamics import (
    DELTX, LHE, LHS, SHA, condensate_evaporation)

# Fraction of the buoyancy force that goes into vertical kinetic energy. ModelE
# applies only a sixth while the parcel is buoyant, but the full force once it
# is negatively buoyant, so overshoots decelerate sharply (MSTCNV.F90).
_A_BUOY_BUOYANT = 1.0 / 6.0
_A_BUOY_OVERSHOOT = 1.0

# Entrainment rate limits [1/m]. The floors apply only where the plume is
# buoyant; the ceiling always applies.
_ENT_FLOOR_LESS_ENTRAINING = 3.0e-4   # iplume == 1, and only below 700 mb
_ENT_FLOOR_MORE_ENTRAINING = 5.0e-4   # iplume == 2
_ENT_MAX = 4.0e-3
_ENT_FLOOR_PRESSURE = 7.0e4           # ModelE tests `pl(l) > 700` in mb

# Cap on `sqrt(kew/(bdzsum*mplume)) - 1`, i.e. on the dilution the closure may
# ask for in a single level.
_MAX_DILUTION = 0.95

# Fixed detrainment rate applied where the buoyancy integral has gone negative,
# expressed per layer depth. ModelE's comment flags this as a placeholder for a
# treatment closer to the original scheme.
_DET_RATE = 0.5

# Ceiling on entrained environmental air as a fraction of the layer's mass.
_REMRAT = 0.333

# Reference blend spectrum: `frac` is each blend's share of the entrained
# environmental air, `fupd` its share of updraft air. Both are ModelE constants.
_BLEND_FRACTION = jnp.array([0.25, 0.50, 0.25])
_BLEND_FUPD_REFERENCE = jnp.array([0.25, 0.50, 0.75])
NMIX = 3

# ModelE's `updairm_max = .99999d0*mplume` guard against entraining more updraft
# air than the plume holds.
_UPDRAFT_FRACTION_MAX = 0.99999

_TEENY = 1e-20


def buoyancy_work_increments(buoyancy: jnp.ndarray,
                             plume_mass: jnp.ndarray,
                             layer_thickness: jnp.ndarray,
                             entrainment_efficiency: jnp.ndarray):
    """Increments to the two running integrals ``kew`` and ``bdzsum``.

    ``kew`` accumulates the buoyancy work done on the plume; ``bdzsum``
    accumulates the same work discounted by the entrainment efficiency. Their
    ratio is what the closure below turns into an entrainment rate, so both must
    be carried up the column together.

    Args:
        buoyancy: ``(tvp - tvl)/tvl - wmp/mplume`` at this level [-].
        plume_mass: Plume mass ``mplume`` entering the level [kg/m^2].
        layer_thickness: ``delz = ma(l)/rho0(l)`` [m]. Note ModelE computes this
            separately from ``gzl(l)`` and flags in-source that the two differ;
            they are not interchangeable.
        entrainment_efficiency: ``enteff``, the target fraction of the maximum
            achievable vertical velocity. Constant 0.67 in the SCM cases tested,
            but in general set per plume from ``closure2`` and the cold-pool
            fraction, so it is an argument rather than a constant.

    Returns:
        ``(d_kew, d_bdzsum)`` to add to the running integrals.
    """
    a_buoy = jnp.where(buoyancy >= 0.0, _A_BUOY_BUOYANT, _A_BUOY_OVERSHOOT)
    bdz = a_buoy * c.grav * buoyancy * layer_thickness
    return bdz * plume_mass, bdz * (1.0 - entrainment_efficiency)


def entrainment_rate(buoyancy: jnp.ndarray,
                     plume_mass: jnp.ndarray,
                     layer_thickness: jnp.ndarray,
                     kew: jnp.ndarray,
                     bdzsum: jnp.ndarray,
                     pressure: jnp.ndarray,
                     iplume: int = 2):
    """Entrainment and detrainment rates [1/m] -- ModelE's energy closure.

    The rate is *not* proportional to buoyancy over ``w^2``. It is chosen so that
    the dilution it causes brings the local vertical velocity down to the target
    fraction ``enteff`` of the maximum the buoyancy history could have supported:

        ent = min(0.95, sqrt(kew/(bdzsum*mplume)) - 1) / delz

    while the buoyancy integral is positive, and a fixed ``-0.5/delz`` once it is
    negative -- which the sign convention then converts into detrainment.

    Args:
        buoyancy: Plume buoyancy at this level [-].
        plume_mass: Plume mass entering the level [kg/m^2].
        layer_thickness: ``delz`` [m].
        kew: Running buoyancy-work integral, **after** adding this level's
            increment.
        bdzsum: Running discounted integral, likewise after this level.
        pressure: Level pressure [**Pa**] (ModelE tests ``pl(l)`` in mb).
        iplume: Which plume this is (1 = less entraining, 2 = more entraining).
            Static configuration, so it branches in Python.

    Returns:
        ``(ent, det)`` -- non-negative entrainment and detrainment rates [1/m].
        At most one of the two is non-zero.
    """
    denominator = bdzsum * plume_mass
    buoyant = kew > denominator

    # `jnp.where` evaluates both branches, so the ratio must be safe even where
    # it is discarded: ModelE never reaches the sqrt with a non-positive
    # denominator (verified 0/502 on the oracle), but an unguarded sqrt of a
    # negative would still poison the gradient here.
    safe_denominator = jnp.where(buoyant, denominator, 1.0)
    ratio = jnp.where(buoyant, kew / safe_denominator, 1.0)

    ent = jnp.minimum(_MAX_DILUTION, jnp.sqrt(ratio) - 1.0) / layer_thickness
    ent = jnp.where(buoyant, ent, -_DET_RATE / layer_thickness)

    # Floors, only where the plume is buoyant. For the less-entraining plume
    # ModelE applies the floor only below 700 mb; for the other, always.
    if iplume == 1:
        floor_applies = (buoyancy > 0.0) & (pressure > _ENT_FLOOR_PRESSURE)
        floor_value = _ENT_FLOOR_LESS_ENTRAINING
    else:
        floor_applies = buoyancy > 0.0
        floor_value = _ENT_FLOOR_MORE_ENTRAINING
    ent = jnp.where(floor_applies, jnp.maximum(ent, floor_value), ent)

    ent = jnp.minimum(ent, _ENT_MAX)

    # A negative rate means detrainment, held as a separate non-negative number.
    det = jnp.maximum(-ent, 0.0)
    ent = jnp.maximum(ent, 0.0)
    return ent, det


def blend_air_masses(plume_mass: jnp.ndarray,
                     cloud_base_mass: jnp.ndarray,
                     plume_mass_lag: jnp.ndarray,
                     entrainment: jnp.ndarray,
                     detrainment: jnp.ndarray,
                     layer_depth: jnp.ndarray,
                     layer_mass: jnp.ndarray):
    """Size the air set aside for sorting and weight the blend spectrum.

    Two regimes, which ModelE writes as an if/else on whether any environmental
    air is entrained:

    * **Normal.** ``envairm`` environmental air is entrained and combined with
      ``updairm`` updraft air across ``NMIX`` blends. The blending ratios drift
      from the reference ``[1/4, 1/2, 3/4]`` toward complete mixing wherever the
      plume has thinned relative to ~1 km below (``refblendwt``), on the
      reasoning that a thin plume leaves less updraft air able to escape mixing.
    * **Overshooting** (``envairm <= 0``, i.e. the buoyancy integral has gone
      negative so ``ent`` is zero). No environmental air is available, and ModelE
      builds a single blend of pure updraft air.

    JAX cannot vary the number of blends, so the overshooting case is expressed
    on the same fixed ``NMIX`` axis by putting all the weight on the first blend
    and zero on the rest. Zero-mass blends carry no mass to any fate, so this is
    equivalent to ModelE's ``nmix = 1``.

    Args:
        plume_mass: ``mplume`` entering the level [kg/m^2].
        cloud_base_mass: ``mplume_b``, the plume's cloud-base mass [kg/m^2].
        plume_mass_lag: ``mplume_lag``, the plume mass roughly 1 km below
            [kg/m^2].
        entrainment: Entrainment rate [1/m].
        detrainment: Detrainment rate [1/m].
        layer_depth: ``gzl(l)`` [m]. Distinct from the ``delz`` used above.
        layer_mass: ``ma(l)`` [kg/m^2].

    Returns:
        ``(environment_air, updraft_air, updraft_factor, environment_factor,
        fupd)`` where the last three have a leading axis of length ``NMIX``.
        ``updraft_factor``/``environment_factor`` multiply the *totals*
        ``updraft_air``/``environment_air`` to give each blend's contribution.
    """
    blend_shape = (NMIX,) + (1,) * jnp.ndim(plume_mass)
    frac = _BLEND_FRACTION.reshape(blend_shape)
    fupd_reference = _BLEND_FUPD_REFERENCE.reshape(blend_shape)

    environment_air = (jnp.minimum(plume_mass, 2.0 * cloud_base_mass)
                       * (entrainment * layer_depth))
    # Numerical stability aid; never binds in the cases tested (0/533).
    environment_air = jnp.minimum(environment_air, layer_mass * _REMRAT)

    entraining = environment_air > 0.0

    # Blending ratios, drifting toward complete mixing as the plume thins.
    reference_weight = jnp.minimum(
        1.0, plume_mass / jnp.maximum(plume_mass_lag, _TEENY))
    fupd_full_mixing = plume_mass / jnp.maximum(
        plume_mass + environment_air, _TEENY)
    fupd = (reference_weight * fupd_reference
            + (1.0 - reference_weight) * fupd_full_mixing)

    # Updraft air implied by those ratios and the entrained environmental air.
    environment_share = jnp.sum(frac * (1.0 - fupd), axis=0)
    updraft_air = environment_air * (
        1.0 / jnp.maximum(environment_share, _TEENY) - 1.0)

    # Never set aside more updraft air than the plume holds; ModelE scales the
    # entrained air back by the same factor, which is a reduction of the
    # effective entrainment rate.
    updraft_air_max = _UPDRAFT_FRACTION_MAX * plume_mass
    excessive = updraft_air > updraft_air_max
    rescale = jnp.where(excessive, updraft_air_max / jnp.maximum(
        updraft_air, _TEENY), 1.0)
    environment_air = environment_air * rescale
    updraft_air = jnp.where(excessive, updraft_air_max, updraft_air)

    fupd_average = jnp.sum(frac * fupd, axis=0)
    updraft_factor = frac * fupd / jnp.maximum(fupd_average, _TEENY)
    environment_factor = (frac * (1.0 - fupd)
                          / jnp.maximum(1.0 - fupd_average, _TEENY))

    # Overshooting: one blend of pure updraft air, no environmental air.
    overshoot_updraft_air = plume_mass * jnp.minimum(
        0.95, detrainment * layer_depth)
    first_blend = (jnp.arange(NMIX).reshape(blend_shape) == 0)
    updraft_factor = jnp.where(
        entraining, updraft_factor, jnp.where(first_blend, 1.0, 0.0))
    environment_factor = jnp.where(entraining, environment_factor, 0.0)
    updraft_air = jnp.where(entraining, updraft_air, overshoot_updraft_air)
    environment_air = jnp.where(entraining, environment_air, 0.0)

    return (environment_air, updraft_air, updraft_factor, environment_factor,
            fupd)


# Virtual-temperature thresholds the sorted blends are tested against, in K
# before division by the environment's virtual temperature (MSTCNV.F90).
_POSITIVE_BUOYANCY = 0.05    # above this a blend rejoins the updraft
_NEGATIVE_BUOYANCY = -0.2    # below this it seeds the downdraft
# Once the plume itself is this negatively buoyant it is overshooting, and
# downdraft formation is switched off -- negative blends detrain locally
# instead. Omitting this guard misroutes blends on exactly the levels where the
# plume is dying (verified: 1566/1572 without it, 1572/1572 with it).
_OVERSHOOT_BUOYANCY = -0.25


class SortedBlends(NamedTuple):
    """Outcome of sorting one level's blends.

    ``*_mass``/``*_heat``/``*_water``/``*_condensate`` are extensive. The three
    fates partition the blend air exactly, and together with the retained plume
    they conserve mass: ``plume_mass + detrained_mass + downdraft_mass`` equals
    the incoming plume mass plus the entrained environmental air.
    """
    plume_mass: jnp.ndarray
    plume_heat: jnp.ndarray
    plume_water: jnp.ndarray
    plume_condensate: jnp.ndarray
    detrained_mass: jnp.ndarray
    detrained_heat: jnp.ndarray
    detrained_water: jnp.ndarray
    detrained_condensate: jnp.ndarray
    downdraft_mass: jnp.ndarray
    downdraft_heat: jnp.ndarray
    downdraft_water: jnp.ndarray
    downdraft_condensate: jnp.ndarray
    mixture_buoyancy: jnp.ndarray


def sort_blends(plume_mass: jnp.ndarray,
                plume_heat: jnp.ndarray,
                plume_water: jnp.ndarray,
                plume_condensate: jnp.ndarray,
                environment_heat: jnp.ndarray,
                environment_water: jnp.ndarray,
                environment_air: jnp.ndarray,
                updraft_air: jnp.ndarray,
                updraft_factor: jnp.ndarray,
                environment_factor: jnp.ndarray,
                exner: jnp.ndarray,
                pressure: jnp.ndarray,
                environment_virtual_temperature: jnp.ndarray,
                buoyancy: jnp.ndarray,
                phase: str = "water") -> SortedBlends:
    """Build the blend spectrum, evaporate, and sort each blend by buoyancy.

    This is the step that makes the scheme what it is. ModelE first *removes*
    ``updraft_air`` from the plume, builds blends of it with entrained
    environmental air, and then only returns the blends that turn out buoyant.
    Everything else is gone: strongly negative blends seed the downdraft and the
    rest detrain. That asymmetry -- remove first, return conditionally -- is
    what lets plume mass decay smoothly with height.

    The evaporation is a **test only**. ModelE evaporates each blend's
    condensate to see whether the resulting cooling makes it negatively buoyant,
    but the mass that rejoins the updraft carries its *original* heat, water and
    condensate (``MSTCNV.F90``: the branches add back the untouched extensive
    ``smmix``/``qmmix``/``wmmix``). Actual phase change is handled by the
    microphysics, not here, so applying the evaporation to the returned
    properties would double-count it.

    Args:
        plume_mass: ``mplume`` entering the level [kg/m^2].
        plume_heat: ``smp``, extensive heat content (potential-temperature-like).
        plume_water: ``qmp``, extensive water vapour.
        plume_condensate: ``wmp``, extensive condensate.
        environment_heat: ``senv``, *intensive* environmental heat.
        environment_water: ``qenv``, *intensive* environmental humidity.
        environment_air: Entrained environmental air [kg/m^2].
        updraft_air: Updraft air set aside for sorting [kg/m^2].
        updraft_factor: Per-blend share of ``updraft_air``, leading axis
            ``NMIX``.
        environment_factor: Per-blend share of ``environment_air``.
        exner: ``plk(l)``, ModelE's ``p[mb]**KAPA``.
        pressure: Level pressure [**Pa**].
        environment_virtual_temperature: ``tvl(l)`` [K].
        buoyancy: Plume buoyancy at this level, for the overshooting guard.
        phase: ``"water"`` or ``"ice"``. Static.

    Returns:
        A :class:`SortedBlends`.
    """
    latent_heat = LHE if phase == "water" else LHS
    slh = latent_heat / SHA

    # Remove the set-aside air from the plume. What survives is the part that
    # never participates in mixing at this level.
    removed_fraction = updraft_air / jnp.maximum(plume_mass, _TEENY)
    retained = 1.0 - removed_fraction
    set_aside_heat = plume_heat * removed_fraction
    set_aside_water = plume_water * removed_fraction
    set_aside_condensate = plume_condensate * removed_fraction

    # Environmental air carries no condensate into the blends.
    entrained_heat = environment_air * environment_heat
    entrained_water = environment_air * environment_water

    # Per-blend extensive amounts. `updraft_factor`/`environment_factor` lead
    # with the NMIX axis, so these do too.
    blend_mass = updraft_factor * updraft_air + environment_factor * environment_air
    blend_heat = (updraft_factor * set_aside_heat
                  + environment_factor * entrained_heat)
    blend_water = (updraft_factor * set_aside_water
                   + environment_factor * entrained_water)
    blend_condensate = updraft_factor * set_aside_condensate

    # Intensive properties. Zero-mass blends (the overshooting branch's unused
    # slots) divide safely and end up strongly negative, but since they carry no
    # mass their fate has no effect.
    safe_mass = jnp.maximum(blend_mass, _TEENY)
    blend_t = blend_heat / safe_mass
    blend_q = blend_water / safe_mass
    blend_condensate_intensive = blend_condensate / safe_mass

    evaporated, _ = condensate_evaporation(
        blend_t, blend_q, exner, 1.0, pressure, blend_condensate_intensive,
        phase)
    test_t = blend_t - slh * evaporated / exner
    test_q = blend_q + evaporated
    test_condensate = blend_condensate_intensive - evaporated

    virtual_t = test_t * exner * (1.0 + DELTX * test_q)
    mixture_buoyancy = ((virtual_t - environment_virtual_temperature)
                        / environment_virtual_temperature - test_condensate)

    overshooting = buoyancy <= _OVERSHOOT_BUOYANCY / environment_virtual_temperature
    rejoins = mixture_buoyancy > _POSITIVE_BUOYANCY / environment_virtual_temperature
    to_downdraft = (
        (mixture_buoyancy < _NEGATIVE_BUOYANCY / environment_virtual_temperature)
        & ~overshooting & ~rejoins)
    detrains = ~rejoins & ~to_downdraft

    def _gather(mask, field):
        return jnp.sum(jnp.where(mask, field, 0.0), axis=0)

    return SortedBlends(
        plume_mass=plume_mass * retained + _gather(rejoins, blend_mass),
        plume_heat=plume_heat * retained + _gather(rejoins, blend_heat),
        plume_water=plume_water * retained + _gather(rejoins, blend_water),
        plume_condensate=(plume_condensate * retained
                          + _gather(rejoins, blend_condensate)),
        detrained_mass=_gather(detrains, blend_mass),
        detrained_heat=_gather(detrains, blend_heat),
        detrained_water=_gather(detrains, blend_water),
        detrained_condensate=_gather(detrains, blend_condensate),
        downdraft_mass=_gather(to_downdraft, blend_mass),
        downdraft_heat=_gather(to_downdraft, blend_heat),
        downdraft_water=_gather(to_downdraft, blend_water),
        downdraft_condensate=_gather(to_downdraft, blend_condensate),
        mixture_buoyancy=mixture_buoyancy,
    )
