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
from jax import lax

from jcm.physics.convection import giss_microphysics as microphysics
from jcm.physics.convection.giss_thermodynamics import (
    DELTX, GRAV, LHE, LHS, RGAS as RGAS_AIR, SHA, condensate_evaporation,
    condensation, safe_divide)

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
    # `GRAV` is ModelE's 9.80665, not jcm's 9.81. The 0.034% difference is
    # negligible as physics but not as a port: it enters the vertical kinetic
    # energy at every level, and the accumulated error in `w` feeds the
    # entrainment closure and the condensate loading, which is enough to move a
    # marginal blend across the buoyancy-sorting threshold.
    bdz = a_buoy * GRAV * buoyancy * layer_thickness
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
    # it is discarded. Two ways it can go wrong, both invisible in the forward
    # value and both fatal to the gradient:
    #   - the non-buoyant branch would take the sqrt of a negative ratio;
    #   - a *terminated* plume has zero mass, hence a zero denominator, while
    #     still carrying kew > 0 from when it was alive -- so `buoyant` is true
    #     and the division is by zero.
    # ModelE reaches the sqrt with a non-positive denominator on 0/502 live
    # levels, so restricting the division to a strictly positive denominator
    # changes nothing physical; it only keeps the dead levels differentiable.
    usable = buoyant & (denominator > 0.0)
    safe_denominator = jnp.where(usable, denominator, 1.0)
    ratio = jnp.where(usable, kew / safe_denominator, 1.0)

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
        1.0, safe_divide(plume_mass, plume_mass_lag))
    fupd_full_mixing = safe_divide(plume_mass,
                                    plume_mass + environment_air)
    fupd = (reference_weight * fupd_reference
            + (1.0 - reference_weight) * fupd_full_mixing)

    # Updraft air implied by those ratios and the entrained environmental air.
    environment_share = jnp.sum(frac * (1.0 - fupd), axis=0)
    updraft_air = environment_air * safe_divide(
        1.0 - environment_share, environment_share)

    # Never set aside more updraft air than the plume holds; ModelE scales the
    # entrained air back by the same factor, which is a reduction of the
    # effective entrainment rate.
    updraft_air_max = _UPDRAFT_FRACTION_MAX * plume_mass
    excessive = updraft_air > updraft_air_max
    rescale = jnp.where(excessive,
                        safe_divide(updraft_air_max, updraft_air, 1.0), 1.0)
    environment_air = environment_air * rescale
    updraft_air = jnp.where(excessive, updraft_air_max, updraft_air)

    fupd_average = jnp.sum(frac * fupd, axis=0)
    updraft_factor = safe_divide(frac * fupd, fupd_average)
    environment_factor = safe_divide(frac * (1.0 - fupd), 1.0 - fupd_average)

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

# `dcw_qc`, the fixed cutoff diameter separating cloud-mode condensate from the
# rest (MSTCNV.F90:277). Distinct from the speed-dependent `DCW` the actual
# precipitation partition uses.
_CLOUD_MODE_DIAMETER = 80.0e-6


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
    blend_mass: jnp.ndarray
    # Total air put into blends at this level, `updairm + envairm`. The three
    # fates partition it, so it is what turns the two exported fates into
    # shares -- without it the rejoining fraction cannot be recovered from the
    # outputs (the mass budget is satisfied for any split), and the sorting
    # cannot be compared against ModelE's per-blend `blend_diag.txt` record.
    blend_mass_total: jnp.ndarray


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
    removed_fraction = safe_divide(updraft_air, plume_mass)
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
    # An empty blend has no intensive properties, and dividing by a floor would
    # hand the saturation calculation a zero temperature. That is finite-valued
    # (`qsat` clamps) but not differentiable: `d(ln qsat)/dT = L/(Rv*T^2)` blows
    # up, and the resulting NaN survives being masked out later. Give empty
    # blends the environment's properties instead -- physically what a blend of
    # no air is, and harmless because it carries no mass to any fate.
    occupied = blend_mass > 0.0
    safe_mass = jnp.where(occupied, blend_mass, 1.0)
    blend_t = jnp.where(occupied, blend_heat / safe_mass, environment_heat)
    blend_q = jnp.where(occupied, blend_water / safe_mass, environment_water)
    blend_condensate_intensive = jnp.where(
        occupied, blend_condensate / safe_mass, 0.0)

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
        blend_mass_total=jnp.sum(blend_mass, axis=0),
        blend_mass=blend_mass,
    )


def vertical_velocity(kew: jnp.ndarray,
                      plume_mass: jnp.ndarray,
                      environment_air: jnp.ndarray) -> jnp.ndarray:
    """Updraft speed, diluted by the air about to be entrained [m/s].

    ModelE carries vertical momentum extensively as ``mw = sqrt(2*kew*mplume)``
    and converts to an intensive speed by dividing by the *post-entrainment*
    mass. The dilution is therefore applied here, before the sorting, using the
    incoming plume mass -- which is why entraining a lot of air slows the plume
    even on a level where it stays buoyant.

    Args:
        kew: Running buoyancy-work integral at this level. Clipped at zero: a
            negative integral means the plume has no kinetic energy left, not an
            imaginary velocity.
        plume_mass: Plume mass entering the level [kg/m^2].
        environment_air: Environmental air being entrained [kg/m^2].

    Returns:
        Updraft speed [m/s].
    """
    # `sqrt` has an infinite derivative at zero, and zero is a perfectly normal
    # argument here: it is what a terminated plume (no mass, no kinetic energy)
    # produces. Selecting the branch before taking the root keeps the gradient
    # finite, which an outer `jnp.maximum` alone would not do.
    energy = 2.0 * jnp.maximum(kew, 0.0) * plume_mass
    moving = energy > 0.0
    momentum = jnp.where(moving, jnp.sqrt(jnp.where(moving, energy, 1.0)), 0.0)
    return safe_divide(momentum, plume_mass + environment_air)


def kinetic_energy(vertical_velocity_: jnp.ndarray,
                   plume_mass: jnp.ndarray) -> jnp.ndarray:
    """Rebuild ``kew`` from the updraft speed once the plume mass has changed.

    The counterpart to :func:`vertical_velocity`, applied at the end of the
    level: the speed is held fixed while the mass is replaced by the
    post-sorting value, so ``kew`` is carried into the next level consistently
    with the mass that survived.

    Args:
        vertical_velocity_: Updraft speed at this level [m/s].
        plume_mass: Plume mass **after** sorting [kg/m^2].

    Returns:
        Buoyancy-work integral to carry upward.
    """
    return 0.5 * vertical_velocity_ ** 2 * plume_mass


# Plume termination thresholds (MSTCNV.F90, checked at the top of each level).
_MIN_PLUME_FRACTION = 5.0e-4      # MINFRAC: mplume vs the layer's own mass
_MIN_CLOUD_BASE_FRACTION = 0.01   # bsort-specific: mplume vs its cloud-base mass
_MAX_OVERSHOOT_DT = 1.0           # max_dt_overshoot [K], before /tvl

# Distance below the current level at which `mplume_lag` is sampled.
_LAG_DISTANCE = 1.0e3


class PlumeAscent(NamedTuple):
    """Per-level results of one plume's ascent. All arrays are ``(nlev, *horiz)``."""
    plume_mass: jnp.ndarray          # mass entering each level
    plume_condensate: jnp.ndarray    # condensate the plume holds at this level
    precipitation: jnp.ndarray       # condpr: what rained out of it here
    mass_lag: jnp.ndarray            # mplume_lag: mass ~1 km below
    vertical_velocity: jnp.ndarray
    entrainment: jnp.ndarray
    detrainment: jnp.ndarray
    entrained_air: jnp.ndarray       # environmental air drawn in
    detrained_mass: jnp.ndarray
    detrained_heat: jnp.ndarray
    detrained_water: jnp.ndarray
    detrained_condensate: jnp.ndarray
    downdraft_mass: jnp.ndarray
    downdraft_heat: jnp.ndarray
    downdraft_water: jnp.ndarray
    downdraft_condensate: jnp.ndarray
    cloud_mode_ratio: jnp.ndarray    # qc_updraft: condensate that may detrain
    blend_mass_total: jnp.ndarray    # updairm + envairm, what the fates split
    # Per-blend buoyancy and mass, leading axis NMIX. These are what the sort
    # actually decides on, so without them the fates can only be audited in
    # aggregate -- and the aggregate hides which blend crossed which threshold.
    blend_buoyancy: jnp.ndarray      # mixbuoy, (NMIX, nlev, *horiz)
    blend_mass: jnp.ndarray          # airmix, (NMIX, nlev, *horiz)
    active: jnp.ndarray              # bool: this level was processed


def plume_ascent(cloud_base: jnp.ndarray,
                 cloud_base_mass: jnp.ndarray,
                 cloud_base_heat: jnp.ndarray,
                 cloud_base_water: jnp.ndarray,
                 cloud_base_condensate: jnp.ndarray,
                 environment_heat: jnp.ndarray,
                 environment_water: jnp.ndarray,
                 environment_virtual_temperature: jnp.ndarray,
                 layer_mass: jnp.ndarray,
                 layer_depth: jnp.ndarray,
                 layer_thickness: jnp.ndarray,
                 height: jnp.ndarray,
                 exner: jnp.ndarray,
                 pressure: jnp.ndarray,
                 entrainment_efficiency: jnp.ndarray,
                 cloud_base_velocity: jnp.ndarray = 0.5,
                 droplet_number: jnp.ndarray = 60.0e6,
                 droplet_radius: jnp.ndarray = 10.0e-6,
                 iplume: int = 2,
                 phase: str = "water") -> PlumeAscent:
    """Run one buoyancy-sorting plume from cloud base to termination.

    Ties together the per-level pieces: the energy closure sets an entrainment
    rate, that sizes the blend spectrum, the blends are sorted by buoyancy, and
    what rejoins becomes the plume entering the next level. Two running
    integrals (``kew``, ``bdzsum``) and the plume's own properties are carried
    upward.

    **Termination.** ModelE ``exit``s the ascent loop; a scan cannot, so the
    plume is masked dead instead and contributes nothing thereafter. The
    conditions are checked on the state *entering* a level, matching ModelE
    (``MSTCNV.F90``): the previous level's ``w`` fell to zero, the plume shrank
    below ``MINFRAC`` of the layer mass or 1% of its cloud-base mass, or it
    became more than ``max_dt_overshoot`` colder than its surroundings.

    **``mplume_lag``.** ModelE walks back *down* the column to the first level
    more than 1 km below, which a scan cannot index because those levels are
    already behind it. Rather than a fixed-length ring buffer, the full incoming
    mass history is carried -- ``nlev`` floats per column, and the profile is
    small enough that the ``O(nlev^2)`` total work is irrelevant. This is exact
    for any layer spacing, whereas a buffer would silently truncate wherever
    layers are thin. The floor of the walk is ``cloud_base - 1``, whose stored
    value is the cloud-base mass.

    Args:
        cloud_base: Index of the first in-cloud level, per column.
        cloud_base_mass: Plume mass there [kg/m^2].
        cloud_base_heat: Extensive heat content there.
        cloud_base_water: Extensive water vapour there.
        cloud_base_condensate: Extensive condensate there.
        environment_heat: ``senv``, intensive, ``(nlev, *horiz)``.
        environment_water: ``qenv``, intensive.
        environment_virtual_temperature: ``tvl`` [K].
        layer_mass: ``ma`` [kg/m^2].
        layer_depth: ``gzl`` [m], used to convert rates to per-layer amounts.
        layer_thickness: ``delz = ma/rho0`` [m], used by the energy closure.
            ModelE keeps these two distinct; they are not interchangeable.
        height: ``zl`` [m], for the 1 km lookback.
        exner: ``plk``.
        pressure: [**Pa**].
        entrainment_efficiency: ``enteff``.
        cloud_base_velocity: ``wbases(iplume)``, the updraft speed at cloud base
            [m/s]. ModelE stores it in `wcu` for every level from `lcl-2` up to
            `lmin`, so it is what the first two levels of the ascent extrapolate
            `wcupass` against. It is `max(0.5, wturb)` for the entraining plume;
            0.5 is the floor and the value BOMEX sits at.
        droplet_number: ``CDNC`` [m^-3] for the precipitation partition. ModelE
            forms this as a land/ocean blend; 60e6 is its ocean value.
        droplet_radius: Assumed cloud droplet volume radius [m].
        iplume: Plume index (static).
        phase: ``"water"`` or ``"ice"`` (static).

    Returns:
        A :class:`PlumeAscent`.
    """
    nlev = layer_mass.shape[0]
    horiz = layer_mass.shape[1:]
    level_axis = jnp.arange(nlev).reshape((nlev,) + (1,) * len(horiz))
    zeros = jnp.zeros(horiz)

    def step(carry, level_inputs):
        (mass, heat, water, condensate, kew, bdzsum, previous_w, previous_w2,
         alive, dumped, history) = carry
        (level, senv, qenv, tvl, ma, gzl, delz, zl, exner_l,
         pressure_l) = level_inputs

        # Seed the plume on the level where it is born.
        at_base = level == cloud_base
        kew = jnp.where(at_base, 0.0, kew)
        bdzsum = jnp.where(at_base, 0.0, bdzsum)
        alive = alive | at_base

        # Arriving at a new level, the plume's conserved heat and water leave it
        # supersaturated, so ModelE re-partitions it and the microphysics rains
        # some of the condensate out before any sorting happens. Skipped at
        # cloud base, where the seed is already the post-condensation state.
        arrival_mass = jnp.where(mass > 0.0, mass, 1.0)
        risen_heat, risen_water, risen_condensate = resaturate_plume(
            arrival_mass, heat, water, condensate, exner_l, pressure_l, phase)
        # Rain out what the microphysics says falls faster than the updraft.
        # ModelE passes an extrapolation of the *previous* levels' `wcu` here
        # (`wcupass`), not this level's, because `wcu` is not known until after
        # the sorting -- so the carried `previous_w` is the right argument.
        # ModelE passes `TPSAV(l) = smp*plk/mplume` (MSTCNV.F90:1690), formed
        # *before* this level's condensation, so the latent heat just released
        # is not in it. Using the post-condensation temperature makes the parcel
        # 0.22% too warm.
        risen_temperature = safe_divide(heat, arrival_mass) * exner_l
        # ModelE forms the volumetric condensate as `CONDMU = (wmp/mplume)*rho0`
        # (MSTCNV.F90:1772) -- `rho0` being the layer's reference air density,
        # not one derived from the plume's own temperature. Using `ma/delz`
        # reproduces the dumped `CONDMU` exactly (ratio 1.00000 over 484
        # levels); a plume-temperature density is 0.08% off.
        air_density = safe_divide(ma, delz)
        water_content = safe_divide(risen_condensate,
                                    arrival_mass) * air_density
        # ModelE hands the microphysics `wcupass`, an extrapolation of the two
        # levels below rather than the level below alone (MSTCNV.F90:1874) --
        # `wcu(l)` is not known until after this level's sorting. Recovering
        # ModelE's critical drop diameter confirms the extrapolation exactly.
        # ModelE's only special case is the literal second model level
        # (`if(l.eq.2) wcupass = wcu(l-1)`), which a plume based this high never
        # reaches, so the extrapolation always applies. Below cloud base `wcu`
        # is not zero: MSTCNV.F90:1636-1640 fills every level from `lcl-2` up to
        # `lmin` with `wbases(iplume)`, so the first two levels of the ascent
        # extrapolate against that seed rather than against nothing.
        extrapolated_w = jnp.maximum(
            0.01, 1.5 * previous_w - 0.5 * previous_w2)
        environment_temperature = safe_divide(
            tvl, 1.0 + DELTX * qenv)
        # Below cloud base and above the plume top there is no parcel, so
        # `risen_temperature` comes out at 0 K. The microphysics is not defined
        # there -- its saturation fits and drop-size powers go non-finite under
        # `grad` even though `rose` discards the value afterwards -- so those
        # levels are handed the environment's own temperature and no water.
        present = mass > 0.0
        risen_temperature = jnp.where(present, risen_temperature,
                                      environment_temperature)
        water_content = jnp.where(present, water_content, 0.0)
        rained = microphysics.precipitate(
            water_content, extrapolated_w, pressure_l, risen_temperature,
            microphysics.scaled_droplet_number(droplet_number, pressure_l,
                                               environment_temperature),
            droplet_radius).precipitated
        # A second partition with ModelE's fixed `dcw_qc` cut gives
        # `qc_updraft`, the condensate that counts as cloud mode
        # (MSTCNV.F90:6957-6967). It is not precipitation -- it is the ceiling
        # on how much condensate the sorted blends are allowed to carry away,
        # and the excess is sent to the rain instead.
        cloud_mode = microphysics.precipitate(
            water_content, extrapolated_w, pressure_l, risen_temperature,
            microphysics.scaled_droplet_number(droplet_number, pressure_l,
                                               environment_temperature),
            droplet_radius, cutoff_diameter=_CLOUD_MODE_DIAMETER).precipitated
        # `qc_updraft = (condmu - tmp_cond)*TLOC*RGAS/PL`, i.e. back to a
        # mixing ratio.
        cloud_mode_ratio = safe_divide(water_content - cloud_mode, air_density)

        # Only the part of the partition realised over this layer's depth.
        rained = rained * microphysics.finite_ascent_fraction(ma)
        rained_mass = jnp.minimum(
            safe_divide(rained * arrival_mass, air_density), risen_condensate)
        risen_condensate = jnp.maximum(risen_condensate - rained_mass, 0.0)
        rose = ~at_base & (mass > 0.0)
        # Only levels the plume actually rose into produced precipitation; the
        # seed level's condensate arrives already rained out.
        rained_mass = jnp.where(rose, rained_mass, 0.0)

        mass = jnp.where(at_base, cloud_base_mass, mass)
        heat = jnp.where(at_base, cloud_base_heat,
                         jnp.where(rose, risen_heat, heat))
        water = jnp.where(at_base, cloud_base_water,
                          jnp.where(rose, risen_water, water))
        condensate = jnp.where(at_base, cloud_base_condensate,
                               jnp.where(rose, risen_condensate, condensate))

        # Levels the plume has not reached carry zero mass. Guarding with
        # `maximum(mass, tiny)` would keep the value finite but hand the
        # gradient a factor of 1/tiny, which overflows to NaN and then survives
        # every downstream mask -- see `safe_divide`. Selecting on both sides
        # keeps those levels out of the gradient; they are neutrally buoyant and
        # inert, which is what `alive` already assumes of them.
        occupied = mass > 0.0
        safe_mass = jnp.where(occupied, mass, 1.0)
        parcel_virtual_t = jnp.where(
            occupied,
            (heat / safe_mass) * exner_l * (1.0 + DELTX * water / safe_mass),
            tvl)
        buoyancy = jnp.where(
            occupied,
            (parcel_virtual_t - tvl) / tvl - condensate / safe_mass, 0.0)

        # Entry conditions, on the state arriving at this level.
        survives = (
            (level >= cloud_base) & alive
            & (at_base | (previous_w > 0.0))
            & (mass > _MIN_PLUME_FRACTION * ma)
            & (mass > _MIN_CLOUD_BASE_FRACTION * cloud_base_mass)
            & (buoyancy > -_MAX_OVERSHOOT_DT / tvl))

        # When the plume fails its entry test, ModelE dumps everything it still
        # carries into that level (MSTCNV.F90, just after the ascent loop:
        # `DM(LMAX) += MPLUME`, `DSM(LMAX) += SMP`, ...). Without this the
        # remaining mass simply vanishes and the environment's mass budget does
        # not close.
        # A plume still rising at the top of the model has nowhere left to go,
        # so it terminates there and dumps what it carries like any other
        # termination. ModelE never needs this -- its `cloud_top` loop is
        # bounded by `lm` and the stratosphere always stops the plume first --
        # but without it a column that stays buoyant to the top leaves the
        # entrained mass with no way back into the environment, and the mass
        # budget does not close.
        at_model_top = level == (nlev - 1)
        terminating = (alive & (~survives | at_model_top) & ~dumped
                       & (level >= cloud_base))

        # Record the mass entering this level; ModelE's `mplumearr(l) = mplume`
        # is likewise the incoming value, not the post-sorting one.
        history = jnp.where(level_axis == level, mass, history)

        d_kew, d_bdzsum = buoyancy_work_increments(
            buoyancy, mass, delz, entrainment_efficiency)
        kew_here = kew + d_kew
        bdzsum_here = bdzsum + d_bdzsum

        ent, det = entrainment_rate(
            buoyancy, mass, delz, kew_here, bdzsum_here, pressure_l, iplume)

        # `mplume_lag`: highest stored level that is more than 1 km below, or
        # the cloud-base mass if the plume has not yet climbed that far.
        eligible = ((level_axis >= cloud_base) & (level_axis < level)
                    & (zl - height > _LAG_DISTANCE))
        found = jnp.any(eligible, axis=0)
        chosen = jnp.max(jnp.where(eligible, level_axis, -1), axis=0)
        lag = jnp.sum(jnp.where(level_axis == chosen, history, 0.0), axis=0)
        lag = jnp.where(found, lag, cloud_base_mass)

        environment_air, updraft_air, updraft_factor, environment_factor, _ = (
            blend_air_masses(mass, cloud_base_mass, lag, ent, det, gzl, ma))
        w = vertical_velocity(kew_here, mass, environment_air)
        sorted_blends = sort_blends(
            mass, heat, water, condensate, senv, qenv, environment_air,
            updraft_air, updraft_factor, environment_factor, exner_l,
            pressure_l, tvl,
            buoyancy, phase)

        def keep(value):
            return jnp.where(survives, value, 0.0)

        next_mass = jnp.where(survives, sorted_blends.plume_mass, mass)
        def dump(value):
            return jnp.where(terminating, value, 0.0)

        carry = (next_mass,
                 jnp.where(survives, sorted_blends.plume_heat, heat),
                 jnp.where(survives, sorted_blends.plume_water, water),
                 jnp.where(survives, sorted_blends.plume_condensate,
                           condensate),
                 jnp.where(survives, kinetic_energy(w, next_mass), kew),
                 jnp.where(survives, bdzsum_here, bdzsum),
                 jnp.where(survives, w, previous_w),
                 jnp.where(survives, previous_w, previous_w2),
                 survives,
                 dumped | terminating,
                 history)
        outputs = (keep(mass), keep(condensate), keep(rained_mass), keep(lag),
                   keep(w), keep(ent), keep(det),
                   keep(environment_air),
                   keep(sorted_blends.detrained_mass) + dump(mass),
                   keep(sorted_blends.detrained_heat) + dump(heat),
                   keep(sorted_blends.detrained_water) + dump(water),
                   keep(sorted_blends.detrained_condensate) + dump(condensate),
                   keep(sorted_blends.downdraft_mass),
                   keep(sorted_blends.downdraft_heat),
                   keep(sorted_blends.downdraft_water),
                   keep(sorted_blends.downdraft_condensate),
                   keep(cloud_mode_ratio),
                   keep(sorted_blends.blend_mass_total),
                   keep(sorted_blends.mixture_buoyancy),
                   keep(sorted_blends.blend_mass),
                   survives)
        return carry, outputs

    # `previous_w`/`previous_w2` start at the cloud-base updraft speed, which is
    # what ModelE has stored in `wcu` below cloud base, so the `wcupass`
    # extrapolation is right from the first level of the ascent.
    seed_w = jnp.broadcast_to(jnp.asarray(cloud_base_velocity, zeros.dtype),
                              zeros.shape)
    initial = (zeros, zeros, zeros, zeros, zeros, zeros, seed_w, seed_w,
               jnp.zeros(horiz, dtype=bool), jnp.zeros(horiz, dtype=bool),
               jnp.zeros((nlev,) + horiz))
    _, outputs = lax.scan(
        step, initial,
        (level_axis, environment_heat, environment_water,
         environment_virtual_temperature, layer_mass, layer_depth,
         layer_thickness, height, exner, pressure))
    return PlumeAscent(*outputs)


def resaturate_plume(plume_mass: jnp.ndarray,
                     plume_heat: jnp.ndarray,
                     plume_water: jnp.ndarray,
                     plume_condensate: jnp.ndarray,
                     exner: jnp.ndarray,
                     pressure: jnp.ndarray,
                     phase: str = "water",
                     previous_phase: str = None):
    """Re-partition the plume between vapour and condensate at a new level.

    Run on arrival at each level, before the sorting. ModelE does not
    incrementally condense: it **evaporates all existing condensate back into
    vapour**, then recomputes the split from scratch at the new level's
    pressure (``MSTCNV.F90``, around the ``get_dq_cond`` call). Total water is
    therefore conserved exactly here, and only its division between phases --
    and the heat that division releases -- changes.

    Verified against the oracle over all 484 BOMEX level transitions: the
    resulting vapour and heat match ModelE to ~1e-6 relative. The condensate
    does *not* match, and is not expected to: precipitation is removed
    afterwards by the microphysics, which is a separate routine.

    Args:
        plume_mass: ``mplume`` [kg/m^2], unchanged by this step.
        plume_heat: ``smp``, extensive.
        plume_water: ``qmp``, extensive vapour.
        plume_condensate: ``wmp``, extensive condensate.
        exner: ``plk`` at the **new** level.
        pressure: [**Pa**] at the new level.
        phase: Phase at the new level (static).
        previous_phase: Phase used when the condensate formed, for the
            evaporation term. ModelE uses ``VLAT(l-1)`` here against ``PLK(l)``,
            so the two can differ across a freezing level. Defaults to
            ``phase``.

    Returns:
        ``(heat, water, condensate)`` after re-saturation.
    """
    previous_latent = (LHE if (previous_phase or phase) == "water" else LHS)
    latent = LHE if phase == "water" else LHS

    # Undo the previous level's condensation, returning all water to vapour.
    heat = plume_heat - previous_latent * plume_condensate / SHA / exner
    water = plume_water + plume_condensate

    condensed, _ = condensation(heat, water, exner, plume_mass, pressure, phase)
    return (heat + (latent / SHA) * condensed / exner,
            water - condensed,
            condensed)
