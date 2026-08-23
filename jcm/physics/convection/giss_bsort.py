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

import jax.numpy as jnp

import jcm.constants as c

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
