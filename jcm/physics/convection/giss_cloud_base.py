"""GISS convective trigger: lifting condensation level (cloud base).

Ports the cloud-base / lifting-condensation-level (LCL) detection from the GISS
``MSTCNV`` plume code (``modelE/model/MSTCNV.F90``). In the Fortran, a
boundary-layer parcel with potential temperature ``sparcel`` and specific
humidity ``qparcel`` is lifted level by level and the cloud base ``lcl`` is the
first level where it saturates::

    do lcl = lmin0+1, lmin+1
      if (qparcel > qsat(sparcel*plk(lcl), lhe, pres(lcl))) exit
    enddo

``sparcel*plk(l)`` is the parcel's *temperature* at level ``l``: lifting at
constant potential temperature is a dry adiabat, ``T(p) = T0 * (p/p0)**KAPA``.
Because we form the adiabat from the parcel's own origin ``(T0, p0)``, the
pressure-ratio is reference-independent and we don't need ModelE's particular
``plk`` reference pressure.

The Fortran's "first level scanning upward, then exit" loop becomes a vectorized
``argmax`` over the saturation mask (the same device the SPEEDY convection
trigger uses), so the identical code runs broadcasting-native on a single column
``(nlev,)`` or a vectorized block ``(nlev, ncols)``.

Differentiability
-----------------
The continuous pieces -- the dry adiabat and ``qsat`` -- are fully
differentiable. The cloud-base *index* itself comes from ``argmax`` and is a
**discontinuous** trigger (it jumps between levels), exactly the kind of
convective-triggering branch the project notes flag. Gradients flow through the
thermodynamic quantities, not through the integer level index.

Units (ModelE-native): temperature K, pressure **Pa**, specific humidity kg/kg.
"""

import jax.numpy as jnp

from jcm.physics.convection.giss_thermodynamics import (
    KAPA,
    LHE,
    LHS,
    SHA,
    saturation_specific_humidity,
    virtual_temperature,
)


def dry_adiabatic_temperature(parcel_temperature: jnp.ndarray,
                              parcel_pressure: jnp.ndarray,
                              level_pressure: jnp.ndarray) -> jnp.ndarray:
    """Temperature of a parcel lifted dry-adiabatically to ``level_pressure``.

    ``T(p) = T0 * (p / p0) ** KAPA`` (constant potential temperature).

    Args:
        parcel_temperature: Parcel origin temperature ``T0`` [K], per column.
        parcel_pressure: Parcel origin pressure ``p0`` [Pa], per column.
        level_pressure: Pressure(s) to lift to [Pa]; vertical on axis 0.

    Returns:
        Parcel temperature at each ``level_pressure`` [K].
    """
    return parcel_temperature * (level_pressure / parcel_pressure) ** KAPA


def lifting_condensation_level(parcel_temperature: jnp.ndarray,
                               parcel_pressure: jnp.ndarray,
                               parcel_specific_humidity: jnp.ndarray,
                               level_pressure: jnp.ndarray):
    """Find the cloud base: first level where a lifted parcel saturates.

    The parcel is assumed to originate at the lowest provided level and lift
    upward (vertical on axis 0; index 0 = surface / highest pressure). At each
    level the dry-adiabatic parcel temperature is compared against saturation:
    the cloud base is the first level where ``parcel_q > qsat(T_parcel, p)``.

    Args:
        parcel_temperature: Parcel origin temperature [K], per column ``(...)``.
        parcel_pressure: Parcel origin pressure [Pa], per column ``(...)``.
        parcel_specific_humidity: Parcel specific humidity [kg/kg], ``(...)``.
        level_pressure: Column pressure profile [Pa], ``(nlev, ...)``.

    Returns:
        ``(cloud_base, condenses)`` where ``cloud_base`` ``(...)`` is the level
        index of the cloud base (or ``nlev`` if the parcel never saturates), and
        ``condenses`` ``(...)`` is a boolean: whether the parcel reaches
        saturation anywhere in the column.
    """
    nlev = level_pressure.shape[0]
    parcel_t = dry_adiabatic_temperature(
        parcel_temperature, parcel_pressure, level_pressure)
    qsat_level = saturation_specific_humidity(parcel_t, level_pressure)

    saturated = parcel_specific_humidity > qsat_level   # (nlev, ...)
    condenses = jnp.any(saturated, axis=0)              # (...)
    first_saturated = jnp.argmax(saturated, axis=0)     # first True (0 if none)
    # argmax returns 0 both for "saturated at the surface" and "never saturated";
    # the condenses mask disambiguates -- a non-saturating parcel gets the
    # sentinel ``nlev`` (an out-of-range index meaning "no cloud base").
    cloud_base = jnp.where(condenses, first_saturated, nlev)
    return cloud_base, condenses


def cloud_base_instability(parcel_potential_temperature: jnp.ndarray,
                           parcel_specific_humidity: jnp.ndarray,
                           parcel_condensate: jnp.ndarray,
                           above_potential_temperature: jnp.ndarray,
                           above_specific_humidity: jnp.ndarray,
                           above_condensate: jnp.ndarray,
                           exner: jnp.ndarray,
                           pressure: jnp.ndarray,
                           phase: str = "water") -> jnp.ndarray:
    """Cloud-base moist instability metric ``DMSE`` (units of K).

    Faithful port of the trigger in ``MSTCNV`` ``cloud_base_closure``::

        DMSE = (SVUP-SVDN)*PLK + SLH*(QSAT(SUP*PLK, LHX, pres) - QDN)

    evaluated at the interface just above cloud base (``lmin+1``). ``SUP``/``SDN``
    are *potential* temperatures (``SM = TH*MA``); ``PLK`` is the Exner function,
    so ``SUP*PLK`` is the layer-above temperature. ``SVUP``/``SVDN`` are the
    corresponding virtual potential temperatures (vapour + condensate loading);
    ``SLH = L/cp``. The first term is the parcel-vs-environment virtual
    *temperature* difference; the second is the latent contribution from the
    parcel's vapour excess over saturation aloft.

    Sign convention (confirmed from source): ``DMSE < 0`` => **unstable**.
    Larger parcel buoyancy or vapour excess drives it more negative.

    NOTE: structure and sign are read from the Fortran but **not yet validated
    numerically** against a convectively active oracle (the available DYCOMS case
    never triggers convection). Treat absolute values as provisional until a
    BOMEX/RICO oracle is available; the tests below assert structure, not
    oracle agreement.

    Args:
        parcel_potential_temperature: Source-parcel potential temperature ``SDN`` [K].
        parcel_specific_humidity: Source-parcel specific humidity ``QDN`` [kg/kg].
        parcel_condensate: Source-parcel condensate ``WMDN`` [kg/kg].
        above_potential_temperature: Layer-above potential temperature ``SUP`` [K].
        above_specific_humidity: Layer-above specific humidity ``QUP`` [kg/kg].
        above_condensate: Layer-above condensate ``WMUP`` [kg/kg].
        exner: Exner function ``PLK`` at the interface (= ``(p/p0)**KAPA``).
        pressure: Interface pressure [Pa].
        phase: ``"water"`` (``LHE``) or ``"ice"`` (``LHS``). Static.

    Returns:
        ``DMSE`` [K]; negative => convectively unstable.
    """
    sv_dn = virtual_temperature(
        parcel_potential_temperature, parcel_specific_humidity, parcel_condensate)
    sv_up = virtual_temperature(
        above_potential_temperature, above_specific_humidity, above_condensate)
    slh = (LHE if phase == "water" else LHS) / SHA
    qsat_above = saturation_specific_humidity(
        above_potential_temperature * exner, pressure, phase)
    return (sv_up - sv_dn) * exner + slh * (qsat_above - parcel_specific_humidity)


def cloud_base_triggers(parcel_potential_temperature: jnp.ndarray,
                        parcel_specific_humidity: jnp.ndarray,
                        parcel_condensate: jnp.ndarray,
                        above_potential_temperature: jnp.ndarray,
                        above_specific_humidity: jnp.ndarray,
                        above_condensate: jnp.ndarray,
                        exner: jnp.ndarray,
                        pressure: jnp.ndarray,
                        phase: str = "water"):
    """Whether moist convection triggers at the cloud-base interface.

    Combines the two ``MSTCNV`` ``cloud_base_closure`` gates at level ``lmin+1``:

    1. the source parcel is **saturated** when lifted to the interface
       (``QDN >= QSAT(SDN*PLK, pres)``), and
    2. the layer is **unstable** (:func:`cloud_base_instability` ``< 0``).

    Both gates are necessary; either failing means "try the next level / no
    convection here".

    The boolean trigger is a **discontinuous** switch (the convective-triggering
    branch the project notes flag); the underlying ``DMSE`` is continuous and
    differentiable. Not yet oracle-validated (see
    :func:`cloud_base_instability`).

    Returns:
        ``(triggers, dmse)`` -- a boolean per column and the continuous metric.
    """
    qsat_parcel = saturation_specific_humidity(
        parcel_potential_temperature * exner, pressure, phase)
    saturated = parcel_specific_humidity >= qsat_parcel
    dmse = cloud_base_instability(
        parcel_potential_temperature, parcel_specific_humidity, parcel_condensate,
        above_potential_temperature, above_specific_humidity, above_condensate,
        exner, pressure, phase)
    return saturated & (dmse < 0.0), dmse
