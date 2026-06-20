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
    saturation_specific_humidity,
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
