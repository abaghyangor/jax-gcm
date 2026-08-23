"""GISS ModelE saturation thermodynamics and moist static energy.

First ported pieces of the GISS moist-convection scheme (``MSTCNV``). These are
the small, self-contained thermodynamic primitives the convective trigger and
plume code build on. They are deliberate, faithful ports of the GISS Fortran:

* ``saturation_vapor_pressure`` -- the Murphy & Koop (2005) expression in
  ``wv_psat`` (``modelE/model/shared/Utilities.F90``). ModelE's default is
  ``use_mk2005 = .true.``, which the verified DYCOMS oracle run used, so we port
  that branch (not the older Clausius-Clapeyron form).
* ``saturation_specific_humidity`` -- ModelE ``QSAT(TM, LH, PR) = mrat *
  wv_psat(TM, LH) / PR`` (same file).
* ``moist_static_energy`` -- dry static energy ``sha*T + geopotential`` plus the
  latent term ``L*q``, matching how ``MSTCNV`` forms moist static energy from
  ``SHA``, geopotential, and ``LHE``/``LHS``.

Why GISS-specific constants (not :mod:`jcm.constants`)
-----------------------------------------------------
To reproduce the ModelE oracle, these functions must use ModelE's own constant
values (e.g. ``sha = rgas/kapa`` works out to ~1002.9 J/kg/K, slightly different
from JCM's ``cpd``). We therefore define the GISS constant set here, derived from
the same base values as ``modelE/model/shared/Constants_mod.F90``. This mirrors
how the SPEEDY port keeps its scheme-specific ``alhc`` separate from the shared
SI constant. General-purpose JCM code should still use :mod:`jcm.constants`.

Units (ModelE-native; convert at the term boundary)
---------------------------------------------------
* temperature: K
* pressure: **Pa** (ModelE ``QSAT`` takes Pa; the oracle ``p_3d`` is hPa, so
  multiply by 100 before calling)
* specific humidity: kg/kg (JCM ``PhysicsState`` uses g/kg -- convert at the edge)
* geopotential: m^2/s^2 (= g * height)

Differentiability note
----------------------
We do **not** port ModelE's analytic ``DLNQSATDT`` -- JAX provides
``d(qsat)/dT`` (and any other derivative) by autodiff straight through
``saturation_specific_humidity``. That is the whole point of the JAX port and is
exercised by the gradient tests.

Following the convention of :mod:`jcm.physics.convection.saturation`, these
functions are not ``@jit``-ed (they are inlined into the model's outer jit), and
``phase`` is a static Python string resolved at trace time.
"""

import jax.numpy as jnp

# --- GISS / ModelE constants, derived exactly as in Constants_mod.F90 ---------
_GASC = 8.314510          # universal gas constant [J/mol/K]
_MAIR = 28.9655           # molar mass of dry air [g/mol]
_MWAT = 18.015            # molar mass of water [g/mol]
_SRAT = 1.401             # c_p/c_v for dry air

RGAS = 1e3 * _GASC / _MAIR        # dry-air gas constant [J/kg/K]  (~287.05)
RVAP = 1e3 * _GASC / _MWAT        # water-vapour gas constant [J/kg/K] (~461.52)
MRAT = _MWAT / _MAIR              # molar-mass ratio (~0.62194)
KAPA = (_SRAT - 1.0) / _SRAT      # R/c_p (~0.28622)
SHA = RGAS / KAPA                 # dry-air specific heat at const p [J/kg/K] (~1002.9)
GRAV = 9.80665                    # gravitational acceleration [m/s^2]
LHE = 2.5e6                       # latent heat of evaporation [J/kg]
LHM = 3.34e5                      # latent heat of melting [J/kg]
LHS = LHE + LHM                   # latent heat of sublimation [J/kg]
DELTX = _MAIR / _MWAT - 1.0       # virtual-temperature humidity coeff. (~0.6078)

# Murphy & Koop (2005) fit validity limits used by ModelE wv_psat.
_MK_WATER_TMIN, _MK_WATER_TMAX = 123.0, 332.0
_MK_ICE_TMIN = 110.0


def saturation_vapor_pressure(temperature: jnp.ndarray,
                              phase: str = "water") -> jnp.ndarray:
    """Saturation vapour pressure [Pa], Murphy & Koop (2005).

    Faithful port of the ``use_mk2005 = .true.`` branch of ModelE ``wv_psat``
    (``modelE/model/shared/Utilities.F90``). Temperature is clipped to the fit's
    validity range exactly as the Fortran does.

    Args:
        temperature: Temperature [K].
        phase: ``"water"`` (uses ``LHE`` branch) or ``"ice"`` (``LHS`` branch).
            Static, resolved at trace time.

    Returns:
        Saturation vapour pressure [Pa].
    """
    if phase == "water":
        tn = jnp.clip(temperature, _MK_WATER_TMIN, _MK_WATER_TMAX)
        ln_tn = jnp.log(tn)
        return jnp.exp(
            54.842763 - 6763.22 / tn - 4.210 * ln_tn + 0.000367 * tn
            + jnp.tanh(0.0415 * (tn - 218.8))
            * (53.878 - 1331.22 / tn - 9.44523 * ln_tn + 0.014025 * tn)
        )
    if phase == "ice":
        tn = jnp.maximum(temperature, _MK_ICE_TMIN)
        return jnp.exp(
            9.550426 - 5723.265 / tn + 3.53068 * jnp.log(tn) - 0.00728332 * tn
        )
    raise ValueError(f"phase must be 'water' or 'ice', got {phase!r}")


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 phase: str = "water") -> jnp.ndarray:
    """Saturation specific humidity [kg/kg], ModelE ``QSAT``.

    ``QSAT(TM, LH, PR) = mrat * wv_psat(TM, LH) / PR``
    (``modelE/model/shared/Utilities.F90``).

    Args:
        temperature: Temperature [K].
        pressure: Air pressure [**Pa**] (oracle ``p_3d`` is hPa -> *100).
        phase: ``"water"`` or ``"ice"`` (static).

    Returns:
        Saturation specific humidity [kg/kg].
    """
    return MRAT * saturation_vapor_pressure(temperature, phase) / pressure


def moist_static_energy(temperature: jnp.ndarray,
                        geopotential: jnp.ndarray,
                        specific_humidity: jnp.ndarray,
                        phase: str = "water") -> jnp.ndarray:
    """Moist static energy [J/kg]: ``sha*T + geopotential + L*q``.

    Matches how ``MSTCNV`` forms moist static energy from the dry static energy
    (``SHA*T`` plus geopotential) and the latent contribution, with ``L = LHE``
    for the water phase and ``LHS`` for ice.

    Args:
        temperature: Temperature [K].
        geopotential: Geopotential [m^2/s^2] (= g * height).
        specific_humidity: Specific humidity [kg/kg].
        phase: ``"water"`` (``LHE``) or ``"ice"`` (``LHS``). Static.

    Returns:
        Moist static energy [J/kg].
    """
    latent_heat = LHE if phase == "water" else LHS
    return SHA * temperature + geopotential + latent_heat * specific_humidity


def d_ln_qsat_dt(temperature: jnp.ndarray, phase: str = "water") -> jnp.ndarray:
    """``d(ln qsat)/dT = L / (RVAP * T^2)`` -- ModelE ``DLNQSATDT``.

    Port of ModelE ``DLNQSATDT`` (``shared/Utilities.F90``), used by the
    cloud-base mass-flux closure's saturation-adjustment terms. Note ModelE uses
    this Clausius-Clapeyron form (with ``C = 1/RVAP``) even when ``wv_psat`` is
    the Murphy & Koop expression -- a deliberate approximation we reproduce.

    (Elsewhere we let JAX autodiff differentiate ``qsat`` directly; this analytic
    form exists specifically to match the closure's Fortran arithmetic.)

    Args:
        temperature: Temperature [K].
        phase: ``"water"`` (``LHE``) or ``"ice"`` (``LHS``). Static.

    Returns:
        ``d(ln qsat)/dT`` [1/K].
    """
    latent_heat = LHE if phase == "water" else LHS
    return latent_heat / (RVAP * temperature ** 2)


def virtual_temperature(temperature: jnp.ndarray,
                        specific_humidity: jnp.ndarray,
                        condensate: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Virtual (density) temperature [K]: ``T * (1 + DELTX*q - wm)``.

    This is the buoyancy variable ``MSTCNV`` uses in the cloud-base closure
    (``SVDN = SDN*(1 + DELTX*QDN - WMDN)``): water vapour makes a parcel lighter
    (``+DELTX*q``) while suspended condensate loads it down (``-wm``). Applying
    the same multiplicative factor to potential temperature instead of ``T``
    yields the virtual *potential* temperature used for the edge values
    ``SVUP``/``SVDN``.

    Args:
        temperature: Temperature (or potential temperature) [K].
        specific_humidity: Water-vapour specific humidity [kg/kg].
        condensate: Suspended condensate (liquid + ice) mixing ratio [kg/kg].
            Defaults to 0 (no condensate loading).

    Returns:
        Virtual temperature [K] (or virtual potential temperature, matching the
        input).
    """
    return temperature * (1.0 + DELTX * specific_humidity - condensate)


# `get_dq_evap` runs a fixed three-iteration Newton solve. The trip count is
# part of the scheme's definition, not a convergence criterion -- ModelE stops
# after three regardless of the residual -- so it is reproduced verbatim rather
# than replaced by an iterate-to-tolerance loop, which would change answers.
_EVAP_ITERATIONS = 3
_TEENY = 1e-20


def condensate_evaporation(dry_static_energy: jnp.ndarray,
                           water_mass: jnp.ndarray,
                           exner: jnp.ndarray,
                           mass: jnp.ndarray,
                           pressure: jnp.ndarray,
                           condensate: jnp.ndarray,
                           phase: str = "water"):
    """Evaporate condensate toward saturation -- ModelE ``get_dq_evap``.

    Port of ``modelE/model/CLOUDS_COM.F90:903``. Given an air parcel holding
    ``condensate``, find how much of it evaporates as the parcel relaxes toward
    saturation, cooling as it goes. The buoyancy-sorting plume calls this on
    every updraft/environment blend before testing the blend's buoyancy
    (``MSTCNV.F90``, ``plume_ent_det_w2_bsort``), which is what lets an
    entraining mixture become negatively buoyant through evaporative cooling --
    the mechanism that drives most of the plume's mass loss.

    The Newton step solves ``q - mass*qsat(T) = 0`` with ``T`` responding to the
    latent heating, linearised through ``d(ln qsat)/dT``:

        dq = (q - mass*qsat(T)) / (1 + (L/cp)*qsat(T)*dlnqsatdt(T))

    Evaporation is ``dq < 0`` (the parcel is subsaturated), which accumulates as
    a positive ``dqsum``; the result is clipped to ``[0, condensate]`` so a
    supersaturated blend neither condenses here nor evaporates more than it has.

    Args:
        dry_static_energy: ``SM``, potential-temperature-like heat content,
            such that ``T = dry_static_energy*exner/mass`` [K * mass units].
        water_mass: ``QM``, water-vapour content in the same mass units.
        exner: ``PLK``, ModelE's ``p[mb]**KAPA``. Note this is *not* normalised
            by a reference pressure -- see the module docstring.
        mass: ``MASS``, air mass of the parcel. The plume calls this with
            ``mass = 1`` and intensive ``sm``/``qm``.
        pressure: Air pressure [**Pa**]. ModelE passes ``pres(l)`` in mb; the
            caller must convert, since this module's ``qsat`` is Pa-based.
        condensate: Available condensate, the cap on evaporation [same units as
            ``water_mass``].
        phase: ``"water"`` (``LHE``) or ``"ice"`` (``LHS``). Static. ModelE
            selects this per level via ``vlat(l)``.

    Returns:
        ``(dqsum, fevp)`` -- the evaporated water (>= 0, capped by
        ``condensate``) and the fraction of the condensate it represents.
    """
    latent_heat = LHE if phase == "water" else LHS
    slh = latent_heat / SHA

    # `saturation_vapor_pressure` clips its argument to the fit's valid range,
    # but `d_ln_qsat_dt = L/(Rv*T^2)` does not, and a caller can legitimately
    # hand this routine a degenerate parcel (a zero-mass blend, an inactive
    # column). At T -> 0 that term is infinite: harmless in value, since it only
    # divides, but it makes the gradient NaN and the NaN survives being masked
    # out downstream. Clip to the same floor the saturation fit uses.
    parcel_t = jnp.maximum(dry_static_energy * exner / mass, _MK_ICE_TMIN)
    remaining_water = water_mass
    dqsum = jnp.zeros_like(parcel_t)

    for _ in range(_EVAP_ITERATIONS):
        qst = saturation_specific_humidity(parcel_t, pressure, phase)
        dq = ((remaining_water - mass * qst)
              / (1.0 + slh * qst * d_ln_qsat_dt(parcel_t, phase)))
        parcel_t = parcel_t + slh * dq / mass
        remaining_water = remaining_water - dq
        dqsum = dqsum - dq

    # ModelE guards the whole body with `if (COND > 0)`, leaving both outputs
    # zero otherwise; the clip already forces dqsum to 0 when condensate is 0,
    # so only the fevp division needs protecting against 0/0.
    dqsum = jnp.clip(dqsum, 0.0, condensate)
    fevp = dqsum / jnp.maximum(condensate, _TEENY)
    return dqsum, fevp
