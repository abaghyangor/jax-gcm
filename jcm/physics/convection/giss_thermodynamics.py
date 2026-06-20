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
