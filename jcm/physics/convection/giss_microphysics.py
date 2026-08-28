"""Convective precipitation -- GISS ModelE ``PRECIPLIQ_GAMMA``.

Port of the warm-rain precipitation partition in ``CONVECTIVE_MICROPHYSICS``
(``MSTCNV.F90``). This decides how much of the condensate a rising plume holds
is left behind as rain rather than carried upward, and it is what supplies the
downdraft with water to evaporate.

**Why it matters more than its size suggests.** The convective downdraft is
driven by evaporative cooling of falling precipitation, and that coupling turns
out to be a *runaway*: too little precipitation leaves the downdraft warm, a warm
downdraft tests positively buoyant and sheds most of its mass, and a smaller
downdraft has less air to evaporate into. A fitted stand-in for this routine
collapsed the modelled downdraft within two levels where ModelE's roughly
doubles on the way down.

The scheme
----------
The condensate is split into two populations, each with a gamma size
distribution, and drops falling faster than the updraft are lost as rain:

* a **cloud mode** of fixed capacity ``CDNC·rho_w·(4/3)pi·rvl^3`` -- as many
  droplets as there are nuclei, each of the assumed cloud droplet radius;
* a **rain mode** holding whatever condensate exceeds that capacity.

Nothing precipitates until the cloud mode is full, which makes precipitation a
*threshold* process in liquid water content. (Verified against the oracle on
630/630 records.)

Two branch traps, both resolved from the oracle rather than the source
---------------------------------------------------------------------
``use_s08_fallspeed_mstcnv`` and ``use_gammadsd_mstcnv`` both default to
``.true.``, which makes the 15-iteration Newton solve for the critical diameter
and the Marshall-Palmer ``PRECIP_MP`` call **dead code** -- the analytic fall
speed and this routine are what run. ``PRECIPLIQ_GAMMA`` is itself defined
twice, with ``#define FAST_MICROPHYSICS`` selecting the later copy.

Also: ``PL`` inside ``CONVECTIVE_MICROPHYSICS`` is in **Pa** despite the name;
the caller passes ``PRES(L)``, not ``pl(l)``.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import gammainc

from jcm.physics.convection.giss_thermodynamics import safe_divide

# Water density and the volume factor of a sphere, as ModelE uses them.
_RHO_WATER = 1000.0
_PI_6 = jnp.pi / 6.0

# Analytic terminal-fall-speed inversion (`use_s08_fallspeed_mstcnv`). The drop
# whose fall speed matches the updraft is the largest one the plume can hold.
_FALL_A, _FALL_B, _FALL_C = 9.65, 9.8, 600.0
_FALL_PRESSURE_EXPONENT = 0.4
# When the updraft outruns every drop size, ModelE takes log(teeny)/(-vrc).
_DIAMETER_MAX = -jnp.log(1e-20) / _FALL_C

# Gamma-distribution shape parameters.
_MU_CLOUD_MAX = 15.0
_MU_RAIN = 2.5
# Rain intercept blends between these two limits across a threshold water
# content `qr0`, via a tanh.
_N0_HIGH, _N0_LOW = 9.0e9, 2.0e6
_RAIN_THRESHOLD = 1.0e-4

_LAM_RAIN_FACTOR = ((_MU_RAIN + 1.0) * (_MU_RAIN + 2.0)
                    * (_MU_RAIN + 3.0)) ** (1.0 / 3.0)
_RGAS = 287.05

# Only part of a layer's precipitation forms within the distance the plume
# actually ascends in one step, so ModelE scales `CONDP` down by the layer's
# mass against a reference depth (`cond_repart_dpscale = 50 mb`). Verified
# against the oracle: a dumped factor of 0.2035 corresponds exactly to the
# dumped layer mass of 103.76 kg/m^2.
_ASCENT_REFERENCE_MASS = 50.0 * 100.0 / 9.80665      # `cond_repart_dmscale`

# ModelE specifies the droplet concentration at a reference state and scales it
# by the local air density (`do_scale_nc`, MSTCNV.F90:1864). The reference is
# `nc_pref = 900 hPa` and `nc_tref = tf + 10 K` (CLOUDS_COM.F90:31-32).
_NC_REFERENCE_PRESSURE = 900.0e2
_NC_REFERENCE_TEMPERATURE = 273.15 + 10.0


def scaled_droplet_number(droplet_number: jnp.ndarray,
                          pressure: jnp.ndarray,
                          environment_temperature: jnp.ndarray) -> jnp.ndarray:
    """Droplet concentration scaled from its reference state to the local air.

    ``CDNC * (p/T) * (nc_tref/nc_pref)``. The concentration is *specified* per
    unit volume at a reference density, so it must be scaled by the local one;
    using the unscaled value leaves the cloud-mode capacity wrong by several
    percent, which matters because precipitation is a threshold process in it.

    Args:
        droplet_number: Reference concentration [m^-3] (60e6 over ocean).
        pressure: [**Pa**].
        environment_temperature: Layer temperature ``tl(l)`` [K] -- the
            *environment's*, not the plume's.

    Returns:
        Scaled concentration [m^-3].
    """
    return (droplet_number * safe_divide(pressure, environment_temperature)
            * (_NC_REFERENCE_TEMPERATURE / _NC_REFERENCE_PRESSURE))


def finite_ascent_fraction(layer_mass: jnp.ndarray) -> jnp.ndarray:
    """Fraction of the partitioned precipitation realised in one layer.

    ``min(1, ma / cond_repart_dmscale)``. Thin layers give the drops less
    distance to fall out in, so less of the partition is realised.

    Args:
        layer_mass: ``ma`` [kg/m^2].

    Returns:
        Fraction in (0, 1].
    """
    return jnp.minimum(1.0, layer_mass / _ASCENT_REFERENCE_MASS)


class Precipitation(NamedTuple):
    """Partition of a plume's condensate."""
    precipitated: jnp.ndarray      # water content rained out [kg/m^3]
    cloud_water: jnp.ndarray       # retained in the cloud mode [kg/m^3]
    rain_water: jnp.ndarray        # in the rain mode before the size cut
    critical_diameter: jnp.ndarray  # `Dc` [m]


def critical_diameter(updraft_speed: jnp.ndarray,
                      pressure: jnp.ndarray) -> jnp.ndarray:
    """Drop diameter whose terminal fall speed equals the updraft [m].

    Drops larger than this fall out of the plume. ModelE inverts its fall-speed
    relation analytically here; the alternative Newton solve in the source is
    not the active branch.

    Args:
        updraft_speed: ``w`` [m/s], clipped at zero.
        pressure: [**Pa**].

    Returns:
        Critical diameter [m].
    """
    speed = jnp.maximum(updraft_speed, 0.0)
    scaled = speed * (pressure / 1.0e5) ** _FALL_PRESSURE_EXPONENT
    # Above `vra` the updraft outruns every drop and nothing can fall out.
    outruns = scaled >= _FALL_A
    ratio = jnp.where(outruns, 1.0, (scaled - _FALL_A) / (-_FALL_B))
    return jnp.where(outruns, _DIAMETER_MAX,
                     jnp.maximum(0.0, jnp.log(ratio) / (-_FALL_C)))


def precipitate(condensate: jnp.ndarray,
                updraft_speed: jnp.ndarray,
                pressure: jnp.ndarray,
                temperature: jnp.ndarray,
                droplet_number: jnp.ndarray,
                droplet_radius: jnp.ndarray) -> Precipitation:
    """Split a plume's condensate into what falls out and what rises on.

    Args:
        condensate: Total condensed water content ``twc`` [kg/m^3]. Note this is
            a *content*, not a mixing ratio: convert from the plume's extensive
            condensate with ``q_c · rho_air``.
        updraft_speed: ``w`` [m/s], setting the critical drop size.
        pressure: [**Pa**].
        temperature: Parcel temperature [K], for the air density used by the
            rain intercept.
        droplet_number: ``CDNC`` [m^-3]. In ModelE this is a land/ocean blend,
            not an aerosol calculation.
        droplet_radius: Assumed cloud droplet volume radius ``rvl`` [m].

    Returns:
        A :class:`Precipitation`.
    """
    air_density = pressure / (_RGAS * temperature)
    diameter = critical_diameter(updraft_speed, pressure)

    # The cloud mode holds one droplet per nucleus at the assumed radius; the
    # rest of the condensate has nowhere to go but the rain mode.
    capacity = (droplet_number * _RHO_WATER * 4.0 / 3.0 * jnp.pi
                * droplet_radius ** 3)
    cloud_water = jnp.minimum(capacity, condensate)
    rain_water = jnp.maximum(condensate - cloud_water, 0.0)

    # --- cloud mode -------------------------------------------------------
    mu_cloud = jnp.minimum(_MU_CLOUD_MAX, 1.0e9 / droplet_number + 2.0)
    has_cloud = cloud_water > 0.0
    # The cube root has an infinite derivative at zero, and zero is a normal
    # argument here (a condensate-free parcel), so keep it out of the power.
    lam_cloud_cubed = safe_divide(
        _RHO_WATER * _PI_6 * droplet_number
        * (mu_cloud + 1.0) * (mu_cloud + 2.0) * (mu_cloud + 3.0),
        cloud_water)
    lam_cloud = jnp.where(
        has_cloud,
        jnp.where(has_cloud, lam_cloud_cubed, 1.0) ** (1.0 / 3.0), 0.0)
    # `incompleteGamma2` is the LOWER regularized incomplete gamma: the mass
    # fraction *below* the critical size, i.e. the drops slow enough to stay
    # with the plume. The name invites the opposite reading, and the sign of the
    # whole scheme turns on it -- the proof is that these are subtracted from
    # the total below to give the precipitation.
    retained_cloud = jnp.where(
        has_cloud,
        cloud_water * gammainc(mu_cloud + 4.0, lam_cloud * diameter),
        0.0)

    # --- rain mode --------------------------------------------------------
    has_rain = rain_water > 0.0
    threshold = _RAIN_THRESHOLD * air_density
    n0_rain = (0.5 * (_N0_HIGH - _N0_LOW)
               * jnp.tanh(safe_divide(threshold - rain_water, 4.0 * threshold))
               + 0.5 * (_N0_HIGH + _N0_LOW))
    # Likewise the fourth root.
    number_rain_fourth = safe_divide(n0_rain ** 3 * rain_water,
                                      _RHO_WATER * _PI_6)
    number_rain = jnp.where(
        has_rain,
        jnp.where(has_rain, number_rain_fourth, 1.0) ** 0.25, 0.0)
    lam_rain = jnp.where(
        has_rain, safe_divide(n0_rain, number_rain) * _LAM_RAIN_FACTOR, 0.0)
    retained_rain = jnp.where(
        has_rain,
        rain_water * gammainc(_MU_RAIN + 4.0, lam_rain * diameter),
        0.0)

    precipitated = jnp.maximum(
        0.0, condensate - retained_cloud - retained_rain)
    return Precipitation(precipitated=precipitated, cloud_water=cloud_water,
                         rain_water=rain_water, critical_diameter=diameter)
