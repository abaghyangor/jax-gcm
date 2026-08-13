"""GISS ModelE moist convection as a composable PhysicsTerm.

The JCM landing zone for the GISS ModelE moist convection routine ``MSTCNV``
(``modelE/model/MSTCNV.F90``, called from ``CONDSE_column`` in
``modelE/model/CLOUDS_DRV.F90``). Per the repo's by-process organization the
convection *term* lives here under ``jcm/physics/convection/`` (named after the
scheme); the parameter/diagnostic structs live under :mod:`jcm.physics.modele`.
The ModelE oracle reading/conversion tooling lives in a *separate* repository;
this repo only carries committed fixture arrays (``jcm/data/test/modele/``) for
tests. See ``ARCHITECTURE.md`` for the full port design.

Status: diagnostics always; full tendencies under ``allow_mc``
-------------------------------------------------------------
:class:`GissConvection` always *diagnoses* two things from the state (when the
column pressure / layer-mass diagnostics are present):

1. ``cloud_base`` -- the convective cloud base (lifting condensation level) of a
   surface parcel, via :func:`~jcm.physics.convection.giss_cloud_base.lifting_condensation_level`.
2. ``cloud_base_mass_flux`` -- the ``MASS_FLUX2`` closure plume mass ``fmp2``
   [kg/m^2] at cloud base (:func:`cloud_base_closure_mass_flux`). The *validated*
   piece: on the BOMEX oracle it gives physical values (``fmp2 ~ 2-8 kg/m^2``).

When ``allow_mc`` is set (mirroring ModelE ``SCMopt%allowMC``; **default off**)
the term also runs the **full tendency chain** (:meth:`_convective_tendencies`):
closure ``fmp2`` -> full-column entraining plume ascent -> compensating
subsidence + detrainment deposition -> ``dth_mc``/``dq_mc``, returned as
temperature/humidity tendencies.

**Validation status of the tendencies:** structurally sensible but **not
magnitude-validated**. On BOMEX the peak ``dth_mc`` lands within ~2x of ModelE
for well-triggered columns and the plume tops out near the observed inversion,
but the scheme still under-triggers on some columns and is missing cooling terms
(evaporation, entrainment removal), so the vertical shape is imperfect. A proper
fix/validation needs the plume-internal ModelE oracle (deferred -- see
``STATUS.md``). Off by default for this reason.

Conventions
-----------
* JCM orders the vertical with **index 0 = top, last index = surface**; the
  ported GISS routines expect **surface-first**, so profiles are flipped
  (``jnp.flip(axis=0)``) before use and indices are returned surface-first.
* Operates broadcasting-native on column-vectorized state ``(nlev, ncols)``;
  the per-column cloud base is handled with :func:`jnp.take_along_axis` (a gather
  with per-column indices), so no ``vmap`` is needed.
* ``state.specific_humidity`` is g/kg (JCM); the GISS routines use kg/kg.
* ``pressure_full`` [Pa], ``layer_thickness`` [m], ``air_density`` [kg/m^3] are
  read from the diagnostics dict when present (``air_mass = density*thickness =
  dp/g``). They are read optionally so the term still runs as a pure scaffold
  (sentinel cloud base, zero mass flux) without them.
* ``PhysicsTendency`` fields are per second.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.physics_term import PhysicsTerm
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.modele.params import GissConvectionParameters
from jcm.physics.modele.physics_data import GissConvectionData
from jcm.physics.convection.giss_cloud_base import lifting_condensation_level
from jcm.physics.convection.giss_mass_flux import cloud_base_mass_flux
from jcm.physics.convection.giss_plume import plume_ascent_column
from jcm.physics.convection.giss_tendencies import convective_tendencies
from jcm.physics.convection.giss_thermodynamics import (
    KAPA,
    saturation_specific_humidity,
)

# Reference pressure for the Exner function. The closure recovers temperature as
# ``theta*exner``, so the reference cancels and its specific value is arbitrary;
# 1000 hPa is the conventional choice and matches the BOMEX validation.
_P_REF = 100000.0

# Plume-ascent constants used by the (allow_mc) tendency path. These are physical
# tunables that should graduate to differentiable ``GissConvectionParameters``
# leaves; kept as module constants for now because the tendency magnitudes are
# not yet validated (the plume over-penetrates -- see the module docstring), so
# tuning them is premature.
_CONTCE = 0.6          # Gregory entrainment strength (more-entraining plume)
_CLOUD_BASE_W = 0.5    # cloud-base updraft seed [m/s]


def cloud_base_closure_mass_flux(temperature: jnp.ndarray,
                                 specific_humidity: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 air_mass: jnp.ndarray,
                                 cloud_base: jnp.ndarray):
    """Cloud-base plume fraction / mass from the ``MASS_FLUX2`` closure.

    Broadcasting-native wrapper around
    :func:`~jcm.physics.convection.giss_mass_flux.cloud_base_mass_flux` that
    builds the three-level cloud-base stencil from full column profiles and a
    per-column cloud-base index.

    All profile inputs are **surface-first** ``(nlev, ...)`` (index 0 = surface);
    ``cloud_base`` ``(...)`` is the lifting-condensation-level index (the sentinel
    ``nlev`` means no cloud base). The closure source level is ``lmin =
    cloud_base - 1``; the stencil spans ``[lmin, lmin+1, lmin+2]`` and its source
    level (index 0) is replaced by the **surface** parcel -- the boundary-layer
    air that actually feeds the plume, matching the BOMEX validation.

    Args:
        temperature: Temperature [K], ``(nlev, ...)``.
        specific_humidity: Specific humidity [kg/kg], ``(nlev, ...)``.
        pressure: Pressure [Pa], ``(nlev, ...)``.
        air_mass: Layer air mass ``dp/g`` [kg/m^2], ``(nlev, ...)``.
        cloud_base: Cloud-base (LCL) level index per column, ``(...)``.

    Returns:
        ``(fplume, fmp2)`` -- the plume mass fraction and mass [kg/m^2] per
        column, ``(...)``; both zero where there is no cloud base.
    """
    nlev = temperature.shape[0]
    exner = (pressure / _P_REF) ** KAPA
    theta = temperature / exner

    # Source level; clip so the 3-level stencil stays in range (masked out below
    # when there is no cloud base anyway).
    lmin = jnp.clip(cloud_base - 1, 0, nlev - 3)

    def gather(arr, size):
        offsets = jnp.arange(size).reshape((size,) + (1,) * lmin.ndim)
        return jnp.take_along_axis(arr, lmin[None, ...] + offsets, axis=0)

    theta3 = gather(theta, 3).at[0].set(theta[0])        # source = surface parcel
    q3 = gather(specific_humidity, 3).at[0].set(specific_humidity[0])
    air_mass3 = gather(air_mass, 3)
    exner2 = gather(exner, 2)
    pressure2 = gather(pressure, 2)

    fplume, fmp2, _ = cloud_base_mass_flux(theta3, q3, air_mass3, exner2, pressure2)

    has_cloud = cloud_base < nlev
    return jnp.where(has_cloud, fplume, 0.0), jnp.where(has_cloud, fmp2, 0.0)


class GissConvection(PhysicsTerm):
    """GISS ModelE moist convection (``MSTCNV``) as a PhysicsTerm.

    Returns zero tendencies and writes a
    :class:`~jcm.physics.modele.physics_data.GissConvectionData` under the public
    ``"convection"`` diagnostics key, populated with the diagnosed ``cloud_base``
    and closure ``cloud_base_mass_flux`` (the rest zero). ``requires`` is empty:
    the pressure / layer-mass diagnostics are read *optionally* so the term runs
    standalone; the tendency-producing chain (which will need them) is not wired
    in yet. See the module docstring.
    """

    name: ClassVar[str] = "giss_convection"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("convection",)

    def __init__(
        self,
        params: GissConvectionParameters | None = None,
        allow_mc: bool = False,
    ):
        """Hold scheme parameters and the static moist-convection switch.

        Args:
            params: Float-valued, differentiable parameters held as an
                ``nnx.Param`` so gradients flow through them once the tendency
                path lands.
            allow_mc: Static switch mirroring ModelE ``SCMopt%allowMC``. When
                ``True`` the full convective tendency chain runs (see
                :meth:`_convective_tendencies`); when ``False`` (default) the term
                only *diagnoses* cloud base + mass flux and returns zero
                tendencies. It is a plain (non-differentiable) attribute -- a
                trace-time code-path switch, so it is read with a Python ``if``.
        """
        self.params = nnx.Param(params or GissConvectionParameters.default())
        self.allow_mc = allow_mc

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute GISS convective tendencies (or just diagnose, if ``allow_mc``).

        Always diagnoses ``cloud_base`` + ``cloud_base_mass_flux``. If
        ``allow_mc`` is set, also runs the full chain (plume ascent ->
        compensating subsidence -> ``dth_mc``/``dq_mc``) and returns real
        temperature/humidity tendencies; otherwise returns zero tendencies. The
        tendency path is **not magnitude-validated** (the plume over-penetrates;
        see the module docstring), which is why it is off by default.

        Args:
            state: Column-vectorized ``PhysicsState`` (``(nlev, ncols)``).
            diagnostics: Forward-flowing diagnostics; ``pressure_full`` [Pa],
                ``layer_thickness`` [m], ``air_density`` [kg/m^3] are read when
                present (all three are required for the tendency path).
            forcing: Boundary-condition forcing (unused).
            terrain: Terrain boundary conditions (unused).

        Returns:
            ``(PhysicsTendency, diagnostics)`` with the ``GissConvectionData``
            (``cloud_base``, ``cloud_base_mass_flux``, and -- under ``allow_mc`` --
            ``dth_mc``/``dq_mc``) under ``"convection"``.
        """
        dtsrc = self.params.get_value().dtsrc
        shape = state.temperature.shape          # (nlev, ncols) in column mode
        nlev = shape[0]
        nodal_shape = shape[1:]

        cloud_base, fmp2 = self._diagnose(state, diagnostics, nlev, nodal_shape)

        if self.allow_mc:
            dtemp_dt, dq_dt, dth_mc, dq_mc = self._convective_tendencies(
                state, diagnostics, cloud_base, fmp2, nlev, nodal_shape, dtsrc)
        else:
            # Convection off: zero tendencies. Multiplying by dtsrc*0 keeps the
            # parameter in the trace so nnx.grad has a defined gradient path.
            dtemp_dt = jnp.zeros(shape) * dtsrc * 0.0
            dq_dt = jnp.zeros(shape)
            dth_mc = jnp.zeros((nlev,) + nodal_shape)
            dq_mc = jnp.zeros((nlev,) + nodal_shape)

        tendency = PhysicsTendency.zeros(shape).copy(
            temperature=dtemp_dt, specific_humidity=dq_dt)
        convection = GissConvectionData.zeros(nodal_shape, nlev).copy(
            cloud_base=cloud_base, cloud_base_mass_flux=fmp2,
            dth_mc=dth_mc, dq_mc=dq_mc)
        return tendency, {**diagnostics, "convection": convection}

    def _convective_tendencies(self, state, diagnostics, cloud_base, fmp2,
                               nlev, nodal_shape, dtsrc):
        """Full convective tendencies via the ported chain (the ``allow_mc`` path).

        Assembles: closure ``fmp2`` -> full-column entraining plume ascent
        (:func:`~jcm.physics.convection.giss_plume.plume_ascent_column`, launched
        at the traced cloud base) -> compensating subsidence + detrainment
        deposition (:func:`~jcm.physics.convection.giss_tendencies.convective_tendencies`)
        for potential temperature and specific humidity. Works **surface-first**
        internally (the GISS routines' convention) and flips the result back to
        the JCM top-first state for the ``PhysicsTendency``.

        NOT magnitude-validated: the plume over-penetrates the trade inversion, so
        the tendencies are structurally sensible (heating/moistening in the cloud
        layer) but their magnitude and vertical shape are not yet trustworthy.

        Needs ``pressure_full``/``layer_thickness``/``air_density``; without them
        returns zeros.

        Returns:
            ``(dT/dt [K/s], dq/dt [g/kg/s])`` top-first, and ``(dth_mc, dq_mc)``
            surface-first per-step diagnostics.
        """
        shape = state.temperature.shape
        pressure_full = diagnostics.get("pressure_full")
        thickness = diagnostics.get("layer_thickness")
        density = diagnostics.get("air_density")
        if pressure_full is None or thickness is None or density is None:
            zero2d = jnp.zeros(shape)
            zero3d = jnp.zeros((nlev,) + nodal_shape)
            return zero2d, zero2d, zero3d, zero3d

        # Surface-first environment (JCM state is top-first); humidity g/kg->kg/kg.
        p = jnp.flip(pressure_full, axis=0)
        t = jnp.flip(state.temperature, axis=0)
        q = jnp.flip(state.specific_humidity, axis=0) / 1000.0
        phi = jnp.flip(state.geopotential, axis=0)
        dz = jnp.flip(thickness, axis=0)
        air_mass = jnp.flip(density, axis=0) * dz
        exner = (p / _P_REF) ** KAPA
        theta_env = t / exner

        # Cloud-base source parcel: the surface (boundary-layer) potential
        # temperature lifted dry-adiabatically to the cloud-base level, saturated.
        cb = jnp.clip(cloud_base, 0, nlev - 1)

        def at_base(a):
            return jnp.take_along_axis(a, cb[None, ...], axis=0)[0]

        t_base = theta_env[0] * at_base(exner)
        q_base = saturation_specific_humidity(t_base, at_base(p))

        parcel_t, _cond, _buoy, mass_flux, det, _top = plume_ascent_column(
            cloud_base, t_base, q_base, at_base(phi),
            jnp.asarray(_CLOUD_BASE_W), fmp2, t, q, phi, p, dz, air_mass,
            contce=_CONTCE)

        # The plume profiles are zero outside the live cloud; use the environment
        # there so qsat/theta stay finite (they are multiplied by a zero mass flux
        # / zero detrainment anyway, but NaN*0 = NaN in JAX, so guard the inputs).
        parcel_t_safe = jnp.where(parcel_t > 1.0, parcel_t, t)
        theta_plume = parcel_t_safe / exner
        q_plume = saturation_specific_humidity(parcel_t_safe, p)   # in-cloud vapour

        dth_mc = convective_tendencies(mass_flux, theta_plume, det, dz,
                                       theta_env, air_mass)
        dq_mc = convective_tendencies(mass_flux, q_plume, det, dz, q, air_mass)

        # Per-step changes -> per-second rates; potential-temperature change ->
        # temperature change (x Exner). Flip back to JCM top-first for the state.
        dT_dt = jnp.flip(dth_mc * exner / dtsrc, axis=0)
        dq_dt = jnp.flip(dq_mc * 1000.0 / dtsrc, axis=0)          # kg/kg -> g/kg
        return dT_dt, dq_dt, dth_mc, dq_mc

    def _diagnose(self, state, diagnostics, nlev, nodal_shape):
        """Cloud base + closure mass flux per column (surface-first).

        Reads ``pressure_full`` (and, for the mass flux, ``layer_thickness`` /
        ``air_density``) from the diagnostics dict. Profiles are flipped to
        surface-first for the ported GISS routines. Without a pressure profile
        the term degrades to the pure scaffold (sentinel cloud base, zero mass
        flux); without the layer-mass diagnostics the cloud base is still
        diagnosed but the mass flux is zero.
        """
        pressure_full = diagnostics.get("pressure_full")
        if pressure_full is None:
            return (jnp.full(nodal_shape, nlev, dtype=int),
                    jnp.zeros(nodal_shape))

        # Surface-first profiles (JCM state is top-first); humidity g/kg -> kg/kg.
        p_sf = jnp.flip(pressure_full, axis=0)
        t_sf = jnp.flip(state.temperature, axis=0)
        q_sf = jnp.flip(state.specific_humidity, axis=0) / 1000.0

        cloud_base, _ = lifting_condensation_level(t_sf[0], p_sf[0], q_sf[0], p_sf)

        thickness = diagnostics.get("layer_thickness")
        density = diagnostics.get("air_density")
        if thickness is None or density is None:
            return cloud_base, jnp.zeros(nodal_shape)

        air_mass = jnp.flip(density, axis=0) * jnp.flip(thickness, axis=0)
        _, fmp2 = cloud_base_closure_mass_flux(
            t_sf, q_sf, p_sf, air_mass, cloud_base)
        return cloud_base, fmp2
