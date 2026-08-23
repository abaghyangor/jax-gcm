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
from jcm.physics.convection.giss_mass_flux import cloud_base_mass_flux_column
from jcm.physics.convection.giss_plume import plume_ascent_column
from jcm.physics.convection.giss_tendencies import convective_tendencies
from jcm.physics.convection.giss_thermodynamics import (
    GRAV,
    KAPA,
    SHA,
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

# Cloud-base adjustment timescale [s]. ``MSTCNV`` relaxes the closure mass flux
# toward neutrality over ``tadj`` rather than applying it all in one step:
# ``FMP2 = FMP2 * min(1, dtime/tadj)`` (line 2769). The caller passes
# ``tadj*seconds_per_hour``, so ``tadj`` is in **seconds** there; the default
# ``tadjmc(1) = 1`` hour gives 3600 s, i.e. a factor of 0.5 at dtsrc = 1800 s.
# Without this the scheme applies twice the convective mass flux per step.
_TADJ_SECONDS = 3600.0


def surface_flux_scales(sensible_heat_flux, evaporation, friction_velocity,
                        surface_density, surface_specific_humidity,
                        tqstar_factor: float = 1.0):
    """Surface-layer ``tstar``/``qstar`` scales that enhance the source parcel.

    ``MSTCNV`` warms and moistens the boundary-layer source parcel by the
    surface-flux scales before running the closure (``CLOUDS_DRV.F90:642``)::

        tstar = min(1,        max(0, SHF / (rho*cp*ustar)))
        qstar = min(0.2*q_sfc, max(0, evap / (rho*ustar)))

    scaled by ``mc_tqstar_fac`` (1.0 in the default preset). Both are floored at
    zero -- only an *upward* surface flux enhances the parcel -- and capped, so a
    strongly forced surface cannot run away.

    This matters more than its size suggests: paired with the multi-source
    closure (whose blended source humidity is nearly invariant during the
    bisection) a ~0.2 g/kg moisture boost shifts the whole ``DMSE1`` curve, and
    the latent term carries ``LHE/SHA ≈ 2490 K`` per kg/kg.

    Args:
        sensible_heat_flux: **Upward** sensible heat flux [W/m^2].
        evaporation: Surface evaporation rate [kg/m^2/s].
        friction_velocity: ``ustar`` [m/s].
        surface_density: Surface air density [kg/m^3].
        surface_specific_humidity: Surface specific humidity [kg/kg] (caps qstar).
        tqstar_factor: ``mc_tqstar_fac``.

    Returns:
        ``(tstar, qstar)`` -- source-parcel temperature [K] and humidity [kg/kg]
        enhancements.
    """
    scale = surface_density * jnp.maximum(friction_velocity, 1.0e-6)
    tstar = jnp.clip(sensible_heat_flux / (scale * SHA), 0.0, 1.0)
    qstar = jnp.clip(evaporation / scale, 0.0,
                     0.2 * surface_specific_humidity)
    return tqstar_factor * tstar, tqstar_factor * qstar


def convective_velocity_scale(sensible_heat_flux, evaporation,
                              boundary_layer_height, surface_density,
                              reference_theta, minimum=0.5):
    """Cloud-base updraft seed from the convective velocity scale ``w*``.

    ``MSTCNV`` seeds the plume with ``wbases = max(0.5, wturb)`` -- the
    boundary-layer turbulent velocity from its own BL scheme (``MSTCNV`` line
    ~2807). JCM has no direct equivalent, so we use the standard convective
    velocity scale built from the surface **buoyancy** flux and the boundary-layer
    depth::

        w* = (g * z_i * (w'theta' + 0.61*theta*w'q') / theta)**(1/3)

    On BOMEX this gives ~0.6 m/s against the oracle's observed 0.57 -- the right
    scale, from inputs we actually have.

    The seed matters more than its size suggests: the entrainment rate goes as
    ``B/w**2``, so a plume seeded too slowly entrains hard, dilutes, and dies
    early, while one that gets moving stays undilute. The ``max(0.5, ...)`` floor
    is ModelE's.
    """
    heat_flux = sensible_heat_flux / (surface_density * SHA)     # K m/s
    moisture_flux = evaporation / surface_density                # kg/kg m/s
    buoyancy_flux = heat_flux + 0.61 * reference_theta * moisture_flux
    w_cubed = (GRAV * boundary_layer_height
               * jnp.maximum(buoyancy_flux, 0.0) / reference_theta)
    return jnp.maximum(w_cubed ** (1.0 / 3.0), minimum)


def cloud_base_closure_mass_flux(temperature: jnp.ndarray,
                                 specific_humidity: jnp.ndarray,
                                 pressure: jnp.ndarray,
                                 air_mass: jnp.ndarray,
                                 cloud_base: jnp.ndarray,
                                 boundary_layer_top=None,
                                 source_dtheta=0.0,
                                 source_dq=0.0):
    """Cloud-base plume fraction / mass from the ``MASS_FLUX2`` closure.

    Broadcasting-native wrapper around
    :func:`~jcm.physics.convection.giss_mass_flux.cloud_base_mass_flux` that
    builds the three-level cloud-base stencil from full column profiles and a
    per-column cloud-base index.

    All profile inputs are **surface-first** ``(nlev, ...)`` (index 0 = surface);
    ``cloud_base`` ``(...)`` is the lifting-condensation-level index (the sentinel
    ``nlev`` means no cloud base). The stencil spans ``[lmin, lmin+1, lmin+2]``
    and its source level (index 0) is replaced by the **surface** parcel -- the
    boundary-layer air that actually feeds the plume.

    The closure level is ``lmin = cloud_base``. This is set by the plume oracle
    (``modele_patches/`` in the bridge repo): ModelE's ``LMIN`` for the plume that
    actually fires sits at the LCL or one level above it across the BOMEX
    periods, never below. The closure is *very* sensitive to this -- on period 47
    it returns ``fmp2`` of 6.8 / 9.4 / 29.9 kg/m² at ``lmin`` = LCL-1 / LCL / LCL+1
    -- so evaluating it a level or two low (as ``cloud_base - 1`` did) badly
    under-computes the cloud-base mass flux.

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

    fplume, fmp2, _ = cloud_base_mass_flux_column(
        theta, specific_humidity, air_mass, exner, pressure, cloud_base,
        boundary_layer_top=boundary_layer_top,
        source_dtheta=source_dtheta, source_dq=source_dq)

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

        # Cloud-base source parcel. The plume is seeded with the **same** parcel
        # the closure is built on: the mass-weighted boundary-layer blend plus
        # the surface-flux enhancement, lifted to cloud base. Seeding instead
        # with raw surface air re-saturated at cloud base discards the parcel's
        # moisture excess -- the very water whose condensation warms the plume --
        # so it starts marginally (often negatively) buoyant, entrains hard
        # (``eps ~ B/w^2``) and dies early. Validated against the plume oracle:
        # this moves the BOMEX cloud top from 10/10/15/15 to 15/15/16/16 for
        # periods 12/24/36/47, against ModelE's 16/17/18/20.
        cb = jnp.clip(cloud_base, 0, nlev - 1)

        def at_base(a):
            return jnp.take_along_axis(a, cb[None, ...], axis=0)[0]

        theta_source, q_source, w_base = self._plume_seed(
            diagnostics, theta_env, q, air_mass, nlev)
        t_base = theta_source * at_base(exner)

        # Relax the closure mass flux over the cloud-base adjustment timescale
        # (MSTCNV line 2769): only the fraction of the neutralising mass flux
        # that fits in one physics step is applied. The `cloud_base_mass_flux`
        # diagnostic stays the raw (pre-relaxation) closure value.
        m_base = fmp2 * jnp.minimum(1.0, dtsrc / _TADJ_SECONDS)

        parcel_t, _cond, _buoy, mass_flux, det, _top = plume_ascent_column(
            cloud_base, t_base, q_source, at_base(phi),
            w_base, m_base, t, q, phi, p, dz, air_mass,
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
        blt, dtheta, dq = self._source_parcel_inputs(
            diagnostics, q_sf, jnp.flip(density, axis=0)[0], nlev)

        # The closure runs at the level where **ModelE's** parcel -- the
        # boundary-layer *blend*, not the surface air -- reaches saturation. The
        # blend is drier than the surface parcel, so it saturates a level higher,
        # and the closure is very level-sensitive, so using the surface LCL here
        # evaluates it one level too low and the closure bottoms out. Validated
        # against the ModelE closure oracle: the blend's LCL reproduces `LMIN` on
        # 6 of 7 sampled BOMEX periods (the surface LCL on 2).
        #
        # The *reported* ``cloud_base`` stays the surface-parcel LCL, which is the
        # quantity validated against ModelE's ``cldmc`` (47/48 exact).
        closure_base = self._blended_parcel_cloud_base(t_sf, q_sf, p_sf,
                                                       air_mass, blt, nlev)
        _, fmp2 = cloud_base_closure_mass_flux(
            t_sf, q_sf, p_sf, air_mass, closure_base,
            boundary_layer_top=blt, source_dtheta=dtheta, source_dq=dq)
        return cloud_base, fmp2

    def _plume_seed(self, diagnostics, theta_env, q_sf, air_mass, nlev):
        """Source parcel and updraft seed for the plume ascent.

        Returns the mass-weighted boundary-layer blend of potential temperature
        and humidity -- enhanced by the surface-flux scales, exactly as the
        closure's parcel is -- together with the cloud-base updraft speed from
        :func:`convective_velocity_scale`. Falls back to surface air and the
        ``0.5 m/s`` floor when the boundary-layer/surface diagnostics are absent.
        """
        density = diagnostics.get("air_density")
        surface_density = (1.2 if density is None
                           else jnp.flip(density, axis=0)[0])
        blt, dtheta, dq = self._source_parcel_inputs(
            diagnostics, q_sf, surface_density, nlev)

        level = jnp.arange(nlev).reshape((nlev,) + (1,) * (theta_env.ndim - 1))
        top = nlev - 3 if blt is None else blt
        weights = jnp.where(level <= top, air_mass, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=0), 1.0e-20)
        theta_source = jnp.sum(theta_env * weights, axis=0) + dtheta
        q_source = jnp.sum(q_sf * weights, axis=0) + dq

        surface = diagnostics.get("surface")
        pbl_height = diagnostics.get("boundary_layer_height")
        if surface is None or pbl_height is None:
            return theta_source, q_source, jnp.asarray(_CLOUD_BASE_W)
        w_base = convective_velocity_scale(
            surface.sensible_heat_flux, surface.evaporation, pbl_height,
            surface_density, theta_source, minimum=_CLOUD_BASE_W)
        return theta_source, q_source, w_base

    def _blended_parcel_cloud_base(self, t_sf, q_sf, p_sf, air_mass,
                                   boundary_layer_top, nlev):
        """Level at which the mass-weighted boundary-layer blend saturates.

        ``MSTCNV``'s source parcel is the ``fpi``-weighted blend of the
        boundary-layer levels, so its saturation level -- not the surface
        parcel's -- sets the closure level ``LMIN``.
        """
        level = jnp.arange(nlev).reshape((nlev,) + (1,) * (t_sf.ndim - 1))
        top = nlev - 3 if boundary_layer_top is None else boundary_layer_top
        weights = jnp.where(level <= top, air_mass, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=0), 1.0e-20)
        exner = (p_sf / _P_REF) ** KAPA
        theta_blend = jnp.sum((t_sf / exner) * weights, axis=0)
        q_blend = jnp.sum(q_sf * weights, axis=0)
        base, _ = lifting_condensation_level(
            theta_blend * exner[0], p_sf[0], q_blend, p_sf)
        return jnp.clip(base, 0, nlev - 3)

    def _source_parcel_inputs(self, diagnostics, q_surface_first,
                              surface_density, nlev):
        """Boundary-layer top and source-parcel enhancement, if available.

        The closure blends its source parcel over the boundary layer and boosts
        it by the surface-flux scales. Both inputs are read *optionally*:

        * ``boundary_layer_height`` [m] (with ``height_full``) gives ``dcl``, the
          level **below** the boundary-layer top -- ``MSTCNV`` excludes source
          levels above it so a parcel displaced through the top is not mixed with
          free-tropospheric air. Note the off-by-one: ``searchsorted`` returns the
          insertion index, and ``dcl`` is one below it.
        * ``surface`` (a ``SurfaceData``) supplies the fluxes for
          :func:`surface_flux_scales`; ``ustar`` comes from the momentum fluxes,
          ``ustar = sqrt(|tau|/rho)``.

        Without them the closure falls back to blending over the whole sub-cloud
        column with no surface enhancement, which under-computes the cloud-base
        mass flux (see ``STATUS.md``).
        """
        height = diagnostics.get("height_full")
        pbl_height = diagnostics.get("boundary_layer_height")
        boundary_layer_top = None
        if height is not None and pbl_height is not None:
            height_sf = jnp.flip(height, axis=0)
            below_top = jnp.sum(
                (height_sf < pbl_height[None, ...]).astype(int), axis=0) - 1
            boundary_layer_top = jnp.clip(below_top, 0, nlev - 3)

        surface = diagnostics.get("surface")
        if surface is None:
            return boundary_layer_top, 0.0, 0.0
        tau = jnp.sqrt(surface.momentum_flux_u ** 2
                       + surface.momentum_flux_v ** 2)
        ustar = jnp.sqrt(tau / jnp.maximum(surface_density, 1.0e-6))
        dtheta, dq = surface_flux_scales(
            surface.sensible_heat_flux, surface.evaporation, ustar,
            surface_density, q_surface_first[0])
        return boundary_layer_top, dtheta, dq
