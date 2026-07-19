"""GISS ModelE moist convection as a composable PhysicsTerm.

The JCM landing zone for the GISS ModelE moist convection routine ``MSTCNV``
(``modelE/model/MSTCNV.F90``, called from ``CONDSE_column`` in
``modelE/model/CLOUDS_DRV.F90``). Per the repo's by-process organization the
convection *term* lives here under ``jcm/physics/convection/`` (named after the
scheme); the parameter/diagnostic structs live under :mod:`jcm.physics.modele`.
The ModelE oracle reading/conversion tooling lives in a *separate* repository;
this repo only carries committed fixture arrays (``jcm/data/test/modele/``) for
tests. See ``ARCHITECTURE.md`` for the full port design.

Status: cloud-base + cloud-base mass-flux diagnostics; zero tendencies
---------------------------------------------------------------------
:class:`GissConvection` returns **zero** tendencies but *diagnoses* two things
from the state when the column pressure / layer-mass diagnostics are available:

1. ``cloud_base`` -- the convective cloud base (lifting condensation level) of a
   surface parcel, via :func:`~jcm.physics.convection.giss_cloud_base.lifting_condensation_level`.
2. ``cloud_base_mass_flux`` -- the ``MASS_FLUX2`` closure plume mass ``fmp2``
   [kg/m^2] at cloud base (:func:`cloud_base_closure_mass_flux`). This is the
   *validated* piece: on the BOMEX oracle it gives physical values
   (``fmp2 ~ 2-8 kg/m^2``) for well-triggered columns.

The tendency-producing chain (plume ascent -> compensating subsidence ->
``dth_mc``/``dq_mc``) is **not wired in yet**: the ported plume over-penetrates
the trade inversion (see ``STATUS.md``), so its magnitudes are not validated.
When it is wired in it will be gated behind ``allow_mc`` (default off) so the
model ignores the unvalidated tendencies by default. The zero tendencies here
are honest, not a validation result.

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
from jcm.physics.convection.giss_thermodynamics import KAPA

# Reference pressure for the Exner function. The closure recovers temperature as
# ``theta*exner``, so the reference cancels and its specific value is arbitrary;
# 1000 hPa is the conventional choice and matches the BOMEX validation.
_P_REF = 100000.0


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
            allow_mc: Static switch mirroring ModelE ``SCMopt%allowMC``. Reserved
                for gating the (not-yet-wired) convective tendencies; defaults to
                ``False``. A plain (non-differentiable) attribute.
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
        """Diagnose cloud base + cloud-base mass flux; return zero tendencies.

        Args:
            state: Column-vectorized ``PhysicsState`` (``(nlev, ncols)``).
            diagnostics: Forward-flowing diagnostics; ``pressure_full`` [Pa],
                ``layer_thickness`` [m], ``air_density`` [kg/m^3] are read when
                present.
            forcing: Boundary-condition forcing (unused).
            terrain: Terrain boundary conditions (unused).

        Returns:
            ``(PhysicsTendency, diagnostics)`` with a zero tendency and a
            ``GissConvectionData`` (``cloud_base`` + ``cloud_base_mass_flux``
            populated) under ``"convection"``.
        """
        # Keep the parameter in the trace so nnx.grad has a defined (zero)
        # gradient path while the tendencies are zero.
        dtsrc = self.params.get_value().dtsrc
        shape = state.temperature.shape          # (nlev, ncols) in column mode
        nlev = shape[0]
        nodal_shape = shape[1:]

        zero_field = jnp.zeros(shape) * dtsrc * 0.0
        tendency = PhysicsTendency.zeros(shape).copy(
            temperature=zero_field,
            specific_humidity=zero_field,
        )

        cloud_base, fmp2 = self._diagnose(state, diagnostics, nlev, nodal_shape)
        convection = GissConvectionData.zeros(nodal_shape, nlev).copy(
            cloud_base=cloud_base, cloud_base_mass_flux=fmp2)
        return tendency, {**diagnostics, "convection": convection}

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
