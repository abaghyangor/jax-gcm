"""GISS ModelE moist convection as a composable PhysicsTerm -- SCAFFOLD ONLY.

This is the JCM landing zone for the GISS ModelE moist convection routine
``MSTCNV`` (``modelE/model/MSTCNV.F90``, called from ``CONDSE_column`` in
``modelE/model/CLOUDS_DRV.F90``), being converted using the verified local
ModelE DYCOMS-II RF02 single-column run as a Fortran oracle.

Per the repo's by-process organization, the convection *term* lives here under
``jcm/physics/convection/`` (named after the scheme), while the model-specific
*infrastructure* it consumes -- oracle reader, parameters, data struct -- lives
under :mod:`jcm.physics.modele`.

Status: Milestone 1 scaffold
----------------------------
:class:`GissConvection` currently returns **zero** tendencies and zero
diagnostics. It is NOT a validated ModelE convection implementation. Zero output
happens to agree with the verified DYCOMS-II RF02 oracle, but ONLY because moist
convection is inactive there (``dq_mc``/``dth_mc``/``mcp`` are identically zero
across all 48 periods -- all precipitation is stratiform). That agreement is
trivial and must not be read as scientific validation. See
``jcm/physics/modele/README.md``.

Next steps ("Option B", DYCOMS/SCM subset of ``MSTCNV``)
-------------------------------------------------------
1. Choose a convectively active oracle (BOMEX/RICO, or DYCOMS with
   ``SCMopt%allowMC`` enabled) -- this oracle cannot exercise nonzero MC.
2. Port ``CLOUD_BASE`` / ``cloud_base_closure`` triggering + cloud-base mass
   flux, then the plume lift/condensation loop (``CLOUD_TOP``,
   ``plume_ent_det_w2``), downdrafts, subsidence, ``EVAP_PRECIP``.
3. Populate ``GissConvectionData`` and the ``PhysicsTendency`` from those.
4. Start with process-level/diagnostic agreement; tighten tolerances later.

Track every omitted ModelE branch here as the port grows. Convective triggering
introduces hard branches; preserve JAX differentiability with ``jnp.where`` /
``jax.lax.cond`` and document discontinuities.

Conventions
-----------
* Operates on column-vectorized state ``(nlev, ncols)`` with vertical index 0 =
  surface, matching the composable physics convention.
* ``state.specific_humidity`` is g/kg (JCM); ModelE oracle ``q`` is kg/kg.
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


class GissConvection(PhysicsTerm):
    """GISS ModelE moist convection (``MSTCNV``) as a PhysicsTerm. SCAFFOLD.

    Returns zero tendencies and writes a zero-filled
    :class:`~jcm.physics.modele.physics_data.GissConvectionData` under the
    public ``"convection"`` diagnostics key. The scheme requires no upstream
    diagnostics yet (a zero scaffold reads only ``state``), so ``requires`` is
    empty; it ``provides`` ``"convection"``.
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
                ``nnx.Param`` so gradients flow through them once physics lands.
            allow_mc: Static switch mirroring ModelE ``SCMopt%allowMC``. The
                verified DYCOMS oracle ran with moist convection inactive, so
                this defaults to ``False``. It is a plain (non-differentiable)
                attribute, not an ``nnx.Param``.
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
        """Compute GISS convective tendencies. SCAFFOLD: returns zeros.

        Args:
            state: Column-vectorized ``PhysicsState`` (``(nlev, ncols)``).
            diagnostics: Forward-flowing diagnostics dict (unused by the
                scaffold beyond shape inference; real physics will read
                pressure/thickness diagnostics here).
            forcing: Boundary-condition forcing (unused in the scaffold).
            terrain: Terrain boundary conditions (unused in the scaffold).

        Returns:
            ``(PhysicsTendency, diagnostics)`` where the tendency is zero and a
            zero ``GissConvectionData`` is written under ``"convection"``. The
            term is a flax ``nnx.Module``; ``nnx.grad`` flows through
            ``self.params`` (gradients are zero while the scaffold returns zero).
        """
        # Reference self.params so the parameter participates in the trace and
        # gradients are defined (zero) w.r.t. it. Multiplying zeros by the
        # parameter keeps the tendency identically zero while wiring autodiff.
        dtsrc = self.params.get_value().dtsrc
        shape = state.temperature.shape          # (nlev, ncols) in column mode
        nlev = shape[0]
        nodal_shape = shape[1:]

        zero_field = jnp.zeros(shape) * dtsrc * 0.0
        tendency = PhysicsTendency.zeros(shape).copy(
            temperature=zero_field,
            specific_humidity=zero_field,
        )

        convection = GissConvectionData.zeros(nodal_shape, nlev)
        return tendency, {**diagnostics, "convection": convection}
