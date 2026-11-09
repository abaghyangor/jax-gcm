"""Diagnostic PyTree for the GISS ModelE convection port (DYCOMS-II SCM subset).

Mirrors the per-scheme ``ConvectionData`` pattern used by the Tiedtke-Nordeng
and SPEEDY convection schemes: a ``@tree_math.struct`` stored in the composable
``diagnostics`` dict under the public ``"convection"`` key (so it flows to
user-facing xarray output as ``convection.<field>``).

Arrays are in column-vectorized order ``(nlev, ncols)`` to match the composable
physics convention; per-column scalars are ``(ncols,)``. Vertical index 0 is the
surface (matching the ModelE oracle convention; the oracle reading/conversion
tooling lives in the separate ModelE-bridge repository).
"""

import jax.numpy as jnp
import tree_math
from jax import tree_util


@tree_math.struct
class GissConvectionData:
    """Moist-convective diagnostics from the GISS convection scheme.

    Attributes:
        dq_mc: Moist-convective specific-humidity tendency, ``(nlev, ncols)``
            (oracle diagnostic ``dq_mc``).
        dth_mc: Moist-convective potential-temperature tendency, ``(nlev, ncols)``
            (oracle diagnostic ``dth_mc``).
        mcp: Moist-convective surface precipitation rate [mm/day],
            ``(ncols,)`` (oracle ``mcp``, same units). Non-zero only on
            the buoyancy-sorting ``allow_mc`` path, which is the only one
            that accounts the plumes' rain.
        cloud_base: Convective cloud-base level index per column, ``(ncols,)``,
            in **surface-first** (ModelE) order -- index 0 is the surface, and the
            sentinel value ``nlev`` means the surface parcel never saturates (no
            cloud base / no convection). Integer-valued.
        cloud_base_mass_flux: Cloud-base convective plume mass ``fmp2`` [kg/m^2]
            per column, ``(ncols,)`` (the ``MASS_FLUX2`` closure output; zero
            where there is no cloud base). Sets the scale of the -- not yet wired
            -- convective tendencies.
        precip_conv: The same surface convective precipitation as ``mcp``, in
            the units and under the name the rest of jcm uses for it
            [kg/m^2/s], ``(ncols,)``. Both are carried because they serve
            different readers: ``EchamSurfaceExchange`` sums
            ``convection.precip_conv`` into the total surface precipitation and
            every other scheme in the tree reports a flux, while ``mcp`` is the
            ModelE oracle's own diagnostic name and rate, which the
            single-column validation compares against directly. 1 mm of water
            is 1 kg/m^2, so they differ only by the 86400 s in a day.

    Not carried: ``ktype``, ECHAM's convective-type code, which radiation reads
    through ``lagged_convection_type`` to pick a cloud-inhomogeneity factor.
    That reader already falls back to 0 for a struct without it, and this
    scheme has no equivalent trigger: ModelE's buoyancy sort does not classify
    a column as penetrative, shallow or mid-level, so a value here would be an
    invention that quietly moved radiation. The cost is that radiation treats a
    convecting column as non-convective for that one factor.

    """

    dq_mc: jnp.ndarray
    dth_mc: jnp.ndarray
    mcp: jnp.ndarray
    cloud_base: jnp.ndarray
    cloud_base_mass_flux: jnp.ndarray
    precip_conv: jnp.ndarray

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        """Zero-filled diagnostics. ``nodal_shape`` is ``(ncols,)`` in column
        mode (or the trailing grid shape in 3D mode).
        """
        return cls(
            dq_mc=jnp.zeros((nlev,) + nodal_shape),
            dth_mc=jnp.zeros((nlev,) + nodal_shape),
            mcp=jnp.zeros(nodal_shape),
            cloud_base=jnp.zeros(nodal_shape, dtype=int),
            cloud_base_mass_flux=jnp.zeros(nodal_shape),
            precip_conv=jnp.zeros(nodal_shape),
        )

    @classmethod
    def ones(cls, nodal_shape, nlev):
        return cls(
            dq_mc=jnp.ones((nlev,) + nodal_shape),
            dth_mc=jnp.ones((nlev,) + nodal_shape),
            mcp=jnp.ones(nodal_shape),
            cloud_base=jnp.ones(nodal_shape, dtype=int),
            cloud_base_mass_flux=jnp.ones(nodal_shape),
            precip_conv=jnp.ones(nodal_shape),
        )

    def copy(self, dq_mc=None, dth_mc=None, mcp=None, cloud_base=None,
             cloud_base_mass_flux=None, precip_conv=None):
        return GissConvectionData(
            dq_mc=dq_mc if dq_mc is not None else self.dq_mc,
            dth_mc=dth_mc if dth_mc is not None else self.dth_mc,
            mcp=mcp if mcp is not None else self.mcp,
            cloud_base=cloud_base if cloud_base is not None else self.cloud_base,
            cloud_base_mass_flux=(cloud_base_mass_flux
                                  if cloud_base_mass_flux is not None
                                  else self.cloud_base_mass_flux),
            precip_conv=(precip_conv if precip_conv is not None
                         else self.precip_conv),
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(
            lambda x, y: x or y, tree_util.tree_map(jnp.any, self)
        )
