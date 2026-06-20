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
        mcp: Moist-convective precipitation, ``(ncols,)`` (oracle ``mcp``).
    """

    dq_mc: jnp.ndarray
    dth_mc: jnp.ndarray
    mcp: jnp.ndarray

    @classmethod
    def zeros(cls, nodal_shape, nlev):
        """Zero-filled diagnostics. ``nodal_shape`` is ``(ncols,)`` in column
        mode (or the trailing grid shape in 3D mode)."""
        return cls(
            dq_mc=jnp.zeros((nlev,) + nodal_shape),
            dth_mc=jnp.zeros((nlev,) + nodal_shape),
            mcp=jnp.zeros(nodal_shape),
        )

    @classmethod
    def ones(cls, nodal_shape, nlev):
        return cls(
            dq_mc=jnp.ones((nlev,) + nodal_shape),
            dth_mc=jnp.ones((nlev,) + nodal_shape),
            mcp=jnp.ones(nodal_shape),
        )

    def copy(self, dq_mc=None, dth_mc=None, mcp=None):
        return GissConvectionData(
            dq_mc=dq_mc if dq_mc is not None else self.dq_mc,
            dth_mc=dth_mc if dth_mc is not None else self.dth_mc,
            mcp=mcp if mcp is not None else self.mcp,
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        return tree_util.tree_reduce(
            lambda x, y: x or y, tree_util.tree_map(jnp.any, self)
        )
