"""Parameters for the GISS ModelE convection port (DYCOMS-II SCM subset).

Milestone 1 keeps this intentionally small: only the constants needed to wire
shapes and (eventually) tendency-unit conversions. Real ModelE tuning constants
from ``MSTCNV.F90`` / ``CLOUDS_COM`` will be added as the physics is ported.

This is *model infrastructure* and lives under ``jcm/physics/modele/`` alongside
the oracle reader and data structs, mirroring how ``speedy/`` and ``icon/`` hold
their own parameter containers. The convection *term* that consumes these lives
under ``jcm/physics/convection/`` (named after the scheme), per the repo's
by-process organization.
"""

import tree_math
import jax.numpy as jnp
from jax import tree_util


@tree_math.struct
class GissConvectionParameters:
    """Static, differentiable parameters for the GISS convection scheme.

    Held by :class:`~jcm.physics.convection.giss_mstcnv.GissConvection` as an
    ``nnx.Param`` so gradients can flow through them once real physics lands.
    Only float-valued (differentiable) parameters belong here; static switches
    such as ``allow_mc`` are plain ``__init__`` kwargs on the term.

    Attributes:
        dtsrc: Source/physics time step in seconds. The verified DYCOMS run uses
            1800 s (30-min sub-daily output, 48 periods over 24 h).
    """

    dtsrc: jnp.ndarray

    @classmethod
    def default(cls):
        return cls(dtsrc=jnp.array(1800.0))

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
