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
        max_dt_overshoot: Largest plume-environment virtual temperature
            difference, in kelvin, that a plume may overshoot into before the
            ascent stops (``MSTCNV.F90:1923``). The default of 1.0 is ModelE's
            own untuned default (``:297``) and is what the instrumented SCM runs
            this port is validated against used, confirmed in their printout, so
            it is the value that reproduces the oracle. GISS's *tuned* presets
            raise it to 1.5876319, 1.6188563 or 2.0862949 (``:369``, ``:410``,
            ``:460``), which lets a plume penetrate further before terminating.
            It is a parameter rather than a constant because that test is what
            ends every deep plume -- 30% of ModelE's TWP-ICE plumes and 80% once
            the term's prognostic inputs are absent, as they are in a global
            run -- so the global depth comparison against Tiedtke cannot be made
            without being able to set it. See
            ``docs/source/design/giss_convection_port.md``.

    """

    dtsrc: jnp.ndarray
    # Carries its default on the field, so adding a tunable here does not break
    # every partial construction -- `default()` still states it explicitly, and
    # the two are pinned equal by `TestMaxOvershootIsATunableParameter`.
    max_dt_overshoot: jnp.ndarray = 1.0

    @classmethod
    def default(cls):
        return cls(dtsrc=jnp.array(1800.0),
                   max_dt_overshoot=jnp.array(1.0))

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
