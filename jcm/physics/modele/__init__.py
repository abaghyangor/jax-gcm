"""ModelE convection port -- model infrastructure (DYCOMS-II RF02 SCM subset).

This package holds the GISS ModelE-specific *infrastructure* for converting the
moist convection routine ``MSTCNV`` (``modelE/model/MSTCNV.F90``) into JCM,
using the verified local ModelE DYCOMS-II RF02 single-column run as a Fortran
oracle. Mirroring how ``speedy/`` and ``icon/`` hold their model infrastructure,
this package contains:

* :mod:`jcm.physics.modele.oracle` -- reads named diagnostics out of the packed
  ModelE sub-daily NetCDF output (``allsteps.subdddycoms_scm.nc``).
* :mod:`jcm.physics.modele.adapter` -- builds a column ``PhysicsState`` from an
  oracle period.
* :mod:`jcm.physics.modele.params` -- ``GissConvectionParameters`` PyTree.
* :mod:`jcm.physics.modele.physics_data` -- ``GissConvectionData`` diagnostics.

The convection *term* itself is a composable ``PhysicsTerm`` and -- per the
repo's by-process organization -- lives under
``jcm/physics/convection/giss_mstcnv.py`` (:class:`~jcm.physics.convection.giss_mstcnv.GissConvection`).
It is currently a zero-returning scaffold, NOT a validated port; see
``README.md`` for the next-step plan and the important zero-moist-convection
finding for this DYCOMS case.
"""

from jcm.physics.modele import oracle  # noqa: F401
from jcm.physics.modele import adapter  # noqa: F401
