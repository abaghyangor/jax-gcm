"""Adapt ModelE oracle columns into a JCM ``PhysicsState``.

Bridges the pure NetCDF reader (:mod:`jcm.physics.modele.oracle`, numpy) to the
composable-physics state representation (jax). Kept separate from ``oracle.py``
so the reader stays dependency-light (numpy / netCDF4 only).

Conventions applied here (see :mod:`jcm.physics.modele.oracle` for provenance):

* Output arrays are column-vectorized ``(nlev, ncols)`` with vertical index 0 =
  surface; for the SCM oracle ``ncols == 1``.
* ModelE ``q`` is kg/kg; JCM ``PhysicsState.specific_humidity`` is g/kg, so we
  multiply by 1000.
* ModelE ``z`` is geopotential *height* in metres; ``PhysicsState.geopotential``
  is geopotential energy per unit mass, so we multiply by ``g``.
* ``normalized_surface_pressure`` is ``p_surface / p0``. We take the oracle's
  surface-layer pressure ``p_3d[0]`` (hPa) over a reference ``p0`` (hPa).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jcm.physics_interface import PhysicsState
from jcm.physics.modele import oracle

# Standard gravity used by ModelE to convert geopotential <-> height.
_G = 9.80665
# Reference pressure for normalized surface pressure (hPa). ModelE's p0 is
# 1013.25 hPa (1 standard atmosphere); the oracle surface layer sits near this.
_P0_HPA = 1013.25


def oracle_to_physics_state(path: str, period: int, *, p0_hpa: float = _P0_HPA):
    """Build a column ``PhysicsState`` from one oracle period.

    Args:
        path: Oracle NetCDF path (e.g. ``oracle.fixture_path()``).
        period: Period index (0..47).
        p0_hpa: Reference pressure (hPa) for the normalized surface pressure.

    Returns:
        A :class:`~jcm.physics_interface.PhysicsState` with ``(nlev, ncols)``
        arrays (``ncols == 1`` for the SCM oracle), ``specific_humidity`` in
        g/kg, ``geopotential`` in m^2/s^2, and ``u``/``v`` winds from the oracle.
    """
    def col(name):
        # (jm, im, lm) -> (lm, ncols)
        native = oracle.read_state_field(path, name, period=period)
        return jnp.asarray(oracle.to_columns(np.asarray(native)))

    t = col("t")
    q_kgkg = col("q")
    z_m = col("z")
    p_hpa = col("p_3d")

    nlev, ncols = t.shape
    p_surface = p_hpa[0]  # surface layer, (ncols,)

    return PhysicsState(
        u_wind=col("u"),
        v_wind=col("v"),
        temperature=t,
        specific_humidity=q_kgkg * 1000.0,  # kg/kg -> g/kg
        geopotential=z_m * _G,              # height (m) -> geopotential (m^2/s^2)
        normalized_surface_pressure=p_surface / p0_hpa,
        tracers={},
    )
