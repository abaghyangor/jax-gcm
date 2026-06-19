"""Reader for the ModelE DYCOMS-II RF02 SCM sub-daily oracle NetCDF.

The verified local ModelE run writes packed sub-daily diagnostics to::

    /Users/gor/ModelE_Support/huge_space/dycoms_scm/allsteps.subdddycoms_scm.nc

ModelE stores groups of diagnostics as one packed array plus a parallel
fixed-width character *name table*. For example::

    aijlh1(nperiod_aijlh1, jm, im, kaijlh1, lm_aijlh1)   # 3D state fields
    sname_aijlh1(kaijlh1, sname_strlen)                  # names of those fields
    scale_aijlh1(kaijlh1)                                # raw -> physical scale
    denom_aijlh1(kaijlh1)                                # ratio denominator index

To read a field by name you look up its name in ``sname_*`` to get the ``k``
index, then slice the packed array along the diagnostic axis and multiply by
``scale_*[k]`` to recover physical units.

This module provides that lookup. Milestone 1 ("Option C") only needs reliable,
documented extraction -- no physics. A small self-contained fixture extracted
from the verified oracle ships at ``jcm/data/test/modele/dycoms_oracle_subset.nc``
so tests do not depend on the external run directory.

Diagnostic groups used here
---------------------------
``aijlh1`` (3D atmospheric state), names ``sname_aijlh1``:
    ``u, v, t, th, q, rhw_rhi, rhw, z, p_3d``
``cijlh1`` (3D cloud/convection), names ``sname_cijlh1``:
    ``qcl, qci, cldss, cldmc, cfr, dq_mc, dth_mc, dq_ss, dth_ss``
``cijh1`` (2D column cloud/precip), names ``sname_cijh1``:
    ``prec, mcp, ssp, cldmc_2d, cldss_2d, cldtot_2d, lwp, iwp, tau_ss, tau_mc``

Conventions (verified against the oracle, see ``README.md``)
-----------------------------------------------------------
* The SCM grid is a single column: ``im == 1`` (x/lon) and ``jm == 1`` (y/lat).
* There are 48 periods (30-min sub-daily output over the 24 h DYCOMS case).
* There are ``lm = 63`` vertical layers. **Vertical index 0 is the surface /
  lowest layer** (``p_3d[0] ~= 1013 hPa``); index 62 is model top
  (``p_3d[62] ~= 0.14 hPa``). This matches ModelE's ``L=1`` = lowest layer.
* Stored arrays are returned with the diagnostic axis removed. State/cloud
  profile readers can optionally transpose a single period to JCM column order
  ``(vertical, x, y) = (lm, im, jm)`` to match the SPEEDY ``(kx, ix, il)`` test
  convention, preserving the singleton ``im``/``jm`` axes.

Units after ``apply_scale=True`` (raw * scale)
----------------------------------------------
============  =========================  ===============================
name          units                      notes
============  =========================  ===============================
``t``         K                          absolute temperature
``th``        K                          potential temperature (scale 7.22)
``q``         kg/kg                       water vapour mixing ratio
``z``         m                           geopotential height (scale 1/g)
``p_3d``      hPa                         layer pressure
``dq_mc``     kg/kg/day                   moist-convective q tendency
``dth_mc``    K/day                       moist-convective theta tendency
``qcl``       kg/kg                       cloud liquid water
``qci``       kg/kg                       cloud ice water
``cldmc``     fraction                    convective cloud fraction
``prec``      mm/day                      total precipitation
``mcp``       mm/day                      moist-convective precipitation
``ssp``       mm/day                      stratiform (large-scale) precip
============  =========================  ===============================

Note ``q`` is **kg/kg** here. JCM ``PhysicsState.specific_humidity`` is g/kg;
multiply by 1000 when populating a ``PhysicsState`` (see README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources

import netCDF4
import numpy as np

# Live oracle written by the verified ModelE DYCOMS-II RF02 SCM run.
DEFAULT_ORACLE_PATH = (
    "/Users/gor/ModelE_Support/huge_space/dycoms_scm/allsteps.subdddycoms_scm.nc"
)

# Which packed array / name table each named diagnostic lives in.
_GROUP_AIJLH1 = ("aijlh1", "sname_aijlh1", "scale_aijlh1")  # 3D state
_GROUP_CIJLH1 = ("cijlh1", "sname_cijlh1", "scale_cijlh1")  # 3D cloud/convection
_GROUP_CIJH1 = ("cijh1", "sname_cijh1", "scale_cijh1")      # 2D column

# name -> group, for convenience lookups.
STATE_FIELDS = ("u", "v", "t", "th", "q", "rhw_rhi", "rhw", "z", "p_3d")
CONVECTION_FIELDS = (
    "qcl", "qci", "cldss", "cldmc", "cfr", "dq_mc", "dth_mc", "dq_ss", "dth_ss",
)
COLUMN_FIELDS = (
    "prec", "mcp", "ssp", "cldmc_2d", "cldss_2d", "cldtot_2d",
    "lwp", "iwp", "tau_ss", "tau_mc",
)

# Documented physical units (after apply_scale=True).
UNITS = {
    "u": "m/s", "v": "m/s", "t": "K", "th": "K", "q": "kg/kg",
    "rhw_rhi": "fraction", "rhw": "fraction", "z": "m", "p_3d": "hPa",
    "qcl": "kg/kg", "qci": "kg/kg", "cldss": "fraction", "cldmc": "fraction",
    "cfr": "fraction", "dq_mc": "kg/kg/day", "dth_mc": "K/day",
    "dq_ss": "kg/kg/day", "dth_ss": "K/day",
    "prec": "mm/day", "mcp": "mm/day", "ssp": "mm/day",
    "cldmc_2d": "fraction", "cldss_2d": "fraction", "cldtot_2d": "fraction",
    "lwp": "kg/m2", "iwp": "kg/m2", "tau_ss": "1", "tau_mc": "1",
}


def fixture_path() -> str:
    """Return the path to the committed self-contained oracle fixture.

    Returns
    -------
    str
        Filesystem path to ``jcm/data/test/modele/dycoms_oracle_subset.nc``.
    """
    return str(resources.files("jcm.data.test") / "modele" / "dycoms_oracle_subset.nc")


def default_path(prefer_live: bool = False) -> str:
    """Pick an oracle path: the committed fixture, or the live run if present.

    Args:
        prefer_live: If True and the live oracle exists, return it; otherwise
            fall back to the committed fixture.

    Returns:
        A path that exists, preferring the fixture for reproducibility.
    """
    if prefer_live and os.path.exists(DEFAULT_ORACLE_PATH):
        return DEFAULT_ORACLE_PATH
    fp = fixture_path()
    if os.path.exists(fp):
        return fp
    return DEFAULT_ORACLE_PATH


def decode_names(char_rows) -> list[str]:
    """Decode a fixed-width NetCDF character name table into stripped strings.

    Args:
        char_rows: 2D character array ``(k, strlen)`` from a ``sname_*`` variable.

    Returns:
        List of ``k`` whitespace-stripped names.
    """
    return [str(netCDF4.chartostring(row)).strip() for row in np.asarray(char_rows)]


def read_named_diagnostic(
    path: str,
    data_var: str,
    name_var: str,
    target_name: str,
    *,
    scale_var: str | None = None,
    apply_scale: bool = True,
):
    """Extract one named diagnostic slice from a packed ModelE diagnostic array.

    The diagnostic axis (``k...``) is the second-to-last axis for the 3D groups
    (before ``lm``) and the last axis for the 2D ``cijh1`` group. This function
    locates ``target_name`` in ``name_var`` and slices that axis out.

    Args:
        path: NetCDF file path.
        data_var: Packed data variable name, e.g. ``"aijlh1"``.
        name_var: Parallel name-table variable, e.g. ``"sname_aijlh1"``.
        target_name: Diagnostic name to extract, e.g. ``"dth_mc"``.
        scale_var: Optional ``scale_*`` variable for unit recovery. If None it is
            inferred as ``data_var`` with an ``"s"``-prefixed sibling lookup is
            not attempted; pass explicitly or rely on the group helpers below.
        apply_scale: If True, multiply the slice by ``scale_var[k]`` to get
            physical units.

    Returns:
        numpy.ndarray with the diagnostic axis removed. For ``aijlh1``/``cijlh1``
        the returned shape is ``(nperiod, jm, im, lm)``; for ``cijh1`` it is
        ``(nperiod, jm, im)``.

    Raises:
        KeyError: If ``target_name`` is not present in the name table.
    """
    with netCDF4.Dataset(path) as ds:
        names = decode_names(ds.variables[name_var][:])
        if target_name not in names:
            raise KeyError(
                f"{target_name!r} not in {name_var} (have: {names})"
            )
        k = names.index(target_name)
        data = np.asarray(ds.variables[data_var][:])
        # diagnostic axis: cijh1 is (nperiod, jm, im, k); the *lh1 groups are
        # (nperiod, jm, im, k, lm).
        if data.ndim == 4:
            sliced = data[:, :, :, k]
        elif data.ndim == 5:
            sliced = data[:, :, :, k, :]
        else:
            raise ValueError(f"Unexpected ndim {data.ndim} for {data_var}")

        if apply_scale and scale_var is not None:
            scale = float(np.asarray(ds.variables[scale_var][:])[k])
            sliced = sliced * scale
    return sliced


def _read_group_field(group, path, name, *, apply_scale=True):
    data_var, name_var, scale_var = group
    return read_named_diagnostic(
        path, data_var, name_var, name,
        scale_var=scale_var, apply_scale=apply_scale,
    )


def to_jcm_column(period_profile: np.ndarray) -> np.ndarray:
    """Transpose a single-period ``(jm, im, lm)`` profile to JCM ``(lm, im, jm)``.

    JCM SPEEDY uses ``(vertical, x, y)`` = ``(kx, ix, il)``. ModelE stores
    ``(y=jm, x=im, vertical=lm)``. This reorders to ``(vertical=lm, x=im, y=jm)``
    while preserving the singleton ``im``/``jm`` axes.

    Args:
        period_profile: Array shaped ``(jm, im, lm)`` (one period).

    Returns:
        Array shaped ``(lm, im, jm)``.
    """
    assert period_profile.ndim == 3, period_profile.shape
    # (jm, im, lm) -> (lm, im, jm)
    return np.transpose(period_profile, (2, 1, 0))


def to_columns(period_profile: np.ndarray) -> np.ndarray:
    """Reshape a single-period ``(jm, im, lm)`` profile to ``(lm, ncols)``.

    This is the column-vectorized order used by the composable physics terms
    (``state.temperature.shape == (nlev, ncols)``), with ``ncols = im * jm``.
    For the SCM oracle ``im == jm == 1`` so ``ncols == 1``.

    Args:
        period_profile: Array shaped ``(jm, im, lm)`` (one period).

    Returns:
        Array shaped ``(lm, ncols)``.
    """
    assert period_profile.ndim == 3, period_profile.shape
    jm, im, lm = period_profile.shape
    # (jm, im, lm) -> (lm, im, jm) -> (lm, ncols)
    return np.transpose(period_profile, (2, 1, 0)).reshape(lm, im * jm)


def read_state_field(
    path: str, name: str, *, period: int | None = None,
    apply_scale: bool = True, jcm_column: bool = False,
) -> np.ndarray:
    """Read a 3D atmospheric state field from ``aijlh1`` by name.

    Args:
        path: NetCDF path.
        name: One of :data:`STATE_FIELDS` (e.g. ``"t"``, ``"q"``, ``"p_3d"``).
        period: If given, select this period index (0..47); otherwise return all.
        apply_scale: Recover physical units (see module docstring table).
        jcm_column: If True (requires ``period``), return ``(lm, im, jm)``
            JCM column order; otherwise the native ModelE order.

    Returns:
        ``(nperiod, jm, im, lm)`` if ``period`` is None, else ``(jm, im, lm)``,
        or ``(lm, im, jm)`` when ``jcm_column`` is set.
    """
    arr = _read_group_field(_GROUP_AIJLH1, path, name, apply_scale=apply_scale)
    if period is None:
        return arr
    pslice = arr[period]
    return to_jcm_column(pslice) if jcm_column else pslice


def read_convection_field(
    path: str, name: str, *, period: int | None = None,
    apply_scale: bool = True, jcm_column: bool = False,
) -> np.ndarray:
    """Read a 3D cloud/convection field from ``cijlh1`` by name.

    Args:
        path: NetCDF path.
        name: One of :data:`CONVECTION_FIELDS` (e.g. ``"dq_mc"``, ``"dth_mc"``).
        period: Optional period index; otherwise all periods.
        apply_scale: Recover physical units.
        jcm_column: If True (requires ``period``), return ``(lm, im, jm)``.

    Returns:
        Same shape convention as :func:`read_state_field`.
    """
    arr = _read_group_field(_GROUP_CIJLH1, path, name, apply_scale=apply_scale)
    if period is None:
        return arr
    pslice = arr[period]
    return to_jcm_column(pslice) if jcm_column else pslice


def read_column_field(
    path: str, name: str, *, period: int | None = None, apply_scale: bool = True,
) -> np.ndarray:
    """Read a 2D column cloud/precip diagnostic from ``cijh1`` by name.

    Args:
        path: NetCDF path.
        name: One of :data:`COLUMN_FIELDS` (e.g. ``"prec"``, ``"mcp"``).
        period: Optional period index; otherwise all periods.
        apply_scale: Recover physical units.

    Returns:
        ``(nperiod, jm, im)`` if ``period`` is None, else ``(jm, im)``.
    """
    arr = _read_group_field(_GROUP_CIJH1, path, name, apply_scale=apply_scale)
    return arr if period is None else arr[period]


@dataclass(frozen=True)
class OracleColumn:
    """A single SCM column / period extracted from the oracle, in JCM order.

    All profile arrays are ``(lm, im, jm) = (vertical, x, y)`` with vertical
    index 0 = surface. ``specific_humidity`` is converted to g/kg for direct
    use in a JCM ``PhysicsState``; ``q_kgkg`` keeps the raw ModelE kg/kg.
    Tendency targets ``dq_mc``/``dth_mc`` are per ModelE units (per day).
    """

    period: int
    t: np.ndarray              # K
    th: np.ndarray             # K (potential temperature)
    q_kgkg: np.ndarray         # kg/kg (ModelE native)
    specific_humidity: np.ndarray  # g/kg (JCM PhysicsState convention)
    z: np.ndarray              # m (geopotential height)
    p_3d: np.ndarray           # hPa
    dq_mc: np.ndarray          # kg/kg/day
    dth_mc: np.ndarray         # K/day
    prec: np.ndarray           # mm/day (scalar column, shape (im, jm))
    mcp: np.ndarray            # mm/day (moist-convective precip)


def read_oracle_column(path: str, period: int) -> OracleColumn:
    """Bundle the fields needed for the convection port for one period.

    Args:
        path: NetCDF path.
        period: Period index (0..47).

    Returns:
        An :class:`OracleColumn` with profiles in JCM ``(lm, im, jm)`` order and
        documented units.
    """
    q_kgkg = read_state_field(path, "q", period=period, jcm_column=True)
    return OracleColumn(
        period=period,
        t=read_state_field(path, "t", period=period, jcm_column=True),
        th=read_state_field(path, "th", period=period, jcm_column=True),
        q_kgkg=q_kgkg,
        specific_humidity=q_kgkg * 1000.0,  # kg/kg -> g/kg for PhysicsState
        z=read_state_field(path, "z", period=period, jcm_column=True),
        p_3d=read_state_field(path, "p_3d", period=period, jcm_column=True),
        dq_mc=read_convection_field(path, "dq_mc", period=period, jcm_column=True),
        dth_mc=read_convection_field(path, "dth_mc", period=period, jcm_column=True),
        prec=read_column_field(path, "prec", period=period),
        mcp=read_column_field(path, "mcp", period=period),
    )
