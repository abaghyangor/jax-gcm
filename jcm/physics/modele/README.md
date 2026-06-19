# ModelE convection port (`jcm.physics.modele` + `convection/giss_mstcnv.py`)

This converts the GISS ModelE moist convection routine `MSTCNV`
(`modelE/model/MSTCNV.F90`, called from `CONDSE_column` in
`modelE/model/CLOUDS_DRV.F90`) into the JCM / JAX-GCM codebase, using the
verified local **ModelE DYCOMS-II RF02 single-column run** as a Fortran oracle.

## Layout (follows the repo's by-process organization)

The convection **term** is a composable `PhysicsTerm` and lives under
`jcm/physics/convection/` (named after the scheme). The ModelE-specific
**infrastructure** lives here under `jcm/physics/modele/`, mirroring how
`speedy/` and `icon/` hold their own params/coords/data:

| File | Purpose |
|------|---------|
| `convection/giss_mstcnv.py` | **`GissConvection(PhysicsTerm)`** — scaffold, returns zeros. |
| `modele/oracle.py` | Read named diagnostics from the packed ModelE sub-daily NetCDF. |
| `modele/adapter.py` | Build a column `PhysicsState` from an oracle period. |
| `modele/params.py` | `GissConvectionParameters` PyTree (`dtsrc`). |
| `modele/physics_data.py` | `GissConvectionData` diagnostics PyTree (`dq_mc`, `dth_mc`, `mcp`). |
| `convection/giss_mstcnv_test.py` | Term interface / composition / differentiability / oracle-zero-MC tests. |
| `modele/oracle_test.py` | Extraction / shape / unit / orientation tests. |
| `../../data/test/modele/dycoms_oracle_subset.nc` | Self-contained fixture extracted from the verified oracle. |

**`GissConvection` is a scaffold. It returns zero tendencies and is not a
validated ModelE port.** See the "Important finding" below for why zero output
*happens* to agree with this oracle — and why that agreement is trivial.

## Interface

`GissConvection` is a `flax.nnx.Module` subclass of `PhysicsTerm` and composes
via `ComposablePhysics` (the only physics API on the current branch):

```python
GissConvection().__call__(state, diagnostics, forcing, terrain)
    -> (PhysicsTendency, diagnostics)   # writes diagnostics["convection"]
```

State is column-vectorized `(nlev, ncols)`; `GissConvectionData` is written
under the public `"convection"` diagnostics key, matching `tiedtke_nordeng` and
`speedy_convection`.

## Environment note

The current branch requires `dinosaur >= 1.3.6` (it imports
`compute_diagnostic_state_hybrid`). The ECHAM/RRTMGP radiation path also needs
the `jax_solar` package; if it is not installed those radiation/ECHAM tests
error on import, but the ModelE/GISS tests here do not depend on it.

## Oracle data conventions (verified)

The oracle is the verified ModelE run output (a sub-daily NetCDF such as
`<ModelE_Support>/huge_space/dycoms_scm/allsteps.subdddycoms_scm.nc`, an external
machine-specific path set via the `MODELE_DYCOMS_ORACLE` env var; a committed
subset is used by default — see `oracle.fixture_path()`). It has
48 sub-daily periods, single column `im=jm=1`, `lm=63`. ModelE stores groups
of diagnostics as a packed array plus a `sname_*` name table and a `scale_*`
factor:

* `aijlh1` (3D state) — names `sname_aijlh1`: `u v t th q rhw_rhi rhw z p_3d`
* `cijlh1` (3D cloud/convection) — names `sname_cijlh1`: `qcl qci cldss cldmc cfr dq_mc dth_mc dq_ss dth_ss`
* `cijh1` (2D column) — names `sname_cijh1`: `prec mcp ssp cldmc_2d cldss_2d cldtot_2d lwp iwp tau_ss tau_mc`

Physical value = raw × `scale_*[k]` (when `denom_*[k] == 0`, as here).

### Vertical orientation

**Vertical index 0 is the surface / lowest layer** (`p_3d[0] ≈ 1013 hPa`),
index 62 is model top (`p_3d[62] ≈ 0.14 hPa`), matching ModelE `L=1` = lowest
layer. Profiles decrease in pressure with increasing index.

### Dimension order

* Native ModelE per-period slice: `(jm, im, lm) = (y, x, vertical)`.
* `jcm_column=True` transposes one period to `(lm, im, jm) = (vertical, x, y)`,
  matching the SPEEDY `(kx, ix, il)` test convention, preserving singleton axes.

### Units (after `apply_scale=True`)

| name | units | notes |
|------|-------|-------|
| `t` | K | absolute temperature |
| `th` | K | potential temperature (scale 7.22) |
| `q` | kg/kg | ModelE native — **multiply by 1000 for JCM g/kg** |
| `z` | m | geopotential height (scale 1/g) |
| `p_3d` | hPa | layer pressure |
| `dq_mc` | kg/kg/day | moist-convective q tendency |
| `dth_mc` | K/day | moist-convective θ tendency |
| `prec`, `mcp`, `ssp` | mm/day | total / moist-conv / stratiform precip |

`oracle.read_oracle_column(path, period)` bundles `t, th, q (kg/kg & g/kg),
z, p_3d, dq_mc, dth_mc, prec, mcp` in JCM column order with these units.

## Important finding: moist convection is inactive in this DYCOMS run

Across **all 48 periods**, `dq_mc`, `dth_mc`, and `mcp` are **identically zero**.
All precipitation is stratiform (`ssp == prec`, both nonzero). DYCOMS-II RF02 is
a stratocumulus case and this rundeck runs with moist convection effectively
off.

Consequences:

* The zero-returning scaffold trivially "matches" the oracle's MC tendencies.
  **This is not scientific validation** — it is a no-convection consistency
  check, and the tests label it as such.
* A faithful `MSTCNV` port **cannot be validated for nonzero MC behaviour
  against this oracle.** To exercise real convection you need either a
  convectively active SCM case (e.g. `SCM_BOMEX.R`, `SCM_RICO.R`) or DYCOMS
  rerun with `SCMopt%allowMC` enabled.

## Next steps ("Option B": DYCOMS/SCM subset of `MSTCNV`)

1. Choose a convectively active oracle (BOMEX/RICO, or DYCOMS with `allowMC`),
   regenerate sub-daily output, and extract fixtures the same way.
2. Port `CLOUD_BASE` / `cloud_base_closure` triggering + cloud-base mass flux.
3. Port the plume lift/condensation loop (`CLOUD_TOP`, `plume_ent_det_w2`), then
   downdrafts, subsidence, `EVAP_PRECIP`.
4. Fill `ModelEConvectionData.dq_mc/dth_mc/mcp` and the `PhysicsTendency`; start
   with **process-level / diagnostic** agreement, tighten tolerances later.
5. Track every omitted ModelE branch in `convection.py`. Preserve JAX
   differentiability with `jnp.where`/`jax.lax.cond` and document the
   triggering discontinuities.

## Running the tests

```sh
# from the jax-gcm repo root
JAX_PLATFORMS=cpu python -m pytest \
    jcm/physics/modele jcm/physics/convection/giss_mstcnv_test.py -q
```

## Regenerating the fixture

The committed fixture is a subset of the verified oracle. To regenerate it from
a fresh ModelE run, copy the `aijlh1/cijlh1/cijh1` packed arrays plus their
`sname_*`/`scale_*`/`denom_*` tables into
`jcm/data/test/modele/dycoms_oracle_subset.nc` (see git history of this package
for the extraction script). Do **not** overwrite the verified oracle output.
