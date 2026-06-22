# GISS ModelE convection in JCM

This is the JCM-side home of the GISS ModelE moist-convection port. The Fortran
source is `MSTCNV` (`modelE/model/MSTCNV.F90`, a two-plume mass-flux scheme that
represents both **shallow (cumulus)** and **deep** convection), called from
`CONDSE_column` in `modelE/model/CLOUDS_DRV.F90`. We convert it to JCM using a
verified ModelE single-column run as a Fortran oracle.

## What lives where

The convection **scheme/term** is in JCM, organized by process; the ModelE
**reading/conversion tooling** is kept in a *separate repository*.

| Piece | Location |
|------|----------|
| Convection term `GissConvection(PhysicsTerm)` | `jcm/physics/convection/giss_mstcnv.py` |
| Scheme parameters `GissConvectionParameters` | `jcm/physics/modele/params.py` |
| Scheme diagnostics `GissConvectionData` | `jcm/physics/modele/physics_data.py` |
| Committed test fixture (one DYCOMS column + MC targets) | `jcm/data/test/modele/dycoms_period24.npz` |
| Scheme tests | `jcm/physics/convection/giss_mstcnv_test.py` |
| **ModelE oracle reader + `PhysicsState` adapter** | **separate ModelE-bridge repo** (not in JCM) |

The bridge repo reads ModelE's packed sub-daily NetCDF output and emits the
committed fixture arrays used here — mirroring how the SPEEDY convection tests
consume committed `.npy` oracle arrays rather than reading raw model output.

## Status: Milestone 1 scaffold

`GissConvection` returns **zero** tendencies and zero diagnostics. It is **not**
a validated ModelE convection implementation.

**Important — why "matches DYCOMS" is trivial here:** moist convection is
*inactive* in the DYCOMS-II RF02 case (`dq_mc`/`dth_mc`/`mcp` are identically
zero across all 48 periods — DYCOMS is stratocumulus, handled by large-scale
condensation, not the convective plumes). So zero output matching DYCOMS is not
evidence of correct physics. Validating real `MSTCNV` physics requires a
convectively active case — **BOMEX** (shallow cumulus), then **RICO** (cumulus
with precipitation) — which specifically exercise the shallow-convection plumes.

## Conventions

* State is column-vectorized `(nlev, ncols)`, vertical index 0 = surface.
* `PhysicsState.specific_humidity` is g/kg (ModelE oracle `q` is kg/kg).
* `PhysicsTendency` fields are per second.

## Running the tests

```sh
# from the jax-gcm repo root
JAX_PLATFORMS=cpu python -m pytest \
    jcm/physics/modele jcm/physics/convection/giss_mstcnv_test.py -q
```

## Next steps

1. ~~Port the saturation / moist-static-energy thermodynamics~~ **done** —
   `jcm/physics/convection/giss_thermodynamics.py` (Murphy & Koop 2005 saturation,
   ModelE `QSAT`, moist static energy), with standalone + gradient + broadcasting
   tests in the style of the other JCM convection schemes.
2. ~~Port the convective trigger / cloud base (LCL detection)~~ **done** —
   `jcm/physics/convection/giss_cloud_base.py` (dry-adiabatic parcel lift +
   first-saturated-level cloud base), with standalone + broadcasting tests.
3. ~~Cloud-base instability criterion (`DMSE`) + saturation gate~~ **done** —
   `cloud_base_instability` / `cloud_base_triggers` in `giss_cloud_base.py`
   (structure/sign confirmed from source; not yet oracle-validated).
4. ~~Cloud-base mass flux (`MASS_FLUX2`, single-source-level)~~ **done** —
   `giss_mass_flux.py` (bisection to neutral `DMSE1`, with layer redistribution
   + precip re-evaporation). First nonzero-rate piece; magnitude not yet
   oracle-validated; multi-source (`nlpi>1`) deferred.
5. ~~Wire cloud-base diagnosis into `GissConvection.__call__`~~ **done** (reads
   `pressure_full`; still zero tendencies).
6. **Validated against active convection (BOMEX):** the cloud-base/LCL detection
   reproduces ModelE's actual convective cloud base to within one model level
   across all 48 BOMEX periods (38/48 exact). The BOMEX oracle and this numerical
   validation live in the private `modele-jcm-bridge` repo (real NASA-derived
   oracle data).
7. ~~Plume above cloud base — moist-adiabatic ascent~~ **done** —
   `giss_plume.py` (undilute saturated ascent + condensation).
8. ~~Entraining updraft — Gregory (2001) cumulus velocity + buoyancy-sorting
   entrainment~~ **done** — `giss_plume.py` (`entrainment_rate`,
   `updraft_velocity`): integrate `w²` up, entrainment as drag, cloud top where
   `w² ≤ 0`. Detrainment is an input (its closure is a separate port).
9. ~~Couple entrainment into the ascent~~ **done** — `giss_plume.py`
   `entraining_plume_ascent` (+ `_saturation_adjust`): a self-consistent
   single-plume march where the plume is carried as moist static energy + total
   water, saturation-adjusted each level, its buoyancy drives the updraft and the
   entrainment that then dilutes it — buoyancy → updraft → entrainment feedback,
   physically-determined cloud top. (No detrainment/precip yet.)
10. The compensating **subsidence + detrainment** that turn the plume mass flux
    into the environmental tendencies, to reproduce the `dq_mc`/`dth_mc`
    *magnitudes* the BOMEX fixture holds. Consider rerunning BOMEX with
    `SCM_PlumeDiag=1` for per-level plume diagnostics to validate the plume
    internals (parcel T, updraft velocity, cloud top) directly.
11. Generate the RICO oracle (adds convective precipitation) the same way.
