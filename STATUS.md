# Status: ModelE → JCM Convection Conversion

**Last updated:** 2026-09-06
**Branch:** `modele-convection-port` (based on `upstream/dev`), 698 commits.
**Companion repo:** `modele-jcm-bridge` (ModelE oracle reading + state conversion).

The long-form engineering record, including every wrong turn and its correction,
is `BSORT_PORT_PLAN.md` (70 sections). This file is the summary.

---

## One-paragraph summary

The goal is to convert the GISS ModelE moist-convection routine `MSTCNV`
(~8,800 lines of Fortran; a mass-flux scheme with buoyancy-sorted entraining
plumes, downdrafts and convective microphysics) into a differentiable JAX
`PhysicsTerm` in JCM, using a verified ModelE single-column run as the oracle.
**The physics is ported and validated.** Driven from an identical column state,
the JAX scheme reproduces ModelE's convective heating with median correlation
**+0.977 on BOMEX and +0.988 on RICO** across all 48 periods of each, finds the
same cloud base ModelE does on **94 of 94** convecting periods, and correctly
**declines to convect** on all 48 DYCOMS periods and both quiet RICO periods.
Gradients through the whole chain match float64 finite differences to
**1.0000000x**. What remains is not the core scheme: it is mixed phase (needs a
deep-convection oracle), the second closure, cold pools, and running the term
inside a free-running JCM integration.

**Repository layout (two repos):**

- **jax-gcm** (this repo, branch `modele-convection-port`): the convection
  scheme, `jcm/physics/convection/giss_*.py`, plus parameter/diagnostic structs
  under `jcm/physics/modele/` and committed fixtures under
  `jcm/data/test/modele/`.
- **modele-jcm-bridge** (separate repo): the ModelE oracle *reading*, the
  like-for-like single-column harness, and the figure generation. Kept separate
  so JCM has no NetCDF or ModelE dependency.

---

## What is ported

Roughly 3,500 lines of JAX across nine modules, with 247 tests.

| Module | Lines | What it is |
| --- | --- | --- |
| `giss_thermodynamics.py` | 345 | Murphy & Koop saturation vapour pressure, ModelE `QSAT`, moist static energy, virtual temperature with condensate loading |
| `giss_cloud_base.py` | 195 | Parcel-lift cloud base (LCL) and the cloud-base instability criterion |
| `giss_mass_flux.py` | 562 | The `MASS_FLUX2` closure: the bisection on `dmse1`, and the four vetoes that decide whether convection happens at all |
| `giss_plume.py` | 612 | Entraining/detraining plume ascent |
| `giss_bsort.py` | 932 | Buoyancy sorting: the blend spectrum, its fate distribution, and the reference/full-mix blend weighting |
| `giss_plume_driver.py` | 755 | The plume loop, the displacement top (`dp_disp`) that selects the cloud base, and detrainment deposition |
| `giss_downdraft.py` | 381 | Downdraft shaft, entrainment, evaporation, and the three-branch detrainment |
| `giss_microphysics.py` | 239 | Condensate-to-precipitation conversion |
| `giss_tendencies.py` | 327 | Compensating subsidence and the assembly of `dth_mc` / `dq_mc` |
| `giss_mstcnv.py` | 716 | The `PhysicsTerm` itself: state marshalling, unit conventions, composition |

All of it is `jax.jit`-compatible and differentiable end to end. There are no
Python-level branches on traced values; ModelE's `if`/`return` control flow is
expressed as `jnp.where` masks throughout.

---

## Validation

### Method

The bridge runs a **like-for-like single-column comparison**: the *same* column
state is fed to both models and the resulting convective tendency is compared.
Getting this honest took most of the effort, and three harness bugs — not port
bugs — accounted for nearly all of the disagreement seen through August:

1. **Pre-convection reconstruction.** The oracle snapshot is taken *after* both
   convection and large-scale condensation, so recovering ModelE's input state
   requires undoing `dth_mc + dth_ss` and `dq_mc + dq_ss`, not just the `mc`
   halves. Asserted per level now, because a column median hid an error that
   changed sign with height.
2. **Surface pressure.** `prsurf`, not the lowest layer midpoint `p_3d[0]`,
   which halved the bottom layer.
3. **Layer edges.** ModelE's `pl(l) = ½(pedn(l) + pedn(l+1))` inverted as a
   recursion from the surface. Treating midpoints as edges is exact only where
   the spacing is uniform; it was wrong by 2.5% at level 8.

Metrics are reported as **median and worst decile**, never median alone. A
median has twice pointed the wrong way here.

### Results

Three SCM cases, 48 periods each, port free-running from the column state:

| `dth_mc`, whole column | BOMEX | RICO | DYCOMS-II RF02 |
| --- | --- | --- | --- |
| what the case is | shallow trade cumulus | precipitating shallow cumulus | stratocumulus |
| ModelE convects | 48/48 periods | 46/48 | **0/48** |
| correlation, median / worst decile | +0.977 / +0.887 | **+0.988 / +0.938** | — |
| nRMSE, median / worst decile | 0.037 / 0.089 | **0.034 / 0.067** | — |
| peak ratio, median | 1.009 | 0.956 | — |
| peak level offset, median | 0 | 0 | — |
| sign mismatch, median | 0.000 | 0.000 | — |
| closure base vs ModelE `lmin` | **48/48 exact** | **46/46 exact** | — |
| port declines where ModelE declines | (none to test) | **2/2** | **48/48** |

`dq_mc` correlates +0.988 median / +0.951 worst decile on BOMEX and
+0.986 / +0.967 on RICO.

**RICO is the load-bearing result.** BOMEX was the case every constant was
checked against, so BOMEX agreement alone cannot distinguish correct physics
from code shaped around one case. RICO was added in a single session — a
rundeck, one ModelE run, no change to the port — and scores *better* than BOMEX
in the tail on every heating metric.

### Differentiability

`giss_mstcnv_test.TestGissConvectionGradient` compares reverse-mode gradients of
a convective-heating loss against central finite differences in float64:
ratios **1.0000000x**. The earlier float32 probe gave ratios of 0.71 and 3.04
and looked like a gradient bug; it was resolution, not correctness.

---

## What the three cases can and cannot show

This matters more than the numbers, because two of the last three defects found
were things BOMEX is structurally incapable of showing.

- **BOMEX and RICO are both all-liquid** (minimum parcel temperature 281 K and
  280 K against ModelE's `tfmc = 258 K`) and **both top out at level 19, about
  3.4 km**. RICO is a second *independent* case, not a *deeper* one.
- **No available case exercises the ice phase**, so the mixed-phase path is
  deliberately not written (see below).
- **DYCOMS is a negative control only.** Its convective tendencies are
  identically zero; it validates the vetoes and nothing else. The DYCOMS test
  that existed before September was near-vacuous — it passed `{}` for
  diagnostics, degrading to the scaffold path that returns zeros
  unconditionally, so it asserted zeros against zeros.
- **A case where the answer is always "yes" cannot test declining.** Three
  closure vetoes were missing until DYCOMS was run through the active path, at
  which point the port convected in 48 periods out of 48.

---

## What is not done

- **Mixed phase.** ModelE's phase criterion depends on the *parcel* temperature
  during ascent and latches through `VLAT(L)`, so it is path-dependent state
  carried through the scan; there is a second phase field `lhp(l)` for
  precipitation and a melting/freezing heat redistribution between them
  (`MSTCNV.F90:1716-1721`, `:4295-4302`). Sequence this with **TWP-ICE**, whose
  ARM forcing is already on disk (`scm_inputs/twp10mb180iopsndgvarana_*.cdf`) and
  which has a ModelE template deck. Writing it before there is a case that can
  check it would be writing physics that cannot be validated.
- **The second closure (`closure2`)** and **cold pools** — both off in the
  preset this port targets.
- **Multi-plume.** The port runs the single plume the BOMEX/RICO preset selects.
- **Free-running integration.** The term has been validated per timestep against
  the oracle but not yet run inside a multi-step JCM integration, where errors
  accumulate rather than being teacher-forced.
- `allow_mc` remains **default off**, mirroring ModelE's own `SCMopt%allowMC`.

---

## Reproducing the results

```sh
cd jax-gcm
JAX_PLATFORMS=cpu ./.venv/bin/python -m pytest jcm/physics/convection -q
# expected: 388 passed, 3 skipped
```

The comparison figures and scorecards (needs the ModelE run directories, which
are not committed):

```sh
cd modele-jcm-bridge
SCM_CASE=rico python -m modele_jcm_bridge.bomex_compare_plots out/
```

`SCM_CASE` takes `bomex`, `rico` or `dycoms`. Adding a case is two lines in the
`CASES` table plus a ModelE rundeck; nothing in the harness or the scoring is
case-specific, which is the point.

The ModelE side, if the oracle needs regenerating:

```sh
cd modelE/decks
MODELERC=$PWD/modelErc.bomex_scm.local gmake setup RUN=rico_scm
MODELERC=$PWD/modelErc.bomex_scm.local ../exec/runE rico_scm -cold-restart -np 1
```

`SCM_PlumeDiag=1` in the rundeck is what writes the plume-internal dumps
(`mass_budget.txt`, `blend_diag.txt`, `downdraft_diag.txt` and five others).
`gmake setup` regenerates the run directory's `I` from the rundeck, so setting
it to 0 there silently disables every dump on the next rebuild.

---

## Environment notes

- `dinosaur >= 1.3.6` (dev imports `compute_diagnostic_state_hybrid`).
- The radiation stack from `requirements.txt`: `jax-solar`, `jax-rrtmgp >= 0.2.0`.
- ModelE builds with `gfortran`, `MPI=NO`, NetCDF from Homebrew; see
  `modelE/decks/modelErc.bomex_scm.local`.
