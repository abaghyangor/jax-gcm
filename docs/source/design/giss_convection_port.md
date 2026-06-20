# GISS ModelE moist convection → JCM: design and porting notes

Status: in progress. This note documents the conversion of the GISS ModelE
moist-convection routine `MSTCNV` (Fortran) into JCM, the design decisions, and
the physical meaning of each ported piece. It is the narrative companion to the
code under `jcm/physics/convection/giss_*` and the infrastructure under
`jcm/physics/modele/`. See also `jcm/physics/modele/README.md` (layout) and
`STATUS.md` (current state).

## 1. Goal and method

Convert one parameterization — GISS moist convection (`MSTCNV`,
`modelE/model/MSTCNV.F90`, ~8,700 lines) — from Fortran into the differentiable
JAX model JCM. Faithfulness is checked against a **Fortran oracle**: outputs
saved from a verified ModelE single-column (SCM) run, used as an answer key.

`MSTCNV` is a **two-plume mass-flux scheme** that represents both shallow
(cumulus) and deep convection — confirmed in source: it tracks `SHALLOWMC`
(shallow) and `DEEPMC` (deep) cloud cover and separate shallow/deep cloud
profiles, driven by two plumes with different entrainment. This is why
boundary-layer cloud field campaigns are the relevant test cases.

### The oracle and the validation boundary

The available oracle is **DYCOMS-II RF02**, a *stratocumulus* case. Its
convective diagnostics (`dq_mc`, `dth_mc`, `mcp`) are **identically zero**: the
plume scheme never triggers (stratocumulus is handled by large-scale
condensation, not convective plumes). Consequences:

- **Pointwise physics** (saturation, energies, buoyancy, cloud-base level) is
  validated by physical reasoning and unit tests — independent of the oracle.
- **Magnitudes** (mass flux, and eventually tendencies) cannot be validated on
  DYCOMS. They need a *convectively active* case — **BOMEX** (shallow cumulus),
  then **RICO** (cumulus with precipitation) — which specifically exercise the
  shallow-convection plumes.

This boundary is the single most important thing to keep in mind when reading
the code: every port is marked as "validated by reasoning" or "magnitude pending
an active oracle".

## 2. Repository layout (separation of concerns)

Two kinds of code were separated:

- **`modele-jcm-bridge`** (separate repo): ModelE-specific *data tooling* — the
  NetCDF oracle reader and the oracle→`PhysicsState` adapter. JCM should not know
  ModelE's file formats.
- **`jax-gcm`** (this repo): the convection *scheme* — `jcm/physics/convection/giss_*`
  — plus its parameter/diagnostic structs under `jcm/physics/modele/`, and small
  committed fixture arrays under `jcm/data/test/modele/` (the SPEEDY-style `.npy`
  pattern). The bridge generates those fixtures; JCM consumes them, so JCM has no
  dependency on ModelE I/O.

## 3. The physics, as the steps of a rising parcel

The ported pieces are, in order, the logical steps of moist convection. Units
are ModelE-native (T in K, pressure in **Pa**, specific/condensate humidity in
kg/kg, geopotential in m²/s²); conversion to JCM conventions happens at the term
boundary.

### 3.1 Saturation — how much water the air can hold
`jcm/physics/convection/giss_thermodynamics.py`

Warm air holds more vapor than cold air (Clausius-Clapeyron, ~7 %/K). The
ceiling is the saturation vapour pressure `es(T)`; condensation begins when a
cooling parcel reaches it.

- `saturation_vapor_pressure` — **Murphy & Koop (2005)**, the `use_mk2005=.true.`
  branch of ModelE `wv_psat` (`shared/Utilities.F90`), which the oracle run used.
  Temperature is clipped to the fit's validity range (water ≤ 332 K) exactly as
  the Fortran does — a faithful limitation, covered by a dedicated test.
- `saturation_specific_humidity` — ModelE `QSAT = mrat · es / p` (mrat ≈ 0.622).

Constants are ModelE's own values (e.g. `sha = rgas/kapa ≈ 1002.9`), derived as
in `Constants_mod.F90` and defined locally, because matching the oracle requires
ModelE's numbers — mirroring how the SPEEDY port keeps its own `alhc`.

### 3.2 Moist static energy — the parcel's energy ledger

`MSE = cp·T + g·z + L·q` (sensible + potential + latent). Conserved as a parcel
rises, even through condensation (latent → sensible). The currency for buoyancy
and instability. `moist_static_energy(T, geopotential, q)`.

### 3.3 Virtual temperature — buoyancy

Buoyancy is about density. Vapour lightens air; suspended condensate loads it
down: `Tv = T·(1 + DELTX·q − w)`, with `DELTX = M_air/M_water − 1 ≈ 0.6078`
(ModelE `deltx`). `virtual_temperature(T, q, condensate)`. Applying the same
factor to potential temperature gives virtual potential temperature, used by the
trigger.

`d_ln_qsat_dt` (= `L/(RVAP·T²)`, ModelE `DLNQSATDT`) is also ported — the
saturation-curve slope the mass-flux closure needs. (Elsewhere JAX autodiff
supplies `d(qsat)/dT`; this analytic form exists to match the closure's Fortran
arithmetic, which uses it even with the MK2005 `es`.)

### 3.4 Cloud base / LCL — where cloud forms
`jcm/physics/convection/giss_cloud_base.py`

A humid surface parcel lifted dry-adiabatically (`T ∝ p^κ`) cools until it
saturates — the Lifting Condensation Level, the flat cumulus base. The Fortran
scans upward and `exit`s at the first saturated level. `lifting_condensation_level`
reproduces this; see the JAX pattern in §4.

### 3.5 Instability trigger — does it actually convect?

A cloud base is necessary but not sufficient: convection fires only if the moist
parcel is buoyant relative to the air above. ModelE's criterion:

```
DMSE = (SVUP − SVDN)·PLK + (L/cp)·(qsat_above − q_parcel)
```

where `SVUP`/`SVDN` are virtual *potential* temperatures (since `SM = TH·MA`, the
state variable is potential temperature) and `PLK` is the Exner function (so
`SUP·PLK` is a temperature). `DMSE < 0` ⇒ unstable, **gated** by the parcel being
saturated at the interface. `cloud_base_instability` / `cloud_base_triggers`.

The sign convention was verified by reading the source (`SM=TH*MA`, `PLK=PL^KAPA`)
and corroborated by worked unstable / stable-but-saturated / subsaturated test
cases. Structure is faithful; absolute values remain pending an active oracle.

### 3.6 Mass-flux closure — how vigorous
`jcm/physics/convection/giss_mass_flux.py`

Convection acts as a thermostat: it carries up exactly enough mass to remove the
instability. `MASS_FLUX2` finds, by **bisection**, the plume mass fraction
`FPLUME` that restores the cloud base to neutral (`DMSE1 → 0`), including
compensating subsidence of the layers above and a small precip re-evaporation.

`cloud_base_mass_flux` is the **single-source-level (`nlpi=1`)** port (the common
case; the multi-source blend is deferred). Magnitude is **not yet
oracle-validated**: tests assert closure *behaviour* (the bisection drives
`DMSE1` toward neutral; a more unstable column needs more flux), not Fortran
agreement.

### 3.7 Term integration — cloud-base diagnostic from model state
`jcm/physics/convection/giss_mstcnv.py`

`GissConvection.__call__` now diagnoses the cloud base from a real
`PhysicsState` when a `pressure_full` (Pa) diagnostic is present (consumed from
the diagnostics dict, as tiedtke/betts_miller do). It writes
`diagnostics["convection"].cloud_base`. **Tendencies remain zero** — converting
the mass-flux closure into temperature/humidity tendencies is the next step and
needs an active oracle to validate.

Two integration details:
- **Vertical orientation.** JCM orders the column index 0 = top / last = surface;
  the ported GISS routines are surface-first. The term flips the column before
  the lift (documented in `_diagnose_cloud_base`).
- **Units.** `PhysicsState.specific_humidity` is g/kg; converted to kg/kg at the
  edge.

## 4. Fortran → JAX translation patterns

Three recurring transforms make stateful Fortran into pure, differentiable,
shape-static JAX:

1. **Scan-and-`exit` → mask-and-`argmax`.** A loop that exits at the first level
   satisfying a condition (cloud base) becomes a boolean mask plus
   `jnp.argmax(mask, axis=0)` (first True) and `jnp.any` (any True). Same device
   SPEEDY's trigger uses.
2. **Data-dependent `if` → `jnp.where`.** The bisection's
   `if(DMSE>0) FPLUME-=DFP` becomes `jnp.where(...)`, preserving
   differentiability. An early `exit` on convergence is implicit: inside the
   tolerance band neither branch fires.
3. **Fixed loops → unrolled.** The 9-iteration bisection has a static count and
   is a plain Python `for`, baked into the compiled graph.

A consequence of (1)–(3): the code is **broadcasting-native** (vertical on axis
0, horizontal axes broadcast), so identical functions run on a single column
`(nlev,)` or a vectorized block `(nlev, ncols)` — verified by per-column equality
tests.

## 5. Differentiability

The continuous physics (saturation, energies, virtual temperature, the
`jnp.where` bisection) is differentiable — the purpose of the JAX port; gradient
tests (`jax.grad`, `check_vjp`/`check_jvp`) cover it. Convective **triggering** is
genuinely discontinuous: the `argmax` cloud-base index and the unstable/stable
boolean are step functions. Gradients flow through the thermodynamic quantities,
not through these integer/boolean switches; the discontinuities are documented at
each site.

## 6. Status and next steps

Ported and tested (≈46 unit tests): saturation, moist static energy, virtual
temperature, `d_ln_qsat_dt`; cloud-base/LCL detection; the instability trigger
(`DMSE`); the single-source mass-flux closure; and the term-level cloud-base
diagnostic. The term computes a real diagnostic from state but **produces no
tendencies yet**.

Remaining:
1. Convert the mass-flux closure into temperature/humidity tendencies and wire it
   into the term (needs the dynamic cloud-base level selection).
2. Port the plume above cloud base (entrainment, condensation, updraft velocity,
   downdrafts, precipitation).
3. Generalize the mass-flux closure to multiple source levels (`nlpi > 1`).
4. Obtain BOMEX/RICO oracles (NASA), regenerate nonzero fixture targets via the
   bridge repo, and validate magnitudes.
