# GISS ModelE moist convection → JCM: design and porting notes

Status: the scheme runs end to end and produces tendencies, validated in single
column against a ModelE SCM oracle; it is **not** a supported global
configuration. This note documents the conversion of the GISS ModelE
moist-convection routine `MSTCNV` (Fortran) into JCM, the design decisions, and
the physical meaning of each ported piece. It is the narrative companion to the
code under `jcm/physics/convection/giss_*` and the infrastructure under
`jcm/physics/modele/`. See also `jcm/physics/modele/README.md` (layout).

Run it with `python -m jcm.main physics=echam-giss-conv`, which puts the term in
ECHAM's convection slot with `allow_mc: true`. The term's own default is
`allow_mc=False`, a diagnostic-only path that returns zero tendencies.

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

Three ModelE SCM oracles are available, in increasing order of convective
activity, and the validation boundary has moved with them:

| case | character | what it exercises |
|------|-----------|-------------------|
| DYCOMS-II RF02 | stratocumulus | nothing: `dq_mc`, `dth_mc`, `mcp` are identically zero, the plume never triggers |
| BOMEX | shallow cumulus | the shallow plume, one cloud base per step, almost no rain |
| RICO | cumulus with precipitation | the microphysics partition |
| TWP-ICE | tropical deep convection (Darwin, 766 steps) | everything: multiple bases, deep ascent, downdrafts, the ice transition |

TWP-ICE is the case the port is now measured on, because it is the only one that
exercises the deep plume and the phase change. Against it the port currently
produces **0.910** of ModelE's convective rain and delivers **0.923** of its
surface flux, with the cloud-base count per step (1.744 against 1.744) and the
mean cloud top both matching.

Instrumented ModelE dumps, not SUBDD output, are what make this possible. The
`modele-jcm-bridge` repository carries the patches that add them (20 dumps, from
the closure's bisection trace to the per-level plume mass budget and the entry
column) and `BSORT_PORT_PLAN.md` records every measurement against them. Several
conclusions in this document would have been wrong without that: the SUBDD
`mc_*` diagnostics this port was first written against are self-registering and
this SCM configuration never emits them.

**What the oracle cannot settle.** Three of ModelE's inputs are prognostic there
and are read here as optional diagnostics that default to zero when absent — the
cold-pool fraction `fcp`, the penetrative mass `mwpen`, and the running
cloud-top pressure `pmct`. A port validated against TWP-ICE with those supplied
is not yet a GCM-ready scheme, and that gap is a design question rather than a
measurement one.

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
cases. It is now also checked against the oracle: `dmse0` matches ModelE's to
three decimals once ModelE's own `sm`/`qm` are supplied, which is what localised
the remaining cloud-base losses to the environment left by the port's *earlier*
plumes rather than to the trigger itself.

### 3.6 Mass-flux closure — how vigorous
`jcm/physics/convection/giss_mass_flux.py`

Convection acts as a thermostat: it carries up exactly enough mass to remove the
instability. `MASS_FLUX2` finds, by **bisection**, the plume mass fraction
`FPLUME` that restores the cloud base to neutral (`DMSE1 → 0`), including
compensating subsidence of the layers above and a small precip re-evaporation.

`cloud_base_mass_flux` implements the **multi-source (`nlpi > 1`)** blend with
the boundary-layer-top `fpi` zeroing, not just the `nlpi = 1` special case: with
the removal spread over several source levels `DMSE1` is nearly insensitive to
`fplume`, which is what lets ModelE's bisection climb to 0.68 where a
single-source form stops near 0.31. The source parcel also carries ModelE's
`tstar`/`qstar` surface-flux enhancement; the two are synergistic and neither
reproduces the oracle alone.

Magnitude **is** oracle-validated now. Over the count-matched TWP-ICE periods
the port's cloud-base mass is 0.977 of ModelE's in aggregate, with a median of
1.001 per base and exact agreement (0.998-1.001) for every base at level 8 and
above; it errs only at the lowest bases, `lmin` 4 (0.762) and 7 (0.909). The
cold-pool source blend ModelE applies where a downdraft exists is reproduced
bit-for-bit when its four inputs are supplied.

The candidate sweep is `_MAX_PLUME_CANDIDATES = 16` wide, counting down from the
displacement top. ModelE has no such limit — its `cloud_base` loop runs from
`lmcm-1` down to `dcl` — so this is a port artifact, and 16 is where TWP-ICE's
base count stops changing. Do not raise it without re-timing: 8 costs 3.72
s/period, 16 costs 5.41 s, and 24 costs **129.91 s** while changing no result.

### 3.7 Term integration — cloud-base diagnostic from model state
`jcm/physics/convection/giss_mstcnv.py`

`GissConvection.__call__` diagnoses the cloud base from a real `PhysicsState`
and, with `allow_mc=True`, runs the full chain: the closure, the candidate sweep
over cloud bases, the buoyancy-sorting plume ascent, the downdraft descent, and
the tendency assembly. It writes a `GissConvectionData` under the public
`"convection"` diagnostics key and returns temperature and humidity tendencies.

Three integration details:
- **Vertical orientation.** JCM orders the column index 0 = top / last = surface;
  the ported GISS routines are surface-first. The term flips the column at the
  edge and flips the tendencies back.
- **Units.** `PhysicsState.specific_humidity` is **kg/kg** (jax-gcm#824), which
  is also the GISS routines' own unit, so no conversion happens at the edge.
- **Surface flux.** `mcp` is the production minus *both* evaporation sinks, the
  clear-air term and the downdraft term, because ModelE's surface precipitation
  is a falling flux depleted on the way down (`prcpmc = prcpmc + prcp_d +
  prcp_e`). Netting only the clear-air term left the port delivering 95.7% of
  its production against ModelE's 63.3%.

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

## 6. Status

**What runs.** The full chain, producing tendencies: saturation and the energy
ledger, cloud-base/LCL detection, the instability trigger, the multi-source
mass-flux closure with the cold-pool blend and the surface-flux enhancement, the
candidate sweep over cloud bases with the environment carried between them, the
buoyancy-sorting plume ascent (`plume_ent_det_w2_bsort`, the routine
`bsort_entdet` actually selects — an earlier version of this port implemented the
dead one), the microphysics partition, the downdraft descent, and the tendency
assembly. Coverage is 100% on the bsort, mass-flux, microphysics, plume,
tendencies and thermodynamics modules; the convection suite is 639 tests.

**Measured against TWP-ICE** (223 active periods, the harness reconstructing the
entry column as a production port would have to):

| quantity | port / ModelE |
|----------|---------------|
| convective rain produced | **0.910** |
| surface flux delivered | **1.017** (0.960 on the oracle entry column) |
| cloud bases per step | 1.744 / 1.744 |
| mean cloud top | 27.0 levels, bias −0.08 |
| cloud-base mass | 0.977 aggregate, 1.001 median |

**The surface flux is read off the descent, not rebuilt.** ModelE's convective
surface rain is the flux that survives the fall — `prcp_d + prcp_e` as the
descent leaves them (`MSTCNV.F90:5156-5157`) — and it is not reconstructible as
"produced minus evaporated", because the flux is also *fed* by `wmdnl`, the
condensate the buoyancy sort carries into the downdraft, which never enters
`condpr`. ModelE's own BOMEX budget closes only with it: `condpr + wmdnl -
dqevp - dqm_evp` gives 0.9142 mm/day against its reported 0.911, where dropping
`wmdnl` leaves 0.3008. `wmdnl` is 42% of BOMEX's production and 7% of
TWP-ICE's. Carrying the flux out of the descent instead took BOMEX from 0.455
to **0.981** (absolute error 27× smaller) and TWP-ICE from 0.927 to **1.017**,
with the median period moving from 0.80 to 1.00; it is also non-negative by
construction, since each evaporation is capped at the stream it draws from.

**The detrained condensate reaches the cloud scheme.** `MSTCNV` splits what a
plume condenses three ways — the rain it sheds (`condpr`), the condensate the
buoyancy sort carries into the downdraft (`wmdnl`) and the condensate it
detrains into the environment as cloud (`dwm`, accumulated into `detrQC` and
passed to `LSCOND`, ModelE's stratiform scheme, at `MSTCNV.F90:2294`). Measured
from `dwm` in `sweep_state_diag.txt`, the third part is **27%** of BOMEX's
plume condensate and **12.7%** of TWP-ICE's. The port computed it, applied
ModelE's cloud-mode cap to it, and then dropped it: it is condensate, so it is
not in `dq_mc`, which is the vapour budget, and nothing else carried it. The
first global run made that visible — `clouds.conv_detrainment_qc` was
identically zero over all 18,432 columns of a T63L47 step.

The amount was never the problem. Comparing the port's own free-running plume
against `sweep_state_diag.txt`, the *proportions* of the split match ModelE on
both cases:

| branch | BOMEX ModelE / port | TWP-ICE ModelE / port |
|---|---|---|
| `condpr` (rain) | 51.4% / 51.0% | 81.5% / 83.3% |
| `dwm` (detrained) | 26.9% / **26.9%** | 12.4% / **11.3%** |
| `wmdnl` (downdraft) | 21.7% / 22.2% | 6.2% / 5.5% |
| total condensate, port/ModelE | 0.969 | 0.809 |

The per-branch ratios track the total, not each other: on TWP-ICE the port
produces 0.809 of ModelE's condensate and the detrained branch comes out at
0.737, which is that production deficit showing through rather than a second
error in the detrainment — the deficit this section's sibling above decomposes.
The port put the right fraction in the right bucket and then dropped the
bucket.

It is now published as `clouds.conv_detrainment_qc`/`_qi` and as a qc/qi
tracer tendency, which is how `TiedtkeConvection` hands ECHAM's `pxtecl`/
`pxteci` to the same consumer, with the phase read back from the ascent's own
`lhx` rather than from a second temperature test. With the
condensate published the column's water budget closes: vapour removed = surface
rain + condensate detrained, to float32 roundoff (3.9e-6 of 6.86 kg/m² on the
test column). Before it, that residual *was* the whole detrainment.

**A Tiedtke control confirms both the defect and its scale.** Run at the
identical configuration, `physics=echam` writes a non-zero
`clouds.conv_detrainment_qc` (max 4.9e-08) where the pre-fix GISS run wrote
exactly zero in all 18,432 columns — the defect seen beside a term that does it
correctly. The same control also sizes the one global number this run could not
interpret on its own: Tiedtke's convection is active in 4.6% of columns (14.9%
within 20° of the equator) against the GISS term's 0.3% (1.1%), a factor of
~15. It does not follow that the GISS closure under-rains: at the same instant
it delivers *more* convective precipitation than Tiedtke (0.0051 against 0.0014
mm/day area-weighted, Tiedtke raining in no column at all), both schemes being
one day out of an initial state. An instantaneous snapshot one day in is a poor
instrument for either quantity; the firing-rate gap is real and open, and
sizing it needs the 30-day A/B.

**The term is coupled to its neighbours in both directions.** Publishing the
detrainment was the output half; the input half is that `GissConvection` read
`state` for its environment, i.e. the column as the step *started*. ECHAM hands
`cumastr` the provisional `ztp1 = ptm1 + ptte·dt` — the column after vertical
diffusion and the surface exchange have run (`physc`) — and ModelE's ordering is
the same, `MSTCNV` being called from `CLOUDS_DRV` downstream of the boundary
layer. Convection is driven by boundary-layer instability, so reading the
step-start column meant the closure never saw the heat and moisture the surface
supplied on the step it was convecting. The term now reads `thermo_run`, as
`TiedtkeConvection` does, and advances it afterwards, which it had never done,
so the cloud microphysics downstream no longer condenses against the
pre-convection column.

Nothing that validated this port supplies `thermo_run` — not the oracle
harness, not the single column, not the unit tests — so the fallback to the raw
state keeps every SCM measurement on this page exactly as it was. The change is
live only in a composition that seeds it, which is the ECHAM stack.

**The downdraft descent is exact.** Every argument
`giss_downdraft.downdraft_descent` takes is dumped, so ModelE's own shaft can be
replayed through it level by level with the ascent taken out of the question.
Doing that reproduces ModelE's `dqevp` — the precipitation evaporated into the
downdraft, which is 98% of what depletes the falling rain — to **1.0000** on all
52 BOMEX plumes and **1.0002** over 1,031 TWP-ICE downdrafts, 96.3% of them
inside 1%. The bridge's `downdraft_replay_test.py` pins this on the committed
BOMEX oracle. An earlier reading of "0.46x ModelE's on TWP-ICE" compared the
port over the first 60 active periods with ModelE over all 223, and TWP-ICE's
rain falls in the later monsoon burst; on a common period set the port
evaporates 40.9% of its production into the downdraft against ModelE's 44.3%.

**What the remaining production deficit is**, from `BSORT_PORT_PLAN.md` sections
113–128, because its composition matters more than its size:

- about 30% is two timesteps where ModelE convects from a **detached
  single-layer source** high in the column (levels 45 and 50). The branch is six
  lines of Fortran and its inputs are dumped, but it fires above the
  displacement top, which this sweep cannot reach at any affordable width.
  Detached sources are 8 of ModelE's 9,240 converting bases on this case;
- about 30% is the remaining base-count periods, where the **vetoed base is
  nearly irrelevant** — on the six tractable ones it carries 0.8–18.4% of the
  rain and accounts for only 10.8% of their deficit, the rest being the bases
  the port *does* convert;
- roughly a quarter is **not the port**: the harness reconstructs MSTCNV's entry
  column from an end-of-step snapshot and folds some cloud condensate into
  vapour. `run_column(..., entry_column="oracle")` reads the real column
  instead, which halves the per-period scatter and drops the worst period from
  8.45x ModelE's rain to 1.56x;
- most of the rest is the **buoyancy sort's threshold sensitivity**, not a
  defect. Its fates turn on ~1.7e-4 in blend buoyancy and the port reproduces
  that quantity to ~2e-4, so a single blend differing by 0.07 K flips between
  rejoining the updraft and detraining. A plume given ModelE's input, cloud base
  and top reproduces ModelE's rain to **1%**; per-period agreement is therefore
  the wrong acceptance test for this scheme and aggregate agreement is the right
  one.

**Accepted limitations**, each with its measured size:

1. **Ice microphysics.** The *condensation* phase is chosen per level from the
   parcel's temperature against `tfmc` = 258.16 K, as ModelE does, and that was
   worth 14.3% of the deficit and 1.5 levels of cloud top. But `PRECIPICE_GAMMA`
   and the graupel blend are unported, so the *precipitation* phase is
   all-liquid, as is the melting/freezing heat redistribution. Measured at
   **1.1%** of the deficit — and that number is specific to a case where only
   12% of ModelE's rain falls at ice levels, so it must be re-sized before this
   term is trusted outside the tropics. See `MIXED_PHASE_SUPPORTED`.
2. **The terminating-level rain partition.** The port rains 0.650 of ModelE's
   condensate at the level a plume dies on and over-rains by 3.1% below, so the
   column total is 0.990 — a redistribution rather than a loss, worth ~1%. Its
   effect on the *downdraft* profile is unmeasured, and that coupling is a known
   runaway.
3. **Cross-plume phase latching.** ModelE shares `TPSAV`/`VLAT` across every
   plume of a timestep; the port decides per plume. Measured inert here: the two
   disagree on 3 of 3,772 shared `(step, level)` pairs, none on an active period.

**Not yet done.**

1. A **validated global run**. The term now *runs* globally: `+configuration=
   t63-echam-1m physics=echam-giss-conv run=smoke` completes a day at T63L47
   with RRTMGP, every output field finite, under the runner's health gates
   (`bail_on_unhealthy`) and without tripping them. Convection is active across
   the grid — a cloud base in nearly every column, `cloud_base_mass_flux` to
   149 kg/m², `mcp` to 13.8 mm/day. That is a smoke test and nothing more: the
   global magnitudes have no oracle behind them, and the A/B against
   `physics=echam` that would size the term's effect on the simulated climate
   has not been run. It also took 1 h 54 m on this CPU host, essentially all of
   it XLA compilation, so the A/B needs a GPU.

   The single-column half of the coupling is more than a smoke test: the preset's own term
   list, built through Hydra with the grey two-stream standing in for RRTMGP,
   integrated as a radiative-convective column through the same
   `compute_tendencies` path the model uses. Over 10 days (720 steps) it stays
   finite and takes the column to a surface temperature of 303.61 K against the
   stock `physics=echam` preset's 303.59 K, with the same drift — so the term is
   not injecting or removing heat the rest of the stack has to absorb, and the
   downdraft runaway `giss_microphysics` warns about does not appear through the
   coupling. `giss_rce_integration_test.py` holds that as a slow test. The
   column stabilises after spin-up, though, so convection is only exercised
   while it settles. There is no reason to expect the tropical-case tunings to
   transfer, and the three diagnostics it reads optionally (`fcp`, `mwpen`,
   `pmct`) are zero in a configuration that does not supply them. What *is*
   covered is the term's own shape handling: the `allow_mc` chain runs on a
   `(nlev, nlon, nlat)` grid and every column matches the same column run alone
   to 1e-5, with four distinct cloud bases and two non-convecting columns among
   six, so a rank assumption or a mis-broadcast level index would fail there.
2. A decision on whether the port carries a **cold-pool state of its own** or
   stays a diagnostic-fed comparison. That is the blocker on the cold-pool
   source blend, whose own effect measured 3.2% and was accepted.
