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
Both figures predate the terminal-dump fix, which raises the delivered flux
again — see "Landing the dump correctly costs surface rain on TWP-ICE" below.

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
instrument for either quantity; the 30-day A/B below is what sized it.

**The 30-day A/B sizes the firing rate, and makes it a blocker.** Run to the
local gate's recipe (T63L47, 30 days from `echam-1m-t63_v31_warm.msgpack`
starting 2000-12-31, 5-day chunks, one A100-40GB), `physics=echam-giss-conv`
against `physics=echam`, with a noise twin of the control at
`+constants.grav=9.81000981` to measure weather divergence rather than assume
it. Area weights are Gauss-Legendre on the Gaussian grid and the surface level
is resolved from `pressure_full`. The 30-day means, each change given as a
multiple of the noise twin's drift in the same quantity:

| quantity | control | GISS | change | × noise |
| --- | --- | --- | --- | --- |
| TOA net | +3.9 W/m² | −11.0 W/m² | **−14.9 W/m²** | 69 |
| TOA SW up | 98.0 W/m² | 111.9 W/m² | +14.2 % | 45 |
| TOA OLR | 243.9 W/m² | 244.9 W/m² | +0.42 % | 11 |
| liquid water path | 59.0 g/m² | 84.7 g/m² | +43.4 % | 167 |
| ice water path | 26.6 g/m² | 26.5 g/m² | −0.17 % | 0.4 |
| convective precipitation | 1.954 mm/day | 1.569 mm/day | −19.7 % | 18 |
| large-scale rain | 0.827 mm/day | 1.088 mm/day | +31.5 % | 395 |
| total precipitation | 2.963 mm/day | 2.828 mm/day | −4.6 % | 6 |
| near-surface humidity | 8.90 g/kg | 9.42 g/kg | +5.9 % | 79 |
| near-surface temperature | 286.08 K | 286.02 K | −0.02 % | 1.9 |

Water paths are the vertical integral of the **state** tracers `qc`/`qi`, which
is where `RRTMGPRadiation` reads condensate from (`rrtmgp.py`); the
`clouds.qc` diagnostic gives +33.5 % rather than +43.4 % for the same run, and
it is not the field the radiation sees.

The gate asks for a 1 % global-mean move or 0.5 W/m² at TOA, at ≥3× the noise
twin. This clears it on every row but temperature and ice water path, the TOA
term by a factor of 69. The days 5–10 pre-divergence window agrees with the
30-day mean to 0.4 percentage points on convective precipitation, so these are
systematic properties of the scheme and not weather.

One chain explains nearly all of it. The term removes 19.7 % less water as
convective rain; the column it leaves behind is 5.9 % moister; the stratiform
scheme condenses that excess, lifting liquid water path 43.4 % and large-scale
rain 31.5 %; the resulting clouds reflect 13.9 W/m² more shortwave. That is
93 % of the TOA bias, and the clear-sky fluxes put 14.89 of those 14.94 W/m² in
the cloud radiative effect, so it is an albedo error rather than a greenhouse
one. OLR contributes the remaining 1.0 W/m², and in the opposite sense: the
port emits *more* OLR than the control, from an upper troposphere its shallow
plumes leave too dry (see below). Total precipitation falls only 4.6 %
because the stratiform compensation is nearly complete, which is why a
near-correct global precipitation here is a compensation artifact and must not
be read as the scheme being approximately right.

**The convective deficit is in intensity, not in triggering.** Read on the
days 5–10 means, the port convects over *more* area than Tiedtke, not less —
70.9 % of columns carry some convective rain against 54.1 % (tropics 74.4 %
against 62.8 %), and `cloud_base_mass_flux` is non-zero in 70.3 %. The deficit
appears entirely at the heavy end. Area raining above 5 mm/day is halved,
14.0 % against 7.1 % globally and 27.5 % against 12.8 % in the tropics, and the
conditional intensity where a column is active runs at 0.73 of Tiedtke's
(0.53 at the 90th percentile). Above the 99th percentile the port is *stronger*
than Tiedtke, 1.55× at the 99.9th.

**The missing rain is missing plume depth.** Taking the top of each column's
convective heating profile, the port's median tropical plume top is **405 hPa
against Tiedtke's 180 hPa**, and only 6.9 % of its convecting columns carry a
plume into 150–50 hPa where 40.5 % of Tiedtke's do — a factor of 0.17. It
instead fills the 600–250 hPa band that Tiedtke leaves nearly empty, at 4.7×
and 4.5× Tiedtke's share there. Tiedtke is bimodal in depth by construction,
having separate shallow and deep closures; the port is unimodal and
mid-level. A plume that stops near 450 hPa cannot rain like one reaching the
tropopause, and that is where the 3–30 mm/day band goes.

The depth gap is robust to how the top is defined. The two arms publish
*different* fields — the GISS term `convection.dth_mc`, Tiedtke
`convection.heating_rate` — so an absolute cutoff would compare two quantities
of different magnitude and measure the cutoff as much as the physics. Taking
the highest level carrying more than a fixed fraction of each column's own peak
heating, over the tropics in the pre-divergence window:

| threshold | Tiedtke median top | GISS median top |
| --- | --- | --- |
| 1 % of column max | 132 hPa | 340 hPa |
| 5 % | 156 hPa | 417 hPa |
| 10 % | 181 hPa | 457 hPa |
| 25 % | 376 hPa | 536 hPa |

The gap survives every cutoff, so it is a property of the schemes.

### But the port is not where the depth gap comes from

`ascent_exit_diag.txt` records which of the five `cloud_top` exits ended each of
ModelE's plumes, and `PlumeAscent.termination_reason` is the port's counterpart,
labelled in the same order. ModelE runs 1.76 plumes per step on TWP-ICE and the
port's single-column harness runs one, so its raw plume list carries shallow
secondaries the port never attempts; reducing ModelE to the deepest plume per
step is what makes the two populations comparable:

| exit | ModelE, deepest per step | port |
| --- | --- | --- |
| 1 `wcu2(l-1) <= 0`, no kinetic energy | 52.7 % (median top L18) | 52.5 % (L16) |
| 5 overshoot limit | 29.7 % (L46) | 27.5 % (L45) |
| 3 `mplume <= 1 %` of base | 15.8 % (L22) | 16.1 % (L21) |
| 2 `mplume <= minfrac*ma` | 1.7 % (L19) | 3.9 % (L12) |
| **median top, all plumes** | **L23.0** | **L22.0** |

751 ModelE plumes against 765 port plumes, and the termination mix agrees to
within 2 percentage points on every exit with the median top one level apart.
**On the deep case the port's ascent terminates where ModelE's does.** So the
global gap is not the port failing to reproduce `MSTCNV`'s depth, and the
earlier reading of it — that the termination criterion is a port defect — does
not survive the measurement. What the global run compares is the port against
*Tiedtke*; `MSTCNV` is not in it.

Two configuration differences remain as candidates, and they are not the same
claim as a defect in the ascent:

- **`max_dt_overshoot` is a tuning parameter, and it was hardcoded.** Exit 5
  thresholds the plume-environment temperature difference at
  `max_dt_overshoot`, which `MSTCNV.F90:297` defaults to 1.0 — the value the SCM
  oracle ran at, confirmed in its printout, so 1.0 is what reproduces every
  number in this document. GISS's own tuned presets use **1.5876319, 1.6188563
  and 2.0862949** (`:369`, `:410`, `:460`), which allow a plume to overshoot
  further before stopping. It is now a `GissConvectionParameters` leaf
  defaulting to 1.0 rather than a module constant, so the comparison is a Hydra
  override and not a code edit::

      python -m jcm.main physics=echam-giss-conv \
        ++physics.terms.giss_convection.params.max_dt_overshoot=2.0862949

  Making it a leaf is also what the repo asks of any numeric tunable — a
  constant is invisible to `jax.grad`. `TestMaxOvershootIsATunableParameter`
  pins the default against the module constant, pins that passing the default
  explicitly changes nothing, and pins the *direction*: against a 3 K lid the
  default stops the plume on exit 5 and GISS's tuned value carries it a level
  further, where it runs out of kinetic energy instead.
- **The three prognostic inputs are zero in a global run.** The harness injects
  ModelE's `cold_pool_fraction`, `penetrative_mass` and
  `convective_cloud_top_pressure` from its dumps; `jcm` reads them as optional
  diagnostics defaulting to zero, and `cold_pool_fraction` feeds
  `entrainment_efficiency` directly. Re-running the TWP-ICE tabulation with all
  three zeroed moves the port's median plume top from **L22 to L46** and makes
  the overshoot limit the terminating test on **80.5 %** of plumes, up from
  27.5 %. So the missing inputs do not make the port shallow — they make it
  *deeper*, and they move the termination onto the one test whose threshold is
  set to an untuned default.

Which of those explains the 457 hPa against Tiedtke's 181 hPa is not settled
here, and one caveat bounds all of it: TWP-ICE is a single deep tropical
monsoon case, while the global figure is a tropical median over 18,432 columns,
so the SCM population may simply not cover what the global run samples. What
can be said is that the port reproduces `MSTCNV` on the case where `MSTCNV` can
be seen, and that the next measurement is a global A/B at GISS's own
`max_dt_overshoot` rather than a rewrite of the ascent.

An earlier reading of this run attributed the deficit to the term triggering
too rarely, from an instantaneous one-day snapshot putting it at 0.3 % of
columns against Tiedtke's 4.6 %. That snapshot is not a measure of how often
the scheme convects over a climate — both arms were one day out of an initial
state — and the 5-day means above contradict it on area. The open question is
the closure's response function, not the trigger.

Double counting of the detrained condensate is excluded as a cause: the port's
handoff is `TiedtkeConvection`'s, field for field, and
`cloud_inputs.cloud_scheme_inputs` subtracts the detrainment from the running
upstream tendency before adding it back, so it enters the cloud scheme exactly
once. The liquid water path rise is the physical consequence of under-convecting,
not a plumbing defect.

### The detrained-ice deficit is its own defect, not a consequence of the depth

Against Tiedtke the port also detrains 58 % less liquid and **84 % less ice**,
and tropical snow is 77.5 % lower. A shallower plume would explain that on its
own — a plume that stops warm never reaches the levels where its condensate
freezes — but the run says the opposite. The port detrains **44.9 %** of its
tropical condensate above the 258.16 K glaciation threshold against the
control's 9.7 %: it detrains *more* in the cold, not less.

What it detrains there is the wrong phase. The ice fraction of detrained
condensate falls monotonically **upward** through the tropical troposphere —
0.84 at 368 hPa, 0.55 at 235 hPa, 0.33 at 180 hPa, **0.004 at 91 hPa (191 K)** —
where it must rise to 1. **23.4 %** of everything the port detrains is liquid at
an environment below 240 K, which needs an 18 K parcel excess to be physical.

The cause is one level of misplacement in `plume_ent_det_w2_bsort`. A level's
detrained condensate is the buoyancy sort's share plus, where the plume stops,
everything it is still carrying. The port emitted that terminal dump at the
level whose entry test the plume **failed**; ModelE emits it one level lower,
decrementing `lmax` to "the highest layer actually reached" (`:2163`) before
`DM(LMAX) += MPLUME` and `DWM(LMAX) += WMP` (`:2205-2210`).

The phase travels with the level, so the misplacement stranded it. `lhx` is
recorded only on levels the plume survived — true of the level it reached,
false of the one it failed — so the dump landed where `lhx` was 0, and the
`latent_heat > ½(LHE+LHS)` test that splits liquid from ice reads 0 as liquid.
Every plume's last and largest act of detrainment was published as liquid
however cold it was. ModelE has no such gap: it assigns `LHXMC(L) = VLAT(L)`
across the whole depth `lcl..lmax` (`:2334-2337`) *before* folding the dump into
`detrQC` (`:2340`), applying one phase to the sort's share and the dump alike.
Landing the dump on the level the plume reached fixes both at once, because that
level condensed and therefore has a phase.

`continuity_diag.txt` measures this directly, and nothing had read it — the
Fortran's own comment records it as "the one part of the tendency chain never
checked against its own oracle". On BOMEX step 1 the first plume runs levels
10–16 in both codes, and ModelE puts all of `dm(16) = 16.8468` on its top level
where the port split it 8.434 at level 16 and 8.413 at 17. The port now
reproduces ModelE's `dm` on every level of that plume to 4 significant figures,
and a test pins the whole row.

The existing glaciation tests all masked on `ascent.active`, the survival mask,
so they examined every level except the broken one.

Splitting the deficit: of the 76 % shortfall in tropical detrained ice, **43 %
is this phase defect** and **57 % is the missing condensate** — the latter
genuinely downstream of the shallow plumes and closing only when the ascent
termination criterion does. Reassigning every gram of sub-258.16 K liquid to
ice takes the port from 0.24× the control's detrained ice to 0.57×, so 43 % is
an upper bound on the phase share: the criterion is the environment's
temperature, while the port decides on the parcel's, which runs a few kelvin
warmer. Holding the parcel excess to an implausible 18 K still leaves 38 %.
The realised figure is now measured: the re-run below takes the global ice
water path bias from +6.2 % to −0.17 %, i.e. the whole of it, by *removing*
suspended ice rather than adding it.

### Landing the dump correctly costs surface rain on TWP-ICE

The fix moves heat and water as well as phase, and the surface precipitation it
delivers moves with them. Measured by one script over the same periods on both
sides of the change (BOMEX 48, TWP-ICE 749 of 766 with ModelE's `mcp` above
zero — a wider population than the 223-period table above, so these numbers
compare only with each other):

| case | before | after | median period before → after |
|------|--------|-------|------------------------------|
| BOMEX | 0.981 | **0.994** | 0.997 → 1.001 |
| TWP-ICE | 0.995 | **1.095** | 1.002 → 1.076 |

The sign is the same in both: the port rains more. On BOMEX, which was short,
that closes the gap. On TWP-ICE, which was already at 0.995, it overshoots by
9.5 %, and the error against ModelE grows roughly eighteenfold. Periods off by
more than 3× fall slightly, 60 to 51.

The placement itself is not in doubt — ModelE's own `dm` per level is exact
after the change and was split across two levels before it, and no reading of
`continuity_diag.txt` makes the old split right. What the overshoot says is
that the misplacement was **compensating** for something: depositing a plume's
terminal heat one layer too high stabilised the air above each plume top
against the next candidate, suppressing convection the port needed suppressed
for the wrong reason. With that gone, deep convection rains too much on the
case with the most plume terminations — and the effect scales with depth, which
is why BOMEX moves 1 % and TWP-ICE 10 %.

What should not be concluded is that TWP-ICE's 0.995 was evidence of fidelity:
it was a compensation, of the same kind as the near-correct global total
precipitation above.

### The re-run sizes the fix globally, and leaves the blocker where it was

The A/B was re-run on the fixed term, reusing the `control` and `noise` arms —
both `physics=echam`, so untouched by the change, and their global means come
back bit-identical, which is what makes the two GISS arms comparable to each
other. The table above is the post-fix run. Against the pre-fix one:

| quantity | control | pre-fix | post-fix | what the fix did |
| --- | --- | --- | --- | --- |
| TOA net | +3.9 W/m² | −10.5 W/m² | −11.0 W/m² | **−0.5 W/m², the wrong way** |
| TOA SW up | 98.0 W/m² | +15.2 % | +14.2 % | −1.0 W/m² of excess reflection |
| TOA OLR | 243.9 W/m² | −0.19 % | +0.42 % | +1.5 W/m², overshooting the control |
| liquid water path | 59.0 g/m² | +45.7 % | +43.4 % | −1.4 g/m² |
| ice water path | 26.6 g/m² | +6.2 % | −0.17 % | **bias removed** |
| convective precipitation | 1.954 mm/day | −20.7 % | −19.7 % | +1.0 pp of the deficit |
| large-scale rain | 0.827 mm/day | +33.3 % | +31.5 % | −1.8 pp of the excess |

Three readings, in descending confidence.

**The ice water path bias is gone.** It was +6.2 % and is now −0.17 %, inside
the noise twin's own drift. That is the fix doing exactly what the phase
argument said it would, and it is the one row that moves decisively. Note the
direction: *correcting* the phase to ice **lowered** the ice water path. The
stranded condensate was being published as liquid at 190–240 K, where the
stratiform scheme froze it anyway and then held it aloft; landing it a level
lower, with the phase the level actually condensed at, puts it where it
precipitates instead. The pre-fix +6.2 % was never ice the convection
detrained — the port detrains 84 % *less* ice than Tiedtke — it was the
stratiform scheme's repair of a misplaced liquid field.

**The port now emits too much OLR, where before it matched — and the clear-sky
fluxes say that is a cancellation coming apart, not a new defect.** +0.42 % at
30 days and +0.51 % in the pre-divergence window, 11× and 20× the noise twin.
Splitting the pre-divergence window with `toa_lw_up_clear`:

| W/m² (pre-divergence, global) | control | pre-fix | post-fix |
| --- | --- | --- | --- |
| OLR, all-sky | 243.38 | 243.44 (+0.06) | 244.63 (+1.25) |
| OLR, clear-sky | 260.56 | 262.74 (+2.18) | 262.96 (+2.40) |
| LW cloud effect | 17.17 | 19.30 (+2.13) | 18.33 (+1.15) |

The port's *clear-sky* OLR was already 2.18 W/m² high before the fix and the fix
barely moves it (+0.22). What the fix changed is the cloud term: the longwave
cloud effect was 2.13 W/m² **too large** and is now 1.15 — the excess is
halved. Pre-fix all-sky OLR agreed with the control to 0.06 W/m² because those
two errors, each about 2 W/m², very nearly cancelled. The fix corrected one of
them, so the other is now visible.

This is the compensation pattern seen on TWP-ICE surface rain, but the sign of
the cloud term is the opposite of what a depth deficit alone would suggest: the
port traps *more* longwave in cloud than Tiedtke, not less, because its
suspended upper-level ice runs 3–7× the control's at 110–270 hPa even while its
high cloud *fraction* is lower. Depositing each plume's terminal condensate one
level above where the plume reached inflated that further, and landing it
correctly removed half the excess.

**The TOA bias is unchanged to slightly worse, and is still the blocker.**
−14.4 → −14.9 W/m², which is 0.5 W/m² at 30 days and 1.0 W/m² in the
pre-divergence window — small, but both windows agree on the sign and both
exceed the noise twin's TOA drift (0.04 W/m² pre-divergence, 0.22 W/m² at 30
days) by more than an order of magnitude, so it is not weather. The shortwave
improvement that partly offsets it is **not** firm: it is 1.0 W/m² at 30 days
but only 0.2 W/m² at days 5–10, and the two windows disagreeing that way is
the signature of an adjustment rather than a systematic property.

**The clear-sky OLR excess is the depth deficit in a third channel.** It is
not a radiation or cloud defect at all: the port's tropical humidity profile is
*moister than the control below 890 hPa* (+9 % at 993 hPa) and **drier through
the whole mid and upper troposphere** — −14 % at 805 hPa, −35 % at 302 hPa,
−37 % at 237 hPa. Upper-tropospheric water vapour is what sets clear-sky OLR,
and convection that stops at 405 hPa does not put moisture above it. So the
same defect now has three independent signatures: 19.7 % less convective rain,
a 43 % liquid water path excess, and a dry upper troposphere worth +2.4 W/m² of
clear-sky OLR. None of the three is a free parameter.

The global run also confirms the phase fix directly, in the detrainment fields
themselves. Tropical `clouds.conv_detrainment_qc` above 420 hPa was non-zero up
to **74 hPa** before the fix — liquid detrained at 190–220 K, which is the
defect — and is now **exactly zero** at every level above 452 hPa, while
`conv_detrainment_qi` at 110–237 hPa rises from 0.17–0.68 to 0.54–0.99
(×10⁻⁹). The port's detrainment phase structure now has the same shape as
Tiedtke's: ice aloft, liquid only in the lower troposphere.

So the detrained-phase defect was real, is fixed, and is worth about 1 W/m² in
each direction at TOA with the two nearly cancelling. It is not the −14.9 W/m²
bias. That remains overwhelmingly shortwave — of the +14.94 W/m² of extra
reflection in the pre-divergence window, **+14.89 is cloud radiative effect**,
so the albedo reading is exact rather than inferred — driven by a liquid water
path still 43 % high, which is downstream of the ascent terminating ~225 hPa too
low. **The termination criterion is the merge blocker, and it is now the only
one with a measured climate effect.**

**One diagnostic was published upside down.** `convection.dth_mc` and
`dq_mc` were written in the plume code's own vertical frame — ModelE's, with
`L=1` at the ground — while every other published field is in the state's.
A deep plume's heating maximum therefore appeared at 17 hPa with the whole
troposphere reading as exactly zero. Only the diagnostics were affected: the
tendency the model integrates is flipped back at its own return, which is why
no budget, conservation check or column case caught it, and why the simulated
climate above is unaffected. The plume tops in this section are read from the
corrected orientation. `TestTheDiagnosticsAreInTheStatesFrame` now pins both
fields against the tendencies they are computed from, and against a plume
being unable to heat the model top.

Correcting it broke the single-column harness in the bridge repo, which had
been reading `dth_mc`/`dq_mc` straight out of the term and scoring them against
ModelE's surface-first columns — correct only for as long as the term published
in ModelE's frame. Every profile the harness scored was then reversed: RICO's
per-period heating correlation reads −0.07 against +0.989 flipped. Nothing in
`jcm` can catch this, because the harness is the thing that holds the two
conventions together; `TestCrossCaseAgreement` caught it, one commit late. The
lesson for the port is that **the bridge suite is part of the gate for any
change to a published convection field**, not only for changes to the physics.

**Cost.** The term integrates 30 days in 157.4 min against the Tiedtke
preset's 12.0 min on the same A100 — **13.1×**, or ~11 simulated days per hour
against ~180. Running it with the full 40 GB rather than the 9 GB left by a
careless host process bought 3.6 %, so this is the term's cost and not a
memory-pressure artifact. The plume scan, not the compile, dominates: XLA
compilation is a one-off ~30 min and the persistent compile cache removes it on
a rerun.

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
   term is trusted outside the tropics. See `MIXED_PHASE_SUPPORTED`. The global
   run's 84 % detrained-ice deficit is **not** this gap: part of it was the
   terminal dump's lost phase, now fixed — worth the whole of the +6.2 % ice
   water path bias — and the rest is the depth.
2. **The terminating-level rain partition.** The port rains 0.650 of ModelE's
   condensate at the level a plume dies on and over-rains by 3.1% below, so the
   column total is 0.990 — a redistribution rather than a loss, worth ~1%. Its
   effect on the *downdraft* profile is unmeasured, and that coupling is a known
   runaway.
3. **Cross-plume phase latching.** ModelE shares `TPSAV`/`VLAT` across every
   plume of a timestep; the port decides per plume. Measured inert here: the two
   disagree on 3 of 3,772 shared `(step, level)` pairs, none on an active period.

**Not yet done.**

1. The **convective intensity distribution**, which the 30-day A/B above
   turned from an open question into the port's one blocking defect, and which
   the post-fix re-run leaves untouched. The term produces 19.7 % less
   convective rain, and that propagates to a −14.9 W/m² TOA bias — 93 % of it
   shortwave — through a moister column and a brighter stratiform deck.
   It is not a triggering failure: the port convects over more area than
   Tiedtke and finds a cloud base in 70 % of columns. It is that the heavy
   events are missing — half the area above 5 mm/day — while the extreme tail
   is heavier, a bimodal response pointing at the closure rather than at the
   trigger or the plume. The column cases bound what the plume can explain:
   production is 0.91 of ModelE's on TWP-ICE and BOMEX, far too little for a
   20.7 % global deficit. Until this is closed the term is not usable for a
   coupled climate run, whatever its column-level fidelity.

   What *is* settled is that it runs globally and stably. `+configuration=
   t63-echam-1m physics=echam-giss-conv` completes 30 days at T63L47 with
   RRTMGP, every output field finite, under the runner's health gates
   (`bail_on_unhealthy`) and without tripping them, with a cloud base in nearly
   every column, `cloud_base_mass_flux` to 149 kg/m² and `mcp` to 13.8 mm/day.
   Near-surface temperature moves 0.01 % over the 30 days, within the noise
   twin, so nothing in the term is injecting or removing heat the rest of the
   stack has to absorb at global scale.

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
