# Porting plan: `plume_ent_det_w2_bsort` → JAX

Status: **plan, not yet implemented.** Written after the mass-budget oracle
(bridge repo `f07b3d2`) established that ModelE runs the buoyancy-sorting plume
routine, not the one currently ported.

## 1. Why this is being rewritten

`MSTCNV.F90:1885` selects between two plume ascent routines on `bsort_entdet`,
which `sync_param` defaults to `.true.` (`MSTCNV.F90:489`); the BOMEX rundeck
does not override it. So:

| routine | lines | status |
|---|---|---|
| `plume_ent_det_w2` | 2823–3178 | **dead code** — what `giss_plume.py` ports |
| `plume_ent_det_w2_bsort` | 3180–3901 | **what actually runs** |

The oracle dump `oracle_data/bomex_mass_budget.txt` (536 plume levels) shows the
budget `mplume_out = mplume_in·(1−frem) + addback − cap_shed` closing to 1.5e-7,
with median net mass change per level of **−31%** and local detrainment firing on
**409/536** levels versus the downdraft's 268. Our port's only mass sink is the
`fpl·ETADN` downdraft diversion, so it sheds far too little mass — the leading
candidate for the ~1.7× tendency overshoot.

## 2. Scope: what changes, what survives

**Survives unchanged**
- `giss_mass_flux.py` — the `MASS_FLUX2` cloud-base closure. Independent of the
  ascent routine; currently 47/48 exact on cloud base, `fmp2` 0.98–1.10×.
- `giss_mstcnv.py` — `surface_flux_scales`, `convective_velocity_scale`,
  `_source_parcel_inputs`, the `PhysicsTerm` shell and its interfaces.
- The parcel thermodynamics inside `_plume_core` (lift, saturation adjustment).
- `giss_tendencies.py` **advective** subsidence form.

**Rewritten**
- The entrainment rate: Gregory `ε = (1/6)·contce·g·B/w²` → the energy closure
  in §3.1.
- The entire mass-exchange core: single `ε`/`δ` pair → set-aside + blend
  spectrum + three-way sort.
- `w` evolution: dilution-based (§3.3) rather than the Gregory (1997) `w²`
  equation.

**Deleted**
- `plume_mixing_fraction` (the `get_fpl` port). It implements downdraft
  initiation for the *non-bsort* routine; bsort initiates downdrafts from the
  sorting itself. Its 2×2-step Newton solves and `_ETADN` become dead code.
- The `1.5·ma(l)` mass cap need **not** be ported: it never fires in BOMEX
  (max `mplume/ma` = 0.59 vs a threshold of 1.5) and its own comment says it
  exists only to bound advective substepping.

## 3. Algorithm specification

Per level `l`, in order. Fortran line numbers are current `MSTCNV.F90`.

### 3.1 Entrainment rate (3428–3481)

Running integrals carried up the column:

```
a_buoy  = 1/6 if buoy(l) >= 0 else 1        # full deceleration in overshoots
bdz     = a_buoy * grav * buoy(l) * delz
kew    += bdz * mplume                       # buoyancy-work integral
bdzsum += bdz * (1 - enteff)
```

Then

```
ent = min(0.95, sqrt(kew/(bdzsum*mplume)) - 1) / delz     if kew > bdzsum*mplume
ent = -0.5/delz                                            otherwise
```

with floors applied only when `buoy(l) > 0` (`3e-4` for `iplume==1 & p>700 mb`,
else `5e-4`), then ceiling `ent = min(ent, 4e-3)`. Finally
`if ent < 0: det = -ent; ent = 0` else `det = 0`.

Note `delz = ma(l)/rho0(l)`, which the source itself flags as differing from
`gzl(l)` — both appear and must not be conflated.

**Interpretation** (from the routine header): the rate is chosen so that its
dilution reduces local `w` to a target fraction `enteff` of the maximum that the
buoyancy history could support. It is a rate of *mixture formation per distance
ascended*, not a rate of mixing into the whole updraft.

### 3.2 Blend air sizing (3484–3585)

```
envairm = min(mplume, 2*mplume_b) * (ent * gzl(l))
envairm = min(envairm, ma(l)*remrat)                 # remrat = 0.333
refblendwt   = min(1, mplume/mplume_lag)
fupd_fullmix = mplume/(mplume + envairm)
frac = [0.25, 0.50, 0.25]                            # env air share per blend
fupd = refblendwt*[0.25,0.50,0.75] + (1-refblendwt)*fupd_fullmix
updairm = envairm * (1/sum(frac*(1-fupd)) - 1)
```

then clamp `updairm <= 0.99999*mplume`, rescaling `envairm` by the same factor.
Per-blend weights:

```
fupdavg   = sum(frac*fupd)
updfac(n) = frac(n)*fupd(n)/fupdavg
envfac(n) = frac(n)*(1-fupd(n))/(1-fupdavg)
```

Overshooting branch (`envairm <= 0`): `updairm = mplume*min(0.95, det*gzl)`,
`nmix=1`, `updfac=1`, `envfac=0`.

`refblendwt` pushes the blending ratios toward complete mixing where plume mass
has decreased over the last ~1 km — the header's rationale is that a thin plume
leaves less updraft air able to escape mixing.

The 4th precip-induced downdraft blend (3540–3568) is **disabled in source**
(`ddmassp = 0.` at 3546). Do not port it; leave a comment saying why.

### 3.3 Vertical velocity (3505–3508, 3893–3897)

```
mw     = sqrt(2*max(0,kew)*mplume)
wcu(l) = mw/(mplume + envairm)          # dilution by the entrained air
```
and after the plume mass is updated, convert back:
`mw = wcu(l)*mplume; kew = mw*mw/(2*mplume)`.

### 3.4 Set aside and build blends (3600–3712)

`frem = updairm/mplume`; scale `mplume, smp, qmp, wmp` by `(1-frem)` and hold
the removed part as `updsm/updqm/updwm`. For each blend `n`:

```
airmix = updfac(n)*updairm + envfac(n)*envairm
smix, qmix, wmix = corresponding mass-weighted means / airmix
```

### 3.5 Evaporate, then sort (3713–3730)

`get_dq_evap` (`CLOUDS_COM.F90:903`) is a **3-iteration Newton** solve toward
saturation, clamped to the available condensate:

```
for n in 1..3:
    qst = qsat(TP, lhx, pl)
    dq  = (QMT - mass*qst)/(1 + slh*qst*dlnqsatdt(TP,lhx))
    TP += slh*dq/mass;  QMT -= dq;  DQSUM -= dq
DQSUM = clip(DQSUM, 0, cond)
```

Fixed trip count, no data-dependent loop — ports directly and stays
differentiable. Then

```
smix -= slh*dqevp/plk(l);  qmix += dqevp;  wmix -= dqevp
tvmix   = smix*plk(l)*(1 + deltx*qmix)
mixbuoy = (tvmix - tvl(l))/tvl(l) - wmix
```

Three-way fate, with `posbuoy=+0.05`, `negbuoy=-0.2` (3322–3324):

| condition | fate | mass effect |
|---|---|---|
| `mixbuoy > posbuoy/tvl(l)` | rejoin updraft | `mplume += airmix` |
| `mixbuoy < negbuoy/tvl(l)` and not overshooting | downdraft | `ddr(l) += airmix` |
| otherwise | detrain locally | `dm(l) += airmix` |

Mass not rejoining simply stays out — it was already removed by `frem`.

## 4. JAX structure

Broadcasting-native as per `CLAUDE.md`: vertical on axis 0, no `vmap`.

- **`kew`, `bdzsum`, `mplume`, `smp`, `qmp`, `wmp` are scan carries.** The ascent
  is already a `lax.scan` in `plume_ascent_column`; these join the carry.
- **The `nmix=3` blend loop is fully unrolled** — it is a static 3, and all three
  blends are independent given the carry. Vectorize over a length-3 axis and
  reduce; no scan needed.
- **The three-way sort becomes two nested `jnp.where`.** Cheap, but see §6.
- **`mplume_lag` is the hard part.** ModelE walks back down the column until
  `zl(l) - zl(ll) > 1 km` (`MSTCNV.F90:1893–1896`), reading `mplumearr`. Inside a
  scan we cannot index a not-yet-written array. Options:
  1. Carry a fixed-length ring buffer of the last `N` levels' `mplume` and `zl`,
     and select with a masked reduction. Exact if `N` covers 1 km (BOMEX layers
     near cloud base are ~100 m, so `N=16` is ample); costs `N` floats of carry.
  2. Two-pass: run the ascent once to get `mplume(l)`, then rerun with
     `mplume_lag` available. Doubles cost and is not self-consistent.
  **Recommend option 1.** Verify `N` against `zl` spacing at build time and
  assert.

## 5. Phasing, each with an oracle checkpoint

Every phase is validated against `oracle_data/bomex_mass_budget.txt` — column by
column, not just on final tendencies. This is what makes the rewrite tractable.

| phase | deliverable | oracle check |
|---|---|---|
| 0 | **done** — dump extended with `enteff`, `mplume_lag`, `gzl`, `delz`, `buoy`, `bdzsum`, `kew`, and the per-blend `mixbuoy`/`airmix`/fate; rerun; §3 re-derived from it (see §5a) | — |
| 1 | **done** — `condensate_evaporation` in `giss_thermodynamics.py` + 8 tests | matches a literal transcription of the Fortran loop in the Newton interior and at both clips |
| 2 | **done** — `entrainment_rate` in `giss_bsort.py` | `ent` **536/536**, `det` **536/536** vs dump, max rel 2.4e-6 |
| 3 | **done** — `blend_air_masses` in `giss_bsort.py` | `envairm` 518/518 entraining, `updairm` **536/536**, `fupd`/`updfac`/`envfac` **1554/1554** |
| 4 | **done** — `sort_blends` in `giss_bsort.py` | `mplume_out`, `detrained`, `downdraft` all **536/536**, max rel 8.2e-7; mass conservation 3.6e-16 |
| 5 | **done** — `vertical_velocity` / `kinetic_energy` | `wcu(l)` **536/536**, max rel 6.7e-7 |
| 5a | **done** — `plume_ascent` driver (scan, lookback, termination) | `mplume_lag` **536/536**; cloud-base level exact |
| 5b | **done** — `condensation` + `resaturate_plume`; precipitation via a calibrated stand-in | heat and vapour **484/484**; **column closes**: mass 0.06%, w 0.14%, detrained 0.12% median error over 534 levels |
| 6 | Rewire `giss_tendencies.py` to the new detrainment sources (`dm`, `dmr`, `ddr`) | BOMEX heating/moistening profiles vs `dth_mc`/`dq_mc` |

## 5b. What blocks a full-column chain

The driver reproduces the cloud-base level exactly and its lookback matches
536/536, but chaining it up a whole column diverges from ModelE after the first
level — **not because the sorting is wrong** (that is 536/536 when fed the
oracle's per-level state) but because ModelE does more between plume levels
than sorting. From `MSTCNV.F90` around the ascent loop, each level also runs:

1. `get_dq_cond` — condensation of the lifted parcel, releasing latent heat;
2. `CONVECTIVE_MICROPHYSICS` — which removes condensate as precipitation.

Measured on the first BOMEX column, the plume entering level 11 carries
**0.0049 less total water** than sorting alone predicts, and correspondingly
more condensate and heat. That is the precipitation and the latent release.

### What the inter-level step decomposes into

Dumping bsort's *output* state as well as its input made this exactly
separable, and the split is cleaner than expected:

* **Plume mass is untouched** between levels — 484/484 transitions carry it
  over unchanged. The microphysics moves water, not air.
* **Re-saturation sets heat and vapour exactly.** ModelE does not condense
  incrementally: it evaporates *all* existing condensate back to vapour, then
  recomputes the split from scratch at the new level. `resaturate_plume` +
  `condensation` reproduce the oracle's heat and vapour **484/484** to ~1e-6,
  and conserve total water to 2e-16.
* **The entire remaining gap is one scalar per level:** the condensate we keep
  and ModelE rains out. Median **16.6%** of the condensate, max 62.5% — which
  is 0.17–8.5% of the plume's total water per level.

So the column is blocked on **precipitation only**, and its size is now known
rather than guessed.

### The precipitation piece

`CONVECTIVE_MICROPHYSICS` (`MSTCNV.F90:6462-6814`, ~350 lines) is a scheme in
its own right: Marshall-Palmer size distributions, cloud-droplet number
concentration, graupel fraction, particle volume/area/fall-speed, and an ice
habit selection. It also takes aerosol-derived `CDNC` as input, so it reaches
outside `MSTCNV`.

It is comparable in size to the bsort port itself, and it is a *microphysics*
scheme rather than the convection routine the deliverable names, so rather than
port it now the column is closed with a **calibrated stand-in**,
`precipitation_fraction`.

### The stand-in, and what it is not

`fraction = 1 - exp(-(qc/scale)**exponent)` on the in-plume condensate mixing
ratio. Its shape was chosen from the oracle, not assumed: the precipitated
fraction correlates **+0.97** with `qc` alone, and adding a residence-time
factor `gzl/w` makes the fit *worse* (+0.89). The fitted exponent comes out at
**~2.1** — the quadratic collection dependence Kessler-type autoconversion
assumes, which is a reassuring result for a fit that was not constrained to
find it. RMS error 0.029 in the removed fraction, against 0.136 for the best
flat fraction.

Both coefficients are differentiable arguments rather than hard-coded
constants, so they remain available for later calibration, and the seam is
clean if the real microphysics is ported.

**This is the one piece of the port that is calibrated rather than derived.**
Any result that depends on convective moisture should say so.

### The column, closed

With re-saturation and the stand-in in place the ascent tracks ModelE level by
level over all 52 BOMEX columns (534 active plume levels):

| quantity | median error | 90th percentile |
|---|---|---|
| plume mass flux | **0.06%** | 1.83% |
| updraft speed `w` | **0.14%** | 0.85% |
| detrained mass | **0.12%** | 1.41% |

Figures: `plume_scatter.png` and `plume_profiles.png`, reproducible via
`python -m modele_jcm_bridge.plume_compare_plots` in the bridge repo.

Phase 0 first: `enteff` in particular is **not** a constant *in general* — it is
set at `MSTCNV.F90:1645–1665` from `closure2` (itself a per-step condition,
`lmin == lmax_disp2 .and. lmax_disp2 > lmax_disp1`, line 1463) and from the cold
pool fraction `fcp`. Guessing it would repeat the mistake that produced the
false "over-penetration" diagnosis earlier in this port. Dump it.

## 5a. Phase 0 results — spec verified against the oracle

Done. `oracle_data/bomex_mass_budget.txt` (536 levels) and
`oracle_data/bomex_blend_diag.txt` (1572 blend records) now carry the closure
inputs and the per-blend sorting decision. Re-deriving §3 from those columns
reproduces ModelE exactly, so **phases 2, 3 and the sort in 4 have an exact
target before any JAX is written**:

| quantity | agreement |
|---|---|
| `ent(l)`, `det(l)` from §3.1 | **536/536**, max rel err 2.4e-6 (= dump precision) |
| `envairm` from §3.2 | exact; `ma·remrat` cap binds **0/533** |
| `fupd(n)` | **1554/1554** |
| `updfac(n)`, `envfac(n)` | **1551/1551** |
| three-way fate | **1572/1572** |

Simplifications this establishes for BOMEX (assert, don't assume, for other cases):

- **`enteff = 0.67` constant and `iplume = 2` on every level.** The `closure2`
  branch never fires, so the `iplume==1` entrainment floor (3e-4) is
  unreachable — the floor is always **5e-4**.
- **`ma(l)·remrat` never binds** (0/533), like the `1.5·ma` cap.
- **`nmix` is 1 exactly when `det(l) > 0`** (equivalently `envairm = 0`, the
  overshooting branch), and 3 otherwise. 18 of 536 levels take that branch.
- **The fate rule needs the overshooting guard**, which is *not* optional: 6
  blends would otherwise be sent to the downdraft instead of detraining.
  `in_overshooting_regime = buoy(l) <= -0.25/tvl(l)` (`MSTCNV.F90:3470`).
  Without it the sort reproduces 1566/1572; with it, 1572/1572.
- **`pres(l)` is in Pa but `pl(l)` is in mb**, and both appear in this routine.
  `get_dq_evap` is called with `pres(l)`, so the blend saturation calculation is
  Pa-based while the `pl(l) > 700` entrainment-floor test is mb-based. Treating
  `pres` as mb put `qsat` off by a factor of 100 and reduced the `mixbuoy`
  agreement to 2/1572; it is the only unit trap found in this routine.
- **The evaporation is a test, not a state update.** ModelE evaporates each
  blend's condensate solely to decide its buoyancy; the branches then add back
  the untouched pre-evaporation extensive `smmix`/`qmmix`/`wmmix`. Applying the
  evaporation to the returned properties would create vapour out of nothing —
  `test_conserves_water_because_evaporation_is_only_a_test` guards this.
- **Residual `mixbuoy` disagreement is dump precision, not physics.** Absolute
  error is median 8.4e-8 / max 4.0e-7 against a decision scale of 1.7e-4, which
  is what the dump's ~7-significant-figure `smix` predicts. Caveat: the closest
  any blend comes to a threshold is 3.5e-7, marginally below that max error, so
  fate agreement is not *structurally* guaranteed for blends sitting that close
  — none flipped here, and the mass budget confirms it.
- `mplume_lag == mplume_b` on 444/536 levels — the 1 km lookback usually reaches
  cloud base in shallow BOMEX. This does **not** remove the need for the §4
  ring buffer, but it does give a cheap regression check: a buffer bug will show
  up as disagreement on the other 92 levels.

## 6. Risks

- **Differentiability of the sort.** The three-way fate is a hard threshold on
  `mixbuoy`. `jnp.where` gives a correct value and a piecewise-constant
  selection, so gradients flow through the *blend properties* but not through
  the *choice*. With only 3 blends the choice is coarse, so `d(tendency)/d(param)`
  will have genuine kinks. This matches ModelE's own behaviour and is the honest
  port. If gradient quality proves inadequate for optimization, a smooth
  sigmoid over `mixbuoy` is a drop-in alternative — but it changes the physics
  and must be an explicit, off-by-default option, not a silent substitution.
- **`mplume_lag` buffer length** — assert coverage rather than assume.
- **`iplume`** selects entrainment floors and `bsort_enteff`. The plume oracle
  showed only plume 2 fires under `lessent_scheme=2` (default, line 622), so the
  `iplume==1` floor branch is likely unreachable here — confirm from the phase-0
  dump before hard-coding either way.
- **Scope**: this is a core rewrite of `giss_plume.py` (~560 LOC, 28 tests).
  Expect the tests to be rewritten alongside, not merely extended.

## 7. Deliberately not ported

- 4th precip-downdraft blend — disabled at source (`ddmassp = 0.`, 3546).
- `1.5·ma(l)` mass cap — never fires; numerical-stability aid only.
- `SIMPLER_ENT` branch — not compiled in this configuration.
- Tracers (`TRACERS_ON`/`TRACERS_WATER`) and momentum (`ump`/`vmp`) blend
  transport — out of scope for the current `PhysicsTerm` interface, which
  carries no tracer or momentum tendency from convection. Momentum should be
  revisited once the thermodynamic path matches.

## 7a. Scoping `dd_evap_precip_loop` (the downdraft)

Assessed rather than started, since it decides whether the tendency comparison
rests on one approximation or two.

### What the 808 lines actually are

| | lines |
|---|---|
| total | 808 |
| comments / blank | 189 |
| tracer-guarded (`TRACERS_ON` is **off** in this build) | ~196 |
| declarations | ~30 |
| **live logic** | **~390** |
| — of which: the descent loop, non-tracer | 252 |
| — of which: pre-loop (precip phase, melting, `mcfrac`) | ~140 |

BOMEX is all-liquid (`lhx = LHE` on every oracle level), so the melting,
freezing and snow-phase machinery — most of the pre-loop's complexity — is
**inert**. Tracers are out of scope for the `PhysicsTerm` regardless.

### The core is smaller than the line count suggests

Per level, descending from cloud top:

1. accumulate the downdraft mass the sorting produced: `ddraft += ddr(l)`;
2. evaporate precipitation into it — **`get_dq_evap`, already ported and tested**
   as `condensate_evaporation`;
3. cool and moisten it: `smdn -= slh*dqevp/plk`, `qmdn += dqevp`;
4. test buoyancy against the environment (`svmix` vs `svm1`) and detrain when it
   turns positively buoyant, with forced detrainment once inside the boundary
   layer;
5. deposit the detrained air into `dm`/`dsm`/`dqm`.

That is a **downward scan mirroring `plume_ascent`'s upward one**, on machinery
already built and proven. Estimate: ~120-160 lines of JAX plus tests, comparable
to `sort_blends`, plus an oracle dump of `ddm`/`thdn`/`qldn`/`dqevp` to validate
against — the step that has made every previous phase land correctly.

Tunables come from the same preset block already confirmed active (the one
carrying `bsort_enteff2 = 0.67`): `dd_evpeff_qp_scale = 0.001`,
`mc_fddrt = 0.5`, `geometric_fevap = .true.`

### The catch, stated plainly

`dd_evap_precip_loop` consumes `condpr` — the precipitation produced by
`CONVECTIVE_MICROPHYSICS`, which is **not** ported. Our stand-in supplies a
precipitated mass per level, so it can drive the downdraft, but the evaporative
cooling then inherits the stand-in's error.

**So porting the downdraft does not reduce the approximation count to one.** It
changes the situation from

* *downdraft air deposited at the wrong level, with no evaporative cooling at
  all* — a structural error in where and how the convection cools and moistens,
  affecting 29.8% of everything leaving the plume;

to

* *downdraft descending and evaporating in the right place, by the right
  mechanism, with an approximate precipitation supply.*

That is a real improvement in kind, not just in magnitude, and it is the
difference between a profile that is biased and one that is misshapen. But the
precipitation caveat survives it either way.

## 7b. Downdraft port — state

`giss_downdraft.py` implements the descent core. Every parameter was extracted
from a dump taken inside ModelE's own descent loop
(`oracle_data/bomex_downdraft.txt`, 873 records over 52 columns), not guessed:

| quantity | value | how established |
|---|---|---|
| descent budget | `dd_out = dd_in + ddr + edraft − detr` | closes to 1.1e-5 (dump precision) |
| detrained fraction, buoyant | **0.75** | measured; implies `fddet = 0.25`, which has no assignment left in `MSTCNV.F90` |
| detrained fraction, in BL | **0.50** | measured; matches `detfac = 0.5` (`MSTCNV.F90:4230`) |
| `dd_detbyent` | 0 | active preset, line 288 |
| downdraft entrainment | `etal = 2.0e-4 · gzl` | `etal/gzl` constant to 0.8% over 561 records |
| precip share to downdraft | `fddrt = 0.5` | active preset |
| evaporation efficiency scale | `dd_evpeff_qp_scale = 1e-3` | active preset |

Mechanism counts over 854 active levels: detrainment fires on 565 (377 from
positive buoyancy, **188 from the boundary-layer forced branch**), entrainment
on 373, evaporation on 813 — so all three paths are live and none can be
dropped.

### Not yet verified

The module passes a smoke test (mass balances, gradients finite, the downdraft
accumulates then sheds in the boundary layer) but has **not** been validated
level-by-level against the oracle. Doing so needs one more dump round: the
descent trace carries `ddr` but not `smdnl`/`qmdnl` (the heat and water the
plume routes into the downdraft), and the environment profile below cloud base,
which the mass-budget dump does not cover because the plume never goes there.

Until that check is run, the module should be treated as written-but-unverified
— which, on the evidence of every earlier phase here, is not the same as
correct.
