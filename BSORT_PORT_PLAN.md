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

### Validation round: what it settled, and what it exposed

**Verified directly against the oracle:**

* The **branch logic is exactly right.** ModelE's detrained fraction comes out
  at precisely 0.75 on positively buoyant levels (n=377), 0.50 on
  boundary-layer-forced levels (n=188) and 0.00 elsewhere (n=185) — the three
  cases this port implements, with the constants it uses.
* The descent budget `dd_out = dd_in + ddr + edraft − detr` closes to 1.1e-5.
* With no evaporation supplied, downdraft mass and entrainment reproduce the
  oracle exactly on the upper levels (median relative error 0.000%), before the
  missing cooling makes the two solutions diverge.

**Not verified, and now understood to be unverifiable in isolation:** the
descent chain level by level. It depends on the precipitation falling through
the downdraft, which comes from `CONVECTIVE_MICROPHYSICS` — not ported.

### The finding that matters more than the port

The downdraft is **acutely sensitive to the precipitation supply**, far more
than expected:

| precipitation supplied | downdraft mass, median error vs oracle |
|---|---|
| none | 0.000% on upper levels, diverging below |
| over-supplied (0.05 kg/m² per level) | **322%** |

The mechanism is a feedback: evaporation cools the downdraft, cooling keeps it
negatively buoyant, staying negatively buoyant suppresses detrainment, and
suppressed detrainment lets the mass run away. Get the precipitation wrong and
the downdraft's whole structure is wrong, not merely its magnitude.

**This inverts the reasoning behind porting it.** The expectation was that
porting the downdraft would leave one approximation (precipitation magnitude)
in place of two. Instead, the downdraft *amplifies* the precipitation
approximation: it converts an error in how much rain forms into an error in
where and whether the downdraft deposits its air at all.

So the honest position is:

* the downdraft port is **structurally correct and parameter-exact**, and it is
  the right thing to have;
* but it cannot be validated, and should not be trusted quantitatively, until
  the precipitation it consumes is real rather than fitted;
* and the precipitation stand-in is now a **more** load-bearing approximation
  than it was before the downdraft existed, not less.

Porting `CONVECTIVE_MICROPHYSICS` has therefore moved from "arguably out of
scope" to the critical path for any quantitative convective-moisture result.

## 8. Next: `CONVECTIVE_MICROPHYSICS` — scoping

Now on the critical path (see §7b). Scoped, not started. **This corrects the
earlier estimate in two directions.**

### Easier than assumed

`CDNC` is **not** an aerosol computation. It is
`cdnc_ocean_mc*(1-pearth) + cdnc_land_mc*pearth` (`MSTCNV.F90:1269`) — a
land/ocean blend, so a constant `60` over BOMEX's ocean. The worry that this
routine reaches outside `MSTCNV` for aerosol input was wrong.

### But two more live/dead branch traps, of exactly the kind that cost this port before

Both flags default to `.true.`, so the obvious-looking code is **dead**:

| flag | default | consequence |
|---|---|---|
| `use_s08_fallspeed_mstcnv` | `.true.` | the 15-iteration Newton solve for `DCW` is **not** used; an analytic log form is |
| `use_gammadsd_mstcnv` | `.true.` | `PRECIP_MP` (Marshall-Palmer) is **not** called; `PRECIPLIQ_GAMMA` is |

And `PRECIPLIQ_GAMMA` is defined twice, resolved by `#define FAST_MICROPHYSICS`
at the top of the file: the copy at 6963 compiles as `PRECIPLIQ_GAMMA_orig` and
is dead; the **active** one is at **7863-8114**, ~250 lines.

So the routine to port is `PRECIPLIQ_GAMMA` (7863-8114) plus:

* `DCW` — the drop diameter whose terminal fall speed matches the updraft, from
  the analytic form
  `DCW = max(0, log(((w*(p/1e5)^0.4) - 9.65)/(-9.8))/(-600))`, saturating when
  `w*(p/1e5)^0.4 >= 9.65`;
* `CDNC` (constant here) and `cloudrvl_mstcnv = 10` µm;
* the mixed-phase and pure-ice branches, **inert for BOMEX** but needed for
  other cases.

### Physical picture

An assumed gamma drop-size distribution with number `CDNC` and mean volume
radius `cloudrvl_mstcnv`; drops larger than `DCW` fall out of the updraft and
become `CONDP`. So precipitation is set by the competition between updraft speed
and drop fall speed — which is why the calibrated stand-in's dependence on
condensate loading alone was only ever an approximation, and why it is
`w`-dependent in reality.

### Method

Same as every prior phase: dump `CONDP` with its inputs (`CONDMU`, `WCU`, `TP`,
`PL`, `CDNC`, `DCW`) from inside `CONVECTIVE_MICROPHYSICS`, rerun, then port
against it. Do **not** trust the constants block — the active preset is lines
~265-295, and every value used so far (`bsort_enteff2 = 0.67`,
`cloudrvl_mstcnv = 10`, `cdnc_ocean_mc = 60`, `dd_evpeff_qp_scale = 0.001`,
`dd_detbyent = 0`, `max_dt_overshoot = 1`, `mc_fddrt = 0.5`) came from it, but
`fddet` showed that some are not in the source at all.

### 8a. Microphysics oracle — first round

`oracle_data/bomex_microphys.txt`, 630 records. Two pieces already verified
exactly, and one finding that invalidates the *form* of the current stand-in.

**The critical drop diameter is exact.** The active analytic form

```
DCW = max(0, log(((w·(p/1e5)^0.4) − 9.65)/(−9.8)) / (−600))
```

reproduces the dumped `DCW` on the first record checked by hand
(1.124894e-4 m at `w = 0.5`, `p = 94886`). Note `PL` inside
`CONVECTIVE_MICROPHYSICS` is in **Pa** despite the name — the caller passes
`PRES(L)`, not `pl(l)`. That is the second time this pair has bitten.

**The cloud/rain partition is exact, 630/630.** `PRECIPLIQ_GAMMA` splits the
condensate into a cloud mode of fixed capacity and a rain mode holding the rest:

```
lwc_cloud = min(twc, CDNC·ρw·(4/3)π·rvl³)      # ≈ 2.0-2.6e-4 kg/m³ here
lwc_rain  = twc − lwc_cloud
```

Precipitation occurs **iff** `lwc_rain > 0`, and that predicate agrees with the
oracle on every one of the 630 records. Of the rain-mode water, the
precipitated fraction runs 0.022-1.03, median 0.795, correlating with
`lwc_rain` (+0.58) far more than with `DCW` (+0.10) or `w` (+0.11).

**This invalidates the shape of the current stand-in.**
`giss_bsort.precipitation_fraction` is a smooth Weibull in the condensate
*mixing ratio*. The real scheme is a **hard threshold on liquid water
content**, at a capacity set by droplet number and size — nothing precipitates
at all below it. The stand-in's +0.97 correlation with `qc` was picking up the
threshold's shadow, not its mechanism, which is why it degraded at low
condensate.

Two consequences:

* The stand-in can be **improved immediately**, without the gamma integration,
  by replacing the Weibull with the exact threshold plus a fitted fraction of
  the rain-mode water. That is strictly better grounded than what is there now.
* The remaining unknown is narrow: what fraction of `lwc_rain` falls out, which
  is the gamma integration above `DCW`.

**Portability of the gamma integration:** `jax.scipy.special.gammainc` is
differentiable in both arguments, and `mu_rain = 2.5` is a constant, so only the
`x` argument varies. No obstacle.

### 8b. The intermediate stand-in does not pay off — skip to the gamma integration

Tested, and **rejected**. Replacing the Weibull with the exact cloud/rain
threshold plus a fitted rain-mode fraction makes the answer *worse*, not better:

| model | RMS error in precipitated fraction |
|---|---|
| current Weibull in `qc` (wrong mechanism) | **0.0277** |
| exact threshold + constant rain fraction (0.815) | 0.0417 |
| exact threshold + content-dependent rain fraction | 0.0358 |

The threshold itself is exact — it predicts precipitation-or-not on 630/630
records. What cannot be faked is the *rain-mode fraction*, which varies from
0.02 to 1.03 across the dataset. A constant throws that variation away, and a
one-parameter saturating fit in `lwc_rain` recovers only part of it, because the
true fraction is a gamma integral above a drop diameter that depends on the
updraft speed.

So the empirical Weibull, despite having demonstrably the wrong mechanism,
happens to fit this data better than a physically-motivated but incomplete
replacement. That is a useful reminder that "more physical" and "more accurate"
are not the same thing when the physics is only half-installed — and a reason
not to ship the intermediate step just because its derivation is nicer.

**Decision: leave `precipitation_fraction` alone and port the gamma integration
outright.** The threshold work is not wasted — it is the first half of that
port, already verified.

### 8c. The gamma integration, reduced to a spec

`PRECIPLIQ_GAMMA` (active copy, `MSTCNV.F90:7880-8131`) turns out to be short
once the diagnostic outputs are set aside. The whole precipitation calculation
is:

```
lwc_cloud      = min(twc, CDNC·ρw·(4/3)π·rvl³)          # verified 630/630
lwc_rain       = twc − lwc_cloud
lam_cloud      = ((ρw·(π/6)·CDNC·(μc+1)(μc+2)(μc+3)) / lwc_cloud)^(1/3)
lwc_detr_cloud = lwc_cloud · P(μc+4, lam_cloud·Dc)
lwc_detr_rain  = lwc_rain  · P(μr+4, lam_rain·Dc)
CONDP          = max(0, twc − lwc_detr_cloud − lwc_detr_rain)
```

with `μr = 2.5` constant and `Dc` the critical drop diameter from §8a.

**`incompleteGamma2` is the *lower* regularized incomplete gamma**, i.e.
`jax.scipy.special.gammainc`, not `gammaincc`. The naming invites the opposite
reading, and the sign of the whole scheme depends on it. The proof is the last
line: `lwc_detr_*` is subtracted from the total to give the precipitation, so it
must be the water in *small* drops — the ones whose fall speed is below the
updraft and which therefore stay with the plume. `detr` here means "detrained
with the cloud", not "removed as rain".

The recurrences in the Fortran (`gam_ingam_mup2 = mup1·gam_ingam_mup1 − expmx`,
and so on) are just the upward recurrence for the incomplete gamma, used to get
all four orders from one evaluation. A port can call `gammainc(μ+4, x)` directly
and skip them.

**Still to extract** (a few lines either side of what is transcribed above):
`mu_cloud` and its `mu_cld_max = 15` cap, and `lam_rain`/`nc_rain` from
`n0_rain = (n1−n2)/2·tanh((qr0·ρair − lwc_rain)/(4·qr0·ρair)) + (n1+n2)/2`
with `n1 = 9e9`, `n2 = 2e6`, `qr0 = 1e-4`.

Then: implement, and validate `CONDP` against `oracle_data/bomex_microphys.txt`
(630 records) exactly as every prior phase was validated.

## 9. Heating-rate comparison — diagnosis

Attempted, and **not yet presentable**. Recording why, because the remaining gap
is now precisely identified.

Against ModelE's `dth_mc` (SUBDD, 48 periods), our mean heating profile gives
**correlation +0.29** and a peak ratio of 1.34, with the peak about three levels
too high. ModelE cools the sub-cloud layers (−1.3 to −1.6 K/day at levels 0-3)
where we produce nothing.

### The sub-cloud source is now oracle-exact

`oracle_data/bomex_source.txt` dumps the actual draw
(`dmr(lll) = -mplume*fpi(...)`, `MSTCNV.F90:1573`). For the first plume it is
levels 0-6 — the *whole* sub-cloud layer, mass-weighted, with the two levels
above the BL top zeroed — summing exactly to the cloud-base mass. The earlier
placeholder put it in the four layers just below cloud base, which is a
different region entirely, and explains the missing sub-cloud cooling.

### But supplying it changed nothing, and the reason matters

Feeding the true source made no difference to the profile, because the
**environmental profile at the sub-cloud levels was a constant fill**: the
mass-budget dump only covers levels the plume reaches. A removal term is
`-removed_air · senv`, and its heating effect comes entirely from the *contrast*
between what is removed and what subsides in to replace it. With a uniform
`senv` that contrast is zero, so the term is inert.

This is worth stating as a general point about the tendency operator: in a
uniform environment the correct answer is exactly zero everywhere — nothing can
change if every layer is identical. A test that produces non-zero heating from a
uniform column has a mass source, not physics.

### What the heating plot actually needs

1. **The full environmental profile**, all levels, from the SUBDD output
   (`th`, `q`, `p_3d`) rather than from the plume dump — `single_column_harness`
   already reads these.
2. **Per-period plume matching.** There are 52 plume columns against 48 output
   periods, so some periods fire more than one plume and their tendencies must
   be summed before comparing.
3. Then the two known approximations remain: the downdraft is deposited where it
   forms rather than where it descends, and precipitation is the fitted
   stand-in.

Items 1 and 2 are wiring, not physics, and are what stand between here and a
publishable heating comparison.

### 9a. Items 1 and 2 done — heating comparison now tracks, but is not finished

Both wiring items are in place:

* **Step markers.** A saved counter incremented at `MSTCNV` entry is written
  into the mass-budget and source dumps. It resolves to **48 distinct steps**,
  exactly the number of SUBDD periods, with 9-18 level-records each — so several
  steps do fire more than one plume, and their tendencies are now summed before
  comparison.
* **Full environmental profile** read from the SUBDD output (`th`, `q`, `p_3d`)
  at every level, instead of the plume dump which only covers levels the plume
  reaches. `p_3d` is in **mb**; `senv = th/1000^kappa`.

Effect on the comparison against `dth_mc`:

| | before | after |
|---|---|---|
| mean-profile correlation | +0.29 | **+0.82** |
| peak ratio | 1.34 | **1.08** |

Per-period correlation is median +0.76, 10th percentile +0.47.

### What is still wrong, and where to look

The main heating layer (levels 4-8) now matches closely — within ~10% at the
peak. Two regions do not:

* **Levels 0-3 (sub-cloud).** ModelE cools by 1.3-1.6 K/day; we give roughly
  zero, and at level 3 we produce +2.9 where ModelE has −0.6. Wrong sign, so
  this is structural rather than a magnitude error. The prime suspect is the
  downdraft, which ModelE lands in exactly this layer after it descends, and
  which we currently deposit where it formed.
* **Levels 9-15.** ModelE keeps heating 1.2-3.9 K/day where ours decays to zero,
  so our detrainment is not reaching high enough.

Both are consistent with the two known placeholders, and the sub-cloud sign
error in particular is what the (already written, still unvalidated) downdraft
descent exists to fix.

`figures/heating_preliminary.png` is committed and labelled preliminary. It
should **not** be shown externally in this state.

### 9b. Off-by-one between ModelE level indices and SUBDD arrays — invalidates §9a

**The +0.82 correlation in §9a is not trustworthy and must be recomputed.**

ModelE's level index `l` is **1-based**; the SUBDD arrays are 0-based. The
harness used the dumped `l` directly to index `th`, `q`, `p_3d` and the derived
layer mass, so every environmental quantity was shifted one level relative to
the plume. It shows up cleanly in the layer mass: the derived `MA[L]` equals the
true `ma[L+1]`.

Consequence, on step 1's first plume:

| | active levels | downdraft | detrained |
|---|---|---|---|
| no shift (as in §9a) | **1** | 0.000 | 51.78 |
| `l → l-1` | **7** | 9.84 | 124.79 |
| ModelE (step 1, *two* plumes) | 14 records | 18.19 | 171.87 |

Unshifted, the plume dies immediately and dumps its entire mass at cloud base.
That still produces heating in roughly the right *place*, which is why the
correlation looked reasonable — the §9a number was obtained from plumes that
were not actually ascending.

With the shift the numbers line up: ModelE's 14 records for that step are two
plumes of seven levels each, and our single plume gives seven active levels with
about half the total downdraft mass. That consistency is the real check.

This also explains why wiring the descending downdraft changed nothing: the
plume was producing **no downdraft mass at all** to descend.

**Standing lesson for this port, now the second instance** (after the `dcl`
off-by-one): a diagnostic that looks plausible is not evidence. The correlation
improved for a reason unrelated to the physics being right, and only a
structural check — layer masses against layer masses — exposed it.

Next: redo §9a with the shift applied, then re-test the descending downdraft
against the sub-cloud sign error.

### 9c. Heating comparison, corrected for the off-by-one

Redone with `l → l-1` applied to every SUBDD-indexed quantity. Plumes now
genuinely ascend — mean 10.2 active levels, against 1 before.

| | correlation | peak ratio |
|---|---|---|
| §9a (invalid, plumes not ascending) | +0.82 | 1.08 |
| corrected, downdraft folded in place | **+0.893** | **1.62** |
| corrected, downdraft descending | **+0.896** | **1.58** |

Correlation is genuinely better than the invalid number. The peak ratio is
worse, and that is the honest direction: with the plume actually rising it
deposits far more heat, and we now overshoot ModelE by ~60%.

**The descending downdraft is not the fix for the sub-cloud layer.** It moves
the correlation by 0.003 and the peak by 0.04. Levels 0-3 still show roughly
zero or the wrong sign where ModelE cools by 1.3-1.6 K/day. The hypothesis in
§9a — that misplaced downdraft air explained the sub-cloud error — is therefore
**not supported** once the plume is actually ascending.

Two candidates remain for the sub-cloud discrepancy, in order of suspicion:

1. **The downdraft is not cold enough.** Its cooling comes from evaporating
   precipitation, and the precipitation supply is the fitted stand-in, which
   §8b showed has the wrong mechanism. A downdraft that arrives too warm
   deposits without cooling — consistent with what is seen.
2. **The 60% overshoot aloft** may itself be the sub-cloud story: too much mass
   detraining high means too little returning low.

Both point back at the microphysics rather than at the downdraft, which
reverses the priority set in §7b. The next diagnostic should be the *moisture*
tendency `dq_mc`, not more heating work: it isolates the precipitation term far
more directly than heating does, since the heating is dominated by subsidence
while the moistening is dominated by what the plume actually sheds.

### 9d. `dq_mc` — the two errors are one error

| | correlation | peak ratio |
|---|---|---|
| `dq_mc`, downdraft folded | +0.836 | **0.70** |
| `dq_mc`, downdraft descending | +0.849 | **0.70** |

Again the descending downdraft barely moves it (+0.013), confirming §9c: the
downdraft is not what is wrong.

**The diagnostic signal is the sign error at levels 7-9.** ModelE *moistens*
there (+0.0005 to +0.002); we *dry* (−0.0014 to −0.0059). Combined with the
heating being 1.58x too strong in the same layer, both point at a single cause:

> the compensating subsidence is too strong relative to the detrainment
> deposition in the mid-cloud layer.

Subsidence warms and dries; detrainment of plume air cools and moistens. Too
much of the first relative to the second gives exactly this pair of symptoms —
excess heating *and* drying where ModelE has moistening. That is one error
showing in two diagnostics, not two independent problems.

This matters because **the plume mass flux itself matches ModelE to 0.06%**
(§7). So the error is not in the plume; it is in how the tendency operator turns
that flux into environmental tendencies.

Two candidates, in order:

1. **The environment profile is the post-convection state.** The SUBDD `th`/`q`
   are written after physics has already modified the column, so the contrast
   driving both the subsidence and the deposition terms is taken against the
   wrong profile. This exact bias was found and corrected once before in this
   project, for the single-column harness; it appears to have re-entered here
   through the SUBDD route. `single_column_harness.run_column` already has the
   pre-convection correction (subtracting `dth_mc`/`dq_mc` times the step) and
   should be reused rather than re-deriving the environment.
2. The detrainment deposition may be under-weighted relative to the interface
   flux in `bsort_environment_tendencies` — but this is less likely given the
   mass budget closes.

Next step is (1): drive the comparison from `single_column_harness`'s
pre-convection state instead of raw SUBDD fields.

### 9e. Pre-convection state applied — large gain, and the sign error localises

The SUBDD `th`/`q` are the post-convection state. Recovering the pre-convection
profile (subtract one step of `dth_mc`/`dq_mc`, as
`single_column_harness.run_column` already does) gives:

| | dth corr | dth peak | dq corr | dq peak |
|---|---|---|---|---|
| post-convection | +0.893 | 1.62 | +0.836 | 0.70 |
| **pre-convection** | **+0.925** | 1.58 | **+0.938** | **0.96** |

Moisture is now essentially right in magnitude (0.96) and well correlated. This
confirms §9d's diagnosis: the contrast driving both terms was being taken
against the wrong profile.

**Lesson worth keeping:** this correction already existed in the harness and was
bypassed by building a fresh path from raw SUBDD fields. The bug was not new — it
was re-imported.

### The remaining sign error, and a concrete suspect

Levels 7-9 still have ModelE moistening (+0.0005 to +0.002) where we dry
(−0.0015 to −0.005). For plume 1, cloud base is level 9 (0-based), and the
source draw covers levels 0-6. So levels 7-8 sit in a gap: no source removal, no
detrainment, only subsidence — hence drying.

**Suspect: the boundary-layer top fed to the downdraft is wrong.** The harness
sets it to the last non-zero source level (6), but ModelE uses
`max(lcl-1, dcl)` — with `lcl` the cloud base, `lcl-1` is level 8 (0-based).
The downdraft's forced detrainment should therefore reach levels 7-8, which is
exactly the gap where the sign is wrong.

That would also explain why wiring the descending downdraft looked ineffective
in §9c and §9d: it was being told to dump its air two levels too low, below the
layer that needed it.

Next: set `boundary_layer_top = cloud_base - 1` (falling back to `dcl` where
that is higher) and re-test. This is a one-line change with a sharp prediction —
the levels 7-8 sign should flip.

### 9f. The boundary-layer hypothesis is falsified; the runaway is the real mechanism

Setting `boundary_layer_top = cloud_base - 1` changed **nothing** — the numbers
are bit-identical to using the source top. The §9e hypothesis is wrong.

The reason is visible in the downdraft mass profile for step 1's first plume:

| level | ModelE `ddm` | ours |
|---|---|---|
| 13 | 9.86 | 8.72 |
| 11 | 17.10 | 2.25 |
| 8 | 18.36 | 3.39 |
| 6 | 19.06 | 0.21 |
| 2 | 1.37 | ~0 |

**ModelE's downdraft grows as it descends**, roughly doubling from formation to
the boundary layer, and only sheds its mass in the lowest few levels. Ours
collapses within two levels and never arrives. Changing where forced detrainment
begins is irrelevant when there is no downdraft left to detrain.

### The runaway

Our evaporation at level 13 is 0.0014 against ModelE's 0.0060. That is enough to
start a feedback:

> too little precipitation to evaporate → downdraft not cooled → tests
> positively buoyant → sheds 75% of its mass → less mass to evaporate into →
> less cooling still.

Each step makes the next worse, which is why the collapse is so abrupt. ModelE
avoids it because its downdraft stays negatively buoyant the whole way down.

The efficiency factor is not the limiter — for these values
`min(1, (ma·kg2mb/30)·(prcp_mixrat/1e-3)^0.6)` saturates at 1. The limiter is the
**precipitation supply itself**, which is the fitted stand-in.

### Consequence for the plan

This is now the third independent line of evidence pointing at
`CONVECTIVE_MICROPHYSICS`:

* §8b — the stand-in's mechanism is demonstrably wrong (threshold, not Weibull);
* §7b — the downdraft is acutely sensitive to the precipitation supply;
* §9f — that sensitivity is a *runaway*, not a gradual degradation.

It also means **the downdraft port cannot be validated at all** until
precipitation is real: every test of it so far has been a test of the stand-in.
The verdict in §9c that "the downdraft is not the fix" should be read narrowly —
the downdraft as currently *fed* is not the fix. Whether the port is correct
remains unknown.

Current standing with the pre-convection state and the downdraft wired:
`dth` +0.931 / peak 1.53, `dq` +0.946 / peak 0.96.

## 10. `PRECIPLIQ_GAMMA` ported and exact — but it does not fix the downdraft

`giss_microphysics.py`. Validated against the 630-record oracle:

| quantity | agreement |
|---|---|
| critical drop diameter `Dc` | **630/630**, max rel 5.8e-7 |
| precipitated water | **570/570**, median rel 1.9e-6 |
| precipitates-or-not | **630/630** |

The scheme is a two-mode gamma distribution: a cloud mode of fixed capacity
`CDNC·rho_w·(4/3)pi·rvl^3`, a rain mode holding the excess, and drops falling
faster than the updraft are lost. `incompleteGamma2` is the **lower**
regularized incomplete gamma, confirmed by the sign of the result.

A third fractional-power NaN gradient turned up and was fixed (`x^(1/3)` and
`x^(1/4)` at zero condensate) — the same class as the two in `giss_bsort` and
`precipitation_fraction`. Worth treating as a standing hazard in this port:
**any non-integer power whose base can legitimately be zero.**

### The result that matters: precipitation was necessary but not sufficient

Substituting the real microphysics for the stand-in raises the precipitation
supply (0.0050 -> 0.0079) and the evaporation into the downdraft (0.0049 ->
0.0075), a 50% increase — and leaves the downdraft mass profile **bit-identical**
(8.72, 2.25, 3.39, 0.21 at levels 13/11/8/6, against ModelE's 9.86, 17.10,
18.36, 19.06).

The reason is that detrainment is a *step function* of the buoyancy sign, not of
its magnitude: 0.75 when buoyant, 0.5 in the boundary layer, 0 otherwise. More
cooling that does not flip the sign changes nothing at all. ModelE's downdraft
evaporates roughly 3x more than ours even with the correct microphysics.

So §9f's diagnosis was right about the mechanism (a runaway) but wrong about the
cause being *only* the precipitation supply. Something in the downdraft port
itself keeps it too warm. Candidates, untested:

* `environment_condensate` is passed as zero in the harness, where ModelE uses
  `qcl + qci`. That *raises* our environmental virtual temperature relative to
  ModelE's, which should make our downdraft *less* likely to test buoyant — so
  it cannot explain the discrepancy and may be masking it.
* The evaporation efficiency factor saturates at 1 here, so it is not limiting.
* The precipitation is being re-split `fddrt` each level in the port; ModelE's
  `prcp_d` stays roughly constant down the column while ours decays. Worth
  checking whether the split is being applied to the right quantity.

The last of these is the most concrete and is where to look next.

## 11. The port now has no fitted parameters

Wiring `giss_microphysics` into the plume's own condensate budget removed the
last calibrated term. `precipitation_fraction` is deleted; every value in the
scheme is now derived from ModelE rather than fitted to it.

### One missing piece cost 3 orders of magnitude, and the oracle caught it

The first wiring attempt regressed the plume from 0.06% to **53.6%** median mass
error, because ModelE scales `CONDP` down by `min(1, ma/cond_repart_dmscale)` —
only part of the partition is realised over the distance the plume ascends in
one layer. Confirming it took one arithmetic check: a dumped factor of 0.2035
times the reference mass 509.86 gives 103.76, exactly the dumped layer mass.

With the rescale, the real microphysics **beats** the stand-in it replaced:

| | fitted stand-in | real microphysics |
|---|---|---|
| active levels | 534/536 | **535/536** |
| plume mass | 0.06% | 0.072% |
| updraft `w` | 0.14% | **0.093%** |
| detrained mass | 0.12% | **0.017%** |

### The downdraft port is validated after all

Driven with the oracle's own inputs rather than our plume's, the descent
reproduces ModelE essentially exactly — mass 9.859/16.546/17.101/17.574/17.986/
18.361/18.714/19.062 against the oracle's identical values, and detrainment
9.531/4.941/2.561 exact. **The port was never wrong**; it was being fed a
downdraft source that our plume generated differently under a reconstructed
environment. §7b's "cannot be validated" and §9c's "not the fix" were both
consequences of the harness, not the physics.

### Tendency standing

| | correlation | peak ratio |
|---|---|---|
| `dth_mc` | **+0.946** | 1.42 |
| `dq_mc` | **+0.945** | **0.96** |

### What is still wrong

Unchanged by the microphysics, which is itself informative — these are not
precipitation problems:

* **Levels 0-3**: ModelE cools 1.3-1.6 K/day, we produce ~0.
* **Levels 7-9**: ModelE moistens, we dry.
* **Heating 42% too strong** while moisture is right to 4%.

Heating too strong with moisture correct points away from the mass flux (which
would move both) and toward the *heat* carried by detrained air — i.e. the
plume's temperature, not its mass. The plume's own `w` and detrainment now match
to 0.1%, so the next place to look is the heat content of what it deposits,
and specifically whether the latent heat released on re-saturation is being
double-counted against what ModelE already accounts for in `CDHEAT`.

## 12. The sub-cloud gap is environmental precipitation evaporation

The `CDHEAT` double-counting hypothesis was **wrong**: `CDHEAT` is a pure
diagnostic (summed at `MSTCNV.F90:5645`, never fed back), and §5b had already
shown plume heat matching 484/484.

First, the plume is now confirmed correct in *every* output:

| plume output | median error vs oracle |
|---|---|
| mass | 0.072% |
| updraft `w` | 0.093% |
| detrained mass | 0.017% |
| **downdraft source** | **0.099%** |
| **entrained air** | **0.110%** |

The last two had never been checked. With all five matching to ~0.1%, the
tendency discrepancy cannot be coming from the plume.

### The missing term

ModelE splits precipitation by `fddrt`: half falls through the downdraft, half
through the **environment**, where it evaporates directly into the layer
(`dsm_evp`, `dqm_evp`, applied at `MSTCNV.F90:4743`). This module's docstring
listed it as deliberately out of scope. Dumping it shows it is not optional:

| level | ModelE `dth_mc` | environmental-evaporation contribution |
|---|---|---|
| 0 | −1.627 | **−1.133** |
| 1 | −1.296 | **−0.920** |
| 2 | −1.313 | **−0.687** |
| 3 | −0.566 | **−0.447** |
| 4 | +6.132 | −0.188 |
| ≥6 | — | 0.000 |

It accounts for **70-90% of exactly the sub-cloud cooling we are missing**, and
is identically zero above level 5.

### What it does not explain

The levels 7-9 sign error and the 42% heating excess aloft are untouched by it.
Those remain open, and are now the only unexplained discrepancies. Since every
plume output matches to 0.1% and the environmental evaporation is confined
below level 5, the remaining error is in the **tendency operator's treatment of
the mid-cloud layer** — most likely the balance between the interface flux and
the detrainment deposition, which §9d already identified from the sign pattern.

### To port it

```
if prcp_e > 0 and menv > 0:
    smenv, qmenv = sm(l)/ma*menv, qm(l)/ma*menv
    dqevp = get_dq_evap(smenv, qmenv, plk, menv, lhx, pres, evap_max=prcp_e)
    dqm_evp(l) = dqevp
    dsm_evp(l) = -(slh*dqevp + heat1(l))/plk
```

Everything is available except **`menv`** — the environmental air mass taking
part — which, like `fddet` and `etal` before it, is used at `MSTCNV.F90:4603`
but never assigned in the file. It needs one more dump round. `heat1` is the
phase-change correction from `MSTCNV.F90:4226`, zero for all-liquid BOMEX.
