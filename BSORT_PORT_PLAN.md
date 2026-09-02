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

## 13. Correction: `fddet` and `menv` are both properly defined

**Retracting a claim made twice in this document and once in a commit message.**
§7b and §9f state that `fddet` "has no assignment left in `MSTCNV.F90`" and that
its value had to be recovered by measurement. That is wrong.

`grep` is case-sensitive; **Fortran is not.** Searching for the lowercase spelling
used at the call sites missed the uppercase declarations:

* `FDDET = 0.25d0` is a **`real*8, parameter` at `MSTCNV.F90:42`**, with a
  docstring comment. The measured 0.25 was correct, but it was never a mystery —
  only a badly-searched constant.
* `MENV = FEVAP(L)*MA(L)` at `MSTCNV.F90:4587`, likewise properly assigned, with
  `!@var MENV air mass available for re-evaporation of precip into the
  environment` two lines above its declaration.

Before finding this I had checked that `menv` was deterministic across two runs
(it is, bit-identical) and was preparing to report an uninitialized-variable bug
in ModelE. That would have been wrong, and embarrassing to have raised.

**Standing rule for this port: search Fortran case-insensitively (`grep -i`).**
The codebase declares in upper case and calls in lower case, so a case-sensitive
search reliably finds the uses and hides the definitions — the exact pattern
that produces false "this is never defined" conclusions.

### The environmental evaporation chain, fully specified

With `geometric_fevap = .true.` (the default):

```
prwtsum = 1e-30;  prwtmcsum = 0;  fevap_extra = 0
for l = lmax down to 1:
    if l >= lcl:
        mcfc        = 0.5*(mcfrac(l-1) + mcfrac(l))
        fevap_extra = mcfc*(1 - mcfc)*geometric_fevapfac
        prwtsum    += condpr(l)
        prwtmcsum  += condpr(l)*mcfc
    else:
        mcfc = 0
    fevap(l) = min(1, max(0, prwtmcsum/prwtsum - mcfc) + fevap_extra)

menv(l) = fevap(l)*ma(l)
```

then, where `prcp_e > 0` and `menv > 0`:

```
dqm_evp(l) = get_dq_evap(sm(l)/ma*menv, qm(l)/ma*menv, plk, menv, pres,
                         evap_max = prcp_e)
dsm_evp(l) = -(slh*dqm_evp(l) + heat1(l))/plk
```

`geometric_fevapfac = 0` in the active preset (`MSTCNV.F90:290`), so
`fevap_extra` vanishes and `fevap` reduces to the precipitation-weighted
convective-fraction excess. `heat1` is zero for all-liquid BOMEX (verified in
the dump). `condpr` and `mcfrac` are both available: the first from
`giss_microphysics`, the second dumped.

Nothing here is unknown any more — this is ready to port.

## 14. Environmental evaporation implemented — and a stream I had conflated

`downdraft_descent` now computes `dsm_evp`/`dqm_evp` inside the descent scan,
where ModelE computes them (`MSTCNV.F90:4587-4643`), returning them as
`environment_heat` / `environment_water`.

### Two precipitation streams, not one

The first attempt produced identically zero, which exposed a distinction I had
collapsed. ModelE carries **two separate precipitation quantities**:

* **`wmdnl`** — condensate the sorted blends carry *into the downdraft*. This,
  and only this, feeds the descending flux: `prcp_d` starts at zero
  (`MSTCNV.F90:4277`) and the loop adds nothing else to it.
* **`condpr`** — precipitation *produced by the plume*, from the microphysics.
  It never enters the downdraft flux at all. Its only roles are weighting how
  much rain evaporates into the clear air, and the phase-change term `heat1`.

Passing the flux where the produced precipitation belonged made the weighting
degenerate: with a single stream, the precipitation-weighted mean convective
fraction equals the local one, their difference is zero, and no air participates.
They are now separate arguments.

### Why the term is a sub-cloud effect

`fevap(l) = max(0, prwtmcsum/prwtsum - mcfc)`. Above cloud base `mcfc` is the
local convective fraction and largely cancels the weighted mean, so little
evaporates. **Below cloud base `mcfc` is zero**, so `fevap` becomes the full
precipitation-weighted mean convective fraction and the term switches on.

Checked against the oracle: the weighted mean is 0.016, and
`menv = 0.016 x 103.76 = 1.66` — exactly the dumped `menv`. That is why the
missing cooling was confined to levels 0-3.

### Remaining to close the loop

Feed `condpr` from `giss_microphysics` into `produced_precipitation`, and add
the returned `environment_heat`/`environment_water` to the tendency. Both are
plumbing; the physics is in place and the constants are all resolved
(`geometric_fevapfac = 0`, `heat1 = 0` for all-liquid).

Suite: 307 passed, 3 skipped.

## 15. Wired — heating correlation 0.977, peak ratio 1.15

Feeding `condpr` from `giss_microphysics` and adding the returned environment
tendencies:

| | dth corr | dth peak | dq corr | dq peak |
|---|---|---|---|---|
| §11 standing | +0.946 | 1.42 | +0.945 | 0.96 |
| correct `wmdnl` stream only | +0.968 | 1.19 | **+0.980** | 0.96 |
| **+ environmental evaporation** | **+0.977** | **1.15** | **+0.980** | **0.95** |

Two separate gains. Most of it came from feeding the descent the *right*
precipitation stream — `downdraft_condensate` (`wmdnl`) rather than the total
precipitation — which alone took heating from 1.42 to 1.19 and moisture
correlation from 0.945 to 0.980. The environmental evaporation added the rest.

### The sub-cloud prediction was right in sign, short in magnitude

Predicted: levels 0-3 move from ~0 to roughly ModelE's −1.3 to −1.6 K/day.

| level | ModelE | before | after |
|---|---|---|---|
| 0 | −1.627 | −0.061 | **−0.427** |
| 1 | −1.296 | −0.001 | **−0.572** |
| 2 | −1.313 | +0.118 | **−0.437** |
| 3 | −0.566 | +0.815 | +0.363 |

The sign is now right where it was wrong or absent, but the magnitude is about a
third of ModelE's. So environmental evaporation is *a* cause of the sub-cloud
cooling, not the whole of it — the earlier estimate that it accounted for 70-90%
was based on ModelE's own dumped term, and our reproduction of that term is
evidently weaker than ModelE's. The likely reason is `mcfrac`: the harness
passes a constant 0.02 where ModelE computes a profile, and `fevap` is built
entirely from precipitation-weighted `mcfrac`.

### Still open

* Levels 7-9 remain the wrong sign in moisture (ours −0.0005 to −0.0012 against
  ModelE +0.0005 to +0.0021), though roughly four times smaller than before.
* Heating remains 15% strong.

Both are now small enough that `mcfrac` — currently a hand-set constant in the
harness — is the most likely single remaining cause. It is dumped, so this is a
harness fix rather than a physics one.

## 16. The `mcfrac` hypothesis is wrong

§15 predicted that replacing the harness's constant `mcfrac = 0.02` with the
dumped profile would deepen the sub-cloud cooling toward ModelE's. It does the
opposite:

| | dth corr | dth peak | dq corr |
|---|---|---|---|
| `mcfrac = 0.02` | **+0.977** | **1.15** | **+0.980** |
| `mcfrac` from the oracle | +0.972 | 1.18 | +0.978 |

Sub-cloud cooling at levels 0-2 goes from −0.43/−0.57/−0.44 to
−0.31/−0.21/−0.03 — weaker, not stronger.

The mechanism is clear in hindsight. Below cloud base `mcfc = 0`, so `fevap`
reduces to the precipitation-weighted mean convective fraction. A constant 0.02
makes that mean exactly 0.02; the real profile averages closer to 0.01, so it
evaporates *less*. ModelE's own `fevap` is 0.016, between the two.

So the constant is not the cause, and it is closer to right than the real
profile by accident. The remaining gap must be in how the weighted mean is
accumulated — most likely that our `condpr` weighting differs from ModelE's, or
that the `l >= lcl` gate should use the lifting condensation level rather than
the plume's cloud base, which the harness currently conflates.

**Best standing configuration** (constant `mcfrac`, everything else real):

| | correlation | peak ratio |
|---|---|---|
| `dth_mc` | **+0.977** | **1.15** |
| `dq_mc` | **+0.980** | **0.95** |

That is the number to quote, with the caveat that one harness input is a
constant that happens to be favourable rather than a derived profile.

## 17. `condpr` traced to a harness bug; the result is robust

### The bug

The harness was computing `condpr` from `detrained_condensate +
downdraft_condensate` — the condensate *leaving* the plume — because
`PlumeAscent` did not expose the condensate *in* it. That made our `CONDMU`
3-5x smaller than ModelE's (ratios 0.22-0.30), which put it below the cloud-mode
capacity so that `CONDP` came out essentially zero.

`plume_ascent` now exposes both `plume_condensate` and `precipitation`
(the `condpr` it already computed internally), so the harness no longer
recomputes anything. `CONDMU` ratios move to 0.59-1.03.

### Why fixing it changed nothing, and why that matters

With `mcfrac` held constant the precipitation-weighted mean convective fraction
*equals that constant regardless of the weights*, so `condpr` cannot affect
`fevap` at all. The §16 test of "real `mcfrac`" was therefore run against a
broken `condpr`, and the two inputs only mean anything together.

Running both correctly:

| | dth corr | dth peak | dq corr | dq peak |
|---|---|---|---|---|
| constant `mcfrac`, plume `condpr` | +0.977 | 1.15 | +0.980 | 0.95 |
| **real `mcfrac` + real `condpr`** | **+0.978** | **1.16** | **+0.977** | **0.96** |

Essentially identical. That resolves the §16 worry that the constant was
"favourable by accident": it is not favourable, the answer is simply **robust**
to this input. Sub-cloud level 0 does improve, −0.427 to −0.644 against ModelE's
−1.627.

**Standing result, every input derived:**

| | correlation | peak ratio |
|---|---|---|
| `dth_mc` | **+0.978** | **1.16** |
| `dq_mc` | **+0.977** | **0.96** |

### What is left

* Sub-cloud cooling reaches about 40% of ModelE's.
* Levels 7-9 remain the wrong sign in moisture, though ~4x smaller than before.
* Heating 16% strong.

`figures/tendencies.png` shows both fields for three periods.

## 18. Continuity oracle: the remaining gap is the downdraft, and it is isolated

`oracle_data/bomex_continuity.txt` dumps `cm`, `dm`, `dmr` and their heat/water
counterparts from inside `apply_continuity_tendencies` — the one component never
checked against its own oracle.

### What it shows

| level | `dm` ModelE | `dm` ours | `dmr` ModelE | `dmr` ours |
|---|---|---|---|---|
| 0-6 (sub-cloud) | 0.42 → 10.84 | **~0** | −11.64 → −12.04 | −11.64 |
| 8 | 17.19 | 17.21 | −11.86 | −11.46 |
| 9 | 48.82 | 48.89 | −32.11 | −31.69 |

`dmr` matches, and the plume's own `dm` matches at levels 8-9. **The entire
discrepancy is `dm` in the sub-cloud layers**, which is the downdraft's
detrainment: ModelE's `dm` at level 6 is 10.84, and the downdraft dump shows
`detr = 9.53` there. Our downdraft never arrives.

### A harness bug found on the way

The harness was passing SUBDD-derived `exner`/`pressure` where every other
plume input used ModelE's own dumped values. In the harness configuration the
plume's detrained mass diverged from level 10 onward (31.28 against 10.43),
while the standalone configuration matched at 0.017%. Using ModelE's `plk`/
`pres` at plume levels restores it to **0.067%**.

That is worth noting as a pattern: the standalone validations and the harness
were not running the same configuration, so a component could be "validated" and
still be wrong where it was actually used. The plume is now verified *in the
configuration the tendency comparison uses*.

Standing after the fix: `dth` **+0.979 / 1.16**, `dq` **+0.981 / 0.95**.

### The downdraft collapse is now isolated

ModelE's downdraft detrains **nothing** between its formation level and the
boundary layer, then sheds 9.53/4.94/2.56 at levels 6/5/4. Ours detrains 10.19
at the formation level itself — the buoyant branch firing where ModelE's does
not.

Ruled out by direct test:

* the descent port itself — driven with oracle `ddr`/`smdnl`/`qmdnl` it
  reproduces ModelE's mass and detrainment essentially exactly;
* the descent's auxiliary inputs — substituting real `qcl+qci`, real `mcfrac`
  and the `lcl`-based boundary-layer top changes the result *not at all*;
* the plume's downdraft source, now at 0.1% in the standalone configuration.

What remains untested is the downdraft source **in the harness configuration**,
where `ddr` is 8.70 against ModelE's 9.86 — 12% low. Given §7b established this
system is a runaway, 12% at the formation level is not obviously too small to
explain a collapse. That is the next thing to measure.

## 19. Root cause: five branch flips, amplified by the runaway

Measured in the harness configuration, every plume output now matches:

| | median | 90th |
|---|---|---|
| plume mass | 0.077% | 0.41% |
| plume detrained | 0.067% | 0.38% |
| downdraft source mass / heat / water | 0.109% | 1.63% |

The 12% figure in §18 was from before the `plk`/`pres` fix and no longer holds.

### The actual failure

Over 268 levels where either model routes blends to the downdraft:

| | count |
|---|---|
| both fire | **263** |
| ModelE only (we miss) | **5** |
| ours only | 3 |
| total downdraft mass, ours / ModelE | **0.994** |

In aggregate the downdraft source is right to 0.6%. But the misses are not
distributed harmlessly. For step 1's first plume, level 13 is one of the five:
ModelE routes 6.05 kg/m² to the downdraft there and we route none. That is 38%
of that column's downdraft, and it is at a *formation* level.

Because the downdraft is a runaway (§7b, §9f), starting with 9.86 instead of
15.9 is not a 38% error in the result — the smaller draught cools less, tests
buoyant, sheds 75% of itself, and is gone within two levels. ModelE's grows to
19.1. **A 2% error in the branch decision produces a total loss of the
sub-cloud detrainment.**

### This is the §6 caveat coming true

The port's differentiability note recorded that the three-way sort is a hard
threshold and that "agreement is not *structurally* guaranteed for blends
sitting within ~4e-7 of a threshold" — noted then as a theoretical limitation.
It is now the leading-order error in the tendency comparison.

The standalone check found 1572/1572 fates correct *when fed the oracle's blend
properties*. In the harness the plume self-propagates, so small differences
accumulate until a marginal blend lands on the other side of the threshold.

### What this does and does not mean

* It is **not** a porting error. Every formula matches; the sort reproduces
  ModelE exactly given identical inputs.
* It is a genuine **conditioning** property of the scheme: buoyancy sorting with
  three discrete outcomes and a runaway downstream amplifies small state
  differences into large tendency differences.
* It bounds what level-by-level agreement can be expected from *any* independent
  implementation, including a Fortran one compiled differently.

The remaining sub-cloud gap should therefore be quoted as a sensitivity of the
scheme, not as an outstanding bug — while noting that a smoother sort (§6's
sigmoid option, currently off) would remove the amplification at the cost of
departing from ModelE.

## 20. Correction: there is one error, not two

§19 closed by saying the mid-level heating excess was "not downdraft-related",
reasoning that ModelE's downdraft does not detrain at levels 7-11. That was the
wrong test — the question is not whether *ModelE's* downdraft detrains there,
but whether *ours* does. It does.

Splitting the detrainment by source, summed over all 48 steps:

| level | plume, ModelE / ours | downdraft, ModelE / ours |
|---|---|---|
| 4 | 0.0 / 0.0 | **172.0 / 27.6** |
| 5 | 0.0 / 0.0 | **234.6 / 33.3** |
| 6 | 703.8 / 703.8 | 70.3 / 47.6 |
| 9 | 593.1 / 593.3 | 155.6 / 250.8 |
| 10 | 245.5 / 253.4 | **19.2 / 143.0** |
| 11 | 183.8 / 203.3 | 39.9 / 104.8 |

**The plume's detrainment matches at every level.** The downdraft's is displaced
upward: roughly seven times too little at levels 4-5, seven times too much at
level 10.

The heat it carries is also right — `dsm/dm`, the specific heat of the detrained
air, agrees with ModelE to four or five significant figures at every level
(41.4840 vs 41.4891 at level 6, 41.6256 vs 41.6286 at level 9). So the air is at
the correct temperature; it is simply deposited in the wrong place.

### One mechanism explains both symptoms

* Missing sub-cloud cooling — the downdraft never arrives.
* Excess mid-level heating — because it unloaded there instead.

Both follow from the collapse traced in §19: five branch flips at formation
levels, amplified by the runaway. This is a single defect with two faces, not
two independent problems, and it means the remaining discrepancy is smaller and
better understood than the two open items in §19 suggested.

### What would actually close it

Nothing in the port is wrong. Closing the gap requires the sorted blends to
land on ModelE's side of the buoyancy threshold on those five levels, which
needs either

* **higher input precision** — the dumps carry ~7 significant figures and the
  marginal blends sit within ~4e-7 of the threshold, so some of the five flips
  may be measurement artefact rather than genuine divergence; or
* **a smoother sort** — the sigmoid option, which removes the amplification but
  departs from ModelE and should stay off by default.

The first is a measurement change and is worth doing before concluding anything
about the second.

## 21. Precision check: the port is bit-exact; one coupling term is not

Re-ran every dump at full double precision (`es24.16` instead of `es14.6`).

### The branch flips are real

Bit-identical results: the same 263 shared downdraft levels, the same **5**
ModelE-only and **3** ours-only, the same 0.994 mass ratio, the same 0.0779% /
0.0689% plume errors to four significant figures. **The flips are genuine
divergence, not measurement artefact.** The sigmoid question in §20 is therefore
a real question rather than a workaround for bad instrumentation — though see
below for why it is still not the next step.

### The formulas are exact, which the old dumps were hiding

Every per-level component, given ModelE's own state at full precision:

| component | old (7 sig figs) | **double precision** |
|---|---|---|
| entrainment closure `ent` | 2.39e-06 | **0.0** |
| detrainment `det` | 4.15e-07 | **0.0** |
| updraft set-aside | 8.31e-07 | **3.7e-16** |
| sort: plume out | 8.23e-07 | **2.4e-15** |
| sort: detrained | 6.19e-07 | **3.8e-16** |
| sort: downdraft | 7.43e-07 | **4.4e-16** |
| vertical velocity `wcu` | 6.73e-07 | **0.0** |
| inter-level re-saturation, heat | ~1e-06 | **3.6e-16** |
| inter-level re-saturation, vapour | ~1e-06 | **4.0e-15** |

**The port reproduces ModelE to machine precision.** Every "1e-6 agreement"
reported earlier in this document was the dump's precision, not the port's
error. That is a materially stronger claim than anything previously recorded
here, and it should replace the older numbers when this work is described.

### The one term that is not exact

Applying `precipitate` *in situ* — from our own re-saturated condensate rather
than from ModelE's dumped `CONDMU` — leaves a **2.5% median** error in the
removed condensate, against 1.9e-06 when fed ModelE's `CONDMU` directly. So the
routine is right and the **coupling into it** is not.

Two candidate causes tested and rejected:

* `wcupass`, which ModelE extrapolates as `1.5*w(l-1) - 0.5*w(l-2)`
  (`MSTCNV.F90:1874`) where we pass `w(l-1)`. Using the extrapolation makes it
  *worse*: 3.19% against 2.46%.
* the density in `CONDMU = (wmp/mplume)*rho0(l)` (`MSTCNV.F90:1772`), which is
  the environmental `rho0 = ma/delz` rather than a plume density. Using it:
  2.68% against 2.46%.

So the conversion into volumetric units is not the discrepancy either. This is
where to resume.

### Why this matters more than the 2.5% suggests

A 2.5% error in the condensate removed per level feeds the next level's
buoyancy, and the sort is a hard threshold. Five blends out of 268 land on the
wrong side, the downdraft collapses, and the deposition moves from levels 4-6 to
levels 9-13 — the single defect of §20. **Closing the 2.5% would very likely
close the whole remaining gap**, and it requires no change to the physics, which
makes it strictly preferable to the sigmoid.

## 22. Chasing the 2.5%: one real fix found, the rest still open

### Found and fixed: the volumetric conversion density

`CONDMU = (wmp/mplume)*rho0(l)` (`MSTCNV.F90:1772`) uses the **layer reference
density**, not one derived from the plume's own temperature. With
`rho0 = ma/delz`, our volumetric condensate reproduces the dumped `CONDMU`
**exactly** — ratio 1.00000 at both the 10th and 90th percentile over 484
levels. The plume-temperature density is 0.08% off.

`plume_ascent` now uses `ma/delz`. Effect:

| | before | after |
|---|---|---|
| plume mass | 0.0779% | **0.0733%** |
| plume detrained | 0.0689% | **0.0616%** |
| downdraft branch (both / ModelE-only / ours-only) | 263 / 5 / 3 | **264 / 4 / 2** |
| downdraft mass ratio | 0.99411 | **1.00189** |

**One of the five branch flips is fixed.** The tendencies barely move
(`dth` +0.9793/1.163, `dq` +0.9813/0.947) because the corrected column was not
one of the dominant ones — but the defect count is genuinely down.

### Rejected by direct test

* **`wcupass`.** ModelE extrapolates `1.5*w(l-1) - 0.5*w(l-2)`
  (`MSTCNV.F90:1874`) where we pass `w(l-1)`. Using the extrapolation is
  *worse*: 3.19% against 2.46%.
* **The back-conversion density.** `TLOC` is the caller's `tl(l)`, the
  environmental temperature, so ModelE converts forward with `rho0` and back
  with `PL/(R*tl)`. Reproducing that asymmetry changes nothing — the two
  densities agree closely enough (2.678% either way).
* **`CDNC`.** It varies 45.7-61.8 cm^-3 against our constant 60. Substituting
  the dumped values makes the residual *worse* (4.37%), but that test is
  unreliable: two plumes per step call the microphysics at identical pressures,
  so matching records by pressure picks arbitrarily between them. **Retest this
  once the microphysics dump carries a plume index** — it is the most plausible
  remaining candidate.

### The open thread

`wmp` is **not** assigned after the `CONVECTIVE_MICROPHYSICS` call. The routine
computes `CONDV = CONDMU - CONDP` and converts both to plume-mass units, but the
plume's condensate is not visibly updated from `CONDV` in the ascent loop. Our
port removes the precipitation directly. Whether ModelE removes it elsewhere, or
carries it and removes it inside the sorting, is the next thing to establish —
it decides whether our removal belongs where we put it at all.

## 23. Where ModelE removes the precipitation

`wmp` is never assigned after the microphysics call because it is passed **into**
it. The caller supplies `wmp` at the `condv` argument slot
(`MSTCNV.F90:1876-1888`), so the plume's condensate is **replaced** by

```
CONDV   = CONDMU - CONDP                 ! CONDMU formed with rho0
wmp_new = CONDV * mplume * TLOC*R/PL     ! converted back with rho_env
```

not decremented by `CONDP`. The forward and backward conversions use different
densities — `rho0 = ma/delz` going in, `rho_env = PL/(R*tl(l))` coming out — so
in principle ModelE also rescales the surviving condensate by `rho0/rho_env`.

**Numerically the distinction does not matter here**: replacement and
subtraction give bit-identical residuals (0.5316% either way), because the two
densities agree closely in this case. Worth recording anyway, since a case with
a larger plume/environment temperature contrast would separate them.

### Confirming the removal is real

Between bsort at level `l` and bsort at level `l+1`, ModelE loses a median
1.60e-3 of total water. Our re-saturation reproduces the vapour **exactly**
(median difference 0.0e0) and the condensate high by **exactly that amount** —
`loss / condensate excess = 1.00000`. So the missing water is precipitation, and
nothing else about the inter-level step is wrong.

The sorting itself is also exact in condensate, not just in mass: feeding
`sort_blends` the oracle's state reproduces `plume_heat`, `plume_water` **and**
`plume_condensate` to 2.4e-15, and water conserves through it to 4.2e-16. That
check had never been run — only the masses had been.

### Residual: 0.53%, cause unknown

Our condensate after precipitation is 0.53% high (median). Candidates tested
against the oracle and **rejected**:

| candidate | result |
|---|---|
| `wcupass = 1.5*w(l-1) - 0.5*w(l-2)` | worse (3.19% vs 2.46% on the removal) |
| back-conversion through `rho_env` | identical, 0.5316% |
| `CDNC` from the oracle (45.7-60.3) | **worse: 0.96% vs 0.53%** |

The `CDNC` test is now trustworthy — the microphysics dump carries `mplume`, so
records match on pressure *and* plume mass, 484/484 unambiguously. Substituting
ModelE's own `CDNC` degrading the fit is genuinely odd and suggests something
else compensates for the constant-60 assumption.

At 0.53% in the condensate this is a small residual, but it is what compounds
into the four remaining branch flips, so it is still the thing worth chasing.
Untested candidates: the parcel temperature `TP` handed to the microphysics, and
`rvl`/`cloudrvl_mstcnv`, which we hold at 10 µm.

## 24. Retraction: the branch flips were not the cause, and the bsort stack is not wired in

### The prediction that failed

§20 closed with: *"Closing the 2.5% would very likely close the whole remaining
gap."* That is now falsified and is retracted here, as the §19 claim that the
mid-level excess was "not downdraft-related" was retracted in §20.

The three microphysics-coupling corrections (`wcupass` extrapolation,
pre-condensation `TP`, scaled CDNC) landed as intended. Measured against the
oracle they took the condensate removal to 0.0001% median, the free-running
plume mass to 0.00210%, the detrained mass to 0.00443%, and **eliminated all
four downdraft branch flips** (268 both / 0 ModelE-only / 0 ours-only).

The tendency comparison did not move at all: dth +0.9793 / peak 1.158 against
+0.9793 / 1.163 before, dq +0.9812 / 0.947 against +0.9813 / 0.947.

### Why it did not move

`GissConvection` — the `PhysicsTerm` the single-column harness runs, and the
only thing the tendency comparison measures — does not call any of this code.
It calls `giss_plume.plume_ascent_column` (a single entraining plume with a
saturation-adjustment detrainment) and `giss_tendencies.convective_tendencies`
(a plain mass-flux operator). `grep` for callers of
`giss_downdraft.downdraft_descent` outside its own tests returns nothing;
`giss_bsort.plume_ascent` and `bsort_environment_tendencies` are likewise
reachable only from their tests.

Confirmed directly: stashing `giss_downdraft.py` and re-running the comparison
reproduced the metrics to the digit.

So every hypothesis in §17-§23 about *why the tendencies disagree* was tested
against a scheme that contains none of the ported physics. Those sections
remain valid as oracle-agreement results for the bsort components in isolation
— that is how they were measured — but none of them says anything about the
tendency gap, and the sub-cloud, mid-level and heating-magnitude discrepancies
were never evidence about the port at all.

### What the tendency comparison actually measures today

Free-running harness, all active BOMEX periods, `GissConvection` as wired:

| | corr | rms ratio |
|---|---|---|
| `dth_mc` | +0.7517 | 2.119 |
| `dq_mc` | +0.6051 | 1.384 |

Median peak-heating ratio JCM/ModelE 1.87x. Cloud base within one level 48/48.
These are the numbers for the *old* scheme; the +0.979 correlations quoted
earlier came from a single oracle-driven column, not this harness.

### Bugs the exercise did find

Chasing the (irrelevant) tendency gap still surfaced three real porting bugs in
the downdraft, all fixed in `f5fbc1b`: `condpr` never entered the falling
precipitation flux, `dp_from_cldtop` was measured from the model top rather
than the plume top, and `prcp_mixrat` was missing its `mb2kg` factor. Together
these left the shaft dry, so it read as buoyant and shed 75% of itself per
level, collapsing in three levels. It now descends to the surface, entrains at
every level, and detrains exactly 0.5 per level below `dcl`, as ModelE does.

### Next

Wiring the bsort stack into `GissConvection` in place of
`giss_plume.plume_ascent_column` is now the blocking item — no tendency
measurement is informative about the port until it is done. That needs the
plume spectrum (the `iplume` loop), per-plume cloud-base mass flux, the
continuity/subsidence step, and `bsort_environment_tendencies` replacing
`convective_tendencies`.

Open and *not* explained by any of the above: one sort flip at l=14, where the
blend buoyancy sits on the -0.2 K threshold and a 0.03% upstream mass drift is
enough to move it. It costs 6.048 of ModelE's 15.907 total downdraft source.

## 25. Scope: wiring the bsort stack into `GissConvection`

### Configuration facts that shrink the job

Read from the Fortran defaults and confirmed against the BOMEX oracle:

* `lessent_scheme` defaults to **2** (`MSTCNV.F90:629`). That sets
  `mplumes(1) = 0` (`MSTCNV.F90:2814`), so the less-entraining plume never
  fires and `plumes_per_base` collapses to a single plume, `iplume=2`. The
  oracle agrees: every one of the 52 plumes in the 48-step BOMEX run has
  `iplume=2`.
* Consequently `contce = entrainment_cont2 = 0.6` (`MSTCNV.F90:506`), a
  constant, not a per-plume spectrum.
* The base loop runs **downward**, `lmin = lmcm-1 → dcl` (`MSTCNV.F90:1452-1459`),
  with `lmcm = ls1-1`.
* Plume bases in BOMEX sit at levels 6-9, ~1.1 plumes per step. The trip count
  is short in practice but must be static in JAX: the number of candidate
  `lmin` values, roughly 10-15.

### The structural constraint

`apply_continuity_tendencies` is called **inside** the plume loop
(`MSTCNV.F90:2376`), so `sm`/`qm` are updated after every plume and the next
plume ascends through an environment its predecessors already modified. The
loop is therefore inherently sequential: a `lax.scan` over a fixed-length list
of candidate `lmin` values carrying the environment, with inactive iterations
masked out. It cannot be vectorised over plumes.

### Work items

| | Item | Where | Est. LOC | Notes |
|---|---|---|---|---|
| W1 | `cloud_base_closure` wrapper | new, wraps `giss_mass_flux` | ~120 | See risks R1/R2 |
| W2 | Per-plume driver: ascent → descent → tendency | new module | ~150 | Also builds `tvl`, `gzl`, `delz`, `zl`, `fpi` source weights |
| W3 | Extend `bsort_environment_tendencies` to consume the descent | `giss_tendencies.py` | ~40 | |
| W4 | Subsidence substepping | `giss_tendencies.py` | ~50 | |
| W5 | Outer `lmin` scan with sequential environment carry | new module | ~80 | |
| W6 | Rewire `GissConvection.__call__` | `giss_mstcnv.py` | ~60 | Keep the `giss_plume` path behind a flag |

**W3** is the smallest and most overdue. `bsort_environment_tendencies` still
deposits downdraft air at the level where it formed, and its docstring still
says `dd_evap_precip_loop` "is not ported" — stale since `giss_downdraft.py`
landed. It needs to take the descent's detrained mass/heat/water and the
environmental evaporation (`dsm_evp`, `dqm_evp`) instead.

**W4**: the function computes a `courant` field and never uses it. ModelE
substeps up to `ksubmax=20` times whenever `cmneg(l) > 0.999*ml(l)`
(`MSTCNV.F90:4915-4935`). Static 20-trip scan, most iterations no-ops.

### Risks

* **R1 — `wturb` is not available.** `wbases = max(0.5, [2,1]*maxval(wturb(lmin0+1:lmin+1)))`
  needs the PBL turbulent velocity profile, which no jcm diagnostic currently
  publishes. Derive it from the TTE-TKE term when that is in the stack,
  otherwise fall back to the convective velocity scale already computed in
  `giss_mstcnv.surface_flux_scales`. Affects the plume's initial `w`, which
  feeds the entrainment closure.
* **R2 — the closure has no oracle.** `giss_mass_flux.py` ports `MASS_FLUX2`
  for the single-source case (`nlpi=1`) and is validated for *behaviour*, not
  level-by-level agreement. The `closure_diag.txt` dump (unit 772) exists in
  the instrumented Fortran but is gated behind `SCMopt%PlumeDiag`, whose rerun
  crashed, so no closure oracle has been collected. This is the largest
  correctness risk: the closure sets the mass-flux *magnitude*, which is
  exactly what the 1.87x peak-heating ratio is about.
* **R3 — cost.** Static trip count `nlmin` x (ascent scan + descent scan +
  substep scan), all over `nlev`, per column per step. Benchmark before
  running anything at grid resolution.
* **R4 — `dcl`** (dry convective layer top) sets both the `lmin` range and the
  downdraft's `boundary_layer_top`. The harness currently reads it from the
  oracle; it needs a real diagnosis.
* **R5** — the l=14 knife-edge sort flip from §24, still open.

### Suggested order

W3 → W2 → W4 → W5 → W6, leaving **W1 last** and driving the first end-to-end
run with cloud-base mass fluxes taken from the oracle. That holds the closure
fixed while the ported physics is measured, so a residual tendency gap can be
attributed to one side or the other instead of to both at once. It also means
R1 and R2 do not block the first useful measurement.

## 26. The closure oracle already existed; what it says about W1

### There was nothing to fix

`closure_diag.txt` (unit 772) and `bisect_diag.txt` (unit 773) were written by
the same instrumented BOMEX run as every other dump in `oracle_data/`, with the
same timestamp. `SCMopt%PlumeDiag` is on, and the run exits cleanly
(`run_status` 13, "terminated normally"). The blocker recorded earlier was
about the SUBDD *solo-variable* registration path (`mc_w_p1` and friends),
which these plain Fortran writes do not go through. **R2's premise was wrong:
the closure oracle has been on disk all along.**

Both files are now in the bridge's `oracle_data/`, with
`oracle.read_closure_diag`, `read_bisect_diag` and `read_source_weights`.
832 closure calls, 832 bisection traces, one-to-one. Every one of the 52
launched plumes matches a closure row's `fmp2` exactly, which confirms the
column mapping end to end.

### R2 is replaced by a sharper problem

`nlpi` -- the number of blended boundary-layer source levels -- is **6, 7, 8 or
9 in all 832 calls, and never 1**; `lmin0` is always 1. `giss_mass_flux.py`
ports the `nlpi=1` case explicitly ("the multi-source generalization is
deferred"), so the ported closure covers a case this configuration never
reaches. That is a structural gap, not a tuning gap.

It is bounded, though. `MASS_FLUX2` blends the source as
`SDN = SUM(SMO1*FPIBYAML)` (`MSTCNV.F90:8960`) and spreads the removal as
`SMN1(:) = SMO1(:)*(1 - fmp2*fpibyaml)` (`MSTCNV.F90:8925`), so

    SDN = SUM(SMO1*fpibyaml) - fmp2*SUM(SMO1*fpibyaml^2)

The generalization is a **second moment** of the same weights, not a new loop:
our three-level stencil keeps its shape, with `theta[0]` becoming the weighted
blend and one extra term carrying `fmp2`'s effect on it. Estimate ~50 LOC in
`giss_mass_flux.py`, now validatable term by term against `bisect_diag.txt`,
which dumps `SDN, SUP, QDN, QUP, SVDN, SVUP, DMSE1` at every iteration.

`fpi` itself is recoverable from the existing source dump (`read_source_weights`).
In BOMEX it is uniform: 1/7 over seven levels, zero above.

### The bisection usually does not converge

Iterations per call run 4-9. In call 0, `fplume` climbs 0.5 -> 0.998 while
`dmse1` stays negative throughout (-1.687 -> -0.843): the sign never flips, so
the bisection saturates against the `fplume` ceiling rather than reaching the
`|DMSE1| <= 1e-3` band. Across all 832 calls the final `|dmse|` has median 1.83
and max 2.69, and `fplume` lands on bisection lattice points (0.0625, ..., 0.999).

`giss_mass_flux.py`'s unit tests assert that the bisection *drives* `DMSE1`
toward zero. That is true directionally but the endpoint is a saturated bound,
not neutrality, so a test asserting convergence to the tolerance band would be
asserting something ModelE does not do. Match the trace, not the ideal.

### Revised W1

Still last in the order, but no longer blocked and no longer unmeasurable:
generalize the source blend to `nlpi > 1` (~50 LOC), then validate iteration by
iteration against the bisection trace before wiring the closure in. R1
(`wturb` for `wbases`) is unchanged and still the one input with no oracle.

## 27. W3 done: the descent is coupled, and the continuity chain has an oracle

### What changed

`bsort_environment_tendencies` now takes the descent's outputs rather than the
plume's formation-level downdraft:

* `downdraft_detrained_*` -- where the shaft actually hands its air back, which
  is several hundred metres below where the sort routed it.
* `downdraft_entrained_air` -- `edraft`, a removal alongside the plume's own.
* `evaporation_heat` / `evaporation_water` -- `dsm_evp` / `dqm_evp`. ModelE
  applies these straight to `sm`/`qm` at `MSTCNV.F90:4743`, *before* continuity
  runs, so they belong in the state the subsidence advects, not in the
  deposition. They add vapour without adding air, so they do not enter the mass
  tendency.

`deposit_downdraft_locally` is gone; it only ever chose between two known-wrong
placements.

### A second bug the oracle exposed

Both ends of the circulation dump their remainder, and only one was ported:

* `DM(LMAX) += MPLUME` (`MSTCNV.F90:1998`) -- already implemented, via the
  `dump()` path in `plume_ascent`.
* `dm(ldmin) += ddraft` (`MSTCNV.F90:4809`) -- **was missing**. The descent
  detrains nothing at level 0, so the shaft's remaining mass was vanishing and
  the environment's budget did not close. Now handed to `ldmin`.

### Validation

`continuity_diag.txt` is this chain's own oracle and had never been used --
§24 flagged it as "the one part of the tendency chain never checked against its
own oracle". Two tests:

1. **The integration itself.** Feeding ModelE's own `dm`/`dmr` through our
   `cm(l) = cm(l-1) - dm(l) - dmr(l)` reproduces its `cm` to a worst relative
   error of **6.3e-7** across all 52 plume blocks -- dump precision (`es14.6`).
   The continuity operator is exact.
2. **The coupled chain**, driven from our own free-running plume and descent:

| level | `dm` ours / ModelE | `dmr` ours / ModelE |
|---|---|---|
| 9 | 32.493 / 32.493 | -21.895 / -22.037 |
| 10 | 10.428 / 10.428 | -19.542 / -19.698 |
| 11 | 25.065 / 25.081 | -13.189 / -13.408 |
| 0 | 0.267 / 0.369 | -7.397 / -7.397 |

`dmr` agrees within ~1% everywhere and to four figures in the boundary layer,
which validates the source draw, the plume's entrainment and the downdraft's
entrainment together. `dm` agrees exactly where the plume dominates.

The two places it does not agree are both already-identified upstream issues,
not tendency-side ones:

* level 13: 18.163 vs 12.097 -- the §24 sort flip, which routes 6.048 to
  detrainment instead of the downdraft. Its excess then propagates down the
  `cm` cumsum, which is the whole of the remaining mid-level `cm` gap.
* levels 0-6: our downdraft carries 0.62-0.72 of ModelE's mass, so it detrains
  proportionally less. Same root cause -- the missing 6.048 of source.

Both reduce to the single knife-edge blend at l=14. That is now the highest-value
open item: it is the only thing standing between this chain and a clean match.

### Note for later

`ruff` is not installed in the jax-gcm venv, so the changed files were not
linted. Tests: 308 pass, 3 skipped.

## 28. Chasing l=14: two real bugs, and the accuracy wall behind them

### It was never knife-edge in the way §24 assumed

§24 called the l=14 flip a knife-edge that a 0.03% mass drift could move. The
blend dump says otherwise: `mixbuoy` there is -7.216e-4 against a `negbuoy/tvl`
of -6.864e-4, a margin of -3.53e-5, or **5% of the threshold**. A 0.03% drift
cannot do that. Something systematic was wrong, and two things were.

### Bug 1: the wrong gravitational constant

`buoyancy_work_increments` used `jcm.constants.grav` (9.81) while every other
constant in the port comes from `giss_thermodynamics.GRAV`, ModelE's 9.80665 as
declared in `Constants_mod.F90`. `giss_downdraft` was separately hardcoding
9.80665, so the port disagreed with itself.

The evidence was exact: our `kew` over ModelE's was **1.0003416**, and
9.81/9.80665 = **1.0003416**. Fixing it took the error in `w` at the first
level above cloud base from +1.71e-4 to -3.3e-6, a factor of 50.

`jcm/constants.py` still defaults to 9.81 model-wide. Changing that is a
separate decision with a much wider blast radius.

### Bug 2: `wcupass` extrapolated against nothing

The port special-cased the first level above cloud base, reasoning that no
`wcu(l-2)` existed yet. Wrong twice: ModelE's special case is the *literal*
second model level (`if(l.eq.2)`, `MSTCNV.F90:1868`), which a plume based at
level 10 never reaches; and `wcu` below cloud base is not unset --
`MSTCNV.F90:1636-1640` fills `lcl-2 .. lmin` with `wbases(iplume)`. The
extrapolation always applies, against that seed.

`plume_ascent` now takes `cloud_base_velocity`, defaulting to ModelE's
`max(0.5, wturb)` floor of 0.5. Free-running BOMEX, this moves `condpr` at the
first precipitating level from **+4.1% to -0.4%** and the downdraft's surface
mass ratio from **0.7255 to 0.9994**.

### The wall

Fixing these did not produce a clean match, and the reason is worth stating
plainly. Two blends in this column sit on their sorting thresholds:

| blend | `mixbuoy` | threshold | margin |
|---|---|---|---|
| l=11, n=2 | +1.734929e-4 | `posbuoy/tvl` +1.697421e-4 | +3.75e-6 (**2%**) |
| l=14, n=1 | -7.216168e-4 | `negbuoy/tvl` -6.863541e-4 | -3.53e-5 (**5%**) |

Getting both right at once requires `mixbuoy` accurate to ~2%. With the
corrected `wcupass` the l=14 blend now sorts to the downdraft correctly, and
the l=11 blend flips out of the updraft. A scan over `cloud_base_velocity`
locates the cliff between 0.55 and 0.58 and shows the result is otherwise
**insensitive** across [0.6, 2.0] -- so `wbase` is not a tuning lever, it is a
switch between which of the two blends is wrong.

The aggregate free-run plume-mass error is therefore 4.2e-1 at the faithful
`wbase = 0.5` and 3.3e-4 at 0.7. The larger number is the honest one: at 0.7 a
wrong `wcupass` was compensating, and the agreement was partly luck. This was
put to the user, who chose the faithful value.

### What this changes about the remaining work

`wturb` (scope risk R1) is no longer a side issue -- it is the one unmeasured
input standing between us and knowing whether `wbase` is 0.5 or something
larger. It needs either a dump added to the instrumented ModelE plus a BOMEX
rerun, or a derivation from the PBL scheme.

Beyond that, the binding constraint is the accuracy of the plume's accumulated
thermodynamic state, not any single formula. Every component checked so far is
exact when teacher-forced -- the sort to 2.4e-15, re-saturation to 3.6e-16,
`blend_air_masses` to machine precision at every blend, the continuity
integration to 6.3e-7. The error is in the *accumulation*, and the sorting
thresholds are what make a sub-1% accumulation error visible as a discrete
branch flip.

## 29. R1 closed: `wbases` measured, and two oracle files corrected

### The answer

A new dump (unit 771, `MSTCNV.F90` end of `cloud_base_closure`) records
`wbases`, `mplumes` and the `wturb` maximum they come from. Rebuilt and reran
BOMEX:

* **`wbases(2) = 0.5 in all 52 plumes.`** `max(wturb)` over the source block
  runs 0.346 to 0.372 and never reaches the 0.5 floor, so the floor binds
  throughout. The `cloud_base_velocity = 0.5` default chosen in §28 is exact
  for this case, not a guess.
* `mplumes(1) = 0` in every plume, confirming directly what §25 inferred from
  the ascent dumps: `lessent_scheme = 2` collapses the spectrum to one plume.

**Scope risk R1 is closed for BOMEX.** It returns for any case where the
boundary layer is turbulent enough to lift `wturb` above 0.5, which this one
never is; wiring a real `wturb` remains necessary before trusting the closure
outside these conditions.

### Two oracle files were contaminated

`closure_diag.txt` and `bisect_diag.txt`, added in §26, had been copied from
files that accumulated across 16 runs -- the dumps open with
`position='append'` and the run directory had never been cleared. Their row
counts were inflated: 832 -> **52**, 7984 -> **499**.

Every distribution reported from them in §26 holds unchanged, because the
surplus was the same run repeated: `nlpi` is 6-9 and never 1, `lmin0` is always
1, `fplume` spans 0.0625-0.999, `|dmse|` has median 1.83 and max 2.69, and all
52 launched plumes match a closure row's `fmp2` exactly. Only the counts were
wrong, and §26's conclusions about W1 stand.

The other six oracle files were compared byte-for-byte against the fresh run
and are **identical**, so nothing derived from the plume, blend, downdraft,
source or continuity dumps is affected -- and the rebuild reproduces the run
bit-for-bit, which is itself a useful check that adding the dump perturbed
nothing.

*Standing rule: clear the run directory before a diagnostic rerun, or the
dumps silently concatenate.*

### Where the tendency chain now stands

Free-running plume and descent, against `continuity_diag.txt`:

| level | `dm` ours / ModelE | `dmr` ours / ModelE | `cm` ours / ModelE |
|---|---|---|---|
| 0 | 0.368 / 0.369 | -7.397 / -7.397 | 7.029 / 7.028 |
| 1 | 0.348 / 0.356 | -7.418 / -7.410 | 14.099 / 14.082 |
| 2 | 0.656 / 0.687 | -7.436 / -7.422 | 20.880 / 20.817 |
| 9 | 32.493 / 32.493 | -21.983 / -22.037 | 25.586 / 22.964 |

The **boundary layer now matches to under 1% on `cm`** -- the layers whose
missing cooling started this whole investigation back in §19. The downdraft
reaches the surface with the right mass (ratio 0.9994), detrains nothing above
`dcl` as ModelE does, and entrains within ~15% at every level.

The whole of the remaining disagreement is the single l=11 blend from §28,
which swaps `dm` between levels 10 and 11 (31.283/10.428 against 10.428/25.081)
and propagates up the `cm` cumsum. That one blend clears its threshold by 2%,
and closing it needs `mixbuoy` accurate to better than that.

## 30. W2 done: the plume runs on state, not on dumps

`giss_plume_driver` composes the chain and derives what it needs from the
column. Until now every profile the ported pieces consumed came from an oracle
dump, so none of the port had been run on inputs a model could supply.

### What it derives, and how it checks out

| quantity | source | agreement |
|---|---|---|
| `delz = ma/rho0` | hydrostatic | exact, every level |
| `tvl = tl*(1+deltx*qv)` | state | exact, every level |
| `gzl` | centred geopotential difference (`CLOUDS_DRV.F90:582`) | exact at every level with valid neighbours |
| `fpi` -> `source_removal` | `MSTCNV.F90:2672-2675` | **exact**: 7.397145 from each of seven layers, zero above `dcl`, summing to `mplume` |
| `mcfrac` | ascent mass flux and `wcu` (`MSTCNV.F90:5136`) | right shape, ~2x the oracle at cloud base |
| source parcel | `MSTCNV.F90:1583-1601` | see below |

`fpi` matching exactly is the useful one: it exercises the mass weighting, the
rule that drops source layers above the mixed-layer top, and the
renormalisation, all at once.

### Where the driver stands against the oracle

Driving the whole chain from the reconstructed BOMEX column, cloud-base mass is
exact and the first ascent step is within 1.7e-4. But at the cloud-base level
our sort routes 32.5144 to the **downdraft** where ModelE **detrains** 32.493 --
the same mass, the opposite branch.

The seed explains it:

| | ours | ModelE | rel |
|---|---|---|---|
| heat | 2154.913 | 2155.917 | -4.7e-4 |
| water | 0.827414 | 0.834436 | -8.4e-3 |
| condensate | 0.036227 | 0.030678 | +1.8e-1 |

Re-saturation conserves total water, so ModelE's raw parcel held 0.865114 and
ours 0.863641 -- 0.17% less. Since `source_removal` is *exact*, that difference
is entirely the sub-cloud humidity the harness reconstructs by subtracting
`dq_mc*dt` from the post-convection state: 0.028 g/kg, which is well inside what
that reconstruction can be trusted to. It is a harness limitation, not a driver
one, and there is no dump of the sub-cloud `sm`/`qm` to settle it against.

The consequence is §28's wall once more, now reached at the cloud-base level
rather than at l=11 or l=14: a 0.17% input error decides a branch.

### Three NaN-gradient bugs, all pre-existing

The driver's gradient test is the first thing to differentiate the composed
chain, and every gradient through the scheme was NaN. The scheme's whole purpose
is to be differentiable, so this was the most valuable thing W2 turned up:

1. `giss_bsort` still had one `maximum(mass, tiny)` -- the last instance of the
   very pattern `safe_divide` was introduced to replace, and which its own
   docstring warns about.
2. The microphysics ran at levels the plume never reached, where the parcel
   temperature is 0 K and the Murphy & Koop fits are outside their validity
   range.
3. `giss_downdraft` called `condensate_evaporation` at 0 K the same way, and
   raised the rain ratio to the power 0.6 at zero -- finite in value, infinite
   in derivative, and most levels have no rain.

All three are value-neutral: the free-running comparison against
`continuity_diag.txt` is unchanged to every digit.

*Standing rule: a NaN gradient hides until something differentiates the whole
composed chain. Unit-level gradient tests on each piece did not find any of
these, because each piece is only degenerate in the context the others put it
in.*

### Note on units

`giss_plume_driver` takes ModelE's convention: `exner = (p in mb)**kappa` paired
with `potential_temperature = theta/1000**kappa`, so that their product is the
temperature. Passing a conventional potential temperature against this Exner
function overstates every temperature about sevenfold, which lands outside the
saturation fits rather than merely being inaccurate. This cost an hour when the
first test column was built the conventional way; the module docstring now says
so explicitly.

### Next

W4 (subsidence substepping) and W5 (the `lmin` scan with sequential environment
carry) remain, then W6 to rewire `GissConvection`. W1 is unblocked but still
wants the `nlpi > 1` generalisation from §26.

## 31. W4 done: the subsidence is substepped

`bsort_environment_tendencies` advected in a single pass with the whole
interface flux, which draws a layer past empty whenever the flux exceeds its
mass. ModelE splits the flux until no step takes more than 99.9% of a layer,
updating the layer mass between steps (`MSTCNV.F90:4931-4941`), and calls a
layer driven negative fatal. `courant` was computed and never used; it now
reports how far over the limit the column went.

The trip count is fixed at ModelE's `ksubmax = 20`, since JAX needs it static.
Once the flux is exhausted the clip returns zero, so the remaining substeps are
exact no-ops -- a column under the limit gets bit-identical results to the old
single pass, which the `continuity_diag.txt` comparison confirms unchanged to
every digit.

Exercised at Courant 4.8 (a 600 kg/m^2 mixed layer under 25 kg/m^2 layers):

| substeps | humidity at level 1 |
|---|---|
| 1 | 0.005000 (spike barely moved) |
| 2 | 0.006177 |
| 3 | 0.006126 |
| 5 | 0.006477 |
| 20 | 0.006477 (converged) |

Upwind monotonicity holds throughout, so no new extreme appears -- overshooting
one is exactly how the unsubstepped scheme drives humidity negative.

### A deliberate divergence

ModelE's scalars go through `adv1d`, a quadratic-upstream scheme carrying
moments (`smom`, `qmom`). The `PhysicsState` has no moments, so this port
applies the same flux splitting to the plain upwind transport it already used --
which is what ModelE itself does for momentum. The splitting is the part that
matters for stability; the moments are an accuracy refinement that would need
the state to carry them.

### The top boundary, found by a failing test

The mass-conservation test failed at -526 kg/m^2 after the change. Cause: in a
column with no inversion the plume stays buoyant to the model top, never
terminates, and never dumps -- so the mass it entrained had no way back. The
old single pass hid this, because the uncancelled flux at the top exactly
offset the accumulated exchange; the clip cannot reproduce that.

ModelE never meets the case: its ascent loop is bounded by `lm` and the
stratosphere stops the plume first. A plume still rising at the highest level
now terminates there and dumps like any other termination. The test column also
gained a trade inversion, so it tests the physics rather than the boundary.

*This is the second time a conservation test has caught a missing termination
dump -- the downdraft's `ldmin` remainder in §27 was the first. Both ends of
every circulation need one.*

## 32. W5 done: the plume sequence, and the first end-to-end measurement

`convective_column` sweeps candidate cloud-base levels and applies each plume
before the next begins. Under `lessent_scheme = 2` the spectrum loop collapses
to a single plume, so this is one descending sweep from `lmcm-1` to `dcl`
rather than two nested loops.

**The plumes are coupled.** Each ascends through what its predecessors left
behind, so the sequence is not the sum of the parts -- a test asserts the two
differ, because if they ever agree the environment is not being carried.

**`ma` does not carry.** The compensating subsidence exactly cancels each
plume's mass flux: `cm(l) = cm(l-1) - dm - dmr` makes the per-level mass
tendency identically zero, and ModelE never reassigns `MA` inside the loop. Only
heat and water change between plumes.

Also ports `lmin0` -- the lowest layer within 300 mb of the base
(`MSTCNV.F90:2646-2648`), which is why BOMEX always draws from the surface.

### Two traps from the fixed trip count

JAX needs a static trip count, so every candidate is traced whether it convects
or not, and both of these are consequences:

* A non-convecting candidate must be given a **dummy 1 kg/m^2 plume**, not a
  zero-mass one. A zero-mass parcel has no temperature, the saturation
  adjustment divides by it, and the NaN survives being multiplied by zero.
  `jnp.where` on the output is not enough; the input has to be well-posed.
* The scan carry is cast to the **promoted dtype** up front. A host mixing
  single-precision state with anything double-precision otherwise promotes
  inside the loop and the carry fails to typecheck -- which is exactly what
  happened the first time the oracle column was fed in.

### First end-to-end measurement of the port

All 48 BOMEX periods, the whole chain, driven with the oracle's cloud-base
masses so the unported closure is **not** part of the measurement:

| | value |
|---|---|
| plume count vs ModelE | **48/48** |
| `dth_mc` correlation | **+0.8169** (rms ratio 1.120) |
| `dq_mc` correlation | **+0.8380** (rms ratio 1.031) |
| median peak heating ratio | **0.924** |

| level | dth ours / ModelE | dq ours / ModelE |
|---|---|---|
| 0 | -2.44 / -1.63 | +0.28 / -1.14 |
| 4 | 5.39 / 6.13 | -17.83 / -22.03 |
| 5 | 4.18 / 4.27 | -8.32 / -6.88 |
| 6 | 5.05 / 5.28 | +0.93 / -0.34 |
| 7 | 5.50 / 5.65 | +1.01 / +0.53 |

**These are not comparable to §24's +0.7517 / +0.6051 / 1.87x.** Those were the
*old* scheme, free-running with its own closure over the same periods. This run
holds the closure at ModelE's values, so it measures the ported physics alone.
Putting the two side by side would credit the port for the closure being exact.

What it does show: the mid-level heating and drying track closely, and the peak
heating ratio is 0.924 rather than an order-of-magnitude miss. The sub-cloud
layers remain the weak point -- cooling about 1.5x too strong, and moistening
where ModelE dries -- which is the same signature §30 traced to the 0.17% error
in the reconstructed sub-cloud humidity.

### A unit trap in the oracle reader

`oracle.read_convection_field(..., "dq_mc")` returns **kg/kg/day**, not
g/kg/day; `bomex_compare_plots` scales it by 1000 internally. Compared without
that factor the oracle appears to dry a thousand times too slowly. The tell was
that `|dq|/|dth|` came out at a constant 0.0025 across all 48 periods -- a fixed
ratio between two independent fields is a unit error, never physics.

### Next

W6 rewires `GissConvection` onto `convective_column`, at which point the
harness measures the port instead of the old scheme. W1 (`nlpi > 1`) is the
remaining piece before the closure can come from the model rather than the
oracle, and until it does, every number above depends on oracle-supplied
cloud-base mass.

## 33. W6 done: the harness finally measures the port

`GissConvection` ran `giss_plume.plume_ascent_column` -- a single entraining
plume with a saturation-adjustment detrainment. Under `allow_mc` it now runs
`convective_column`. The old path stays behind `bsort=False`, so both can be
measured against the same oracle rather than one replacing the other silently.

Scope is one cloud base (`max_plumes=1`): ModelE runs the closure at every
candidate from `lmcm-1` down to `dcl`, and the ported closure solves a single
base.

### The numbers, and what they finally separate

| | old scheme | ported | ported, ModelE's closure |
|---|---|---|---|
| `dth_mc` correlation | +0.7517 | **+0.8734** | +0.8169 |
| `dq_mc` correlation | +0.6051 | **+0.8608** | +0.8380 |
| peak `dth` ratio | 1.87x | 2.11x | **0.924** |
| `dth` rms ratio | 2.119 | 2.482 | **1.120** |
| `dq` rms ratio | 1.384 | 2.279 | **1.031** |

The shape improved substantially and the magnitude got slightly worse. That
looks equivocal until the two right-hand columns are read together: the same
chain, driven with ModelE's own cloud-base masses, lands at 0.924 peak and rms
ratios near 1.

The closure explains the gap exactly. Across all 48 periods the ported closure
hands over **2.185x** ModelE's total `mplume` (median; mean 2.094), and the
harness peak heating ratio is **2.11x**. The tendency is near-linear in the
cloud-base mass flux, so 2.185 in gives 2.11 out.

**The entire remaining magnitude error is the closure, and none of it is the
ported physics.** Section 26 already established why: `giss_mass_flux` ports the
`nlpi = 1` single-source case, and BOMEX runs `nlpi` between 6 and 9 in all 832
closure calls.

### Where the port stands

Every structural piece is now ported, wired and oracle-checked:

| piece | agreement |
|---|---|
| blend masses (`blend_air_masses`) | machine precision, every blend |
| blend sort | 2.4e-15 |
| re-saturation | 3.6e-16 |
| condensate removal (microphysics) | 0.0001% median |
| continuity integration | 6.3e-7 (dump precision) |
| source removal (`fpi`) | exact |
| `delz`, `tvl` | exact |
| downdraft branch structure | detrains 0.5 below `dcl`, as ModelE |
| plume count over BOMEX | 48/48 |

What is *not* ported is the closure's multi-source generalisation, and it is now
the only thing standing between the harness and a like-for-like magnitude.

### Next

W1, with a target: the closure must come down by a factor of about 2.19, and
`bisect_diag.txt` gives `SDN, SUP, QDN, QUP, SVDN, SVUP, DMSE1` at every
iteration of all 832 calls to check it term by term. After that, raise
`max_plumes` so the sweep runs every candidate base rather than one.

## 34. W1 done: the closure spans a multi-layer source

`cloud_base_closure` handles any `nlpi`. The three-level function it replaces is
now its `nlpi = 1` special case, and a test asserts the general form reproduces
it exactly when all the weight sits on one layer.

Three things differ once the source spans more than one layer: the plume draws
`fmp2*fpi(l)` from each; removing it makes the block subside internally, layer
`l` carrying its own air down into `l-1` (`MSTCNV.F90:8945-8957`); and
`SDN`/`QDN` become `fpi`-weighted blends of the updated layers. The Fortran's
cascade walks down the block subtracting `fpi(l)` from a running total -- the
flux crossing the bottom of layer `l` is `fmp2` times the weight *below* it,
which vectorises as an exclusive cumulative sum, no scan needed.

### The arithmetic is verified

`bisect_diag.txt` dumps the closure's internals at every iteration of all 52
calls, which allows checking the formulas **without needing ModelE's state**:

| check | error |
|---|---|
| `FMP2 == FPLUME*AML(NLPI)` | 5.2e-07 relative |
| `SVDN`/`SVUP` virtual-temperature form | implied condensate 2.4e-07 (BOMEX has none) |
| `DMSE1` formula | 2.0e-04 absolute, on an O(1) quantity |

All at the dump's own `es14.6` precision.

### Effect on the harness

| | before W1 | after W1 | target |
|---|---|---|---|
| closure / ModelE `mplume` | 2.185x | **1.696x** | 1.0 |
| peak `dth` ratio | 2.110 | **1.773** | 1.0 |
| `dth` rms ratio | 2.482 | **2.057** | 1.0 |
| `dq` rms ratio | 2.279 | **1.854** | 1.0 |
| `dth` / `dq` correlation | +0.873 / +0.861 | +0.870 / +0.859 | |

### The residual is the input state, not the closure

Everything the closure is *given* checks out, and the one thing it *derives*
from the state does not:

| quantity | ours / ModelE |
|---|---|
| `AML(NLPI)` (= `ma(lmin)`) | **exact** |
| `PRESL(NLPI+1)` | **exact** |
| `PLKL(NLPI+1)` | 0.999972 |
| evaluation level `lmin` | **46/48 exact** |
| `DQSUM0` | **0.910** (range 0.88-1.00) |

`DQSUM0` is the lifted blend's supersaturation -- a difference of two
near-equal saturation terms, so the 0.17% error in the harness's reconstructed
sub-cloud humidity (section 30) is more than enough to move it 9%. `DMSE1` is
the same kind of difference, and the bisection's sensitivity turns a few tenths
of a Kelvin into a factor of order two in `fmp2`.

**This is the third distinct place the same 0.17% has surfaced**: as a branch
flip at cloud base (section 30), as sub-cloud tendency errors (section 32), and now
as the closure's magnitude. It is one root cause, and it is in the bridge's
state reconstruction (`th - dth_mc*dt` from the post-convection SUBDD output),
not in the port.

### What would settle it

A dump of `sm`/`qm` as the closure sees them -- roughly fifteen lines of Fortran
and a two-minute rebuild, the same procedure that produced `wbases`. Every
remaining discrepancy in the port now traces to the reconstructed state, so
that dump is worth more than any further work on the ported code.

### Where the port stands

All six work items are done. Every component agrees with its oracle at dump
precision, the chain runs end to end, and with ModelE's own cloud-base masses it
reproduces the tendencies at a peak ratio of 0.924 and rms ratios of 1.12 and
1.03 (section 32). Driven by its own closure it is 1.77x strong, and that gap is
now attributed rather than merely measured.

## 35. The closure-state dump, and three missing terms

Section 34 attributed the closure's residual to the harness's reconstructed
state. Dumping what the closure actually sees (`closure_state_diag.txt`, unit
770: both the raw `sm`/`qm` and the enhanced `smo1`/`qmo1`, plus `fpi`, `aml`,
`plkl`, `presl`, `tstar`, `qstar`) showed that attribution was **wrong**, and
found three terms the port was missing.

### The reconstruction was never the problem

| | ours / ModelE |
|---|---|
| `th` | 1.000029 (**+0.003%**) |
| `qv` | 1.000503 (**+0.050%**) |

The 0.17% of section 30 was a *blend* comparison, and what it was really seeing
was the missing enhancement below.

### Three terms, all in ModelE, none in the port

1. **Surface-flux enhancement** (`MSTCNV.F90:2725-2746`). `tstar` and `qstar`
   are added to every source layer at or below `dcl` before the closure or the
   plume sees it; the moisture term is capped at half the layer's humidity and
   the environment is left untouched. On BOMEX it is **1.06% of the source
   humidity** -- twenty times the reconstruction error. `giss_mstcnv` already
   computed the scales (`surface_flux_scales` matches ModelE's `tstar`/`qstar`
   to 0.49%); the bsort path just never applied them.
2. **`tadj` relaxation** (`MSTCNV.F90:2820`). `FMP2 = FMP2*min(1, dtime/tadj)`,
   inside the closure and before ModelE's own dump. At `dtsrc = 1800` and
   `tadj = 3600` that is exactly one half.
3. **Saturation guard** (`MSTCNV.F90:2804`). ModelE returns before the
   bisection when the lifted blend never saturates. **51 of the 98** BOMEX
   closure calls take that exit; without the guard the port returns a mass flux
   for columns ModelE declines to convect at all.

The `tadj` factor was diagnosable only because the dump made an exact-input
comparison possible: the ratio came out at **2.00000**, and a clean integer
ratio is a missing constant, never physics -- the same tell as the constant
`|dq|/|dth|` in section 32.

### The closure now reproduces ModelE

Driven with ModelE's exact `smo1`/`qmo1`/`fpi`/`aml`:

| | |
|---|---|
| ratio median | **1.000000** |
| within 0.1% | 23/47 |
| within 1% | 43/47 |

### End to end on the harness

| | W6 | after W1 | now |
|---|---|---|---|
| peak `dth` ratio | 2.110 | 1.773 | **1.286** |
| `dth` rms ratio | 2.482 | 2.057 | **1.398** |
| `dq` rms ratio | 2.279 | 1.854 | **1.191** |
| `dth` correlation | +0.873 | +0.870 | +0.862 |
| `dq` correlation | +0.861 | +0.859 | +0.849 |

At the heating peak, level 4 now gives 7.06 against 6.13 K/day and **-22.41
against -22.03 g/kg/day**.

### What is left

The sub-cloud layers: level 0 cools 3.44 against ModelE's 1.63 and moistens
slightly where ModelE dries 1.14. That is the last structured disagreement, and
with the closure and the state both now accounted for it is no longer
attributable to either.

*Standing lesson, now three times over: when a ratio comes out at a clean
constant -- 2.185 tracking a 2.11 heating ratio, `|dq|/|dth|` at 0.0025, `fmp2`
at exactly 2.00000 -- look for a missing factor before looking for physics.*

## 36. The sub-cloud layers: the per-plume operator is not the problem

The last structured disagreement is sub-cloud: level 0 cools 2.46 K/day against
ModelE's 1.63, and moistens 0.19 g/kg/day where ModelE dries 1.14. Four
candidates were tested against the oracle and three were eliminated.

### Eliminated

| candidate | measurement | verdict |
|---|---|---|
| rain evaporating into clear air | `dqm_evp` ours 0.001641 vs 0.001815 at level 0 | **10% weaker** than ModelE -- would cool and moisten *less* |
| downdraft temperature | `thdn` 41.2565 vs 41.0369, i.e. ~1.6 K **warmer** at the surface | detrains warmer air -- cools *less* |
| downdraft humidity | `qldn` 0.015211 vs 0.015427, **drier** | moistens *less* |
| `fevap` formula | matches `MSTCNV.F90:4228-4242`; `geometric_fevapfac = 0` in the active preset | correct |

All three physical candidates point the *wrong way*: each would make the
sub-cloud tendency weaker, not stronger.

### The per-plume operator is right

Teacher-forcing a single plume with the oracle's own inputs and comparing every
sub-cloud term against `continuity_diag.txt`:

| term | agreement at levels 0-5 |
|---|---|
| `dsmr` (removal heat) | 0.5% |
| `dqmr` (removal water) | ~1% |
| `dsm` (deposited heat) | ~2% |
| `dqm` (deposited water) | ~2% |
| `cm` (interface flux) | 0.03% |
| advection-state `th` | **0.004%** |
| advection-state `qv` | **0.07%** |

The state the subsidence advects, and the flux that advects it, both reproduce
ModelE. Whatever is wrong is not in the tendency operator.

### A comparison that was not valid

Much of this section was initially read off a comparison of **one plume's**
tendency against the SUBDD total for a **two-plume step** -- -2.19 against
-3.95, which looked like severe under-cooling and pointed at entirely the wrong
things. `dth_mc` is the step total over every plume; a single-plume result can
only be compared against the summed continuity dump, never against SUBDD.

### An interaction worth knowing

Adding the `tstar`/`qstar` enhancement improved the self-consistent harness
(peak 1.773 -> 1.286) but *worsened* the run driven with ModelE's cloud-base
masses pinned (0.924 -> 1.266, section 32). That is not a contradiction: pinning
ModelE's `mplume` while enriching the parcel breaks the consistency between
closure and parcel that ModelE maintains. The oracle-pinned diagnostic was a
clean measurement of the ported physics only while the parcel matched ModelE's;
now that both the parcel and the closure are ported, the self-consistent harness
is the meaningful one.

### Where that leaves it

Every per-plume component is verified against its own oracle, and the plume
count matches 48/48. The residual sub-cloud discrepancy is therefore in how the
sweep composes plumes, or in a term ModelE applies once per step rather than per
plume -- neither of which the current dumps resolve, since `continuity_diag.txt`
is written per plume and `dth_mc` only per step.

The dump that would settle it is `sm`/`qm` after each plume's
`apply_continuity_tendencies`, which would let the sweep be checked plume by
plume instead of only at its end.
