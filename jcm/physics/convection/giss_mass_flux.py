"""GISS cloud-base mass-flux closure (``MASS_FLUX2``), single-source-level case.

Ports the closure magnitude from ``MSTCNV`` ``MASS_FLUX2``
(``modelE/model/MSTCNV.F90``): a bisection that finds the cloud-base plume mass
fraction ``FPLUME`` which, after the plume removes that mass from the cloud-base
layers (compensating subsidence/mixing of the layers above) plus a small precip
re-evaporation, restores the cloud base to a neutral moist-static-energy state
(``DMSE1 -> 0``).

Scope: single source level (``nlpi = 1``)
----------------------------------------
The Fortran handles a blend of ``nlpi`` boundary-layer source levels with an
inner redistribution loop. This port implements the common **single-source-level
case** (``nlpi = 1``), where that loop collapses: the plume originates in one
layer (``lmin``) and mixes with the two layers above (``lmin+1``, ``lmin+2``).
The multi-source generalization is deferred. All other elements -- the 9-step
bisection, the layer mass redistribution, and the precip re-evaporation
(``FEVAP = 0.005*FPLUME``) -- are ported faithfully for this case.

Validation status
-----------------
Structure is a faithful read of the Fortran. On the convectively active **BOMEX**
oracle the closure produces physical values (``FPLUME ~ 0.04-0.14``,
``fmp2 ~ 2-8 kg/m^2``) for well-triggered columns, and the downstream ``dth_mc``
lands at the right order of magnitude (see ``STATUS.md``). It is not a
level-by-level Fortran match; the unit tests assert closure *behaviour* -- the
bisection drives ``DMSE1`` toward zero, a more unstable column yields a larger
mass flux -- rather than exact agreement.

JAX notes
---------
The fixed 9-iteration bisection is unrolled (static count); the Fortran's
``if(DMSE1>..) FPLUME-=DFP`` branches become ``jnp.where`` so the routine is
differentiable and runs broadcasting-native on ``(...)``-shaped column inputs.
The early ``exit`` when ``|DMSE1| <= 1e-3`` is implicit: inside that band neither
``where`` branch fires, so ``FPLUME`` simply stops moving.

Units: potential temperature K, pressure Pa, specific/condensate humidity kg/kg,
layer mass kg/m^2, Exner dimensionless.
"""

import jax.numpy as jnp

from jcm.physics.convection.giss_thermodynamics import (
    DELTX,
    LHE,
    SHA,
    d_ln_qsat_dt,
    saturation_specific_humidity,
    virtual_temperature,
)

_DMSE_TOL = 1.0e-3   # neutrality band (K), matches the Fortran exit threshold
_N_ITER = 9          # fixed bisection count, as in MASS_FLUX2
_FEVAP_FRAC = 0.005  # precip re-evaporation fraction of FPLUME (Fortran FEVAP)
_TEENY = 1.0e-20     # guards divisions at masked (zero-mass) levels


def cloud_base_mass_flux(theta, specific_humidity, air_mass, exner, pressure,
                         condensate=None, phase: str = "water"):
    """Cloud-base plume mass fraction that neutralizes the cloud base.

    Single-source-level (``nlpi=1``) port of ``MASS_FLUX2``. Inputs are the
    three-level cloud-base stencil, **vertical on axis 0**: the source layer
    (``lmin``) and the two layers above (``lmin+1``, ``lmin+2``). The routine
    forms the mass-weighted ``SM = theta*mass`` internally to match the Fortran
    redistribution. Broadcasting-native: the trailing axes are horizontal.

    Args:
        theta: Potential temperature [K], ``(3, ...)`` at ``[lmin, lmin+1,
            lmin+2]``. ``theta[0]`` is the boundary-layer source parcel.
        specific_humidity: Specific humidity [kg/kg], ``(3, ...)`` at the same
            levels.
        air_mass: Layer air mass [kg/m^2], ``(3, ...)`` at the same levels.
        exner: Exner function, ``(2, ...)`` at ``[lmin, lmin+1]``.
        pressure: Pressure [Pa], ``(2, ...)`` at ``[lmin, lmin+1]``.
        condensate: Optional cloud water [kg/kg], ``(2, ...)`` at ``[lmin,
            lmin+1]`` (defaults to zero -- the loading terms ``wm_dn``/``wm_up``).
        phase: ``"water"`` or ``"ice"`` (static).

    Returns:
        ``(fplume, fmp2, dmse1)`` -- the plume mass fraction, the plume mass
        ``fplume*air_mass[0]`` [kg/m^2], and the residual cloud-base instability
        after closure (≈0 when neutralized). ``fplume`` is returned raw; the
        caller applies the ``MSTCNV`` limiters (``MINFRAC``, the ``0.5*ma`` caps).
    """
    # Unpack the three-level stencil (axis 0) into the per-level names the
    # MASS_FLUX2 redistribution below is written in.
    theta_dn, theta_up, theta_up2 = theta[0], theta[1], theta[2]
    q_dn, q_up, q_up2 = specific_humidity[0], specific_humidity[1], specific_humidity[2]
    mass1, mass2, mass3 = air_mass[0], air_mass[1], air_mass[2]
    exner1, exner2 = exner[0], exner[1]
    pressure1, pressure2 = pressure[0], pressure[1]
    wm_dn = 0.0 if condensate is None else condensate[0]
    wm_up = 0.0 if condensate is None else condensate[1]

    # In MASS_FLUX2 both SLH and SLHE are formed from LHE (= LHE/SHA),
    # regardless of the LHX phase passed in for the qsat calls.
    slh = LHE / SHA

    # Mass-weighted (extensive) layer quantities, as the Fortran mixes.
    smo1, qmo1 = theta_dn * mass1, q_dn * mass1
    smo2, qmo2 = theta_up * mass2, q_up * mass2
    smo3, qmo3 = theta_up2 * mass3, q_up2 * mass3

    # Saturation-adjustment slopes for the precip re-evaporation (intensive).
    tplift = exner2 * theta_dn
    qsatc0 = saturation_specific_humidity(tplift, pressure2, phase)
    dqsum0 = (q_dn - qsatc0) / (1.0 + slh * qsatc0 * d_ln_qsat_dt(tplift, phase))

    t1 = exner1 * theta_dn
    qsatc1 = saturation_specific_humidity(t1, pressure1, phase)
    dq1 = (qsatc1 - q_dn) / (1.0 + slh * qsatc1 * d_ln_qsat_dt(t1, phase))

    t2 = exner2 * theta_up
    qsatc2 = saturation_specific_humidity(t2, pressure2, phase)
    dq2 = (qsatc2 - q_up) / (1.0 + slh * qsatc2 * d_ln_qsat_dt(t2, phase))

    fplume = jnp.broadcast_to(jnp.asarray(0.5), jnp.shape(theta_dn * 1.0)).astype(float)
    dfp = jnp.full_like(fplume, 0.5)
    dmse1 = jnp.zeros_like(fplume)

    for _ in range(_N_ITER):
        dfp = dfp * 0.5
        fmp2 = fplume * mass1
        frat1 = fmp2 / mass2
        frat2 = fmp2 / mass3

        # Compensating subsidence/mixing of the layers above into the plume gap.
        smn2 = smo2 * (1.0 - frat1) + frat2 * smo3
        qmn2 = qmo2 * (1.0 - frat1) + frat2 * qmo3
        smn1 = smo1 * (1.0 - fmp2 / mass1) + frat1 * smo2
        qmn1 = qmo1 * (1.0 - fmp2 / mass1) + frat1 * qmo2

        # Precip re-evaporation: moisten/cool using a small fraction of the
        # cloud air, capped by the available supersaturation DQSUM.
        dqsum = fmp2 * dqsum0
        fevap = _FEVAP_FRAC * fplume
        positive = dqsum > 0.0

        dq_a = jnp.where(positive, jnp.minimum(fevap * mass2 * dq2, dqsum), 0.0)
        smn2 = smn2 - slh * dq_a / exner2
        qmn2 = qmn2 + dq_a
        dqsum = dqsum - dq_a

        dq_b = jnp.where(dqsum > 0.0, jnp.minimum(fevap * mass1 * dq1, dqsum), 0.0)
        smn1 = smn1 - slh * dq_b / exner1
        qmn1 = qmn1 + dq_b

        # Recompute the cloud-base instability after this trial mass flux.
        sdn, sup = smn1 / mass1, smn2 / mass2
        qdn, qup = qmn1 / mass1, qmn2 / mass2
        sv_dn = virtual_temperature(sdn, qdn, wm_dn)
        sv_up = virtual_temperature(sup, qup, wm_up)
        qsat_up = saturation_specific_humidity(sup * exner2, pressure2, phase)
        dmse1 = (sv_up - sv_dn) * exner2 + slh * (qsat_up - qdn)

        # Bisection: too stable -> less mass flux; too unstable -> more.
        fplume = jnp.where(
            dmse1 > _DMSE_TOL, fplume - dfp,
            jnp.where(dmse1 < -_DMSE_TOL, fplume + dfp, fplume))

    return fplume, fplume * mass1, dmse1


def cloud_base_mass_flux_column(theta, specific_humidity, air_mass, exner,
                                pressure, cloud_base, boundary_layer_top=None,
                                source_dtheta=0.0, source_dq=0.0,
                                condensate=None, phase: str = "water"):
    """``MASS_FLUX2`` with the full **multi-source-level** (``nlpi>1``) blend.

    :func:`cloud_base_mass_flux` treats the plume as drawn from a single layer.
    ``MSTCNV`` instead blends the source over every boundary-layer level up to
    cloud base, mass-weighted (``fpi = aml/sum(aml)``, line 2653), removing
    ``fmp2*fpi(l)/aml(l)`` from each and cascading the resulting sub-cloud
    subsidence downward (lines 8707-8719).

    **Why this matters.** Spreading the removal makes the blended source parcel
    nearly *invariant* during the bisection (the ModelE trace shows ``QDN``
    moving only 0.0185→0.0184 across all 9 iterations), so ``DMSE1`` is far less
    sensitive to ``fplume`` and the closure converges at a much larger plume
    fraction. With a single source level the parcel relaxes quickly instead and
    the closure converges early, under-computing the cloud-base mass flux ~2x.

    ``source_dtheta``/``source_dq`` carry ``MSTCNV``'s surface-flux enhancement
    of the source parcel (``tstar``/``qstar``, ``mc_tqstar_fac``; see
    :func:`~jcm.physics.convection.giss_mstcnv.surface_flux_scales`). These are
    **synergistic with the blend, not independent**: because the blend keeps
    ``qdn`` nearly fixed, a small moisture boost shifts the whole ``DMSE1`` curve
    rather than being relaxed away, and the latent term carries
    ``LHE/SHA ≈ 2490 K`` per kg/kg of ``qdn``. Validated against the ModelE
    closure oracle on BOMEX period 47 (`fplume` 0.306 → 0.497 with the boost, vs
    ModelE 0.682).

    Args:
        theta, specific_humidity, air_mass, exner, pressure: column profiles,
            **surface-first** ``(nlev, ...)``.
        cloud_base: closure level ``lmin``, ``(...)``.
        boundary_layer_top: ``dcl``. ``MSTCNV`` excludes source levels above the
            boundary-layer top so a parcel displaced through it is not mixed with
            free-tropospheric air (line 2655). Note ``dcl`` is the level *below*
            the boundary-layer height. ``None`` disables the exclusion.
        source_dtheta, source_dq: surface-flux enhancement of the source parcel.
        condensate: optional cloud water [kg/kg]. phase: ``"water"``/``"ice"``.

    Returns:
        ``(fplume, fmp2, dmse1)`` -- ``fplume`` is the fraction of the *top*
        source layer's mass (``FMP2 = FPLUME*AML(NLPI)``).
    """
    nlev = theta.shape[0]
    level = jnp.arange(nlev).reshape((nlev,) + (1,) * (theta.ndim - 1))
    slh = LHE / SHA

    lmin = jnp.clip(cloud_base, 0, nlev - 3)
    source_top = lmin if boundary_layer_top is None else jnp.minimum(
        lmin, boundary_layer_top)

    def at(arr, offset):
        idx = jnp.clip(lmin + offset, 0, nlev - 1)[None, ...]
        return jnp.take_along_axis(arr, idx, axis=0)[0]

    # Source blend, mass-weighted and cut off at the boundary-layer top.
    is_source = level <= source_top
    fpi = jnp.where(is_source, air_mass, 0.0)
    fpi = fpi / jnp.maximum(jnp.sum(fpi, axis=0), _TEENY)
    by_aml = fpi / jnp.maximum(air_mass, _TEENY)

    # Surface-flux enhancement applies to the boundary-layer source levels.
    theta_src = theta + jnp.where(is_source, source_dtheta, 0.0)
    q_src = specific_humidity + jnp.where(is_source, source_dq, 0.0)
    smo1, qmo1 = theta_src * air_mass, q_src * air_mass

    mass_top = at(air_mass, 0)
    mass_up1, mass_up2 = at(air_mass, 1), at(air_mass, 2)
    exner_top, exner_up1 = at(exner, 0), at(exner, 1)
    pressure_top, pressure_up1 = at(pressure, 0), at(pressure, 1)
    theta_up1, theta_up2 = at(theta, 1), at(theta, 2)
    q_up1, q_up2 = at(specific_humidity, 1), at(specific_humidity, 2)
    smo2, qmo2 = theta_up1 * mass_up1, q_up1 * mass_up1
    smo3, qmo3 = theta_up2 * mass_up2, q_up2 * mass_up2
    wm_dn = 0.0 if condensate is None else jnp.sum(condensate * fpi, axis=0)
    wm_up = 0.0 if condensate is None else at(condensate, 1)

    # Saturation slopes for the precip re-evaporation, from the blended parcel.
    theta_dn0 = jnp.sum(theta_src * fpi, axis=0)
    q_dn0 = jnp.sum(q_src * fpi, axis=0)
    t_lift = exner_up1 * theta_dn0
    qsat_lift = saturation_specific_humidity(t_lift, pressure_up1, phase)
    dqsum0 = (q_dn0 - qsat_lift) / (
        1.0 + slh * qsat_lift * d_ln_qsat_dt(t_lift, phase))
    t_top = exner_top * at(theta_src, 0)
    qsat_top = saturation_specific_humidity(t_top, pressure_top, phase)
    dq_top = (qsat_top - at(q_src, 0)) / (
        1.0 + slh * qsat_top * d_ln_qsat_dt(t_top, phase))
    t_up = exner_up1 * theta_up1
    qsat_up = saturation_specific_humidity(t_up, pressure_up1, phase)
    dq_up = (qsat_up - q_up1) / (
        1.0 + slh * qsat_up * d_ln_qsat_dt(t_up, phase))

    at_top = level == lmin
    fplume = jnp.zeros_like(theta_dn0) + 0.5
    dfp = jnp.zeros_like(theta_dn0) + 0.5
    dmse1 = jnp.zeros_like(theta_dn0)

    for _ in range(_N_ITER):
        dfp = dfp * 0.5
        fmp2 = fplume * mass_top
        frat1, frat2 = fmp2 / mass_up1, fmp2 / mass_up2

        smn2 = smo2 * (1.0 - frat1) + frat2 * smo3
        qmn2 = qmo2 * (1.0 - frat1) + frat2 * qmo3

        # Each source level loses fmp2*fpi/aml; the top one also gains from above.
        smn1 = smo1 * (1.0 - fmp2 * by_aml)
        qmn1 = qmo1 * (1.0 - fmp2 * by_aml)
        smn1 = smn1 + jnp.where(at_top, frat1 * smo2, 0.0)
        qmn1 = qmn1 + jnp.where(at_top, frat1 * qmo2, 0.0)

        # Sub-cloud subsidence cascade: the flux through level l is fmp2 times
        # the cumulative fpi *below* it, carrying that level's properties down.
        # The Fortran cascade runs over every level from lmin down to 1
        # (``do ll=nlpi,2,-1``), *including* levels above the boundary-layer top
        # whose ``fpi`` was zeroed -- those still pass the subsidence flux down
        # even though they contribute nothing to the source blend. Masking this
        # to the source levels drops the transfer at ``lmin`` and collapses the
        # closure.
        below = jnp.cumsum(fpi, axis=0) - fpi
        flux = jnp.where((level <= lmin) & (level > 0), fmp2 * below, 0.0)
        fs, fq = flux * theta_src, flux * q_src
        zero = jnp.zeros((1,) + fs.shape[1:], dtype=fs.dtype)
        smn1 = smn1 - fs + jnp.concatenate([fs[1:], zero], axis=0)
        qmn1 = qmn1 - fq + jnp.concatenate([fq[1:], zero], axis=0)

        # Precip re-evaporation, capped by the available supersaturation.
        dqsum = fmp2 * dqsum0
        fevap = _FEVAP_FRAC * fplume
        dq_a = jnp.where(dqsum > 0.0,
                         jnp.minimum(fevap * mass_up1 * dq_up, dqsum), 0.0)
        smn2 = smn2 - slh * dq_a / exner_up1
        qmn2 = qmn2 + dq_a
        dqsum = dqsum - dq_a
        dq_b = jnp.where(dqsum > 0.0,
                         jnp.minimum(fevap * mass_top * dq_top, dqsum), 0.0)
        smn1 = smn1 + jnp.where(at_top, -slh * dq_b / exner_top, 0.0)
        qmn1 = qmn1 + jnp.where(at_top, dq_b, 0.0)

        sdn = jnp.sum(smn1 * by_aml, axis=0)
        qdn = jnp.sum(qmn1 * by_aml, axis=0)
        sup, qup = smn2 / mass_up1, qmn2 / mass_up1
        sv_dn = virtual_temperature(sdn, qdn, wm_dn)
        sv_up = virtual_temperature(sup, qup, wm_up)
        qsat_new = saturation_specific_humidity(
            sup * exner_up1, pressure_up1, phase)
        dmse1 = (sv_up - sv_dn) * exner_up1 + slh * (qsat_new - qdn)

        fplume = jnp.where(
            dmse1 > _DMSE_TOL, fplume - dfp,
            jnp.where(dmse1 < -_DMSE_TOL, fplume + dfp, fplume))

    return fplume, fplume * mass_top, dmse1
