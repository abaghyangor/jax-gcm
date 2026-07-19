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
