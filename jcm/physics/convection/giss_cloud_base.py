"""GISS convective trigger: lifting condensation level (cloud base).

Ports the cloud-base / lifting-condensation-level (LCL) detection from the GISS
``MSTCNV`` plume code (``modelE/model/MSTCNV.F90``). In the Fortran, a
boundary-layer parcel with potential temperature ``sparcel`` and specific
humidity ``qparcel`` is lifted level by level and the cloud base ``lcl`` is the
first level where it saturates::

    do lcl = lmin0+1, lmin+1
      if (qparcel > qsat(sparcel*plk(lcl), lhe, pres(lcl))) exit
    enddo

``sparcel*plk(l)`` is the parcel's *temperature* at level ``l``: lifting at
constant potential temperature is a dry adiabat, ``T(p) = T0 * (p/p0)**KAPA``.
Because we form the adiabat from the parcel's own origin ``(T0, p0)``, the
pressure-ratio is reference-independent and we don't need ModelE's particular
``plk`` reference pressure.

The Fortran's "first level scanning upward, then exit" loop becomes a vectorized
``argmax`` over the saturation mask (the same device the SPEEDY convection
trigger uses), so the identical code runs broadcasting-native on a single column
``(nlev,)`` or a vectorized block ``(nlev, ncols)``.

Differentiability
-----------------
The continuous pieces -- the dry adiabat and ``qsat`` -- are fully
differentiable. The cloud-base *index* itself comes from ``argmax`` and is a
**discontinuous** trigger (it jumps between levels), exactly the kind of
convective-triggering branch the project notes flag. Gradients flow through the
thermodynamic quantities, not through the integer level index.

Units (ModelE-native): temperature K, pressure **Pa**, specific humidity kg/kg.
"""

import jax.numpy as jnp

from jcm.physics.convection.giss_thermodynamics import (
    DELTX,
    KAPA,
    LHE,
    LHS,
    SHA,
    d_ln_qsat_dt,
    saturation_specific_humidity,
    virtual_temperature,
)


def dry_adiabatic_temperature(parcel_temperature: jnp.ndarray,
                              parcel_pressure: jnp.ndarray,
                              level_pressure: jnp.ndarray) -> jnp.ndarray:
    """Temperature of a parcel lifted dry-adiabatically to ``level_pressure``.

    ``T(p) = T0 * (p / p0) ** KAPA`` (constant potential temperature).

    Args:
        parcel_temperature: Parcel origin temperature ``T0`` [K], per column.
        parcel_pressure: Parcel origin pressure ``p0`` [Pa], per column.
        level_pressure: Pressure(s) to lift to [Pa]; vertical on axis 0.

    Returns:
        Parcel temperature at each ``level_pressure`` [K].
    """
    return parcel_temperature * (level_pressure / parcel_pressure) ** KAPA


def lifting_condensation_level(parcel_temperature: jnp.ndarray,
                               parcel_pressure: jnp.ndarray,
                               parcel_specific_humidity: jnp.ndarray,
                               level_pressure: jnp.ndarray):
    """Find the cloud base: first level where a lifted parcel saturates.

    The parcel is assumed to originate at the lowest provided level and lift
    upward (vertical on axis 0; index 0 = surface / highest pressure). At each
    level the dry-adiabatic parcel temperature is compared against saturation:
    the cloud base is the first level where ``parcel_q > qsat(T_parcel, p)``.

    Args:
        parcel_temperature: Parcel origin temperature [K], per column ``(...)``.
        parcel_pressure: Parcel origin pressure [Pa], per column ``(...)``.
        parcel_specific_humidity: Parcel specific humidity [kg/kg], ``(...)``.
        level_pressure: Column pressure profile [Pa], ``(nlev, ...)``.

    Returns:
        ``(cloud_base, condenses)`` where ``cloud_base`` ``(...)`` is the level
        index of the cloud base (or ``nlev`` if the parcel never saturates), and
        ``condenses`` ``(...)`` is a boolean: whether the parcel reaches
        saturation anywhere in the column.
    """
    nlev = level_pressure.shape[0]
    parcel_t = dry_adiabatic_temperature(
        parcel_temperature, parcel_pressure, level_pressure)
    qsat_level = saturation_specific_humidity(parcel_t, level_pressure)

    saturated = parcel_specific_humidity > qsat_level   # (nlev, ...)
    condenses = jnp.any(saturated, axis=0)              # (...)
    first_saturated = jnp.argmax(saturated, axis=0)     # first True (0 if none)
    # argmax returns 0 both for "saturated at the surface" and "never saturated";
    # the condenses mask disambiguates -- a non-saturating parcel gets the
    # sentinel ``nlev`` (an out-of-range index meaning "no cloud base").
    cloud_base = jnp.where(condenses, first_saturated, nlev)
    return cloud_base, condenses


def cloud_base_instability(parcel_potential_temperature: jnp.ndarray,
                           parcel_specific_humidity: jnp.ndarray,
                           parcel_condensate: jnp.ndarray,
                           above_potential_temperature: jnp.ndarray,
                           above_specific_humidity: jnp.ndarray,
                           above_condensate: jnp.ndarray,
                           exner: jnp.ndarray,
                           pressure: jnp.ndarray,
                           phase: str = "water") -> jnp.ndarray:
    """Cloud-base moist instability metric ``DMSE`` (units of K).

    Faithful port of the trigger in ``MSTCNV`` ``cloud_base_closure``::

        DMSE = (SVUP-SVDN)*PLK + SLH*(QSAT(SUP*PLK, LHX, pres) - QDN)

    evaluated at the interface just above cloud base (``lmin+1``). ``SUP``/``SDN``
    are *potential* temperatures (``SM = TH*MA``); ``PLK`` is the Exner function,
    so ``SUP*PLK`` is the layer-above temperature. ``SVUP``/``SVDN`` are the
    corresponding virtual potential temperatures (vapour + condensate loading);
    ``SLH = L/cp``. The first term is the parcel-vs-environment virtual
    *temperature* difference; the second is the latent contribution from the
    parcel's vapour excess over saturation aloft.

    Sign convention (confirmed from source): ``DMSE < 0`` => **unstable**.
    Larger parcel buoyancy or vapour excess drives it more negative.

    NOTE: structure and sign are read from the Fortran but **not yet validated
    numerically** against a convectively active oracle (the available DYCOMS case
    never triggers convection). Treat absolute values as provisional until a
    BOMEX/RICO oracle is available; the tests below assert structure, not
    oracle agreement.

    Args:
        parcel_potential_temperature: Source-parcel potential temperature ``SDN`` [K].
        parcel_specific_humidity: Source-parcel specific humidity ``QDN`` [kg/kg].
        parcel_condensate: Source-parcel condensate ``WMDN`` [kg/kg].
        above_potential_temperature: Layer-above potential temperature ``SUP`` [K].
        above_specific_humidity: Layer-above specific humidity ``QUP`` [kg/kg].
        above_condensate: Layer-above condensate ``WMUP`` [kg/kg].
        exner: Exner function ``PLK`` at the interface (= ``(p/p0)**KAPA``).
        pressure: Interface pressure [Pa].
        phase: ``"water"`` (``LHE``) or ``"ice"`` (``LHS``). Static.

    Returns:
        ``DMSE`` [K]; negative => convectively unstable.
    """
    sv_dn = virtual_temperature(
        parcel_potential_temperature, parcel_specific_humidity, parcel_condensate)
    sv_up = virtual_temperature(
        above_potential_temperature, above_specific_humidity, above_condensate)
    slh = (LHE if phase == "water" else LHS) / SHA
    qsat_above = saturation_specific_humidity(
        above_potential_temperature * exner, pressure, phase)
    return (sv_up - sv_dn) * exner + slh * (qsat_above - parcel_specific_humidity)


def cloud_base_triggers(parcel_potential_temperature: jnp.ndarray,
                        parcel_specific_humidity: jnp.ndarray,
                        parcel_condensate: jnp.ndarray,
                        above_potential_temperature: jnp.ndarray,
                        above_specific_humidity: jnp.ndarray,
                        above_condensate: jnp.ndarray,
                        exner: jnp.ndarray,
                        pressure: jnp.ndarray,
                        phase: str = "water"):
    """Whether moist convection triggers at the cloud-base interface.

    Combines the two ``MSTCNV`` ``cloud_base_closure`` gates at level ``lmin+1``:

    1. the source parcel is **saturated** when lifted to the interface
       (``QDN >= QSAT(SDN*PLK, pres)``), and
    2. the layer is **unstable** (:func:`cloud_base_instability` ``< 0``).

    Both gates are necessary; either failing means "try the next level / no
    convection here".

    The boolean trigger is a **discontinuous** switch (the convective-triggering
    branch the project notes flag); the underlying ``DMSE`` is continuous and
    differentiable. Not yet oracle-validated (see
    :func:`cloud_base_instability`).

    Returns:
        ``(triggers, dmse)`` -- a boolean per column and the continuous metric.
    """
    qsat_parcel = saturation_specific_humidity(
        parcel_potential_temperature * exner, pressure, phase)
    saturated = parcel_specific_humidity >= qsat_parcel
    dmse = cloud_base_instability(
        parcel_potential_temperature, parcel_specific_humidity, parcel_condensate,
        above_potential_temperature, above_specific_humidity, above_condensate,
        exner, pressure, phase)
    return saturated & (dmse < 0.0), dmse


def cloud_base_mass_flux(theta: jnp.ndarray,
                         specific_humidity: jnp.ndarray,
                         air_mass: jnp.ndarray,
                         exner: jnp.ndarray,
                         pressure: jnp.ndarray,
                         condensate: jnp.ndarray = None,
                         phase: str = "water",
                         n_iter: int = 9):
    """Cloud-base convective mass flux from the ``MSTCNV`` ``MASS_FLUX2`` closure.

    This sets the **absolute scale** of the scheme: the bisection finds the plume
    mass fraction ``FPLUME`` such that removing ``FMP2 = FPLUME·AML(lmin)`` from
    the cloud-base layer and redistributing it upward (compensating subsidence)
    drives the cloud-base moist-static-energy jump ``DMSE1`` to neutral. Every
    downstream tendency magnitude scales with the returned ``fmp2``.

    Faithful port of ``MASS_FLUX2`` (``modelE/model/MSTCNV.F90``). The active
    ``DMSE1`` uses ``SVUP − SVDN`` directly (ModelE computes a ``THBAR`` edge
    value but does **not** use it in the live residual), so no ``THBAR`` is
    needed here.

    Assumptions / scope (documented per the project's "state assumptions" rule):

    * **Single cloud-base sub-level** (``nlpi = 1``) -- the standard case, and the
      BOMEX case. ModelE's ``fpi``-weighted redistribution across multiple
      cloud-base sub-levels (``nlpi > 1``) is not ported.
    * The bisection runs a **fixed** ``n_iter`` (default 9, matching Fortran's
      ``do ITER=1,9``); ModelE's ``|DMSE1| <= 1e-3`` early exit is replaced by a
      :func:`jnp.where` step (zero update once converged) so the routine is
      ``vmap``/``grad`` safe rather than data-dependent control flow.
    * Includes the small precipitation re-evaporation correction (``FEVAP`` /
      ``MCLOUD``) exactly as in ModelE.
    * Returns the **pre-relaxation** ``fmp2 = FPLUME·AML(lmin)``. The caller
      applies ModelE's convective-timescale factor ``min(1, dtsrc/tadj)`` and the
      ``fmp2`` caps (lines ~2769-2802 of ``MSTCNV``) -- those are a separate step.

    Broadcasting-native: vertical on axis 0. ``theta``, ``specific_humidity``,
    ``air_mass`` are ``(3, ...)`` at ``[lmin, lmin+1, lmin+2]``; ``exner``,
    ``pressure`` are ``(2, ...)`` at ``[lmin, lmin+1]``; optional ``condensate``
    is ``(2, ...)`` cloud water at ``[lmin, lmin+1]`` (defaults to zero).

    Returns:
        ``(fplume, fmp2)`` -- the plume mass fraction and the cloud-base plume
        mass [kg/m²] per column.
    """
    th0, th1, th2 = theta[0], theta[1], theta[2]
    q0, q1 = specific_humidity[0], specific_humidity[1]
    m0, m1, m2 = air_mass[0], air_mass[1], air_mass[2]
    by0, by1, by2 = 1.0 / m0, 1.0 / m1, 1.0 / m2
    plk0, plk1 = exner[0], exner[1]
    pr0, pr1 = pressure[0], pressure[1]
    wmdn = 0.0 if condensate is None else condensate[0]
    wmup = 0.0 if condensate is None else condensate[1]
    slh = LHE / SHA                                      # SLH = SLHE = LHE/SHA

    # Mass-weighted SM/QM (ModelE SM = theta * air_mass).
    sm0, sm1, sm2 = th0 * m0, th1 * m1, th2 * m2
    qm0, qm1, qm2 = q0 * m0, q1 * m1, specific_humidity[2] * m2

    # Supersaturation of the lifted cloud-base parcel at lmin+1, and the local
    # saturation deficits used by the precip re-evaporation (computed once).
    tplift = plk1 * th0
    qsatc = saturation_specific_humidity(tplift, pr1, phase)
    dqsum0 = (q0 - qsatc) / (1.0 + slh * qsatc * d_ln_qsat_dt(tplift, phase))
    t1 = plk0 * th0
    qsatc = saturation_specific_humidity(t1, pr0, phase)
    dq1 = (qsatc - q0) / (1.0 + slh * qsatc * d_ln_qsat_dt(t1, phase))
    t2 = plk1 * th1
    qsatc = saturation_specific_humidity(t2, pr1, phase)
    dq2 = (qsatc - q1) / (1.0 + slh * qsatc * d_ln_qsat_dt(t2, phase))

    fplume = jnp.full_like(jnp.asarray(th0, dtype=jnp.result_type(float)), 0.5)
    dfp = 0.5
    for _ in range(n_iter):
        dfp = dfp * 0.5
        fmp2 = fplume * m0
        frat1, frat2 = fmp2 * by1, fmp2 * by2

        # Plume removes fmp2 from the cloud-base layer and pushes mass up; the
        # layer above sheds frat1 of its mass and gains frat2 from lmin+2.
        smn2 = sm1 * (1.0 - frat1) + frat2 * sm2
        qmn2 = qm1 * (1.0 - frat1) + frat2 * qm2
        smn1 = sm0 * (1.0 - fmp2 * by0) + frat1 * sm1
        qmn1 = qm0 * (1.0 - fmp2 * by0) + frat1 * qm1

        # Precipitation re-evaporation moistens/cools, only where the lifted
        # parcel is supersaturated (dqsum > 0); cascades lmin+1 then lmin.
        dqsum = fmp2 * dqsum0
        fevap = 0.005 * fplume
        dq = jnp.minimum(fevap * m1 * dq2, dqsum)
        dq = jnp.where(dqsum > 0.0, dq, 0.0)
        smn2 = smn2 - slh * dq / plk1
        qmn2 = qmn2 + dq
        dqsum = dqsum - dq
        dqa = jnp.minimum(fevap * m0 * dq1, dqsum)
        dqa = jnp.where(dqsum > 0.0, dqa, 0.0)
        smn1 = smn1 - slh * dqa / plk0
        qmn1 = qmn1 + dqa

        sdn, sup = smn1 * by0, smn2 * by1
        qdn, qup = qmn1 * by0, qmn2 * by1
        svdn = sdn * (1.0 + DELTX * qdn - wmdn)
        svup = sup * (1.0 + DELTX * qup - wmup)
        dmse1 = (svup - svdn) * plk1 + slh * (
            saturation_specific_humidity(sup * plk1, pr1, phase) - qdn)

        # Bisection: stable jump (dmse1 > 0) -> weaker plume; unstable -> stronger.
        step = jnp.where(dmse1 > 1e-3, -dfp,
                         jnp.where(dmse1 < -1e-3, dfp, 0.0))
        fplume = fplume + step

    return fplume, fplume * m0
