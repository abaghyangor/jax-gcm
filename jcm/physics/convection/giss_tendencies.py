"""GISS convective environmental tendencies from the plume mass flux.

The convective plume's effect on the *environment* -- the ``dq_mc``/``dth_mc``
tendencies -- comes from **compensating subsidence**: by mass continuity, the
mass the plume carries up must sink in the surrounding environment, advecting the
environmental profile and warming/drying it. ``MSTCNV`` implements this as an
**upwind advection** of the mass-weighted environmental quantities by the
convective mass flux (``apply_continuity_tendencies`` / the subsidence loop).

This module ports that upwind operator. It takes a convective mass flux at the
interior interfaces and an environmental profile, and returns the per-layer
tendency from compensating subsidence.

Scope / status
--------------
This is the tendency *operator* (the mechanism). Producing the actual
``dq_mc``/``dth_mc`` still needs the convective mass-flux profile assembled from
the plume(s) (cloud-base closure magnitude + entrainment/detrainment continuity)
and the detrainment deposition added -- those are the remaining ports. The
single-step upwind here omits ``MSTCNV``'s CFL substepping (valid while the mass
flux is below the layer mass each step, as in shallow convection).

Sign convention follows ``MSTCNV``'s ``cmn``: a **positive** interface flux
advects the layer *below* the interface upward; negative advects the layer above
downward. Broadcasting-native: vertical on axis 0.
"""

import jax.numpy as jnp


def subsidence_tendency(interface_flux: jnp.ndarray,
                        prop: jnp.ndarray,
                        layer_mass: jnp.ndarray) -> jnp.ndarray:
    """Per-layer tendency of ``prop`` from compensating-subsidence advection.

    Faithful single-step port of the ``MSTCNV`` subsidence upwind flux: across
    each interior interface the mass flux carries the *upwind* (donor) layer's
    property; the per-layer change is the flux convergence, divided by the layer
    mass to give an intensive tendency.

    Args:
        interface_flux: Convective mass flux [kg/m²] at the ``n-1`` interior
            interfaces, ``(n-1, ...)``, ordered like the layers (interface ``i``
            sits between layers ``i`` and ``i+1``). Positive advects the lower
            layer up (``MSTCNV`` ``cmn`` sign).
        prop: Intensive environmental property per layer, ``(n, ...)`` (e.g.
            potential temperature, dry static energy, or specific humidity).
        layer_mass: Layer air mass ``MA = dp/g`` [kg/m²], ``(n, ...)``.

    Returns:
        Per-layer change in ``prop`` over the step (same units as ``prop``),
        ``(n, ...)``. The mass-weighted column total is zero (pure internal
        advection conserves the property).
    """
    # Upwind donor across each interior interface: lower layer if the flux is
    # upward (>0), upper layer if downward.
    donor = jnp.where(interface_flux > 0.0, prop[:-1], prop[1:])
    flux = interface_flux * donor                       # (n-1, ...)

    zero = jnp.zeros((1,) + flux.shape[1:], dtype=flux.dtype)
    flux_below = jnp.concatenate([zero, flux], axis=0)  # interface below each layer
    flux_above = jnp.concatenate([flux, zero], axis=0)  # interface above each layer

    # Mass-weighted convergence -> intensive tendency.
    return (flux_below - flux_above) / layer_mass


def convective_tendencies(mass_flux, plume_property, detrainment_rate,
                          layer_thickness, env_property, layer_mass):
    """Environmental tendency of a property from one plume (per step).

    Combines the two environmental effects of a convective plume on a conserved
    property (potential temperature for ``dth_mc``, specific humidity for
    ``dq_mc``):

    1. **Compensating subsidence** -- the plume's upward mass flux forces an
       equal environmental subsidence, advected by :func:`subsidence_tendency`
       (the environment sinks, so the interface flux is ``−mass_flux``).
    2. **Detrainment deposition** -- where the plume sheds mass
       (``DM = mass_flux · det · dz``) it deposits its own (warmer/moister)
       property into the layer: ``DM·(plume − env)/MA``.

    This is a single plume and omits precipitation/evaporation and the
    entrainment-removal bookkeeping; the **absolute magnitude scales with the
    cloud-base mass flux** (the closure ``fmp2``), so it is calibrated only once
    that scale and the two-plume sum are wired in. Broadcasting-native; the
    returned tendency is a change over the step the mass flux represents.

    Args:
        mass_flux: Plume mass at each level above cloud base [kg/m²], ``(n, ...)``.
        plume_property: Plume property per level (θ [K] or q [kg/kg]), ``(n, ...)``.
        detrainment_rate: Detrainment rate [1/m], ``(n, ...)``.
        layer_thickness: Layer thickness ``dz`` [m], ``(n, ...)``.
        env_property: Environmental property per level, ``(n, ...)``.
        layer_mass: Layer air mass ``MA`` [kg/m²], ``(n, ...)``.

    Returns:
        Per-layer environmental tendency of the property, ``(n, ...)``.
    """
    # Interface mass flux (between layers i and i+1) for the subsidence advection.
    mf_interface = 0.5 * (mass_flux[:-1] + mass_flux[1:])
    subsidence = subsidence_tendency(-mf_interface, env_property, layer_mass)

    # Detrainment deposition of the plume property into each layer.
    detrained_mass = mass_flux * detrainment_rate * layer_thickness
    deposition = detrained_mass * (plume_property - env_property) / layer_mass

    return subsidence + deposition
