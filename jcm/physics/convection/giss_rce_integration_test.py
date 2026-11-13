"""The GISS convection term inside the real ECHAM term list, integrated in time.

Everything else that exercises this term runs it alone: ``giss_mstcnv_test``
calls it directly (on a column and, since the grid tests, on a
``(nlev, nlon, nlat)`` block), and ``giss_hydra_config_test`` builds
``physics=echam-giss-conv`` but never runs it. So the term had never been
*stepped* with the rest of the physics around it -- vertical diffusion and the
surface exchange feeding it a boundary layer, the cloud microphysics consuming
its precipitation, radiation cooling the column between steps -- and a port that
is correct in isolation can still destabilise a column through that coupling.
The module docstring of :mod:`jcm.physics.convection.giss_microphysics` names
the specific runaway to fear: too little precipitation leaves the downdraft
warm, a warm downdraft sheds its mass, and a smaller downdraft evaporates less
still.

These tests close that gap as far as a single column can. They build the
**preset's own term list** through Hydra -- so the ordering under test is the
one a run would use -- swap RRTMGP for the grey two-stream to make it
affordable, and integrate a radiative-convective column through the same
``compute_tendencies`` path the global model uses.

What they do **not** establish is a global run: fixed SST, one column, grey
radiation, and a column that stabilises after spin-up so convection is only
exercised while it is settling. Magnitudes remain validated only against the
ModelE SCM oracle (see ``docs/source/design/giss_convection_port.md``).

Measured at 10 days (720 steps) when this was written, against the stock
``physics=echam`` preset run identically as a reference:

===================================  ==============  ==============
quantity                             GISS            Tiedtke
===================================  ==============  ==============
surface temperature, start -> end    298.91 -> 303.61  299.76 -> 303.59
column mean q, start -> end (g/kg)   2.525 -> 2.610    2.609 -> 2.705
temperature drift over last fifth    4.71 K            4.75 K
===================================  ==============  ==============

i.e. the two schemes take the column to the same place. The GISS term's own
``mcp`` peaks at 98 mm/day during spin-up and falls away as the column
stabilises, which is the expected behaviour for a column that has stopped being
unstable, not a shutdown: the cloud scheme's stratiform rain is ~2e-18 for both
schemes over the same window.
"""

import unittest
from pathlib import Path

import numpy as np
import pytest


_PRESET = "echam-giss-conv"
#: Days to integrate. 2 days is 144 steps at the RCE column's 1200 s step --
#: long enough that an instability in the coupling shows up (the 10-day runs
#: above diverge from each other by nothing) and short enough for a test.
_DAYS = 2.0


def _physics(preset):
    """Build a preset's term list, with the grey two-stream for radiation."""
    from hydra import compose, initialize_config_dir

    import jcm
    from jcm import runners
    from jcm.physics.echam.echam_terms import default_radiation_parameters
    from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation

    config_dir = str(Path(jcm.__file__).resolve().parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"physics={preset}"])
    physics = runners.build_physics(cfg)
    return physics.replace(
        "radiation",
        GreyTwoStreamRadiation(params=default_radiation_parameters("macv2sp")))


def _run(preset, days=_DAYS):
    from jcm.rce import rce_column, rce_initial_state, run_rce

    scm = rce_column(nlev=47, physics=_physics(preset),
                     interactive_humidity=True)
    state = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7)
    return run_rce(scm, state, n_days=days)


@pytest.mark.slow
class TestGissConvectionInTheEchamStack(unittest.TestCase):
    """Integrate the preset's own term list and check it does not misbehave."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip("hydra")
        cls.pred = _run(_PRESET)
        cls.temperature = np.asarray(cls.pred.relaxed_states["temperature"])
        cls.humidity = np.asarray(
            cls.pred.relaxed_states["specific_humidity"])

    def test_the_convection_slot_holds_the_giss_term(self):
        """Guard against the run below silently testing Tiedtke."""
        names = [type(t).__name__ for t in _physics(_PRESET).terms]
        self.assertIn("GissConvection", names)
        self.assertNotIn("TiedtkeConvection", names)

    def test_the_integration_stays_finite(self):
        """No NaN or inf anywhere in the evolved column."""
        self.assertTrue(np.isfinite(self.temperature).all())
        self.assertTrue(np.isfinite(self.humidity).all())

    def test_the_column_stays_physical(self):
        """Temperature in a plausible range and humidity never negative.

        A coupling instability shows up here long before it produces a NaN: the
        first symptom of the downdraft runaway would be a column cooling without
        bound at the levels the shaft detrains into.
        """
        self.assertGreater(self.temperature.min(), 150.0)
        self.assertLess(self.temperature.max(), 350.0)
        self.assertGreaterEqual(self.humidity.min(), 0.0)
        self.assertLess(self.humidity.max(), 0.1)

    def test_convection_actually_runs(self):
        """Otherwise every assertion here is about an inactive term."""
        conv = self.pred.physics_data["convection"]
        self.assertGreater(float(np.asarray(conv.cloud_base_mass_flux).max()),
                           0.0, "the closure never produced a mass flux")
        self.assertTrue(np.any(np.asarray(conv.dth_mc) != 0.0),
                        "the term never produced a heating profile")

    def test_the_surface_precipitation_is_finite_and_non_negative(self):
        """``mcp`` feeds ``precip_conv``, which the surface exchange sums in."""
        mcp = np.asarray(self.pred.physics_data["convection"].mcp)
        self.assertTrue(np.isfinite(mcp).all())
        self.assertTrue((mcp >= 0.0).all())

    def test_it_does_not_diverge_from_the_stock_preset(self):
        """The column must land where ``physics=echam`` lands it.

        Not a validation of either scheme -- this column is radiatively
        controlled at fixed SST, so a scheme cannot move the equilibrium far --
        but it is a strong check that the GISS term is not injecting or removing
        heat the rest of the stack has to absorb. Measured at 10 days the two
        surface temperatures agree to 0.02 K; the tolerance here is set for the
        shorter run and for float32.
        """
        other = _run("echam")
        mine = self.temperature[-1]
        theirs = np.asarray(other.relaxed_states["temperature"])[-1]
        self.assertLess(float(np.abs(mine[-1] - theirs[-1])), 2.0,
                        "surface temperatures diverge")
        self.assertLess(float(np.abs(mine - theirs).max()), 5.0,
                        "the profiles diverge somewhere in the column")


if __name__ == "__main__":
    unittest.main()
