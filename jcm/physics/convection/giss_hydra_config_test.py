"""Hydra wiring for the GISS convection term (physics config group).

The term was reachable from no config at all for a long time, which meant it
could not be run through ``python -m jcm.main`` -- a standing violation of the
"new configurations go through Hydra" rule in ``CLAUDE.md`` and one of the four
things blocking a merge. ``physics=echam-giss-conv`` is that entry point, and
these tests pin the two things about it that are easy to break silently.
"""

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


def _cfg(overrides):
    import jcm

    config_dir = str(Path(jcm.__file__).resolve().parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


class GissConvectionHydraConfigTest(unittest.TestCase):
    PRESET = "physics=echam-giss-conv"

    def test_the_term_sits_in_the_convection_slot(self):
        """Order is execution order, and this preset is standalone for it.

        ``cfg.physics.terms`` is an *ordered* mapping consumed in order by
        ``runners.build_physics``. OmegaConf merges dicts by key, so inheriting
        ``echam`` and nulling ``tiedtke_convection`` appends the replacement at
        the end -- running convection after the cloud microphysics and the
        gravity-wave drag. That is why the preset lists every term itself, and
        this test is what catches someone "simplifying" it back to a defaults
        merge.
        """
        names = list(_cfg([self.PRESET]).physics.terms)
        self.assertIn("giss_convection", names)
        self.assertNotIn("tiedtke_convection", names)
        i = names.index("giss_convection")
        # ECHAM's physc ordering: vdiff and the surface exchange before
        # convection, the cloud microphysics after it.
        self.assertEqual(names[i - 1], "echam_surface")
        self.assertEqual(names[i + 1], "echam_1m_microphysics")
        self.assertLess(names.index("tte_tke_vertical_diffusion"), i)

    def test_allow_mc_is_on(self):
        """`allow_mc` defaults to False, which returns zero tendencies.

        A preset that forgot it would run, produce nothing, and look like a
        physics result rather than a misconfiguration.
        """
        term = _cfg([self.PRESET]).physics.terms.giss_convection
        self.assertEqual(
            term._target_,
            "jcm.physics.convection.giss_mstcnv.GissConvection")
        self.assertTrue(term.allow_mc)

    def test_it_builds(self):
        from jcm import runners

        physics = runners.build_physics(_cfg([self.PRESET]))
        built = [type(t).__name__ for t in physics.terms]
        self.assertIn("GissConvection", built)
        self.assertNotIn("TiedtkeConvection", built)
        self.assertEqual(built[built.index("GissConvection") + 1],
                         "Echam1MMicrophysics")

    def test_the_two_settings_stay_in_step_with_echam(self):
        """The preset copies `echam`'s top-level settings, so pin them."""
        giss = _cfg([self.PRESET]).physics
        echam = _cfg(["physics=echam"]).physics
        self.assertEqual(giss.vectorize_columns, echam.vectorize_columns)
        self.assertEqual(giss.checkpoint_terms, echam.checkpoint_terms)
        # Same term count: one convection scheme swapped for another.
        self.assertEqual(len(giss.terms), len(echam.terms))
