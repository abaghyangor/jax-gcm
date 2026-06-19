"""Tests for the ModelE DYCOMS oracle reader.

These tests run against the committed self-contained fixture
``jcm/data/test/modele/dycoms_oracle_subset.nc`` (extracted from the verified
ModelE DYCOMS-II RF02 SCM run). They prove that named diagnostics can be
resolved and extracted with documented shapes, units, and orientation. They do
NOT assert any JAX-vs-Fortran physics agreement (Milestone 1 is data only).
"""

import unittest

import numpy as np

from jcm.physics.modele import oracle


class TestOracleNames(unittest.TestCase):
    def setUp(self):
        self.path = oracle.fixture_path()

    def test_fixture_exists(self):
        import os
        self.assertTrue(os.path.exists(self.path), self.path)

    def test_decode_state_names(self):
        import netCDF4
        with netCDF4.Dataset(self.path) as ds:
            names = oracle.decode_names(ds.variables["sname_aijlh1"][:])
        # names must be stripped of fixed-width padding
        self.assertEqual(names, list(oracle.STATE_FIELDS))
        self.assertTrue(all(n == n.strip() for n in names))

    def test_decode_convection_and_column_names(self):
        import netCDF4
        with netCDF4.Dataset(self.path) as ds:
            conv = oracle.decode_names(ds.variables["sname_cijlh1"][:])
            col = oracle.decode_names(ds.variables["sname_cijh1"][:])
        self.assertEqual(conv, list(oracle.CONVECTION_FIELDS))
        self.assertEqual(col, list(oracle.COLUMN_FIELDS))

    def test_missing_name_raises(self):
        with self.assertRaises(KeyError):
            oracle.read_state_field(self.path, "not_a_real_field")


class TestOracleShapes(unittest.TestCase):
    def setUp(self):
        self.path = oracle.fixture_path()
        self.nperiod = 48
        self.lm = 63

    def test_state_field_all_periods_shape(self):
        t = oracle.read_state_field(self.path, "t")
        # (nperiod, jm, im, lm)
        self.assertEqual(t.shape, (self.nperiod, 1, 1, self.lm))

    def test_state_field_single_period_native_and_jcm(self):
        native = oracle.read_state_field(self.path, "t", period=24)
        self.assertEqual(native.shape, (1, 1, self.lm))  # (jm, im, lm)
        jcm = oracle.read_state_field(self.path, "t", period=24, jcm_column=True)
        self.assertEqual(jcm.shape, (self.lm, 1, 1))  # (vertical, x, y)

    def test_convection_fields_extractable(self):
        for name in ("dq_mc", "dth_mc", "qcl", "qci", "cldmc"):
            arr = oracle.read_convection_field(self.path, name, period=0,
                                               jcm_column=True)
            self.assertEqual(arr.shape, (self.lm, 1, 1), name)

    def test_column_fields_extractable(self):
        for name in ("prec", "mcp"):
            allp = oracle.read_column_field(self.path, name)
            self.assertEqual(allp.shape, (self.nperiod, 1, 1), name)
            one = oracle.read_column_field(self.path, name, period=0)
            self.assertEqual(one.shape, (1, 1), name)


class TestOracleUnitsAndOrientation(unittest.TestCase):
    """Verify physical units/scaling and the surface=index0 orientation."""

    def setUp(self):
        self.path = oracle.fixture_path()

    def test_temperature_physical_range(self):
        t = oracle.read_state_field(self.path, "t", period=24, jcm_column=True)
        t = np.asarray(t).ravel()
        self.assertTrue(np.all(t > 150) and np.all(t < 330), (t.min(), t.max()))

    def test_pressure_orientation_surface_is_index0(self):
        # p_3d[0] ~ surface (~1013 hPa), monotonically decreasing upward.
        p = np.asarray(
            oracle.read_state_field(self.path, "p_3d", period=24, jcm_column=True)
        ).ravel()
        self.assertGreater(p[0], 900.0)         # surface
        self.assertLess(p[-1], 1.0)             # model top
        self.assertTrue(np.all(np.diff(p) < 0))  # decreasing upward

    def test_scale_changes_values(self):
        # 'th' has scale 7.22; raw and scaled must differ.
        raw = oracle.read_state_field(self.path, "th", period=24, apply_scale=False)
        scaled = oracle.read_state_field(self.path, "th", period=24, apply_scale=True)
        self.assertFalse(np.allclose(np.asarray(raw), np.asarray(scaled)))

    def test_humidity_units_kgkg(self):
        q = np.asarray(
            oracle.read_state_field(self.path, "q", period=24, jcm_column=True)
        )
        # kg/kg water vapour: small positive, well under 0.05.
        self.assertTrue(np.all(q >= 0))
        self.assertLess(np.nanmax(q), 0.05)


class TestOracleColumnBundle(unittest.TestCase):
    def setUp(self):
        self.path = oracle.fixture_path()

    def test_read_oracle_column(self):
        col = oracle.read_oracle_column(self.path, period=24)
        self.assertEqual(col.t.shape, (63, 1, 1))
        # specific_humidity is g/kg = q_kgkg * 1000
        self.assertTrue(
            np.allclose(col.specific_humidity, col.q_kgkg * 1000.0)
        )
        self.assertGreater(np.nanmax(col.specific_humidity), np.nanmax(col.q_kgkg))

    def test_moist_convection_is_inactive_in_dycoms(self):
        """Key finding: DYCOMS-II RF02 oracle has zero moist convection.

        Across all 48 periods, dq_mc, dth_mc and mcp are identically zero -- all
        precipitation is stratiform. The eventual physics port therefore cannot
        be validated for nonzero MC behaviour against this oracle; a different
        SCM case (or allowMC enabled) is required. This test documents and locks
        that fact.
        """
        dq = np.asarray(oracle.read_convection_field(self.path, "dq_mc"))
        dth = np.asarray(oracle.read_convection_field(self.path, "dth_mc"))
        mcp = np.asarray(oracle.read_column_field(self.path, "mcp"))
        self.assertEqual(np.max(np.abs(dq)), 0.0)
        self.assertEqual(np.max(np.abs(dth)), 0.0)
        self.assertEqual(np.max(np.abs(mcp)), 0.0)

        # Sanity: total precip is nonzero and equals stratiform precip here.
        prec = np.asarray(oracle.read_column_field(self.path, "prec"))
        ssp = np.asarray(oracle.read_column_field(self.path, "ssp"))
        self.assertGreater(np.max(prec), 0.0)
        self.assertTrue(np.allclose(prec, ssp))


if __name__ == "__main__":
    unittest.main()
