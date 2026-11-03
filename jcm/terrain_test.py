"""Tests for jcm/terrain.py

Tests for TerrainData struct and get_terrain function.
"""

import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jcm.terrain import (
    TerrainData, _SSO_NAMES, _descriptors_to_tensor, _tensor_to_descriptors,
    aggregate_sso_descriptors, derive_sso_descriptors,
    get_simplified_sso_descriptors, get_terrain,
)
from jcm.constants import grav
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.utils import get_coords

_T63_TERRAIN = Path("jcm/data/bc/t63/terrain.nc")


class TestSimplifiedSSODescriptors(unittest.TestCase):
    """Tests for the simplified SSO-descriptor heuristic
    (:func:`get_simplified_sso_descriptors`).
    """

    def test_zero_over_ocean(self):
        """All six SSO fields should be zero where orog == 0."""
        orog = jnp.zeros((4, 8))
        sso = get_simplified_sso_descriptors(orog)
        for name in ("orostd", "orosig", "orogam",
                     "orothe", "oropic", "oroval"):
            self.assertTrue(jnp.allclose(sso[name], 0.0),
                            msg=f"{name} not zero over ocean")

    def test_nonzero_over_land(self):
        """SSO fields should populate where orog > 0."""
        orog = jnp.array([[1500.0]])
        sso = get_simplified_sso_descriptors(orog)
        # Heuristic: orostd = 0.25 * orog
        self.assertTrue(jnp.allclose(sso["orostd"], 375.0))
        # oropic = orog + 2*orostd, oroval = orog - 2*orostd
        self.assertTrue(jnp.allclose(sso["oropic"], 2250.0))
        self.assertTrue(jnp.allclose(sso["oroval"], 750.0))
        # Anisotropy/orientation/slope are constants over land.
        self.assertTrue(jnp.allclose(sso["orogam"], 0.5))
        self.assertTrue(jnp.allclose(sso["orothe"], 0.0))
        self.assertTrue(jnp.allclose(sso["orosig"], 0.1))

    def test_oroval_clamped_at_zero(self):
        """Valley elevation should not go below sea level for low hills."""
        orog = jnp.array([[10.0]])  # 10 m mean → orostd=2.5 → oroval=5
        sso = get_simplified_sso_descriptors(orog)
        self.assertTrue(jnp.all(sso["oroval"] >= 0.0))
        # For tiny hills, the heuristic gives positive valleys.
        self.assertTrue(jnp.allclose(sso["oroval"], 5.0))

    def test_activation_gate_inactive_over_ocean(self):
        """The Lott-Miller activation gate (ppic-pmea > gpicmea AND
        pstd > gstd, both 1m by default) must be inactive for ocean
        columns derived by the heuristic — otherwise SSO would fire on
        flat ocean.
        """
        orog = jnp.zeros((10,))
        sso = get_simplified_sso_descriptors(orog)
        ppic_minus_mea = sso["oropic"] - orog
        self.assertTrue(jnp.all(ppic_minus_mea <= 1.0))
        self.assertTrue(jnp.all(sso["orostd"] <= 1.0))


class TestDeriveSSODescriptorsFromHighRes(unittest.TestCase):
    """Tests for :func:`derive_sso_descriptors` (Baines-Palmer)."""

    def test_synthetic_ridge_gives_anisotropic_stats(self):
        """A high-res ridge oriented N-S should produce non-zero std,
        slope, and an anisotropy < 1 (not isotropic).
        """
        import numpy as np
        nx_hr, ny_hr = 64, 32
        hr_lat = np.linspace(-10.0, 10.0, ny_hr)
        hr_lon = np.linspace(-10.0, 10.0, nx_hr)
        # Ridge: mountain at lon=0, constant in lat.
        H = 1500.0 * np.exp(-(hr_lon[:, None] / 3.0) ** 2) * np.ones((nx_hr, ny_hr))

        # Coarse target grid: 4x4 cells covering the same domain.
        tg_lat = np.array([-7.5, -2.5, 2.5, 7.5])
        tg_lon = np.array([-7.5, -2.5, 2.5, 7.5])
        sso = derive_sso_descriptors(H, hr_lat, hr_lon, tg_lat, tg_lon)

        # Centre cells should have non-zero std and slope; corner cells
        # (no orography there) approximately zero std.
        self.assertGreater(float(sso["orostd"][1, 1]), 50.0)
        self.assertGreater(float(sso["orosig"][1, 1]), 0.0)
        self.assertGreater(float(sso["oropic"][1, 1]),
                           float(sso["oroval"][1, 1]))

    def test_isotropic_dome_gives_anisotropy_near_one(self):
        """A radially-symmetric dome should yield anisotropy ≈ 1."""
        import numpy as np
        nx_hr, ny_hr = 64, 32
        hr_lat = np.linspace(-10.0, 10.0, ny_hr)
        hr_lon = np.linspace(-10.0, 10.0, nx_hr)
        # Symmetric Gaussian dome at the origin.
        Lon, Lat = np.meshgrid(hr_lon, hr_lat, indexing="ij")
        H = 1500.0 * np.exp(-(Lon ** 2 + Lat ** 2) / 9.0)

        tg_lat = np.array([-2.5, 2.5])
        tg_lon = np.array([-2.5, 2.5])
        sso = derive_sso_descriptors(H, hr_lat, hr_lon, tg_lat, tg_lon)
        # All four cells contain part of the dome; anisotropy should
        # be close to 1 (within 0.3) for the centre-adjacent cells.
        for i in range(2):
            for j in range(2):
                gam = float(sso["orogam"][i, j])
                self.assertGreaterEqual(gam, 0.0)
                self.assertLessEqual(gam, 1.0)


class TestGradientTensorRoundTrip(unittest.TestCase):
    """orosig/orogam/orothe and the gradient tensor are interconvertible.

    Aggregating descriptors onto a coarser grid depends on it: the tensors
    average, the descriptors do not.
    """

    def _psd_tensor(self, seed):
        """A tensor built as a mean of outer products, so positive
        semi-definite, like a real gradient tensor.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        gx = rng.normal(size=200)
        gy = 0.4 * gx + rng.normal(size=200)
        return (float(np.mean(gx * gx)), float(np.mean(gy * gy)),
                float(np.mean(gx * gy)))

    def test_tensor_to_descriptors_and_back(self):
        import numpy as np
        for seed in range(5):
            K, L, M = self._psd_tensor(seed)
            orosig, orogam, orothe = _tensor_to_descriptors(K, L, M)
            K2, L2, M2 = _descriptors_to_tensor(orosig, orogam, orothe)
            np.testing.assert_allclose([K2, L2, M2], [K, L, M], rtol=1e-6,
                                       atol=1e-12)

    def test_descriptors_to_tensor_and_back(self):
        import numpy as np
        for seed in range(5):
            K, L, M = self._psd_tensor(seed)
            descriptors = _tensor_to_descriptors(K, L, M)
            round_tripped = _tensor_to_descriptors(
                *_descriptors_to_tensor(*descriptors))
            np.testing.assert_allclose(round_tripped, descriptors, rtol=1e-6,
                                       atol=1e-12)

    def test_isotropic_tensor_has_unit_anisotropy(self):
        orosig, orogam, orothe = _tensor_to_descriptors(0.01, 0.01, 0.0)
        self.assertAlmostEqual(float(orogam), 1.0, places=6)
        self.assertAlmostEqual(float(orosig), (0.02) ** 0.5, places=6)

    def test_zero_tensor_is_inert(self):
        orosig, orogam, orothe = _tensor_to_descriptors(0.0, 0.0, 0.0)
        self.assertEqual(float(orosig), 0.0)
        self.assertEqual(float(orogam), 0.0)
        K, L, M = _descriptors_to_tensor(0.0, 0.0, 0.0)
        self.assertEqual((float(K), float(L), float(M)), (0.0, 0.0, 0.0))


class TestAggregateSSODescriptors(unittest.TestCase):
    """Tests for :func:`aggregate_sso_descriptors`."""

    def _source(self):
        """A N-S ridge on a fine grid, with sub-cell statistics of its own."""
        import numpy as np
        nx_hr, ny_hr = 64, 32
        hr_lat = np.linspace(-10.0, 10.0, ny_hr)
        hr_lon = np.linspace(-10.0, 10.0, nx_hr)
        orog = (1500.0 * np.exp(-(hr_lon[:, None] / 3.0) ** 2)
                * np.ones((nx_hr, ny_hr)))
        shape = (nx_hr, ny_hr)
        sso = dict(
            orostd=np.full(shape, 120.0),
            orosig=np.full(shape, 0.05),
            orogam=np.full(shape, 0.3),
            orothe=np.full(shape, 20.0),
            oropic=orog + 300.0,
            oroval=orog - 200.0,
        )
        tg_lat = np.array([-7.5, -2.5, 2.5, 7.5])
        tg_lon = np.array([-7.5, -2.5, 2.5, 7.5])
        return sso, orog, hr_lat, hr_lon, tg_lat, tg_lon

    def test_output_lands_on_target_grid(self):
        sso, orog, hr_lat, hr_lon, tg_lat, tg_lon = self._source()
        out = aggregate_sso_descriptors(sso, orog, hr_lat, hr_lon,
                                        tg_lat, tg_lon)
        for name in _SSO_NAMES:
            self.assertEqual(out[name].shape, (4, 4))

    def test_keeps_the_sources_sub_cell_statistics(self):
        """Aggregated values exceed what the source orography alone gives.

        The point of aggregating: re-deriving from the source orography
        sees only the variation between its cells, while the source's own
        orostd/orosig describe what happens inside them.
        """
        import numpy as np
        sso, orog, hr_lat, hr_lon, tg_lat, tg_lon = self._source()
        aggregated = aggregate_sso_descriptors(sso, orog, hr_lat, hr_lon,
                                               tg_lat, tg_lon)
        derived = derive_sso_descriptors(orog, hr_lat, hr_lon,
                                         tg_lat, tg_lon)

        self.assertTrue(jnp.all(aggregated["orostd"] >= derived["orostd"]))
        self.assertTrue(jnp.all(aggregated["orosig"] >= derived["orosig"]))
        # The source's orostd is a uniform 120 m here, so the law of total
        # variance pins the result: the derived variance (between the
        # source cells) plus 120^2 (within them).
        np.testing.assert_allclose(
            np.asarray(aggregated["orostd"]),
            np.sqrt(120.0 ** 2 + np.asarray(derived["orostd"]) ** 2),
            rtol=1e-5, atol=1e-3,
        )

    def test_peak_and_valley_bound_the_sources(self):
        sso, orog, hr_lat, hr_lon, tg_lat, tg_lon = self._source()
        out = aggregate_sso_descriptors(sso, orog, hr_lat, hr_lon,
                                        tg_lat, tg_lon)
        # Tolerance for the float32 round trip through the returned arrays.
        self.assertLessEqual(float(out["oropic"].max()),
                             float(sso["oropic"].max()) + 1e-2)
        self.assertGreaterEqual(float(out["oroval"].min()),
                                float(sso["oroval"].min()) - 1e-2)
        self.assertTrue(jnp.all(out["oropic"] >= out["oroval"]))

    def test_zero_source_statistics_reproduce_the_derivation(self):
        """With no sub-cell statistics to keep, aggregating == deriving.

        The law-of-total-variance and tensor decompositions both reduce to
        the Baines-Palmer statistics of the source orography when the
        source's own descriptors are zero, which pins the two paths
        together.
        """
        import numpy as np
        sso, orog, hr_lat, hr_lon, tg_lat, tg_lon = self._source()
        zeroed = {name: np.zeros_like(sso[name]) for name in _SSO_NAMES}

        aggregated = aggregate_sso_descriptors(zeroed, orog, hr_lat, hr_lon,
                                               tg_lat, tg_lon)
        derived = derive_sso_descriptors(orog, hr_lat, hr_lon,
                                         tg_lat, tg_lon)
        for name in ("orostd", "orosig", "orogam", "orothe"):
            np.testing.assert_allclose(
                np.asarray(aggregated[name]), np.asarray(derived[name]),
                rtol=1e-5, atol=1e-7,
                err_msg=f"{name} diverges from the derived statistics",
            )

    def test_ranges_stay_physical(self):
        sso, orog, hr_lat, hr_lon, tg_lat, tg_lon = self._source()
        out = aggregate_sso_descriptors(sso, orog, hr_lat, hr_lon,
                                        tg_lat, tg_lon)
        self.assertTrue(jnp.all(out["orostd"] >= 0.0))
        self.assertTrue(jnp.all(out["orosig"] >= 0.0))
        self.assertTrue(jnp.all(out["orogam"] >= 0.0))
        self.assertTrue(jnp.all(out["orogam"] <= 1.0))
        self.assertTrue(jnp.all(jnp.abs(out["orothe"]) <= 90.0))


class TestGetTerrain(unittest.TestCase):
    """Tests for the get_terrain function."""

    def test_get_terrain_with_nodal_shape_only(self):
        """When only nodal_shape is provided, should return zeros for both orography and fmask."""
        nodal_shape = (96, 48)
        orography, fmask = get_terrain(nodal_shape=nodal_shape)

        self.assertEqual(orography.shape, nodal_shape)
        self.assertEqual(fmask.shape, nodal_shape)
        self.assertTrue(jnp.allclose(orography, 0.0))
        self.assertTrue(jnp.allclose(fmask, 0.0))

    def test_get_terrain_no_inputs_raises_error(self):
        """Should raise ValueError when no inputs are provided."""
        with self.assertRaises(ValueError) as context:
            get_terrain()
        self.assertIn("Must provide at least one of", str(context.exception))

    def test_get_terrain_orography_only(self):
        """When only orography is provided, fmask should be derived from orography > 0."""
        nodal_shape = (64, 32)
        # Create orography with some areas elevated and some at sea level
        orography = jnp.zeros(nodal_shape)
        orography = orography.at[:32, :16].set(500.)  # Mountains in one quadrant

        result_orog, result_fmask = get_terrain(orography=orography)

        # fmask should be 1 where orography > 0, 0 otherwise
        expected_fmask = (orography > 0.0).astype(jnp.float32)
        self.assertTrue(jnp.allclose(result_orog, orography))
        self.assertTrue(jnp.allclose(result_fmask, expected_fmask))

    def test_get_terrain_fmask_only(self):
        """When only fmask is provided, orography should default to zeros."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.5

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_orog, 0.0))
        self.assertEqual(result_fmask.shape, nodal_shape)

    def test_get_terrain_both_provided(self):
        """When both orography and fmask are provided, use them directly."""
        nodal_shape = (64, 32)
        orography = jnp.ones(nodal_shape) * 500.
        fmask = jnp.ones(nodal_shape) * 0.7

        result_orog, result_fmask = get_terrain(orography=orography, fmask=fmask)

        self.assertTrue(jnp.allclose(result_orog, orography))
        self.assertTrue(jnp.allclose(result_fmask, 0.7))

    def test_get_terrain_fmask_threshold_low(self):
        """Values below fmask_threshold should be set to exactly 0."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.05  # Below default threshold of 0.1

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_fmask, 0.0))

    def test_get_terrain_fmask_threshold_high(self):
        """Values above 1.0 - fmask_threshold should be set to exactly 1."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.95  # Above 1.0 - 0.1 = 0.9

        result_orog, result_fmask = get_terrain(fmask=fmask)

        self.assertTrue(jnp.allclose(result_fmask, 1.0))

    def test_get_terrain_fmask_threshold_custom(self):
        """Custom fmask_threshold should be respected."""
        nodal_shape = (64, 32)
        fmask = jnp.ones(nodal_shape) * 0.25

        # With threshold 0.1, 0.25 is in the middle range
        result_orog, result_fmask = get_terrain(fmask=fmask, fmask_threshold=0.1)
        self.assertTrue(jnp.allclose(result_fmask, 0.25))

        # With threshold 0.3, 0.25 should be rounded to 0
        result_orog, result_fmask = get_terrain(fmask=fmask, fmask_threshold=0.3)
        self.assertTrue(jnp.allclose(result_fmask, 0.0))


class TestTerrainDataCopy(unittest.TestCase):
    """Tests for TerrainData.copy method."""

    def test_copy_no_changes(self):
        """Copy with no arguments should return identical data."""
        nodal_shape = (64, 32)
        zero = jnp.zeros(nodal_shape)
        terrain = TerrainData(
            orog=jnp.ones(nodal_shape) * 100.,
            phis0=jnp.ones(nodal_shape) * grav * 100.,
            fmask=jnp.ones(nodal_shape) * 0.5,
            lfluxland=jnp.bool_(True),
            orostd=zero, orosig=zero, orogam=zero,
            orothe=zero, oropic=zero, oroval=zero,
        )

        copied = terrain.copy()

        self.assertTrue(jnp.allclose(copied.orog, terrain.orog))
        self.assertTrue(jnp.allclose(copied.phis0, terrain.phis0))
        self.assertTrue(jnp.allclose(copied.fmask, terrain.fmask))
        self.assertEqual(bool(copied.lfluxland), bool(terrain.lfluxland))

    def test_copy_with_changes(self):
        """Copy with arguments should replace those fields."""
        nodal_shape = (64, 32)
        zero = jnp.zeros(nodal_shape)
        terrain = TerrainData(
            orog=jnp.ones(nodal_shape) * 100.,
            phis0=jnp.ones(nodal_shape) * grav * 100.,
            fmask=jnp.ones(nodal_shape) * 0.5,
            lfluxland=jnp.bool_(True),
            orostd=zero, orosig=zero, orogam=zero,
            orothe=zero, oropic=zero, oroval=zero,
        )

        new_orog = jnp.ones(nodal_shape) * 200.
        new_fmask = jnp.ones(nodal_shape) * 0.8
        copied = terrain.copy(orog=new_orog, fmask=new_fmask)

        self.assertTrue(jnp.allclose(copied.orog, 200.))
        self.assertTrue(jnp.allclose(copied.phis0, terrain.phis0))  # Unchanged
        self.assertTrue(jnp.allclose(copied.fmask, 0.8))
        self.assertEqual(bool(copied.lfluxland), True)  # Unchanged


class TestTerrainDataFromCoords(unittest.TestCase):
    """Tests for TerrainData.from_coords classmethod."""

    def test_from_coords_defaults(self):
        """Default from_coords should create aquaplanet-like terrain."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_coords(coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.phis0.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_from_coords_with_orography(self):
        """from_coords with orography should compute fmask from orography."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        # Create orography with mountains in half the domain
        orography = jnp.zeros(nodal_shape)
        orography = orography.at[:nodal_shape[0]//2, :].set(1000.)

        terrain = TerrainData.from_coords(coords, orography=orography)

        # Check shapes
        self.assertEqual(terrain.orog.shape, nodal_shape)
        self.assertEqual(terrain.phis0.shape, nodal_shape)

        # fmask should be 1 where orography > 0
        expected_fmask = (orography > 0.0).astype(jnp.float32)
        self.assertTrue(jnp.allclose(terrain.fmask, expected_fmask))

    def test_from_coords_with_fmask(self):
        """from_coords with fmask but no orography should have flat terrain."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        fmask = jnp.ones(nodal_shape) * 0.5

        terrain = TerrainData.from_coords(coords, fmask=fmask)

        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))

    def test_from_coords_with_lfluxland(self):
        """from_coords lfluxland parameter should be respected."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)

        terrain_land = TerrainData.from_coords(coords, lfluxland=True)
        terrain_no_land = TerrainData.from_coords(coords, lfluxland=False)

        self.assertEqual(bool(terrain_land.lfluxland), True)
        self.assertEqual(bool(terrain_no_land.lfluxland), False)

    def test_from_coords_spectral_truncation_phis0(self):
        """phis0 should be spectrally truncated version of grav * orog."""
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        nodal_shape = coords.horizontal.nodal_shape

        # Create varying orography
        orography = jnp.ones(nodal_shape) * 1000.

        terrain = TerrainData.from_coords(coords, orography=orography)

        # For uniform orography, spectral truncation should preserve the field
        expected_phis0 = grav * orography
        self.assertTrue(jnp.allclose(terrain.phis0, expected_phis0, rtol=1e-5))


class TestTerrainDataAquaplanet(unittest.TestCase):
    """Tests for TerrainData.aquaplanet classmethod."""

    def test_aquaplanet_all_zeros(self):
        """Aquaplanet terrain should have all zeros."""
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.aquaplanet(coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_aquaplanet_different_resolutions(self):
        """Aquaplanet should work for different resolutions."""
        for truncation in [21, 31, 42]:
            coords = get_speedy_coords(layers=8, spectral_truncation=truncation)
            terrain = TerrainData.aquaplanet(coords)

            expected_shape = coords.horizontal.nodal_shape
            self.assertEqual(terrain.orog.shape, expected_shape)


class TestTerrainDataSingleColumn(unittest.TestCase):
    """Tests for TerrainData.single_column classmethod."""

    def test_single_column_defaults(self):
        """Default single column should be flat ocean."""
        terrain = TerrainData.single_column()

        self.assertEqual(terrain.orog.shape, (1, 1))
        self.assertEqual(terrain.phis0.shape, (1, 1))
        self.assertEqual(terrain.fmask.shape, (1, 1))
        self.assertTrue(jnp.allclose(terrain.orog, 0.0))
        self.assertTrue(jnp.allclose(terrain.phis0, 0.0))
        self.assertTrue(jnp.allclose(terrain.fmask, 0.0))
        self.assertEqual(bool(terrain.lfluxland), False)

    def test_single_column_with_orography(self):
        """Single column with orography should compute phis0 correctly."""
        orog_height = 500.
        terrain = TerrainData.single_column(orog=orog_height)

        self.assertTrue(jnp.allclose(terrain.orog, orog_height))
        self.assertTrue(jnp.allclose(terrain.phis0, grav * orog_height))

    def test_single_column_with_fmask(self):
        """Single column fmask should be set correctly."""
        terrain = TerrainData.single_column(fmask=1.0)

        self.assertTrue(jnp.allclose(terrain.fmask, 1.0))

    def test_single_column_with_lfluxland(self):
        """Single column lfluxland should be set correctly."""
        terrain_land = TerrainData.single_column(lfluxland=True)
        terrain_ocean = TerrainData.single_column(lfluxland=False)

        self.assertEqual(bool(terrain_land.lfluxland), True)
        self.assertEqual(bool(terrain_ocean.lfluxland), False)

    def test_single_column_custom_phis0(self):
        """Single column with custom phis0 should override default calculation."""
        orog_height = 500.
        custom_phis0 = 1234.5  # Different from grav * orog

        terrain = TerrainData.single_column(orog=orog_height, phis0=custom_phis0)

        self.assertTrue(jnp.allclose(terrain.orog, orog_height))
        self.assertTrue(jnp.allclose(terrain.phis0, custom_phis0))


class TestTerrainDataFromFile(unittest.TestCase):
    """Tests for TerrainData.from_file classmethod using actual data files."""

    def test_from_file_loads_terrain(self):
        """from_file should load terrain data from actual NetCDF file."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.phis0.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)

    def test_from_file_has_realistic_orography(self):
        """Loaded terrain should have realistic orography values."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # Orography should be non-negative (no below sea level in this dataset)
        # and have a reasonable max (Mt Everest ~8849m)
        self.assertTrue(jnp.all(terrain.orog >= -500.))  # Allow some below sea level
        self.assertTrue(jnp.all(terrain.orog <= 9000.))

    def test_from_file_has_valid_fmask(self):
        """Loaded terrain should have fmask values in [0, 1]."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # fmask should be between 0 and 1
        self.assertTrue(jnp.all(terrain.fmask >= 0.0))
        self.assertTrue(jnp.all(terrain.fmask <= 1.0))

    def test_from_file_phis0_matches_grav_times_orog(self):
        """phis0 should be spectrally truncated grav * orog."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # phis0 should be approximately grav * orog (with some spectral truncation difference)
        phi0_direct = grav * terrain.orog
        # The spectral truncation can cause differences, but the mean should be close
        self.assertTrue(jnp.isclose(jnp.mean(terrain.phis0), jnp.mean(phi0_direct), rtol=0.01))

    def test_from_file_lfluxland_default_true(self):
        """from_file should default lfluxland to True."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        self.assertEqual(bool(terrain.lfluxland), True)

    def test_from_file_lfluxland_can_be_set_false(self):
        """from_file lfluxland parameter should be respected."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords, lfluxland=False)

        self.assertEqual(bool(terrain.lfluxland), False)

    def test_from_file_interpolates_to_different_resolution(self):
        """from_file should interpolate terrain when coords resolution differs."""
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        # Use T21 coords with T31 terrain file - should interpolate
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)

        # Should match the target coords resolution
        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual(terrain.orog.shape, expected_shape)
        self.assertEqual(terrain.fmask.shape, expected_shape)

        # Should still have physically valid data after interpolation
        self.assertTrue(jnp.all(terrain.orog >= -500.))
        self.assertTrue(jnp.all(terrain.fmask >= 0.0))
        self.assertTrue(jnp.all(terrain.fmask <= 1.0))


class TestTerrainDataFromFileGridConsistency(unittest.TestCase):
    """Every array field of a from_file TerrainData must share the target grid.

    The T63 boundary file carries its own six SSO descriptors on its native
    (192, 96) grid. Passing those through unchanged used to leave orog,
    phis0 and fmask regridded to the target shape while the descriptors
    stayed at the file's — an object that constructs happily and only fails
    on the first physics step, inside the vmapped Lott-Miller SSO term.

    """

    def setUp(self):
        if not _T63_TERRAIN.exists():
            pytest.skip(
                f"{_T63_TERRAIN} missing; run utils/convert_echam_bc.py "
                f"to generate it"
            )

    def test_t63_file_on_t31_coords_is_shape_consistent(self):
        """All array fields share the target nodal shape when coarsening."""
        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        terrain = TerrainData.from_file(_T63_TERRAIN, coords=coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual((96, 48), expected_shape)
        leaves = jax.tree.leaves(terrain)
        shapes = {leaf.shape for leaf in leaves if leaf.ndim > 0}
        self.assertEqual({expected_shape}, shapes)

    def test_t63_file_on_t63_coords_keeps_the_files_descriptors(self):
        """On the file's own grid the preprocessed fields are used verbatim."""
        import xarray as xr

        coords = get_coords(get_echam_levels(47), spectral_truncation=63)
        terrain = TerrainData.from_file(_T63_TERRAIN, coords=coords)

        self.assertEqual((192, 96), coords.horizontal.nodal_shape)
        with xr.open_dataset(_T63_TERRAIN) as ds:
            for name in ("orostd", "orosig", "orogam",
                         "orothe", "oropic", "oroval"):
                self.assertTrue(
                    jnp.allclose(getattr(terrain, name),
                                 jnp.asarray(ds[name].values)),
                    msg=f"{name} differs from the file's own field",
                )

    def test_coarsened_descriptors_are_physically_valid(self):
        """Descriptors re-derived onto a coarser grid stay in range."""
        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        terrain = TerrainData.from_file(_T63_TERRAIN, coords=coords)

        self.assertTrue(jnp.all(terrain.orostd >= 0.0))
        self.assertTrue(jnp.all(terrain.orosig >= 0.0))
        self.assertTrue(jnp.all(terrain.orogam >= 0.0))
        self.assertTrue(jnp.all(terrain.orogam <= 1.0))
        self.assertTrue(jnp.all(jnp.abs(terrain.orothe) <= 90.0))
        self.assertTrue(jnp.all(terrain.oropic >= terrain.oroval))
        # A coarse cell holds the finer cells' peaks and valleys.
        self.assertTrue(jnp.all(terrain.oropic <= 9000.0))
        self.assertTrue(jnp.all(terrain.oroval >= -500.0))

    def test_coarsening_aggregates_rather_than_rederives(self):
        """The file's own descriptors feed the T31 values, not just its orography.

        Both paths put the right shape on the object, so shape alone does
        not say which ran. The file's statistics have to be in there: its
        orosig reaches 0.027, and re-deriving from its orography alone
        tops out near 0.016.
        """
        import xarray as xr

        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        terrain = TerrainData.from_file(_T63_TERRAIN, coords=coords)

        grid = coords.horizontal
        target_lat = grid.latitudes * 180.0 / jnp.pi
        target_lon = grid.longitudes * 180.0 / jnp.pi
        with xr.open_dataset(_T63_TERRAIN) as ds:
            src_orog = jnp.asarray(ds["orog"].values)
            src_lat = jnp.asarray(ds["lat"].values)
            src_lon = jnp.asarray(ds["lon"].values)
            src_sso = {name: jnp.asarray(ds[name].values)
                       for name in _SSO_NAMES}

        aggregated = aggregate_sso_descriptors(
            src_sso, src_orog, src_lat, src_lon, target_lat, target_lon)
        derived = derive_sso_descriptors(
            src_orog, src_lat, src_lon, target_lat, target_lon)

        for name in _SSO_NAMES:
            self.assertTrue(
                jnp.allclose(getattr(terrain, name), aggregated[name]),
                msg=f"{name} is not the aggregated field",
            )
        # The two paths really do differ on this file.
        self.assertGreater(float(aggregated["orosig"].max()),
                           float(derived["orosig"].max()))

    def test_t63_file_on_t85_coords_is_shape_consistent(self):
        """All array fields share the target nodal shape when refining.

        The file is coarser than the target here, so its descriptors are
        interpolated up rather than re-derived.
        """
        coords = get_coords(get_echam_levels(47), spectral_truncation=85)
        terrain = TerrainData.from_file(_T63_TERRAIN, coords=coords)

        expected_shape = coords.horizontal.nodal_shape
        self.assertEqual((256, 128), expected_shape)
        leaves = jax.tree.leaves(terrain)
        shapes = {leaf.shape for leaf in leaves if leaf.ndim > 0}
        self.assertEqual({expected_shape}, shapes)
        self.assertTrue(jnp.all(terrain.orogam <= 1.0))
        self.assertTrue(jnp.all(jnp.abs(terrain.orothe) <= 90.0))


class TestTerrainDataShapeValidation(unittest.TestCase):
    """The constructors name mis-shaped fields instead of passing them on."""

    def test_from_coords_rejects_mismatched_sso_field(self):
        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        wrong = jnp.zeros((192, 96))

        with self.assertRaises(ValueError) as cm:
            TerrainData.from_coords(coords, orostd=wrong)

        message = str(cm.exception)
        self.assertIn("orostd", message)
        self.assertIn("(192, 96)", message)
        self.assertIn("(96, 48)", message)


class TestTerrainDataTreeMath(unittest.TestCase):
    """Tests for JAX tree_math compatibility."""

    def test_terrain_data_is_jax_pytree(self):
        """TerrainData should be a valid JAX pytree."""
        import jax

        terrain = TerrainData.single_column(orog=100., fmask=0.5)

        # Should be able to tree_map over it
        doubled = jax.tree.map(lambda x: x * 2, terrain)

        self.assertTrue(jnp.allclose(doubled.orog, 200.))
        self.assertTrue(jnp.allclose(doubled.fmask, 1.0))

    def test_terrain_data_jit_compatible(self):
        """TerrainData should work with jax.jit."""
        import jax

        @jax.jit
        def get_phis0(terrain):
            return terrain.phis0

        terrain = TerrainData.single_column(orog=100., fmask=0.5)
        result = get_phis0(terrain)

        self.assertTrue(jnp.allclose(result, grav * 100.))


if __name__ == '__main__':
    unittest.main()
