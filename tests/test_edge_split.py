"""The edge-splitting pair search (``edge_slop``) -- the gates of stage C.

E1  option plumbing: validation, inert at full resolution (pair lists
    bitwise those of the plain instance), level_table / provenance, pickling,
    float32 search precision refused
E2  slop 0 is an exact regrouping: the child pixel pairs of the accepted
    pairs are exactly the full-resolution pairs of every bin (a partition);
    w(theta) == full resolution <= 1e-12 and Sigma w_i w_j <= 1e-10, every k,
    with and without a mask; xi+-, gamma_t of gg / kg / three_by_two_pt ==
    the per-pair moment reference over the mixed-level list <= 1e-10 (pixel
    ends without moments), for quad / dipole / plain cells
E3  every accepted pair satisfies the acceptance rule (range check on the
    stored list), at slop 0 and at slop 0.3
E4  Global2PCF(edge_slop=0) == one-patch Integrated3PCF(edge_slop=0)
    <= 1e-12, any tiling; several CPU "devices" == one; packed == unpacked
    (to the angle quantisation; kk bitwise); pack_host_pairs == pack_pairs
E5  pair files: round trip bitwise (Integrated3PCF format 5 attribute,
    Global2PCF format 3 attribute); a file with another edge_slop is refused,
    one with a slop is adopted (warning) by a None instance
E6  emulated GPU (unpacked and packed moment tiles with pixel ends) == CPU
E7  the preflight pair-count model is within 20 % of the split count
"""

import os
import pickle
import tempfile
import unittest
import warnings

import healpy as hp
import numpy as np
import pytest

from CosmoFuse import Global2PCF, Integrated3PCF
from CosmoFuse.edge_split import SplitNodes, check_accepted_pairs
from CosmoFuse.moments import (
    cell_frames,
    cell_moments_numpy,
    gammat_pair_terms,
    moment_factors,
    pair_angles,
    xipm_pair_terms,
)
from CosmoFuse.pair_geometry import PairGeometry
from CosmoFuse.treecode import estimate_pairs_per_bin, validate_edge_slop
from CosmoFuse.utils import pixel2RaDec
from CosmoFuse.zeta_writer import ZetaWriter

from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
CENTRE = (np.radians(60.0), np.radians(30.0))  # (theta, phi)
RADIUS = 8.0  # degrees
# nside 64 / 32 pixels: 55' / 110'.  k = 1.5 over 60'-480' gives levels
# [64, 64, 64, 32, 32]: the two coarse bins mix pixels and nside-32 cells.
BINS = dict(nbins=5, theta_min=60.0, theta_max=480.0)
K = 1.5


def _mask(holes=True):
    disc = hp.query_disc(NSIDE, hp.ang2vec(*CENTRE), np.radians(RADIUS))
    mask = np.zeros(NPIX, dtype=bool)
    mask[disc] = True
    if holes:
        rng = np.random.default_rng(1)
        mask[rng.choice(disc, disc.size // 20, replace=False)] = False
        hole = hp.query_disc(
            NSIDE, hp.ang2vec(np.radians(57.0), np.radians(27.0)), np.radians(1.2)
        )
        mask[hole] = False
    return mask


def _maps(seed=3, nz=1):
    rng = np.random.default_rng(seed)
    shear = rng.normal(size=(nz, 2, NPIX)) * 0.03
    dens = rng.normal(size=(nz, NPIX))
    w_s = rng.uniform(0.5, 2.0, size=(nz, NPIX))
    w_d = rng.uniform(0.5, 2.0, size=(nz, NPIX))
    return shear, dens, w_s, w_d


def _kw(**extra):
    kw = dict(
        mask=None,
        device="cpu",
        map_precision="float64",
        rotation_precision="float64",
        resolution_factor=K,
        cell_moments="quad",
        edge_slop=0.0,
    )
    kw.update(BINS)
    kw.update(extra)
    return kw


def _one_patch(mask, find=True, **extra):
    """An Integrated3PCF with ONE patch holding the whole footprint."""
    kw = _kw(mask=mask, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = Integrated3PCF(
            NSIDE,
            np.array([CENTRE[1]]),
            np.array([CENTRE[0]]),
            patch_size=(RADIUS + 0.5) * 60.0,
            theta_Q=10.0,
            **kw,
        )
    if find:
        corr.preprocess()
    return corr


def _global(mask, **extra):
    kw = _kw(mask=mask, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Global2PCF(NSIDE, **kw)
    g.preprocess()
    return g


def _rel(a, b):
    b = np.asarray(b)
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def _nodes(corr):
    """The row-space node table of a one-patch instance (its cells are the
    footprint's cells)."""
    return SplitNodes.from_rows(
        corr.nside, corr.map_inds, corr._treecode, corr.n_aperture_cells
    )


def _pair_bins(corr, patch=0):
    bins = np.asarray(corr.bins[patch], dtype=np.int64)
    return np.repeat(np.arange(bins.size), bins)


def _child_pixels(corr, nodes, local):
    """RING pixels of every node (a pixel: itself; a cell: its members)."""
    tree = corr._treecode
    n_active = corr.n_active
    starts = tree.level_starts(first=corr.n_aperture_cells)
    out = []
    for v in np.asarray(local):
        if v < n_active:
            out.append(np.array([int(nodes.gid[v])]))
        else:
            c = v - n_active
            level = int(np.searchsorted(starts, c, side="right") - 1)
            indptr, indices = tree.pixel_children(level)
            k = c - starts[level]
            out.append(indices[indptr[k] : indptr[k + 1]])
    return out


# ── E1 ────────────────────────────────────────────────────────────────────


class TestOptionPlumbing(unittest.TestCase):
    def test_validation(self):
        self.assertIsNone(validate_edge_slop(None))
        self.assertIsNone(validate_edge_slop(False))
        self.assertEqual(validate_edge_slop(0), 0.0)
        self.assertEqual(validate_edge_slop(0.25), 0.25)
        for bad in (True, -0.1, "0", float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_edge_slop(bad)
        with self.assertRaisesRegex(ValueError, "float64"):
            Integrated3PCF(
                NSIDE,
                np.zeros(1),
                np.zeros(1),
                edge_slop=0.0,
                pair_search_precision="float32",
                **BINS,
            )

    def test_inert_at_full_resolution(self):
        mask = _mask()
        plain = _one_patch(mask, resolution_factor=None, edge_slop=None)
        split = _one_patch(mask, resolution_factor=None, edge_slop=0.0)
        self.assertIsNone(split._treecode)
        np.testing.assert_array_equal(plain.bins[0], split.bins[0])
        np.testing.assert_array_equal(plain.pair_inds[0], split.pair_inds[0])
        np.testing.assert_array_equal(plain.pair_exp2phi[0], split.pair_exp2phi[0])
        self.assertEqual(split.level_table["edge_slop"], 0.0)
        self.assertIsNone(plain.level_table["edge_slop"])

    def test_provenance_and_pickle(self):
        corr = _one_patch(_mask(), find=False)
        self.assertEqual(corr.level_table["edge_slop"], 0.0)
        prov = ZetaWriter._collect_provenance(corr)
        self.assertEqual(prov["edge_slop"], 0.0)
        self.assertIn('"edge_slop": 0.0', prov["level_table"])
        back = pickle.loads(pickle.dumps(corr))
        self.assertEqual(back.edge_slop, 0.0)
        self.assertNotIn(
            "edge_slop",
            ZetaWriter._collect_provenance(_one_patch(_mask(), False, edge_slop=None)),
        )


# ── E2 / E3 ───────────────────────────────────────────────────────────────


class TestExactRegrouping(unittest.TestCase):
    """E2, E3: slop 0 partitions the full-resolution pairs bin by bin."""

    @classmethod
    def setUpClass(cls):
        cls.shear, cls.dens, cls.w_s, cls.w_d = _maps()

    def _full_pairs_per_bin(self, full):
        ids = np.asarray(full.pair_inds[0], dtype=np.int64)
        lo, hi = np.minimum(ids[0], ids[1]), np.maximum(ids[0], ids[1])
        code = lo * NPIX + hi
        bins = _pair_bins(full)
        return [np.sort(code[bins == b]) for b in range(full.nbins)]

    def _split_children_per_bin(self, split):
        nodes = _nodes(split)
        ids = np.asarray(split.pair_inds[0], dtype=np.int64)
        local = nodes.local_of_gid(ids, NPIX, split.n_active)
        kids_a = _child_pixels(split, nodes, local[0])
        kids_b = _child_pixels(split, nodes, local[1])
        bins = _pair_bins(split)
        out = [[] for _ in range(split.nbins)]
        n_mixed = n_cell = 0
        for ka, kb, b in zip(kids_a, kids_b, bins):
            p = np.repeat(ka, kb.size)
            q = np.tile(kb, ka.size)
            if ka.size > 1 and kb.size > 1:
                n_cell += 1
            elif ka.size > 1 or kb.size > 1:
                n_mixed += 1
            out[b].append(np.minimum(p, q) * NPIX + np.maximum(p, q))
        return [np.sort(np.concatenate(v)) for v in out], n_mixed, n_cell

    def test_partition_and_weight_sums(self):
        for holes in (False, True):
            mask = _mask(holes)
            full = _one_patch(mask, resolution_factor=None, edge_slop=None)
            ref_bins = self._full_pairs_per_bin(full)
            for k in (K, 2.0):
                with self.subTest(holes=holes, k=k):
                    split = _one_patch(mask, resolution_factor=k)
                    self.assertGreater(len(set(split.level_nside.tolist())), 1)
                    got, n_mixed, n_cell = self._split_children_per_bin(split)
                    # mixed-level pairs and cell pairs both occur
                    self.assertGreater(n_mixed, 0)
                    self.assertGreater(n_cell, 0)
                    self.assertLess(split.ntotpairs, full.ntotpairs)
                    for b in range(full.nbins):
                        np.testing.assert_array_equal(got[b], ref_bins[b])
                    # w(theta) has no phase: equals full resolution
                    ref = full.kk(
                        self.dens, self.w_d, apertures=False, return_device=False
                    ).xi_g
                    xi = split.kk(
                        self.dens, self.w_d, apertures=False, return_device=False
                    ).xi_g
                    self.assertLess(_rel(xi, ref), 1e-12)

    def test_global_weight_sums_equal_full_resolution(self):
        mask = _mask()
        full = _global(mask, resolution_factor=None, edge_slop=None)
        split = _global(mask)
        self.assertEqual(
            split.ntotpairs, sum(int(np.sum(b)) for b in split._engine.bins)
        )
        a = full.kk(self.dens, self.w_d, return_device=False)
        b = split.kk(self.dens, self.w_d, return_device=False)
        self.assertLess(_rel(b.weight, a.weight), 1e-10)
        self.assertLess(_rel(b.num["xi"], a.num["xi"]), 1e-10)
        g_full = full.gg(self.shear, self.w_s, flip_g1=True, return_device=False)
        g_split = split.gg(self.shear, self.w_s, flip_g1=True, return_device=False)
        self.assertLess(_rel(g_split.weight, g_full.weight), 1e-10)
        # xi+ has no phase window beyond the transport: close, not equal
        self.assertLess(_rel(g_split.xi_p, g_full.xi_p), 1e-3)

    def _reference(self, corr, order):
        """Per-bin (xi+, xi-, gamma_t) of patch 0 from the per-pair moment
        expansion over the stored mixed-level pair list: cells carry the
        moments of their members, pixels only their value."""
        nodes = _nodes(corr)
        tree = corr._treecode
        n_active = corr.n_active
        g = (self.shear[0, 0] + 1j * self.shear[0, 1])[corr.map_inds]
        w = self.w_s[0][corr.map_inds]
        d = self.dens[0][corr.map_inds]
        wd = self.w_d[0][corr.map_inds]
        nm = {0: 1, 1: 3, 2: 6}[order]
        pix_geom = cell_frames(*pixel2RaDec(np.asarray(corr.map_inds), NSIDE))
        mom_g = np.zeros((nodes.gid.size, 6), dtype=complex)
        mom_d = np.zeros((nodes.gid.size, 6), dtype=complex)
        wsum_s = np.zeros(nodes.gid.size)
        wsum_d = np.zeros(nodes.gid.size)
        mom_g[:n_active, 0] = w * g
        mom_d[:n_active, 0] = wd * d
        wsum_s[:n_active] = w
        wsum_d[:n_active] = wd
        starts = tree.level_starts(first=corr.n_aperture_cells)
        lut = corr._global_to_row_lut()
        for level in range(tree.n_levels):
            indptr, indices = tree.pixel_children(level)
            rows = indices if lut is None else lut[indices]
            geom = cell_frames(tree.cell_ra[level], tree.cell_dec[level])
            cell = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
            t, dx, dy = moment_factors(geom[cell], pix_geom[rows, 0:3])
            mg, _ = cell_moments_numpy(indptr, rows, t, dx, dy, w, g, max(order, 1))
            md, _ = cell_moments_numpy(indptr, rows, t, dx, dy, wd, d, max(order, 1))
            sl = slice(n_active + starts[level], n_active + starts[level + 1])
            mom_g[sl, : mg.shape[1]] = mg
            mom_d[sl, : md.shape[1]] = md
            wsum_s[sl] = np.add.reduceat(w[rows], indptr[:-1])
            wsum_d[sl] = np.add.reduceat(wd[rows], indptr[:-1])
            if order == 0:
                # plain cells: the untransported mean
                mom_g[sl, 0] = np.add.reduceat((w * g)[rows], indptr[:-1])
        ids = np.asarray(corr.pair_inds[0], dtype=np.int64)
        a, b = nodes.local_of_gid(ids, NPIX, n_active)
        phi_a, phi_b, inv_d = pair_angles(nodes.geom[a], nodes.geom[b])
        if order == 0:
            # plain cells rotate with e^{2i phi}: the moment formula at
            # order 0 with the stored rotation is the same thing
            pass
        xip, xim = xipm_pair_terms(
            mom_g[a, :nm], mom_g[b, :nm], phi_a, phi_b, inv_d, order
        )
        gt_ab = gammat_pair_terms(
            mom_d[a, :nm], mom_g[b, :nm], phi_a, phi_b, inv_d, order
        )
        gt_ba = gammat_pair_terms(
            mom_d[b, :nm], mom_g[a, :nm], phi_b, phi_a, inv_d, order
        )
        w_ss = wsum_s[a] * wsum_s[b]
        w_ds = wsum_d[a] * wsum_s[b] + wsum_d[b] * wsum_s[a]
        bins = _pair_bins(corr)
        nb = corr.nbins
        sums = lambda v: np.bincount(bins, weights=v, minlength=nb)  # noqa: E731
        return (
            sums(xip) / sums(w_ss),
            sums(xim) / sums(w_ss),
            sums(gt_ab + gt_ba) / sums(w_ds),
        )

    def test_kernels_equal_the_mixed_level_reference(self):
        mask = _mask()
        for cm, order in (("quad", 2), ("dipole", 1), (None, 0)):
            with self.subTest(cell_moments=cm):
                corr = _one_patch(mask, cell_moments=cm)
                want = self._reference(corr, order)
                gg = corr.gg(self.shear, self.w_s, apertures=False, return_device=False)
                kg = corr.kg(
                    self.dens, self.shear, self.w_d, self.w_s, return_device=False
                )
                res = corr.three_by_two_pt(
                    shear_maps=self.shear,
                    k_maps=self.dens,
                    weights=(self.w_s, self.w_d),
                    return_device=False,
                )
                got = {
                    "gg.xi_p": gg.xi_p[0, 0],
                    "gg.xi_m": gg.xi_m[0, 0],
                    "kg.xi_t": kg.xi_t[0, 0],
                    "3x2.xi_p": res.xi_p[0, 0],
                    "3x2.xi_m": res.xi_m[0, 0],
                    "3x2.xi_t": res.xi_t[0, 0],
                }
                ref = {
                    "gg.xi_p": want[0],
                    "gg.xi_m": want[1],
                    "kg.xi_t": want[2],
                    "3x2.xi_p": want[0],
                    "3x2.xi_m": want[1],
                    "3x2.xi_t": want[2],
                }
                for key in got:
                    self.assertLess(_rel(got[key], ref[key]), 1e-10, key)

    def test_acceptance_rule(self):
        mask = _mask()
        for slop in (0.0, 0.3):
            with self.subTest(slop=slop):
                corr = _one_patch(mask, edge_slop=slop)
                nodes = _nodes(corr)
                check = check_accepted_pairs(
                    nodes,
                    corr.pair_inds[0],
                    corr.bins[0],
                    corr.binedges,
                    corr.level_nside,
                    slop,
                )
                self.assertTrue(np.all(check["allowed"]))
                self.assertTrue(np.all(check["in_bin"]))
                self.assertLessEqual(check["overhang"].max(), slop + 1e-12)
                self.assertTrue(np.all(check["overhang"][check["pixel_pair"]] == 0))
                self.assertGreater(np.sum(~check["pixel_pair"]), 0)
                if slop > 0:
                    # a looser slop accepts coarser pairs: fewer of them
                    exact = _one_patch(mask, edge_slop=0.0)
                    self.assertLess(corr.ntotpairs, exact.ntotpairs)
                    self.assertGreater(check["overhang"].max(), 0.0)


# ── E4 ────────────────────────────────────────────────────────────────────


class TestEstimatorsAndBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _maps(nz=2)

    def _measure_global(self, g):
        gg = g.gg(self.shear, self.w_s, flip_g1=True, return_device=False)
        kk = g.kk(self.dens, self.w_d, return_device=False)
        kg = g.kg(
            self.dens, self.shear, self.w_d, self.w_s, flip_g1=True, return_device=False
        )
        return {"xi_p": gg.xi_p, "xi_m": gg.xi_m, "xi": kk.xi, "xi_t": kg.xi}

    def _measure_patch(self, p):
        gg = p.gg(
            self.shear, self.w_s, apertures=False, flip_g1=True, return_device=False
        )
        kk = p.kk(self.dens, self.w_d, apertures=False, return_device=False)
        kg = p.kg(
            self.dens, self.shear, self.w_d, self.w_s, flip_g1=True, return_device=False
        )
        return {
            "xi_p": gg.xi_p[:, 0],
            "xi_m": gg.xi_m[:, 0],
            "xi": kk.xi_g[:, 0],
            "xi_t": kg.xi_t[:, 0],
        }

    def test_global_equals_one_patch(self):
        p = _one_patch(self.mask)
        ref = self._measure_patch(p)
        for tile_nside in (2, 4, 8):
            with self.subTest(tile_nside=tile_nside):
                g = _global(self.mask, tile_nside=tile_nside)
                self.assertEqual(g.ntotpairs, p.ntotpairs)
                self.assertGreater(g._engine._pair_moments, 0)
                got = self._measure_global(g)
                for key in ref:
                    self.assertLess(_rel(got[key], ref[key]), 1e-12, key)

    def test_several_devices_equal_one(self):
        single = _global(self.mask, tile_nside=8)
        ref = self._measure_global(single)
        for devices in (["cpu", "cpu"], ["cpu", "cpu", "cpu"]):
            with self.subTest(n_devices=len(devices)):
                multi = _global(self.mask, tile_nside=8, device=devices)
                self.assertEqual(multi.ntotpairs, single.ntotpairs)
                got = self._measure_global(multi)
                for key in ref:
                    self.assertLess(_rel(got[key], ref[key]), 1e-13, key)

    def test_packed_equals_unpacked(self):
        exact = _one_patch(self.mask)
        packed = _one_patch(self.mask, pack_pairs=True)
        a, b = self._measure_patch(exact), self._measure_patch(packed)
        # kk reads no rotation: bitwise; the others to the angle quantisation
        np.testing.assert_array_equal(a["xi"], b["xi"])
        for key in ("xi_p", "xi_m", "xi_t"):
            self.assertLess(_rel(b[key], a[key]), 2e-4, key)
        # the host-packed payload builds the same device state
        host = _one_patch(self.mask, pack_pairs=True, pack_host_pairs=True)
        self.assertIsNone(host.pair_inds)
        c = self._measure_patch(host)
        for key in a:
            np.testing.assert_array_equal(c[key], b[key])
        g_exact = _global(self.mask, tile_nside=8)
        g_host = _global(self.mask, tile_nside=8, pack_pairs=True, pack_host_pairs=True)
        g_pack = _global(self.mask, tile_nside=8, pack_pairs=True)
        ra, rb, rc = (self._measure_global(g) for g in (g_exact, g_host, g_pack))
        for key in ra:
            np.testing.assert_array_equal(rb[key], rc[key])
            self.assertLess(_rel(rb[key], ra[key]), 2e-4, key)


# ── E5 ────────────────────────────────────────────────────────────────────


class TestFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _maps()

    def _measure(self, corr):
        gg = corr.gg(self.shear, self.w_s, apertures=False, return_device=False)
        kg = corr.kg(self.dens, self.shear, self.w_d, self.w_s, return_device=False)
        return gg.xi_p, gg.xi_m, kg.xi_t

    def test_integrated_round_trip_and_refusals(self):
        corr = _one_patch(self.mask)
        want = self._measure(corr)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            corr.save_pairs(path)
            import h5py

            with h5py.File(path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 5)
                self.assertEqual(float(fp.attrs["edge_slop"]), 0.0)
            same = _one_patch(self.mask, find=False)
            same.load_pairs(path)
            for a, b in zip(self._measure(same), want):
                np.testing.assert_array_equal(a, b)
            # a None instance adopts the file's slop, with a warning
            adopt = _one_patch(self.mask, find=False, edge_slop=None)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                adopt.load_pairs(path)
            self.assertTrue(any("edge_slop=0.0" in str(x.message) for x in w))
            self.assertEqual(adopt.edge_slop, 0.0)
            for a, b in zip(self._measure(adopt), want):
                np.testing.assert_array_equal(a, b)
            # another slop is another estimator
            other = _one_patch(self.mask, find=False, edge_slop=0.2)
            with self.assertRaisesRegex(ValueError, "edge_slop"):
                other.load_pairs(path)
            # a static file into a split instance, likewise
            static = _one_patch(self.mask, edge_slop=None)
            static.save_pairs(path)
            with self.assertRaisesRegex(ValueError, "edge_slop"):
                _one_patch(self.mask, find=False).load_pairs(path)
            # a full-resolution file is the same estimator whatever the slop
            full = _one_patch(self.mask, resolution_factor=None, edge_slop=None)
            full.save_pairs(path)
            loader = _one_patch(self.mask, find=False, resolution_factor=None)
            loader.load_pairs(path)
            self.assertEqual(loader.ntotpairs, full.ntotpairs)

    def test_global_round_trip_and_refusals(self):
        g = _global(self.mask, tile_nside=8)
        want = g.gg(self.shear, self.w_s, return_device=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g2.h5")
            g.save_pairs(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                same = Global2PCF(NSIDE, tile_nside=8, **_kw(mask=self.mask))
            same.load_pairs(path)
            got = same.gg(self.shear, self.w_s, return_device=False)
            np.testing.assert_array_equal(got.xi_p, want.xi_p)
            np.testing.assert_array_equal(got.xi_m, want.xi_m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                other = Global2PCF(
                    NSIDE, tile_nside=8, **_kw(mask=self.mask, edge_slop=0.1)
                )
            with self.assertRaisesRegex(ValueError, "edge_slop"):
                other.load_pairs(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adopt = Global2PCF(
                    NSIDE, tile_nside=8, **_kw(mask=self.mask, edge_slop=None)
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                adopt.load_pairs(path)
            self.assertTrue(any("edge_slop=0.0" in str(x.message) for x in w))
            self.assertEqual(adopt._engine.edge_slop, 0.0)
            got = adopt.gg(self.shear, self.w_s, return_device=False)
            np.testing.assert_array_equal(got.xi_m, want.xi_m)


# ── E6 ────────────────────────────────────────────────────────────────────


class TestEmulatedGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _maps(nz=2)

    def _measure(self, corr):
        res = corr.three_by_two_pt(
            shear_maps=self.shear,
            k_maps=self.dens,
            weights=(self.w_s, self.w_d),
            return_device=False,
        )
        gg = corr.gg(
            self.shear, self.w_s, flip_g1=True, return_device=False, apertures=False
        )
        kg = corr.kg(self.dens, self.shear, self.w_d, self.w_s, return_device=False)
        kk = corr.kk(self.dens, self.w_d, return_device=False, apertures=False)
        return (
            res.xi_p,
            res.xi_m,
            res.xi_g,
            res.xi_t,
            gg.xi_p,
            gg.xi_m,
            kg.xi_t,
            kk.xi_g,
        )

    def test_emulated_gpu_equals_cpu(self):
        for pack in (False, True):
            with self.subTest(pack_pairs=pack):
                corr = _one_patch(self.mask, pack_pairs=pack)
                cpu = self._measure(corr)
                with emulated_gpu(corr) as calls:
                    gpu = self._measure(corr)
                self.assertGreater(calls["xipm_tomo_moments_kernel"], 0)
                self.assertGreater(calls["kernel_density_shear_tomo_moments"], 0)
                self.assertGreater(calls["kernel_3x2pt_tomo_moments"], 0)
                for a, b in zip(cpu, gpu):
                    np.testing.assert_allclose(
                        b, a, rtol=0, atol=1e-12 * np.max(np.abs(a))
                    )


# ── E7 ────────────────────────────────────────────────────────────────────


class TestPreflightModel(unittest.TestCase):
    def test_model_within_20_percent(self):
        for holes in (False, True):
            mask = _mask(holes)
            for k in (K, 2.0):
                with self.subTest(holes=holes, k=k):
                    corr = _one_patch(mask, resolution_factor=k)
                    n_pix = np.array([PairGeometry.patch_pixels(corr, 0).size])
                    model = estimate_pairs_per_bin(
                        n_pix,
                        corr.binedges,
                        corr.level_nside,
                        NSIDE,
                        np.radians(corr.patch_size / 60.0),
                        corr.edge_slop,
                    )
                    actual = np.asarray(corr.bins[0], dtype=np.float64)
                    self.assertLess(abs(model.sum() / actual.sum() - 1.0), 0.2)
                    # and the split projection is far below the static one
                    # of the same k only where it should be: more pairs
                    static = estimate_pairs_per_bin(
                        n_pix,
                        corr.binedges,
                        corr.level_nside,
                        NSIDE,
                        np.radians(corr.patch_size / 60.0),
                        None,
                    )
                    coarse = corr.level_nside != NSIDE
                    self.assertTrue(np.all(model[coarse] > static[coarse]))
                    np.testing.assert_allclose(model[~coarse], static[~coarse])

    def test_preflight_uses_the_split_count(self):
        from CosmoFuse.pair_geometry import PairGeometry as PG

        mask = _mask()
        split = _one_patch(mask, find=False, memory_budget_gb=1e6)
        static = _one_patch(mask, find=False, edge_slop=None, memory_budget_gb=1e6)
        need_split = PG.preflight_pair_memory(split)["projected_gb"]
        need_static = PG.preflight_pair_memory(static)["projected_gb"]
        self.assertGreater(need_split, 2 * need_static)
        tight = _one_patch(
            mask, find=False, memory_budget_gb=0.5 * (need_split + need_static)
        )
        with self.assertRaisesRegex(MemoryError, "edge_slop=0"):
            tight.find_pairs()


# ── real GPU ──────────────────────────────────────────────────────────────


@pytest.mark.gpu
class TestRealGPU(unittest.TestCase):
    """The CUDA moment tiles with pixel ends (unpacked and packed) against
    the CPU backend, float64 maps, <= 1e-12; and the GPU search equals the
    CPU one (the search is host code either way)."""

    def test_pairs(self):
        mask = _mask()
        shear, dens, w_s, w_d = _maps(nz=2)
        for pack in (False, True):
            with self.subTest(pack_pairs=pack):
                cpu = _one_patch(mask, pack_pairs=pack)
                gpu = _one_patch(mask, pack_pairs=pack, device="gpu")
                self.assertEqual(gpu.ntotpairs, cpu.ntotpairs)
                ref = None
                for est in (cpu, gpu):
                    r = est.three_by_two_pt(
                        shear_maps=shear,
                        k_maps=dens,
                        weights=(w_s, w_d),
                        return_device=False,
                    )
                    g = est.gg(
                        shear, w_s, flip_g1=True, return_device=False, apertures=False
                    )
                    t = est.kg(dens, shear, w_d, w_s, return_device=False)
                    k = est.kk(dens, w_d, return_device=False, apertures=False)
                    out = [
                        r.xi_p,
                        r.xi_m,
                        r.xi_g,
                        r.xi_t,
                        g.xi_p,
                        g.xi_m,
                        t.xi_t,
                        k.xi_g,
                    ]
                    if ref is None:
                        ref = out
                    else:
                        for a, b in zip(ref, out):
                            np.testing.assert_allclose(
                                np.asarray(b),
                                np.asarray(a),
                                rtol=0,
                                atol=1e-12 * np.max(np.abs(a)),
                            )

    def test_global(self):
        mask = _mask()
        shear, dens, w_s, w_d = _maps(nz=2)
        cpu = _global(mask, tile_nside=8, pack_pairs=True)
        gpu = _global(mask, tile_nside=8, pack_pairs=True, device="gpu")
        a = cpu.gg(shear, w_s, flip_g1=True, return_device=False)
        b = gpu.gg(shear, w_s, flip_g1=True, return_device=False)
        self.assertLess(_rel(b.xi_m, a.xi_m), 1e-12)
        a = cpu.kg(dens, shear, w_d, w_s, return_device=False)
        b = gpu.kg(dens, shear, w_d, w_s, return_device=False)
        self.assertLess(_rel(b.xi, a.xi), 1e-12)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
