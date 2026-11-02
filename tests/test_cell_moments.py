"""Moment-carrying treecode cells (``cell_moments``) -- the gates of stage B.

C1  the plain cells are untouched: ``cell_moments=None`` builds no moment
    geometry and ``moment_order`` is 0; full resolution ignores the option
C2  conventions: the angles of ``moments.pair_angles`` reproduce the pair
    finder's e^{2i phi}; per coarse pair the mono / dipole / quad residuals
    against the exact child sums scale as 4 dphi : 8 dphi^2 : (4 dphi)^3 / 6
    for xi- and 2 dphi : 2 dphi^2 : (2 dphi)^3 / 6 for gamma_t -- which fixes
    the sign S and the second-order terms
C3  the CPU kernels equal the numpy reference sums over the coarse pairs
    (<= 1e-10 relative) for xi+-, gamma_t, in gg / kg / three_by_two_pt;
    xi_g and kk are bitwise the plain cells
C4  the quad estimator of a smooth field is within 0.5 % of the exact child
    sums in every coarse bin, and much closer than the plain cells
C5  emulated GPU (unpacked and packed) == CPU; packed == unpacked to the
    angle quantisation; Global2PCF == one-patch Integrated3PCF
"""

import unittest
import warnings

import healpy as hp
import numpy as np
import pytest

from pixpoint import Global2PCF, Integrated3PCF, U_crittenden
from pixpoint.moments import (
    MOMENT_SIGN,
    cell_frames,
    cell_moments_numpy,
    frames,
    gammat_pair_terms,
    moment_factors,
    pair_angles,
    unit_vectors,
    xipm_pair_terms,
)
from pixpoint.utils import pixel2RaDec

from ._results import fields_of
from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
# nside 128 / 64 / 32 pixels: 27' / 55' / 110'.  k = 2 over 60'-600' gives
# levels [128, 128, 64, 32, 32]: two fine bins and three coarse ones.
BINS = dict(nbins=5, theta_min=60.0, theta_max=600.0)
PATCH = dict(patch_size=700.0, theta_Q=200.0)
PHI_C = np.radians([40.0, 70.0])
THETA_C = np.radians([95.0, 60.0])


def _mask():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (np.degrees(phi) > 20.0) & (np.degrees(phi) < 100.0)
    rng = np.random.default_rng(5)
    mask[rng.choice(NPIX, NPIX // 50, replace=False)] = False
    return mask


def _corr(mask, cell_moments, k=2.0, **kw):
    args = dict(
        mask=mask,
        device="cpu",
        map_precision="float64",
        rotation_precision="float64",
        resolution_factor=k,
        cell_moments=cell_moments,
    )
    args.update(BINS)
    args.update(PATCH)
    args.update(kw)
    corr = Integrated3PCF(NSIDE, PHI_C, THETA_C, **args)
    corr.preprocess()
    return corr


def _random_maps(seed=3, nz=2):
    rng = np.random.default_rng(seed)
    shear = rng.normal(size=(nz, 2, NPIX)) * 0.03
    dens = rng.normal(size=(nz, NPIX))
    w_s = rng.uniform(0.5, 2.0, size=(nz, NPIX))
    w_d = rng.uniform(0.5, 2.0, size=(nz, NPIX))
    return shear, dens, w_s, w_d


def _smooth_maps(seed=7, nz=1):
    """Gaussian convergence -> E-mode shear, with power down to the pixel
    scale so that xi- is not negligible against xi+ at 100'-600'."""
    lmax = 3 * NSIDE - 1
    ell = np.arange(lmax + 1)
    cl = 1e-6 * (np.maximum(ell, 1) / 20.0) ** -1.0 * np.exp(-((ell / 300.0) ** 2))
    cl[:2] = 0
    rng = np.random.default_rng(seed)
    shear = np.zeros((nz, 2, NPIX))
    dens = np.zeros((nz, NPIX))
    for z in range(nz):
        np.random.seed(int(rng.integers(1 << 31)))
        kappa = hp.synfast(cl, NSIDE, lmax=lmax)
        f = np.sqrt(
            np.clip((ell + 2) * (ell - 1), 0, None) / np.maximum(ell * (ell + 1), 1)
        )
        elm = hp.almxfl(hp.map2alm(kappa, lmax=lmax), f)
        g1, g2 = hp.alm2map_spin([elm, 0 * elm], NSIDE, 2, lmax)
        shear[z, 0], shear[z, 1] = g1, g2
        dens[z] = kappa / np.std(kappa)
    w = np.ones((nz, NPIX))
    return shear, dens, w, w


# ── exact child sums with PixPoint's conventions ──────────────────────────


def _pixel_geom(rows, corr):
    pix = np.asarray(corr.map_inds, dtype=np.int64)[rows]
    ra, dec = pixel2RaDec(pix, corr.nside)
    return cell_frames(ra, dec)


def _coarse_pairs(corr, patch, b):
    """Device rows of the pairs of bin ``b`` of ``patch`` (coarse level)."""
    bins = np.asarray(corr.bins[patch], dtype=np.int64)
    edges = np.concatenate(([0], np.cumsum(bins)))
    lo, hi = int(edges[b]), int(edges[b + 1])
    lut = corr._global_to_row_lut()
    rows = corr._pair_ids_to_rows(corr.pair_inds[patch][:, lo:hi], lut)
    return rows[0].astype(np.int64), rows[1].astype(np.int64)


def _level_of_bin(corr, b):
    tree = corr._treecode
    return tree.coarse_nsides.index(int(corr.level_nside[b]))


def _children(corr, level, cells):
    """Pixel rows of every cell (cells given as device rows)."""
    tree = corr._treecode
    indptr, indices = tree.pixel_children(level)
    lut = corr._global_to_row_lut()
    starts = tree.level_starts(first=corr.n_aperture_cells)
    local = np.asarray(cells) - corr.n_active - starts[level]
    out = []
    for c in local:
        pix = indices[indptr[c] : indptr[c + 1]]
        out.append(pix if lut is None else lut[pix])
    return out


def _exact_pair_sums(corr, level, cells_a, cells_b, g, w, delta, w_d):
    """Per coarse pair: exact sums over the children of (xi+, xi-, w_ss,
    gamma_t (a lens, b source), gamma_t (b lens, a source), w_ds_ab, w_ds_ba)
    with the PixPoint rotation e^{2i phi}."""
    kids_a = _children(corr, level, cells_a)
    kids_b = _children(corr, level, cells_b)
    out = np.zeros((len(cells_a), 7))
    for p, (ka, kb) in enumerate(zip(kids_a, kids_b)):
        ga, gb = _pixel_geom(ka, corr), _pixel_geom(kb, corr)
        xa, xb = ga[:, 0:3], gb[:, 0:3]
        phi_a = np.arctan2(ga[:, 6:9] @ xb.T, ga[:, 3:6] @ xb.T)  # (na, nb)
        phi_b = np.arctan2(xa @ gb[:, 6:9].T, xa @ gb[:, 3:6].T)
        ra, rb = np.exp(2j * phi_a), np.exp(2j * phi_b)
        A = g[ka][:, None] * ra
        B = g[kb][None, :] * rb
        ww = w[ka][:, None] * w[kb][None, :]
        out[p, 0] = np.sum(ww * np.real(A * np.conj(B)))
        out[p, 1] = np.sum(ww * np.real(A * B))
        out[p, 2] = np.sum(ww)
        # gamma_t: lens a / source b and lens b / source a
        wab = w_d[ka][:, None] * w[kb][None, :]
        wba = w_d[kb][None, :] * w[ka][:, None]
        out[p, 3] = np.sum(wab * delta[ka][:, None] * (-np.real(B)))
        out[p, 4] = np.sum(wba * delta[kb][None, :] * (-np.real(A)))
        out[p, 5] = np.sum(wab)
        out[p, 6] = np.sum(wba)
    return out


def _cell_moments(corr, level, g, w, delta, w_d, order):
    """Unnormalised shear, scalar and weight moments of every cell of a
    level (appended-row space of that level), float64."""
    tree = corr._treecode
    indptr, indices = tree.pixel_children(level)
    lut = corr._global_to_row_lut()
    rows = indices if lut is None else lut[indices]
    geom = cell_frames(tree.cell_ra[level], tree.cell_dec[level])
    pix = _pixel_geom(np.arange(corr.n_active), corr)[:, 0:3]
    cell = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    t, dx, dy = moment_factors(geom[cell], pix[rows])
    mom_g, _ = cell_moments_numpy(indptr, rows, t, dx, dy, w, g, order)
    mom_d, _ = cell_moments_numpy(indptr, rows, t, dx, dy, w_d, delta, order)
    return geom, mom_g, mom_d


def _moment_pair_sums(corr, level, cells_a, cells_b, g, w, delta, w_d, order):
    """Per coarse pair from the moments: (xi+, xi-, gamma_t ab, gamma_t ba)."""
    tree = corr._treecode
    starts = tree.level_starts(first=corr.n_aperture_cells)
    la = np.asarray(cells_a) - corr.n_active - starts[level]
    lb = np.asarray(cells_b) - corr.n_active - starts[level]
    geom, mom_g, mom_d = _cell_moments(corr, level, g, w, delta, w_d, max(order, 1))
    nm = {0: 1, 1: 3, 2: 6}[order]
    mg_a, mg_b = mom_g[la, :nm], mom_g[lb, :nm]
    md_a, md_b = mom_d[la, :nm], mom_d[lb, :nm]
    phi_a, phi_b, inv_d = pair_angles(geom[la], geom[lb])
    xip, xim = xipm_pair_terms(mg_a, mg_b, phi_a, phi_b, inv_d, order)
    gt_ab = gammat_pair_terms(md_a, mg_b, phi_a, phi_b, inv_d, order)
    gt_ba = gammat_pair_terms(md_b, mg_a, phi_b, phi_a, inv_d, order)
    return np.stack((xip, xim, gt_ab, gt_ba), axis=1)


class TestPlainCellsUntouched(unittest.TestCase):
    """C1"""

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()

    def test_none_builds_nothing(self):
        corr = _corr(self.mask, None)
        self.assertEqual(corr.moment_order, 0)
        self.assertEqual(corr._pair_moments, 0)
        self.assertIsNone(corr.pair_phi)
        self.assertIsNone(corr.compute_context.moment_base_dev)
        self.assertIsNone(corr.level_table["cell_moments"])

    def test_full_resolution_is_inert(self):
        corr = _corr(self.mask, "quad", k=None)
        self.assertEqual(corr.moment_order, 2)
        self.assertEqual(corr._pair_moments, 0)
        self.assertIsNone(corr.pair_phi)
        shear, dens, w_s, w_d = _random_maps()
        plain = _corr(self.mask, None, k=None)
        a = fields_of(corr.gg(shear, w_s, return_device=False), "M_a", "xi_p", "xi_m")
        b = fields_of(plain.gg(shear, w_s, return_device=False), "M_a", "xi_p", "xi_m")
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            _corr(self.mask, "octupole")


class TestConventions(unittest.TestCase):
    """C2"""

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.corr = _corr(cls.mask, "quad")
        cls.shear, cls.dens, cls.w_s, cls.w_d = _random_maps()

    def test_levels(self):
        self.assertEqual(self.corr.level_nside.tolist(), [128, 128, 64, 32, 32])
        self.assertEqual(self.corr._first_coarse_bin(), 2)
        n_coarse = [int(np.sum(b[2:])) for b in self.corr.bins]
        self.assertEqual([p.shape[1] for p in self.corr.pair_phi], n_coarse)
        self.assertTrue(all(n > 50 for n in n_coarse))

    def test_pair_angles_reproduce_the_pair_finder(self):
        """e^{2i phi} of moments.pair_angles == the pair finder's rotation
        (fine-level pairs, both ends), so the moment kernels and the plain
        kernels rotate into the same frame."""
        corr = self.corr
        a, b = _coarse_pairs(corr, 0, 0)  # a fine bin: pixel rows
        ga, gb = _pixel_geom(a, corr), _pixel_geom(b, corr)
        phi_a, phi_b, inv_d = pair_angles(ga, gb)
        exp = corr.pair_exp2phi[0][:, : a.size]
        np.testing.assert_allclose(np.exp(2j * phi_a), exp[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.exp(2j * phi_b), exp[1], rtol=0, atol=1e-9)
        # separations inside the bin
        d = 1.0 / inv_d
        self.assertTrue(np.all(d > corr.binedges[0]) and np.all(d < corr.binedges[1]))
        # and the stored coarse geometry is what pair_angles gives
        bins = np.asarray(corr.bins[0], dtype=np.int64)
        tree = corr._treecode
        starts = tree.level_starts(first=corr.n_aperture_cells)
        for b_ in range(2, 5):
            ca, cb = _coarse_pairs(corr, 0, b_)
            lev = _level_of_bin(corr, b_)
            geom = cell_frames(tree.cell_ra[lev], tree.cell_dec[lev])
            la = ca - corr.n_active - starts[lev]
            lb = cb - corr.n_active - starts[lev]
            pa, pb, idd = pair_angles(geom[la], geom[lb])
            lo = int(np.sum(bins[2:b_]))
            hi = lo + int(bins[b_])
            np.testing.assert_allclose(corr.pair_phi[0][0, lo:hi], pa, atol=1e-12)
            np.testing.assert_allclose(corr.pair_phi[0][1, lo:hi], pb, atol=1e-12)
            np.testing.assert_allclose(corr.pair_inv_d[0][lo:hi], idd, rtol=1e-12)

    def test_frames_without_trigonometry(self):
        rng = np.random.default_rng(0)
        ra = rng.uniform(0, 2 * np.pi, 1000)
        dec = rng.uniform(-1.4, 1.4, 1000)
        xyz = unit_vectors(ra, dec)
        e, n = frames(xyz)
        np.testing.assert_allclose(
            e, np.stack((-np.sin(ra), np.cos(ra), 0 * ra), 1), atol=1e-14
        )
        np.testing.assert_allclose(
            n,
            np.stack(
                (-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)), 1
            ),
            atol=1e-14,
        )

    def test_per_pair_expansion(self):
        """mono : dipole : quad residuals vs the exact child sums scale as the
        expansion predicts -- the gate that fixes S and the sign of the
        second-order terms (a wrong sign makes the next order *worse*)."""
        corr = self.corr
        g = self.shear[0, 0][corr.row_pix] + 1j * self.shear[0, 1][corr.row_pix]
        w = self.w_s[0][corr.row_pix]
        delta = self.dens[0][corr.row_pix]
        w_d = self.w_d[0][corr.row_pix]
        rng = np.random.default_rng(11)
        self.assertEqual(MOMENT_SIGN, -1.0)
        checked = 0
        for b in range(2, 5):
            ca, cb = _coarse_pairs(corr, 0, b)
            if ca.size == 0:
                continue
            sel = rng.choice(ca.size, size=min(300, ca.size), replace=False)
            ca, cb = ca[sel], cb[sel]
            lev = _level_of_bin(corr, b)
            exact = _exact_pair_sums(corr, lev, ca, cb, g, w, delta, w_d)
            p = np.sqrt(hp.nside2pixarea(int(corr.level_nside[b])))
            theta = np.sqrt(corr.binedges[b] * corr.binedges[b + 1])
            dphi = 0.41 * p / theta
            pred_m = np.array([4 * dphi, 8 * dphi**2, (4 * dphi) ** 3 / 6])
            pred_t = np.array([2 * dphi, 2 * dphi**2, (2 * dphi) ** 3 / 6])
            rms_m, rms_t, rms_p = [], [], []
            for order in (0, 1, 2):
                est = _moment_pair_sums(corr, lev, ca, cb, g, w, delta, w_d, order)
                rms_p.append(np.sqrt(np.mean((est[:, 0] - exact[:, 0]) ** 2)))
                rms_m.append(np.sqrt(np.mean((est[:, 1] - exact[:, 1]) ** 2)))
                gt = np.concatenate((est[:, 2], est[:, 3]))
                gt_ex = np.concatenate((exact[:, 3], exact[:, 4]))
                rms_t.append(np.sqrt(np.mean((gt - gt_ex) ** 2)))
            scale_m = np.sqrt(np.mean(exact[:, 1] ** 2))
            scale_t = np.sqrt(np.mean(np.concatenate((exact[:, 3], exact[:, 4])) ** 2))
            scale_p = np.sqrt(np.mean(exact[:, 0] ** 2))
            rel_m, rel_t = np.array(rms_m) / scale_m, np.array(rms_t) / scale_t
            msg = f"bin {b}: xi- {rel_m}, gamma_t {rel_t}, predicted {pred_m}, {pred_t}"
            # each order improves on the last
            self.assertLess(rel_m[1], 0.5 * rel_m[0], msg)
            self.assertLess(rel_m[2], 0.5 * rel_m[1], msg)
            self.assertLess(rel_t[1], 0.5 * rel_t[0], msg)
            self.assertLess(rel_t[2], 0.5 * rel_t[1], msg)
            # and lands within a factor 3 of the predicted scaling (8 for the
            # quad residual: (c dphi)^3 / 6 is one of several third-order
            # terms, and a white-noise field is the worst case for the rest)
            for k, (got, pred) in enumerate(
                zip(np.concatenate((rel_m, rel_t)), np.concatenate((pred_m, pred_t)))
            ):
                factor = 8.0 if k % 3 == 2 else 3.0
                self.assertLess(got, factor * pred, msg)
                self.assertGreater(got, pred / factor, msg)
            # xi+ needs only the transport.  What is left per pair is second
            # order in p / theta (0.45-0.6 in the coarse bins here) plus the
            # spherical term O(p theta) of the transported J frame (p and
            # theta in radians; 1e-5 at production resolutions, see RESULTS.md)
            self.assertLess(
                rms_p[0] / scale_p, 2e-2 * (p / theta) ** 2 + 4.0 * p * theta, msg
            )
            checked += 1
        self.assertEqual(checked, 3)


class TestKernelsAgainstReference(unittest.TestCase):
    """C3"""

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _random_maps()
        cls.corrs = {cm: _corr(cls.mask, cm) for cm in (None, "dipole", "quad")}

    def _reference(self, corr, order):
        """Numerators and denominators of every (patch, bin) from the per-pair
        sums: coarse bins from the moments, fine bins unchanged."""
        nz = 2
        shape = (corr.n_patches, corr.nbins)
        out = {
            k: np.zeros((nz * (nz + 1) // 2,) + shape) for k in ("xip", "xim", "wss")
        }
        out.update({k: np.zeros((nz * nz,) + shape) for k in ("xit", "wds")})
        rows = corr.row_pix
        for ip in range(corr.n_patches):
            for b in range(corr._first_coarse_bin(), corr.nbins):
                ca, cb = _coarse_pairs(corr, ip, b)
                lev = _level_of_bin(corr, b)
                k = 0
                for i in range(nz):
                    for j in range(i, nz):
                        for ori, (za, zb) in enumerate(((i, j), (j, i))):
                            if ori and i == j:
                                continue
                            ga = self.shear[za, 0][rows] + 1j * self.shear[za, 1][rows]
                            gb = self.shear[zb, 0][rows] + 1j * self.shear[zb, 1][rows]
                            # moments of end a with field za, end b with zb
                            _, mga, _ = _cell_moments(
                                corr,
                                lev,
                                ga,
                                self.w_s[za][rows],
                                0 * ga.real,
                                0 * ga.real + 1,
                                max(order, 1),
                            )
                            geom, mgb, _ = _cell_moments(
                                corr,
                                lev,
                                gb,
                                self.w_s[zb][rows],
                                0 * gb.real,
                                0 * gb.real + 1,
                                max(order, 1),
                            )
                            tree = corr._treecode
                            starts = tree.level_starts(first=corr.n_aperture_cells)
                            la, lb = (
                                ca - corr.n_active - starts[lev],
                                cb - corr.n_active - starts[lev],
                            )
                            nm = {0: 1, 1: 3, 2: 6}[order]
                            pa, pb, idd = pair_angles(geom[la], geom[lb])
                            xip, xim = xipm_pair_terms(
                                mga[la, :nm], mgb[lb, :nm], pa, pb, idd, order
                            )
                            out["xip"][k, ip, b] += xip.sum()
                            out["xim"][k, ip, b] += xim.sum()
                            Wa = np.array(
                                [
                                    self.w_s[za][rows][c].sum()
                                    for c in _children(corr, lev, ca)
                                ]
                            )
                            Wb = np.array(
                                [
                                    self.w_s[zb][rows][c].sum()
                                    for c in _children(corr, lev, cb)
                                ]
                            )
                            out["wss"][k, ip, b] += np.sum(Wa * Wb)
                        k += 1
                for ll in range(nz):
                    for s in range(nz):
                        k = ll * nz + s
                        d = self.dens[ll][rows]
                        wd = self.w_d[ll][rows]
                        gs = self.shear[s, 0][rows] + 1j * self.shear[s, 1][rows]
                        ws = self.w_s[s][rows]
                        geom, mg, md = _cell_moments(
                            corr, lev, gs, ws, d, wd, max(order, 1)
                        )
                        tree = corr._treecode
                        starts = tree.level_starts(first=corr.n_aperture_cells)
                        la, lb = (
                            ca - corr.n_active - starts[lev],
                            cb - corr.n_active - starts[lev],
                        )
                        nm = {0: 1, 1: 3, 2: 6}[order]
                        pa, pb, idd = pair_angles(geom[la], geom[lb])
                        gt_ab = gammat_pair_terms(
                            md[la, :nm], mg[lb, :nm], pa, pb, idd, order
                        )
                        gt_ba = gammat_pair_terms(
                            md[lb, :nm], mg[la, :nm], pb, pa, idd, order
                        )
                        out["xit"][k, ip, b] += gt_ab.sum() + gt_ba.sum()
                        Wd_a = np.array([wd[c].sum() for c in _children(corr, lev, ca)])
                        Wd_b = np.array([wd[c].sum() for c in _children(corr, lev, cb)])
                        Ws_a = np.array([ws[c].sum() for c in _children(corr, lev, ca)])
                        Ws_b = np.array([ws[c].sum() for c in _children(corr, lev, cb)])
                        out["wds"][k, ip, b] += np.sum(Wd_a * Ws_b) + np.sum(
                            Wd_b * Ws_a
                        )
        return out

    def assert_coarse(self, got, want_num, want_den, b_c, name, tol=1e-10):
        est = np.where(
            want_den != 0, want_num / np.where(want_den != 0, want_den, 1), 0
        )
        scale = np.max(np.abs(est[..., b_c:]))
        np.testing.assert_allclose(
            got[..., b_c:], est[..., b_c:], rtol=0, atol=tol * scale, err_msg=name
        )

    def test_gg_kg_three_by_two_pt(self):
        plain = self.corrs[None]
        xi_g_plain = plain.kk(
            self.dens, self.w_d, return_device=False, apertures=False
        ).xi_g
        gg_plain = fields_of(
            plain.gg(self.shear, self.w_s, return_device=False, apertures=False),
            "xi_p",
            "xi_m",
        )
        xi_g_fused_plain = plain.three_by_two_pt(
            shear_maps=self.shear,
            k_maps=self.dens,
            weights=(self.w_s, self.w_d),
            return_device=False,
        ).xi_g
        for cm, order in (("dipole", 1), ("quad", 2)):
            corr = self.corrs[cm]
            b_c = corr._first_coarse_bin()
            ref = self._reference(corr, order)
            xip, xim = fields_of(
                corr.gg(self.shear, self.w_s, return_device=False, apertures=False),
                "xi_p",
                "xi_m",
            )
            self.assert_coarse(xip, ref["xip"], ref["wss"], b_c, f"{cm} gg xi+")
            self.assert_coarse(xim, ref["xim"], ref["wss"], b_c, f"{cm} gg xi-")
            # the fine bins are untouched by the moments
            for a, b in zip((xip, xim), gg_plain):
                np.testing.assert_array_equal(a[..., :b_c], b[..., :b_c])
            xit = corr.kg(
                self.dens, self.shear, self.w_d, self.w_s, return_device=False
            ).xi_t
            self.assert_coarse(xit, ref["xit"], ref["wds"], b_c, f"{cm} kg xi_t")
            res = corr.three_by_two_pt(
                shear_maps=self.shear,
                k_maps=self.dens,
                weights=(self.w_s, self.w_d),
                return_device=False,
            )
            np.testing.assert_allclose(
                res.xi_p, xip, rtol=0, atol=1e-12 * np.max(np.abs(xip))
            )
            np.testing.assert_allclose(
                res.xi_m, xim, rtol=0, atol=1e-12 * np.max(np.abs(xim))
            )
            np.testing.assert_allclose(
                res.xi_t, xit, rtol=0, atol=1e-12 * np.max(np.abs(xit))
            )
            # xi_g has no phase: bitwise the plain cells, in and out of 3x2pt
            xi_g = corr.kk(
                self.dens, self.w_d, return_device=False, apertures=False
            ).xi_g
            self.assertTrue(np.array_equal(xi_g, xi_g_plain))
            self.assertTrue(np.array_equal(res.xi_g, xi_g_fused_plain))
            # and the moments change the coarse bins of xi-
            self.assertFalse(
                np.allclose(xim[..., b_c:], gg_plain[1][..., b_c:], rtol=1e-3)
            )


class TestSmoothField(unittest.TestCase):
    """C4"""

    def test_quad_is_close_to_the_exact_child_sums(self):
        """k = 3: levels [128, 128, 128, 64, 32], p / theta = 0.18 and 0.23 in
        the coarse bins.  Errors against the exact child sums (same bin
        assignment) in units of the largest exact value of the patch."""
        mask = _mask()
        shear, dens, w_s, w_d = _smooth_maps()
        results = {}
        exact_num = den = None
        b_c = None
        for cm in (None, "dipole", "quad"):
            corr = _corr(mask, cm, k=3.0)
            self.assertEqual(corr.level_nside.tolist(), [128, 128, 128, 64, 32])
            rows = corr.row_pix
            g = shear[0, 0][rows] + 1j * shear[0, 1][rows]
            w = w_s[0][rows]
            res = corr.gg(shear, w_s, return_device=False, apertures=False)
            results[cm] = (res.xi_p[0], res.xi_m[0])
            if exact_num is None:
                b_c = corr._first_coarse_bin()
                exact_num = np.zeros((2, corr.n_patches, corr.nbins))
                den = np.zeros((corr.n_patches, corr.nbins))
                for ip in range(corr.n_patches):
                    for b in range(b_c, corr.nbins):
                        ca, cb = _coarse_pairs(corr, ip, b)
                        lev = _level_of_bin(corr, b)
                        ex = _exact_pair_sums(
                            corr, lev, ca, cb, g, w, dens[0][rows], w_d[0][rows]
                        )
                        exact_num[0, ip, b] = ex[:, 0].sum()
                        exact_num[1, ip, b] = ex[:, 1].sum()
                        den[ip, b] = ex[:, 2].sum()
        assert exact_num is not None and den is not None and b_c is not None
        exact_p = exact_num[0, :, b_c:] / den[:, b_c:]
        exact_m = exact_num[1, :, b_c:] / den[:, b_c:]
        scale_p = np.max(np.abs(exact_p), axis=1, keepdims=True)
        scale_m = np.max(np.abs(exact_m), axis=1, keepdims=True)
        err = {}
        for cm, (xip, xim) in results.items():
            err[cm] = (
                float(np.max(np.abs(xip[:, b_c:] - exact_p) / scale_p)),
                float(np.max(np.abs(xim[:, b_c:] - exact_m) / scale_m)),
            )
        # xi+ with the transported mean: exact up to the spherical O(p theta)
        # term of the transported J frame (p = 110', theta <= 600' here, both
        # in radians: 4e-3; 1e-5 at production resolutions, RESULTS.md)
        p_max = np.sqrt(hp.nside2pixarea(int(corr.level_nside.min())))
        self.assertLess(err["quad"][0], 4.0 * p_max * float(corr.binedges[-1]), err)
        # xi-: quad within 1.5 % of the largest exact value and 3.5x closer
        # than the plain cells (the probe: 5-10x at k = 2); the dipole alone
        # flips the sign of the mean error and need not be closer
        self.assertLess(err["quad"][1], 1.5e-2, err)
        self.assertLess(err["quad"][1], 0.3 * err[None][1], err)


class TestBackendsAndEstimators(unittest.TestCase):
    """C5"""

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _random_maps()

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
        return (res.xi_p, res.xi_m, res.xi_g, res.xi_t, gg.xi_p, gg.xi_m, kg.xi_t)

    def test_emulated_gpu_equals_cpu(self):
        for pack in (False, True):
            with self.subTest(pack_pairs=pack):
                corr = _corr(self.mask, "quad", pack_pairs=pack)
                cpu = self._measure(corr)
                with emulated_gpu(corr) as calls:
                    gpu = self._measure(corr)
                self.assertGreater(calls["kernel_moment_level"], 0)
                self.assertGreater(calls["xipm_tomo_moments_kernel"], 0)
                self.assertGreater(calls["kernel_density_shear_tomo_moments"], 0)
                self.assertGreater(calls["kernel_3x2pt_tomo_moments"], 0)
                for a, b in zip(cpu, gpu):
                    np.testing.assert_allclose(
                        b, a, rtol=0, atol=1e-12 * np.max(np.abs(a))
                    )

    def test_packed_equals_unpacked_to_the_quantisation(self):
        exact = self._measure(_corr(self.mask, "quad"))
        packed = self._measure(_corr(self.mask, "quad", pack_pairs=True))
        for a, b in zip(exact, packed):
            np.testing.assert_allclose(b, a, rtol=0, atol=2e-4 * np.max(np.abs(a)))

    def test_global2pcf_equals_one_patch(self):
        """Global2PCF with moments == an Integrated3PCF with one patch holding
        the whole footprint (an independent pair search), <= 1e-12."""
        theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
        centre = (np.radians(60.0), np.radians(30.0))
        vec = hp.ang2vec(*centre)
        disc = hp.query_disc(NSIDE, vec, np.radians(9.0))
        mask = np.zeros(NPIX, dtype=bool)
        mask[disc] = True
        rng = np.random.default_rng(2)
        mask[rng.choice(disc, disc.size // 40, replace=False)] = False
        kw = dict(
            nbins=4,
            theta_min=90.0,
            theta_max=700.0,
            device="cpu",
            map_precision="float64",
            rotation_precision="float64",
            resolution_factor=2.0,
            cell_moments="quad",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Global2PCF(NSIDE, mask=mask, tile_nside=8, **kw)
        g.preprocess()
        one = Integrated3PCF(
            NSIDE,
            np.array([centre[1]]),
            np.array([centre[0]]),
            mask=mask,
            patch_size=9.0 * 60.0 + 30.0,
            theta_Q=100.0,
            **kw,
        )
        one.preprocess()
        self.assertGreater(g._engine._pair_moments, 0)
        self.assertGreater(one._pair_moments, 0)
        res = g.gg(self.shear, self.w_s, flip_g1=True, return_device=False)
        ref = fields_of(
            one.gg(
                self.shear, self.w_s, flip_g1=True, return_device=False, apertures=False
            ),
            "xi_p",
            "xi_m",
        )
        for got, want in zip((res.xi_p, res.xi_m), ref):
            want = want[:, 0, :]
            np.testing.assert_allclose(
                got, want, rtol=0, atol=1e-12 * np.max(np.abs(want))
            )
        res_t = g.kg(self.dens, self.shear, self.w_d, self.w_s, return_device=False)
        want_t = one.kg(
            self.dens, self.shear, self.w_d, self.w_s, return_device=False
        ).xi_t[:, 0, :]
        np.testing.assert_allclose(
            res_t.xi, want_t, rtol=0, atol=1e-12 * np.max(np.abs(want_t))
        )


class TestFilesAndHostPacking(unittest.TestCase):
    """C5: pair files (format 5) and the packed host payload with moments."""

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _random_maps()

    def _measure(self, corr):
        gg = corr.gg(self.shear, self.w_s, return_device=False, apertures=False)
        kg = corr.kg(self.dens, self.shear, self.w_d, self.w_s, return_device=False)
        return gg.xi_p, gg.xi_m, kg.xi_t

    def test_round_trip(self):
        import os
        import tempfile

        import h5py

        corr = _corr(self.mask, "quad")
        ref = self._measure(corr)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.h5")
            corr.save_pairs(path)
            with h5py.File(path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 5)
                self.assertEqual(fp.attrs["cell_moments"], "quad")
                self.assertIn("pair_phi", fp)
            for cm in ("quad", "dipole"):
                other = Integrated3PCF(
                    NSIDE,
                    PHI_C,
                    THETA_C,
                    mask=self.mask,
                    device="cpu",
                    map_precision="float64",
                    rotation_precision="float64",
                    resolution_factor=2.0,
                    cell_moments=cm,
                    **BINS,
                    **PATCH,
                )
                other.load_pairs(path)
                got = self._measure(other)
                if cm == "quad":
                    for a, b in zip(got, ref):
                        self.assertTrue(np.array_equal(a, b))
                else:
                    self.assertFalse(np.allclose(got[1], ref[1]))
            # a sliced load keeps the coarse geometry aligned
            part = Integrated3PCF(
                NSIDE,
                PHI_C[1:],
                THETA_C[1:],
                mask=self.mask,
                device="cpu",
                map_precision="float64",
                rotation_precision="float64",
                resolution_factor=2.0,
                cell_moments="quad",
                **BINS,
                **PATCH,
            )
            part.load_pairs(path, 1, 2)
            got = self._measure(part)
            for a, b in zip(got, ref):
                self.assertTrue(np.array_equal(a, b[:, 1:]))
            # a plain-cell file cannot serve a moment estimator
            plain = _corr(self.mask, None)
            plain_path = os.path.join(tmp, "plain.h5")
            plain.save_pairs(plain_path)
            with h5py.File(plain_path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 3)
            with self.assertRaises(ValueError):
                other.load_pairs(plain_path)

    def test_host_packing(self):
        import os
        import tempfile

        import h5py

        device = _corr(self.mask, "quad", pack_pairs=True)
        host = _corr(self.mask, "quad", pack_pairs=True, pack_host_pairs=True)
        self.assertIsNone(host.pair_phi)
        self.assertIsNotNone(host.packed_inv_d)
        ref = self._measure(device)
        # packing at pair-finding time == packing in prepare(), bitwise
        for a, b in zip(self._measure(host), ref):
            self.assertTrue(np.array_equal(a, b))
        ctx_d, ctx_h = device.compute_context, host.compute_context
        for name in ("moment_phi_dev", "moment_inv_d_dev", "moment_base_dev"):
            np.testing.assert_array_equal(getattr(ctx_d, name), getattr(ctx_h, name))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "packed.h5")
            host.save_pairs(path)
            with h5py.File(path, "r") as fp:
                self.assertEqual(int(fp.attrs["format_version"]), 5)
                self.assertIn("packed_inv_d", fp)
            other = _corr(self.mask, "quad", pack_pairs=True)
            other.load_pairs(path)
            for a, b in zip(self._measure(other), ref):
                self.assertTrue(np.array_equal(a, b))

    def test_global2pcf_files(self):
        import os
        import tempfile

        import h5py

        theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
        mask = (np.degrees(phi) < 40.0) & (np.abs(90 - np.degrees(theta)) < 20)
        kw = dict(
            nbins=4,
            theta_min=90.0,
            theta_max=700.0,
            device="cpu",
            map_precision="float64",
            rotation_precision="float64",
            resolution_factor=2.0,
            cell_moments="quad",
            tile_nside=8,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Global2PCF(NSIDE, mask=mask, **kw)
            g.preprocess()
            ref = g.gg(self.shear, self.w_s, return_device=False)
            gp = Global2PCF(
                NSIDE, mask=mask, pack_pairs=True, pack_host_pairs=True, **kw
            )
            gp.preprocess()
            gd = Global2PCF(NSIDE, mask=mask, pack_pairs=True, **kw)
            gd.preprocess()
        packed_ref = gd.gg(self.shear, self.w_s, return_device=False)
        for name in ref.names:
            self.assertTrue(
                np.array_equal(
                    gp.gg(self.shear, self.w_s, return_device=False).num[name],
                    packed_ref.num[name],
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            for est, want, version in ((g, ref, 3), (gp, packed_ref, 3)):
                path = os.path.join(tmp, "g.h5")
                est.save_pairs(path)
                with h5py.File(path, "r") as fp:
                    self.assertEqual(int(fp.attrs["format_version"]), version)
                    self.assertEqual(fp.attrs["cell_moments"], "quad")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    other = Global2PCF(NSIDE, mask=mask, pack_pairs=est is gp, **kw)
                other.load_pairs(path)
                got = other.gg(self.shear, self.w_s, return_device=False)
                for name in want.names:
                    self.assertTrue(np.array_equal(got.num[name], want.num[name]))


class TestApertureMoments(unittest.TestCase):
    """C6: the aperture statistics on a coarse aperture level.

    Reference: the exact sums over the children of every disc cell with the
    coarse disc membership and the coarse centre pixel (the window the
    estimator defines), so only the within-cell expansion is measured.
    """

    AP = 64  # base 128 -> aperture level 64: p / theta_Q = 0.18 at theta_Q = 300'
    BASE = dict(
        device="cpu",
        map_precision="float64",
        rotation_precision="float64",
        nbins=3,
        theta_min=60.0,
        theta_max=300.0,
        patch_size=400.0,
        theta_Q=300.0,
    )

    @classmethod
    def setUpClass(cls):
        cls.mask = _mask()
        cls.shear, cls.dens, cls.w_s, cls.w_d = _smooth_maps(seed=3)
        cls.corrs = {
            cm: Integrated3PCF(
                NSIDE,
                PHI_C,
                THETA_C,
                mask=cls.mask,
                aperture_nside=cls.AP,
                cell_moments=cm,
                **cls.BASE,
            )
            for cm in (None, "dipole", "quad")
        }
        for c in cls.corrs.values():
            c.preprocess()
        cls.exact = cls._exact_apertures(cls.corrs[None])

    @classmethod
    def _exact_apertures(cls, corr):
        """(M_a, M_g with U) per (z, patch) from the exact child sums of the
        coarse disc cells about the coarse centre pixel."""
        from pixpoint import Q_crittenden

        rows = corr.row_pix
        cell_pix, indptr, child_rows = corr._aperture_cells()
        pix = _pixel_geom(np.arange(corr.n_active), corr)
        pe, pn, px = pix[:, 3:6], pix[:, 6:9], pix[:, 0:3]
        nz = cls.shear.shape[0]
        ma = np.zeros((nz, corr.n_patches))
        mg = np.zeros((nz, corr.n_patches))
        tq = corr.theta_Q
        for ip in range(corr.n_patches):
            q_inds = np.asarray(corr.Q_inds[ip], dtype=np.int64) - NPIX
            pc = hp.ang2pix(cls.AP, THETA_C[ip], PHI_C[ip])
            xc = unit_vectors(*pixel2RaDec([pc], cls.AP))[0]
            area = corr.Q_patch_area[ip]
            for z in range(nz):
                g = cls.shear[z, 0][rows] + 1j * cls.shear[z, 1][rows]
                w, d, wd = cls.w_s[z][rows], cls.dens[z][rows], cls.w_d[z][rows]
                na = nw = ng = nwd = 0.0
                for c in q_inds:
                    kids = child_rows[indptr[c] : indptr[c + 1]]
                    psi = np.arctan2(pn[kids] @ xc, pe[kids] @ xc)
                    th = np.arccos(np.clip(px[kids] @ xc, -1, 1))
                    na += np.sum(
                        w[kids]
                        * Q_crittenden(th, tq)
                        * (-np.real(g[kids] * np.exp(2j * psi)))
                    )
                    nw += np.sum(w[kids])
                    ng += np.sum(wd[kids] * U_crittenden(th, tq) * d[kids])
                    nwd += np.sum(wd[kids])
                ma[z, ip] = area * na / nw
                mg[z, ip] = area * ng / nwd
        return ma, mg

    def test_filter_derivatives(self):
        from pixpoint import Q_crittenden, Q_schneider, U_crittenden, U_schneider
        from pixpoint.correlation_helpers import filter_derivatives

        th = np.linspace(0.002, 0.25, 500)
        for fn in (Q_crittenden, U_crittenden, Q_schneider, U_schneider):
            d1, d2 = filter_derivatives(fn, th, 90.0)

            def custom(t, tq=90.0, fn=fn):
                return fn(t, tq)

            f1, f2 = filter_derivatives(custom, th, 90.0)  # finite differences
            # away from Schneider's edge (a kink at theta_Q)
            keep = np.abs(th - np.radians(1.5)) > 0.003
            scale1, scale2 = np.max(np.abs(d1)), np.max(np.abs(d2))
            np.testing.assert_allclose(f1[keep], d1[keep], rtol=0, atol=5e-6 * scale1)
            np.testing.assert_allclose(f2[keep], d2[keep], rtol=0, atol=5e-6 * scale2)

    def test_full_resolution_apertures_are_untouched(self):
        plain = _corr(
            self.mask,
            None,
            k=None,
            **{
                k: v
                for k, v in self.BASE.items()
                if k not in ("device", "map_precision", "rotation_precision")
            },
        )
        quad = _corr(
            self.mask,
            "quad",
            k=None,
            **{
                k: v
                for k, v in self.BASE.items()
                if k not in ("device", "map_precision", "rotation_precision")
            },
        )
        self.assertEqual(quad._aperture_moments, 0)
        a = plain.m_a(self.shear, self.w_s, return_device=False)
        b = quad.m_a(self.shear, self.w_s, return_device=False)
        self.assertTrue(np.array_equal(a, b))

    def test_each_order_closer_to_the_exact_child_sums(self):
        ma_ex, mg_ex = self.exact
        err_a, err_g = {}, {}
        for cm, corr in self.corrs.items():
            ma = corr.m_a(self.shear, self.w_s, return_device=False)
            mg = corr.m_g(
                self.dens, self.w_d, aperture_filter=U_crittenden, return_device=False
            )
            err_a[cm] = np.max(np.abs(ma - ma_ex) / np.abs(ma_ex))
            err_g[cm] = np.max(np.abs(mg - mg_ex) / np.abs(mg_ex))
            # the patch mean has a constant filter: identical for every order
            pm = corr.m_g(self.dens, self.w_d, return_device=False)
            self.assertTrue(
                np.array_equal(
                    pm, self.corrs[None].m_g(self.dens, self.w_d, return_device=False)
                )
            )
        msg = f"M_a {err_a}, M_g {err_g}"
        self.assertGreater(err_a[None], 1e-2, msg)  # the plain cells are off by %
        self.assertLess(err_a["quad"], 2e-3, msg)
        self.assertLess(err_a["quad"], 0.1 * err_a[None], msg)
        self.assertLess(err_a["dipole"], err_a[None], msg)
        self.assertLess(err_g["quad"], 1e-3, msg)
        self.assertLess(err_g["quad"], 0.1 * err_g[None], msg)

    def test_three_by_two_pt_matches_m_a_m_g_and_emulated_gpu(self):
        for cm in ("dipole", "quad"):
            corr = self.corrs[cm]
            ma = corr.m_a(self.shear, self.w_s, return_device=False)
            mg = corr.m_g(
                self.dens, self.w_d, aperture_filter=U_crittenden, return_device=False
            )
            res = corr.three_by_two_pt(
                shear_maps=self.shear,
                k_maps=self.dens,
                weights=(self.w_s, self.w_d),
                k_aperture_filter=U_crittenden,
                return_device=False,
            )
            np.testing.assert_allclose(
                res.M_a, ma, rtol=0, atol=1e-12 * np.max(np.abs(ma))
            )
            np.testing.assert_allclose(
                res.M_g, mg, rtol=0, atol=1e-12 * np.max(np.abs(mg))
            )
            with emulated_gpu(corr) as calls:
                corr.compute_context.Q_inds_dev = None
                ma_g = corr.m_a(self.shear, self.w_s, return_device=False)
                mg_g = corr.m_g(
                    self.dens,
                    self.w_d,
                    aperture_filter=U_crittenden,
                    return_device=False,
                )
                res_g = corr.three_by_two_pt(
                    shear_maps=self.shear,
                    k_maps=self.dens,
                    weights=(self.w_s, self.w_d),
                    k_aperture_filter=U_crittenden,
                    return_device=False,
                )
            corr.compute_context.Q_inds_dev = None
            self.assertGreater(calls["aperture_tomo_moments_kernel"], 0)
            self.assertGreater(calls["kernel_3x2pt_aperture_moments"], 0)
            for got, want in ((ma_g, ma), (mg_g, mg), (res_g.M_a, ma), (res_g.M_g, mg)):
                np.testing.assert_allclose(
                    got, want, rtol=0, atol=1e-12 * np.max(np.abs(want))
                )


@pytest.mark.gpu
class TestRealGPU(unittest.TestCase):
    """The real CUDA kernels against the CPU backend (float64 maps, <= 1e-12),
    unpacked and packed, pairs and apertures."""

    def _pair(self, pack):
        mask = _mask()
        shear, dens, w_s, w_d = _random_maps()
        cpu = _corr(mask, "quad", pack_pairs=pack)
        args = dict(
            mask=mask,
            device="gpu",
            map_precision="float64",
            rotation_precision="float64",
            resolution_factor=2.0,
            cell_moments="quad",
            pack_pairs=pack,
        )
        args.update(BINS)
        args.update(PATCH)
        gpu = Integrated3PCF(NSIDE, PHI_C, THETA_C, **args)
        gpu.preprocess()
        return cpu, gpu, (shear, dens, w_s, w_d)

    def test_pairs(self):
        for pack in (False, True):
            with self.subTest(pack_pairs=pack):
                cpu, gpu, (shear, dens, w_s, w_d) = self._pair(pack)
                for est in (cpu, gpu):
                    est_out = []
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
                    est_out = [r.xi_p, r.xi_m, r.xi_g, r.xi_t, g.xi_p, g.xi_m, t.xi_t]
                    if est is cpu:
                        ref = est_out
                    else:
                        for a, b in zip(ref, est_out):
                            np.testing.assert_allclose(
                                np.asarray(b),
                                np.asarray(a),
                                rtol=0,
                                atol=1e-12 * np.max(np.abs(a)),
                            )

    def test_apertures(self):
        mask = _mask()
        shear, dens, w_s, w_d = _smooth_maps(seed=3)
        base = dict(TestApertureMoments.BASE)
        outs = []
        for device in ("cpu", "gpu"):
            base["device"] = device
            corr = Integrated3PCF(
                NSIDE,
                PHI_C,
                THETA_C,
                mask=mask,
                aperture_nside=TestApertureMoments.AP,
                cell_moments="quad",
                **base,
            )
            corr.preprocess()
            r = corr.three_by_two_pt(
                shear_maps=shear,
                k_maps=dens,
                weights=(w_s, w_d),
                k_aperture_filter=U_crittenden,
                return_device=False,
            )
            outs.append(
                (
                    corr.m_a(shear, w_s, return_device=False),
                    corr.m_g(
                        dens, w_d, aperture_filter=U_crittenden, return_device=False
                    ),
                    r.M_a,
                    r.M_g,
                )
            )
        for a, b in zip(*outs):
            np.testing.assert_allclose(
                np.asarray(b), np.asarray(a), rtol=0, atol=1e-12 * np.max(np.abs(a))
            )


if __name__ == "__main__":
    unittest.main()
