"""``OnePoint``: one-point statistics of aperture fields (8.0.0 stage D).

Gates (never loosened):

1. the per-aperture values are *bitwise* ``Integrated3PCF.m_a`` / ``m_g`` on
   the same centres (same kernels), and ``M_x`` equals an independent numpy
   cross projection to 1e-12;
2. histogram and moments equal ``numpy`` on the per-aperture values to
   1e-12, under/overflow accounted, non-finite apertures left out;
3. coverage: a centre whose disc is masked below ``min_coverage`` is absent,
   ``min_coverage=0`` keeps every footprint centre with an observed pixel;
4. the HEALPix-pixel top hat equals the masked ``ud_grade`` of the map;
5. several devices add to one device (histogram bitwise, sums to 1e-13), the
   emulated GPU path equals the CPU backend, and ``aperture_nside`` runs
   without pairs and equals the patch estimator's coarse apertures.
"""

import contextlib
import unittest

import healpy as hp
import numpy as np

from CosmoFuse import (
    Integrated3PCF,
    OnePoint,
    OnePointResult,
    Q_crittenden,
    U_crittenden,
)
from CosmoFuse.onepoint import default_centre_nside, footprint_centres
from CosmoFuse.results import onepoint_cross_index

from .test_gpu_kernel_emulation import emulated_gpu

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
SCALES = (60.0, 120.0)
EDGES = np.linspace(-0.05, 0.05, 11)


def footprint_mask():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta > 1.2) & (theta < 1.9) & (phi > 0.5) & (phi < 2.5)
    # a hole, so that coverage has something to reject
    vec = hp.ang2vec(1.55, 1.5)
    mask[hp.query_disc(NSIDE, vec, np.radians(2.0))] = False
    return mask


def maps(nz=2, seed=1):
    rng = np.random.default_rng(seed)
    shear = rng.normal(size=(nz, 2, NPIX)) * 0.05
    kappa = rng.normal(size=(nz, NPIX)) * 0.02
    w = 0.5 + rng.random((nz, NPIX))
    return shear, kappa, w


def make(mask, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("rotation_precision", "float64")
    kw.setdefault("scales", SCALES)
    return OnePoint(NSIDE, mask, **kw)


def patch_estimator(op, statistic, s, mask, **kw):
    """An ``Integrated3PCF`` with the accepted centres of scale ``s`` as patches."""
    c = op.centres(statistic, s)
    theta = float(op.scales[s])
    kw.setdefault("device", "cpu")
    kw.setdefault("rotation_precision", "float64")
    corr = Integrated3PCF(
        NSIDE, c["phi"], c["theta"], patch_size=theta, theta_Q=theta, mask=mask, **kw
    )
    corr.radius_filter = op.radius_factor * theta
    return corr


def cross_reference(corr, mask, shear, w):
    """A_c Σ w γ_x Q / Σ w with γ_x = γ1 sin 2φ − γ2 cos 2φ, from scratch."""
    out = np.zeros((shear.shape[0], corr.n_patches))
    for p in range(corr.n_patches):
        vec = hp.ang2vec(corr.theta_center[p], corr.phi_center[p])
        pc = hp.ang2pix(NSIDE, corr.theta_center[p], corr.phi_center[p])
        disc = hp.query_disc(NSIDE, vec, np.radians(corr.radius_filter / 60.0))
        disc = disc[mask[disc]]
        disc = disc[disc != pc]
        th, ph = hp.pix2ang(NSIDE, disc)
        thc, phc = hp.pix2ang(NSIDE, pc)
        dec, ra = np.pi / 2 - th, ph
        decc, rac = np.pi / 2 - thc, phc
        cosv = np.cos(ra - rac) * np.cos(decc) * np.cos(dec) + np.sin(decc) * np.sin(
            dec
        )
        sinv = np.sqrt(1 - cosv**2)
        cphi = np.sin(ra - rac) * np.cos(decc) / sinv
        sphi = (
            np.cos(dec) * np.sin(decc) - np.sin(dec) * np.cos(decc) * np.cos(ra - rac)
        ) / sinv
        c2, s2 = cphi**2 - sphi**2, 2 * sphi * cphi
        Q = Q_crittenden(np.arccos(cosv), corr.theta_Q)
        A = disc.size * hp.nside2pixarea(NSIDE)
        gx = shear[:, 0, disc] * s2 - shear[:, 1, disc] * c2
        out[:, p] = A * np.sum(w[:, disc] * gx * Q, axis=1) / np.sum(w[:, disc], axis=1)
    return out


def numpy_reduction(values, edges):
    """Histogram (half-open bins), power sums and cross sums of ``values
    (nz, n)`` over the apertures finite in every bin."""
    v = values[:, np.all(np.isfinite(values), axis=0)]
    nz = v.shape[0]
    power = np.stack([np.sum(v**k, axis=1) for k in range(1, 6)], axis=1)
    idx = onepoint_cross_index(nz)
    cross = {
        "ij": np.array([np.sum(v[i] * v[j]) for i, j in idx["ij"]]),
        "iij": np.array([np.sum(v[i] ** 2 * v[j]) for i, j in idx["iij"]]),
        "ijk": np.array([np.sum(v[i] * v[j] * v[k]) for i, j, k in idx["ijk"]]),
    }
    hist = np.zeros((nz, edges.size - 1))
    under = np.zeros(nz)
    over = np.zeros(nz)
    for z in range(nz):
        under[z] = np.count_nonzero(v[z] < edges[0])
        over[z] = np.count_nonzero(v[z] >= edges[-1])
        inside = v[z][(v[z] >= edges[0]) & (v[z] < edges[-1])]
        hist[z] = np.bincount(
            np.searchsorted(edges, inside, side="right") - 1, minlength=edges.size - 1
        )
    return v.shape[1], power, cross, hist, under, over


class TestValuesAreTheApertureKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = footprint_mask()
        cls.shear, cls.kappa, cls.w = maps()
        cls.op = make(cls.mask, edges=EDGES)
        cls.res_g = cls.op.g(cls.shear, cls.w, flip_g1=True, values=True)
        cls.res_k = cls.op.k(cls.kappa, cls.w, values=True)

    def test_m_ap_is_m_a_bitwise(self):
        for s in range(len(SCALES)):
            corr = patch_estimator(self.op, "g", s, self.mask)
            ref = corr.m_a(self.shear, self.w, flip_g1=True, return_device=False)
            self.assertTrue(np.array_equal(ref, self.res_g.values["M_ap"][s]))

    def test_m_x_is_the_cross_projection(self):
        for s in range(len(SCALES)):
            corr = patch_estimator(self.op, "g", s, self.mask)
            shear = self.shear.copy()
            shear[:, 0] *= -1  # flip_g1
            ref = cross_reference(corr, self.mask, shear, self.w)
            got = self.res_g.values["M_x"][s]
            self.assertLess(np.max(np.abs(ref - got)), 1e-12 * np.max(np.abs(ref)))

    def test_cross_false_measures_only_m_ap(self):
        res = self.op.g(self.shear, self.w, flip_g1=True, cross=False)
        self.assertEqual(res.names, ("M_ap",))
        np.testing.assert_array_equal(
            res.power_sums["M_ap"], self.res_g.power_sums["M_ap"]
        )

    def test_k_tophat_is_m_g_bitwise(self):
        for s in range(len(SCALES)):
            corr = patch_estimator(self.op, "k", s, self.mask)
            ref = corr.m_g(self.kappa, self.w, return_device=False)
            self.assertTrue(np.array_equal(ref, self.res_k.values["k"][s]))

    def test_k_filter_is_m_g_with_the_filter_bitwise(self):
        op = make(self.mask, k_aperture_filter=U_crittenden)
        res = op.k(self.kappa, self.w, values=True)
        for s in range(len(SCALES)):
            corr = patch_estimator(op, "k", s, self.mask)
            ref = corr.m_g(
                self.kappa, self.w, aperture_filter=U_crittenden, return_device=False
            )
            self.assertTrue(np.array_equal(ref, res.values["k"][s]))
        # a filter shares the shear discs: same centres as g
        np.testing.assert_array_equal(
            op.centres("k", 0)["pix"], op.centres("g", 0)["pix"]
        )

    def test_centres_are_footprint_pixels_with_coverage(self):
        c = self.op.centres("g", 0)
        self.assertEqual(c["centre_nside"], NSIDE)
        self.assertTrue(np.all(c["coverage"] >= 0.8))
        self.assertTrue(np.all(np.diff(c["pix"]) > 0))
        self.assertEqual(c["pix"].size, self.res_g.n_apertures[0])
        np.testing.assert_array_equal(self.op.n_apertures("g"), self.res_g.n_apertures)
        prov = self.op.provenance
        self.assertEqual(prov["aperture_filter"], "Q_crittenden")
        self.assertEqual(prov["n_apertures"]["shear"], list(self.res_g.n_apertures))


class TestReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = footprint_mask()
        cls.shear, cls.kappa, cls.w = maps(nz=3)
        cls.op = make(cls.mask, edges=EDGES)
        cls.res = cls.op.g(cls.shear, cls.w, values=True)

    def test_histogram_and_moments_match_numpy(self):
        for name in self.res.names:
            for s in range(len(SCALES)):
                n, power, cross, hist, under, over = numpy_reduction(
                    self.res.values[name][s], EDGES
                )
                self.assertEqual(self.res.count[name][s], n)
                self.assertEqual(n, self.res.n_apertures[s])
                np.testing.assert_allclose(
                    self.res.power_sums[name][:, s], power, rtol=1e-12, atol=0
                )
                for k in ("ij", "iij", "ijk"):
                    np.testing.assert_allclose(
                        self.res.cross_sums[name][k][:, s], cross[k], rtol=1e-12
                    )
                np.testing.assert_array_equal(self.res.hist[name][:, s], hist)
                np.testing.assert_array_equal(self.res.underflow[name][:, s], under)
                np.testing.assert_array_equal(self.res.overflow[name][:, s], over)
                total = hist.sum(axis=1) + under + over
                np.testing.assert_array_equal(total, n)

    def test_estimates(self):
        m = self.res.moments("M_ap")
        v = self.res.values["M_ap"][0]
        np.testing.assert_allclose(m[:, 0, 0], v.mean(axis=1), rtol=1e-12)
        c = self.res.central_moments("M_ap")
        np.testing.assert_allclose(c[:, 0, 0], v.var(axis=1), rtol=1e-9)
        pdf = self.res.pdf("M_ap")
        width = np.diff(EDGES)
        inside = np.sum(pdf[:, 0] * width, axis=1)
        frac = (
            1
            - (self.res.underflow["M_ap"][:, 0] + self.res.overflow["M_ap"][:, 0])
            / v.shape[1]
        )
        np.testing.assert_allclose(inside, frac, rtol=1e-12)
        cm = self.res.cross_moments("M_ap")
        self.assertEqual(cm["ij"].shape, (3, 2))
        self.assertEqual(cm["iij"].shape, (6, 2))
        self.assertEqual(cm["ijk"].shape, (1, 2))
        self.assertEqual(self.res.cross_index["ijk"], ((0, 1, 2),))

    def test_edges_per_call_and_shapes(self):
        e2 = np.stack([EDGES, 2 * EDGES])  # (nscale, nedges)
        res = self.op.g(self.shear, self.w, edges=e2, cross=False)
        self.assertEqual(res.edges.shape, (3, 2, EDGES.size))
        np.testing.assert_array_equal(res.edges[1, 1], 2 * EDGES)
        e3 = np.broadcast_to(EDGES, (3, 2, EDGES.size)).copy()
        e3[2] *= 0.5
        res = self.op.g(self.shear, self.w, edges=e3, cross=False)
        n, _, _, hist, _, _ = numpy_reduction(self.res.values["M_ap"][1], 0.5 * EDGES)
        np.testing.assert_array_equal(res.hist["M_ap"][2, 1], hist[2])
        with self.assertRaises(ValueError):
            self.op.g(self.shear, self.w, edges=EDGES[::-1])
        with self.assertRaises(ValueError):
            self.op.g(self.shear, self.w, edges=np.zeros((5, EDGES.size)))
        res = make(self.mask).g(self.shear, self.w, cross=False)
        self.assertIsNone(res.hist)
        with self.assertRaises(ValueError):
            res.pdf("M_ap")

    def test_non_finite_apertures_are_left_out(self):
        # bin 1 has no weight in the eastern half: every aperture reaching
        # into it has Σw = 0 there and is left out of the sums of *all* bins
        w = self.w.copy()
        theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
        w[1, phi > 1.5] = 0.0
        res = self.op.g(self.shear, w, cross=False, values=True)
        v = res.values["M_ap"][0]
        self.assertTrue(np.any(~np.isfinite(v[1])))
        n, power, _, hist, under, over = numpy_reduction(v, EDGES)
        self.assertLess(n, res.n_apertures[0])
        self.assertEqual(res.count["M_ap"][0], n)
        np.testing.assert_allclose(res.power_sums["M_ap"][:, 0], power, rtol=1e-12)
        np.testing.assert_array_equal(res.hist["M_ap"][:, 0], hist)

    def test_add(self):
        a = self.res
        b = self.op.g(self.shear, self.w, values=True)
        c = a + b
        self.assertIsInstance(c, OnePointResult)
        np.testing.assert_array_equal(c.n_apertures, 2 * a.n_apertures)
        np.testing.assert_array_equal(c.hist["M_x"], 2 * a.hist["M_x"])
        np.testing.assert_array_equal(c.power_sums["M_ap"], 2 * a.power_sums["M_ap"])
        self.assertEqual(c.values["M_ap"][0].shape[1], 2 * a.values["M_ap"][0].shape[1])
        np.testing.assert_allclose(c.moments("M_ap"), a.moments("M_ap"), rtol=1e-12)
        with self.assertRaises(ValueError):
            a + self.op.g(self.shear, self.w, cross=False)
        with self.assertRaises(ValueError):
            a + self.op.g(self.shear, self.w, edges=2 * EDGES)


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.mask = footprint_mask()

    def test_masked_centres_are_rejected(self):
        strict = make(self.mask, scales=(60.0,), min_coverage=0.95)
        loose = make(self.mask, scales=(60.0,), min_coverage=0.0)
        cs = strict.centres("g", 0)
        cl = loose.centres("g", 0)
        self.assertLess(cs["pix"].size, cl["pix"].size)
        self.assertTrue(np.all(cs["coverage"] >= 0.95))
        self.assertTrue(np.all(np.isin(cs["pix"], cl["pix"])))
        # min_coverage=0: every footprint centre whose disc holds an observed pixel
        fp = footprint_centres(self.mask, NSIDE, cl["centre_nside"])
        self.assertEqual(cl["n_candidates"], fp.size)
        self.assertTrue(np.all(cl["disc_size"] > 0))
        np.testing.assert_array_equal(cl["pix"], fp[np.isin(fp, cl["pix"])])
        rejected = np.setdiff1d(fp, cl["pix"])
        # the only rejection at min_coverage=0 is an empty disc
        self.assertEqual(rejected.size, 0)
        # a centre inside the hole is gone even at loose coverage once the
        # disc is small (the top hat of radius theta): the hole is 2 deg wide
        tophat = make(self.mask, scales=(60.0,), min_coverage=0.5)
        ct = tophat.centres("k", 0)
        hole = hp.query_disc(NSIDE, hp.ang2vec(1.55, 1.5), np.radians(0.5))
        self.assertFalse(np.any(np.isin(hole, ct["pix"])))

    def test_coarser_centres(self):
        op = make(self.mask, scales=(60.0,), centre_nside=16)
        c = op.centres("g", 0)
        self.assertEqual(c["centre_nside"], 16)
        self.assertTrue(np.all(c["pix"] < hp.nside2npix(16)))
        self.assertEqual(default_centre_nside(60.0, NSIDE), NSIDE)
        self.assertEqual(default_centre_nside(60.0, 1024), 128)  # 27' pixel ~ 30'
        self.assertEqual(default_centre_nside(20.0, 1024), 256)  # 13.7' ~ 10'
        with self.assertRaises(ValueError):
            make(self.mask, centre_nside=(16,))  # one per scale
        with self.assertRaises(ValueError):
            make(self.mask, centre_nside=48)
        with self.assertRaises(ValueError):
            make(self.mask, min_coverage=1.5)
        with self.assertRaises(ValueError):
            make(self.mask, k_aperture_filter="tophat")

    def test_no_centre_raises(self):
        # a 5 theta disc of 50 deg radius is never 80 % inside this footprint
        with self.assertRaises(ValueError):
            make(self.mask, scales=(600.0,)).g(*maps()[::2])


class TestPixelTopHat(unittest.TestCase):
    def test_matches_masked_ud_grade(self):
        mask = footprint_mask()
        _, kappa, w = maps()
        for ap in (None, 32):
            op = make(
                mask,
                scales=(60.0,),
                k_aperture_filter="healpix",
                centre_nside=16,
                aperture_nside=ap,
                min_coverage=0.5,
            )
            res = op.k(kappa, w, values=True)
            c = op.centres("k", 0)
            num = hp.ud_grade((w * kappa * mask), 16)
            den = hp.ud_grade((w * mask), 16)
            ref = num[:, c["pix"]] / den[:, c["pix"]]
            np.testing.assert_allclose(res.values["k"][0], ref, rtol=1e-12, atol=1e-15)
            # coverage = unmasked fraction of the centre pixel's map pixels,
            # with and without a coarse aperture level
            frac = hp.ud_grade(mask.astype(float), 16)[c["pix"]]
            np.testing.assert_allclose(c["coverage"], frac, rtol=1e-12)
            self.assertTrue(np.all(c["coverage"] >= 0.5))
        with self.assertRaises(ValueError):
            make(
                mask,
                scales=(60.0,),
                k_aperture_filter="healpix",
                centre_nside=64,
                aperture_nside=32,
            ).k(kappa, w)


class TestDevicesAndWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = footprint_mask()
        cls.shear, cls.kappa, cls.w = maps()

    def test_two_devices_equal_one(self):
        one = make(self.mask, edges=EDGES)
        two = make(self.mask, edges=EDGES, device=("cpu", "cpu"))
        a = one.g(self.shear, self.w)
        b = two.g(self.shear, self.w)
        np.testing.assert_array_equal(a.n_apertures, b.n_apertures)
        for name in a.names:
            np.testing.assert_array_equal(a.hist[name], b.hist[name])
            np.testing.assert_array_equal(a.count[name], b.count[name])
            np.testing.assert_allclose(
                a.power_sums[name], b.power_sums[name], rtol=1e-13, atol=0
            )
        ak = one.k(self.kappa, self.w)
        bk = two.k(self.kappa, self.w)
        np.testing.assert_allclose(ak.power_sums["k"], bk.power_sums["k"], rtol=1e-13)
        # a device array is refused on several devices
        self.assertEqual(len(two._parts("shear", 0)), 2)

    def test_emulated_gpu_equals_cpu(self):
        """The real cupy wrappers (numpy-emulated kernels) on every engine."""
        op = make(self.mask, edges=EDGES)
        ref = op.g(self.shear, self.w, values=True)
        refk = op.k(self.kappa, self.w, values=True)
        parts = [op._parts("shear", s)[0] for s in range(len(SCALES))] + [
            op._parts("tophat", s)[0] for s in range(len(SCALES))
        ]
        with contextlib.ExitStack() as stack:
            calls = [stack.enter_context(emulated_gpu(p)) for p in parts]
            got = op.g(self.shear, self.w, values=True)
            gotk = op.k(self.kappa, self.w, values=True)
            self.assertEqual(
                sum(c["aperture_tomo_shear_kernel"] for c in calls), 2 * len(SCALES)
            )
            # k's values, plus the weights-only Σw pass of both g and k
            self.assertEqual(
                sum(c["aperture_tomo_density_kernel"] for c in calls), 3 * len(SCALES)
            )
        for s in range(len(SCALES)):
            for name in ref.names:
                np.testing.assert_allclose(
                    got.values[name][s], ref.values[name][s], rtol=1e-12, atol=1e-16
                )
            np.testing.assert_allclose(
                gotk.values["k"][s], refk.values["k"][s], rtol=1e-12, atol=1e-16
            )
        for name in ref.names:
            np.testing.assert_array_equal(got.hist[name], ref.hist[name])
            np.testing.assert_allclose(
                got.power_sums[name], ref.power_sums[name], rtol=1e-12
            )

    def test_aperture_nside_without_pairs(self):
        """Coarse apertures (with quad cells) need no pair geometry and are
        the patch estimator's coarse apertures, bitwise."""
        op = make(self.mask, aperture_nside=32, min_coverage=0.5)
        res = op.g(self.shear, self.w, cross=False, values=True)
        resk = op.k(self.kappa, self.w, values=True)
        for s in range(len(SCALES)):
            corr = patch_estimator(op, "g", s, self.mask, aperture_nside=32)
            ref = corr.m_a(self.shear, self.w, return_device=False)
            self.assertTrue(np.array_equal(ref, res.values["M_ap"][s]))
            corr = patch_estimator(op, "k", s, self.mask, aperture_nside=32)
            ref = corr.m_g(self.kappa, self.w, return_device=False)
            self.assertTrue(np.array_equal(ref, resk.values["k"][s]))
        self.assertEqual(op.provenance["aperture_nside"], 32)


if __name__ == "__main__":
    unittest.main()
