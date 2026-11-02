"""``ResultWriter`` (the shared streaming HDF5 base) and ``OnePointWriter``.

Gates: every streamed row equals the result it came from; the constants
(scales, edges) and the provenance are written once; ``n_flushed`` is the
durable prefix and ``resume`` continues from it; a changed layout is
refused; a ``TwoPointResult`` streams through the plain base via
``result_arrays``; the ``ZetaWriter`` gates in ``test_zeta_writer.py`` are
untouched by the refactor (they run on the same base).
"""

import json
import os
import tempfile
import unittest

import h5py
import healpy as hp
import numpy as np

from CosmoFuse import OnePoint, OnePointWriter, ResultWriter, result_arrays
from CosmoFuse.results import TwoPointResult

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
EDGES = np.linspace(-0.1, 0.1, 9)


def make_onepoint():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta > 1.2) & (theta < 1.9) & (phi > 0.5) & (phi < 2.5)
    return OnePoint(
        NSIDE, mask, scales=(120.0, 240.0), device="cpu", edges=EDGES, min_coverage=0.5
    )


def make_results(op, n=4, values=False):
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        shear = rng.normal(size=(2, 2, NPIX)) * 0.05
        w = 0.5 + rng.random((2, NPIX))
        out.append(op.g(shear, w, values=values))
    return out


class WriterBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = make_onepoint()
        cls.results = make_results(cls.op)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "out.h5")

    def tearDown(self):
        self._tmp.cleanup()


class TestOnePointWriter(WriterBase):
    def test_rows_equal_the_results(self):
        with OnePointWriter(self.path, self.op, flush_every=2) as out:
            for r in self.results:
                out.submit(r)
            self.assertEqual(out.n_submitted, len(self.results))
        with h5py.File(self.path, "r") as f:
            g = f["onepoint"]
            self.assertEqual(int(f["n_flushed"][()]), len(self.results))
            for i, r in enumerate(self.results):
                arrays = OnePointWriter.arrays(r)
                for name, arr in arrays.items():
                    np.testing.assert_array_equal(g[name][i], arr, err_msg=name)
                self.assertEqual(set(g["hist"]), set(r.names))
            np.testing.assert_array_equal(f["constants/scales"][:], self.op.scales)
            np.testing.assert_array_equal(
                f["constants/edges"][:], self.results[0].edges
            )
            self.assertEqual(f.attrs["statistic"], "g")
            self.assertEqual(json.loads(f.attrs["names"]), ["M_ap", "M_x"])
            self.assertEqual(json.loads(f.attrs["cross_index"])["ij"], [[0, 1]])
            self.assertEqual(int(f.attrs["nside"]), NSIDE)
            self.assertEqual(f.attrs["estimator"], "OnePoint")
            self.assertEqual(f.attrs["aperture_filter"], "Q_crittenden")
            self.assertIn("cosmofuse_version", f.attrs)
            # a sum streamed row by row adds up to the summed result
            total = sum(self.results[1:], self.results[0])
            np.testing.assert_array_equal(
                g["power_sums/M_ap"][:].sum(axis=0), total.power_sums["M_ap"]
            )

    def test_values_and_cross_sums_layout(self):
        res = make_results(self.op, n=2, values=True)
        with OnePointWriter(self.path, self.op) as out:
            for r in res:
                out.submit(r)
        with h5py.File(self.path, "r") as f:
            g = f["onepoint"]
            for s in range(2):
                np.testing.assert_array_equal(
                    g[f"values/M_ap/{s}"][1], res[1].values["M_ap"][s]
                )
            self.assertIn("cross_sums/M_ap/ij", g)
            self.assertNotIn("cross_sums/M_ap/ijk", g)  # empty at nz = 2
        # a result with another layout is refused (surfaced on close)
        out = OnePointWriter(self.path, self.op)
        out.submit(res[0])
        out.submit(self.results[0])
        with self.assertRaises(RuntimeError):
            out.close()

    def test_rejects_anything_but_a_result(self):
        with OnePointWriter(self.path, self.op) as out:
            with self.assertRaises(TypeError):
                out.submit({"x": np.zeros(3)})

    def test_resume_continues_after_the_flushed_prefix(self):
        with OnePointWriter(self.path, self.op, flush_every=2) as out:
            for r in self.results[:3]:
                out.submit(r)
        with h5py.File(self.path, "r") as f:
            self.assertEqual(int(f["n_flushed"][()]), 3)  # close() flushes
            self.assertEqual(f["onepoint/n_apertures"].shape[0], 3)
        # simulate a crash after the flush at 2: truncate n_flushed
        with h5py.File(self.path, "a") as f:
            f["n_flushed"][()] = 2
        with OnePointWriter(self.path, self.op, flush_every=2, resume=True) as out:
            out.submit(self.results[3])
        with h5py.File(self.path, "r") as f:
            self.assertEqual(f["onepoint/n_apertures"].shape[0], 3)
            np.testing.assert_array_equal(
                f["onepoint/power_sums/M_x"][2], self.results[3].power_sums["M_x"]
            )
            self.assertEqual(int(f["n_flushed"][()]), 3)


class TestPlainResultWriter(WriterBase):
    def test_two_point_results_stream_through_result_arrays(self):
        rng = np.random.default_rng(1)
        edges = np.geomspace(1e-3, 1e-2, 4)
        combos = ((0, 0), (0, 1), (1, 1))
        results = [
            TwoPointResult(
                "gg",
                edges,
                combos,
                {"xi_p": rng.normal(size=(3, 3)), "xi_m": rng.normal(size=(3, 3))},
                rng.random((3, 3)),
            )
            for _ in range(3)
        ]
        prov = {"nside": 64, "level_table": {"nside": [64, 64]}, "flag": True}
        with ResultWriter(
            self.path, prov, group="gg", constants={"binedges": edges}
        ) as out:
            for r in results:
                out.submit(result_arrays(r))
        with h5py.File(self.path, "r") as f:
            for i, r in enumerate(results):
                np.testing.assert_array_equal(f["gg/num/xi_p"][i], r.num["xi_p"])
                np.testing.assert_array_equal(f["gg/weight"][i], r.weight)
            np.testing.assert_array_equal(f["constants/binedges"][:], edges)
            self.assertEqual(int(f.attrs["nside"]), 64)
            self.assertEqual(json.loads(f.attrs["level_table"]), {"nside": [64, 64]})
            self.assertTrue(bool(f.attrs["flag"]))
            self.assertEqual(int(f["n_flushed"][()]), 3)

    def test_mismatched_shapes_fail_on_close(self):
        out = ResultWriter(self.path, {})
        out.submit({"a": np.zeros(3)})
        out.submit({"a": np.zeros(4)})
        with self.assertRaises(RuntimeError):
            out.close()
        with self.assertRaises(TypeError):
            ResultWriter(self.path, {}).submit({})
        with self.assertRaises(TypeError):
            result_arrays(np.zeros(3))


if __name__ == "__main__":
    unittest.main()
