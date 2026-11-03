"""The labelled result objects: i3PCF ζ from an IntegratedResult, host
copies of device results, and the small accessors and guards of the global
and one-point results.  Built directly from random arrays, no measurement."""

import itertools
import unittest

import numpy as np

from pixpoint import (
    Component,
    Integrated3PCF,
    IntegratedResult,
    OnePointResult,
    ThreePointResult,
    TwoPointResult,
)

from ._results import ZETA_DEFINITIONS, zeta_reference

NBINS = 4
N_PATCHES = 9


class _DeviceArray:
    """Stands in for a cupy array: not an ndarray, copied to the host by
    ``get()``."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def get(self):
        return self._a.copy()


def _integrated(n_source, n_lens, seed=0):
    rng = np.random.default_rng(seed)
    gg = tuple(itertools.combinations_with_replacement(range(n_source), 2))
    kk = tuple(itertools.combinations_with_replacement(range(n_lens), 2))
    kg = tuple((lens, s) for lens in range(n_lens) for s in range(n_source))
    shape = (N_PATCHES, NBINS)
    return IntegratedResult(
        binedges=np.radians(np.geomspace(10, 100, NBINS + 1) / 60),
        M_a=rng.normal(size=(n_source, N_PATCHES)),
        M_g=rng.normal(size=(n_lens, N_PATCHES)),
        xi_p=rng.normal(size=(len(gg),) + shape),
        xi_m=rng.normal(size=(len(gg),) + shape),
        xi_g=rng.normal(size=(len(kk),) + shape),
        xi_t=rng.normal(size=(len(kg),) + shape),
        gg_combinations=gg,
        kk_combinations=kk,
        kg_combinations=kg,
    )


class TestIntegratedZetas(unittest.TestCase):
    def test_integrated_zetas_matches_explicit_patch_average(self):
        # 4 source and 2 lens bins: no annulus has the combination count of
        # a triangle over its centre's bins unless it really is one, so the
        # inferred layouts are unambiguous.
        res = _integrated(n_source=4, n_lens=2)
        zetas = res.zetas()
        self.assertEqual(set(zetas), set(ZETA_DEFINITIONS))
        for name, value in zetas.items():
            with self.subTest(estimator=name):
                ref = zeta_reference(res, name)
                self.assertEqual(value.shape, ref.shape)
                np.testing.assert_allclose(value, ref, rtol=1e-12, atol=1e-15)

    def test_documented_row_order(self):
        res = _integrated(n_source=4, n_lens=2)
        zetas = res.zetas()
        self.assertEqual(
            zetas["zeta_a_plus"].shape[0], len(Integrated3PCF.zeta_triplets(4))
        )
        self.assertEqual(
            zetas["zeta_g_g"].shape[0], len(Integrated3PCF.zeta_triplets(2))
        )
        self.assertEqual(
            zetas["zeta_g_plus"].shape[0],
            len(Integrated3PCF.zeta_cross_triplets(2, len(res.gg_combinations))),
        )
        self.assertEqual(
            zetas["zeta_a_t"].shape[0],
            len(Integrated3PCF.zeta_cross_triplets(4, len(res.kg_combinations))),
        )

    def test_symmetric_override_for_equal_bin_counts(self):
        # 2 source and 2 lens bins: xi_p has 3 combinations, the size of a
        # triangle over 2 lens bins, so zeta_g_plus is ambiguous and needs
        # the documented override to get the cross layout.
        res = _integrated(n_source=2, n_lens=2)
        zetas = res.zetas(symmetric={"zeta_g_plus": False, "zeta_a_g": False})
        for name in ("zeta_g_plus", "zeta_a_g"):
            with self.subTest(estimator=name):
                np.testing.assert_allclose(
                    zetas[name], zeta_reference(res, name), rtol=1e-12, atol=1e-15
                )

    def test_unmeasured_fields_form_no_zeta(self):
        res = _integrated(n_source=2, n_lens=1)
        res.M_g = res.xi_g = res.xi_t = None
        self.assertEqual(set(res.zetas()), {"zeta_a_plus", "zeta_a_minus"})


class TestToNumpy(unittest.TestCase):
    def test_integrated_to_numpy(self):
        host = _integrated(n_source=2, n_lens=1)
        host.xi_g = None
        device = IntegratedResult(
            binedges=host.binedges,
            **{
                k: (_DeviceArray(v) if v is not None else None)
                for k, v in (
                    (k, getattr(host, k))
                    for k in ("M_a", "M_g", "xi_p", "xi_m", "xi_g", "xi_t")
                )
            },
            gg_combinations=host.gg_combinations,
            kk_combinations=host.kk_combinations,
            kg_combinations=host.kg_combinations,
        )
        for res in (host, device):
            out = res.to_numpy()
            self.assertIsNot(out, res)
            self.assertIsNone(out.xi_g)
            self.assertEqual(out.gg_combinations, host.gg_combinations)
            self.assertEqual(out.kg_combinations, host.kg_combinations)
            np.testing.assert_array_equal(out.binedges, host.binedges)
            for name, value in out.fields().items():
                with self.subTest(source=type(res.M_a).__name__, field=name):
                    self.assertIsInstance(value, np.ndarray)
                    np.testing.assert_array_equal(value, getattr(host, name))

    def test_two_point_to_numpy(self):
        rng = np.random.default_rng(1)
        num = {"xi_p": rng.normal(size=(3, NBINS)), "xi_m": rng.normal(size=(3, NBINS))}
        weight = rng.uniform(1, 2, size=(3, NBINS))
        weight[0, 0] = 0.0  # a bin without pairs
        args = ("gg", np.radians([0.1, 0.2, 0.4, 0.8, 1.6]), ((0, 0), (0, 1), (1, 1)))
        host = TwoPointResult(*args, num, weight)
        device = TwoPointResult(
            *args, {k: _DeviceArray(v) for k, v in num.items()}, _DeviceArray(weight)
        )
        out = device.to_numpy()
        self.assertEqual(out.statistic, "gg")
        self.assertEqual(out.combinations, host.combinations)
        self.assertIsInstance(out.weight, np.ndarray)
        for name in num:
            self.assertIsInstance(out.num[name], np.ndarray)
        np.testing.assert_array_equal(out.xi_p, host.xi_p)
        np.testing.assert_array_equal(out.xi_m, host.xi_m)
        self.assertEqual(out.xi_p[0, 0], 0.0)


def _three_point(n_max=2, label="x", seed=2):
    rng = np.random.default_rng(seed)
    shape = (2 * n_max + 1, 1, 1, 1, 2, 2)
    return ThreePointResult(
        n_max=n_max,
        binedges=np.radians(np.array([10.0, 40.0, 160.0]) / 60),
        components=(Component(label, ("g", "g", "g")),),
        spins={"g": 2},
        num={label: rng.normal(size=shape) + 1j * rng.normal(size=shape)},
        norm={label: rng.normal(size=shape) + 1j * rng.normal(size=shape)},
    )


class TestThreePointResult(unittest.TestCase):
    def test_three_point_properties_and_add_mismatch(self):
        res = _three_point()
        np.testing.assert_allclose(res.theta, [20.0, 80.0], rtol=1e-12)
        self.assertEqual(res.labels, ("x",))
        np.testing.assert_array_equal(res.n, [-2, -1, 0, 1, 2])
        for other in (
            _three_point(n_max=1),
            _three_point(label="y"),
            ThreePointResult(
                res.n_max,
                res.binedges * 2,
                res.components,
                res.spins,
                res.num,
                res.norm,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "same definition"):
                res + other

    def test_binned_default_edges(self):
        res = _three_point()
        default = res.binned()
        explicit = res.binned(np.linspace(-np.pi, np.pi, 17))
        self.assertEqual(default["x"].shape, (1, 1, 1, 2, 2, 16))
        np.testing.assert_array_equal(default["x"], explicit["x"])
        np.testing.assert_array_equal(
            res.binned(nphi=8)["x"], res.binned(np.linspace(-np.pi, np.pi, 9))["x"]
        )


def _one_point(seed, values=False):
    rng = np.random.default_rng(seed)
    nz, nscale = 3, 2
    cross_index = {
        "ij": tuple(itertools.combinations(range(nz), 2)),
        "iij": tuple((i, j) for i in range(nz) for j in range(nz) if i != j),
        "ijk": tuple(itertools.combinations(range(nz), 3)),
    }
    return OnePointResult(
        statistic="k",
        scales=np.array([10.0, 20.0]),
        names=("k",),
        n_apertures=np.array([100, 80]),
        count={"k": np.array([97.0, 80.0])},
        power_sums={"k": rng.normal(size=(nz, nscale, 5))},
        cross_sums={
            "k": {
                key: rng.normal(size=(len(idx), nscale))
                for key, idx in cross_index.items()
            }
        },
        cross_index=cross_index,
    )


class TestOnePointResult(unittest.TestCase):
    def test_onepoint_shape_properties_and_add_without_histogram(self):
        a, b = _one_point(3), _one_point(4)
        self.assertEqual(a.nz, 3)
        self.assertEqual(a.nscale, 2)
        s = a + b
        self.assertIsNone(s.edges)
        self.assertIsNone(s.hist)
        self.assertIsNone(s.underflow)
        self.assertIsNone(s.overflow)
        self.assertIsNone(s.values)
        np.testing.assert_array_equal(s.n_apertures, [200, 160])
        np.testing.assert_array_equal(s.count["k"], a.count["k"] + b.count["k"])
        np.testing.assert_array_equal(
            s.power_sums["k"], a.power_sums["k"] + b.power_sums["k"]
        )
        for key in a.cross_index:
            np.testing.assert_array_equal(
                s.cross_sums["k"][key], a.cross_sums["k"][key] + b.cross_sums["k"][key]
            )
        with self.assertRaisesRegex(ValueError, "no histogram"):
            s.pdf("k")


if __name__ == "__main__":
    unittest.main()
