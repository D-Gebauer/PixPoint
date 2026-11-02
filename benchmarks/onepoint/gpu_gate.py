#!/usr/bin/env python
"""Real-GPU gate for ``OnePoint`` (stage D): the cupy backend against the CPU
backend on a small footprint, float64 maps and rotations.

Per-aperture values must agree to 1e-12 (the GPU aperture kernels reduce in
a different order), the histogram bitwise, the sums to 1e-12; a one-element
device list equals the plain device; ``aperture_nside`` with quad cells and
the ``"healpix"`` top hat run on the GPU.  Exit status 0 = pass.

    benchmarks/static_treecode/remote_run.sh "python benchmarks/onepoint/gpu_gate.py"
"""

import sys

import healpy as hp
import numpy as np

from CosmoFuse import OnePoint, U_crittenden

NSIDE = 128
NPIX = hp.nside2npix(NSIDE)


def footprint():
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta > 1.2) & (theta < 1.9) & (phi > 0.5) & (phi < 2.5)
    mask[hp.query_disc(NSIDE, hp.ang2vec(1.55, 1.5), np.radians(2.0))] = False
    return mask


def compare(label, a, b, rtol):
    err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)
    ok = err <= rtol
    print(f"{'PASS' if ok else 'FAIL'} {label}: max rel diff {err:.2e} (tol {rtol:g})")
    return ok


def check(label, cond):
    print(f"{'PASS' if cond else 'FAIL'} {label}")
    return bool(cond)


def main():
    import cupy as cp

    rng = np.random.default_rng(3)
    mask = footprint()
    nz = 3
    shear = rng.normal(size=(nz, 2, NPIX)) * 0.05
    kappa = rng.normal(size=(nz, NPIX)) * 0.02
    w = 0.5 + rng.random((nz, NPIX))
    w[1, hp.pix2ang(NSIDE, np.arange(NPIX))[1] > 2.3] = 0.0  # a bin with holes
    edges = np.linspace(-0.05, 0.05, 41)
    ok = True
    for kw in (
        {},
        {"aperture_nside": 64},
        {"k_aperture_filter": U_crittenden},
        {"k_aperture_filter": "healpix", "centre_nside": 32, "aperture_nside": 64},
    ):
        label = str(kw) or "default"
        common = dict(scales=(30.0, 60.0), rotation_precision="float64", edges=edges)
        common.update(kw)
        cpu = OnePoint(NSIDE, mask, device="cpu", **common)
        gpu = OnePoint(NSIDE, mask, device="gpu", **common)
        lst = OnePoint(NSIDE, mask, device=[0], **common)
        for stat in ("g", "k"):
            if stat == "g" and "k_aperture_filter" in kw:
                continue
            args = (shear, w) if stat == "g" else (kappa, w)
            rc = getattr(cpu, stat)(*args, values=True)
            rg = getattr(gpu, stat)(*args, values=True)
            rl = getattr(lst, stat)(*args, values=True)
            for name in rc.names:
                for s in range(2):
                    vc, vg = rc.values[name][s], rg.values[name][s]
                    # an empty aperture is NaN on the CPU and 0 on the GPU
                    # (kernel conventions); OnePoint drops it from the sums
                    # either way, so only the finite values are compared
                    fin = np.isfinite(vc)
                    ok &= compare(
                        f"{label} {stat} {name} scale {s} values",
                        vg[fin],
                        vc[fin],
                        1e-12,
                    )
                    ok &= check(
                        f"{label} {stat} {name} scale {s} device list bitwise",
                        np.array_equal(vg[fin], rl.values[name][s][fin]),
                    )
                ok &= check(
                    f"{label} {stat} {name} hist bitwise",
                    np.array_equal(rc.hist[name], rg.hist[name]),
                )
                ok &= check(
                    f"{label} {stat} {name} count",
                    np.array_equal(rc.count[name], rg.count[name]),
                )
                ok &= compare(
                    f"{label} {stat} {name} power sums",
                    rg.power_sums[name],
                    rc.power_sums[name],
                    1e-12,
                )
                for k in ("ij", "iij", "ijk"):
                    if rc.cross_sums[name][k].size:
                        ok &= compare(
                            f"{label} {stat} {name} cross {k}",
                            rg.cross_sums[name][k],
                            rc.cross_sums[name][k],
                            1e-12,
                        )
            print(
                f"      {label} {stat}: n_apertures {rc.n_apertures}, count {rc.count[rc.names[0]]}"
            )
        cp.get_default_memory_pool().free_all_blocks()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
