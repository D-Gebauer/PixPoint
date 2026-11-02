#!/usr/bin/env python
"""B5: the moment 3x2pt tile, one walk (tile_multi_mom, 82 accumulators, spills
~500 B at nz = 4 quad) against the standalone moment tiles (tile_xipm_mom +
tile_ds_mom + the plain xi_g tile, three walks, no spill).

nside 1024, 200 patches of Q90, k = 4, nz = 4, quad and dipole, packed and
unpacked, device-resident float32 maps, median of 7.  The standalone path is
forced by taking the multi moment kernel off the backend; the plain tiles of
the fine bins are untouched either way.

    benchmarks/static_treecode/remote_run.sh "python benchmarks/moments/b5_3x2pt_probe.py"
"""

import json
import time

import cupy as cp
import healpy as hp
import numpy as np

from CosmoFuse import Integrated3PCF
from CosmoFuse.utils import select_patch_centers

P = "/project/ocean1-seitz/users/dgebauer"
MASK = f"{P}/research/CosmoFuse-treecode-data/DESY3_Mask.fits"


def sync():
    cp.cuda.runtime.deviceSynchronize()


def timed(fn, n=7):
    fn()
    sync()
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def main():
    nside, nz = 1024, 4
    m = hp.read_map(MASK)
    mask = hp.ud_grade(m.astype(np.float64), nside) != 0
    phi_c, theta_c = select_patch_centers(mask, 8, patch_size=90.0, theta_Q=90.0)
    phi_c, theta_c = phi_c[:200], theta_c[:200]
    n = int(np.count_nonzero(mask))
    rng = np.random.default_rng(1)
    shear = cp.asarray(rng.normal(0, 0.03, (nz, 2, n)).astype(np.float32))
    w = cp.asarray(rng.uniform(0.5, 1.5, (nz, n)).astype(np.float32))
    dens = cp.asarray(rng.normal(0, 1, (nz, n)).astype(np.float32))
    for pack in (False, True):
        for cm in ("dipole", "quad"):
            corr = Integrated3PCF(
                nside,
                phi_c,
                theta_c,
                nbins=9,
                theta_min=7.0,
                theta_max=123.0,
                patch_size=90.0,
                theta_Q=90.0,
                mask=mask,
                device="gpu",
                map_precision="float32",
                rotation_precision="float32",
                accumulation_precision="float64",
                resolution_factor=4.0,
                memory_budget_gb=float("inf"),
                pack_pairs=pack,
                cell_moments=cm,
            )
            corr.preprocess(release_host_pairs=True)
            sync()
            run = {"pack_pairs": pack, "cell_moments": cm, "pairs": int(corr.ntotpairs)}
            call = lambda: corr.three_by_two_pt(
                shear, dens, weights=(w, w)
            )  # noqa: E731
            run["multi_ms"] = timed(call)
            ref = corr.three_by_two_pt(shear, dens, weights=(w, w))
            multi = corr.backend.kernel_3x2pt_tomo_moments
            corr.backend.kernel_3x2pt_tomo_moments = None
            run["standalone_ms"] = timed(call)
            alt = corr.three_by_two_pt(shear, dens, weights=(w, w))
            corr.backend.kernel_3x2pt_tomo_moments = multi
            run["bitwise"] = all(
                bool(cp.array_equal(getattr(ref, k), getattr(alt, k)))
                for k in ("xi_p", "xi_m", "xi_g", "xi_t")
            )
            print(json.dumps(run), flush=True)
            del corr
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
