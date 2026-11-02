#!/usr/bin/env python
"""Stage B5: cost of the moment-carrying cells in the library (A100).

Times the pair pass (``gg(apertures=False)``, ``kg``, ``three_by_two_pt``) of an
``Integrated3PCF`` on the DES Y3 mask for ``cell_moments`` None / dipole / quad,
unpacked and packed, nz = 1 and 4, and the aperture pass (``m_a``, ``m_g`` with a
filter) at a coarse aperture level for the three variants.  Device-resident
inputs, median of several calls, CUDA-synchronised.

Usage (el9, A100):
    benchmarks/static_treecode/remote_run.sh \\
        "python benchmarks/moments/bench_cell_moments.py --out benchmarks/moments/results/bench_512.json"
"""

import argparse
import json
import os
import time

import healpy as hp
import numpy as np

P = "/project/ocean1-seitz/users/dgebauer"
DEFAULT_MASK = f"{P}/research/CosmoFuse-treecode-data/DESY3_Mask.fits"


def timed(fn, sync, n=7):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--npatch", type=int, default=450)
    ap.add_argument("--k", type=float, nargs="+", default=[4.0])
    ap.add_argument("--theta-min", type=float, default=7.0)
    ap.add_argument("--theta-max", type=float, default=123.0)
    ap.add_argument("--nbins", type=int, default=9)
    ap.add_argument("--Q", type=float, default=90.0)
    ap.add_argument("--aperture-nside", type=int, nargs="*", default=[128, 256])
    ap.add_argument("--nz", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pixpoint import Integrated3PCF, U_crittenden
    from pixpoint.utils import select_patch_centers

    gpu = args.device == "gpu"
    if gpu:
        import cupy as cp

        def sync():
            cp.cuda.runtime.deviceSynchronize()

        xp = cp
    else:
        xp = np

        def sync():
            pass

    m = hp.read_map(args.mask)
    mask = (
        hp.ud_grade(m.astype(np.float64), args.nside)
        if hp.npix2nside(m.size) != args.nside
        else m
    ) != 0
    phi_c, theta_c = select_patch_centers(mask, 8, patch_size=args.Q, theta_Q=args.Q)
    phi_c, theta_c = phi_c[: args.npatch], theta_c[: args.npatch]
    row_pix = np.flatnonzero(mask)
    rng = np.random.default_rng(1)
    report = {
        "args": vars(args),
        "n_patches": int(phi_c.size),
        "n_active": int(row_pix.size),
        "runs": [],
    }

    def make(cm, pack=False, ap=None, k=None):
        return Integrated3PCF(
            args.nside,
            phi_c,
            theta_c,
            nbins=args.nbins,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            patch_size=args.Q,
            theta_Q=args.Q,
            mask=mask,
            device=args.device,
            map_precision="float32",
            rotation_precision="float32",
            accumulation_precision="float64",
            resolution_factor=args.k[0] if k is None else k,
            pair_search_precision="float64",
            memory_budget_gb=float("inf"),
            pack_pairs=pack,
            cell_moments=cm,
            aperture_nside=ap,
        )

    for nz in args.nz:
        shear = xp.asarray(
            rng.normal(0, 0.03, (nz, 2, row_pix.size)).astype(np.float32)
        )
        w = xp.asarray(rng.uniform(0.5, 1.5, (nz, row_pix.size)).astype(np.float32))
        dens = xp.asarray(rng.normal(0, 1, (nz, row_pix.size)).astype(np.float32))
        for k in args.k:
            for pack in (False, True):
                for cm in (None, "dipole", "quad"):
                    t0 = time.perf_counter()
                    corr = make(cm, pack=pack, k=k)
                    corr.preprocess(release_host_pairs=True)
                    sync()
                    t_pre = time.perf_counter() - t0
                    run = {
                        "nz": nz,
                        "k": k,
                        "pack_pairs": pack,
                        "cell_moments": cm,
                        "preprocess_s": t_pre,
                        "pairs": int(corr.ntotpairs),
                        "levels": [int(v) for v in corr.level_nside],
                    }
                    run["gg_pairs_ms"] = timed(
                        lambda: corr.gg(shear, w, flip_g1=True, apertures=False), sync
                    )
                    run["kg_ms"] = timed(lambda: corr.kg(dens, shear, w, w), sync)
                    run["three_by_two_pt_ms"] = timed(
                        lambda: corr.three_by_two_pt(shear, dens, weights=(w, w)), sync
                    )
                    print(json.dumps(run), flush=True)
                    report["runs"].append(run)
                    del corr
                    if gpu:
                        cp.get_default_memory_pool().free_all_blocks()
        # aperture pass
        for ap_nside in args.aperture_nside:
            for cm in (None, "dipole", "quad"):
                corr = make(cm, pack=True, ap=ap_nside)
                corr.preprocess(release_host_pairs=True)
                sync()
                run = {"nz": nz, "aperture_nside": ap_nside, "cell_moments": cm}
                run["m_a_ms"] = timed(lambda: corr.m_a(shear, w, flip_g1=True), sync)
                run["m_g_filter_ms"] = timed(
                    lambda: corr.m_g(dens, w, aperture_filter=U_crittenden), sync
                )
                run["gg_with_apertures_ms"] = timed(
                    lambda: corr.gg(shear, w, flip_g1=True), sync
                )
                print(json.dumps(run), flush=True)
                report["runs"].append(run)
                del corr
                if gpu:
                    cp.get_default_memory_pool().free_all_blocks()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
