#!/usr/bin/env python
"""Stage C5: cost of the edge-splitting pair search in the library (A100).

For ``Integrated3PCF`` (DES Y3 mask, patches of Q90) and ``Global2PCF`` (the
whole footprint) at nside 512 / 1024 and k = 4 (quad cells): the pair-search
time of the static list and of the split (``edge_slop=0``), the pair counts,
the device memory of the geometry (unpacked / packed) and the per-map cost of
``gg(apertures=False)``, ``kg`` and ``three_by_two_pt`` with device-resident
float32 maps (median of several calls, CUDA-synchronised).

Usage (el9, A100):
    benchmarks/static_treecode/remote_run.sh \\
        "python benchmarks/edge_split/bench_edge_split.py --nside 512 \\
         --out benchmarks/edge_split/results/bench_512.json"
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


def device_bytes(est):
    """Bytes of the pair geometry an estimator holds on the device."""
    import cupy as cp

    engines = getattr(est, "_engines", None) or [est]
    total = 0
    for e in engines:
        ctx = e.compute_context
        for name in (
            "inds_dev",
            "exp2phi_dev",
            "inds_i_dev",
            "inds_j_dev",
            "packed_pairs_dev",
            "packed_perm_dev",
            "moment_phi_dev",
            "moment_inv_d_dev",
            "moment_d_codes_dev",
            "packed_cell_of_row_dev",
        ):
            a = getattr(ctx, name, None)
            if isinstance(a, cp.ndarray):
                total += int(a.nbytes)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--npatch", type=int, default=450)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--theta-min", type=float, default=7.0)
    ap.add_argument("--theta-max", type=float, default=123.0)
    ap.add_argument("--nbins", type=int, default=9)
    ap.add_argument("--Q", type=float, default=90.0)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--global-theta-max", type=float, default=250.0)
    ap.add_argument("--global-nbins", type=int, default=8)
    ap.add_argument("--global-theta-min", type=float, default=15.0)
    ap.add_argument("--tile-nside", type=int, default=None)
    ap.add_argument("--skip-global", action="store_true")
    ap.add_argument("--skip-patches", action="store_true")
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from CosmoFuse import Global2PCF, Integrated3PCF
    from CosmoFuse.utils import select_patch_centers

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
    nz = args.nz
    shear = xp.asarray(rng.normal(0, 0.03, (nz, 2, row_pix.size)).astype(np.float32))
    w = xp.asarray(rng.uniform(0.5, 1.5, (nz, row_pix.size)).astype(np.float32))
    dens = xp.asarray(rng.normal(0, 1, (nz, row_pix.size)).astype(np.float32))
    report = {
        "args": vars(args),
        "n_patches": int(phi_c.size),
        "n_active": int(row_pix.size),
        "runs": [],
    }

    def common(**kw):
        base = dict(
            device=args.device,
            map_precision="float32",
            rotation_precision="float32",
            accumulation_precision="float64",
            resolution_factor=args.k,
            pair_search_precision="float64",
            cell_moments="quad",
        )
        base.update(kw)
        return base

    def measure(est, run):
        # the patch estimator's gg also measures apertures unless told not to
        gg_kw = {} if isinstance(est, Global2PCF) else {"apertures": False}
        run["gg_pairs_ms"] = timed(
            lambda: est.gg(shear, w, flip_g1=True, **gg_kw), sync
        )
        run["kg_ms"] = timed(lambda: est.kg(dens, shear, w, w), sync)
        # Integrated3PCF takes weights=(shear_w, k_w); Global2PCF takes
        # shear_weights, k_weights positionally after the maps.
        if isinstance(est, Global2PCF):
            run["three_by_two_pt_ms"] = timed(
                lambda: est.three_by_two_pt(shear, dens, w, w), sync
            )
        else:
            run["three_by_two_pt_ms"] = timed(
                lambda: est.three_by_two_pt(shear, dens, weights=(w, w)), sync
            )

    if not args.skip_patches:
        for slop in (None, 0.0):
            for pack in (False, True):
                corr = Integrated3PCF(
                    args.nside,
                    phi_c,
                    theta_c,
                    nbins=args.nbins,
                    theta_min=args.theta_min,
                    theta_max=args.theta_max,
                    patch_size=args.Q,
                    theta_Q=args.Q,
                    mask=mask,
                    memory_budget_gb=float("inf"),
                    pack_pairs=pack,
                    edge_slop=slop,
                    **common(),
                )
                t0 = time.perf_counter()
                corr.find_pairs()
                t_search = time.perf_counter() - t0
                t0 = time.perf_counter()
                corr.prepare(release_host_pairs=True)
                sync()
                t_prepare = time.perf_counter() - t0
                run = {
                    "estimator": "Integrated3PCF",
                    "edge_slop": slop,
                    "pack_pairs": pack,
                    "find_pairs_s": t_search,
                    "prepare_s": t_prepare,
                    "pairs": int(corr.ntotpairs),
                    "levels": [int(v) for v in corr.level_nside],
                    "device_bytes": device_bytes(corr) if gpu else None,
                }
                measure(corr, run)
                print(json.dumps(run), flush=True)
                report["runs"].append(run)
                del corr
                if gpu:
                    cp.get_default_memory_pool().free_all_blocks()

    if not args.skip_global:
        for slop in (None, 0.0):
            for pack in (False, True):
                try:
                    g2 = Global2PCF(
                        args.nside,
                        mask=mask,
                        nbins=args.global_nbins,
                        theta_min=args.global_theta_min,
                        theta_max=args.global_theta_max,
                        tile_nside=args.tile_nside,
                        pack_pairs=pack,
                        edge_slop=slop,
                        **common(),
                    )
                    t0 = time.perf_counter()
                    g2.find_pairs()
                    t_search = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    g2.prepare(release_host_pairs=True)
                    sync()
                    t_prepare = time.perf_counter() - t0
                except (ValueError, MemoryError) as exc:
                    run = {
                        "estimator": "Global2PCF",
                        "edge_slop": slop,
                        "pack_pairs": pack,
                        "error": str(exc)[:300],
                    }
                    print(json.dumps(run), flush=True)
                    report["runs"].append(run)
                    if gpu:
                        cp.get_default_memory_pool().free_all_blocks()
                    continue
                run = {
                    "estimator": "Global2PCF",
                    "edge_slop": slop,
                    "pack_pairs": pack,
                    "tile_nside": int(g2.tile_nside),
                    "n_tiles": int(g2.n_tiles),
                    "find_pairs_s": t_search,
                    "prepare_s": t_prepare,
                    "pairs": int(g2.ntotpairs),
                    "levels": [int(v) for v in g2.level_table["nside"]],
                    "device_bytes": device_bytes(g2) if gpu else None,
                }
                measure(g2, run)
                print(json.dumps(run), flush=True)
                report["runs"].append(run)
                del g2
                if gpu:
                    cp.get_default_memory_pool().free_all_blocks()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
