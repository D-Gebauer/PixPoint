"""Cost of the tiling of Global2PCF with packed pairs (8.1.0).

    python bench_tiles.py --nside 1024 --mask /path/DESY3_Mask.fits \
        --tiles 8 16 0 --resolution-factor 2.9 --out tiles_1024_k2.9.json

``--tiles 0`` is the automatic choice (``tile_nside=None``).  For every tiling:
construction (including the packed-block check), ``find_pairs``
(``pack_host_pairs=True``), ``prepare``, the median per-map ``gg``/``kk``/``kg``
(float32 maps, float64 accumulation, device-resident, synchronised), pairs,
tiles, the bound on the largest packed row block against the largest real one,
and the device pool.  Random maps; the pair count does not depend on the
tiling, so the times compare one estimator.
"""

import argparse
import gc
import json
import time
import warnings

import healpy as hp
import numpy as np


def sync():
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def free_device():
    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def pool_gb():
    try:
        import cupy

        return cupy.get_default_memory_pool().total_bytes() / 1e9
    except Exception:
        return None


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), ts


def run_tiling(args, mask, maps, tile):
    from pixpoint import Global2PCF

    out = {"tile_nside_requested": tile or None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        g = Global2PCF(
            args.nside,
            mask=mask,
            nbins=args.nbins,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            tile_nside=tile or None,
            device=args.device,
            map_precision="float32",
            accumulation_precision="float64",
            resolution_factor=args.resolution_factor,
            cell_moments=args.cell_moments,
            pack_pairs=True,
            pack_host_pairs=True,
        )
        out["construct_s"] = time.perf_counter() - t
    out["warnings"] = [str(w.message) for w in caught]
    out.update(
        tile_nside=g.tile_nside,
        n_tiles=g.n_tiles,
        packed_block_rows_bound=g.packed_block_rows_bound,
        levels=np.asarray(g.level_table["nside"]).tolist(),
    )
    try:
        t = time.perf_counter()
        g.find_pairs()
        out["find_s"] = time.perf_counter() - t
    except ValueError as exc:
        out["status"] = "error"
        out["error"] = str(exc)
        return out
    sizes = [np.asarray(s) for e in g._engines for s in e.packed_block_sizes]
    out["packed_block_rows_max"] = int(max(int(s.max()) for s in sizes if s.size))
    t = time.perf_counter()
    g.prepare(release_host_pairs=True)
    sync()
    out["prepare_s"] = time.perf_counter() - t
    out["pairs"] = int(g.ntotpairs)
    shear, w, kappa = maps
    if args.device == "gpu":
        import cupy

        shear, w, kappa = cupy.asarray(shear), cupy.asarray(w), cupy.asarray(kappa)
    calls = {
        "gg": lambda: g.gg(shear, w, flip_g1=True),
        "kk": lambda: g.kk(kappa, w),
        "kg": lambda: g.kg(kappa, shear, w, w, flip_g1=True),
    }
    out["per_map_s"] = {}
    for name, fn in calls.items():
        fn()  # compile / warm the caches
        out["per_map_s"][name], _ = timed(fn, args.reps)
    out["device_pool_gb"] = pool_gb()
    out["status"] = "ok"
    del g, calls
    free_device()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=1024)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--tiles", type=int, nargs="+", default=[8, 16, 0])
    ap.add_argument("--resolution-factor", type=float, default=None)
    ap.add_argument("--cell-moments", default="quad")
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--theta-min", type=float, default=2.5)
    ap.add_argument("--theta-max", type=float, default=250.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = hp.read_map(args.mask, dtype=np.float64)
    mask = hp.ud_grade(m, args.nside) >= args.min_coverage
    n = int(mask.sum())
    rng = np.random.default_rng(0)
    maps = (
        (rng.normal(size=(args.nz, 2, n)) * 0.3).astype(np.float32),
        rng.uniform(0.5, 1.5, size=(args.nz, n)).astype(np.float32),
        rng.normal(size=(args.nz, n)).astype(np.float32),
    )
    report = {"args": vars(args), "n_active": n, "runs": []}
    for tile in args.tiles:
        r = run_tiling(args, mask, maps, tile)
        report["runs"].append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=1)


if __name__ == "__main__":
    main()
