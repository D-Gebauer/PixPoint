#!/usr/bin/env python
"""Stage D4 (timing): cost of ``OnePoint`` on an A100.

DES Y3 mask at ``--nside``, scales 20', 45', 90' (default), the default
``centre_nside`` per scale, ``aperture_nside`` in (None, 512, 256) unless
given: the centre selection + geometry build (``prepare``), the device memory
of the aperture geometry, and the per-map cost of ``g`` (with and without
``M_x``), ``k`` (top hat) and ``k`` (HEALPix pixel) with device-resident
float32 maps (median of several CUDA-synchronised calls), per scale.

Synthetic Gaussian maps: this measures time, not the window on the PDF (that
needs the map archive; see the guide's D4).

Usage (el9, A100):
    benchmarks/static_treecode/remote_run.sh \\
        "python benchmarks/onepoint/bench_onepoint.py --nside 1024 \\
         --out benchmarks/onepoint/results/bench_1024.json"
"""

import argparse
import json
import os
import time

import healpy as hp
import numpy as np

P = "/project/ocean1-seitz/users/dgebauer"
DEFAULT_MASK = f"{P}/research/CosmoFuse-treecode-data/DESY3_Mask.fits"


def timed(fn, sync, n=5):
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


def geometry_bytes(op):
    """Device bytes of the aperture geometry every engine holds."""
    import cupy as cp

    total = 0
    for per_scale in op._engines.values():
        for parts in per_scale.values():
            for eng in parts:
                ctx = eng.compute_context
                for name in (
                    "Q_inds_dev",
                    "Q_cos_dev",
                    "Q_sin_dev",
                    "Q_val_dev",
                    "Q_offsets_dev",
                    "Q_patch_area_dev",
                ):
                    a = getattr(ctx, name, None)
                    if isinstance(a, cp.ndarray):
                        total += int(a.nbytes)
                # {id(values): (values, device tuple)}: the scalar-aperture
                # geometries (top hat, filters) this engine uploaded
                for _values, dev in (getattr(ctx, "Q_val_k_dev", None) or {}).values():
                    for a in dev:
                        if isinstance(a, cp.ndarray):
                            total += int(a.nbytes)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=1024)
    ap.add_argument("--scales", type=float, nargs="+", default=[20.0, 45.0, 90.0])
    ap.add_argument(
        "--aperture-nside", type=str, nargs="+", default=["none", "512", "256"]
    )
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from CosmoFuse import OnePoint

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
    row_pix = np.flatnonzero(mask)
    rng = np.random.default_rng(1)
    nz = args.nz
    shear = xp.asarray(rng.normal(0, 0.03, (nz, 2, row_pix.size)).astype(np.float32))
    w = xp.asarray(rng.uniform(0.5, 1.5, (nz, row_pix.size)).astype(np.float32))
    kappa = xp.asarray(rng.normal(0, 0.02, (nz, row_pix.size)).astype(np.float32))
    edges = np.linspace(-0.1, 0.1, 101)

    report = {"args": vars(args), "n_active": int(row_pix.size), "runs": []}
    for ap_s in args.aperture_nside:
        ap_nside = None if ap_s.lower() == "none" else int(ap_s)
        for kind in ("g", "k_tophat", "k_healpix"):
            kw = dict(
                device=args.device,
                map_precision="float32",
                rotation_precision="float32",
                aperture_nside=ap_nside,
                min_coverage=args.min_coverage,
                edges=edges,
            )
            if kind == "k_healpix":
                kw["k_aperture_filter"] = "healpix"
            try:
                for theta in args.scales:
                    op = OnePoint(args.nside, mask, scales=(theta,), **kw)
                    t0 = time.perf_counter()
                    op.prepare(("g",) if kind == "g" else ("k",))
                    sync()
                    t_prepare = time.perf_counter() - t0
                    stat = "g" if kind == "g" else "k"
                    c = op.centres(stat, 0)
                    run = {
                        "kind": kind,
                        "aperture_nside": ap_nside,
                        "scale_arcmin": theta,
                        "centre_nside": int(c["centre_nside"]),
                        "n_candidates": int(c["n_candidates"]),
                        "n_apertures": int(c["pix"].size),
                        "disc_pixels": int(c["disc_size"].sum()),
                        "prepare_s": t_prepare,
                        "device_bytes": geometry_bytes(op) if gpu else None,
                    }
                    if kind == "g":
                        run["g_ms"] = timed(lambda: op.g(shear, w, flip_g1=True), sync)
                        run["g_no_cross_ms"] = timed(
                            lambda: op.g(shear, w, flip_g1=True, cross=False), sync
                        )
                        run["g_values_ms"] = timed(
                            lambda: op.g(shear, w, flip_g1=True, values=True), sync
                        )
                    else:
                        run["k_ms"] = timed(lambda: op.k(kappa, w), sync)
                    if gpu:
                        run["device_bytes"] = geometry_bytes(op)
                    print(json.dumps(run), flush=True)
                    report["runs"].append(run)
                    if gpu:
                        cp.get_default_memory_pool().free_all_blocks()
            except (ValueError, MemoryError) as exc:
                run = {
                    "kind": kind,
                    "aperture_nside": ap_nside,
                    "error": str(exc)[:300],
                }
                print(json.dumps(run), flush=True)
                report["runs"].append(run)
                if gpu:
                    cp.get_default_memory_pool().free_all_blocks()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
