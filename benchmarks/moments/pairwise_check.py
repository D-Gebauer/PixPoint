#!/usr/bin/env python
"""Per-pair check of the moment expansion against exact child sums (CPU, numpy twins).

For a random subset of coarse cell pairs of a static-treecode Integrated3PCF, compares the
mono / dipole / quad xi- contributions with the exact sum over the children (same bin), and
prints the rms relative residual per level.  Expected scaling: mono ~ 4 dphi (leakage, first
order), dipole ~ 8 dphi^2, quad ~ (4 dphi)^3 / 6 with dphi ~ 0.41 p / theta.
"""
import argparse
import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moment_probe import Nodes, exact_children_np, kappa_to_shear, make_corr, pair_arrays, pair_geometry, pair_sums_np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--npairs", type=int, default=400)
    ap.add_argument("--theta-min", type=float, default=15.0)
    ap.add_argument("--theta-max", type=float, default=120.0)
    ap.add_argument("--nbins", type=int, default=6)
    ap.add_argument("--Q", type=float, default=90.0)
    ap.add_argument("--mask", default=os.path.expanduser("~/Documents/research/software/PDFast/data/DESY3_Mask.fits"))
    args = ap.parse_args()
    args.device = "cpu"
    args.npatch = 1
    rng = np.random.default_rng(1)

    m = hp.read_map(args.mask)
    mask = hp.ud_grade(m.astype(np.float64), args.nside) != 0
    from CosmoFuse.utils import select_patch_centers

    phi_c, theta_c = select_patch_centers(mask, 8, patch_size=args.Q, theta_Q=args.Q)
    # a centre at high |dec| makes the frame terms visible
    dec = np.pi / 2 - theta_c
    i0 = int(np.argmax(np.abs(dec)))
    phi_c, theta_c = phi_c[i0 : i0 + 1], theta_c[i0 : i0 + 1]
    print(f"patch centre ra={np.degrees(phi_c[0]):.1f} dec={np.degrees(dec[i0]):.1f}")

    ell = np.arange(3 * args.nside)
    cl = 1e-9 * (np.maximum(ell, 1) / 100.0) ** -1.0 * np.exp(-((ell / (2.5 * args.nside)) ** 2))
    cl[:2] = 0
    kappa = hp.synfast(cl, args.nside, lmax=3 * args.nside - 1)
    g1, g2 = kappa_to_shear(kappa, args.nside)
    row_pix = np.flatnonzero(mask)
    gamma = g1[row_pix] + 1j * g2[row_pix]
    w = np.ones(row_pix.size)
    S = 1

    corr = make_corr(args, mask, phi_c, theta_c, args.k)
    corr.find_pairs()
    nodes = Nodes(corr)
    ni, nj, offsets, counts = pair_arrays(corr, nodes)
    phi_i, phi_j, inv_d = pair_geometry(np, nodes, ni, nj)
    print("levels per bin:", corr.level_nside.tolist(), " pairs per bin:", counts[0].tolist())
    mom_cur, W = nodes.moments(gamma, w, S, transport=False, order=0)
    mom_t, _ = nodes.moments(gamma, w, S, transport=True, order=2)
    pm = w * gamma

    for b in range(args.nbins):
        lo, hi = offsets[b], offsets[b + 1]
        if corr.level_nside[b] == corr.nside or hi == lo:
            continue
        sel = np.sort(rng.choice(np.arange(lo, hi), size=min(args.npairs, hi - lo), replace=False))
        one = np.arange(sel.size + 1)          # every selected pair is its own group
        ex = exact_children_np(ni[sel], nj[sel], one, nodes, pm, w, S)
        rows = {}
        for name, var, mom in (("mono", 0, mom_cur), ("mono_t", 0, mom_t), ("dipole", 1, mom_t), ("quad", 2, mom_t)):
            r = pair_sums_np(ni[sel], nj[sel], phi_i[sel], phi_j[sel], inv_d[sel], mom, W, S, var, one)
            rows[name] = r
        scale_m = np.sqrt(np.mean(ex[:, 1] ** 2))
        scale_p = np.sqrt(np.mean(ex[:, 0] ** 2))
        p = np.sqrt(hp.nside2pixarea(int(corr.level_nside[b])))
        theta = np.sqrt(corr.binedges[b] * corr.binedges[b + 1])
        dphi = 0.41 * p / theta
        print(f"\nbin {b}: nside {corr.level_nside[b]}, theta {np.degrees(theta)*60:.1f}', "
              f"p/theta = {p/theta:.3f}, dphi_est = {dphi:.3f}  (4dphi = {4*dphi:.3f}, 8dphi^2 = {8*dphi**2:.4f}, (4dphi)^3/6 = {(4*dphi)**3/6:.4f})")
        print(f"  {'variant':8s} {'xi- rms resid/rms':>18s} {'xi- mean resid/mean':>20s} {'xi+ rms resid/rms':>18s} {'w exact?':>9s}")
        for name, r in rows.items():
            dm = r[:, 1] - ex[:, 1]
            dp = r[:, 0] - ex[:, 0]
            print(f"  {name:8s} {np.sqrt(np.mean(dm**2))/scale_m:18.3e} {dm.mean()/ex[:,1].mean():20.3e} "
                  f"{np.sqrt(np.mean(dp**2))/scale_p:18.3e} {np.max(np.abs(r[:,2]-ex[:,2])):9.1e}")


if __name__ == "__main__":
    main()
