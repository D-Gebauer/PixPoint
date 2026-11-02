#!/usr/bin/env python
"""Headline metrics of T10 runs against the full-resolution reference:
max |ratio - 1| over the resolved entries, max |dzeta| / sigma_patch and the
quadrature sum over the 140 entries, for zeta_a,+ and zeta_a,-.

    python benchmarks/static_treecode/t10_summary.py k2.9_quad k2_quad k1.5_quad
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t10_zeta_level import zeta_per_patch  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(tags):
    fine = np.load(os.path.join(RES, "t10_fine.npz"))
    print(
        f"{'tag':12s} {'ms/map-set':>10s} | zeta_a,+: max|r-1|  max|d|/sig  quad | zeta_a,-: max|r-1|  max|d|/sig  quad"
    )
    for tag in tags:
        c = np.load(os.path.join(RES, f"t10_{tag}.npz"))
        line = f"{tag:12s} {float(c['ms_per_mapset']):10.1f} |"
        for stat in ("p", "m"):
            qf, _ = zeta_per_patch(fine["M_a"], fine[f"xi_{stat}"])
            qc, _ = zeta_per_patch(c["M_a"], c[f"xi_{stat}"])
            zf, zc = qf.mean(axis=2), qc.mean(axis=2)
            nmaps, _, npatch, _ = qf.shape
            sig_patch = (qf.std(axis=2, ddof=1) / np.sqrt(npatch)).mean(axis=0)
            sig_map = zf.std(axis=0, ddof=1)
            d = (zc - zf).mean(axis=0)
            mf = zf.mean(axis=0)
            ratio = zc.mean(axis=0) / mf
            resolved = np.abs(mf) > 3 * sig_map / np.sqrt(nmaps)
            x = np.abs(d) / sig_patch
            line += (
                f"  {np.nanmax(np.abs(ratio[resolved] - 1)):8.2f}  {x.max():10.2f}"
                f"  {np.sqrt((x**2).sum()):4.1f} |"
            )
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:] or ["k2.9_quad", "k2_quad", "k1.5_quad"])
