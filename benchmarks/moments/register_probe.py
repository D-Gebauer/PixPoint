#!/usr/bin/env python
"""B5: register and local-memory use of the moment pair tiles (needs cupy).

Compiles the moment-corrected kernels of the three pair statistics for
float32 maps / float64 accumulation and prints ptxas' register count and
local (spill) bytes per instantiation: a non-zero ``local`` is the register
spill that made the quad tile 3.4x slower than mono at nz = 4.

    benchmarks/static_treecode/remote_run.sh "python benchmarks/moments/register_probe.py"
"""

import cupy as cp

from CosmoFuse.backend.raw import KernelFile

T, ACC, QT, I = "float", "double", "float", "int"
CASES = [
    # (file, kernel, template args)
    ("tomo_vectorized_xipm.cu", "gpu_tiled_tomo_reduce_xipm_mom", (T, QT, "{ns}", "{mom}", I, ACC)),
    ("tomo_vectorized_xipm.cu", "gpu_tiled_packed_reduce_xipm_mom", (T, "{ns}", "{mom}", ACC)),
    ("density_shear_tomo_vectorized.cu", "gpu_tiled_tomo_reduce_ds_mom", (T, QT, "{nd}", "{ns}", "{mom}", I, ACC)),
    ("density_shear_tomo_vectorized.cu", "gpu_tiled_packed_reduce_ds_mom", (T, "{nd}", "{ns}", "{mom}", ACC)),
    ("tomo_tiled_3x2pt.cu", "gpu_tiled_tomo_reduce_3x2pt_mom", (T, QT, "{nd}", "{ns}", "0", "{mom}", I, ACC)),
    ("tomo_tiled_3x2pt.cu", "gpu_tiled_packed_reduce_3x2pt_mom", (T, "{nd}", "{ns}", "0", "{mom}", ACC)),
]


def main():
    files = {}
    print(f"{'kernel':42s} {'nz':>2s} {'mom':>3s} {'regs':>5s} {'local B':>8s}")
    for filename, name, targs in CASES:
        kf = files.setdefault(filename, KernelFile(cp, filename, "probe"))
        for nz in (1, 4):
            for mom in (1, 2):
                args = tuple(a.format(ns=nz, nd=nz, mom=mom) for a in targs)
                k = kf(name, args)
                if k is None:
                    print(f"{name:42s} {nz:2d} {mom:3d}  compile failed")
                    continue
                attrs = k.raw.attributes
                print(
                    f"{name:42s} {nz:2d} {mom:3d} {k.raw.num_regs:5d} "
                    f"{k.raw.local_size_bytes:8d}   maxthreads {attrs['max_threads_per_block']}"
                )


if __name__ == "__main__":
    main()
