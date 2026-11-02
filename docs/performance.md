# Running on GPUs (and CPUs)

You do not need to know anything about GPU programming to use PixPoint on a
GPU. This page explains what changes, in plain terms, and the handful of
choices that matter for large runs.

## CPU or GPU?

PixPoint runs the same measurements on both. On the CPU the inner loops are
compiled by Numba and use all cores; on an NVIDIA GPU they run as CUDA kernels
through CuPy. `device="auto"` picks the GPU when CuPy is installed and falls
back to the CPU otherwise; `device="cpu"` and `device="gpu"` force one.

What a GPU buys you is throughput on the *repeated* part. Finding the pairs
runs on the CPU either way (it is a search, not arithmetic); measuring a map
is one pass over hundreds of millions of pairs, and that pass is 10 to 100
times faster on a GPU. Some measured numbers (A100, DES Y3 footprint, 4
tomographic bins, float32 maps, float64 sums):

| measurement | pairs | per map on an A100 |
|---|---|---|
| `Global2PCF.gg`, `nside=512`, 20 bins to 250′ | $6.7\times10^8$ | 33 ms |
| `Integrated3PCF.gg`, 992 patches, `nside=512`, 6 bins | $1.1\times10^8$ | 13 ms |
| `Global3PCF.ggg`, `nside=512`, 12 bins, `n_max=20` | $1.3\times10^9$ ordered | 1.6 s |
| `OnePoint.g`, `nside=1024`, one scale | $2\times10^8$ disc pixels | 12 ms |

The same `Integrated3PCF` measurement with TreeCorr on the pixel catalogues
took 58 s on 48 CPU cores. On a laptop CPU, PixPoint's own pair pass at
`nside=512` is a matter of seconds per map, so for a handful of maps the CPU is
fine; a GPU is for suites.

## Memory: the one thing to plan

The pairs live in memory for the whole session. Unpacked they cost 24 bytes
each (in RAM, in the pair file and on the GPU); the stored geometry of the
example above is 16 GB, which fits an 80 GB card easily and a 24 GB desktop
GPU with room for the maps. Three knobs shrink it:

- `pack_pairs=True` stores the pair geometry on the GPU in 8 bytes per pair
  (three times more pairs per card). The rotation angles are then quantised
  to 16 bits, which moves every estimate by about 1e-5 of its statistical
  error and does not bias it; the host arrays and the pair file stay exact.
- `pack_host_pairs=True` packs in RAM and in the file as well (the file is
  then no longer exact; keep an exact copy if you may want to switch back).
- `resolution_factor` (the [treecode](treecode.md)) reduces the number of
  pairs at large separations by orders of magnitude; it changes the
  estimator in a documented way.

Before searching, every estimator estimates the pair memory from the footprint
and the binning and refuses to start if it would not fit the free memory of
the device (`memory_budget_gb` overrides the budget; `float("inf")` disables
the check). `release_host_pairs=True` on `preprocess()` / `load_pairs()` frees
the RAM copy once the GPU holds the pairs.

## Precision

Use `map_precision="float32"` with `accumulation_precision="float64"` on a GPU:
maps and weights are stored and read in single precision, every sum over pairs
is kept in double. Results differ from an all-double run at the 1e-7 level and
show no accumulation error. `map_precision="float64"` is for comparisons
against a double-precision reference. The pair search itself is always double
precision.

## Keeping results and maps on the device

By default a result's arrays are left on the GPU (`return_device=True`); call
`.to_numpy()` or pass `return_device=False` to get NumPy arrays. Leaving them
on the device matters when the result feeds the next step on the GPU (a
writer that reduces before copying, a likelihood evaluated on the device).

Uploading a map-set to the GPU costs about as much as measuring it. For a
suite of maps, two helpers hide the upload of the next map behind the
measurement of the current one:

```python
from pixpoint import MapFileLoader
import numpy as np

def read(path, out):                       # runs in a background thread
    out["shear"][...] = np.load(path, mmap_mode="r")

loader = MapFileLoader(g2, {"shear": (nz, 2, g2.n_active)}, sources=paths,
                       read_fn=read, n_slots=8, n_readers=4,
                       row_pix_hash=g2.row_pix_hash)
for k, dev in loader:                      # dev["shear"] is already on the GPU
    results.append(g2.gg(dev["shear"], w, flip_g1=True))
```

Store the archive in row space (`g2.to_row_space(maps)`, once), so nothing has
to be gathered per map, and pass fixed weight maps as a read-only array
(`w.flags.writeable = False`) so they are uploaded once and cached.
{class}`~pixpoint.MapLoader` does the same for maps that are already in memory.
Both are no-ops on the CPU, so the same script runs everywhere.

The first measurement on a GPU compiles the kernels (a few seconds); make one
throwaway call before timing a loop. On the CPU, `warmup()` triggers the
one-off Numba compilation.

## Several GPUs

Pass a list of GPU ids and the estimator splits its work across them: the
patches of `Integrated3PCF`, the tiles of `Global2PCF`, the centres of
`Global3PCF` and `OnePoint`. Each GPU holds its share of the pairs and a copy
of the maps, so a geometry too large for one card is measured on two, and the
result equals the single-GPU result to the order of the final addition (1e-13
relative; bitwise for the patch estimator). Multi-GPU calls take host arrays
and return host arrays.

```python
g2 = Global2PCF(nside, mask=mask, nbins=20, theta_min=2.5, theta_max=250, device=[0, 1])
g2.load_pairs("pairs.h5")                 # each GPU loads its own slice
res = g2.gg(shear, w, flip_g1=True)
```

Pair files are written from a single-device estimator (or a CPU one) and read
by a group.

## Under the hood, briefly

For those who want to know what runs: the pixel pairs are stored per
(patch or tile, angular bin) with their rotation angles; one GPU kernel per
statistic visits every pair once and accumulates all tomographic combinations
in registers (which is what makes the pass memory-bound and fast); the
aperture statistics run one block of threads per patch; the treecode's merged
cells are built on the GPU per map. Every kernel has a NumPy twin that the
test suite runs on CPU-only machines, so the GPU and CPU paths are checked
against each other and against brute-force references at machine precision.
Details for developers are in the repository's `CLAUDE.md` and the
`benchmarks/` notes.
