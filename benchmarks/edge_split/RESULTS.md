# Edge-splitting pair search (`edge_slop=0`): measured cost — 2026-11-02

Stage C5 of `RELEASE_8.0.0_GUIDE.md`. One A100 80 GB (`usm-cl-seitz1/2`), DES Y3 mask,
`k = 4` (`resolution_factor=4`), `cell_moments="quad"`, 4 tomographic bins, float32 maps and
rotations, float64 accumulation, device-resident maps, median of 7 CUDA-synchronised calls.
Script: `bench_edge_split.py`; raw numbers: `results/patch_runs_{512,1024}_2026-11-02.jsonl`
(patch estimator; the slurm logs `slurm_es*` hold the same lines) and `results/global_{512,1024}.json`.
`find_pairs` of the *first* configuration of each process includes the Numba JIT compile.

## `Integrated3PCF` — 9 bins 7′–123′, θ_Q = 90′, patches of Q90 (73 at nside 512, 69 at 1024)

| nside | search | packed | pairs | find (s) | prepare (s) | device MB | `gg` (ms) | `kg` (ms) | `three_by_two_pt` (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | static | no | 4.02e6 | 19.7 (JIT) | 0.14 | 133 | 1.27 | 1.82 | 2.70 |
| 512 | static | yes | 4.02e6 | 0.21 | 1.18 | 33 | 1.85 | 1.82 | 2.60 |
| 512 | split | no | 6.02e6 | 3.85 (JIT) | 0.15 | 221 | 2.68 | 2.73 | 5.21 |
| 512 | split | yes | 6.02e6 | 0.68 | 0.21 | 54 | 2.84 | 2.70 | 5.20 |
| 1024 | static | no | 2.01e7 | 3.42 (JIT) | 0.59 | 677 | 5.43 | 6.30 | 9.77 |
| 1024 | static | yes | 2.01e7 | 1.03 | 0.43 | 169 | 5.60 | 6.16 | 9.78 |
| 1024 | split | no | 5.29e7 | 6.02 | 1.13 | 2120 | 28.8 | 23.1 | 55.7 |
| 1024 | split | yes | 5.29e7 | 5.85 | 2.55 | 500 | 29.4 | 23.0 | 55.9 |

Level table (k = 4): nside 512 → `[512 ×7, 256 ×2]`, nside 1024 → `[1024 ×5, 512 ×2, 256 ×2]`.
`gg` is `gg(apertures=False)`, i.e. the pair pass alone.

## `Global2PCF` — 8 bins 15′–250′, whole footprint

| nside | tiles | search | packed | pairs | find (s) | prepare (s) | device GB | `gg` (ms) | `kg` (ms) | `three_by_two_pt` (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 8 / 136 | static | no | 5.15e7 | 8.4 | 1.31 | 1.78 | 17.0 | 17.3 | 35.1 |
| 512 | 8 / 136 | static | yes | 5.15e7 | 5.8 | 1.15 | 0.44 | 16.8 | 17.1 | 35.2 |
| 512 | 8 / 136 | split | no | 2.25e8 | 37.9 | 5.8 | 9.40 | 181 | 123 | 313 |
| 512 | 8 / 136 | split | yes | 2.25e8 | 34.4 | 12.2 | 2.19 | 190 | 124 | 323 |
| 1024 | 16 / 455 | static | no | 1.67e8 | 27.7 | 4.9 | 5.88 | 61.5 | 54.0 | 118 |
| 1024 | 16 / 455 | static | yes | 1.67e8 | 24.3 | 4.2 | 1.46 | 58.9 | 53.7 | 115 |
| 1024 | 16 / 455 | split | no | 1.81e9 | 277 | 81 | 78.3 | 1726 | 944 | 2737 |
| 1024 | 16 / 455 | split | yes | 1.81e9 | 274 | 163 | 18.1 | 1748 | 974 | 2781 |

Level table: nside 512 → `[512 ×4, 256 ×2, 128 ×2]`, nside 1024 → `[1024 ×2, 512 ×2, 256 ×2, 128 ×2]`.
`tile_nside=16` at 1024 because the default tiles exceed 65536 rows per packed (tile, bin) block
with the split (CLAUDE.md, "Known limit").

## What the numbers say

- **Pairs.** At k = 4 the exact binning costs 1.5× (patches, 512) to 2.6× (patches, 1024) and
  4.4× (footprint, 512) to 10.8× (footprint, 1024) the pairs of the static list — the ratio grows
  with the number of coarse levels under the bins, because every level seam adds its strips of
  finer pairs. The footprint at 1024 has 1.8e9 pairs: 78 GB unpacked (the A100 is 80 GB), 18 GB packed.
- **Per pair the split is slower than the static list**, not just more of it: 0.27 → 0.54 ns/pair
  on the patches at 1024, 0.37 → 0.95 ns/pair on the footprint. The split's bins mix pixels and
  cells of several levels and most of the extra pairs are fine pixel pairs in narrow strips next to
  the bin edges, i.e. many small row blocks and poorly coalesced gathers, and all of them run
  through the quad moment tiles. So the wall time is 2.6–5× (patches) and 10–28× (footprint) the
  static list: `three_by_two_pt` 9.8 → 56 ms per map on 69 patches at nside 1024, 115 → 2740 ms
  on the footprint. The notebook prototype needed 115 ms/map for the patch case; the library is 2×
  faster than that, and the static list stays 5.7× faster still.
- **Packing is memory, not speed**: 3.9–4.3× less device memory, within ±6 % in time (the
  packed split `prepare` is slower, 2.5 vs 1.1 s and 163 vs 81 s, because packing the pixel-row
  blocks of every level costs the `np.unique` passes).
- **The host search is the cost of the footprint split**: 277 s at nside 1024 (10× the static
  search of 28 s; 38 s vs 8 s at 512) — Numba, 16 cores. The patch search is cheap (6 s). Open
  question 2 of the guide (split on the device) is only worth it for footprint-wide runs, and only
  once the measurement itself is cheaper than the search amortised over the maps.
- **Verdict for production.** `edge_slop=0` is a *validation* tool at k = 4: it measures what the
  static list's binning window costs on a given data vector (ζ_a,+ in
  `benchmarks/moments/RESULTS.md`) at 3–5× the per-map price for patches. For 10⁵-map SBI runs the
  static list with quad cells remains the production estimator; if the exact binning is wanted
  there, the lever is a coarser k (the split's pair count is what makes k = 2 affordable), measured
  per data vector, not a blanket switch.

Not measured here: the sawtooth at the level seams in the noiseless mean (the notebook's
`plot_variant_mean`) — a science plot, to be made with the real map archive when the k = 2 + quad +
split T10 rerun (guide, open question 6) is done.
