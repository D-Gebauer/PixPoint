# Global estimators — first measurements (2026-09-25)

`bench_global.py` / `profile_g3.py` / `profile_step2.py`, seitz1 A100 80 GB PCIe
(idle), cupy 13.3, DES Y3 mask (`hp.ud_grade(mask) >= 0.5`), random maps,
4 source bins, float32 maps + float64 accumulation, `flip_g1=True`,
device-resident inputs, synchronised wall time, median.

## Global2PCF (ξ±, 20 log bins 15–250′, full resolution, `pack_pairs=True`)

| nside | footprint rows | unique pairs | tiles (nside) | find | prepare | `gg()` |
|---|---|---|---|---|---|---|
| 256 | 91 k | — | 8 | 5.6 s | 4.8 s | **2.5 ms** |
| 512 | 365 k | 6.72 × 10⁸ | 135 (8) | 42 s | 158 s | **33 ms** |

`prepare` at nside 512 is dominated by the host-side packing (`np.unique` of
the row block of every (tile, level)); a one-off, but the next thing to make
faster before nside 2048.

### Update 2026-10-21 (7.2.0): `prepare` / `load_pairs`

Profiled before changing anything (laptop, one core, GPU code path with an
identity upload, 1.8 × 10⁸ pairs at nside 512): 90 ns/pair, of which
`np.unique` 54 % (its argsort), `encode_angles` 32 % (`np.angle` alone
20 %), id → row translation 5 %. 7.2.0 replaces the sort by a marker table
(3.9 ns/pair, bitwise `np.unique`) and packs the tiles in parallel threads.
`bench_prepare.py` on seitz1 (A100, 160 threads; another user held ~12 cores),
DES Y3, 20 bins 15–250′, `pack_pairs=True`, both trees in one session:

| nside | k | pairs | find | prepare 7.1.0 → 7.2.0 | load exact 7.1.0 → 7.2.0 | packed prepare / load / file | `gg()` |
|---|---|---|---|---|---|---|---|
| 512 | — | 6.72 × 10⁸ | 26–29 s | 119.8 → 8.7 s | 117.2 → 9.5 s | 1.2 s / 3.3 s / 5.4 GB (exact 16.1) | 36 ms |
| 1024 | 2.9 | 6.98 × 10⁷ | 29 s | 9.9 → 1.1 s | 9.5 → 1.2 s | 0.33 s / 0.55 s / 0.61 GB | 21 ms |
| 2048 | 2.9 | 6.98 × 10⁷ | 31 s | 9.7 → 1.2 s | 9.8 → 1.4 s | 0.36 s / 0.67 s / 0.72 GB | 69 ms |

Device state and `gg` output hash identically for both trees and all paths.
The 158 s of September was measured before the sortless pair search; 7.1.0
measured 120 s in this session.  `find` is now the slowest step: at nside 1024
(laptop profile) 37 of 39 s are inside `unique_pairs`, i.e. the brute-force
candidate tests of the fine levels, not the pairs kept.

## Global3PCF (shear Γ₀…Γ₃, 12 log bins 15–250′, n_max = 20, all 64 ordered z-triplets)

| nside | ordered pairs | find | device memory | first call (+ normalisation) | `ggg()` |
|---|---|---|---|---|---|
| 256 | 8.46 × 10⁷ | 4.2 s | 2.5 GB | 1.7 s | **0.29 s** |
| 512 | 1.34 × 10⁹ | 129 s | 19.1 GB | 3.7 s | **1.94 s** |
| 512 (7.3.0, 2026-10-21) | 1.34 × 10⁹ | 41 s | | | **1.56 s** |

Split at nside 256: step 1 (CUDA kernel) 77 ms, step 2 (contraction) 210 ms.
Step 1 scales with the pairs (×16 per nside doubling), step 2 with the
centres (×4), so at nside 512 step 1 dominates (~1.2 s vs ~0.8 s).

* Step 2 was rewritten from one GEMM per centre bin (0.42 s total at nside
  256) to one batched GEMM per component, `L3ᵀ @ (L2 ⊗ centre)` (0.29 s): the
  complex128 GEMMs run at ~9 TFLOP/s, i.e. at the A100's FP64 rate — the
  remaining levers are fewer flops (symmetries of the components / z-triplets)
  or complex64 with `accumulation_precision="same"`.
* Step 1 runs at ~2.5 TFLOP/s (1.34 × 10⁹ pairs × 276 complex MACs per pair in
  1.2 s), a quarter of the FP64 rate: it is limited by the shared-memory
  reads of the staged neighbours, not by arithmetic.  The obvious next lever.
* The earlier back-of-envelope estimate for nside 512 (0.15–0.4 s per map)
  assumed 10–30 TFLOP/s sustained over both steps; it was ~5× too optimistic.
  10⁵ maps at nside 512 cost ~55 GPU-hours as measured.

Correctness on the same machine: `pytest -m gpu` (9 passed, including the
Global2PCF and Global3PCF GPU-vs-CPU gates at 1e-12 in float64).

## Update 2026-10-21 (7.3.0): Global3PCF steps 1 and 2

`step1_probe.py` / `step2_probe.py` on seitz1 (A100), nside 512, DES Y3,
`ggg`, 4 bins, n_max 20, bitwise / rel-diff checks against the shipped code:

| step 1 variant | time | |
|---|---|---|
| shipped (276 outputs) | 1.147 s | 2.6 TFLOP/s |
| only the read pair products (244) | 1.022 s | bitwise |
| + 4-FMA accumulation, 2-vector shared values, m-major thread map | **0.913 s** | ≤ 5e-15 |
| register tiling, 3–15 multipoles per thread | 0.94–1.91 s | |
| split staging / half power table / NB 16–64 / blocks 64–256 | 1.02–1.31 s | bitwise |
| *ablation: staging only* | *0.073 s* | |
| *ablation: accumulation only (shipped / FMA)* | *0.889 / 0.699 s* | |

Step 1 is the accumulation loop, bound by FP64 instructions, not by shared
memory or the staging (the earlier guess).  Even with FMAs it runs at under
half the FP64 instruction rate; the rest would need tensor cores (DMMA) or
warp-specialised staging, not tried.

| step 2 variant (per `ggg()`, on top of the reduced table) | time | |
|---|---|---|
| shipped | 1.900 s | |
| M: one matmul for gamma0..2 (shared leg-3 factor) | 1.819 s | bitwise |
| D: gamma3 from gamma2 by the leg swap | 1.715 s | 3e-15 |
| M + D | 1.684 s | 3e-15 |

All together: `ggg()` 2.011 → 1.558 s, `kgg()` 2.668 → 2.429 s (no twins
there), max rel diff 5.7e-15 against 7.2.1.

## Update 2026-11-04 (8.1.0): tiles sized for packed pairs

`bench_tiles.py` on usm-cl-seitz2 (A100 80 GB PCIe, el9 SLURM, 64 cores),
DES Y3 (`ud_grade >= 0.5`), 20 log bins 2.5–250′, k = 2.9 with quad moments,
4 source bins, random maps, float32 maps + float64 accumulation,
`pack_pairs` + `pack_host_pairs`, device-resident inputs, median of 5:

| nside 1024, k = 2.9 | tile_nside 8 (136 tiles) | tile_nside 16 (455 tiles) |
|---|---|---|
| construction (incl. the packed-block check) | 0.7 s | 0.7 s |
| `find_pairs` | 52.5 s | 137.7 s (2.6×) |
| `prepare` | 3.3 s | 3.5 s |
| `gg()` | 20.1 ms | 17.5 ms (−13 %) |
| `kk()` | 6.3 ms | 5.2 ms (−18 %) |
| `kg()` | 34.2 ms | 30.9 ms (−10 %) |
| device pool | 2.03 GB | 2.10 GB |
| largest packed row block | 19477 | 5830 |

1.45 × 10⁸ pairs in both (the tiling only regroups them). Finer tiles cost
one-off pair finding (every tile searches its own halo, and at tile_nside 16
the halo of 250′ is larger than the tile) and are 10–18 % faster per map
(smaller row blocks, more local gathers). The automatic choice keeps 8 here
(`tile_nside=None` gave the tile-8 numbers); it refines only when packing
needs it, which on this footprint happens first at nside 2048 (k = 2.9:
largest block 70363 rows at tile_nside 8, bound 72624; fits at 16).

Cost of the check itself (`packed_block_rows_bound` with the packing limit,
el9 CPU node, 32 cores, nside 2048): 1.0 s (tile 8, stops at the first
overflowing block) + 3.5 s (tile 16) with k = 2.9; 16 s for 8 → 64 at full
resolution, where the 250′ halo alone nearly fills a block. Without the
limit (every block refined, as the tests do): 12–28 s with k = 2.9.

The full-resolution nside-1024 run of the same job was killed by the host
memory limit (256 GB) during the pair search and is not reported.
