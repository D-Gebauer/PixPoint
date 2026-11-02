# Moment-carrying treecode cells: accuracy and speed probe (2026-11-02)

`moment_probe.py` (CUDA prototype + numpy twins), `pairwise_check.py` (per-pair formula check).
Raw numbers: `results/probe_1024_v2.json` (noise-free), `results/probe_1024_noise.json`
(DES-like shape noise), per-patch arrays in the matching `.npz` files. Run on usm-cl-seitz2
(A100 80 GB) via `benchmarks/static_treecode/remote_run.sh`.

## Question

The static treecode's ξ− window comes from rotating every child pair with its parents' position
angle: per child pair the phase is off by e^{-4iδφ}, which suppresses ξ− by ⟨cos 4δφ⟩ and leaks
the parity-odd pair product (noise, no signal) with weight sin 4δφ. Can the cells carry the
within-cell gradient information so that a coarse k keeps ξ−'s S/N?

## Method

Expand e^{-4iδφ} in the children's offsets from the cell centroids (gnomonic east/north
coordinates in the cell frame, child shears parallel-transported into that frame — exact,
geometry-only). The cell pair becomes a contraction of per-cell moments of w·γ:

| variant | moments per cell | correction |
|---|---|---|
| mono (= shipped treecode) | M = Σ wγ | none |
| dipole | + D = Σ wγ δr (2 complex) | −4iS δφ₁, δφ₁ = (p_J·δr_j − p_I·δr_i)/d |
| quad | + Q = Σ wγ δr δrᵀ (3 complex) | −8 δφ₁² and the second-order angle term +4iS (a·b)/d² |

p, q are the perpendicular/parallel unit vectors of the pair; the J-end ones are the
parallel-transported ones (p_J = (sin φ_J, −cos φ_J)), which needs the **full** position angle
at both ends (e^{2iφ} alone leaves the relative sign of the two dipole terms ambiguous) and 1/d
per pair. Only ξ− changes: for ξ+ the two ends' phase errors cancel, so the monopole is already
exact up to the frame transport (1e-5 at nside 1024).

Setup: nside 1024, DES Y3 mask, the first 200 production Q90 patches (R = θ_Q = 90′), 9 log bins
7′–123′, CosmoGrid fiducial convergence (bin 2) → KS shear, uniform weights, float32 products +
float64 accumulation. Levels (k = 2): 1024 ×3, 512 ×2, 256 ×2, 128 ×2; (k = 4): 1024 ×5, 512 ×2,
256 ×2; (k = 8): 1024 ×7, 512 ×2.

Two references on the same pair lists: **fine** = full-resolution pixel pairs (3.6 × 10⁸ pairs);
**target** = exact child sums with the *coarse* bin assignment (CUDA kernel over the cell → pixel
CSR, double precision). target − fine is the binning error (common to all variants); variant −
target is the phase error the moments address.

Validation: the probe's mono variant reproduces CosmoFuse's `gg` to ≤ 3.4 × 10⁻⁶ σ_patch at full
resolution and at every k (the production convention is S = +1 on the unflipped shear ≡ CosmoFuse
`flip_g1=True`); float32 vs float64 products ≤ 3.4 × 10⁻⁶ σ_patch; the per-pair residuals scale
as 4δφ : 8δφ² : (4δφ)³/6 (`pairwise_check.py`, nside 512, k = 4: 18 % → 5.5 % → 1.2 % per pair).

## Accuracy, noise-free (relative error of the patch mean, ξ−)

| k | bin θ | level | binning (target−fine) | mono | dipole | quad |
|---|---|---|---|---|---|---|
| 2 | 21′ | 512 | +0.26 % | −10.1 % | +8.9 % | +1.29 % |
| 2 | 29′ | 512 | +1.1 % | −5.4 % | +5.1 % | +0.40 % |
| 2 | 40′ | 256 | −2.8 % | −13.0 % | +11.0 % | +2.35 % |
| 2 | 56′ | 256 | +2.5 % | −7.2 % | +6.5 % | +0.76 % |
| 2 | 76′ | 128 | −2.3 % | −11.4 % | +12.1 % | +2.87 % |
| 2 | 105′ | 128 | +1.9 % | −6.9 % | +7.5 % | +1.00 % |
| 4 | 40′ | 512 | +1.9 % | −2.98 % | +2.81 % | +0.12 % |
| 4 | 56′ | 512 | −0.6 % | −1.55 % | +1.50 % | +0.034 % |
| 4 | 76′ | 256 | −1.6 % | −4.27 % | +3.79 % | +0.26 % |
| 4 | 105′ | 256 | +1.8 % | −2.62 % | +2.03 % | +0.083 % |
| 8 | 76′ | 512 | −0.7 % | −0.91 % | +0.80 % | +0.010 % |
| 8 | 105′ | 512 | +1.3 % | −0.51 % | +0.44 % | +0.003 % |

The dipole alone flips the sign of the mean error (its per-pair coefficient is real,
1 + 8δφ², as predicted) — it is an information fix, not a bias fix. Quad removes the phase
error by 20–50× at k = 4 and 5–10× at k = 2. After quad, the **binning error dominates**
(±2 %, alternating sign at the level seams): that is the part a `bin_slop = 0`-style edge
split would address, not moments. ξ+ phase errors are ≤ 1e-5 for every variant with the frame
transport (≤ 3.5e-5 without it = shipped treecode).

## Information retained, DES-like shape noise (σ = 0.064 per pixel and component)

Per-patch correlation coefficient of ξ− with the exact (target) value across the 200 patches;
its square is the fraction of Fisher information retained. (Noise-free, every variant is
≥ 0.999 because the leaked parity-odd term cancels within a cell for a smooth field; with
pixel noise it does not — hence this is the relevant regime.)

| k | bin θ | level | mono | dipole | quad |
|---|---|---|---|---|---|
| 2 | 21′ | 512 | 0.922 | 0.9952 | 0.9993 |
| 2 | 29′ | 512 | 0.958 | 0.9984 | 0.9999 |
| 2 | 40′ | 256 | 0.877 | 0.9860 | 0.9965 |
| 2 | 56′ | 256 | 0.928 | 0.9961 | 0.9994 |
| 2 | 76′ | 128 | 0.863 | 0.9837 | 0.9941 |
| 2 | 105′ | 128 | 0.933 | 0.9958 | 0.9993 |
| 4 | 40′ | 512 | 0.977 | 0.9994 | 1.0000 |
| 4 | 56′ | 512 | 0.987 | 0.9998 | 1.0000 |
| 4 | 76′ | 256 | 0.966 | 0.9989 | 0.9999 |
| 4 | 105′ | 256 | 0.983 | 0.9998 | 1.0000 |
| 8 | 76′ | 512 | 0.994 | 1.0000 | 1.0000 |
| 8 | 105′ | 512 | 0.997 | 1.0000 | 1.0000 |

mono's correlation equals 1 − (suppression) to within 0.5 %, confirming that for shape-noise
dominated pixels the ξ− S/N loss is the suppression itself (3–4 % at k = 4, 7–14 % at k = 2).
Dipole cuts the loss by ~20× (to 0.05–0.2 % at k = 4), quad by another ~5×. **Quad at k = 2
retains more ξ− information (0.994–0.9999) than mono at k = 4 (0.966–0.987), and dipole at
k = 4 (0.9989–0.9998) more than mono at k = 8 (0.994–0.997).**

## Speed (A100, probe kernels, one map, median of 7)

Per-pair cost, flat across k (all pairs run through the moment kernel in the prototype):

| | nz = 1 | nz = 4 |
|---|---|---|
| mono | 13 ps | 62 ps |
| dipole | 22 ps (1.7×) | 100 ps (1.6×) |
| quad | 28 ps (2.2×) | 182 ps (2.9×) |

Pair counts: full res 3.60 × 10⁸; k = 8 1.69 × 10⁸ (coarse bins 1.41 × 10⁷); k = 4 5.90 × 10⁷
(coarse 8.4 × 10⁶); k = 2 1.74 × 10⁷ (coarse 3.1 × 10⁶). Kernel wall (nz = 1): full-res mono
4.73 ms; k = 8 mono 2.15 ms; k = 4 mono 0.73 / dipole 1.29 / quad 1.60 ms; k = 2 mono 0.24 /
dipole 0.42 / quad 0.53 ms. CosmoFuse `gg` (pairs only) for scale: full 10.2 ms, k = 8 6.6 ms,
k = 4 4.1 ms (packed 3.8), k = 2 4.1 ms — the CosmoFuse call is host/expansion-bound at this
size, so only the kernel numbers compare the variants.

In a library implementation only the **coarse-bin** pairs need moments (fine-level rows have
D = Q = 0). Projected from the per-pair costs: k = 4 quad ≈ 0.86 ms (+17 % over mono) at nz = 1,
≈ 4.8 ms (+26 %) at nz = 4; k = 2 quad ≈ 0.29 ms. Versus the brute-force alternative, mono at
k = 8 (2.15 ms, 2.9× the pairs of k = 4, 9.7× those of k = 2), quad at k = 2 is ~7× cheaper and
more accurate.

## What a library version needs

- Moments as extra virtual rows produced by the existing degrade chain: the CSR gets geometry
  weights (δx, δy, δx², δxδy, δy², transport phase) instead of ones — 6 complex values per cell
  row (dipole: 3). `_expand_rows` layouts generalise (AoS per row is what the kernel wants).
- Pair payload: the **full** angle φ at both ends (the packed uint16 can encode φ instead of 2φ
  at the same cost; the unpacked complex e^{2iφ} is *not* enough) plus 1/d per pair (uint8/16 in
  the bin's range suffices, the correction needs it to ~1 %).
- Kernel: the correction only in `tile_xipm` for coarse-level pairs; 9 (dipole) or 15 (quad)
  extra complex FMAs per pair and 3×/6× the per-end map reads. Register pressure at nz = 4 with
  quad is the thing to watch (the prototype's nz = 4 quad kernel is 2.9× mono, likely spilling).
- `three_by_two_pt`'s fused tile gets the same term for ξ−; γ_t has the analogous spin-2 version
  (coefficients 2 instead of 4) if wanted; ξ+, ξ_g need nothing.
- Binning (±2 % at k = 4 after quad) becomes the leading treecode error. An edge-splitting
  pair search (`bin_slop = 0` behaviour) removes it but was measured at 5× (patches) to 25×
  (footprint) the per-map cost at k = 4 and taken out of the library again (8.0.0 guide,
  stage C); a smooth data vector means full resolution.

## Follow-up: Global2PCF prototype benchmark at nside 1024 with edge splitting (2026-11-02; prototype only, the library's split was removed)

The same variants on CosmoFuse's footprint-wide `Global2PCF` pair lists (DES Y3 multiprobe mocks,
20 maps, 8 bins 15′–250′, source bin 4 with weights, TreeCorr brute as the baseline), plus a
**top-down edge-splitting pair search** (`edge_slop = 0`: a cell pair whose separation range crosses a
bin edge is split until it fits, down to exactly binned pixel pairs; cells coarser than the bin's static
level are split too). Code and runs live in the OptimSBI project on el9
(`src/optimsbi/moment_treecode.py`, `scripts/bench_global2pcf_moments.py`, section "Moment-carrying
cells and edge splitting" of `notebooks/02_global2pcf_treecorr.ipynb`). Headlines:

- Σ w_i w_j per bin equals brute to 2e-10 with the split (static k = 4: up to 2.7 %).
- Quad + split at k = 4: per noisy map 0.001–0.004 σ from brute in ξ− (ξ+ ≤ 7e-5 σ), mean ξ− ≤ 0.04 %;
  1.82 × 10⁹ pairs (11× the static k = 4 list, 17 % of full resolution), 51.5 GB, 115 ms/map vs 3.3 s
  for the chunked full resolution. k = 8 + split has *more* pairs (2.13 × 10⁹) and is not more accurate.
- Without the split, the per-map difference to brute on noisy maps is set by the exchange of pairs
  across bin edges (0.3–0.8 σ for every cell variant, ξ+ alike): a window, not an information loss, but
  one the moments cannot touch. Noiseless, quad cuts the per-map ξ− error 10× (k = 4: 0.6–2.3 σ → 0.1–0.23 σ).
- Mean ξ− error, noiseless: k = 8 plain −0.4…−1.0 % → quad ≤ 0.24 %; k = 4 plain −1.9…−4.9 % → quad
  ≤ 0.3 %; k = 2 plain −8…−19 % → quad ≤ 5 %.

## Library implementation (8.0.0, 2026-11-02)

`cell_moments="quad"` (default) / `"dipole"` / `None` on `Integrated3PCF` and `Global2PCF`:
`src/CosmoFuse/moments.py` (derivation + numpy reference), `cuda/moment_rows.cu` (per-level
moment builder straight from the pixel rows), the `*_mom` tiles of `cuda/pair_tiles.cuh`, and
the aperture kernels of `cuda/aperture_moments.cuh`.  Gates: `tests/test_cell_moments.py`.

### Aperture probe (numpy, nside 128, DES-like mask region, 2 patches)

Reference: exact sums over the children of every disc cell with the coarse disc membership
and the coarse centre pixel (the window the coarse estimator defines), so only the within-cell
expansion is measured.  Per-cell rms residual relative to the rms exact cell term, and the
error of the patch sum, for `M_a` (Q_crittenden) and `M_g` with `U_crittenden`:

| field | aperture nside | θ_Q | p/θ_Q | M_a per cell: mono → dipole → quad | M_a patch sum | M_g per cell | M_g patch sum |
|---|---|---|---|---|---|---|---|
| smooth | 64 | 300′ | 0.18 | 3.8 % → 0.37 % → 0.025 % | −4.2 % → −1.5 % → +0.012 % | 3.5 % → 0.37 % → 0.009 % | −0.8 % → −0.09 % → +0.004 % |
| smooth | 32 | 600′ | 0.18 | 5.3 % → 0.52 % → 0.12 % | −1.6 % → +0.32 % → +0.025 % | 5.2 % → 0.49 % → 0.021 % | −0.6 % → −0.6 % → −0.006 % |
| smooth | 64 | 150′ | 0.37 | 7.5 % → 1.5 % → 0.086 % | −5.0 % → +0.95 % → +0.066 % | 7.8 % → 1.6 % → 0.078 % | −24 % → +2.3 % → +0.08 % |
| white noise | 64 | 300′ | 0.18 | 8.3 % → 0.46 % → 0.055 % | −5.3 % → −1.4 % → +0.07 % | 9.0 % → 0.49 % → 0.023 % | −4.9 % → −1.7 % → +0.04 % |

(one patch each; the second patch of every row behaves the same.)  The dipole removes the
scatter (per-cell residual ÷ 10) and the quad the mean (patch sum ÷ 100–1000), as the ξ−
measurements above and the estimate table of the guide predicted.  No weight moments are
needed: CosmoFuse normalises every aperture by Σw without the filter, so only the numerator
expands; the patch-mean `M_g` (constant filter) needs nothing.

### Cost in the library (A100, DES Y3 mask, 450 patches of Q90, 7′–123′ in 9 bins, k = 4, float32 maps, device-resident input, median of 7)

nside 512: levels `[512 ×7, 256, 256]`, 4.0 × 10⁶ pairs — the calls are **host-bound** (1–3 ms
with ~0.5 ms jitter), so these numbers bound the overhead rather than measure the kernels;
see the nside-1024 run below for the kernel-dominated regime.

| nz | pack | cell_moments | gg (pairs) | kg | three_by_two_pt |
|---|---|---|---|---|---|
| 1 | no | None / dipole / quad | 0.86 / 1.76 / 2.07 ms | 0.91 / 1.35 / 2.26 | 1.88 / 1.49 / 2.41 |
| 1 | yes | None / dipole / quad | 2.08 / 1.33 / 1.87 | 2.84 / 1.57 / 2.42 | 1.56 / 1.85 / 1.86 |
| 4 | no | None / dipole / quad | 1.08 / 1.42 / 2.22 | 2.23 / 2.27 / 3.05 | 2.67 / 2.91 / 2.60 |
| 4 | yes | None / dipole / quad | 1.67 / 2.27 / 2.33 | 1.97 / 2.58 / 2.96 | 1.97 / 2.62 / 2.99 |

Aperture pass (`m_a`; `m_g` with `U_crittenden`), nz = 4: ap 128: 0.53 / 1.44 / 1.68 ms
(mono / dipole / quad), ap 256: 0.47 / 0.59 / 0.66 ms; the moment build of the aperture level
(one `gpu_moment_level` launch over the footprint) is included.

nside 1024, 200 patches of Q90, 7′–123′ in 9 bins (`results/bench_1024.json`), the
kernel-dominated regime.  k = 4: levels `[1024 ×5, 512, 512, 256, 256]`, 2.0 × 10⁷ pairs;
k = 2: `[1024 ×3, 512, 512, 256, 256, 128, 128]`, 5.9 × 10⁶ pairs.  Whole call, ms:

| nz | k | pack | statistic | None | dipole | quad |
|---|---|---|---|---|---|---|
| 1 | 4 | no | gg (pairs) / kg / 3x2pt | 1.19 / 1.94 / 2.26 | 1.60 / 2.21 / 1.99 | 1.45 / 2.65 / 2.07 |
| 1 | 4 | yes | gg / kg / 3x2pt | 1.35 / 1.28 / 2.33 | 1.28 / 1.72 / 1.98 | 1.35 / 1.91 / 2.06 |
| 1 | 2 | yes | gg / kg / 3x2pt | 0.76 / 1.01 / 1.20 | 1.00 / 1.43 / 1.64 | 1.11 / 1.65 / 1.85 |
| 4 | 4 | no | gg / kg / 3x2pt | 1.65 / 2.78 / 3.78 | 2.40 / 3.66 / 5.07 | **5.35 / 5.61 / 8.93** |
| 4 | 4 | yes | gg / kg / 3x2pt | 1.78 / 2.80 / 3.86 | 2.41 / 3.77 / 5.11 | **4.73 / 5.59 / 9.32** |
| 4 | 2 | yes | gg / kg / 3x2pt | 0.93 / 1.51 / 2.00 | 1.34 / 2.13 / 2.81 | 2.21 / 2.78 / 4.28 |

Aperture pass, nz = 4 (`m_a` / `m_g` with U_crittenden): ap 256: 0.45 / 0.38 ms (mono),
1.17 / 0.73 (dipole), 1.14 / 1.07 (quad); ap 128: 0.42 / 0.38, 0.92 / 0.68, 0.96 / 0.72 —
the 2–3× of the estimate, the moment build of the aperture level included.

**Reading.** At nz = 1 the moments cost nothing measurable (host-bound calls; the dipole
and quad tiles over the ~15 % coarse pairs add < 0.3 ms).  At **nz = 4 the quad tile is
3.2× the plain call** (gg 1.65 → 5.35 ms) and the dipole 1.45× — far above the projected
+26 %, and far above what the extra bytes explain.  The cause is the one the prototype
already flagged: `tile_xipm_mom` / `tile_multi_mom` keep the contracted moments of every
tomographic bin of both ends in registers (`EndMoments<T, 2>[NS]` × 2 = 80 values at NS = 4,
on top of 30 double accumulators), which spills to local memory at NS = 4 (the prototype's
nz = 4 quad kernel showed the same 2.9×).  **Open B5 item:** restructure the quad tile so the
per-end contractions are recomputed per combination from the raw moments (L1-resident,
12 T per (cell, bin)) instead of being cached per bin, or run the moment pass per source bin;
then confirm with `-Xptxas -v` that NS = 4 no longer spills.  Until then the quad default
costs ~3× the plain treecode call at nz = 4 in the pair pass (the data vector is unchanged by
the fix: it is a register-allocation matter).

## T10 with moment-carrying cells (A100, 2026-11-02)

`t10_zeta_level.py --measure --k K` now takes `--cell-moments quad|dipole|none` (default
quad; tags `kK_quad`, old plain tags `kK`); el9 paths.  Same setup as `T10_RESULTS.md`
(DES Y3 nside 512, Q110, 917 patches, 56 baryonified N-body map-sets, 4 source bins,
`aperture_nside=None`), summarised by `t10_summary.py`:

| | pairs | ms / map-set | max abs(ratio−1), ζ_a,+ | ζ_a,− | max abs(Δζ)/σ_patch, ζ_a,+ | ζ_a,− | quadrature over the 140 entries, ζ_a,+ | ζ_a,− |
|---|---|---|---|---|---|---|---|---|
| k = 2.9, plain (7.x) | 39.0 M | 37.2 | 0.14 | 0.28 | 0.23 | 0.27 | 0.6 σ | 1.0 σ |
| k = 2.9, quad | 39.0 M | 29.0 | 0.13 | 0.23 | 0.23 | **0.20** | 0.6 σ | **0.7 σ** |
| k = 2, quad | 18.6 M | 30.4 | 0.31 | 0.37 | 0.25 | 0.39 | 1.0 σ | 1.3 σ |
| k = 1.5, plain (7.x) | 7.4 M | 35.5 | 0.33 | 0.59 | 0.29 | 0.81 | 1.2 σ | 3.7 σ |
| k = 1.5, quad | 7.4 M | 21.6 | 0.33 | 0.57 | 0.29 | **0.39** | 1.2 σ | **1.6 σ** |

The moments do what they can: at k = 1.5 the ζ_a,− shift halves (0.81 → 0.39 σ_patch, 3.7 →
1.6 σ in quadrature) and at k = 2.9 it drops below the ζ_a,+ one.  What remains is the
**binning window** — ζ_a,+ is untouched by construction and ζ_a,− now matches it — which is
a binning effect, not the moments' (an edge-splitting search would remove it; it was built,
measured at 5–25× the per-map cost and removed again).  k = 2 with quad (0.39 σ) does not pass
the 0.27 σ criterion that k = 2.9 passes, so the production k stays 2.9; the ζ price of
k = 2.9 itself falls from 0.27 to 0.20 σ_patch with the default `"quad"`.


## B5: the register spill of the moment tiles (2026-11-03)

`EndMoments[NS]` per end -- 10 live values per (end, bin), 80 at nz = 4 -- plus the float64
accumulators spilled the quad tiles to local memory. The three moment tiles now visit every
ordered bin pair (t, u) and load the two ends' contractions inside that double loop
(`tri_index` folds the combination index to a constant once the loops are unrolled); the
arithmetic of every contribution and the order in which a combination accumulates them are
unchanged, so the sums are bitwise the old ones (GPU gates of `tests/test_cell_moments.py`
pass). Price: NS(NS+1) end loads per pair instead of 2 NS.

ptxas (`register_probe.py`, float32 maps, float64 accumulation, A100):

| kernel | nz | dipole regs / local B | quad regs / local B |
|---|---|---|---|
| `gpu_tiled_*_reduce_xipm_mom` | 1 | 48 / 0 | 64 / 0 |
| | 4 | 167 / 0 | 168 / 0 |
| `gpu_tiled_*_reduce_ds_mom` | 1 | 48 / 0 | 72 / 0 |
| | 4 | 218–250 / 0 | 255 / 32 |
| `gpu_tiled_*_reduce_3x2pt_mom` | 1 | 64 / 0 | 95 / 0 |
| | 4 | 255 / 112–120 | 255 / 496–504 |

Timing, nside 1024, 200 patches, k = 4, nz = 4 (`results/bench_1024_b5.json` vs
`results/bench_1024.json`), ms per call:

| pack | cell_moments | gg before → after | kg | three_by_two_pt |
|---|---|---|---|---|
| no | None | 1.65 → 2.55 | 2.78 → 3.44 | 3.78 → 4.04 |
| no | dipole | 2.40 → 2.41 | 3.66 → 4.19 | 5.07 → 5.06 |
| no | quad | **5.35 → 3.72** | 5.60 → 5.03 | 8.93 → 6.99 |
| yes | None | 1.78 → 2.02 | 2.80 → 3.45 | 3.86 → 3.86 |
| yes | dipole | 2.41 → 2.43 | 3.77 → 3.71 | 5.11 → 5.14 |
| yes | quad | **4.73 → 3.20** | 5.59 → 4.99 | 9.32 → 7.04 |

The mono and dipole rows did not change code and show the run-to-run noise of these ~1 ms
host-bound calls (±1 ms). Quad `gg` is now 1.3–1.6× mono instead of 3.2×; `kg` and
`three_by_two_pt` improve by 10–25 %, less because `tile_ds_mom` still has 32 B of local
memory and `tile_multi_mom` 500 B: its 82 float64 accumulators (10 ξ± combinations × 3,
10 w × 2, 16 γ_t × 2) are 164 registers before any geometry. `b5_3x2pt_probe.py` measured
the one-walk multi tile against the standalone moment tiles (three walks, no spill;
`results/b5b_17669553.log`): quad 7.06 vs 7.17 ms (unpacked), 7.05 vs 7.17 (packed), dipole
5.34 vs 5.08 and 5.22 vs 5.30 -- equal within the noise. The 500 B spill of the multi tile costs
nothing measurable against the three extra walks it saves, so the moment path keeps the plain
tile's accumulator budget and the multi tile. (The two paths are not bitwise equal to each
other -- different template instantiations contract FMAs differently under `--use_fast_math` --
both pass the moment gates; that parity was never a gate.)

Verdict: B5 closed. Quad at nz = 4 costs 1.3–1.6× mono on `gg`, 1.5× on `kg`, 1.7–1.8× on
`three_by_two_pt`, all of it the arithmetic of the moment terms, no spill left on the ξ± tile.
