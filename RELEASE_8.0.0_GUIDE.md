# CosmoFuse → PixPoint 8.0.0 — implementation guide

Working document for the 8.0.0 release. It replaces nothing: the planned-improvements
list (`~/Downloads/cosmofuse-8.0-improvements.md`) keeps the roadmap (docs, PyPI,
webpage follow 8.0.0); this file is the engineering plan for the release itself.
Like `GPU_IMPLEMENTATION_GUIDE.md` and `STATIC_TREECODE_GUIDE.md` before it, it is
deleted once every stage is done; the science caveats that outlive the code move
into `CLAUDE.md` then.

Prototype and measurements the new features build on:

- `benchmarks/moments/` — moment-carrying cells (`moment_probe.py`, `pairwise_check.py`,
  `RESULTS.md`): formulas, conventions, accuracy and per-pair kernel costs on an A100.
- OptimSBI (el9, `/project/ocean1-seitz/users/dgebauer/research/OptimSBI`):
  `src/optimsbi/moment_treecode.py` (device moments, `edge_split_pairs`, `MomentEstimator`),
  `scripts/bench_global2pcf_moments.py`, section "Moment-carrying cells and edge splitting"
  of `notebooks/02_global2pcf_treecorr.ipynb` (13 runs at nside 1024).
- `~/Documents/research/software/PDFast` — the harmonic-space 1-pt pipeline the new
  `OnePoint` estimator complements (it keeps mass mapping; we do not).

## 0. Scope, order, ground rules

**8.0.0 contains, in one breaking release:**

| stage | change | kind |
|---|---|---|
| A | remove `ThreePointResult.to_centroid()`; remove per-tile 2PCF sums and `TwoPointResult.jackknife()` | breaking |
| B | moment-carrying treecode cells (`cell_moments = None | "dipole" | "quad"`) | estimator option |
| C | ~~edge-splitting pair search (`edge_slop`)~~ — implemented, measured, removed | — |
| D | 1-point statistics: `OnePoint` (aperture-mass PDF from shear, smoothed-field PDF of κ/δ, moments) | new estimator |
| E | rename to **PixPoint** (`import pixpoint`), HDF5 `kind` strings, URLs | breaking |
| F | version, CHANGELOG with before/after table, GPU test pass, `CLAUDE.md` rewrite | release |

**Order: A → B → C → D → E → F.** The removals go first because they are small and
simplify what B and C touch (`_finish_pair_sums`, `TwoPointResult`). The rename goes
last so that every feature is developed, benchmarked and reviewed against 7.7.0 under
the same import path, the prototype scripts keep running, and the paper repo's pinned
release is undisturbed. Features B and C are independent of each other and of D; D is
independent of both. Each stage is its own commit series with the test suite green.

**Defaults (decided 2026-11-02, revised 2026-11-03): everything stays as in 7.7.0 -- `pack_pairs=False`, full resolution.**

| argument | 7.7.0 | 8.0.0 |
|---|---|---|
| `pack_pairs` | `False` | `False` (unchanged; `True` = 8 B/pair on the device, quantised rotations on both backends, opt-in as before) |
| `pack_host_pairs` | `False` | `False` (opt-in, unchanged) |
| `resolution_factor` | `None` | `None` (full resolution) |
| `aperture_nside` | `None` | `None` (full resolution) |
| `cell_moments` | — | `"quad"` — inert at full resolution; acts whenever `resolution_factor` or `aperture_nside` is set |

~~Consequences of `pack_pairs=True` by default~~ (moot since the default stays `False`; kept for the record): the private
single-map `_*_single` methods, unpacked-only in 7.7.0, are made to work with packing (A3;
the sum-of-weights precompute is gone, see A4); the rotations are the uint16-quantised
ones on the GPU **and** the CPU backend (one estimator), so results differ from 7.7.0's
default at the ~1e-6 relative level; pair files stay exact (packing happens in `prepare()`).
The treecode, the split and the moments remain opt-in: a default construction measures the
exact pixel-pair estimator.

**Ground rules (from `CLAUDE.md`, unchanged in substance):**

- `resolution_factor=None, pack_pairs=False` stays **bit-for-bit identical to 4.20.0** (except
  `M_a` 7.6.0, `M_g` 7.6.2); `resolution_factor=None, pack_pairs=True` stays bit-for-bit
  identical to 7.7.0 with `pack_pairs=True`. Nothing in B–D may touch the full-resolution
  pair path or its packed payload encoding. `cell_moments=None` must
  reproduce 7.7.0 bitwise for every k (same `pack_pairs`).
- Every new `__global__` needs a numpy twin in `tests/cuda_emulation.py`
  (`_KERNEL_EMULATORS`), the typed-launch signature parse must accept it, and
  `tests/test_kernel_signatures.py` must stay green. CPU backend = Numba kernel next to the
  GPU wrapper in the same `backend/` module.
- The machine-precision and global-estimator gates listed in `CLAUDE.md` are never loosened.
  New gates are added, old ones are only removed when the feature they gate is removed (A).
- `mypy --strict`, black/isort/flake8 clean; device-or-host arrays are `Any`.
- Document each stage in `CHANGELOG.md` as it lands (one 8.0.0 entry, grown stage by stage),
  and in `docs/estimator_note.md` where the estimator changes (B, C, D).
- Benchmarks/gates that need a GPU run on `usm-cl-seitz1/2` via
  `benchmarks/static_treecode/deploy_el9.sh` + `remote_run.sh`.

## Stage A — removals

### A1. `ThreePointResult.to_centroid()` — done 2026-11-02

- Delete `to_centroid` (`src/CosmoFuse/results.py:321`) and `_phi_coefficients` if it has no
  other caller after that; delete `TestGlobal3PCF.test_centroid_is_a_phase`
  (`tests/test_global_3pcf.py:306`).
- Remove the mentions in `global_3pcf.py:33`, README (`to_centroid` example at ~459 and the
  sentence at ~473). Replace with the projection convention stated in words (vertex order,
  east→north angles, φ = φ₃ − φ₂, x-projection β per vertex, the sign in f̃ = −f e^{−2iβ},
  conjugation flags) and the one-line recipe "convert theory to the x-projection per triangle".
  The rotation itself lives on in the paper repo's benchmark scripts.

### A2. Per-tile sums and `jackknife()` — done 2026-11-02

- `TwoPointResult`: `tile_num[name]`, `tile_weight` `(ncomb, n_tiles, nbins)` → `num[name]`,
  `weight` `(ncomb, nbins)`. Drop `n_tiles`, `jackknife`; `__add__` sums instead of
  concatenating (results of two devices, two tile sets or two maps add). `estimate` is the
  ratio of the summed arrays as now.
- `global_2pcf.py`: `_TileEngine._finish_pair_sums` keeps capturing the unnormalised sums but
  `Global2PCF` reduces over tiles before building the result. Tiles stay an internal detail of
  the pair search (`tile_nside` keeps its meaning for `TileSet.reach`, balancing, cell
  alignment `tile_nside <= coarsest level`). Multi-device: per-device totals add.
- Tests: `tests/test_global_2pcf.py` G2 block (`test_jackknife`, per-tile row labels) goes;
  the ≡-one-patch gate stays (it compares totals). `tests/test_global_multi_device.py`:
  "Global2PCF bitwise per tile" becomes "per-device sums add to the one-device sums". This
  cannot stay bitwise: the one-device total is one float64 sum over all tiles, the multi-device
  total is a sum per range plus a final add, and float64 addition is not associative. Gate:
  ≤ 1e-13 of the largest sum (the same tolerance `Global3PCF` already uses between device
  counts). **Done 2026-11-02.**
- `global_geometry.py:20`, README features bullet (line 73) and the `Global2PCF` usage block
  (~426–430). The `ZetaWriter` docstring lines (`zeta_writer.py:24–31`) stay: they describe
  what a *user* can still derive from `reduce="none"` per-patch arrays, which remains true.
- Pair file format of `Global2PCF` is unchanged (tiles are still how pairs are stored).

### A3. Packing by default — dropped 2026-11-03: the default stays `pack_pairs=False`

Decision: the default construction stays the exact, bitwise-reproducible estimator (the ground rule
the whole test suite leans on); packing is a memory optimisation one turns on knowingly when the
geometry does not fit, and flipping the default would have moved every result by ~1e-6 relative
and forced the lazy-upload plumbing below for the test references alone. Nothing to do. The plan
as it was:


- **Single-map references** (`_gg_single`, `_kk_single`, `_kg_single`): no package caller,
  ~70 call sites in ten test files as the per-pair reference for the tiled kernels — keep
  them independent (no wrapper over the tomographic path). Replace the raise in
  `_require_unpacked_pairs` by a lazy upload: on first use with `inds_dev is None`, translate
  the host `pair_inds` to rows exactly as `prepare()` does without packing, with the
  **quantised** rotations when `pack_pairs=True` (what the CPU backend already holds), cache
  on the `ComputeContext`, and raise only when the host arrays are gone (`release_host_pairs`,
  `pack_host_pairs`). +24 B/pair while a reference is in use; docstring says so. Gate: single
  ≡ tomographic ≤ 1e-12 with packing on and off, both backends; `test_pair_packing.py`'s
  "raises" assertion becomes "works, and raises only without host arrays". Possible later
  cleanup, not 8.0.0: numpy brute-force references inside the tests and deletion of the
  single kernels.
- ~~Sum-of-weights precompute~~ — removed instead (A4): the tomographic kernels accumulate
  Σ w_i w_j in the same pass over the pairs, so an explicit `sumofweights=` only replaced the
  kernel's own denominators and saved nothing; a packed weights-only pass would have been a
  kernel plus twin for a feature with no effect. `_compute_xipm_sumofweights` /
  `_get_xipm_sumofweights` stay as internals of the single-map references on the GPU (their
  kernels do not accumulate denominators) and follow the lazy upload above.

### A4. API surface that was not the estimator's — done 2026-11-02

Removed with the Stage A survey (no package caller; tests pointed at the real owner):

- `Integrated3PCF.zeta_*` (8) and `Integrated3PCF.calculate_all_zetas` — pass-throughs to
  `correlation_helpers`, duplicated a third time by `IntegratedResult.zetas()`.
- `Integrated3PCF.get_pairs_patch`, `get_pairs_patch_M_a`, `__get_pairs_helper__`,
  `__get_pairs_M_a_helper__`, `aperture_filter_key` — forwarding shims to `PairGeometry`;
  `pair_geometry.py` calls its own static methods now.
- The setters of `inds_dev`, `exp2phi_dev`, `bins_dev`, `tot_bins_dev`,
  `tot_bins_reduceat_dev`, `ntotpairs` — `prepare()` writes the `ComputeContext` directly,
  tests that fake a prepared state do the same; the getters stay.
- `compute_sumofweights` and the `sumofweights=` keyword of `gg`/`kk`/`kg` and of every
  tomographic private method (`_gg_pairs` … `_density_shear_tomo_vectorized`), with
  `_normalize_tomo_sumofweights_*`, `_is_per_comb_sumofweights_shape`,
  `_symmetrised_denominators`, `_compute_tomo_sumofweights` and the `_tomo_sumofweights_cache*`
  fields of `ComputeContext`. Reason above.

## Stage B — moment-carrying cells — implemented and measured 2026-11-02 (one B5 follow-up open)

Shipped as designed below with these resolutions of the open points: (B2.2) the moments are
built *straight from the pixel rows* by `gpu_moment_level` (`cuda/moment_rows.cu`) with the
geometry evaluated on the fly from the stored unit vectors — no per-entry factor arrays; the
cell rows of shear get the transported mean written over the plain one; (B2.4) the coarse pairs
carry `pair_phi (2, n_coarse)` + `pair_inv_d` (a contiguous tail of every patch's list);
packed coarse blocks encode φ in the uint16 angle and a uint16 separation code per pair
(`packing.encode_separations`, linear in the bin); (B2.5) the plain tiles run over *all* bins
and the moment tiles overwrite the coarse bins afterwards (`mom_base[b] < 0` marks a fine
bin) — a per-bin skip in the plain tiles is the B5 optimisation if the ~15 % plain work on
coarse pairs shows; the per-(bin,row) fallbacks refuse moments with a message; (B2.6) S = −1
on the kernel rows (`moments.MOMENT_SIGN`), fixed by `tests/test_cell_moments.py::
TestConventions`; (B2.8) aperture moments need no weight moments: CosmoFuse normalises by
Σw without the filter, so only the numerator expands (`moments.aperture_shear_terms`,
`aperture_scalar_terms`); the moment aperture kernels have only the one-block-per-patch form
(≤ 16 bins). Gates B4.1–B4.7 are `tests/test_cell_moments.py` (C1–C6 + a GPU class).
The ξ+ residual of the transported mean has a spherical O(p θ) term (both in radians) that
the flat-sky transported J frame misses: 4e-3 at nside 128/32, 1e-5 in production.

**B5 measured** (`benchmarks/moments/RESULTS.md`, "Library implementation"): T10 with quad
at k = 2.9 / 2 / 1.5 → ζ_a,− 0.20 / 0.39 / 0.39 σ_patch (plain 0.27 / — / 0.81); ζ_a,+
unchanged (binning); production k stays 2.9 until stage C.  Cost on the A100: free at nz = 1,
but the **quad tile spills registers at nz = 4** (gg 1.65 → 5.35 ms at nside 1024, k = 4;
dipole 2.40 ms) — the open item: recompute the per-end contractions per combination instead of
caching `EndMoments[NS]` for both ends, verify with `-Xptxas -v`.  Aperture moments cost the
estimated 2–3× of the (small) aperture pass.

### B1. What ships

For the coarse (treecode) levels every cell carries, besides `M = Σ w γ̃` and `W = Σ w`,
the dipole `D = Σ w γ̃ δr` (2 complex) and, for `"quad"`, `Q = Σ w γ̃ δr δrᵀ` (3 complex),
with δr the gnomonic east/north offset of the child from the cell centroid in the cell's
tangent frame and γ̃ the child shear parallel-transported into that frame
(γ̃ = γ e^{2iSρ}, ρ = angle of the child's east vector in the cell frame). The ξ− pair
term becomes (p, q the perpendicular/parallel unit vectors of the pair at each end, the
J-end ones transported: p_J = (sin φ_J, −cos φ_J), q_J = (−cos φ_J, −sin φ_J); d = pair
separation):

```
F = M_I M_J
    − 4iS/d   (M_I Dp_J − Dp_I M_J)
    − 8/d²    (M_I Qpp_J − 2 Dp_I Dp_J + Qpp_I M_J)
    + 4iS/d²  (M_I Qqp_J − Dq_J Dp_I − Dq_I Dp_J + Qqp_I M_J)
ξ− += Re(F e^{−2iS(φ_I+φ_J)});   ξ+ uses M_I conj(M_J) only (ends cancel).
```

γ_t (`kg`, `tile_ds`, the γ_t section of `tile_multi`) has the same window with half the
phase: the pair product δ_i γ_j e^{−2iSφ} is off by e^{−2iSδφ}, and **δφ depends on the
offsets at both ends**, so the scalar lens cell contributes its own (real) moments
D_δ = Σ w δ δr, Q_δ = Σ w δ δr δrᵀ:

```
F_t = M_I M_J
    − 2iS/d   (M_I Dp_J − Dp_I M_J)
    − 2/d²    (M_I Qpp_J − 2 Dp_I Dp_J + Qpp_I M_J)
    + 2iS/d²  (M_I Qqp_J − Dq_J Dp_I − Dq_I Dp_J + Qqp_I M_J)
γ_t += Re(F_t e^{−2iSφ_J})      (I = lens, scalar, no transport; J = source, transported)
```

(the ξ− formula with 4 → 2; verify the sign of the second-order term with the per-pair
check exactly as was done for ξ−, B4.2). Nothing else has a phase: ξ+ and ξ_g = w(θ) see
only the binning window (stage C), `Global3PCF` has no cells.

Measured (RESULTS.md): mean ξ− error at k = 4 goes from −2…−5 % to ≤ 0.3 %; information
retained with shape noise 0.97–0.99 → ≥ 0.9999; quad at k = 2 beats mono at k = 4. Dipole
alone has a real coefficient 1 + 8δφ² (bias flips sign, leakage removed) — it is an
information fix, offered because it is 1.6× instead of 2.2–2.9× the kernel cost.

### B2. Design decisions (take them before coding)

1. **Where the moments are made.** Directly from the pixels per level (one CSR pixel → cell
   per treecode level, geometry factors `(δx, δy, δx², δxδy, δy²)` times the transport
   phasor as CSR values), *not* by composing level by level: composing needs a shift of the
   expansion centre and a frame change per level and is exact only to second order. The
   prototype (`Nodes.moment_entries`) is the direct form and is validated. Cost ∝
   n_levels × n_pixels of the footprint, same as the sparse chain's total.
2. **Moments as virtual rows.** Extend the appended-row layout: after `[aperture | level 1 |
   level 2 | …]` the cell rows get companion buffers `D`, `Q` (AoS per row, complex, in the
   map precision) produced by a new degrade kernel `gpu_moment_level` (SEG lanes per cell
   like `gpu_degrade_level`, complex factors, reads the already-signed pixel rows so the
   `flip_g1/flip_g2` fold stays free) and its sparse-chain reference (cupyx/scipy CSR
   mat-vec on the pixel rows). Both go through `_expand_rows` under the existing memo /
   `_expansion_scope`; the layout parameter grows a `moments` flag. Shear rows get complex
   moments (transported), scalar rows get real moments **only when a call needs them**
   (`kg`, `three_by_two_pt`; `kk` never) — `_expansion_scope` already knows the statistic.
   With `cell_moments=None` nothing is allocated or launched (7.7.0 bitwise).
3. **One M.** With moments on, use the *transported* M everywhere (ξ+ phase error improves
   from 3.5e-5 to 1e-5); with moments off, the untransported shipped M. Document that
   `cell_moments` changes ξ+ at the 1e-5 level, not only ξ−.
4. **Pair payload.** The correction needs the **full** position angle at both ends and 1/d
   per pair; `unique_pairs` / the patch finder return e^{2iφ}, which is not enough. For the
   coarse levels store φ (not 2φ): unpacked → `pair_exp2phi` stays for the fine levels, a new
   per-level `pair_phi` (float32) + `pair_inv_d` (float32, or uint16 over the bin's range —
   1 % is enough) for coarse levels; packed → the uint16 angle code of coarse blocks encodes
   φ instead of 2φ (same bytes; 2φ resolution halves on coarse blocks only, fine-level
   blocks and the full-resolution encoding are untouched) plus a uint8/16 1/d column →
   10 B/pair on coarse blocks. `PackedPairs` loader gets a template flag. The **full-
   resolution encoding does not change** (4.20.0 gate).
5. **Which kernels (decided 2026-11-02: parity for every kernel where a phase exists).**
   `tile_xipm` (ξ−), `tile_ds` (γ_t, with the scalar lens moments) and both sections of
   `tile_multi` (`tomo_tiled_3x2pt.cu`), each with a template int `MOM` ∈ {0, 1, 2}; the
   Numba twins and the emulation twins follow. `tile_dd` (ξ_g) is untouched. The
   per-(bin,row) fallback kernels `gpu_fused_tomo_reduce_*` (beyond
   `_MAX_TILED_ACCUMULATORS`) refuse `cell_moments` with a clear error rather than growing
   a third variant. `cell_moments` therefore means the same thing for every pair statistic
   of `Integrated3PCF` and `Global2PCF`.
6. **Sign convention.** The probe's S = +1 on the unflipped shear matches CosmoFuse
   `flip_g1=True`. Derive S once from the kernel's rotation convention on the *signed* rows
   (the fold happens in the degrade), and gate it (B4.2) — do not carry S as a user option.
7. **Defaults (decided, see § 0).** `cell_moments="quad"`: inert at the full-resolution
   defaults, active whenever `resolution_factor` or `aperture_nside` is set, so a treecode
   measurement is the corrected one unless the user asks for `None` (= 7.7.0 cells) or
   `"dipole"`. Goes into the provenance either way.
8. **Aperture moments at `aperture_nside` (decided 2026-11-02: dipole and quad ship in
   8.0.0).** The aperture pass has its own cell window, two effects of the same order: the
   tangential phase about the centre (2δφ, `M_a` only) and the filter value at the cell
   centroid instead of the child (Q′·δr, `M_a` and `M_g`). To first order both are fixed by
   the **dipole** (D of the shear rows, D_δ of the scalar rows, plus the weight dipole
   D_w = Σ w δr for the denominator, which vanishes only for uniform weights):

   ```
   Σ_p w γ_p e^{−2iφ_p} Q(θ_p) ≈ e^{−2iφ_c} [ Q_c M_c − 2iS Q_c/θ_c (p̂_c·D_c) + Q′_c (r̂_c·D_c) ]
   Σ_p w δ_p Q(θ_p)            ≈ Q_c M_c + Q′_c (r̂_c·D_δ,c)
   Σ_p w   Q(θ_p)              ≈ Q_c W_c + Q′_c (r̂_c·D_w,c)
   ```

   (r̂_c, p̂_c = radial / tangential unit vectors from the centre to the cell centroid in the
   cell frame; M_c, D_c transported as in B1; U and its derivative for `M_g` with a filter;
   the patch-mean `M_g` has Q′ = 0 and needs nothing.)
   The first-order terms vanish for a field constant over the cell (the centroid is the
   children's mean), so the dipole removes the *shape-noise scatter* (information loss) of
   the aperture window; the *mean bias* is second order and stays with the dipole. The
   **quad** removes it: with θ_p ≈ θ_c + r̂·δr + (p̂·δr)²/(2θ_c) and φ_p ≈ φ_c + (p̂·δr)/θ_c
   − (r̂·δr)(p̂·δr)/θ_c², the second-order terms are the phase curvature −2 Q_c/θ_c² (p̂·δr)²,
   the phase–gradient cross term −2iS Q′_c/θ_c (r̂·δr)(p̂·δr) together with the second-order
   angle term +2iS Q_c/θ_c² (r̂·δr)(p̂·δr), and the filter curvature ½Q″_c (r̂·δr)² +
   ½ Q′_c/θ_c (p̂·δr)² — each contracted with the cell's Q moments (shear: 3 complex,
   scalar: 3 real, weights: 3 real) as in the ξ− formula. Filters therefore need Q′ and Q″.
   Bytes per disc cell: mono ≈ 28 B (γ, W, index, angle, Q), dipole ≈ 56 B, quad ≈ 96 B →
   plan with 2× / 3× the mono pass (the pair kernel measured 1.7× / 2.2×). Expected
   (estimates, θ_Q = 90′, base 2048; mono scatter = the measured ζ price 0.04 σ at ap 256 /
   0.07 σ at ap 128, flat in θ; quad bias ratios from the ξ− measurements, 1/25 of mono at
   pixel/θ = 0.17 and 1/8 at 0.33; scatter ratios from the ξ− information table):

   | aperture nside | pixel/θ_Q | variant | mean bias | ζ scatter | cost vs 512 mono |
   |---|---|---|---|---|---|
   | 512 | 0.076 | mono | ~0.1 % | ≤ 0.02 σ | 1 |
   | 256 | 0.15 | mono | ~0.5 % | 0.04 σ | 0.25 |
   | 256 | 0.15 | dipole | ~0.5 % | ~0.01 σ | ~0.5 |
   | 256 | 0.15 | quad | ~0.02 % | ~0.003 σ | ~0.75 |
   | 128 | 0.31 | mono | ~2 % | 0.07 σ | 0.06 |
   | 128 | 0.31 | dipole | ~2 % | ~0.02 σ | ~0.12 |
   | 128 | 0.31 | quad | ~0.25 % | ~0.005 σ | ~0.19 |

   256 + quad beats 512 mono on both axes at 3/4 of the cost and 128 + quad beats 256 mono on
   both axes at a slightly lower cost — the same pattern as ξ− (quad k = 2 vs mono k = 4). The
   dipole alone is the information-per-cost choice when the mean rescaling is accepted as a
   shared window (SBI); quad when the data vector must stay close to the full-resolution
   aperture (theory, PDF tails in `OnePoint`).
   Implementation: the aperture-level rows carry D and Q (shear: 2 + 3 complex, scalar:
   2 + 3 real) and D_w, Q_w (2 + 3 real) from the same `gpu_moment_level`; the aperture
   geometry stores the full centre→cell angle φ in place of cos 2φ / sin 2φ (r̂, p̂ follow
   from it), 1/θ_c, and Q′, Q″ next to Q (`aperture_filter` objects get derivatives: analytic
   for the shipped filters, central finite differences otherwise); the three aperture
   kernels (`gpu_aperture_shear_tomo_fused`, `gpu_aperture_density_tomo_fused`,
   `gpu_3x2pt_tomo_aperture_fused`) get a `MOM` ∈ {0, 1, 2} template and the per-(patch, bin)
   fallbacks refuse `MOM > 0`; `OnePoint` (stage D) uses the same kernels and inherits it.
   `cell_moments` governs the aperture pass with the same meaning as the pair pass.
   Full-resolution apertures (`aperture_nside=None`) have D = Q = 0 and stay bitwise.
   **Probe done 2026-11-02** (numpy, nside 128 with aperture levels 64 / 32, p/θ_Q = 0.18 and
   0.37; `benchmarks/moments/RESULTS.md`, "Aperture probe"): per cell mono 4–9 % → dipole
   0.4–0.7 % → quad 0.02–0.2 %; patch sums ±4 % → ±1 % → ±5e-4.  The reference that isolates
   the within-cell expansion is the exact child sum with the *coarse* disc membership and the
   coarse centre pixel (the full-resolution aperture differs by the window, not by the cells);
   the production-regime columns of the table above remain estimates.
9. **Pair file format 5** for B (and C): `pair_phi`, `pair_inv_d`, `cell_moments` attr;
   `load_pairs` reads 2–4 as today; a file without `pair_phi` can be used only with
   `cell_moments=None` (or re-searched) — raise, don't guess.

### B3. Files

- `treecode.py`: cell frames (east/north at the centroid) and per-level moment CSR
  (`TreecodeGeometry.moment_entries(level) -> (indptr, indices, factors (nnz, 6) complex)`);
  `slice_patches` carries them.
- `backend/degrade.py`, `cuda/degrade_rows.cu` (+ `gpu_moment_level`, numpy twin).
- `pair_finder.py`, `global_geometry.py::unique_pairs`: full angles + separation out of the
  Numba search for coarse levels (`ordered_pairs` already returns full angles — reuse its
  angle arithmetic).
- `packing.py`: coarse-block encoding of φ and 1/d; `prepare()` row blocks unchanged.
- `cuda/pair_tiles.cuh` (`tile_xipm<…, MOM>`, `tile_ds<…, MOM>`, loaders),
  `tomo_vectorized_xipm.cu`, the γ_t kernel file, `tomo_tiled_3x2pt.cu`, their wrappers in
  `backend/shear.py`, `backend/density.py`, `backend/three_by_two.py`, Numba twins,
  `tests/cuda_emulation.py`.
- `cuda/aperture_tomo.cu`, `cuda/tomo_fused_3x2pt.cu` (aperture sections, `MOM` template),
  `backend/aperture.py`, `pair_geometry.py` (full angle φ and Q′ in the aperture geometry,
  `KApertureGeometry` values + derivative), the filter functions (derivatives), Numba and
  emulation twins (the aperture twins also assert the strides).
- `correlations.py`: `cell_moments` argument, `_expand_rows*` moments, memo keys, the
  `ComputeContext` buffers; `global_2pcf.py` passes it through; `multi_device.py` nothing
  (each engine holds the full treecode).
- `io_handler.py`, `global_2pcf.py` save/load (format 5), `zeta_writer.py::_collect_provenance`
  (`cell_moments`).
- `docs/estimator_note.md`: the expansion and the transported frame, stated as an estimator.

### B4. Gates (new tests, `tests/test_cell_moments.py`)

1. `cell_moments=None` ≡ 7.7.0 **bitwise** at k ∈ {2, 4} on the stored test maps (CPU) and on
   the GPU (benchmarks `bitwise_check.py`).
2. Per coarse pair, mono / dipole / quad residual vs the exact child sum scales as
   4δφ : 8δφ² : (4δφ)³/6 for ξ− and 2δφ : 2δφ² : (2δφ)³/6 for γ_t (`pairwise_check.py` logic
   as a test, nside 128, tolerance a factor 2 on the rms ratio). This is also the gate that
   fixes the sign of the second-order γ_t term.
3. Quad ξ− of a patch vs exact child sums with the same bin assignment ≤ 0.3 % at k = 4 on
   the test field; ξ+ identical to the exact child sums to 1e-5 (transport).
4. Packed ≡ unpacked ≤ 1e-6 relative (the φ quantisation; the 2φ path keeps its ≤ 1e-12 gate).
5. CPU ≡ GPU ≤ 1e-12; emulation twins; `Global2PCF` ≡ one-patch `Integrated3PCF` with
   `cell_moments="quad"` ≤ 1e-12; multi-device adds bitwise; file round trip (format 5).
6. Aperture moments: `m_a` / `m_g` with `aperture_nside` vs the exact full-resolution aperture
   on a smooth test field: dipole residual per patch ≤ mono / 5 and quad ≤ mono / 20 at
   ap = base/4; ≡ mono bitwise at `aperture_nside=None`; finite-difference filter derivatives
   ≡ the analytic ones ≤ 1e-8; `three_by_two_pt`'s aperture sections ≡ `m_a`/`m_g`.
7. `three_by_two_pt` ξ− ≡ `gg` ξ− and γ_t ≡ `kg` γ_t with moments (fused tile vs standalone
   tiles, ≤ 1e-12); `kk` with `cell_moments="quad"` ≡ `cell_moments=None` bitwise (no
   scalar moments are built for it).

### B5. Measurements before release (A100)

- Kernel cost per pair with moments restricted to coarse blocks (the prototype applied them
  to every pair): expect k = 4 quad ≈ +17 % at nz = 1, +26 % at nz = 4 over mono; check the
  **nz = 4 quad register pressure** (prototype 2.9× mono = spilling) with `-Xptxas -v`.
- T10-style ζ-level test with `"quad"` at k = 2 and 1.5: if quad at k = 2 passes the 0.27 σ
  criterion, the production k can drop and the pair count with it (RESULTS.md: quad k = 2
  retains more ξ− information than mono k = 4).
- Memory: +6 complex per coarse cell row (negligible), +2 B/pair on coarse packed blocks.
- Aperture probe (B2.8) first, then the aperture pass cost with dipole / quad at ap 512 / 256 /
  128 (expect 2× / 3× the mono pass at the same level) and the ζ-level price via
  `t10_zeta_level.py --aperture-nside` for mono, dipole and quad.

## Stage C — edge-splitting pair search — implemented, measured and **removed** (2026-11-02)

`edge_slop` shipped in `a2fe1a1` (search, kernels' pixel-end guard, packing bounds, gates) and
was measured on the A100 (`65f0ece`): at k = 4 the exact binning had 1.5–2.6× (patches) to
4.4–10.8× (footprint) the static pairs and ran ~2× slower per pair — `three_by_two_pt` 9.8 → 56 ms
per map on 69 patches at nside 1024, 0.12 → 2.7 s on the footprint (1.8e9 pairs, 78 GB unpacked,
host search 277 s) — only ~1.5× cheaper than full resolution while still carrying the moments'
phase window. The user decided it does not earn its place next to the static list (fast, with
the moments) and full resolution (exact): reverted in the commit after `3dee385`; the code and
its measurements are recoverable from those two commits. The binning window of the static list
(±2 % at k = 4, alternating at the level seams) stays a known, deterministic property of the
treecode estimator, identical for data and simulations (see the science caveats).

## Stage D — 1-point statistics: `OnePoint` — implemented 2026-11-02 (D4 measurements pending)

**Implementation record** (what shipped vs. the plan below; `src/CosmoFuse/onepoint.py`,
`OnePointResult` in `results.py`, gates `tests/test_onepoint.py`, 18 tests):

- D1/D2.1 as planned: `OnePoint(nside, mask, scales, *, centre_nside, aperture_nside,
  min_coverage, aperture_filter, k_aperture_filter, radius_factor, edges, device, ...)`,
  `g(shear, w, flip_g1, flip_g2, cross, edges, values)`, `k(maps, w, edges, values)`. One
  `_ApertureEngine(Integrated3PCF)` per (geometry kind, scale, device) with the accepted centres
  as patches; `m_a` / `m_g` unchanged, so the values are bitwise the patch estimator's. The
  aperture geometry never needed `pair_inds` (`m_a` builds it on demand), so nothing in
  `correlations.py` changed. **M_× needs no kernel**: γ_× is the tangential kernel applied to
  (γ_2, −γ_1), i.e. `m_a` on the swapped components with `flip_g2` -- exact, the sign fold is
  free; price: a second pass over the disc geometry (`cross=False` skips it). A fused t/× kernel
  is the lever if D4 shows the pass matters.
- **Coverage is defined at the map resolution whatever `aperture_nside` is** (decided during
  the gates): the window must not move the centre set, or two windows could not be compared on
  the same apertures. The `"healpix"` top hat with `aperture_nside` needs `centre_nside <=
  aperture_nside`.
- D2.2 (geometry build) not done: the per-centre `query_disc` loop runs twice (coverage pass,
  then `find_aperture_pairs`); 4.5e3 centres at nside 64 take 0.2 s. Measure at 1e5 centres /
  nside 1024 in D4 before threading it.
- **D2.3 reversed: no `cuda/onepoint.cu`.** The reduction (`_reduce_scale`) runs on the
  backend's array module -- `searchsorted` + `bincount` for the half-open histogram, Σ x^k by
  `power` + `sum`, cross sums by two GEMMs -- on (nz, n_apertures) values that are tiny next to
  the aperture pass. A kernel would add a twin and a signature for nothing measurable; D4 can
  overturn this with a number.
- **D2.4 decided: unit weight per accepted aperture** (DES Y3 moments). An aperture whose disc
  holds no weight in one tomographic bin is dropped from the sums of every bin (`count` reports
  what entered). The real-GPU gate (`benchmarks/onepoint/gpu_gate.py`) caught that this cannot be
  read off the values: the Numba shear kernel returns NaN for Σw = 0, the GPU `safe_divide` and
  every `M_g` path return 0, so a GPU PDF silently counted zeros. `OnePoint` therefore runs a
  weights-only pass per call (`_nonempty_apertures`: `m_g` with the weights as field, unit
  weights, unit filter → `A Σw / N`, exactly 0 iff all weights are) -- one extra geometry read
  per call; the only change to `correlations.py` is that `_k_aperture_geometry_dev` keeps a small
  dict of filter values on the device instead of one slot, so alternating U and the unit filter
  does not re-upload. `moments (nz, nscale, 5)` are k = 1..5 raw (central 2..5 derived).
- D2.5: `device=[...]` splits centres by disc pixel count (`balanced_ranges`), parts add; the
  result holds *sums* (`power_sums`, `cross_sums`, `hist`, `count`, `n_apertures`) so `+` is
  exact. Writer: done -- `ResultWriter` (`result_writer.py`) is the labelled-array base
  (`ZetaWriter` now subclasses it, same file layout and tests), `OnePointWriter` streams the
  sums per map with scales/edges/names written once; `result_arrays()` serves the global results.
- D2.6: documented (module docstring, README, `docs/estimator_note.md` §5).
- D3 gates 1–5 all in `tests/test_onepoint.py` (gate 5's "kernel signature/twin" part is void
  without a new kernel; the emulated-GPU test patches every engine's backend instead), plus the
  real-GPU gate `benchmarks/onepoint/gpu_gate.py` (cupy vs CPU, device list, coarse apertures).
- D4: timing done (`benchmarks/onepoint/RESULTS.md`); the window study on the nside-512 archive
  and the PDFast cross-check are pending.


### D1. What ships

A pair-free estimator on the same row space, map inputs and weights:

- **Aperture mass from shear** (`g`): M_ap and M_× = A Σ w γ_{t,×} Q / Σ w Q at every centre,
  the kernel of `M_a` (`gpu_aperture_shear_tomo_fused`, one block per centre) with the ×
  component added. No mass mapping.
- **Smoothed scalar field** (`k`): κ or δ averaged with a circular top hat (`M_g`'s default
  patch-disc geometry, radius per scale), a compensated U, or — when the treecode rows exist —
  the HEALPix-pixel top hat of a degrade level (histogram of the cell rows).
- Per scale and tomographic bin: a weighted histogram on fixed edges, raw moments 2–5
  (and cross-bin moments ⟨x_i x_j⟩, ⟨x_i² x_j⟩, ⟨x_i x_j x_k⟩ as in the DES Y3 moments
  analysis), the count and summed weight of accepted apertures. Optionally the raw
  per-aperture values (`values=True`) for debugging and quantile estimators.
- Centres: all HEALPix pixels of `centre_nside` (per scale; default pixel ≈ θ/2) inside the
  footprint whose disc **coverage** (fraction of the disc's pixels that are unmasked, from the
  binary mask at geometry time, independent of the map weights) is ≥ `min_coverage`
  (default 0.8). Discs on the `aperture_nside` level as `M_a`/`M_g` do.

### D2. Design decisions

1. **Class, not a mode of `Integrated3PCF`.** `OnePoint(nside, mask, scales, filters,
   centre_nside=None, aperture_nside=None, min_coverage=0.8, edges=..., device=...)`, methods
   `g(shear, w, flip_g1, flip_g2)` and `k(maps, w)`, both returning `OnePointResult`
   (`hist (nz, nscale, nbins)`, `edges`, `moments (nz, nscale, 2..5)`, `cross_moments`,
   `n_apertures`, `coverage`, `+` sums histograms and moments). Internally it reuses
   `PairGeometry.find_aperture_pairs` / `k_aperture_geometry` with the centres as "patches" and
   the aperture kernels unchanged; `prepare()` without pairs needs the aperture geometry
   decoupled from `find_pairs` (`_prepare_aperture_flat` must not require `pair_inds`).
2. **Geometry build.** `k_aperture_geometry` loops `query_disc` per centre in Python: fine for
   10³ patches, minutes for 10⁵–10⁶ centres. Vectorise: `query_disc` per centre stays (it is
   the exact disc) but runs in `map_patches` threads, or build discs from the degraded cells'
   neighbourhood with a vectorised distance test. One-off, cached per (footprint, centres,
   scale) like the patch-mean geometry.
3. **Histogram/moment kernel.** New `cuda/onepoint.cu`: `gpu_hist_moments` — per block a
   shared-memory histogram (atomicAdd on float64/uint32 bins) + moment accumulators, one
   global atomicAdd per bin per block; input the per-aperture values and weights, the
   coverage mask applied by index selection at geometry time (accepted centres only are ever
   computed). Numpy twin; Numba kernel. Edges are fixed per (scale, z-bin) by the user (from
   the sims), out-of-range values go to under/overflow bins that are reported.
4. **Weights.** The aperture statistic is already weight-normalised per aperture; the
   histogram weight per aperture is 1 (default) or Σ w Q / coverage-free area — **decide**
   and document (DES-style analyses use unit weight after the coverage cut).
5. **Multi-device**: `balanced_ranges` over centres, parts add (`+`). `ZetaWriter`: add a
   sibling `OnePointWriter` or generalise `ZetaWriter` to labelled arrays (it is already a
   per-map-set reducer with provenance; the 1-pt arrays are small).
6. **Mass mapping stays outside.** κ from shear (Kaiser–Squires) needs SHTs; PDFast / the
   forward model produce κ and pass it as the `k` field. Say so in the docs.

### D3. Gates (`tests/test_onepoint.py`)

1. Per-aperture `g` values ≡ `Integrated3PCF.m_a` on the same centres (bitwise: same kernel);
   `k` ≡ `m_g`; M_× ≡ a numpy tangential/cross projection ≤ 1e-12.
2. Histogram and moments ≡ `np.histogram` / `np.average` of the per-aperture values (CPU and
   emulated GPU) ≤ 1e-12; under/overflow accounted.
3. Coverage: an aperture crossing the mask edge below `min_coverage` is absent; with
   `min_coverage=0` the count equals the number of footprint centres.
4. HEALPix-pixel top hat ≡ histogram of the degraded map (`hp.ud_grade` of the masked map
   with coverage weights) ≤ 1e-12.
5. Multi-device adds bitwise; CPU ≡ GPU ≤ 1e-12; kernel signature/twin tests.

### D4. Measurements

ms/map per scale at nside 1024 for θ = 20′, 45′, 90′ (expect < 1–10 ms, upload-bound);
the `aperture_nside` window on the PDF (tails!) along the lines of `APERTURE_RESULTS.md`:
base 512 → aperture 256/128, PDF moments and the 1 %/99 % quantiles vs full resolution,
and the same for the HEALPix top hat vs the circular one. Cross-check one configuration
against PDFast's top-hat κ PDF (same κ maps) — expect differences from the mask-edge
treatment, quantify them.

## Stage E — rename to PixPoint

Do this in one commit after D is merged and green.

- `src/CosmoFuse/` → `src/pixpoint/`; every `from CosmoFuse` / `import CosmoFuse` in `src`,
  `tests` (incl. `tests/_results.py`, `cuda_emulation.py`), `benchmarks/**`, `Makefile`,
  `.github/workflows/ci.yml` (`--cov=CosmoFuse`), `pyproject.toml` (`name = "pixpoint"`,
  `[tool.setuptools.package-data] "pixpoint"`, URLs → `D-Gebauer/PixPoint`,
  `pixpoint.readthedocs.io`), `README.md`, `CLAUDE.md`, `docs/estimator_note.md`.
- `__version__ = "8.0.0"` in `src/pixpoint/__init__.py` and `pyproject.toml`.
- HDF5: `FORMAT_KIND = "PixPoint.Global2PCF" / "PixPoint.Global3PCF"`; `load_pairs` accepts the
  `CosmoFuse.*` strings too (test: a 7.7 file loads). Provenance attrs written by `ZetaWriter`
  that carry the package name (version key) — write the new name, read both.
- Remove the stale `src/cosmofuse.egg-info/` (and make sure `.gitignore` covers egg-info).
- Rename the GitHub repo (redirect), then the local checkout directory, `CLAUDE.md` path
  references (`benchmarks/static_treecode/deploy_el9.sh` destination
  `/project/ocean1-seitz/users/dgebauer/research/CosmoFuse-static-treecode` → `PixPoint-…`),
  and the auto-memory directory last (its name is derived from the checkout path).
- Downstream, outside this repo (note in the CHANGELOG, fix when they next pin):
  OptimSBI's `optimsbi.external.cosmofuse_v7`; Perlmutter venvs `~/.venvs/cosmofuse`
  (6.4) and `~/.venvs/cosmofuse7` (7.7) keep their pins — a `pixpoint` venv comes with the
  first project that moves; the paper repo pins a release and converts with it.
- No `CosmoFuse` alias package.

## Stage F — release

- `CHANGELOG.md` 8.0.0 entry in the 7.0.0 style: **Breaking** paragraph, before/after table
  (`import CosmoFuse` → `import pixpoint`; `TwoPointResult.tile_num/jackknife` → `num`;
  `to_centroid` → docs recipe; `kind` strings), then one block per feature (B, C, D) with the
  measured numbers, then "unchanged": `resolution_factor=None` bitwise vs 4.20.0,
  `Global3PCF` untouched by B–D (full resolution only).
- `CLAUDE.md`: rewrite the overview under the new name; add the moments / split / 1-pt
  sections and their caveats (salvaged from this guide, § below); update the gate list.
- `docs/estimator_note.md`: B, C, D definitions (paper-appendix quality).
- `pytest` (CPU, incl. slow) + `pytest -m gpu` on an A100 + `benchmarks/static_treecode/
  bitwise_check.py`; `make lint`; `make build` (CUDA sources in the wheel under the new
  package-data key).
- Delete this guide; move the science caveats into `CLAUDE.md`.

## Science caveats to carry (for `CLAUDE.md` once the guide is deleted)

- Moments remove the ξ− *phase* window of coarse cells (mean bias and the sin 4δφ leakage);
  the *binning* window (pairs exchanged across bin edges, ±2 % at k = 4, sawtooth at level
  seams) remains; an edge-splitting search removes it but costs 5–25× per map at k = 4 and was
  removed from the library (stage C). Both windows are deterministic and shared by data and
  sims; a smooth data vector or comparability with exact codes means full resolution.
- `cell_moments` acts on ξ− (4δφ) and γ_t (2δφ, with scalar lens-cell moments), changes ξ+
  at the 1e-5 level (transported frame) and leaves ξ_g untouched. Store `cell_moments` and
  `level_table` with every data vector (provenance).
- Dipole alone flips the sign of the ξ− mean bias (+8δφ²); only quad restores the modulus.
- Aperture moments: the dipole removes the shape-noise scatter of the `aperture_nside` window
  (information), the quad also its mean rescaling of M_a / M_g (second order). Choose
  `aperture_nside` by information per cost for SBI (128 + dipole at θ_Q = 90′, base 2048),
  by the mean bias for theory comparison and PDF tails (quad; 256 + quad ≈ 512 mono at 3/4
  of the cost).
- The `aperture_nside` window on 1-pt PDFs is **not** the 0.07 σ measured for ζ: PDF tails are
  more sensitive to the effective smoothing scale; measure before using (D4).
- The HEALPix-pixel top hat is not circular and varies over the sky: fine as a shared window
  in SBI, not comparable to circular-filter theory or to PDFast.
- The M_ap PDF from shear needs no mass mapping; κ PDFs need κ from outside the package.

## Open questions / risks (decide before or during the stage)

1. Aperture dipole (B2.8): the probe's measured scatter/bias table decides the recommended
   `aperture_nside` for production, not whether the dipole ships.
2. ~~Split search on the host or device~~ — moot, the split was removed.
3. ~~Row blocks with mixed levels at nside 2048~~ — moot, the split was removed.
4. 1-pt histogram weighting: decided, unit weight per accepted aperture (D2.4); the writer is
   `OnePointWriter` on the shared `ResultWriter` base (D2.5).
5. ~~`edge_slop` on patches and footprint~~ — both had it; removed.
6. T10 re-run with quad: if k = 2 quad passes, change the production recommendation (and the
   default factor behind `resolution_factor=True` / `"default"`).
