# `OnePoint` on an A100 — timing (stage D4, 2026-11-02)

`bench_onepoint.py`, one A100 80 GB (`usm-cl-seitz1/2`), DES Y3 mask at nside 1024
(`n_active` = 1.27e7 rows), 4 tomographic bins, float32 maps and rotations, device-resident
synthetic Gaussian maps, `min_coverage=0.8`, `cell_moments="quad"`, 100 histogram bins, median
of 5 CUDA-synchronised calls. Raw: `results/bench_1024.json` (job 17668831, the run that
includes the Σw pass; `slurm_op1024_17668551.log` is the earlier run without it). The real-GPU
gate of the same code is `results/gpu_gate_17668830.log` (cupy ≡ CPU, 80 checks, all pass).
Synthetic maps: this is the *time*; the window study on the PDF (real maps) is still to do.

## Centres and geometry (independent of `aperture_nside`)

| θ | `centre_nside` | footprint centres | accepted (`g`, 5 θ disc) | accepted (`k`, θ top hat) | disc pixels (`g`) | disc pixels (`k`) |
|---|---|---|---|---|---|---|
| 20′ | 256 | 94 390 | 83 207 | 87 117 | 2.12e8 | 9.1e6 |
| 45′ | 128 | 24 159 | 18 530 | 21 482 | 2.37e8 | 1.12e7 |
| 90′ | 64 | 6 229 | 3 956 | 5 295 | 1.99e8 | 1.09e7 |

The centre spacing scales with θ and the disc area with θ², so the `g` geometry is ~2e8 disc
entries *per scale whatever θ*: 16 B per entry of Q geometry plus 4 B for the unit-filter values
of the Σw pass — 4.8–5.7 GB on the device at full resolution (the benchmark's byte count double
counts the shared index array; the true figure is 20 B/entry), 1.2–1.5 GB with
`aperture_nside=512`, 0.31–0.37 GB with 256. The top hats are 25× smaller. Building the geometry
(`prepare`: coverage pass + `find_aperture_pairs`, two `query_disc` loops over the candidates,
single-threaded) takes 17–28 s per scale at full resolution, 8–17 s at 512, 3–12 s at 256; the
top hats 0.2–0.5 s. One-off per footprint and scale; not worth threading yet.

## Per map, per scale (ms), Σw pass included

| statistic | `aperture_nside` | 20′ | 45′ | 90′ |
|---|---|---|---|---|
| `g` (`M_ap` + `M_x`) | None | 21.9 | 20.3 | 19.7 |
| `g(cross=False)` | None | 13.0 | 12.1 | 12.3 |
| `g` | 512 | 39.9 | 40.7 | 35.4 |
| `g(cross=False)` | 512 | 24.9 | 23.9 | 21.0 |
| `g` | 256 | 25.0 | 19.6 | 18.5 |
| `g(cross=False)` | 256 | 13.9 | 13.8 | 11.1 |
| `k` top hat | None / 512 / 256 | 9.5 / 8.4 / 9.1 | 7.8 / 11.4 / 7.6 | 7.3 / 7.4 / 6.3 |
| `k` HEALPix pixel | None / 512 / 256 | 10.0 / 8.5 / 9.0 | 7.9 / 7.7 / 8.0 | 7.0 / 7.3 / 7.6 |

`values=True` costs nothing measurable (run-to-run noise is ±2 ms). Without the Σw pass (first
run) `g(cross=False)` was 10–11 ms and `k` 6–7 ms: the pass costs ~2 ms per scale at full
resolution — one more read of the 2e8-entry geometry.

## What the numbers say

- **A floor of ~6–7 ms per call** is the reduction, not the apertures: the top hats touch 1e7
  (or 1.4e6) disc entries and still take 7–10 ms. `_reduce_scale` is a few dozen small cupy
  launches per field (searchsorted + bincount per bin, five power sums, two GEMMs, the ijk
  reductions) plus the `.get()` syncs. For 10⁵ maps × 3 scales that floor is 30 min of GPU time
  per statistic — acceptable, and the obvious lever if it ever is not (one fused reduction
  kernel, or batching several maps' values before reducing).
- **One pass over the `g` geometry is 2–4 ms per scale at full resolution** (2e8 entries × 16–20 B
  + 12 B × nz map reads ≈ 400 GB/s). `g(cross=False)` ≈ 12–13 ms = floor + aperture pass + Σw
  pass; `g` ≈ 20–22 ms adds the `M_x` pass. Two levers, each ~2–4 ms per scale: a fused t/×
  kernel (one geometry read for both components) and a cached Σw mask (the weights rarely change
  between maps; a cache keyed on a frozen weights array would drop the pass and the 4 B/entry of
  unit-filter values). Worth it only once the reduction floor is gone.
- **`aperture_nside` buys memory, not time, with quad cells at nz = 4.** 512 is *slower* than full
  resolution (24 vs 12 ms without cross, 40 vs 21 with) and 256 only breaks even: the disc
  entries fall 4× / 16×, but every entry of a coarse aperture level reads the cell's moments —
  `NZ × 6 × 2` floats = 192 B at nz = 4 against 16 B of geometry — through
  `gpu_aperture_shear_tomo_mom`, plus the per-call degrade + moment build (`_expand_moments`) of
  the whole footprint. Same mechanism as the B5 register spill: the moment payload scales with
  nz. So for the PDF: full resolution when it fits (~5 GB per scale here), `aperture_nside` when
  it does not (nside 2048, or many scales at once), and `cell_moments=None` is the knob to test if
  the coarse pass must be fast — its price on the PDF is part of the window study.
- Expectation in the guide ("< 1–10 ms, upload-bound"): met for `k`, nearly for `g(cross=False)`;
  nothing here is upload-bound (device-resident maps), it is reduction- then geometry-bound.

Not measured here (needs the map archive): the `aperture_nside` and pixel-vs-circular top hat
windows on the PDF (moments and 1 % / 99 % quantiles vs full resolution), `cell_moments=None`
vs `"quad"` on the coarse aperture, and the PDFast cross-check.
