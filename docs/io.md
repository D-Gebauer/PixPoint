# Files and streaming

## Pair files

Every estimator that finds pairs can save them and load them back:

```python
g2.preprocess()
g2.save_pairs("pairs.h5")
# ... another session, the same footprint and binning ...
g2.load_pairs("pairs.h5")
```

A pair file is HDF5. It records the footprint (via the mask's pixel list),
the resolution, the bins, the treecode level table and `cell_moments`, and
`load_pairs` refuses a file that does not match the estimator it is loaded
into, so a data vector cannot be measured with the wrong pairs by accident.
Files written by versions before the rename (`kind = "CosmoFuse.Global2PCF"`)
are still read. Patch pair files of `Integrated3PCF` can be sliced
(`load_pairs(path, start_ind, stop_ind)`) to load a range of patches, which is
how a multi-GPU group reads one file.

`pack_host_pairs=True` writes the 8-byte packed pairs instead of the exact
24-byte ones (a third of the size, faster to load; not convertible back to an
exact file).

## Row-space archives

For a simulation suite, store the maps restricted to the footprint:

```python
rows = g2.to_row_space(full_sky_maps)          # (nz, 2, n_active), ascending RING order of the unmasked pixels
np.save("maps_rows.npy", rows)
hash_ = g2.row_pix_hash                        # store next to the archive
```

`row_pix_hash` fingerprints `(nside, mask)`; the loaders compare it to the
archive's so maps written for another mask are rejected. Every method
accepts row-space input directly.

## Streaming results: `ResultWriter` and friends

A long run should not keep ten thousand results in memory and write them at
the end. The writers append each result to an HDF5 file from a background
thread, with the estimator's provenance in the file attributes:

```python
from pixpoint import ResultWriter, OnePointWriter, ZetaWriter, result_arrays

# two-point and three-point results: the sums per map
with ResultWriter("xi.h5", ResultWriter.collect_provenance(g2), group="gg",
                  constants={"theta": g2.bincenters}) as out:
    for shear in maps:
        out.submit(result_arrays(g2.gg(shear, w, flip_g1=True)))

# integrated 3PCF: reduced to zeta per map
with ZetaWriter("zeta.h5", corr, flush_every=50) as out:
    for shear in maps:
        out.submit(corr.gg(shear, w, flip_g1=True))

# one-point statistics: the histogram and moment sums per map
with OnePointWriter("pdf.h5", op) as out:
    for shear in maps:
        out.submit(op.g(shear, w, flip_g1=True))
```

What every writer guarantees:

- `submit()` returns as soon as the arrays are on the host and queued; the
  HDF5 work happens in a thread. It blocks only when `depth` results are
  waiting, so memory stays bounded and a writer that cannot keep up is
  noticed.
- `flush_every` results are flushed together; `n_flushed` in the file is the
  number of results guaranteed to be on disk. After a crash, reopen with
  `resume=True` to continue after that prefix.
- The file attributes hold the provenance: `nside`, the binning, the
  treecode `level_table`, `cell_moments`, the mask fingerprint, the PixPoint
  version. Datasets written once go under `constants/`.
- `swmr=True` lets another process read the file while it grows; it is off
  by default because a crashed SWMR file keeps a stale write lock.

Reading back is plain h5py:

```python
import h5py
with h5py.File("xi.h5") as f:
    num = f["gg/num/xi_p"][:]          # (nmaps, ncomb, nbins)
    weight = f["gg/weight"][:]
    xi_p = num / weight
    print(dict(f.attrs))
```

## Provenance

A data vector is only comparable to one measured with the same estimator:
the same mask, resolution, bins, `resolution_factor`, `cell_moments`,
`aperture_nside`, patch set or aperture centres. All of it is in
`estimator.level_table` / `estimator.provenance` and in the files the writers
produce; store it with everything you measure.
