"""Streaming i3PCF reduction and HDF5 output.

A measurement loop normally keeps every per-patch result in RAM and writes
one array at the end.  At the DES Y3 production geometry (nside 512, 917
patches, 7 angular bins, 4 tomographic bins) one map-set is

    M_a (4, 917) + xi± (10, 917, 7) x 2  =  132,048 doubles  =  1.06 MB,

so a 10,000-realisation suite is ~10.6 GB held until the very end -- and
lost entirely if the run dies at realisation 9,500.

:class:`ZetaWriter` turns that into a stream.  ``zeta`` is an average over
*patches* (:func:`pixpoint.correlation_helpers.calculate_all_zetas`
averages over axis 2 and carries the map axis through untouched), so the
reduction of map-set *k* needs nothing but map-set *k*: it can run the
moment the measurement returns and the per-patch arrays can be dropped.
Reduced, the same map-set is 8 variants x (20 triplets, 7 bins) = **9 kB**,
118x smaller.

Two storage modes:

``reduce="zeta"``
    9 kB per map-set (90 MB for 10,000 realisations).  The patch axis is
    gone, so no jackknife and no re-binning later.
``reduce="none"``
    the raw 1.06 MB per map-set.  Everything -- zeta, a leave-one-patch-out
    jackknife, a different binning -- can still be derived from it.

(Storing *per-patch* zeta contributions is strictly worse than either:
8.2 MB per map-set, 8x the raw arrays, and it holds less information than
they do, since the jackknife is recoverable from the raw arrays exactly.)

Where the work happens
----------------------

``submit()`` runs on the calling thread and never leaves device memory
referenced by the queue -- anything on a GPU is brought to the host inside
``submit()``, after reducing on the device when that shrinks the transfer:

===========================  ==============================  ==================
input                         ``submit()`` (calling thread)   writer thread
===========================  ==============================  ==================
GPU + ``reduce="zeta"``       reduce on device, 9 kB D2H      HDF5 only
GPU + ``reduce="none"``       1.06 MB D2H                     HDF5 only
host (multi-GPU, CPU)         queue the arrays                reduce + HDF5
===========================  ==============================  ==================

:class:`~pixpoint.multi_device.MultiDeviceIntegrated3PCF` concatenates the
per-device outputs on the patch axis and always returns host arrays, so it
lands in the third row: the reduction runs on the CPU, in the writer
thread, off the measurement's critical path.  No device code is involved
and nothing about the group needs to change.

Both numpy and h5py release the GIL for the bulk of their work, so a plain
thread genuinely overlaps with the next map-set's kernels.
"""

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .correlation_helpers import calculate_all_zetas
from .result_writer import (  # noqa: F401  (re-exported for callers of 7.x)
    _CHUNK_BYTES,
    ResultWriter,
    _is_device_array,
    _json_default,
    _to_host,
)
from .results import IntegratedResult

# Central (aperture) and annular (2PCF) fields, in the order of
# results.INTEGRATED_FIELDS.
CENTRAL_FIELDS: Tuple[str, ...] = ("M_a", "M_g")
ANNULAR_FIELDS: Tuple[str, ...] = ("xi_p", "xi_m", "xi_g", "xi_t")

_PROVENANCE_ATTRS = ResultWriter._PROVENANCE_ATTRS


class ZetaWriter(ResultWriter):
    """Reduce each map-set to zeta (or keep it raw) and append it to HDF5.

    Args:
        path: Output HDF5 file.
        corr: The :class:`~pixpoint.correlations.Integrated3PCF` (or
            :class:`~pixpoint.multi_device.MultiDeviceIntegrated3PCF`) that
            produced the measurements.  Read once, for provenance.
        reduce: ``"zeta"`` (default) stores the eight i3PCF variants;
            ``"none"`` stores the raw per-patch arrays.
        flush_every: Flush to disk every this many map-sets.  A crash
            costs at most one window; ``n_flushed`` in the file records
            what is durable.
        depth: Queue depth.  ``submit()`` blocks once this many map-sets
            are waiting, which bounds memory and keeps the writer from
            falling behind unnoticed.
        swmr: Enable HDF5 single-writer/multiple-reader so another process
            can follow the file while it grows.  **Off by default, and it
            is not free** -- see the class notes below.
        resume: Open an existing file and continue appending after its
            last flushed map-set, discarding anything written past it.

    SWMR notes (measured with HDF5 1.14.6):
        * it forces ``libver="latest"`` (superblock v3), so the file needs
          HDF5 >= 1.10 to read at all;
        * readers must pass ``swmr=True``;
        * **after a crash the file keeps a stale write lock**: an ordinary
          ``h5py.File(path, "r")`` then fails with *"file is already open
          for write"* until someone runs ``h5clear -s <file>``.  A
          non-SWMR file written by this class survives the same crash and
          opens normally, because ``flush_every`` already makes the
          flushed prefix durable.
        * HDF5 documents SWMR as unreliable on NFS.

        So SWMR buys exactly one thing -- watching a run from another
        process -- and charges a worse crash story for it.  Leave it off
        unless you want the live view.

    The file layout is that of :class:`~pixpoint.result_writer.ResultWriter`
    with the group ``zeta`` or ``raw``; the per-estimator part is the
    reduction, which runs on the device inside ``submit()`` for device
    results and in the writer thread for host results.
    """

    def __init__(
        self,
        path: str,
        corr: Any,
        *,
        reduce: str = "zeta",
        flush_every: int = 50,
        depth: int = 2,
        swmr: bool = False,
        resume: bool = False,
    ) -> None:
        if reduce not in ("zeta", "none"):
            raise ValueError(f"reduce must be 'zeta' or 'none'; got {reduce!r}")
        self.reduce = reduce
        super().__init__(
            path,
            self.collect_provenance(corr),
            group="zeta" if reduce == "zeta" else "raw",
            flush_every=flush_every,
            depth=depth,
            swmr=swmr,
            resume=resume,
        )

    # ------------------------------------------------------------------ #
    # submission (calling thread)
    # ------------------------------------------------------------------ #

    def submit(self, result: IntegratedResult) -> None:  # type: ignore[override]
        """Queue one map-set.

        ``result`` is what ``Integrated3PCF.gg`` / ``kk`` / ``kg`` /
        ``three_by_two_pt`` return (or an :class:`IntegratedResult` built by
        hand), with no leading map axis.  It must hold at least one central
        field and one annular field.  Blocks when the queue is full.
        """
        if not isinstance(result, IntegratedResult):
            raise TypeError(
                "submit() takes an IntegratedResult, " f"not {type(result).__name__}"
            )
        self._raise_deferred()
        if self._closed:
            raise RuntimeError("ZetaWriter is closed")
        given = result.fields()
        if not any(k in given for k in CENTRAL_FIELDS):
            raise ValueError(f"submit() needs a central field ({CENTRAL_FIELDS})")
        if not any(k in given for k in ANNULAR_FIELDS):
            raise ValueError(f"submit() needs an annular field ({ANNULAR_FIELDS})")
        self._put(self._prepare(given))

    def _prepare(self, given: Mapping[str, Any]) -> Dict[str, Any]:
        """Add the map axis and get everything onto the host.

        Device arrays are reduced *before* the copy when that shrinks it,
        and nothing on a device is ever handed to the writer thread: the
        thread only ever touches host numpy and h5py, so no stream or
        buffer-lifetime hazard can arise.
        """
        batched = {k: v[None, ...] for k, v in given.items()}
        on_device = any(_is_device_array(v) for v in batched.values())

        if on_device and self.reduce == "zeta":
            zetas = calculate_all_zetas(**batched)
            self._require_zetas(zetas)
            return {"zeta": {k: _to_host(v) for k, v in zetas.items()}}
        if on_device:
            return {"raw": {k: _to_host(v) for k, v in batched.items()}}
        # Host input (multi-device group or CPU backend): defer the
        # reduction to the writer thread, off the critical path.
        return {"raw": {k: np.asarray(v) for k, v in batched.items()}}

    @staticmethod
    def _require_zetas(zetas: Mapping[str, Any]) -> None:
        if not zetas:
            raise ValueError(
                "the submitted fields do not form any zeta: pair a central "
                f"field {CENTRAL_FIELDS} with an annular one {ANNULAR_FIELDS}"
            )

    # ------------------------------------------------------------------ #
    # writer thread
    # ------------------------------------------------------------------ #

    def _payload(self, item: Dict[str, Any]) -> Mapping[str, np.ndarray]:
        if "raw" in item and self.reduce == "zeta":
            zetas = calculate_all_zetas(**item["raw"])
            self._require_zetas(zetas)
            return zetas
        return item.get("zeta", item.get("raw", {}))  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return (
            f"<ZetaWriter {self.path!r} reduce={self.reduce!r} "
            f"written={self._n_appended}>"
        )

    @staticmethod
    def _collect_provenance(corr: Any) -> Dict[str, Any]:
        return ResultWriter.collect_provenance(corr)
