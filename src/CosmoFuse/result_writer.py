"""Streaming HDF5 output shared by every estimator: :class:`ResultWriter`.

A measurement loop produces one small labelled result per map (a few kB to
a MB); keeping 10⁴–10⁵ of them in RAM until the end is wasteful and loses
everything if the run dies.  ``ResultWriter`` appends them to an HDF5 file
as they come, from a writer thread, so the measurement never waits for the
disk:

* ``submit(arrays)`` runs on the calling thread, takes a mapping of labelled
  arrays (host or device; device arrays are copied to the host *here*, so
  the writer thread never touches device memory), and queues them;
* the writer thread appends every array to a resizable dataset of the same
  name under one group, one row per map, chunked at ~1 MiB and at most one
  flush window;
* ``flush_every`` map-sets are made durable at a time: ``n_flushed`` in the
  file is the prefix a reader may trust, and ``resume=True`` reopens the
  file and continues after it, discarding anything written past it;
* the estimator's provenance (``nside``, ``level_table``, ...) goes into the
  file attributes once, ``constants`` (edges, scales, ...) into datasets
  written once, so a data vector cannot be silently combined with one
  measured under a different estimator.

Dataset names may contain ``/`` (``cross_sums/M_ap/ij``): h5py makes the
groups.  Both numpy and h5py release the GIL for the bulk of their work, so
the thread genuinely overlaps with the next map's kernels.

The subclasses add the per-estimator shape:
:class:`~CosmoFuse.zeta_writer.ZetaWriter` (``Integrated3PCF``: the zeta
reduction, or the raw per-patch arrays) and :class:`OnePointWriter`
(``OnePoint``: the histogram and moment sums).  A
:class:`~CosmoFuse.results.TwoPointResult` or
:class:`~CosmoFuse.results.ThreePointResult` needs no subclass -- submit
:func:`result_arrays` of it to a plain ``ResultWriter``.

SWMR (``swmr=True``) lets another process follow the file while it grows,
but a crashed SWMR file keeps a stale write lock (``h5clear -s``), which a
plain file does not: off by default.
"""

import json
import queue
import threading
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np

from .results import OnePointResult, ThreePointResult, TwoPointResult

# Target size of one HDF5 chunk along the map axis.
_CHUNK_BYTES = 1 << 20


def _is_device_array(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "cupy"


def _to_host(a: Any) -> np.ndarray:
    if _is_device_array(a):
        import cupy

        return cast(np.ndarray, cupy.asnumpy(a))
    return np.asarray(a)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return str(obj)


def result_arrays(result: Any) -> Dict[str, np.ndarray]:
    """The labelled host arrays of a global result, for a plain
    :class:`ResultWriter`: ``num/<name>`` and ``weight`` of a
    :class:`TwoPointResult`, ``num/<label>`` and ``norm/<label>`` of a
    :class:`ThreePointResult`."""
    if isinstance(result, TwoPointResult):
        out = {f"num/{k}": _to_host(v) for k, v in result.num.items()}
        out["weight"] = _to_host(result.weight)
        return out
    if isinstance(result, ThreePointResult):
        out = {f"num/{k}": _to_host(v) for k, v in result.num.items()}
        out.update({f"norm/{k}": _to_host(v) for k, v in result.norm.items()})
        return out
    raise TypeError(
        "result_arrays() takes a TwoPointResult or a ThreePointResult; "
        f"got {type(result).__name__}"
    )


class ResultWriter:
    """Append one labelled set of arrays per map to an HDF5 file.

    Args:
        path: Output HDF5 file.
        provenance: What produced the arrays (an estimator's ``provenance``
            or :meth:`collect_provenance`), written to the file attributes.
            Non-scalar values are stored as JSON strings.
        group: Name of the HDF5 group the per-map datasets live in.
        constants: Arrays written once (histogram edges, scales, ...), as
            datasets ``constants/<name>``.
        flush_every: Flush to disk every this many map-sets.
        depth: Queue depth; ``submit()`` blocks once this many map-sets are
            waiting, which bounds memory and keeps the writer from falling
            behind unnoticed.
        swmr: HDF5 single-writer/multiple-reader mode (see the module notes).
        resume: Open an existing file and continue after its last flushed
            map-set.
    """

    def __init__(
        self,
        path: str,
        provenance: Optional[Mapping[str, Any]] = None,
        *,
        group: str = "arrays",
        constants: Optional[Mapping[str, Any]] = None,
        flush_every: int = 50,
        depth: int = 2,
        swmr: bool = False,
        resume: bool = False,
    ) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.path = str(path)
        self.group_name = str(group)
        self.flush_every = int(flush_every)
        self.swmr = bool(swmr)
        self._resume = bool(resume)
        self._provenance: Dict[str, Any] = dict(provenance or {})
        self._constants: Dict[str, np.ndarray] = {
            k: _to_host(v) for k, v in (constants or {}).items()
        }

        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(
            maxsize=depth
        )
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._n_submitted = 0
        self._n_appended = 0
        self._closed = False

        self._file: Any = None
        self._group: Any = None
        self._datasets: Dict[str, Any] = {}
        self._n_flushed_ds: Any = None

        self._thread = threading.Thread(
            target=self._drain, name=type(self).__name__, daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # submission (calling thread)
    # ------------------------------------------------------------------ #

    def submit(self, arrays: Mapping[str, Any]) -> None:
        """Queue one map's arrays (host or device), each without a map axis.
        Every map must carry the same names and shapes.  Blocks when the
        queue is full."""
        if not isinstance(arrays, Mapping) or not arrays:
            raise TypeError("submit() takes a non-empty mapping of name -> array")
        self._put({"arrays": {k: _to_host(v)[None, ...] for k, v in arrays.items()}})

    def _put(self, item: Dict[str, Any]) -> None:
        self._raise_deferred()
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        self._queue.put(item)
        with self._lock:
            self._n_submitted += 1

    # ------------------------------------------------------------------ #
    # writer thread
    # ------------------------------------------------------------------ #

    def _payload(self, item: Dict[str, Any]) -> Mapping[str, np.ndarray]:
        """The arrays to append for one queued item (writer thread).  Each
        carries a leading map axis of length 1."""
        return item["arrays"]  # type: ignore[no-any-return]

    def _drain(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    self._consume(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # surfaced on the next submit()/close()
            with self._lock:
                self._error = exc
            self._drain_remaining()

    def _drain_remaining(self) -> None:
        """Keep the queue moving after a failure so submit() cannot wedge."""
        while True:
            item = self._queue.get()
            self._queue.task_done()
            if item is None:
                return

    def _consume(self, item: Dict[str, Any]) -> None:
        payload = self._payload(item)
        if self._file is None:
            self._open(payload)
        self._append(payload)
        self._n_appended += 1
        if self._n_appended % self.flush_every == 0:
            self._flush()

    def _open(self, payload: Mapping[str, np.ndarray]) -> None:
        import h5py

        libver = "latest" if self.swmr else None
        if self._resume:
            self._file = h5py.File(self.path, "a", libver=libver)
            self._group = self._file[self.group_name]
            self._n_flushed_ds = self._file["n_flushed"]
            self._datasets = {k: self._group[k] for k in payload}
            durable = int(self._n_flushed_ds[()])
            # discard anything written past the last flush: those rows may
            # be partial
            for ds in self._datasets.values():
                ds.resize(durable, axis=0)
            self._n_appended = durable
        else:
            self._file = h5py.File(self.path, "w", libver=libver)
            for key, value in self._provenance.items():
                self._file.attrs[key] = self._attr_value(value)
            for name, arr in self._constants.items():
                self._file.create_dataset(f"constants/{name}", data=arr)
            self._group = self._file.create_group(self.group_name)
            self._n_flushed_ds = self._file.create_dataset(
                "n_flushed", shape=(), dtype=np.int64
            )
            for name, arr in payload.items():
                self._datasets[name] = self._group.create_dataset(
                    name,
                    shape=(0,) + arr.shape[1:],
                    maxshape=(None,) + arr.shape[1:],
                    chunks=self._chunks(arr),
                    dtype=arr.dtype,
                )
        # SWMR must be enabled only once every object exists.
        if self.swmr:
            self._file.swmr_mode = True

    @staticmethod
    def _attr_value(value: Any) -> Any:
        if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
            return value
        if isinstance(value, np.ndarray) and value.ndim <= 1:
            return value
        return json.dumps(value, default=_json_default, sort_keys=True)

    def _chunks(self, arr: np.ndarray) -> Tuple[int, ...]:
        """Chunk length in map-sets, capped by the flush window.

        A chunk is the unit HDF5 reads, modifies and writes.  Sizing it at
        ~1 MiB regardless of ``flush_every`` means a 50-map flush touches a
        chunk that spans many more than 50 maps, so every flush rewrites the
        whole partially-filled chunk -- write amplification proportional to
        the ratio, and a file that stays padded to the chunk boundary.
        Capping it at the flush window makes each flush write whole chunks.
        """
        per_map = int(arr.dtype.itemsize * np.prod(arr.shape[1:], dtype=np.int64))
        n = max(1, _CHUNK_BYTES // max(per_map, 1))
        n = min(n, max(1, int(self.flush_every)))
        return (int(n),) + arr.shape[1:]

    def _append(self, payload: Mapping[str, np.ndarray]) -> None:
        missing = set(self._datasets) ^ set(payload)
        if missing:
            raise ValueError(
                "every map-set must carry the same fields; this one differs in "
                f"{sorted(missing)}"
            )
        row = self._n_appended
        for name, arr in payload.items():
            ds = self._datasets[name]
            if arr.shape[1:] != ds.shape[1:]:
                raise ValueError(
                    f"{name} has shape {arr.shape[1:]} but the file holds "
                    f"{ds.shape[1:]}"
                )
            ds.resize(row + 1, axis=0)
            ds[row] = arr[0]

    def _flush(self) -> None:
        self._n_flushed_ds[()] = self._n_appended
        self._file.flush()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _raise_deferred(self) -> None:
        with self._lock:
            exc = self._error
        if exc is not None:
            raise RuntimeError(f"the {type(self).__name__} thread failed") from exc

    @property
    def n_submitted(self) -> int:
        """Map-sets handed to :meth:`submit`."""
        with self._lock:
            return self._n_submitted

    @property
    def n_written(self) -> int:
        """Map-sets appended to the file (flushed or not)."""
        return self._n_appended

    def close(self) -> None:
        """Drain the queue, flush and close.  Idempotent."""
        if self._closed:
            self._raise_deferred()
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        try:
            if self._file is not None:
                try:
                    if self._error is None:
                        self._flush()
                finally:
                    self._file.close()
                    self._file = None
        finally:
            self._raise_deferred()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n_appended

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.path!r} group={self.group_name!r} "
            f"written={self._n_appended}>"
        )

    # ------------------------------------------------------------------ #

    # Attributes of a pair estimator that tell two data vectors apart.
    _PROVENANCE_ATTRS = (
        "nside",
        "nbins",
        "theta_min",
        "theta_max",
        "patch_size",
        "theta_Q",
        "n_patches",
        "resolution_factor",
        "cell_moments",
        "aperture_nside",
        "pack_pairs",
        "pack_host_pairs",
        "pair_search_precision",
    )

    @classmethod
    def collect_provenance(cls, estimator: Any) -> Dict[str, Any]:
        """Everything needed to tell two data vectors apart.

        A treecode measurement is only comparable to another made with the
        same level table, and a row-space archive only to the mask it was
        built for -- so both go in the file.  An estimator with a
        ``provenance`` property (``OnePoint``) contributes that.
        """
        from . import __version__

        out: Dict[str, Any] = {"cosmofuse_version": __version__}
        own = getattr(type(estimator), "provenance", None)
        if isinstance(own, property):
            out.update(estimator.provenance)
            return out
        for name in cls._PROVENANCE_ATTRS:
            value = getattr(estimator, name, None)
            if value is None:
                continue
            out[name] = value if not isinstance(value, bool) else bool(value)
        for name in ("level_table", "row_pix_hash"):
            try:
                value = getattr(estimator, name)
            except Exception:
                continue
            out[name] = (
                value
                if isinstance(value, str)
                else json.dumps(value, default=_json_default, sort_keys=True)
            )
        return out


class OnePointWriter(ResultWriter):
    """Stream :class:`~CosmoFuse.results.OnePointResult` objects to HDF5.

    One row per map under the group ``onepoint``: ``n_apertures``, and per
    field ``count/<name>``, ``power_sums/<name>``, ``cross_sums/<name>/<key>``
    (non-empty keys), with edges also ``hist/<name>``, ``underflow/<name>``,
    ``overflow/<name>``, and with ``values=True`` results ``values/<name>/<s>``
    per scale.  Written once: ``constants/scales``, ``constants/edges``, the
    estimator's :attr:`~CosmoFuse.onepoint.OnePoint.provenance`, ``names``,
    ``statistic`` and ``cross_index`` (JSON) as attributes.  The first result
    fixes the layout; every later one must match it.
    """

    def __init__(
        self,
        path: str,
        estimator: Any,
        *,
        flush_every: int = 50,
        depth: int = 2,
        swmr: bool = False,
        resume: bool = False,
    ) -> None:
        super().__init__(
            path,
            self.collect_provenance(estimator),
            group="onepoint",
            flush_every=flush_every,
            depth=depth,
            swmr=swmr,
            resume=resume,
        )
        self._layout_fixed = False

    def submit(self, result: Any) -> None:
        """Queue one :class:`OnePointResult` (host or device arrays)."""
        if not isinstance(result, OnePointResult):
            raise TypeError(
                "OnePointWriter.submit() takes an OnePointResult, "
                f"not {type(result).__name__}"
            )
        if not self._layout_fixed:
            self._constants["scales"] = np.asarray(result.scales, dtype=np.float64)
            if result.edges is not None:
                self._constants["edges"] = np.asarray(result.edges, dtype=np.float64)
            self._provenance["statistic"] = result.statistic
            self._provenance["names"] = json.dumps(list(result.names))
            self._provenance["cross_index"] = json.dumps(
                {k: [list(t) for t in v] for k, v in result.cross_index.items()}
            )
            self._layout_fixed = True
        self._put({"arrays": {k: v[None, ...] for k, v in self.arrays(result).items()}})

    @staticmethod
    def arrays(result: OnePointResult) -> Dict[str, np.ndarray]:
        """The labelled host arrays of one result (no map axis)."""
        out: Dict[str, np.ndarray] = {
            "n_apertures": np.asarray(result.n_apertures, dtype=np.int64)
        }
        for name in result.names:
            out[f"count/{name}"] = np.asarray(result.count[name], dtype=np.float64)
            out[f"power_sums/{name}"] = _to_host(result.power_sums[name])
            for key, arr in result.cross_sums[name].items():
                if np.asarray(arr).size:
                    out[f"cross_sums/{name}/{key}"] = _to_host(arr)
            if result.hist is not None:
                out[f"hist/{name}"] = _to_host(result.hist[name])
                assert result.underflow is not None and result.overflow is not None
                out[f"underflow/{name}"] = _to_host(result.underflow[name])
                out[f"overflow/{name}"] = _to_host(result.overflow[name])
            if result.values is not None:
                for s, arr in enumerate(result.values[name]):
                    out[f"values/{name}/{s}"] = _to_host(arr)
        return out
