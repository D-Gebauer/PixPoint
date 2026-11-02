"""
Edge-splitting pair search of the static treecode (``edge_slop``).

The static search of :mod:`CosmoFuse.pair_geometry` /
:mod:`CosmoFuse.global_geometry` pairs the cells of *one* level per angular
bin and bins every cell pair by the separation of the two centroids.  A
child pixel pair whose own separation lies on the other side of a bin edge
is then counted in the wrong bin: a deterministic *binning window* of
+-2 % at k = 4 that alternates in sign at the level seams
(``benchmarks/moments/RESULTS.md``).

This module replaces that per-level search by a top-down recursion over
*cell pairs of any two levels* (TreeCorr's ``bin_slop``):

* candidates are the unordered pairs of the coarsest level's cells (self
  pairs included) whose separation range ``[d - s_I - s_J, d + s_I + s_J]``
  reaches the theta range, where ``d`` is the centroid separation and
  ``s_I`` the cell *radius* -- the largest angle between the centroid and a
  member pixel centre, so the range bounds every child pixel pair exactly
  (triangle inequality);
* a cell pair is **accepted** into its *home bin* (the bin of ``d``) when
  both cells are no coarser than that bin's static level (``level_nside``,
  the ``resolution_factor`` table) and the range overhangs the bin's edges
  by at most ``edge_slop`` bin widths; otherwise the cell with the larger
  radius is **split** into its direct children (both on a tie, which keeps
  self pairs to ``c_I <= c_J``), a cell always before a pixel;
* pixel pairs are leaves and are binned **exactly**, with the pair finder's
  rule (strict ``cos`` comparison), so at ``edge_slop = 0`` the sum of
  ``w_i w_j`` over the pairs of a bin equals the full-resolution one.

Every pixel pair in range is covered by exactly one accepted pair (the
recursion is a tree whose leaves partition the pixel pairs), so the
estimator is still an exact regrouping of the full-resolution sums -- what
changes against the static list is only *which* pairs share a geometry.
One bin then holds pairs of several levels (pixel-pixel, pixel-cell,
cell-cell); the row space addresses every level already, and the moment
kernels treat a pixel end as a cell without moments.

Two drivers share the recursion: :func:`split_patch_pairs` for one patch of
:class:`~CosmoFuse.correlations.Integrated3PCF` (its own cells) and
:func:`split_tile_pairs` for one tile of
:class:`~CosmoFuse.global_2pcf.Global2PCF` (the footprint's cells; a top
pair is owned by the tile of its lower global id, so the tiles partition the
same recursion tree the one-patch search walks).  The kernels are Numba
(``parallel=True``, no fastmath: the partition must not depend on the
machine); the search runs at float64 whatever ``pair_search_precision``
says, which only ever mattered for throwaway runs.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import healpy as hp
import numpy as np

from ._numba import njit, prange
from .global_geometry import _angle_terms_jit, _bin_of_jit, _exp2_jit
from .moments import cell_frames
from .treecode import PatchCells, TreecodeGeometry
from .utils import pixel2RaDec

#: Frontier entries (pairs under consideration) a recursion step may hold
#: before the split entries are expanded in chunks: bounds the memory of the
#: breadth-first walk (two int64 per entry) independent of the footprint.
MAX_FRONTIER = 1 << 24


# ── node tables ───────────────────────────────────────────────────────────


@dataclass
class SplitNodes:
    """The elements a split search recurses over, in local indices: the
    pixels of the group first (a patch's disc, or the whole row space), then
    its treecode cells level by level (finest first).  Local index order is
    global-id order, so ``gid[ii] < gid[jj]`` iff ``ii < jj``."""

    gid: np.ndarray  # (n,) int64 global ids (RING pixel, or npix + virtual row)
    geom: np.ndarray  # (n, 9) float64: unit vector, east, north (cell_frames)
    radius: np.ndarray  # (n,) float64 cell radius, 0 for a pixel
    nside: np.ndarray  # (n,) int64 nside of the element's level
    child_indptr: np.ndarray  # (n + 1,) int64 direct children (local indices)
    child_indices: np.ndarray
    top: np.ndarray  # (n_top,) local indices of the coarsest level's elements
    base_nside: int

    @property
    def xyz(self) -> np.ndarray:
        return self.geom[:, 0:3]

    @classmethod
    def from_patch(
        cls,
        nside: int,
        pix_ring: np.ndarray,
        cells: PatchCells,
        coarse_nsides: Sequence[int],
        cell_gid_start: Sequence[int],
    ) -> "SplitNodes":
        """One patch: its unmasked pixels (RING ids ``pix_ring``, ascending)
        and its :class:`~CosmoFuse.treecode.PatchCells`; the cells of level
        ``L`` get global ids ``cell_gid_start[L] + local``."""
        pix_ring = np.asarray(pix_ring, dtype=np.int64)
        n_pix = int(pix_ring.size)
        ra, dec = pixel2RaDec(pix_ring, int(nside))
        geoms = [cell_frames(ra, dec)]
        gids = [pix_ring]
        radii = [np.zeros(n_pix)]
        nsides = [np.full(n_pix, int(nside), dtype=np.int64)]
        indptr_parts = [np.zeros(n_pix + 1, dtype=np.int64)]
        indices_parts: List[np.ndarray] = []
        nnz = 0
        start = n_pix
        level_start = []
        order = np.argsort(pix_ring, kind="stable")
        pix_sorted = pix_ring[order]
        for level, ns in enumerate(coarse_nsides):
            n_c = int(cells.n_cells[level])
            level_start.append(start)
            geoms.append(cell_frames(cells.ra[level], cells.dec[level]))
            gids.append(int(cell_gid_start[level]) + np.arange(n_c, dtype=np.int64))
            radii.append(np.asarray(cells.radius[level], dtype=np.float64))
            nsides.append(np.full(n_c, int(ns), dtype=np.int64))
            idx = np.asarray(cells.child_indices[level], dtype=np.int64)
            if level == 0:
                pos = np.searchsorted(pix_sorted, idx)
                local = order[np.minimum(pos, n_pix - 1)] if n_pix else pos
                if idx.size and not np.array_equal(pix_ring[local], idx):
                    raise ValueError(
                        "treecode cells reference pixels outside the patch"
                    )
            else:
                local = idx + level_start[level - 1]
            ptr = np.asarray(cells.child_indptr[level], dtype=np.int64)
            indptr_parts.append(ptr[1:] + nnz)
            indices_parts.append(local.astype(np.int64))
            nnz += int(local.size)
            start += n_c
        n_top = int(cells.n_cells[-1]) if coarse_nsides else n_pix
        top_start = level_start[-1] if coarse_nsides else 0
        return cls(
            gid=np.concatenate(gids),
            geom=np.ascontiguousarray(np.concatenate(geoms, axis=0)),
            radius=np.concatenate(radii),
            nside=np.concatenate(nsides),
            child_indptr=np.concatenate(indptr_parts),
            child_indices=(
                np.concatenate(indices_parts)
                if indices_parts
                else np.zeros(0, dtype=np.int64)
            ),
            top=top_start + np.arange(n_top, dtype=np.int64),
            base_nside=int(nside),
        )

    @classmethod
    def from_rows(
        cls,
        nside: int,
        row_pix: np.ndarray,
        treecode: TreecodeGeometry,
        n_aperture_cells: int,
    ) -> "SplitNodes":
        """The whole row space: pixel rows then the treecode cells of every
        level, so that a local index is the device row (cells at ``n_active
        + level_start + index``) and the global ids are the ones the pair
        lists carry."""
        row_pix = np.asarray(row_pix, dtype=np.int64)
        n_active = int(row_pix.size)
        npix = int(hp.nside2npix(int(nside)))
        ra, dec = pixel2RaDec(row_pix, int(nside))
        geoms = [cell_frames(ra, dec)]
        gids = [row_pix]
        radii = [np.zeros(n_active)]
        nsides = [np.full(n_active, int(nside), dtype=np.int64)]
        indptr_parts = [np.zeros(n_active + 1, dtype=np.int64)]
        indices_parts: List[np.ndarray] = []
        lut = np.full(npix, -1, dtype=np.int64)
        lut[row_pix] = np.arange(n_active)
        starts = treecode.level_starts(first=int(n_aperture_cells))
        nnz = 0
        for level in range(treecode.n_levels):
            n_c = int(treecode.cells_per_level[level])
            geoms.append(cell_frames(treecode.cell_ra[level], treecode.cell_dec[level]))
            gids.append(npix + starts[level] + np.arange(n_c, dtype=np.int64))
            radii.append(treecode.radius(level))
            nsides.append(
                np.full(n_c, int(treecode.coarse_nsides[level]), dtype=np.int64)
            )
            idx = np.asarray(treecode.child_indices[level], dtype=np.int64)
            if level == 0:
                local = lut[idx]
                if local.size and int(local.min()) < 0:
                    raise ValueError("treecode cells reference pixels outside the mask")
            else:
                local = n_active + int(starts[level - 1]) + idx
            ptr = np.asarray(treecode.child_indptr[level], dtype=np.int64)
            indptr_parts.append(ptr[1:] + nnz)
            indices_parts.append(local.astype(np.int64))
            nnz += int(local.size)
        if treecode.n_levels:
            top_start = n_active + int(starts[treecode.n_levels - 1])
            n_top = int(treecode.cells_per_level[treecode.n_levels - 1])
        else:
            top_start, n_top = 0, n_active
        return cls(
            gid=np.concatenate(gids),
            geom=np.ascontiguousarray(np.concatenate(geoms, axis=0)),
            radius=np.concatenate(radii),
            nside=np.concatenate(nsides),
            child_indptr=np.concatenate(indptr_parts),
            child_indices=(
                np.concatenate(indices_parts)
                if indices_parts
                else np.zeros(0, dtype=np.int64)
            ),
            top=top_start + np.arange(n_top, dtype=np.int64),
            base_nside=int(nside),
        )

    def local_of_gid(self, gids: np.ndarray, npix: int, n_active: int) -> np.ndarray:
        """Local indices of global ids for a :meth:`from_rows` table
        (pixel ids through the row lookup, virtual rows by offset)."""
        gids = np.asarray(gids, dtype=np.int64)
        out = np.empty(gids.shape, dtype=np.int64)
        pix = gids < npix
        if np.any(pix):
            pos = np.searchsorted(self.gid[:n_active], gids[pix])
            pos = np.minimum(pos, max(n_active - 1, 0))
            if n_active == 0 or not np.array_equal(self.gid[:n_active][pos], gids[pix]):
                raise ValueError("pair ids reference pixels outside the row space")
            out[pix] = pos
        out[~pix] = n_active + (gids[~pix] - npix)
        return out


# ── kernels ───────────────────────────────────────────────────────────────


@njit(cache=True, parallel=True)
def _split_step(
    ii: np.ndarray,
    jj: np.ndarray,
    xyz: np.ndarray,
    radius: np.ndarray,
    nside: np.ndarray,
    base_nside: int,
    cos_edges: np.ndarray,
    edges: np.ndarray,
    level_nside: np.ndarray,
    slop: float,
    child_indptr: np.ndarray,
    code: np.ndarray,
    nchild: np.ndarray,
) -> None:
    """Classify every frontier pair: ``code[p] = b >= 0`` accepted into bin
    ``b``; ``-1`` dropped (out of range); ``-2 - mode`` split, ``mode`` 0 =
    children of ii, 1 = children of jj, 2 = both, with ``nchild[p]`` children
    pairs.  Pixel pairs use the pair finder's binning rule."""
    n = ii.size
    nb = edges.size - 1
    e0 = edges[0]
    e1 = edges[nb]
    for p in prange(n):
        a = ii[p]
        b = jj[p]
        xa0 = xyz[a, 0]
        xa1 = xyz[a, 1]
        xa2 = xyz[a, 2]
        xb0 = xyz[b, 0]
        xb1 = xyz[b, 1]
        xb2 = xyz[b, 2]
        dot = xa0 * xb0 + xa1 * xb1 + xa2 * xb2
        cell_a = nside[a] < base_nside
        cell_b = nside[b] < base_nside
        nchild[p] = 0
        if not cell_a and not cell_b:
            code[p] = _bin_of_jit(dot, cos_edges)
            continue
        cx = xa1 * xb2 - xa2 * xb1
        cy = xa2 * xb0 - xa0 * xb2
        cz = xa0 * xb1 - xa1 * xb0
        d = math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot)
        sa = radius[a]
        sb = radius[b]
        lo = d - sa - sb
        hi = d + sa + sb
        if hi <= e0 or lo >= e1:
            code[p] = -1
            continue
        home = -1
        if d > e0 and d < e1:
            lo_k = 0
            hi_k = nb + 1
            while lo_k < hi_k:
                mid = (lo_k + hi_k) >> 1
                if edges[mid] < d:
                    lo_k = mid + 1
                else:
                    hi_k = mid
            home = lo_k - 1
            if home < 0 or home >= nb or not (d < edges[home + 1]):
                home = -1
        accept = False
        if home >= 0:
            if nside[a] >= level_nside[home] and nside[b] >= level_nside[home]:
                width = edges[home + 1] - edges[home]
                over = edges[home] - lo
                if hi - edges[home + 1] > over:
                    over = hi - edges[home + 1]
                if over < 0.0:
                    over = 0.0
                accept = over <= slop * width
        if accept:
            code[p] = home
            continue
        if cell_a and cell_b:
            if sa > sb:
                mode = 0
            elif sb > sa:
                mode = 1
            else:
                mode = 2
        elif cell_a:
            mode = 0
        else:
            mode = 1
        na = child_indptr[a + 1] - child_indptr[a]
        nbb = child_indptr[b + 1] - child_indptr[b]
        if mode == 0:
            nchild[p] = na
        elif mode == 1:
            nchild[p] = nbb
        elif a == b:
            nchild[p] = na * (na + 1) // 2
        else:
            nchild[p] = na * nbb
        code[p] = -2 - mode


@njit(cache=True, parallel=True)
def _split_expand(
    ii: np.ndarray,
    jj: np.ndarray,
    code: np.ndarray,
    offsets: np.ndarray,
    child_indptr: np.ndarray,
    child_indices: np.ndarray,
    out_I: np.ndarray,
    out_J: np.ndarray,
) -> None:
    """Children pairs of the split entries (``code <= -2``) at ``offsets``."""
    n = ii.size
    for p in prange(n):
        c = code[p]
        if c > -2:
            continue
        mode = -2 - c
        a = ii[p]
        b = jj[p]
        o = offsets[p]
        a0 = child_indptr[a]
        a1 = child_indptr[a + 1]
        b0 = child_indptr[b]
        b1 = child_indptr[b + 1]
        if mode == 0:
            for k in range(a0, a1):
                out_I[o] = child_indices[k]
                out_J[o] = b
                o += 1
        elif mode == 1:
            for k in range(b0, b1):
                out_I[o] = a
                out_J[o] = child_indices[k]
                o += 1
        elif a == b:
            for u in range(a0, a1):
                cu = child_indices[u]
                for v in range(u, a1):
                    out_I[o] = cu
                    out_J[o] = child_indices[v]
                    o += 1
        else:
            for u in range(a0, a1):
                cu = child_indices[u]
                for v in range(b0, b1):
                    out_I[o] = cu
                    out_J[o] = child_indices[v]
                    o += 1


@njit(cache=True, parallel=True)
def _pair_geometry(
    ii: np.ndarray,
    jj: np.ndarray,
    geom: np.ndarray,
    c1: np.ndarray,
    s1: np.ndarray,
    c2: np.ndarray,
    s2: np.ndarray,
    n_moments: int,
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    inv_d: np.ndarray,
) -> None:
    """Rotation factors ``e^{2i phi}`` at both ends (the pair finder's
    arithmetic) for every pair, and for the first ``n_moments`` pairs the
    full position angles and the inverse separation (``moments.pair_angles``
    arithmetic) that the moment kernels read."""
    n = ii.size
    for p in prange(n):
        a = ii[p]
        b = jj[p]
        x1 = geom[a, 0]
        y1 = geom[a, 1]
        z1 = geom[a, 2]
        x2 = geom[b, 0]
        y2 = geom[b, 1]
        z2 = geom[b, 2]
        sinC1, cosC1, sinC2, cosC2 = _angle_terms_jit(x1, y1, z1, x2, y2, z2)
        a_re, a_im = _exp2_jit(sinC1, cosC1)
        b_re, b_im = _exp2_jit(sinC2, cosC2)
        c1[p] = a_re
        s1[p] = a_im
        c2[p] = b_re
        s2[p] = b_im
        if p < n_moments:
            ea0 = geom[a, 3]
            ea1 = geom[a, 4]
            ea2 = geom[a, 5]
            na0 = geom[a, 6]
            na1 = geom[a, 7]
            na2 = geom[a, 8]
            eb0 = geom[b, 3]
            eb1 = geom[b, 4]
            eb2 = geom[b, 5]
            nb0 = geom[b, 6]
            nb1 = geom[b, 7]
            nb2 = geom[b, 8]
            phi_a[p] = math.atan2(
                na0 * x2 + na1 * y2 + na2 * z2, ea0 * x2 + ea1 * y2 + ea2 * z2
            )
            phi_b[p] = math.atan2(
                nb0 * x1 + nb1 * y1 + nb2 * z1, eb0 * x1 + eb1 * y1 + eb2 * z1
            )
            cx = y1 * z2 - z1 * y2
            cy = z1 * x2 - x1 * z2
            cz = x1 * y2 - y1 * x2
            dot = x1 * x2 + y1 * y2 + z1 * z2
            d = math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot)
            inv_d[p] = 1.0 / d if d > 0.0 else 0.0


# ── the recursion ─────────────────────────────────────────────────────────


class _Recursion:
    """Breadth-first walk of the split tree from a set of top pairs; the
    accepted pairs are collected per bin (local indices)."""

    def __init__(
        self,
        nodes: SplitNodes,
        binedges: np.ndarray,
        level_nside: np.ndarray,
        slop: float,
        max_frontier: int,
    ) -> None:
        self.nodes = nodes
        self.edges = np.ascontiguousarray(np.asarray(binedges, dtype=np.float64))
        self.cos_edges = np.cos(self.edges)
        self.level_nside = np.ascontiguousarray(np.asarray(level_nside, dtype=np.int64))
        self.slop = float(slop)
        self.max_frontier = max(int(max_frontier), 1)
        self.nbins = int(self.edges.size - 1)
        self.out_I: List[List[np.ndarray]] = [[] for _ in range(self.nbins)]
        self.out_J: List[List[np.ndarray]] = [[] for _ in range(self.nbins)]
        self.visited = 0
        self.xyz = np.ascontiguousarray(nodes.geom[:, 0:3])

    def _emit(self, ii: np.ndarray, jj: np.ndarray, bins: np.ndarray) -> None:
        # radix sort on the small-integer keys: O(n), stable
        order = np.argsort(bins.astype(np.int16), kind="stable")
        bounds = np.searchsorted(bins[order], np.arange(self.nbins + 1))
        for b in range(self.nbins):
            a0, a1 = int(bounds[b]), int(bounds[b + 1])
            if a1 > a0:
                sel = order[a0:a1]
                self.out_I[b].append(ii[sel])
                self.out_J[b].append(jj[sel])

    def run(self, ii: np.ndarray, jj: np.ndarray) -> None:
        ii = np.ascontiguousarray(ii, dtype=np.int64)
        jj = np.ascontiguousarray(jj, dtype=np.int64)
        nodes = self.nodes
        while ii.size:
            self.visited += int(ii.size)
            code = np.empty(ii.size, dtype=np.int64)
            nchild = np.empty(ii.size, dtype=np.int64)
            _split_step(
                ii,
                jj,
                self.xyz,
                nodes.radius,
                nodes.nside,
                int(nodes.base_nside),
                self.cos_edges,
                self.edges,
                self.level_nside,
                self.slop,
                nodes.child_indptr,
                code,
                nchild,
            )
            acc = code >= 0
            if np.any(acc):
                self._emit(ii[acc], jj[acc], code[acc])
            spl = code <= -2
            if not np.any(spl):
                return
            ii, jj, code, nchild = ii[spl], jj[spl], code[spl], nchild[spl]
            csum = np.cumsum(nchild)
            total = int(csum[-1])
            if total > self.max_frontier and ii.size > 1:
                # expand in chunks that fit, each walked to its leaves first
                start = 0
                while start < ii.size:
                    base = int(csum[start - 1]) if start else 0
                    stop = int(np.searchsorted(csum, base + self.max_frontier, "right"))
                    stop = max(stop, start + 1)
                    sl = slice(start, stop)
                    cI, cJ = self._expand(ii[sl], jj[sl], code[sl], nchild[sl])
                    self.run(cI, cJ)
                    start = stop
                return
            ii, jj = self._expand(ii, jj, code, nchild)

    def _expand(
        self, ii: np.ndarray, jj: np.ndarray, code: np.ndarray, nchild: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        offsets = np.zeros(ii.size, dtype=np.int64)
        if ii.size > 1:
            np.cumsum(nchild[:-1], out=offsets[1:])
        total = int(offsets[-1] + nchild[-1]) if ii.size else 0
        out_I = np.empty(total, dtype=np.int64)
        out_J = np.empty(total, dtype=np.int64)
        _split_expand(
            ii,
            jj,
            code,
            offsets,
            self.nodes.child_indptr,
            self.nodes.child_indices,
            out_I,
            out_J,
        )
        return out_I, out_J

    def collect(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Accepted pairs grouped by bin, ``ii <= jj`` (local indices), and the
        per-bin counts."""
        counts = np.array(
            [sum(int(a.size) for a in v) for v in self.out_I], dtype=np.int64
        )
        ii = np.concatenate(
            [a for v in self.out_I for a in v] or [np.zeros(0, dtype=np.int64)]
        )
        jj = np.concatenate(
            [a for v in self.out_J for a in v] or [np.zeros(0, dtype=np.int64)]
        )
        lo = np.minimum(ii, jj)
        hi = np.maximum(ii, jj)
        return lo, hi, counts


def split_pairs(
    nodes: SplitNodes,
    top_I: np.ndarray,
    top_J: np.ndarray,
    binedges: np.ndarray,
    level_nside: np.ndarray,
    edge_slop: float,
    first_coarse_bin: int,
    want_moments: bool,
    max_frontier: int = MAX_FRONTIER,
) -> Dict[str, np.ndarray]:
    """Walk the split tree from the top pairs ``(top_I, top_J)`` (local
    indices of coarsest-level elements) and return the accepted pairs.

    Returns ``ids`` ``(2, n)`` global ids (lower id first), ``exp2phi``
    ``(2, n)`` complex128 rotation factors, ``counts`` ``(nbins,)`` pairs
    per bin (the pairs are grouped by bin), and -- for the pairs of the
    bins from ``first_coarse_bin`` on, a contiguous tail -- ``phi`` ``(2,
    n_tail)`` and ``inv_d`` ``(n_tail,)`` when ``want_moments`` (empty
    otherwise), plus ``visited`` (pairs the recursion looked at).
    """
    rec = _Recursion(nodes, binedges, level_nside, edge_slop, max_frontier)
    rec.run(np.asarray(top_I, dtype=np.int64), np.asarray(top_J, dtype=np.int64))
    ii, jj, counts = rec.collect()
    n = int(ii.size)
    c1 = np.empty(n)
    s1 = np.empty(n)
    c2 = np.empty(n)
    s2 = np.empty(n)
    n_fine = int(np.sum(counts[: int(first_coarse_bin)]))
    n_tail = n - n_fine if want_moments else 0
    phi_a = np.empty(n_tail)
    phi_b = np.empty(n_tail)
    inv_d = np.empty(n_tail)
    if n:
        # the geometry kernel fills the moment arrays for the first n_moments
        # entries it sees: pass the tail first so those are the coarse pairs
        order = np.concatenate((np.arange(n_fine, n), np.arange(n_fine)))
        Io = ii[order]
        Jo = jj[order]
        c1o = np.empty(n)
        s1o = np.empty(n)
        c2o = np.empty(n)
        s2o = np.empty(n)
        _pair_geometry(
            Io, Jo, nodes.geom, c1o, s1o, c2o, s2o, n_tail, phi_a, phi_b, inv_d
        )
        c1[order] = c1o
        s1[order] = s1o
        c2[order] = c2o
        s2[order] = s2o
    exp2phi = np.empty((2, n), dtype=np.complex128)
    exp2phi[0].real = c1
    exp2phi[0].imag = s1
    exp2phi[1].real = c2
    exp2phi[1].imag = s2
    return {
        "ids": np.stack((nodes.gid[ii], nodes.gid[jj])),
        "exp2phi": exp2phi,
        "counts": counts,
        "phi": np.stack((phi_a, phi_b)),
        "inv_d": inv_d,
        "visited": np.array(rec.visited, dtype=np.int64),
    }


def top_pairs_within(
    nodes: SplitNodes,
    core: np.ndarray,
    cand: np.ndarray,
    theta_max: float,
    ordered: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-level candidate pairs ``(ii, jj)``, ``ii`` from ``core`` and ``jj``
    from ``cand`` (local indices of coarsest-level elements), whose range can
    reach ``theta_max``: ``d - s_I - s_J < theta_max``.  With ``ordered``
    only pairs with ``ii <= jj`` (every unordered pair once, self pairs
    included), which is the ownership rule of both drivers."""
    core = np.asarray(core, dtype=np.int64)
    cand = np.asarray(cand, dtype=np.int64)
    if core.size == 0 or cand.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    xyz = nodes.xyz
    ii = np.repeat(core, cand.size)
    jj = np.tile(cand, core.size)
    if ordered:
        keep = ii <= jj
        ii, jj = ii[keep], jj[keep]
    xa, xb = xyz[ii], xyz[jj]
    cross = np.cross(xa, xb)
    d = np.arctan2(
        np.sqrt(np.einsum("ij,ij->i", cross, cross)), np.einsum("ij,ij->i", xa, xb)
    )
    keep = d - nodes.radius[ii] - nodes.radius[jj] < float(theta_max)
    return ii[keep], jj[keep]


# ── drivers ───────────────────────────────────────────────────────────────


def split_patch_pairs(
    nside: int,
    pix_ring: np.ndarray,
    cells: PatchCells,
    coarse_nsides: Sequence[int],
    cell_gid_start: Sequence[int],
    binedges: np.ndarray,
    level_nside: np.ndarray,
    edge_slop: float,
    first_coarse_bin: int,
    want_moments: bool,
    max_frontier: int = MAX_FRONTIER,
) -> Dict[str, np.ndarray]:
    """The edge-splitting search of one patch (see :func:`split_pairs`)."""
    nodes = SplitNodes.from_patch(nside, pix_ring, cells, coarse_nsides, cell_gid_start)
    top_I, top_J = top_pairs_within(
        nodes, nodes.top, nodes.top, float(binedges[-1]), ordered=True
    )
    return split_pairs(
        nodes,
        top_I,
        top_J,
        binedges,
        level_nside,
        edge_slop,
        first_coarse_bin,
        want_moments,
        max_frontier,
    )


def split_tile_pairs(
    nodes: SplitNodes,
    core_gids: np.ndarray,
    cand_gids: np.ndarray,
    npix: int,
    n_active: int,
    binedges: np.ndarray,
    level_nside: np.ndarray,
    edge_slop: float,
    first_coarse_bin: int,
    want_moments: bool,
    max_frontier: int = MAX_FRONTIER,
) -> Dict[str, np.ndarray]:
    """The edge-splitting search of one tile of the footprint
    (:func:`SplitNodes.from_rows` table): the top pairs are the coarsest
    cells of the tile (``core_gids``) against those within reach
    (``cand_gids``), owned by the tile when the core cell's id is the lower
    one (self pairs included)."""
    core = nodes.local_of_gid(core_gids, npix, n_active)
    cand = nodes.local_of_gid(cand_gids, npix, n_active)
    top_I, top_J = top_pairs_within(
        nodes, core, cand, float(binedges[-1]), ordered=True
    )
    return split_pairs(
        nodes,
        top_I,
        top_J,
        binedges,
        level_nside,
        edge_slop,
        first_coarse_bin,
        want_moments,
        max_frontier,
    )


def tile_reach_radius(nodes: SplitNodes, theta_max: float, tile_nside: int) -> float:
    """Search radius for :meth:`~CosmoFuse.global_geometry.TileSet.reach`
    that covers every top pair a tile may own: a top cell pair reaches
    ``theta_max`` when its centroids are within ``theta_max + s_I + s_J``,
    and a cell's centroid lies within the tile's circumradius of the tile
    centre (the centroid of points in a cap stays in the cap), so one extra
    ``max_pixrad(tile_nside)`` covers the centroid being anywhere in its
    tile rather than inside the polygon ``query_disc`` tests."""
    s_top = float(nodes.radius[nodes.top].max()) if nodes.top.size else 0.0
    return float(theta_max) + 2.0 * s_top + float(hp.max_pixrad(int(tile_nside)))


def check_accepted_pairs(
    nodes: SplitNodes,
    ids: np.ndarray,
    counts: np.ndarray,
    binedges: np.ndarray,
    level_nside: np.ndarray,
    edge_slop: float,
    local_of_gid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Acceptance-rule check of a stored pair list (for tests): per pair,
    the overhang of ``[d - s_a - s_b, d + s_a + s_b]`` beyond its bin in
    units of the bin width (``<= edge_slop`` expected; pixel pairs 0), and
    whether both ends are no coarser than the bin's static level.  ``ids``
    are global ids; ``local_of_gid`` maps them to the table (default: the
    table's ``gid`` array is searched)."""
    ids = np.asarray(ids, dtype=np.int64)
    if local_of_gid is None:
        order = np.argsort(nodes.gid, kind="stable")
        pos = np.searchsorted(nodes.gid[order], ids)
        local = order[pos]
        if not np.array_equal(nodes.gid[local], ids):
            raise ValueError("pair ids not in the node table")
    else:
        local = local_of_gid[ids]
    a, b = local[0], local[1]
    edges = np.asarray(binedges, dtype=np.float64)
    bins = np.repeat(np.arange(counts.size), np.asarray(counts, dtype=np.int64))
    xa, xb = nodes.xyz[a], nodes.xyz[b]
    cross = np.cross(xa, xb)
    d = np.arctan2(
        np.sqrt(np.einsum("ij,ij->i", cross, cross)), np.einsum("ij,ij->i", xa, xb)
    )
    s = nodes.radius[a] + nodes.radius[b]
    width = edges[bins + 1] - edges[bins]
    over = np.maximum(np.maximum(edges[bins] - (d - s), (d + s) - edges[bins + 1]), 0)
    static = np.asarray(level_nside, dtype=np.int64)[bins]
    pixel = (nodes.nside[a] >= nodes.base_nside) & (nodes.nside[b] >= nodes.base_nside)
    allowed = (nodes.nside[a] >= static) & (nodes.nside[b] >= static)
    return {
        "overhang": over / width,
        "allowed": allowed,
        "pixel_pair": pixel,
        "in_bin": (d > edges[bins]) & (d < edges[bins + 1]),
        "level_a": nodes.nside[a],
        "level_b": nodes.nside[b],
    }
