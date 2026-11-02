"""
Static treecode: per-angular-bin HEALPix resolution levels.

For every angular bin the pair geometry is built on the coarsest HEALPix
level whose pixel size ``p`` still satisfies ``p <= theta_lo / k`` (``k`` =
``resolution_factor``).  Coarse "cells" are formed *per patch* from the
unmasked base-resolution pixels inside the patch disc, so the patch window
stays an exact top-hat at base resolution.  A cell carries the weighted mean
of its members and the sum of their weights; because

    W_I W_J g_I g_J = sum_{i in I} sum_{j in J} w_i w_j g_i g_j ,

one coarse pair is exactly the sum over its fine child pairs -- the only
change is that the child pairs are binned and rotated with the geometry of
their parent cells (their binary-mask centroids).

Levels are assigned once, at preprocess time ("static"): the measurement
kernels stay branch-free and the estimator is a fixed linear operator on the
maps.  With ``edge_slop`` (:mod:`CosmoFuse.edge_split`) the per-bin level is
the *coarsest* level a bin may use: cell pairs whose separation range
crosses a bin edge are split into their children, down to exactly binned
pixel pairs, so one bin then holds pairs of several levels.

Conventions
-----------
* Host pair / aperture indices use *global ids*: ``id < npix`` is a HEALPix
  RING pixel, ``id >= npix`` is the appended virtual row ``id - npix``.
* Appended rows are numbered ``[aperture block | level 1 | level 2 | ...]``
  with the coarse levels ordered by descending nside; within a level the
  cells of patch ``p`` occupy ``cell_offsets[level, p] : cell_offsets[level,
  p + 1]``, which makes patch-range slicing a pure offset operation.
* NESTED ordering is used internally for the parent relation; everything
  user-facing stays RING.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import healpy as hp
import numpy as np
import scipy.sparse as sp


def is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


# Resolution factor used when the static treecode is switched on without an
# explicit value (``resolution_factor=True`` / ``"default"``): xi- is
# suppressed by <= 5 %, gamma_t by <= 2.5 %, xi+/xi_g by < 1 %.
DEFAULT_RESOLUTION_FACTOR = 4.0


def validate_resolution_factor(resolution_factor: object) -> Optional[float]:
    """Return the resolution factor as a float (``None`` = full resolution).

    ``None`` / ``False`` -> full resolution (the default); ``True`` /
    ``"default"`` -> :data:`DEFAULT_RESOLUTION_FACTOR`; a positive number ->
    that value.
    """
    if resolution_factor is None or resolution_factor is False:
        return None
    if resolution_factor is True or (
        isinstance(resolution_factor, str) and resolution_factor.lower() == "default"
    ):
        return DEFAULT_RESOLUTION_FACTOR
    if not isinstance(resolution_factor, (int, float, np.integer, np.floating)):
        raise ValueError(
            "resolution_factor must be None/False (full resolution), "
            "True/'default' (static treecode with the default factor "
            f"{DEFAULT_RESOLUTION_FACTOR:g}) or a positive number; got "
            f"{resolution_factor!r}"
        )
    k = float(resolution_factor)
    if not (k > 0.0) or np.isnan(k):
        raise ValueError(
            f"resolution_factor must be positive; got {resolution_factor!r}"
        )
    return k


def validate_edge_slop(edge_slop: object) -> Optional[float]:
    """Return the edge slop as a float (``None`` = the static per-bin levels).

    ``None`` / ``False`` -> no edge splitting (the pairs of a bin are the cell
    pairs of its level, binned by centroid separation); a number ``>= 0`` ->
    the edge-splitting search of :mod:`CosmoFuse.edge_split` with that slop
    (``0.0`` = exact binning).
    """
    if edge_slop is None or edge_slop is False:
        return None
    if isinstance(edge_slop, bool) or not isinstance(
        edge_slop, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            "edge_slop must be None (static per-bin levels) or a number >= 0 "
            f"(the tolerated overhang of a cell pair beyond its bin's edges, in "
            f"units of the bin width; 0 = exact binning); got {edge_slop!r}"
        )
    slop = float(edge_slop)
    if not (slop >= 0.0) or np.isnan(slop) or np.isinf(slop):
        raise ValueError(f"edge_slop must be finite and >= 0; got {edge_slop!r}")
    return slop


def assign_levels(
    binedges: np.ndarray, nside: int, resolution_factor: Optional[float]
) -> np.ndarray:
    """nside used by every angular bin.

    ``binedges`` in radians.  For bin ``b`` the result is the smallest
    power-of-two nside whose pixel size (``sqrt`` of the pixel area) is
    ``<= binedges[b] / resolution_factor``, clipped to ``[1, nside]``.
    """
    nbins = len(binedges) - 1
    levels = np.full(nbins, int(nside), dtype=np.int64)
    if resolution_factor is None:
        return levels
    if not is_power_of_two(nside):
        raise ValueError(
            "resolution_factor requires a power-of-two nside (NESTED parent "
            f"relation); got nside={nside}"
        )
    for b in range(nbins):
        limit = float(binedges[b]) / float(resolution_factor)
        ns = int(nside)
        while ns > 1 and hp.nside2resol(ns // 2) <= limit:
            ns //= 2
        levels[b] = ns
    return levels


def level_groups(level_nside: np.ndarray) -> List[Tuple[int, int, int]]:
    """Contiguous bin ranges ``(nside, first_bin, stop_bin)`` in bin order."""
    groups: List[Tuple[int, int, int]] = []
    start = 0
    for b in range(1, len(level_nside) + 1):
        if b == len(level_nside) or level_nside[b] != level_nside[start]:
            groups.append((int(level_nside[start]), start, b))
            start = b
    return groups


def effective_resolution_factor(
    binedges: np.ndarray, level_nside: np.ndarray
) -> np.ndarray:
    """theta_lo / pixel size for every bin."""
    resol = np.array([hp.nside2resol(int(ns)) for ns in level_nside], dtype=np.float64)
    factor: np.ndarray = np.asarray(binedges[:-1], dtype=np.float64) / resol
    return factor


# ── analytic pair-count model (preflight) ─────────────────────────────────


def disc_pair_fraction(theta_lo: float, theta_hi: float, radius: float) -> float:
    """Fraction of point pairs of a uniform disc (flat sky, radius ``radius``)
    whose separation lies in ``[theta_lo, theta_hi]``."""

    def cdf(d: float) -> float:
        t = min(max(d / (2.0 * radius), 0.0), 1.0)
        if t >= 1.0:
            return 1.0
        # P(sep <= d) for two uniform points in a disc
        a = np.arccos(t)
        s = np.sqrt(1.0 - t * t)
        return float(
            1.0
            + (2.0 / np.pi) * ((4.0 * t * t - 1.0) * a)
            - (2.0 / np.pi) * (t * s * (1.0 + 2.0 * t * t))
        )

    return max(cdf(theta_hi) - cdf(theta_lo), 0.0)


def model_cell_radius(nside_level: int, base_nside: int) -> float:
    """Typical radius (radians) of a complete cell of ``nside_level`` built
    from base pixels of ``base_nside``: the distance from the centroid to the
    farthest member pixel *centre*, i.e. the half-diagonal of the cell minus
    the half-diagonal of a base pixel, for a square of side ``nside2resol``
    (zero for the base level; a 2 x 2 cell gives ``resol(base) / sqrt 2``).
    The pair-count model of the edge split uses it; the search itself uses
    the exact per-cell radii (:func:`cell_radii`)."""
    if int(nside_level) >= int(base_nside):
        return 0.0
    return float(
        (hp.nside2resol(int(nside_level)) - hp.nside2resol(int(base_nside)))
        / np.sqrt(2.0)
    )


def _split_strips(
    theta_lo: float,
    theta_hi: float,
    stages: Sequence[Tuple[float, float]],
    slop: float,
) -> List[Tuple[float, float, float]]:
    """Separation intervals accepted at every stage of a bin under the edge
    split.  ``stages``: ``(range, density)`` of the pair types the recursion
    walks through, from the coarsest (cell pairs of the bin's static level)
    to pixel pairs -- the range is ``s_a + s_b`` of the two ends' model
    radii, the density the pair count relative to pixel pairs.  A pair of
    range ``r`` is accepted when ``d +- r`` overhangs the bin by at most
    ``slop`` of the bin width, i.e. for ``d`` in ``A = [theta_lo + r - slop
    w, theta_hi - r + slop w]`` (within the bin); a stage gets ``A`` minus
    what the previous (coarser) stage already took.  Returns ``(a, b,
    density)`` intervals (empty ones dropped)."""
    width = theta_hi - theta_lo
    out: List[Tuple[float, float, float]] = []
    prev: Optional[Tuple[float, float]] = None
    for r, density in stages:
        a = max(theta_lo, theta_lo + r - slop * width)
        b = min(theta_hi, theta_hi - r + slop * width)
        if b <= a:
            continue
        if prev is None:
            out.append((a, b, density))
        else:
            if prev[0] > a:
                out.append((a, prev[0], density))
            if b > prev[1]:
                out.append((prev[1], b, density))
        prev = (a, b)
    return out


def split_stages(nside_level: int, base_nside: int) -> List[Tuple[float, float]]:
    """The ladder of pair types the edge split walks from the cell pairs of
    ``nside_level`` to pixel pairs -- ``(n, n), (n, 2n), (2n, 2n), (2n, 4n),
    ...`` in nside (the larger cell is split first, so same-level pairs pass
    through a mixed stage) -- as ``(range, density)`` for
    :func:`_split_strips`: range ``s_a + s_b`` of the model radii, density
    ``(n_a n_b / base^2)^2`` relative to pixel pairs."""
    base = int(base_nside)
    n = int(nside_level)
    levels = []
    while n < base:
        levels.append(n)
        n *= 2
    levels.append(base)
    stages: List[Tuple[float, float]] = []
    for k, na in enumerate(levels):
        for nb in (na, levels[k + 1]) if k + 1 < len(levels) else (na,):
            r = model_cell_radius(na, base) + model_cell_radius(nb, base)
            density = (float(na) * float(nb) / float(base) ** 2) ** 2
            stages.append((r, density))
    return stages


def estimate_pairs_per_bin(
    n_pix_patch: np.ndarray,
    binedges: np.ndarray,
    level_nside: np.ndarray,
    base_nside: int,
    radius: float,
    edge_slop: Optional[float] = None,
) -> np.ndarray:
    """Projected number of pairs per bin, summed over patches.

    ``n_pix_patch``: unmasked base-resolution pixels per patch.  With
    ``edge_slop`` the projection follows the edge split: every stage of the
    ladder :func:`split_stages` (cell pairs of the bin's static level, then
    mixed and finer pairs down to pixel pairs) contributes the pairs whose
    centroid separation lies in the strips :func:`_split_strips` assigns to
    it (the coarsest stage the interior of the bin, every finer stage two
    strips next to the edges), with that stage's pair density.
    """
    n_pix_patch = np.asarray(n_pix_patch, dtype=np.float64)
    n_pairs_pix = float(np.sum(0.5 * n_pix_patch * n_pix_patch))
    out = np.zeros(len(level_nside), dtype=np.float64)
    for b, ns in enumerate(level_nside):
        lo, hi = float(binedges[b]), float(binedges[b + 1])
        if edge_slop is None or int(ns) >= int(base_nside):
            density = (float(ns) / float(base_nside)) ** 4
            out[b] = n_pairs_pix * density * disc_pair_fraction(lo, hi, radius)
            continue
        total = 0.0
        stages = split_stages(int(ns), int(base_nside))
        for a, bb, density in _split_strips(lo, hi, stages, float(edge_slop)):
            total += n_pairs_pix * density * disc_pair_fraction(a, bb, radius)
        out[b] = total
    return out


# ── per-patch cell construction ───────────────────────────────────────────


@dataclass
class PatchCells:
    """Coarse cells of one patch, one entry per coarse level."""

    n_cells: List[int] = field(default_factory=list)
    ra: List[np.ndarray] = field(default_factory=list)
    dec: List[np.ndarray] = field(default_factory=list)
    # CSR children of every cell: level 0 -> HEALPix RING ids of the member
    # pixels; deeper levels -> patch-local cell index in the previous level.
    child_indptr: List[np.ndarray] = field(default_factory=list)
    child_indices: List[np.ndarray] = field(default_factory=list)
    # per level: the largest angle between a cell's centroid and one of its
    # member pixel centres (the edge split's range bound; 0 for a one-pixel cell)
    radius: List[np.ndarray] = field(default_factory=list)


def cell_radii(
    centroid_xyz: np.ndarray, member_xyz: np.ndarray, cell_of_member: np.ndarray
) -> np.ndarray:
    """Radius of every cell: the largest angle (radians) between its centroid
    ``centroid_xyz[c]`` ``(3, n_cells)`` and a member ``member_xyz[:, m]``
    ``(3, n_members)`` with ``cell_of_member[m] == c``."""
    n_cells = int(centroid_xyz.shape[1])
    radius = np.zeros(n_cells, dtype=np.float64)
    if cell_of_member.size == 0:
        return radius
    c = centroid_xyz[:, cell_of_member]
    dot = np.einsum("ij,ij->j", c, member_xyz)
    cross = np.cross(c, member_xyz, axis=0)
    ang = np.arctan2(np.sqrt(np.einsum("ij,ij->j", cross, cross)), dot)
    np.maximum.at(radius, cell_of_member, ang)
    return radius


def build_patch_cells(
    nside: int, pix_ring: np.ndarray, coarse_nsides: Sequence[int]
) -> PatchCells:
    """Group the unmasked base pixels of one patch into coarse cells.

    ``coarse_nsides`` must be sorted by descending nside.  Cell positions are
    the unit-vector centroids of the member *pixel centres* (binary mask, no
    weights), so the geometry is independent of the maps.
    """
    cells = PatchCells()
    pix_ring = np.asarray(pix_ring, dtype=np.int64)
    if len(coarse_nsides) == 0:
        return cells
    base_order = int(np.log2(nside))
    nest = hp.ring2nest(nside, pix_ring) if pix_ring.size else pix_ring
    vec = (
        np.asarray(hp.pix2vec(nside, pix_ring), dtype=np.float64)
        if pix_ring.size
        else np.zeros((3, 0))
    )

    prev_ids = nest  # NESTED ids of the previous level's cells (level 0: pixels)
    prev_shift = 0
    for level, nside_c in enumerate(coarse_nsides):
        shift = 2 * (base_order - int(np.log2(nside_c)))
        parent_of_fine = nest >> shift
        cell_ids, inv_fine = np.unique(parent_of_fine, return_inverse=True)
        n_cells = int(cell_ids.size)

        cvec: np.ndarray = np.stack(
            [np.bincount(inv_fine, weights=vec[k], minlength=n_cells) for k in range(3)]
        )
        norm = np.sqrt(np.sum(cvec * cvec, axis=0))
        cvec = cvec / np.where(norm > 0, norm, 1.0)
        theta, phi = hp.vec2ang(cvec.T) if n_cells else (np.zeros(0), np.zeros(0))
        cells.n_cells.append(n_cells)
        cells.ra.append(np.asarray(phi, dtype=np.float64))
        cells.dec.append(np.pi / 2.0 - np.asarray(theta, dtype=np.float64))
        cells.radius.append(cell_radii(cvec, vec, inv_fine))

        # children = elements of the previous level
        parent_of_prev = prev_ids >> (shift - prev_shift)
        inv_prev = np.searchsorted(cell_ids, parent_of_prev)
        order = np.argsort(inv_prev, kind="stable")
        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(np.bincount(inv_prev, minlength=n_cells))
        cells.child_indptr.append(indptr)
        cells.child_indices.append(
            pix_ring[order] if level == 0 else order.astype(np.int64)
        )

        prev_ids = cell_ids
        prev_shift = shift
    return cells


def id_range_bounds(
    npix: int, n_aperture_cells: int, treecode: Optional["TreecodeGeometry"]
) -> Optional[np.ndarray]:
    """Boundaries of the global-id ranges a pair list can mix -- the pixels
    ``[0, npix)`` and every treecode level -- for the range-aware row blocks
    of :func:`CosmoFuse.packing.block_local_indices`; ``None`` without a
    treecode (one range)."""
    if treecode is None or treecode.n_levels == 0:
        return None
    starts = treecode.level_starts(first=int(n_aperture_cells))
    bounds = [int(npix)] + [
        int(npix) + int(starts[level]) for level in range(1, treecode.n_levels)
    ]
    return np.asarray(bounds, dtype=np.int64)


# ── global (all patches) treecode geometry ────────────────────────────────


@dataclass
class TreecodeGeometry:
    """Per-patch coarse cells of all patches (host side)."""

    base_nside: int
    coarse_nsides: Tuple[int, ...]
    # (n_levels, n_patches + 1): cumulative cell counts per level
    cell_offsets: np.ndarray
    # per level: CSR over that level's cells (all patches concatenated).
    # level 0 children are HEALPix RING ids, deeper levels index the previous
    # level's cells.
    child_indptr: List[np.ndarray]
    child_indices: List[np.ndarray]
    cell_ra: List[np.ndarray]
    cell_dec: List[np.ndarray]
    # per level: the cell radii (:func:`cell_radii`); ``None`` for a geometry
    # read from a file -- :meth:`radius` then computes them on demand
    cell_radius: Optional[List[np.ndarray]] = None

    @property
    def n_levels(self) -> int:
        return len(self.coarse_nsides)

    @property
    def n_patches(self) -> int:
        return int(self.cell_offsets.shape[1] - 1)

    @property
    def cells_per_level(self) -> np.ndarray:
        return self.cell_offsets[:, -1].astype(np.int64)

    @property
    def n_cells(self) -> int:
        return int(self.cells_per_level.sum())

    def level_starts(self, first: int = 0) -> np.ndarray:
        """Appended-row index of the first cell of every level (+ total)."""
        starts = np.zeros(self.n_levels + 1, dtype=np.int64)
        starts[1:] = np.cumsum(self.cells_per_level)
        return starts + int(first)

    def level_index(self, nside: int) -> int:
        return self.coarse_nsides.index(int(nside))

    @classmethod
    def from_patches(
        cls,
        base_nside: int,
        coarse_nsides: Sequence[int],
        patches: Sequence[PatchCells],
    ) -> "TreecodeGeometry":
        n_levels = len(coarse_nsides)
        n_patches = len(patches)
        cell_offsets = np.zeros((n_levels, n_patches + 1), dtype=np.int64)
        for level in range(n_levels):
            counts = [p.n_cells[level] for p in patches]
            cell_offsets[level, 1:] = np.cumsum(counts)

        child_indptr, child_indices, cell_ra, cell_dec = [], [], [], []
        for level in range(n_levels):
            n_cells = int(cell_offsets[level, -1])
            indptr = np.zeros(n_cells + 1, dtype=np.int64)
            indices = []
            nnz = 0
            for p, patch in enumerate(patches):
                a, b = cell_offsets[level, p], cell_offsets[level, p + 1]
                indptr[a + 1 : b + 1] = patch.child_indptr[level][1:] + nnz
                idx = patch.child_indices[level]
                if level > 0:
                    idx = idx + cell_offsets[level - 1, p]
                indices.append(idx)
                nnz += int(idx.size)
            child_indptr.append(indptr)
            child_indices.append(
                np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
            )
            cell_ra.append(
                np.concatenate([p.ra[level] for p in patches])
                if patches
                else np.zeros(0)
            )
            cell_dec.append(
                np.concatenate([p.dec[level] for p in patches])
                if patches
                else np.zeros(0)
            )
        radii: Optional[List[np.ndarray]] = None
        if all(len(p.radius) == n_levels for p in patches):
            radii = [
                (
                    np.concatenate([p.radius[level] for p in patches])
                    if patches
                    else np.zeros(0)
                )
                for level in range(n_levels)
            ]
        return cls(
            base_nside=int(base_nside),
            coarse_nsides=tuple(int(n) for n in coarse_nsides),
            cell_offsets=cell_offsets,
            child_indptr=child_indptr,
            child_indices=child_indices,
            cell_ra=cell_ra,
            cell_dec=cell_dec,
            cell_radius=radii,
        )

    def slice_patches(self, start: int, stop: int) -> "TreecodeGeometry":
        """Geometry restricted to patches ``start:stop`` (cells renumbered)."""
        n_levels = self.n_levels
        offsets = (
            self.cell_offsets[:, start : stop + 1]
            - self.cell_offsets[:, start : start + 1]
        )
        child_indptr, child_indices, cell_ra, cell_dec = [], [], [], []
        for level in range(n_levels):
            a, b = int(self.cell_offsets[level, start]), int(
                self.cell_offsets[level, stop]
            )
            indptr = self.child_indptr[level][a : b + 1]
            i0, i1 = int(indptr[0]), int(indptr[-1])
            idx = self.child_indices[level][i0:i1]
            if level > 0:
                idx = idx - self.cell_offsets[level - 1, start]
            child_indptr.append(indptr - i0)
            child_indices.append(idx)
            cell_ra.append(self.cell_ra[level][a:b])
            cell_dec.append(self.cell_dec[level][a:b])
        radii = None
        if self.cell_radius is not None:
            radii = [
                self.cell_radius[level][
                    int(self.cell_offsets[level, start]) : int(
                        self.cell_offsets[level, stop]
                    )
                ]
                for level in range(n_levels)
            ]
        return TreecodeGeometry(
            base_nside=self.base_nside,
            coarse_nsides=self.coarse_nsides,
            cell_offsets=offsets,
            child_indptr=child_indptr,
            child_indices=child_indices,
            cell_ra=cell_ra,
            cell_dec=cell_dec,
            cell_radius=radii,
        )

    def radius(self, level: int) -> np.ndarray:
        """Radii of the cells of ``level`` (:func:`cell_radii`), from the
        construction when available, else from :meth:`pixel_children`."""
        if self.cell_radius is not None:
            return np.asarray(self.cell_radius[level], dtype=np.float64)
        cache: Dict[int, np.ndarray] = self.__dict__.setdefault("_radius_cache", {})
        hit = cache.get(level)
        if hit is not None:
            return hit
        indptr, indices = self.pixel_children(level)
        n_cells = int(indptr.size - 1)
        centroid = np.asarray(
            hp.ang2vec(np.pi / 2.0 - self.cell_dec[level], self.cell_ra[level]),
            dtype=np.float64,
        ).reshape(n_cells, 3)
        members = (
            np.asarray(hp.pix2vec(self.base_nside, indices), dtype=np.float64)
            if indices.size
            else np.zeros((3, 0))
        )
        cell_of_member = np.repeat(np.arange(n_cells), np.diff(indptr))
        out = cell_radii(np.ascontiguousarray(centroid.T), members, cell_of_member)
        cache[level] = out
        return out

    def pixel_children(self, level: int) -> Tuple[np.ndarray, np.ndarray]:
        """Direct CSR ``(indptr, indices)`` from the cells of ``level`` to
        their member base pixels (RING ids), composed from the child chain
        and cached.  The moment-carrying cells are built from this (one pass
        pixels -> cells per level, not level by level: composing moments
        needs a shift of the expansion centre and is exact only to second
        order)."""
        cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = self.__dict__.setdefault(
            "_pixel_children_cache", {}
        )
        hit = cache.get(level)
        if hit is not None:
            return hit
        if level == 0:
            out = (
                np.asarray(self.child_indptr[0], dtype=np.int64),
                np.asarray(self.child_indices[0], dtype=np.int64),
            )
        else:
            p_indptr, p_indices = self.pixel_children(level - 1)
            n_prev = int(p_indptr.size - 1)
            prev = sp.csr_matrix(
                (np.ones(p_indices.size), p_indices, p_indptr),
                shape=(n_prev, int(hp.nside2npix(self.base_nside))),
            )
            indptr = np.asarray(self.child_indptr[level], dtype=np.int64)
            indices = np.asarray(self.child_indices[level], dtype=np.int64)
            parent = sp.csr_matrix(
                (np.ones(indices.size), indices, indptr),
                shape=(int(indptr.size - 1), n_prev),
            )
            direct = (parent @ prev).tocsr()
            direct.sum_duplicates()
            direct.sort_indices()
            out = (
                np.asarray(direct.indptr, dtype=np.int64),
                np.asarray(direct.indices, dtype=np.int64),
            )
        cache[level] = out
        return out

    def members_per_cell(self, level: int) -> np.ndarray:
        """Number of base pixels in every cell of ``level`` (for tests)."""
        counts = np.diff(self.child_indptr[0]).astype(np.int64)
        for lv in range(1, level + 1):
            indptr = self.child_indptr[lv]
            csum = np.concatenate(([0], np.cumsum(counts[self.child_indices[lv]])))
            counts = csum[indptr[1:]] - csum[indptr[:-1]]
        return counts
