#!/usr/bin/env python
"""Moment-carrying treecode cells: accuracy and speed probe (prototype, not library code).

The static treecode replaces every child pair (i in I, j in J) of a coarse cell pair by the
parents' geometry.  For xi- the only approximation (given the parents' bin) is the rotation
phase: e^{-2iS(phi_i + phi_j)} = e^{-2iS(phi_I + phi_J)} e^{-4iS dphi}, with dphi the change of
the pair's position angle for the children's offsets from the cell centroids.  Expanding
e^{-4iS dphi} in the offsets, the cell pair becomes a contraction of per-cell *moments* of the
weighted shear, Sum w g {1, dx, dy, dx^2, dx dy, dy^2}:

  mono   : M_I M_J                                        (= the shipped treecode)
  dipole : + first order   (needs the dipoles,   3 complex values per cell)
  quad   : + second order  (needs the quadrupoles, 6 complex values per cell)

Offsets are gnomonic tangent-plane coordinates in the cell's east/north frame; the child shears
are parallel-transported into the cell frame before summing (exact, geometry only; the shipped
treecode skips this, so "mono" below is built without it to reproduce it bitwise-ish, and
"mono_t" shows the transport alone).  The perpendicular/parallel unit vectors at the J end are
the ones transported along the connecting geodesic (p_J = (sin phi_J, -cos phi_J) etc.), so the
flat-sky expansion is exact to the holonomy of the (tiny) child triangle.

References, measured on the same pair lists:
  fine   : the full-resolution pixel-pair estimator (a second Integrated3PCF, k = None);
  target : exact child sums with the *coarse* bin assignment (CUDA kernel over the cell->pixel
           CSR) -- it differs from fine only by the binning, and from the variants only by the
           phase approximation, so the two error sources are separated.

Usage (A100):  python moment_probe.py --nside 1024 --npatch 200 --k 2 4 --out results/probe_1024.json
Local CPU smoke test of the formulas (numpy twins of the kernels):
               python moment_probe.py --device cpu --nside 128 --npatch 3 --theta-min 60 --theta-max 400
                      --nbins 5 --Q 200 --k 2 --synthetic --centres auto --no-pixpoint-timing
"""
import argparse
import gc
import json
import os
import sys
import time

import healpy as hp
import numpy as np
import scipy.sparse as sp

P = "/project/ocean1-seitz/users/dgebauer"
DEFAULT_MASK = f"{P}/research/CosmoFuse-treecode-data/DESY3_Mask.fits"
DEFAULT_KAPPA = f"{P}/sbi/convergence_maps/fiducial/cosmo_fiducial/nzmap_0000.npy"
DEFAULT_CENTRES = f"{P}/sbi/CosmoFuse/Q90/patch_center_original_{{}}.dat"

VARIANTS = {"mono": 0, "dipole": 1, "quad": 2}
NM_OF = {0: 1, 1: 3, 2: 6}
THREADS = 256

# ─────────────────────────────── CUDA kernels ────────────────────────────────

CUDA_SRC = r"""
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ float2 cmulc(float2 a, float2 b) {   // a * conj(b)
    return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cscale(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
__device__ __forceinline__ float2 cmul_i(float2 a) { return make_float2(-a.y, a.x); }   // i * a

template <int NACC>
__device__ __forceinline__ void block_reduce_atomic(double* acc, double* sh, double* out) {
    #pragma unroll
    for (int k = 0; k < NACC; ++k) {
        double v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        acc[k] = v;
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < NACC; ++k) sh[wid * NACC + k] = acc[k];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x >> 5;
        for (int k = 0; k < NACC; ++k) {
            double s = 0.0;
            for (int w = 0; w < nw; ++w) s += sh[w * NACC + k];
            atomicAdd(out + k, s);
        }
    }
}

// Second-order correction factor of one oriented cell pair (I end -> J end).
template <int VAR>
__device__ __forceinline__ float2 pair_factor(
    float2 MI, float2 DpI, float2 DqI, float2 QppI, float2 QqpI,
    float2 MJ, float2 DpJ, float2 DqJ, float2 QppJ, float2 QqpJ,
    float S, float idd)
{
    float2 F = cmul(MI, MJ);
    if (VAR >= 1) {
        // -4 i S dphi_1,  dphi_1 = (p_J.dr_j - p_I.dr_i) / d
        F = cadd(F, cscale(cmul_i(csub(cmul(MI, DpJ), cmul(DpI, MJ))), -4.f * S * idd));
    }
    if (VAR >= 2) {
        // -8 dphi_1^2
        float2 b2 = cadd(csub(cmul(MI, QppJ), cscale(cmul(DpI, DpJ), 2.f)), cmul(QppI, MJ));
        F = cadd(F, cscale(b2, -8.f * idd * idd));
        // -4 i S dphi_2,  dphi_2 = -(a b) with a the parallel component
        float2 ab = cadd(csub(csub(cmul(MI, QqpJ), cmul(DqJ, DpI)), cmul(DqI, DpJ)), cmul(QqpI, MJ));
        F = cadd(F, cscale(cmul_i(ab), 4.f * S * idd * idd));
    }
    return F;
}

// One block row (blockIdx.x) per group = (patch, bin); gridDim.y blocks share a group.
// out[group][comb][3] = (xi+ numerator, xi- numerator, sum of weights), both orientations of a
// cross combination summed, auto combinations once (PixPoint's convention).
template <int NZ, int VAR>
__global__ void pair_sums(
    const int* __restrict__ node_i, const int* __restrict__ node_j,
    const float* __restrict__ phi_i, const float* __restrict__ phi_j,
    const float* __restrict__ inv_d,
    const float2* __restrict__ mom,      // [(node*NZ + z)*NM + m]
    const float* __restrict__ wts,       // [node*NZ + z]
    const long long* __restrict__ offsets,
    const int sign, double* __restrict__ out)
{
    constexpr int NM = VAR == 0 ? 1 : (VAR == 1 ? 3 : 6);
    constexpr int NC = NZ * (NZ + 1) / 2;
    __shared__ double sh[(THREADS_ / 32) * NC * 3];
    const int g = blockIdx.x;
    const long long start = offsets[g], stop = offsets[g + 1];
    const float S = (float)sign;
    double acc[NC * 3];
    #pragma unroll
    for (int k = 0; k < NC * 3; ++k) acc[k] = 0.0;

    for (long long p = start + (long long)blockIdx.y * blockDim.x + threadIdx.x; p < stop;
         p += (long long)gridDim.y * blockDim.x) {
        const int ni = node_i[p], nj = node_j[p];
        float si, ci, sj, cj;
        sincosf(phi_i[p], &si, &ci);
        sincosf(phi_j[p], &sj, &cj);
        const float2 ri = make_float2(ci * ci - si * si, -S * 2.f * si * ci);   // e^{-2iS phi_I}
        const float2 rj = make_float2(cj * cj - sj * sj, -S * 2.f * sj * cj);
        const float2 rij = cmul(ri, rj);     // e^{-2iS(phi_I + phi_J)}  (xi-)
        const float2 rijc = cmulc(ri, rj);   // e^{-2iS(phi_I - phi_J)}  (xi+)
        const float idd = (VAR > 0) ? inv_d[p] : 0.f;
        // perpendicular (p) and parallel (q) unit vectors of the pair at both ends,
        // the J ones parallel-transported along the geodesic
        const float pIx = -si, pIy = ci, qIx = ci, qIy = si;
        const float pJx = sj, pJy = -cj, qJx = -cj, qJy = -sj;

        float2 MI[NZ], MJ[NZ], DpI[NZ], DpJ[NZ], DqI[NZ], DqJ[NZ], QppI[NZ], QppJ[NZ], QqpI[NZ], QqpJ[NZ];
        float WI[NZ], WJ[NZ];
        #pragma unroll
        for (int z = 0; z < NZ; ++z) {
            const float2* mi = mom + ((size_t)ni * NZ + z) * NM;
            const float2* mj = mom + ((size_t)nj * NZ + z) * NM;
            MI[z] = mi[0]; MJ[z] = mj[0];
            WI[z] = wts[(size_t)ni * NZ + z]; WJ[z] = wts[(size_t)nj * NZ + z];
            DpI[z] = DpJ[z] = DqI[z] = DqJ[z] = QppI[z] = QppJ[z] = QqpI[z] = QqpJ[z] = make_float2(0.f, 0.f);
            if (VAR >= 1) {
                const float2 dxi = mi[1], dyi = mi[2], dxj = mj[1], dyj = mj[2];
                DpI[z] = cadd(cscale(dxi, pIx), cscale(dyi, pIy));
                DqI[z] = cadd(cscale(dxi, qIx), cscale(dyi, qIy));
                DpJ[z] = cadd(cscale(dxj, pJx), cscale(dyj, pJy));
                DqJ[z] = cadd(cscale(dxj, qJx), cscale(dyj, qJy));
            }
            if (VAR >= 2) {
                const float2 xxi = mi[3], xyi = mi[4], yyi = mi[5];
                const float2 xxj = mj[3], xyj = mj[4], yyj = mj[5];
                QppI[z] = cadd(cadd(cscale(xxi, pIx * pIx), cscale(xyi, 2.f * pIx * pIy)), cscale(yyi, pIy * pIy));
                QqpI[z] = cadd(cadd(cscale(xxi, qIx * pIx), cscale(xyi, qIx * pIy + qIy * pIx)), cscale(yyi, qIy * pIy));
                QppJ[z] = cadd(cadd(cscale(xxj, pJx * pJx), cscale(xyj, 2.f * pJx * pJy)), cscale(yyj, pJy * pJy));
                QqpJ[z] = cadd(cadd(cscale(xxj, qJx * pJx), cscale(xyj, qJx * pJy + qJy * pJx)), cscale(yyj, qJy * pJy));
            }
        }
        int c = 0;
        #pragma unroll
        for (int z1 = 0; z1 < NZ; ++z1) {
            #pragma unroll
            for (int z2 = z1; z2 < NZ; ++z2, ++c) {
                // orientation A: z1 at I, z2 at J
                float2 F = pair_factor<VAR>(MI[z1], DpI[z1], DqI[z1], QppI[z1], QqpI[z1],
                                            MJ[z2], DpJ[z2], DqJ[z2], QppJ[z2], QqpJ[z2], S, idd);
                float2 G = cmulc(MI[z1], MJ[z2]);
                float xim = F.x * rij.x - F.y * rij.y;
                float xip = G.x * rijc.x - G.y * rijc.y;
                float w = WI[z1] * WJ[z2];
                if (z1 != z2) {   // orientation B: z2 at I, z1 at J
                    F = pair_factor<VAR>(MI[z2], DpI[z2], DqI[z2], QppI[z2], QqpI[z2],
                                         MJ[z1], DpJ[z1], DqJ[z1], QppJ[z1], QqpJ[z1], S, idd);
                    G = cmulc(MI[z2], MJ[z1]);
                    xim += F.x * rij.x - F.y * rij.y;
                    xip += G.x * rijc.x - G.y * rijc.y;
                    w += WI[z2] * WJ[z1];
                }
                acc[c * 3 + 0] += (double)xip;
                acc[c * 3 + 1] += (double)xim;
                acc[c * 3 + 2] += (double)w;
            }
        }
    }
    block_reduce_atomic<NC * 3>(acc, sh, out + (size_t)g * NC * 3);
}

__device__ __forceinline__ void frame_of(const double* x, double* e, double* n) {
    const double alpha = atan2(x[1], x[0]);
    const double delta = asin(fmin(1.0, fmax(-1.0, x[2])));
    double sa, ca, sd, cd;
    sincos(alpha, &sa, &ca);
    sincos(delta, &sd, &cd);
    e[0] = -sa; e[1] = ca; e[2] = 0.0;
    n[0] = -sd * ca; n[1] = -sd * sa; n[2] = cd;
}
__device__ __forceinline__ double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Exact child sums of every (cell) pair of a group with the group's bin: the phase-error-free
// target.  One thread per coarse pair, double precision throughout.  Pixel nodes list
// themselves as their only child, so the same kernel is the double-precision full-res estimator.
__global__ void exact_children(
    const int* __restrict__ node_i, const int* __restrict__ node_j,
    const long long* __restrict__ offsets,
    const long long* __restrict__ cptr, const int* __restrict__ crows,
    const double* __restrict__ pxyz,      // [row*3]
    const double2* __restrict__ pm,       // w*gamma per pixel row (own frame)
    const double* __restrict__ pw,
    const int sign, double* __restrict__ out)   // [group][3]
{
    __shared__ double sh[(THREADS_ / 32) * 3];
    const int g = blockIdx.x;
    const long long start = offsets[g], stop = offsets[g + 1];
    const double S = (double)sign;
    double acc[3] = {0.0, 0.0, 0.0};
    for (long long p = start + (long long)blockIdx.y * blockDim.x + threadIdx.x; p < stop;
         p += (long long)gridDim.y * blockDim.x) {
        const int I = node_i[p], J = node_j[p];
        for (long long a = cptr[I]; a < cptr[I + 1]; ++a) {
            const int i = crows[a];
            const double* xi = pxyz + 3 * i;
            double ei[3], nni[3];
            frame_of(xi, ei, nni);
            const double2 mi = pm[i];
            const double wi = pw[i];
            for (long long b = cptr[J]; b < cptr[J + 1]; ++b) {
                const int j = crows[b];
                const double* xj = pxyz + 3 * j;
                double ej[3], nj[3];
                frame_of(xj, ej, nj);
                const double phi_i = atan2(dot3(nni, xj), dot3(ei, xj));
                const double phi_j = atan2(dot3(nj, xi), dot3(ej, xi));
                double s2i, c2i, s2j, c2j;
                sincos(2.0 * phi_i, &s2i, &c2i);
                sincos(2.0 * phi_j, &s2j, &c2j);
                const double2 mj = pm[j];
                // A = m_i e^{-2iS phi_i}, B = m_j e^{-2iS phi_j}
                const double Ax = mi.x * c2i + mi.y * S * s2i, Ay = mi.y * c2i - mi.x * S * s2i;
                const double Bx = mj.x * c2j + mj.y * S * s2j, By = mj.y * c2j - mj.x * S * s2j;
                acc[0] += Ax * Bx + Ay * By;
                acc[1] += Ax * Bx - Ay * By;
                acc[2] += wi * pw[j];
            }
        }
    }
    block_reduce_atomic<3>(acc, sh, out + (size_t)g * 3);
}
""".replace("THREADS_", str(THREADS))


# ─────────────────────────────── geometry ─────────────────────────────────────


def frames(xyz):
    """Local east/north unit vectors of unit vectors ``xyz`` (N, 3)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    alpha = np.arctan2(y, x)
    delta = np.arcsin(np.clip(z, -1.0, 1.0))
    ca, sa, cd, sd = np.cos(alpha), np.sin(alpha), np.cos(delta), np.sin(delta)
    e = np.stack([-sa, ca, np.zeros_like(ca)], axis=1)
    n = np.stack([-sd * ca, -sd * sa, cd], axis=1)
    return e, n


def kappa_to_shear(kappa, nside):
    if hp.npix2nside(kappa.size) != nside:
        kappa = hp.ud_grade(kappa.astype(np.float64), nside)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    f = np.sqrt(np.clip((ell + 2) * (ell - 1), 0, None) / np.maximum(ell * (ell + 1), 1))
    elm = hp.almxfl(hp.map2alm(kappa.astype(np.float64), lmax=lmax), f)
    return hp.alm2map_spin([elm, 0 * elm], nside, 2, lmax)


class Nodes:
    """Row-space pixels followed by the treecode cells of every level (PixPoint's virtual
    row order), with unit vectors, the cell -> pixel-row CSR and the moment operators."""

    def __init__(self, corr):
        nside = corr.nside
        self.npix = hp.nside2npix(nside)
        self.row_pix = np.asarray(corr.map_inds, dtype=np.int64)
        self.n_active = int(self.row_pix.size)
        lut = np.full(self.npix, -1, dtype=np.int64)
        lut[self.row_pix] = np.arange(self.n_active)
        self.lut = lut
        self.pix_xyz = np.asarray(hp.pix2vec(nside, self.row_pix), dtype=np.float64).T
        assert corr.n_aperture_cells == 0, "probe assumes aperture_nside=None"
        tree = corr._treecode
        xyz = [self.pix_xyz]
        self.levels = []          # scipy CSR cell -> pixel rows (data = 1), per level
        self.level_nside = []
        if tree is not None:
            A = None
            for L in range(tree.n_levels):
                n_c = int(tree.cells_per_level[L])
                idx = np.asarray(tree.child_indices[L], dtype=np.int64)
                ptr = np.asarray(tree.child_indptr[L], dtype=np.int64)
                if L == 0:
                    rows = lut[idx]
                    assert (rows >= 0).all()
                    C = sp.csr_matrix((np.ones(rows.size), rows, ptr), shape=(n_c, self.n_active))
                else:
                    n_prev = int(tree.cells_per_level[L - 1])
                    C = sp.csr_matrix((np.ones(idx.size), idx, ptr), shape=(n_c, n_prev)) @ A
                    C.sum_duplicates()
                    C.sort_indices()
                    assert np.all(C.data == 1.0)
                A = C
                self.levels.append(C)
                self.level_nside.append(int(tree.coarse_nsides[L]))
                xyz.append(np.asarray(hp.ang2vec(np.pi / 2 - tree.cell_dec[L], tree.cell_ra[L]), dtype=np.float64))
        self.xyz = np.ascontiguousarray(np.concatenate(xyz, axis=0))
        self.n_nodes = int(self.xyz.shape[0])
        # unified children CSR over all nodes (pixel node -> itself)
        ptrs = [np.arange(self.n_active + 1, dtype=np.int64)]
        rows = [np.arange(self.n_active, dtype=np.int64)]
        nnz = self.n_active
        for C in self.levels:
            ptrs.append(C.indptr[1:].astype(np.int64) + nnz)
            rows.append(C.indices.astype(np.int64))
            nnz += int(C.nnz)
        self.child_ptr = np.concatenate(ptrs)
        self.child_rows = np.concatenate(rows).astype(np.int32)
        self.e, self.n = frames(self.xyz)

    def node_of_id(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        out = np.where(ids < self.npix, self.lut[np.minimum(ids, self.npix - 1)], self.n_active + (ids - self.npix))
        assert (out >= 0).all() and (out < self.n_nodes).all()
        return out.astype(np.int32)

    def moments(self, gamma_rows, w_rows, S, transport=True, order=2):
        """(n_nodes, 6) complex128 moments of w*gamma and (n_nodes,) weight sums."""
        mom = np.zeros((self.n_nodes, 6), dtype=np.complex128)
        W = np.zeros(self.n_nodes, dtype=np.float64)
        f = w_rows.astype(np.float64) * gamma_rows.astype(np.complex128)
        mom[: self.n_active, 0] = f
        W[: self.n_active] = w_rows
        start = self.n_active
        for C in self.levels:
            n_c = C.shape[0]
            cell = np.repeat(np.arange(n_c), np.diff(C.indptr))
            rows = C.indices
            xc = self.xyz[start + cell]
            ec, nc = self.e[start + cell], self.n[start + cell]
            xp = self.pix_xyz[rows]
            dot = np.einsum("ij,ij->i", xc, xp)
            dx = np.einsum("ij,ij->i", ec, xp) / dot          # gnomonic offsets, radians
            dy = np.einsum("ij,ij->i", nc, xp) / dot
            vals = f[rows]
            if transport:
                ep = self.e[rows]
                rho = np.arctan2(np.einsum("ij,ij->i", nc, ep), np.einsum("ij,ij->i", ec, ep))
                vals = vals * np.exp(2j * S * rho)
            facs = [np.ones_like(dx)]
            if order >= 1:
                facs += [dx, dy]
            if order >= 2:
                facs += [dx * dx, dx * dy, dy * dy]
            for m, fac in enumerate(facs):
                v = vals * fac
                mom[start : start + n_c, m] = (
                    np.bincount(cell, weights=v.real, minlength=n_c)
                    + 1j * np.bincount(cell, weights=v.imag, minlength=n_c)
                )
            W[start : start + n_c] = np.bincount(cell, weights=w_rows[rows], minlength=n_c)
            start += n_c
        return mom, W


def pair_arrays(corr, nodes):
    """All patches' pairs as node indices, grouped by (patch, bin) = PixPoint's host order."""
    counts = np.array([np.asarray(b, dtype=np.int64) for b in corr.bins])      # (npatch, nbins)
    offsets = np.concatenate(([0], np.cumsum(counts.ravel()))).astype(np.int64)
    N = int(offsets[-1])
    ni = np.empty(N, dtype=np.int32)
    nj = np.empty(N, dtype=np.int32)
    pos = 0
    for ids in corr.pair_inds:
        n = ids.shape[1]
        ni[pos : pos + n] = nodes.node_of_id(ids[0])
        nj[pos : pos + n] = nodes.node_of_id(ids[1])
        pos += n
    assert pos == N
    return ni, nj, offsets, counts


def pair_geometry(xp, nodes, ni, nj, chunk=1 << 24):
    """Full position angles at both ends (own east/north frames) and 1/separation, float32."""
    N = ni.size
    xyz = xp.asarray(nodes.xyz)
    e = xp.asarray(nodes.e)
    n = xp.asarray(nodes.n)
    phi_i = xp.empty(N, dtype=xp.float32)
    phi_j = xp.empty(N, dtype=xp.float32)
    inv_d = xp.empty(N, dtype=xp.float32)
    for s in range(0, N, chunk):
        a = xp.asarray(ni[s : s + chunk]).astype(xp.int64)
        b = xp.asarray(nj[s : s + chunk]).astype(xp.int64)
        xa, xb = xyz[a], xyz[b]
        phi_i[s : s + chunk] = xp.arctan2((n[a] * xb).sum(1), (e[a] * xb).sum(1)).astype(xp.float32)
        phi_j[s : s + chunk] = xp.arctan2((n[b] * xa).sum(1), (e[b] * xa).sum(1)).astype(xp.float32)
        cr = xp.cross(xa, xb)
        d = xp.arctan2(xp.sqrt((cr * cr).sum(1)), (xa * xb).sum(1))
        inv_d[s : s + chunk] = (1.0 / d).astype(xp.float32)
    return phi_i, phi_j, inv_d


# ─────────────────────────────── numpy twins ─────────────────────────────────


def pair_sums_np(ni, nj, phi_i, phi_j, inv_d, mom, W, S, var, offsets):
    """Numpy twin of ``pair_sums<1, var>`` (nz = 1): per-group (xi+, xi-, w) sums."""
    phi_i = phi_i.astype(np.float64)
    phi_j = phi_j.astype(np.float64)
    idd = inv_d.astype(np.float64) if var > 0 else 0.0
    si, ci, sj, cj = np.sin(phi_i), np.cos(phi_i), np.sin(phi_j), np.cos(phi_j)
    ri = np.exp(-2j * S * phi_i)
    rj = np.exp(-2j * S * phi_j)
    mi, mj = mom[ni], mom[nj]
    MI, MJ = mi[:, 0], mj[:, 0]
    F = MI * MJ
    if var >= 1:
        pIx, pIy, qIx, qIy = -si, ci, ci, si
        pJx, pJy, qJx, qJy = sj, -cj, -cj, -sj
        DpI = mi[:, 1] * pIx + mi[:, 2] * pIy
        DqI = mi[:, 1] * qIx + mi[:, 2] * qIy
        DpJ = mj[:, 1] * pJx + mj[:, 2] * pJy
        DqJ = mj[:, 1] * qJx + mj[:, 2] * qJy
        F = F + 1j * (MI * DpJ - DpI * MJ) * (-4.0 * S * idd)
    if var >= 2:
        QppI = mi[:, 3] * pIx**2 + mi[:, 4] * 2 * pIx * pIy + mi[:, 5] * pIy**2
        QqpI = mi[:, 3] * qIx * pIx + mi[:, 4] * (qIx * pIy + qIy * pIx) + mi[:, 5] * qIy * pIy
        QppJ = mj[:, 3] * pJx**2 + mj[:, 4] * 2 * pJx * pJy + mj[:, 5] * pJy**2
        QqpJ = mj[:, 3] * qJx * pJx + mj[:, 4] * (qJx * pJy + qJy * pJx) + mj[:, 5] * qJy * pJy
        F = F + (MI * QppJ - 2 * DpI * DpJ + QppI * MJ) * (-8.0 * idd**2)
        F = F + 1j * (MI * QqpJ - DqJ * DpI - DqI * DpJ + QqpI * MJ) * (4.0 * S * idd**2)
    xim = (F * ri * rj).real
    xip = (MI * np.conj(MJ) * ri * np.conj(rj)).real
    w = W[ni] * W[nj]
    n_groups = offsets.size - 1
    gid = np.repeat(np.arange(n_groups), np.diff(offsets))
    out = np.stack(
        [np.bincount(gid, weights=v, minlength=n_groups) for v in (xip, xim, w)], axis=1
    )
    return out


def exact_children_np(ni, nj, offsets, nodes, pm, pw, S):
    """Numpy twin of ``exact_children``: loops over coarse pairs (small inputs only)."""
    n_groups = offsets.size - 1
    out = np.zeros((n_groups, 3))
    gid = np.repeat(np.arange(n_groups), np.diff(offsets))
    ptr, rows = nodes.child_ptr, nodes.child_rows
    for p in range(ni.size):
        ci = rows[ptr[ni[p]] : ptr[ni[p] + 1]]
        cj = rows[ptr[nj[p]] : ptr[nj[p] + 1]]
        xi, xj = nodes.pix_xyz[ci], nodes.pix_xyz[cj]
        ei, nni = frames(xi)
        ej, nj_ = frames(xj)
        phi_i = np.arctan2(nni @ xj.T, ei @ xj.T)        # (ci, cj)
        phi_j = np.arctan2(xi @ nj_.T, xi @ ej.T)
        A = pm[ci][:, None] * np.exp(-2j * S * phi_i)
        B = pm[cj][None, :] * np.exp(-2j * S * phi_j)
        out[gid[p], 0] += (A * np.conj(B)).real.sum()
        out[gid[p], 1] += (A * B).real.sum()
        out[gid[p], 2] += (pw[ci][:, None] * pw[cj][None, :]).sum()
    return out


# ─────────────────────────────── drivers ─────────────────────────────────────


class Device:
    def __init__(self, name):
        self.gpu = name == "gpu"
        if self.gpu:
            import cupy

            self.xp = cupy
            self.mod = cupy.RawModule(
                code=CUDA_SRC,
                options=("-std=c++17", "--use_fast_math"),
                name_expressions=[f"pair_sums<{nz}, {v}>" for nz in (1, 4) for v in (0, 1, 2)]
                + ["exact_children"],
            )
            self.n_sm = cupy.cuda.Device().attributes["MultiProcessorCount"]
        else:
            self.xp = np

    def sync(self):
        if self.gpu:
            self.xp.cuda.runtime.deviceSynchronize()

    def to_host(self, a):
        return self.xp.asnumpy(a) if self.gpu else np.asarray(a)

    def pair_sums(self, nz, var, ni, nj, phi_i, phi_j, inv_d, mom, W, S, offsets, n_groups):
        xp = self.xp
        if not self.gpu:
            assert nz == 1
            return pair_sums_np(ni, nj, phi_i, phi_j, inv_d, mom, W, S, var, offsets)
        NM = NM_OF[var]
        NC = nz * (nz + 1) // 2
        mom_dev = xp.ascontiguousarray(
            xp.broadcast_to(xp.asarray(mom[:, :NM], dtype=xp.complex64)[:, None, :], (mom.shape[0], nz, NM))
        )
        w_dev = xp.ascontiguousarray(xp.broadcast_to(xp.asarray(W, dtype=xp.float32)[:, None], (W.size, nz)))
        out = xp.zeros((n_groups, NC, 3), dtype=xp.float64)
        kern = self.mod.get_function(f"pair_sums<{nz}, {var}>")
        blocks_y = max(1, (4 * self.n_sm) // max(1, n_groups) + 1)
        args = (ni, nj, phi_i, phi_j, inv_d, mom_dev, w_dev, offsets, np.int32(S), out)

        def launch():
            kern((n_groups, blocks_y), (THREADS,), args)

        launch()
        self.sync()
        ts = []
        for _ in range(7):
            out.fill(0.0)
            self.sync()
            t0 = time.perf_counter()
            launch()
            self.sync()
            ts.append(time.perf_counter() - t0)
        return self.to_host(out), float(np.median(ts))

    def exact(self, ni, nj, offsets, nodes, pm, pw, S, n_groups):
        xp = self.xp
        if not self.gpu:
            return exact_children_np(ni, nj, offsets, nodes, pm, pw, S)
        out = xp.zeros((n_groups, 3), dtype=xp.float64)
        kern = self.mod.get_function("exact_children")
        blocks_y = max(1, (8 * self.n_sm) // max(1, n_groups) + 1)
        kern(
            (n_groups, blocks_y),
            (THREADS,),
            (
                ni, nj, offsets,
                xp.asarray(nodes.child_ptr), xp.asarray(nodes.child_rows),
                xp.asarray(nodes.pix_xyz.ravel()), xp.asarray(pm, dtype=xp.complex128),
                xp.asarray(pw, dtype=xp.float64), np.int32(S), out,
            ),
        )
        self.sync()
        return self.to_host(out)


def ratios(sums):
    """(n_groups, [nc,] 3) sums -> xi+, xi- as (npatch*nbins,) arrays (combination 0)."""
    s = sums if sums.ndim == 2 else sums[:, 0, :]
    return s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]


def compare(x, ref, nbins):
    """x, ref: (npatch, nbins)."""
    d = x - ref
    sigma = ref.std(axis=0)
    mean_r, mean_x = ref.mean(axis=0), x.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = mean_x / mean_r - 1.0
        in_sigma = d / sigma
    corr_per_bin = [float(np.corrcoef(x[:, b], ref[:, b])[0, 1]) for b in range(x.shape[1])]
    return {
        "rel_diff_patch_mean_per_bin": rel.tolist(),
        "corrcoef_per_patch_per_bin": corr_per_bin,
        "std_ratio_per_bin": (x.std(axis=0) / sigma).tolist(),
        "abs_diff_patch_mean_per_bin": (mean_x - mean_r).tolist(),
        "patch_mean_ref_per_bin": mean_r.tolist(),
        "per_patch_in_sigma_rms_per_bin": np.sqrt(np.nanmean(in_sigma**2, axis=0)).tolist(),
        "per_patch_in_sigma_max": float(np.nanmax(np.abs(in_sigma))),
        "per_patch_abs_rms": float(np.sqrt(np.mean(d * d))),
    }


def measure_pixpoint(dev, corr, shear, w, flip_g1, timing):
    """PixPoint's gg: per-patch xi+, xi- and (optionally) timings."""
    xp = dev.xp
    res = corr.gg(shear, w, apertures=False, flip_g1=flip_g1, return_device=False)
    xi_p, xi_m = np.asarray(res.xi_p[0], dtype=np.float64), np.asarray(res.xi_m[0], dtype=np.float64)
    out = {"xi_p": xi_p, "xi_m": xi_m}
    if timing and dev.gpu:
        for label, kw in (("pairs_only_ms", {"apertures": False}), ("with_apertures_ms", {"apertures": True})):
            ts = []
            for _ in range(5):
                dev.sync()
                t0 = time.perf_counter()
                corr.gg(shear, w, flip_g1=flip_g1, return_device=True, **kw)
                dev.sync()
                ts.append(time.perf_counter() - t0)
            out[label] = 1e3 * float(np.median(ts))
    return out


def make_corr(args, mask, phi_c, theta_c, k, pack=False):
    from pixpoint import Integrated3PCF

    return Integrated3PCF(
        args.nside, phi_c, theta_c, nbins=args.nbins, theta_min=args.theta_min, theta_max=args.theta_max,
        patch_size=args.Q, theta_Q=args.Q, mask=mask, device=args.device, map_precision="float32",
        rotation_precision="float32", accumulation_precision="float64", resolution_factor=k,
        pair_search_precision="float64", memory_budget_gb=float("inf"), pack_pairs=pack,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=1024)
    ap.add_argument("--npatch", type=int, default=200)
    ap.add_argument("--k", type=float, nargs="+", default=[4.0])
    ap.add_argument("--theta-min", type=float, default=7.0)
    ap.add_argument("--theta-max", type=float, default=123.0)
    ap.add_argument("--nbins", type=int, default=9)
    ap.add_argument("--Q", type=float, default=90.0)
    ap.add_argument("--zbin", type=int, default=2)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--kappa", default=DEFAULT_KAPPA)
    ap.add_argument("--centres", default=DEFAULT_CENTRES, help="pattern with {} for phi/theta, or 'auto'")
    ap.add_argument("--synthetic", action="store_true", help="Gaussian kappa instead of the CosmoGrid map")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="shape noise per pixel and component added to the shear (DES-like: 0.064 at nside 1024)")
    ap.add_argument("--no-pixpoint-timing", action="store_true")
    ap.add_argument("--nz-speed", type=int, nargs="*", default=[1, 4], help="nz values for the kernel timings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = Device(args.device)
    xp = dev.xp
    report = {"args": vars(args), "k": {}}

    # ---- footprint, patches, shear ------------------------------------------------------
    t0 = time.perf_counter()
    m = hp.read_map(args.mask)
    mask = (hp.ud_grade(m.astype(np.float64), args.nside) if hp.npix2nside(m.size) != args.nside else m) != 0
    if args.centres == "auto":
        from pixpoint.utils import select_patch_centers

        phi_c, theta_c = select_patch_centers(mask, 8, patch_size=args.Q, theta_Q=args.Q)
    else:
        phi_c = np.loadtxt(args.centres.format("phi"))
        theta_c = np.loadtxt(args.centres.format("theta"))
    phi_c, theta_c = phi_c[: args.npatch], theta_c[: args.npatch]
    npatch = int(phi_c.size)
    row_pix = np.flatnonzero(mask)
    if args.synthetic:
        ell = np.arange(3 * args.nside)
        cl = 1e-9 * (np.maximum(ell, 1) / 100.0) ** -1.0 * np.exp(-((ell / (2.5 * args.nside)) ** 2))
        cl[:2] = 0
        kappa = hp.synfast(cl, args.nside, lmax=3 * args.nside - 1, verbose=False)
    else:
        kappa = np.load(args.kappa, mmap_mode="r")[args.zbin].astype(np.float64)
    g1, g2 = kappa_to_shear(kappa, args.nside)
    signal = np.stack([g1[row_pix], g2[row_pix]])
    if args.noise:
        signal = signal + np.random.default_rng(1234).normal(0.0, args.noise, signal.shape)
    shear_rows = np.ascontiguousarray(signal[None], dtype=np.float32)  # (1, 2, n)
    w_rows = np.ones((1, row_pix.size), dtype=np.float32)
    report["setup"] = {"n_patches": npatch, "n_active": int(row_pix.size), "map_s": time.perf_counter() - t0}
    print(json.dumps(report["setup"]), flush=True)

    def evaluate(corr, label, mono_only):
        """Pairs, geometry, moments and all variants for one estimator; returns dict."""
        nodes = Nodes(corr)
        ni, nj, offsets, counts = pair_arrays(corr, nodes)
        n_groups = npatch * args.nbins
        N = int(offsets[-1])
        t0 = time.perf_counter()
        ni_d, nj_d = xp.asarray(ni), xp.asarray(nj)
        phi_i, phi_j, inv_d = pair_geometry(xp, nodes, ni_d, nj_d)
        off_d = xp.asarray(offsets)
        dev.sync()
        info = {"label": label, "n_pairs": N, "geometry_s": time.perf_counter() - t0,
                "n_pairs_per_bin": counts.sum(axis=0).tolist(),
                "level_nside": [int(v) for v in corr.level_nside], "n_nodes": nodes.n_nodes,
                "n_cells_per_level": [int(C.shape[0]) for C in nodes.levels]}
        print(json.dumps(info), flush=True)
        return nodes, (ni_d, nj_d, phi_i, phi_j, inv_d, off_d), n_groups, info

    gamma_rows = shear_rows[0, 0].astype(np.float64) + 1j * shear_rows[0, 1].astype(np.float64)
    w1 = w_rows[0].astype(np.float64)

    # ---- full resolution: reference + convention check ------------------------------
    t0 = time.perf_counter()
    corr = make_corr(args, mask, phi_c, theta_c, None)
    corr.preprocess()
    dev.sync()
    t_pre_full = time.perf_counter() - t0
    nodes, dev_pairs, n_groups, info = evaluate(corr, "full", True)
    info["preprocess_s"] = t_pre_full
    cf = measure_pixpoint(dev, corr, shear_rows, w_rows, True, not args.no_pixpoint_timing)
    cf_p, cf_m = cf["xi_p"].reshape(-1), cf["xi_m"].reshape(-1)
    best = None
    conv = {}
    for S in (1, -1):
        for f1 in (1.0, -1.0):
            gam = f1 * gamma_rows.real + 1j * gamma_rows.imag
            mom, W = nodes.moments(gam, w1, S, transport=False, order=0)
            r = dev.pair_sums(1, 0, *dev_pairs[:5], mom, W, S, dev_pairs[5], n_groups)
            sums = r[0] if dev.gpu else r
            xp_, xm_ = ratios(sums)
            dp = float(np.max(np.abs(xp_ - cf_p) / np.std(cf_p)))
            dm = float(np.max(np.abs(xm_ - cf_m) / np.std(cf_m)))
            conv[f"S={S:+d},flip_g1={f1 < 0}"] = {"max_dxip_over_std": dp, "max_dxim_over_std": dm}
            if best is None or dm < best[0]:
                best = (dm, S, f1)
    _, S, f1 = best
    report["convention"] = {"checks": conv, "chosen": {"S": S, "flip_g1": f1 < 0}}
    print("convention", json.dumps(report["convention"]), flush=True)
    gamma_rows = f1 * gamma_rows.real + 1j * gamma_rows.imag
    pm_pix = w1 * gamma_rows

    mom, W = nodes.moments(gamma_rows, w1, S, transport=False, order=0)
    fine = {}
    for nz in args.nz_speed:
        if nz > 1 and not dev.gpu:
            continue
        r = dev.pair_sums(nz, 0, *dev_pairs[:5], mom, W, S, dev_pairs[5], n_groups)
        if dev.gpu:
            sums, t = r
            info[f"mono_nz{nz}_ms"] = 1e3 * t
        else:
            sums = r
        if nz == 1:
            fine["xi_p"], fine["xi_m"] = ratios(sums)
    exact_full = dev.exact(dev_pairs[0], dev_pairs[1], dev_pairs[5], nodes, pm_pix, w1, S, n_groups)
    ex_p, ex_m = ratios(exact_full)
    shape = (npatch, args.nbins)
    info["float32_vs_float64"] = {
        "xi_p": compare(fine["xi_p"].reshape(shape), ex_p.reshape(shape), args.nbins),
        "xi_m": compare(fine["xi_m"].reshape(shape), ex_m.reshape(shape), args.nbins),
    }
    info["probe_vs_pixpoint"] = {
        "xi_p": compare(fine["xi_p"].reshape(shape), cf_p.reshape(shape), args.nbins),
        "xi_m": compare(fine["xi_m"].reshape(shape), cf_m.reshape(shape), args.nbins),
    }
    info["pixpoint"] = {k_: v for k_, v in cf.items() if k_.endswith("_ms")}
    report["full"] = info
    print("full", json.dumps({k_: v for k_, v in info.items() if "vs" not in k_}), flush=True)
    fine_p, fine_m = ex_p.reshape(shape), ex_m.reshape(shape)       # double-precision reference
    del corr, nodes, dev_pairs
    gc.collect()
    if dev.gpu:
        xp.get_default_memory_pool().free_all_blocks()

    # ---- treecode levels ----------------------------------------------------------------
    for k in args.k:
        t0 = time.perf_counter()
        corr = make_corr(args, mask, phi_c, theta_c, k)
        corr.preprocess()
        dev.sync()
        t_pre = time.perf_counter() - t0
        nodes, dev_pairs, n_groups, info = evaluate(corr, f"k={k:g}", False)
        info["preprocess_s"] = t_pre
        cf = measure_pixpoint(dev, corr, shear_rows, w_rows, True, not args.no_pixpoint_timing)
        info["pixpoint"] = {k_: v for k_, v in cf.items() if k_.endswith("_ms")}
        if dev.gpu and not args.no_pixpoint_timing:
            cp_ = make_corr(args, mask, phi_c, theta_c, k, pack=True)
            cp_.preprocess(release_host_pairs=True)
            cfp = measure_pixpoint(dev, cp_, shear_rows, w_rows, True, True)
            info["pixpoint_packed"] = {k_: v for k_, v in cfp.items() if k_.endswith("_ms")}
            del cp_
            gc.collect()
            xp.get_default_memory_pool().free_all_blocks()
        # target: exact child sums with the coarse binning
        t0 = time.perf_counter()
        target = dev.exact(dev_pairs[0], dev_pairs[1], dev_pairs[5], nodes, pm_pix, w1, S, n_groups)
        info["target_s"] = time.perf_counter() - t0
        tg_p, tg_m = (v.reshape(shape) for v in ratios(target))
        results = {"target": (tg_p, tg_m), "fine": (fine_p, fine_m),
                   "pixpoint": (cf["xi_p"].reshape(shape), cf["xi_m"].reshape(shape))}
        mom_cur, W = nodes.moments(gamma_rows, w1, S, transport=False, order=0)
        mom_t, _ = nodes.moments(gamma_rows, w1, S, transport=True, order=2)
        timings = {}
        for name, var, mom in (("mono", 0, mom_cur), ("mono_t", 0, mom_t), ("dipole", 1, mom_t), ("quad", 2, mom_t)):
            for nz in args.nz_speed:
                if nz > 1 and (not dev.gpu or name == "mono_t"):
                    continue
                r = dev.pair_sums(nz, var, *dev_pairs[:5], mom, W, S, dev_pairs[5], n_groups)
                if dev.gpu:
                    sums, t = r
                    timings[f"{name}_nz{nz}_ms"] = 1e3 * t
                else:
                    sums = r
                if nz == 1:
                    results[name] = tuple(v.reshape(shape) for v in ratios(sums))
        info["kernel_ms"] = timings
        comp = {}
        for name in ("pixpoint", "mono", "mono_t", "dipole", "quad", "target"):
            comp[name] = {}
            for ref in ("fine", "target"):
                if name == ref:
                    continue
                comp[name][f"vs_{ref}"] = {
                    "xi_p": compare(results[name][0], results[ref][0], args.nbins),
                    "xi_m": compare(results[name][1], results[ref][1], args.nbins),
                }
        comp["mono"]["vs_pixpoint"] = {
            "xi_p": compare(results["mono"][0], results["pixpoint"][0], args.nbins),
            "xi_m": compare(results["mono"][1], results["pixpoint"][1], args.nbins),
        }
        info["compare"] = comp
        info["theta_arcmin"] = (np.degrees(np.sqrt(corr.binedges[:-1] * corr.binedges[1:])) * 60).tolist()
        report["k"][f"{k:g}"] = info
        # console summary
        print(f"\n=== k = {k:g}: levels {info['level_nside']}, {info['n_pairs']:.3g} pairs ===")
        print("timings (ms):", json.dumps({**info["pixpoint"], **{"packed_" + a: b for a, b in info.get("pixpoint_packed", {}).items()}, **timings}))
        hdr = "bin  theta'  nside | " + " | ".join(f"{n:>9s}" for n in ("binning", "mono", "mono_t", "dipole", "quad"))
        print("xi- relative error of the patch mean (binning = target vs fine; variants vs target):")
        print(hdr)
        for b in range(args.nbins):
            row = [comp["target"]["vs_fine"]["xi_m"]["rel_diff_patch_mean_per_bin"][b]]
            row += [comp[n]["vs_target"]["xi_m"]["rel_diff_patch_mean_per_bin"][b] for n in ("mono", "mono_t", "dipole", "quad")]
            print(f"{b:3d} {info['theta_arcmin'][b]:7.1f} {info['level_nside'][b]:5d} | " + " | ".join(f"{v:+9.2e}" for v in row))
        print("xi+ relative error of the patch mean (same columns):")
        for b in range(args.nbins):
            row = [comp["target"]["vs_fine"]["xi_p"]["rel_diff_patch_mean_per_bin"][b]]
            row += [comp[n]["vs_target"]["xi_p"]["rel_diff_patch_mean_per_bin"][b] for n in ("mono", "mono_t", "dipole", "quad")]
            print(f"{b:3d} {info['theta_arcmin'][b]:7.1f} {info['level_nside'][b]:5d} | " + " | ".join(f"{v:+9.2e}" for v in row))
        print("xi- per patch vs target: correlation coefficient / std ratio (information retained; 1 = exact):")
        for b in range(args.nbins):
            cells = []
            for n in ("mono", "dipole", "quad"):
                c = comp[n]["vs_target"]["xi_m"]
                cells.append(f"{c['corrcoef_per_patch_per_bin'][b]:7.4f} {c['std_ratio_per_bin'][b]:6.3f}")
            print(f"{b:3d} {info['theta_arcmin'][b]:7.1f} {info['level_nside'][b]:5d} | " + " | ".join(cells))
        print("mono vs PixPoint gg(k): max |d xi-| per patch in sigma_patch =",
              f"{comp['mono']['vs_pixpoint']['xi_m']['per_patch_in_sigma_max']:.2e}", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fp:
                json.dump(report, fp, indent=1)
            np.savez(os.path.splitext(args.out)[0] + f"_k{k:g}.npz",
                     **{f"{n}_{c}": results[n][i] for n in results for i, c in enumerate(("xi_p", "xi_m"))})
        del corr, nodes, dev_pairs
        gc.collect()
        if dev.gpu:
            xp.get_default_memory_pool().free_all_blocks()
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(report, fp, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
