/*
 * moment_rows.cu -- Moment-carrying treecode cells: the normalised moments
 *                   of one resolution level straight from the pixel rows.
 *
 * For every (cell, lead) one group of SEG lanes walks the cell's member
 * pixels (the *direct* pixel -> cell CSR of the level, not the child chain:
 * composing moments level by level needs a shift of the expansion centre
 * and is exact only to second order) and accumulates
 *
 *     W   = sum_j w_j
 *     m_k = sum_j w_j v~_j f_k(dr_j),   f = [1, dx, dy, dx^2, dx dy, dy^2]
 *
 * with dr_j the gnomonic offset of pixel j from the cell centroid in the
 * cell's east/north frame and v~_j the pixel value: for shear (KIND 2) the
 * complex gamma parallel-transported into the cell frame, gamma e^{2iS rho}
 * (S = -1, CosmoFuse's rotation convention -- see CosmoFuse/moments.py),
 * for a scalar (KIND 1) the real value, for KIND 0 the constant 1 (moments
 * of the weights).  The geometry is evaluated on the fly in double from the
 * stored unit vectors (cell record: x, e, n; pixel: x) -- no per-entry
 * factor arrays.  Lane 0 divides by W and writes the normalised moments at
 * the output type: out_mom[((cell*n_lead + lead)*NM + k)*2 + {re, im}] (the
 * imaginary parts of real kinds are zero) and out_w[cell*n_lead + lead].
 *
 * Deterministic (fixed lane assignment and shuffle tree, no atomics).
 *
 * TSRC -- type of the row buffers (map dtype)
 * QT   -- type of the stored pixel unit vectors
 * TOUT -- output type (map dtype)
 * KIND -- 0 weights, 1 scalar, 2 shear
 * NM   -- 3 (dipole) or 6 (quad)
 * SEG  -- lanes cooperating on one cell (power of two, 1..32)
 */

__COMMON_CUDA_SOURCE__

#define MOMENT_BLOCK 256

template<typename TSRC, typename QT, typename TOUT, int KIND, int NM, int SEG>
__global__ void gpu_moment_level(
    const long long* __restrict__ indptr,   /* (n_cells + 1)                     */
    const int* __restrict__ indices,        /* member pixel rows                 */
    const double* __restrict__ cell_geom,   /* (n_cells, 9): x, east, north      */
    const QT* __restrict__ pix_xyz,         /* (n_active, 3)                     */
    const TSRC* __restrict__ w_src,
    const long long w_src_base,
    const long long w_src_lead,
    const long long w_src_row,
    const TSRC* __restrict__ v_src,         /* KIND 2: components k = 0, 1       */
    const long long v_src_base,
    const long long v_src_lead,
    const long long v_src_comp,
    const long long v_src_row,
    TOUT* __restrict__ out_mom,             /* (n_cells, n_lead, NM, 2)          */
    TOUT* __restrict__ out_w,               /* (n_cells, n_lead)                 */
    const int n_cells,
    const int n_lead)
{
    const int lane = (int)threadIdx.x % SEG;
    const int group = (int)threadIdx.x / SEG;
    const long long gid =
        (long long)blockIdx.x * (MOMENT_BLOCK / SEG) + (long long)group;
    const long long total = (long long)n_cells * (long long)n_lead;
    if (gid >= total) return;

    const int cell = (int)(gid / n_lead);
    const int lead = (int)(gid - (long long)cell * n_lead);

    const double* g = cell_geom + 9 * (long long)cell;
    const double xc0 = g[0], xc1 = g[1], xc2 = g[2];
    const double ec0 = g[3], ec1 = g[4], ec2 = g[5];
    const double nc0 = g[6], nc1 = g[7], nc2 = g[8];
    const double S = -1.0;

    const TSRC* wrow = w_src + w_src_base + (long long)lead * w_src_lead;
    const TSRC* v0 = v_src + v_src_base + (long long)lead * v_src_lead;
    const TSRC* v1 = v0 + v_src_comp;

    double sw = 0.0;
    double acc_re[NM], acc_im[NM];
#pragma unroll
    for (int k = 0; k < NM; ++k) { acc_re[k] = 0.0; acc_im[k] = 0.0; }

    const long long begin = indptr[cell];
    const long long end = indptr[cell + 1];
    for (long long j = begin + lane; j < end; j += SEG) {
        const long long row = (long long)indices[j];
        const double px = (double)pix_xyz[3 * row];
        const double py = (double)pix_xyz[3 * row + 1];
        const double pz = (double)pix_xyz[3 * row + 2];
        const double inv = 1.0 / (xc0 * px + xc1 * py + xc2 * pz);
        const double dx = (ec0 * px + ec1 * py + ec2 * pz) * inv;
        const double dy = (nc0 * px + nc1 * py + nc2 * pz) * inv;
        const double w = (double)wrow[row * w_src_row];
        sw += w;
        double fr, fi;
        if (KIND == 2) {
            const double cd = sqrt(px * px + py * py);
            double ep0 = 1.0, ep1 = 0.0;
            if (cd > 0.0) { ep0 = -py / cd; ep1 = px / cd; }
            const double rho = atan2(nc0 * ep0 + nc1 * ep1, ec0 * ep0 + ec1 * ep1);
            double tr, ti;
            sincos(2.0 * S * rho, &ti, &tr);
            const double g1 = (double)v0[row * v_src_row];
            const double g2 = (double)v1[row * v_src_row];
            fr = w * (g1 * tr - g2 * ti);
            fi = w * (g1 * ti + g2 * tr);
        } else if (KIND == 1) {
            fr = w * (double)v0[row * v_src_row];
            fi = 0.0;
        } else {
            fr = w;
            fi = 0.0;
        }
        acc_re[0] += fr;        acc_im[0] += fi;
        acc_re[1] += fr * dx;   acc_im[1] += fi * dx;
        acc_re[2] += fr * dy;   acc_im[2] += fi * dy;
        if (NM > 3) {
            acc_re[3] += fr * dx * dx;  acc_im[3] += fi * dx * dx;
            acc_re[4] += fr * dx * dy;  acc_im[4] += fi * dx * dy;
            acc_re[5] += fr * dy * dy;  acc_im[5] += fi * dy * dy;
        }
    }

#pragma unroll
    for (int off = SEG / 2; off > 0; off >>= 1) {
        sw += __shfl_down_sync(0xffffffffu, sw, off, SEG);
#pragma unroll
        for (int k = 0; k < NM; ++k) {
            acc_re[k] += __shfl_down_sync(0xffffffffu, acc_re[k], off, SEG);
            acc_im[k] += __shfl_down_sync(0xffffffffu, acc_im[k], off, SEG);
        }
    }

    if (lane == 0) {
        const double inv_w = (sw != 0.0) ? 1.0 / sw : 0.0;
        out_w[gid] = (TOUT)sw;
        TOUT* o = out_mom + gid * (long long)(2 * NM);
#pragma unroll
        for (int k = 0; k < NM; ++k) {
            o[2 * k] = (TOUT)(acc_re[k] * inv_w);
            o[2 * k + 1] = (TOUT)(acc_im[k] * inv_w);
        }
    }
}
