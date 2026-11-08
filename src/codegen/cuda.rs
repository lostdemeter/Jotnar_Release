//! CUDA backend (FP32 compute, naive correctness-first kernels).
//!
//! The reference CUDA backend is optimized (kernels + cuBLAS). This port is
//! deliberately simple: one launch per op, static shapes, a sync after each
//! op. It is a correctness backend first; the scheduling work is a separate
//! milestone. T streams are refused (CUDA is the F path).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::payload::Feeds;

use super::infer::{infer_shapes, CKind, StreamInfo};
use super::{sanitize_cnames, Artifact, CompileOptions};

const CU_RUNTIME: &str = r#"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(_e)); return 4; } } while (0)

__global__ void k_add(const float *a, const float *b, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] + b[i];
}
__global__ void k_sub(const float *a, const float *b, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] - b[i];
}
__global__ void k_mul(const float *a, const float *b, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] * b[i];
}
__global__ void k_div(const float *a, const float *b, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] / b[i];
}
__global__ void k_silu(const float *a, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) { float x = a[i]; o[i] = x / (1.0f + expf(-x)); }
}
__global__ void k_fill(float *o, int64_t n, float v) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = v;
}
__global__ void k_gather(const float *t, const int64_t *ids, float *o, int64_t rows, int64_t w) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * w; i += blockDim.x * gridDim.x) {
    int64_t r = i / w, j = i % w; o[i] = t[ids[r] * w + j];
  }
}
__global__ void k_argmax(const float *x, int64_t *o, int64_t rows, int64_t w) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    int64_t best = 0; float bv = x[r * w];
    for (int64_t j = 1; j < w; ++j) if (x[r * w + j] > bv) { bv = x[r * w + j]; best = j; }
    o[r] = best;
  }
}
__global__ void k_matmul(const float *a, const float *b, float *o, int64_t m, int64_t k, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += blockDim.x * gridDim.x) {
    int64_t r = i / n, c = i % n; float acc = 0.0f;
    for (int64_t t = 0; t < k; ++t) acc += a[r * k + t] * b[t * n + c];
    o[i] = acc;
  }
}
__global__ void k_bmm(const float *a, const float *b, float *o, int64_t B, int64_t m, int64_t k, int64_t n, int64_t bb) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < B * m * n; i += blockDim.x * gridDim.x) {
    int64_t bi = i / (m * n), rem = i % (m * n), r = rem / n, c = rem % n;
    int64_t bj = (bb == 1) ? 0 : bi;
    float acc = 0.0f;
    for (int64_t t = 0; t < k; ++t) acc += a[(bi * m + r) * k + t] * b[(bj * k + t) * n + c];
    o[i] = acc;
  }
}
__global__ void k_bmmv(const float *a, const float *b, float *o, int64_t B, int64_t M, int64_t N, int64_t K, int64_t LAA, int64_t SA, int64_t LAB, int64_t SB, int TR) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < B * M * N; i += blockDim.x * gridDim.x) {
    int64_t bi = i / (M * N), rem = i % (M * N), mi = rem / N, ni = rem % N;
    float acc = 0.0f;
    for (int64_t kk = 0; kk < K; ++kk) {
      float bv = TR ? b[bi * SB + ni * LAB + kk] : b[bi * SB + kk * LAB + ni];
      acc += a[bi * SA + mi * LAA + kk] * bv;
    }
    o[i] = acc;
  }
}
__global__ void k_rmsnorm(const float *x, const float *w, float *o, int64_t rows, int64_t d, float eps) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float ss = 0.0f;
    for (int64_t j = 0; j < d; ++j) ss += x[r * d + j] * x[r * d + j];
    float rs = rsqrtf(ss / (float)d + eps);
    for (int64_t j = 0; j < d; ++j) o[r * d + j] = x[r * d + j] * rs * w[j];
  }
}
__global__ void k_layernorm(const float *x, const float *w, const float *b, float *o, int64_t rows, int64_t d, float eps) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float mu = 0.0f;
    for (int64_t j = 0; j < d; ++j) mu += x[r * d + j];
    mu /= (float)d;
    float ss = 0.0f;
    for (int64_t j = 0; j < d; ++j) { float dv = x[r * d + j] - mu; ss += dv * dv; }
    float rs = rsqrtf(ss / (float)d + eps);
    for (int64_t j = 0; j < d; ++j) o[r * d + j] = (x[r * d + j] - mu) * rs * w[j] + b[j];
  }
}
__global__ void k_tshift(const float *x, float *o, int64_t rows, int64_t d) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float mx = x[r * d];
    for (int64_t j = 1; j < d; ++j) if (x[r * d + j] > mx) mx = x[r * d + j];
    for (int64_t j = 0; j < d; ++j) o[r * d + j] = x[r * d + j] - mx;
  }
}
__global__ void k_softmax(const float *x, float *o, int64_t rows, int64_t d) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float mx = x[r * d];
    for (int64_t j = 1; j < d; ++j) if (x[r * d + j] > mx) mx = x[r * d + j];
    float se = 0.0f;
    for (int64_t j = 0; j < d; ++j) { float e = expf(x[r * d + j] - mx); o[r * d + j] = e; se += e; }
    for (int64_t j = 0; j < d; ++j) o[r * d + j] /= se;
  }
}
__global__ void k_rotary(const float *x, const int64_t *pos, float *o, int64_t rows, int64_t d, float base) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * (d / 2); i += blockDim.x * gridDim.x) {
    int64_t r = i / (d / 2), h = i % (d / 2);
    float ang = (float)pos[r] * powf(base, -2.0f * (float)h / (float)d);
    float c = cosf(ang), s = sinf(ang);
    float x0 = x[r * d + 2 * h], x1 = x[r * d + 2 * h + 1];
    o[r * d + 2 * h] = x0 * c - x1 * s;
    o[r * d + 2 * h + 1] = x0 * s + x1 * c;
  }
}
__global__ void k_rope(const float *x, const int64_t *pos, float *o, int64_t rows, int64_t nh, int64_t dh, float base) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * nh * (dh / 2); i += blockDim.x * gridDim.x) {
    int64_t r = i / (nh * (dh / 2)), rem = i % (nh * (dh / 2)), h = rem / (dh / 2), p = rem % (dh / 2);
    float ang = (float)pos[r] * powf(base, -2.0f * (float)p / (float)dh);
    float sn = sinf(ang), cs = cosf(ang);
    int64_t i0 = r * nh * dh + h * dh + 2 * p, i1 = i0 + 1;
    float a = x[i0], b = x[i1];
    o[i0] = a * cs - b * sn; o[i1] = a * sn + b * cs;
  }
}
__global__ void k_swiglu(const float *x, float *o, int64_t rows, int64_t inter) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * inter; i += blockDim.x * gridDim.x) {
    int64_t r = i / inter, j = i % inter;
    float up = x[r * 2 * inter + j], g = x[r * 2 * inter + inter + j];
    o[i] = (g / (1.0f + expf(-g))) * up;
  }
}
__global__ void k_transpose(const float *x, float *o, int64_t r, int64_t c) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r * c; i += blockDim.x * gridDim.x) {
    int64_t rr = i / c, cc = i % c; o[cc * r + rr] = x[i];
  }
}
__global__ void k_slice0(const float *x, float *o, int64_t lo, int64_t n, int64_t w) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * w; i += blockDim.x * gridDim.x) o[i] = x[(i / w + lo) * w + (i % w)];
}
__global__ void k_slice1(const float *x, float *o, int64_t lo, int64_t hi, int64_t rows, int64_t w) {
  int64_t ow = hi - lo;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * ow; i += blockDim.x * gridDim.x) {
    int64_t r = i / ow, c = i % ow; o[i] = x[r * w + lo + c];
  }
}
__global__ void k_concat(const float *a, const float *b, float *o, int64_t rows, int64_t wa, int64_t wb) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * (wa + wb); i += blockDim.x * gridDim.x) {
    int64_t r = i / (wa + wb), c = i % (wa + wb);
    o[i] = (c < wa) ? a[r * wa + c] : b[r * wb + c - wa];
  }
}
__global__ void k_select(const int64_t *m, const float *a, const float *b, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = m[i] ? a[i] : b[i];
}
__global__ void k_scatter_row(const float *row, const int64_t *idx, float *o, int64_t c) {
  int64_t r = idx[0];
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < c; i += blockDim.x * gridDim.x) o[r * c + i] = row[i];
}
__global__ void k_topk(const float *x, float *vals, int64_t *idx, int64_t rows, int64_t w, int64_t k) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    int64_t tn = 0;
    for (int64_t c = 0; c < w; ++c) {
      float v = x[r * w + c];
      int64_t p = (tn < k) ? tn : k;
      while (p > 0 && v > vals[r * k + p - 1]) --p;
      if (p >= k) continue;
      int64_t m = (tn < k) ? tn : k - 1;
      for (int64_t s = m; s > p; --s) { vals[r * k + s] = vals[r * k + s - 1]; idx[r * k + s] = idx[r * k + s - 1]; }
      vals[r * k + p] = v; idx[r * k + p] = c;
      if (tn < k) ++tn;
    }
  }
}
__global__ void k_conv(const float *x, const float *ker, float *o, int64_t h, int64_t w, int64_t kh, int64_t kw) {
  int64_t rh = kh / 2, rw = kw / 2;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h * w; i += blockDim.x * gridDim.x) {
    int64_t oy = i / w, ox = i % w; float acc = 0.0f;
    for (int64_t ky = 0; ky < kh; ++ky) {
      int64_t iy = oy + ky - rh; iy = iy < 0 ? 0 : (iy >= h ? h - 1 : iy);
      for (int64_t kx = 0; kx < kw; ++kx) {
        int64_t ix = ox + kx - rw; ix = ix < 0 ? 0 : (ix >= w ? w - 1 : ix);
        acc += x[iy * w + ix] * ker[ky * kw + kx];
      }
    }
    o[i] = acc;
  }
}
__global__ void k_interp(const float *x, float *o, int64_t h, int64_t w, int64_t c, int64_t ho, int64_t wo, float hscale, float wscale) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ho * wo * c; i += blockDim.x * gridDim.x) {
    int64_t oy = i / (wo * c), rem = i % (wo * c), ox = rem / c, cc = rem % c;
    float fsy = ((float)oy + 0.5f) * hscale - 0.5f;
    fsy = fsy < 0 ? 0 : (fsy > (float)(h - 1) ? (float)(h - 1) : fsy);
    int64_t y0 = (int64_t)floorf(fsy); float wy = fsy - y0;
    int64_t y1 = (y0 + 1 >= h) ? h - 1 : y0 + 1;
    float fsx = ((float)ox + 0.5f) * wscale - 0.5f;
    fsx = fsx < 0 ? 0 : (fsx > (float)(w - 1) ? (float)(w - 1) : fsx);
    int64_t x0 = (int64_t)floorf(fsx); float wx = fsx - x0;
    int64_t x1 = (x0 + 1 >= w) ? w - 1 : x0 + 1;
    o[i] = (1 - wy) * (1 - wx) * x[(y0 * w + x0) * c + cc]
         + (1 - wy) * wx * x[(y0 * w + x1) * c + cc]
         + wy * (1 - wx) * x[(y1 * w + x0) * c + cc]
         + wy * wx * x[(y1 * w + x1) * c + cc];
  }
}
__global__ void k_f2i(const float *x, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float v = x[i];
    if (v != v || v >= 9223372036854775808.0f || v < -9223372036854775808.0f) { o[i] = 0; }
    else o[i] = (int64_t)v;
  }
}
__global__ void k_i2f(const int64_t *x, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = (float)x[i];
}
__global__ void k_iadd(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] + (uint64_t)b[i]);
}
__global__ void k_iadd_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] + (uint64_t)b);
}
__global__ void k_isub(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] - (uint64_t)b[i]);
}
__global__ void k_isub_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] - (uint64_t)b);
}
__global__ void k_imul(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] * (uint64_t)b[i]);
}
__global__ void k_imul_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] * (uint64_t)b);
}
__global__ void k_imad(const int64_t *a, const int64_t *b, const int64_t *c, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] * (uint64_t)b[i] + (uint64_t)c[i]);
}
__global__ void k_ixor(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] ^ b[i];
}
__global__ void k_ixor_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] ^ b;
}
__global__ void k_iand(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] & b[i];
}
__global__ void k_iand_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] & b;
}
__global__ void k_ior(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] | b[i];
}
__global__ void k_ior_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] | b;
}
__global__ void k_ishl(const int64_t *a, int64_t s, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] << (uint64_t)s);
}
__global__ void k_ishr(const int64_t *a, int64_t s, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (int64_t)((uint64_t)a[i] >> (uint64_t)s);
}
__global__ void k_isar(const int64_t *a, const int64_t *s, int64_t *o, int64_t n, int *ierr) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    if (s[i] < 0 || s[i] > 63) { atomicExch(ierr, 1); continue; }
    o[i] = a[i] >> s[i];
  }
}
__global__ void k_isar_s(const int64_t *a, int64_t s, int64_t *o, int64_t n, int *ierr) {
  if (s < 0 || s > 63) { atomicExch(ierr, 1); return; }
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] >> s;
}
__global__ void k_ieq(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = (a[i] == b[i]);
}
__global__ void k_ieq_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = (a[i] == b);
}
__global__ void k_ilt(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = (a[i] < b[i]);
}
__global__ void k_ilt_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = (a[i] < b);
}
__global__ void k_imax(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] > b[i] ? a[i] : b[i];
}
__global__ void k_imax_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = a[i] > b ? a[i] : b;
}
__global__ void k_sadd(const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (b[i] > 0 && a[i] > INT64_MAX - b[i]) ? INT64_MAX : ((b[i] < 0 && a[i] < INT64_MIN - b[i]) ? INT64_MIN : a[i] + b[i]);
}
__global__ void k_sadd_s(const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    o[i] = (b > 0 && a[i] > INT64_MAX - b) ? INT64_MAX : ((b < 0 && a[i] < INT64_MIN - b) ? INT64_MIN : a[i] + b);
}
__global__ void k_idiv(const int64_t *a, const int64_t *b, int64_t *o, int64_t n, int *ierr, int mod) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int64_t bb = b[i];
    if (bb == 0 || (a[i] == INT64_MIN && bb == -1)) { atomicExch(ierr, 1); continue; }
    o[i] = mod ? (a[i] % bb) : (a[i] / bb);
  }
}
__global__ void k_idiv_s(const int64_t *a, int64_t b, int64_t *o, int64_t n, int *ierr, int mod) {
  if (b == 0) { atomicExch(ierr, 1); return; }
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    if (a[i] == INT64_MIN && b == -1) { atomicExch(ierr, 1); continue; }
    o[i] = mod ? (a[i] % b) : (a[i] / b);
  }
}
__global__ void k_idivu(const int64_t *a, const int64_t *b, int64_t *o, int64_t n, int *ierr, int mod) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    uint64_t bb = (uint64_t)b[i];
    if (bb == 0) { atomicExch(ierr, 1); continue; }
    uint64_t aa = (uint64_t)a[i];
    o[i] = (int64_t)(mod ? (aa % bb) : (aa / bb));
  }
}
__global__ void k_idivu_s(const int64_t *a, int64_t b, int64_t *o, int64_t n, int *ierr, int mod) {
  uint64_t bb = (uint64_t)b;
  if (bb == 0) { atomicExch(ierr, 1); return; }
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    uint64_t aa = (uint64_t)a[i];
    o[i] = (int64_t)(mod ? (aa % bb) : (aa / bb));
  }
}
__global__ void k_iselect(const int64_t *c, const int64_t *a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = c[i] ? a[i] : b[i];
}
__global__ void k_iselect_sab(const int64_t *c, int64_t a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = c[i] ? a : b;
}
__global__ void k_iselect_sa(const int64_t *c, int64_t a, const int64_t *b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = c[i] ? a : b[i];
}
__global__ void k_iselect_sb(const int64_t *c, const int64_t *a, int64_t b, int64_t *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = c[i] ? a[i] : b;
}
__global__ void k_attn(const float *qr, const float *kf, const float *vf,
                       const int64_t *mask, float *o,
                       int64_t rows, int64_t nh, int64_t nkv, int64_t dh,
                       int64_t S, float neg) {
  int64_t per = nh / nkv;
  int64_t nout = rows * nh * dh;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nout; idx += blockDim.x * gridDim.x) {
    int64_t r = idx / (nh * dh);
    int64_t rem = idx % (nh * dh);
    int64_t head = rem / dh;
    int64_t d = rem % dh;
    int64_t g = head / per;
    int64_t hh = head % per;
    int64_t base = r * nh * dh + head * dh;
    float mx = -3.402823466e+38f;
    for (int64_t t = 0; t < S; ++t) {
      float acc = 0.0f;
      for (int64_t dd = 0; dd < dh; ++dd)
        acc += qr[base + dd] * kf[t * nkv * dh + g * dh + dd];
      if (mask[hh * S + t] == 0) acc = neg;
      if (acc > mx) mx = acc;
    }
    float se = 0.0f;
    for (int64_t t = 0; t < S; ++t) {
      float acc = 0.0f;
      for (int64_t dd = 0; dd < dh; ++dd)
        acc += qr[base + dd] * kf[t * nkv * dh + g * dh + dd];
      if (mask[hh * S + t] == 0) acc = neg;
      se += expf(acc - mx);
    }
    float sum = 0.0f;
    for (int64_t t = 0; t < S; ++t) {
      float acc = 0.0f;
      for (int64_t dd = 0; dd < dh; ++dd)
        acc += qr[base + dd] * kf[t * nkv * dh + g * dh + dd];
      if (mask[hh * S + t] == 0) acc = neg;
      float p = expf(acc - mx) / se;
      sum += p * vf[t * nkv * dh + g * dh + d];
    }
    o[idx] = sum;
  }
}
__global__ void k_gather3(const float *t, const int64_t *ii, const int64_t *jj, float *o, int64_t n, int64_t w, int64_t v2) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * w; i += blockDim.x * gridDim.x) {
    int64_t r = i / w, c = i % w; o[i] = t[(ii[r] * v2 + jj[r]) * w + c];
  }
}
"#;

fn launch(kernel: &str, grid: &str, block: &str, args: &str) -> String {
    format!(
        "  {kernel}<<<{grid}, {block}>>>({args});\n  CK(cudaGetLastError());\n  CK(cudaDeviceSynchronize());"
    )
}

fn grid_expr(n: &str) -> String {
    format!("((({n}) + 255) / 256 > 65535 ? 65535 : (({n}) + 255) / 256)")
}

struct C {
    streams: HashMap<String, StreamInfo>,
    config: crate::ir::Config,
}

impl C {
    fn s(&self, n: &str) -> Result<&StreamInfo> {
        self.streams
            .get(n)
            .ok_or_else(|| Error::asm(format!("CUDA: unknown stream '{n}'")))
    }
    fn n(&self, n: &str) -> Result<usize> {
        self.s(n)?
            .elems()
            .ok_or_else(|| Error::unimplemented(format!("CUDA: dynamic shape in {n}")))
    }
    fn rows(&self, n: &str) -> Result<usize> {
        self.s(n)?
            .rows()
            .ok_or_else(|| Error::unimplemented(format!("CUDA: dynamic rows in {n}")))
    }
    fn w(&self, n: &str) -> Result<usize> {
        self.s(n)?
            .shape
            .last()
            .copied()
            .flatten()
            .ok_or_else(|| Error::unimplemented(format!("CUDA: dynamic width in {n}")))
    }
}

pub fn pattern(mn: &str, outs: &[String], args: &[String], c: &C) -> Result<String> {
    let o = outs.first().map(|s| s.as_str()).unwrap_or("");
    let a = |i: usize| args.get(i).map(|s| s.as_str()).unwrap_or("");
    let one = |kernel: &str, n: usize, args: &str| {
        launch(kernel, &grid_expr(&n.to_string()), "256", args)
    };
    match mn {
        "ADD" | "SUB" | "MUL" | "DIV" | "SILU" => {
            let n = c.n(o)?;
            let k = match mn {
                "ADD" => "k_add",
                "SUB" => "k_sub",
                "MUL" => "k_mul",
                "DIV" => "k_div",
                _ => "k_silu",
            };
            if mn == "SILU" {
                Ok(one(k, n, &format!("{}, {}, {}", a(0), o, n)))
            } else {
                Ok(one(k, n, &format!("{}, {}, {}, {}", a(0), a(1), o, n)))
            }
        }
        "BETA" | "TBETA" => {
            let n = c.n(o)?;
            let v = if mn == "TBETA" {
                c.config
                    .get_f64_or("beta_b", c.config.get_f64_or("beta", 0.5))
            } else {
                c.config.get_f64_or("beta", 0.5)
            };
            Ok(one("k_fill", n, &format!("{}, {}, {:?}f", o, n, v)))
        }
        "GATHER" => {
            let rows = c.rows(&a(1))?;
            let w = c.w(a(0))?;
            Ok(one(
                "k_gather",
                rows * w,
                &format!("{}, {}, {}, {}, {}", a(0), a(1), o, rows, w),
            ))
        }
        "ARGMAX" => {
            let rows = c.rows(a(0))?;
            let w = c.w(a(0))?;
            Ok(one(
                "k_argmax",
                rows,
                &format!("{}, {}, {}, {}", a(0), o, rows, w),
            ))
        }
        "MATMUL" | "BATCH_MATMUL" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            if ash.len() == 2 {
                let (m, k, n) = (ash[0].unwrap(), ash[1].unwrap(), bsh[1].unwrap());
                Ok(one(
                    "k_matmul",
                    m * n,
                    &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n),
                ))
            } else {
                let (b, m, k, n) = (
                    ash[0].unwrap(),
                    ash[1].unwrap(),
                    ash[2].unwrap(),
                    bsh[2].unwrap(),
                );
                let bb = bsh[0].unwrap();
                Ok(one(
                    "k_bmm",
                    b * m * n,
                    &format!(
                        "{}, {}, {}, {}, {}, {}, {}, {}",
                        a(0),
                        a(1),
                        o,
                        b,
                        m,
                        k,
                        n,
                        bb
                    ),
                ))
            }
        }
        "BMMV" => {
            let nums: Vec<i64> = (2..13)
                .map(|i| {
                    a(i).parse::<f64>()
                        .map(|v| v as i64)
                        .map_err(|_| Error::asm(format!("CUDA BMMV: bad literal {}", a(i))))
                })
                .collect::<Result<_>>()?;
            let (bb, m, n, k, laa, sa, lab, sb, _lac, _sc, tr) = (
                nums[0], nums[1], nums[2], nums[3], nums[4], nums[5], nums[6], nums[7], nums[8],
                nums[9], nums[10],
            );
            Ok(one(
                "k_bmmv",
                (bb * m * n) as usize,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                    a(0),
                    a(1),
                    o,
                    bb,
                    m,
                    n,
                    k,
                    laa,
                    sa,
                    lab,
                    sb,
                    tr
                ),
            ))
        }
        "RMSNORM" | "LAYERNORM" => {
            let rows = c.rows(o)?;
            let d = c.w(a(0))?;
            let eps = if mn == "RMSNORM" {
                c.config
                    .get_f64("eps_rms")
                    .ok_or_else(|| Error::asm("CUDA RMSNORM needs CONFIG eps_rms"))?
            } else {
                c.config
                    .get_f64("eps_ln")
                    .ok_or_else(|| Error::asm("CUDA LAYERNORM needs CONFIG eps_ln"))?
            };
            if mn == "RMSNORM" {
                Ok(one(
                    "k_rmsnorm",
                    rows,
                    &format!("{}, {}, {}, {}, {}, {:?}f", a(0), a(1), o, rows, d, eps),
                ))
            } else {
                Ok(one(
                    "k_layernorm",
                    rows,
                    &format!(
                        "{}, {}, {}, {}, {}, {}, {:?}f",
                        a(0),
                        a(1),
                        a(2),
                        o,
                        rows,
                        d,
                        eps
                    ),
                ))
            }
        }
        "TSHIFT" => {
            let rows = c.rows(o)?;
            let d = c.w(a(0))?;
            Ok(one(
                "k_tshift",
                rows,
                &format!("{}, {}, {}, {}", a(0), o, rows, d),
            ))
        }
        "SOFTMAX" | "SOFTMAX_WIDE" => {
            let rows = c.rows(o)?;
            let d = c.w(a(0))?;
            Ok(one(
                "k_softmax",
                rows,
                &format!("{}, {}, {}, {}", a(0), o, rows, d),
            ))
        }
        "ROTARY" => {
            let rows = c.rows(a(0))?;
            let d = c.w(a(0))?;
            let base = c.config.get_f64_or("rope_base", 10000.0);
            Ok(one(
                "k_rotary",
                rows * (d / 2),
                &format!("{}, {}, {}, {}, {}, {:?}f", a(0), a(1), o, rows, d, base),
            ))
        }
        "ROPE" => {
            let rows = c.rows(a(0))?;
            let nh = a(2)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA ROPE nh"))?;
            let dh = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA ROPE dh"))?;
            let base = c.config.get_f64_or("rope_base", 10000.0);
            Ok(one(
                "k_rope",
                rows * nh as usize * (dh / 2) as usize,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {:?}f",
                    a(0),
                    a(1),
                    o,
                    rows,
                    nh,
                    dh,
                    base
                ),
            ))
        }
        "SWIGLU" => {
            let rows = c.rows(a(0))?;
            let inter = a(1)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SWIGLU inter"))?;
            Ok(one(
                "k_swiglu",
                rows * inter as usize,
                &format!("{}, {}, {}, {}", a(0), o, rows, inter),
            ))
        }
        "TRANSPOSE" => {
            let sh = c.s(a(0))?.shape.clone();
            let (r, w) = (sh[0].unwrap(), sh[1].unwrap());
            Ok(one(
                "k_transpose",
                r * w,
                &format!("{}, {}, {}, {}", a(0), o, r, w),
            ))
        }
        "SLICE" => {
            let sh = c.s(a(0))?.shape.clone();
            let ax = a(1)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SLICE axis"))?;
            let lo = a(2)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SLICE lo"))?;
            let hi = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SLICE hi"))?;
            let (r, w) = (sh[0].unwrap(), sh[1].unwrap());
            if ax == 1 {
                Ok(one(
                    "k_slice1",
                    r * (hi - lo) as usize,
                    &format!("{}, {}, {}, {}, {}, {}", a(0), o, lo, hi, r, w),
                ))
            } else {
                Ok(one(
                    "k_slice0",
                    (hi - lo) as usize * w,
                    &format!("{}, {}, {}, {}, {}", a(0), o, lo, (hi - lo) as usize, w),
                ))
            }
        }
        "CONCAT" => {
            let wa = c.w(a(0))?;
            let wb = c.w(a(1))?;
            let rows = c.rows(a(0))?;
            Ok(one(
                "k_concat",
                rows * (wa + wb),
                &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, rows, wa, wb),
            ))
        }
        "SELECT" => {
            let n = c.n(o)?;
            Ok(one(
                "k_select",
                n,
                &format!("{}, {}, {}, {}, {}", a(0), a(1), a(2), o, n),
            ))
        }
        "SCATTER" => {
            let sh = c.s(a(0))?.shape.clone();
            let (r, w) = (sh[0].unwrap(), sh[1].unwrap());
            let copy = format!(
                "  CK(cudaMemcpy({o}, {}, {} * sizeof(float), cudaMemcpyDeviceToDevice));\n",
                a(0),
                r * w
            );
            let launch = one(
                "k_scatter_row",
                w,
                &format!("{}, {}, {}, {}", a(1), a(2), o, w),
            );
            Ok(copy + &launch)
        }
        "TOPK" => {
            let rows = c.rows(a(0))?;
            let w = c.w(a(0))?;
            let k = a(1).parse::<i64>().map_err(|_| Error::asm("CUDA TOPK k"))?;
            Ok(one(
                "k_topk",
                rows,
                &format!("{}, {}, {}, {}, {}, {}", a(0), outs[0], outs[1], rows, w, k),
            ))
        }
        "CONV" => {
            let sh = c.s(a(0))?.shape.clone();
            let ksh = c.s(a(1))?.shape.clone();
            let (h, w, kh, kw) = (
                sh[0].unwrap(),
                sh[1].unwrap(),
                ksh[0].unwrap(),
                ksh[1].unwrap(),
            );
            Ok(one(
                "k_conv",
                h * w,
                &format!("{}, {}, {}, {}, {}, {}, {}", a(0), a(1), o, h, w, kh, kw),
            ))
        }
        "INTERP" => {
            let sh = c.s(a(0))?.shape.clone();
            let sy = a(1)
                .parse::<f64>()
                .map_err(|_| Error::asm("CUDA INTERP sy"))?;
            let sx = a(2)
                .parse::<f64>()
                .map_err(|_| Error::asm("CUDA INTERP sx"))?;
            let (h, w) = (sh[0].unwrap(), sh[1].unwrap());
            let ch = if sh.len() == 3 { sh[2].unwrap() } else { 1 };
            let ho = ((h as f64 * sy).round().max(1.0)) as usize;
            let wo = ((w as f64 * sx).round().max(1.0)) as usize;
            Ok(one(
                "k_interp",
                ho * wo * ch,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {}, {:?}f, {:?}f",
                    a(0),
                    o,
                    h,
                    w,
                    ch,
                    ho,
                    wo,
                    h as f64 / ho as f64,
                    w as f64 / wo as f64
                ),
            ))
        }
        "F2I" => {
            let n = c.n(o)?;
            Ok(one("k_f2i", n, &format!("{}, {}, {}", a(0), o, n)))
        }
        "I2F" => {
            let n = c.n(o)?;
            Ok(one("k_i2f", n, &format!("{}, {}, {}", a(0), o, n)))
        }
        "IADD" | "ISUB" | "IMUL" | "IXOR" | "IAND" | "IOR" | "IEQ" | "ILT" | "IMAX" | "SADD" => {
            let (a, b, o) = (a(0), a(1), o);
            let n = c.n(o)?;
            let kern = match mn {
                "IADD" => "k_iadd",
                "ISUB" => "k_isub",
                "IMUL" => "k_imul",
                "IXOR" => "k_ixor",
                "IAND" => "k_iand",
                "IOR" => "k_ior",
                "IEQ" => "k_ieq",
                "ILT" => "k_ilt",
                "IMAX" => "k_imax",
                _ => "k_sadd",
            };
            if c.streams.contains_key(b) {
                Ok(one(kern, n, &format!("{}, {}, {}, {}", a, b, o, n)))
            } else {
                let lit = b
                    .parse::<i64>()
                    .map_err(|_| Error::asm(format!("CUDA {mn}: bad literal {b}")))?;
                Ok(one(
                    &format!("{kern}_s"),
                    n,
                    &format!("{}, {}, {}, {}", a, lit, o, n),
                ))
            }
        }
        "IMAD" => {
            let (a, b, cc, o) = (a(0), a(1), a(2), o);
            if !c.streams.contains_key(b) || !c.streams.contains_key(cc) {
                return Err(Error::unimplemented(
                    "CUDA IMAD literal operands are not ported yet",
                ));
            }
            let n = c.n(o)?;
            Ok(one(
                "k_imad",
                n,
                &format!("{}, {}, {}, {}, {}", a, b, cc, o, n),
            ))
        }
        "ISHL" | "ISHR" => {
            let (a, b, o) = (a(0), a(1), o);
            let n = c.n(o)?;
            let lit = b
                .parse::<i64>()
                .map_err(|_| Error::asm(format!("CUDA {mn}: literal shift")))?;
            if !(0..=63).contains(&lit) {
                return Err(Error::asm(format!("{mn}: shift must be 0..63, got {lit}")));
            }
            let kern = if mn == "ISHL" { "k_ishl" } else { "k_ishr" };
            Ok(one(kern, n, &format!("{}, {}, {}, {}", a, lit, o, n)))
        }
        "ISAR" => {
            let (a, b, o) = (a(0), a(1), o);
            let n = c.n(o)?;
            let check = "int *_jerr=0; CK(cudaMalloc(&_jerr,sizeof(int))); CK(cudaMemset(_jerr,0,sizeof(int)));";
            let tail = "CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); int _h=0; CK(cudaMemcpy(&_h,_jerr,sizeof(int),cudaMemcpyDeviceToHost)); cudaFree(_jerr); if(_h) return 30;";
            if c.streams.contains_key(b) {
                Ok(format!(
                    "  {{ {check} {}\n  {tail} }}",
                    launch(
                        "k_isar",
                        &grid_expr(&n.to_string()),
                        "256",
                        &format!("{}, {}, {}, {}, _jerr", a, b, o, n)
                    )
                ))
            } else {
                let lit = b
                    .parse::<i64>()
                    .map_err(|_| Error::asm("CUDA ISAR literal"))?;
                Ok(format!(
                    "  {{ {check} {}\n  {tail} }}",
                    launch(
                        "k_isar_s",
                        &grid_expr(&n.to_string()),
                        "256",
                        &format!("{}, {}, {}, {}, _jerr", a, lit, o, n)
                    )
                ))
            }
        }
        "IDIV" | "IMOD" | "IDIVU" | "IMODU" => {
            let (a, b, o) = (a(0), a(1), o);
            let n = c.n(o)?;
            let unsigned = mn == "IDIVU" || mn == "IMODU";
            let modv = if mn == "IMOD" || mn == "IMODU" { 1 } else { 0 };
            let check = "int *_jerr=0; CK(cudaMalloc(&_jerr,sizeof(int))); CK(cudaMemset(_jerr,0,sizeof(int)));";
            let tail = "CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); int _h=0; CK(cudaMemcpy(&_h,_jerr,sizeof(int),cudaMemcpyDeviceToHost)); cudaFree(_jerr); if(_h) return 30;";
            let base = if unsigned { "k_idivu" } else { "k_idiv" };
            if c.streams.contains_key(b) {
                Ok(format!(
                    "  {{ {check} {}\n  {tail} }}",
                    launch(
                        base,
                        &grid_expr(&n.to_string()),
                        "256",
                        &format!("{}, {}, {}, {}, _jerr, {}", a, b, o, n, modv)
                    )
                ))
            } else {
                let lit = b
                    .parse::<i64>()
                    .map_err(|_| Error::asm("CUDA IDIV literal"))?;
                Ok(format!(
                    "  {{ {check} {}\n  {tail} }}",
                    launch(
                        &format!("{base}_s"),
                        &grid_expr(&n.to_string()),
                        "256",
                        &format!("{}, {}, {}, {}, _jerr, {}", a, lit, o, n, modv)
                    )
                ))
            }
        }
        "ISELECT" => {
            let (cc, a, b, o) = (a(0), a(1), a(2), o);
            let n = c.n(o)?;
            let as_ = c.streams.contains_key(a);
            let bs = c.streams.contains_key(b);
            let aexpr = if as_ {
                a.to_string()
            } else {
                a.parse::<i64>()
                    .map_err(|_| Error::asm("CUDA ISELECT literal"))?
                    .to_string()
            };
            let bexpr = if bs {
                b.to_string()
            } else {
                b.parse::<i64>()
                    .map_err(|_| Error::asm("CUDA ISELECT literal"))?
                    .to_string()
            };
            let kern = match (as_, bs) {
                (true, true) => "k_iselect",
                (false, true) => "k_iselect_sa",
                (true, false) => "k_iselect_sb",
                (false, false) => "k_iselect_sab",
            };
            Ok(one(
                kern,
                n,
                &format!("{}, {}, {}, {}, {}", cc, aexpr, bexpr, o, n),
            ))
        }
        "GATHER3" => {
            let (t, ii, jj, o) = (a(0), a(1), a(2), o);
            let sh = c.s(t)?.shape.clone();
            if sh.len() != 3 {
                return Err(Error::unimplemented("CUDA GATHER3 needs 3D table"));
            }
            let rows = c.rows(jj)?;
            let w = sh[2].unwrap();
            let v2 = sh[1].unwrap();
            Ok(one(
                "k_gather3",
                rows * w,
                &format!("{}, {}, {}, {}, {}, {}, {}", t, ii, jj, o, rows, w, v2),
            ))
        }
        "ATTN" => {
            let (qr, kf, vf, mask, o) = (a(0), a(1), a(2), a(3), o);
            let nh = a(4)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA ATTN nh"))?;
            let nkv = a(5)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA ATTN nkv"))?;
            let dh = a(6)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA ATTN dh"))?;
            let sv = a(7).parse::<i64>().map_err(|_| Error::asm("CUDA ATTN S"))?;
            let neg = c.config.get_f64_or("beta", -1e9);
            let rows = c.rows(o)?;
            Ok(one(
                "k_attn",
                rows * nh as usize * dh as usize,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:?}f",
                    qr, kf, vf, mask, o, rows, nh, nkv, dh, sv, neg
                ),
            ))
        }
        other => Err(Error::unimplemented(format!(
            "cuda: no pattern for {other}"
        ))),
    }
}

fn input_decl(name: &str, s: &StreamInfo, _ftype: &str) -> String {
    match s.kind {
        CKind::F => format!("  float *h_{name} = 0, *{name} = 0;"),
        CKind::I => format!("  int64_t *h_{name} = 0, *{name} = 0;"),
        CKind::T => String::new(),
    }
}

fn elem_size(k: CKind, f16: bool) -> usize {
    match k {
        CKind::F => {
            if f16 {
                2
            } else {
                4
            }
        }
        _ => 8,
    }
}

pub fn compile_cuda(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm(
            "compile_program needs outputs=[...] (OUT streams)",
        ));
    }
    let mut bound = crate::asm::assemble(text, &opts.asm)?;
    let mut sample = sample;
    let mut outputs = opts.outputs.clone();
    sanitize_cnames(&mut bound, &mut outputs, &mut sample)?;
    let streams = infer_shapes(&bound, &sample)?;
    for s in streams.values() {
        if s.kind == CKind::T {
            return Err(Error::unimplemented(
                "CUDA: T streams are not supported (decode/quantize first)",
            ));
        }
    }
    for o in &outputs {
        if !streams.contains_key(o) {
            return Err(Error::asm(format!("CUDA: output '{o}' is not a stream")));
        }
    }
    let in_names: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    let f16 = opts.use_fp16;
    let ftype = if f16 { "__half" } else { "float" };
    let mut c = C {
        streams: streams.clone(),
        config: bound.config.clone(),
    };

    let mut src = String::new();
    src.push_str(CU_RUNTIME);
    src.push('\n');
    for n in &in_names {
        src.push_str(&input_decl(n, streams.get(n).unwrap(), ftype));
        src.push('\n');
    }
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        let s = streams.get(o).unwrap();
        match s.kind {
            CKind::F => src.push_str(&format!("float *{o} = 0;\n")),
            CKind::I => src.push_str(&format!("int64_t *{o} = 0; int64_t *h_{o} = 0;\n")),
            CKind::T => {}
        }
    }
    let mut inter: Vec<String> = Vec::new();
    for ins in &bound.instrs {
        for x in &ins.outs {
            if !in_names.contains(x) && !outputs.contains(x) && !inter.contains(x) {
                inter.push(x.clone());
            }
        }
    }
    for x in &inter {
        let s = streams.get(x).unwrap();
        match s.kind {
            CKind::F => src.push_str(&format!("float *{x} = 0;\n")),
            CKind::I => src.push_str(&format!("int64_t *{x} = 0;\n")),
            CKind::T => {}
        }
    }

    // argv count
    let mut n_argv = 0usize;
    for _n in &in_names {
        n_argv += 1;
    }
    for o in &outputs {
        n_argv += if streams.get(o).unwrap().kind == CKind::I {
            1
        } else {
            1
        };
    }
    src.push_str("\nint main(int argc, char **argv) {\n");
    src.push_str(&format!("  if (argc != {}) return 9;\n", n_argv + 1));
    let mut idx = 1usize;
    for n in &in_names {
        let s = streams.get(n).unwrap();
        let es = elem_size(s.kind, f16);
        match s.kind {
            CKind::F if f16 => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    __half *hb = (__half*)malloc(_fb); if (fread(hb,1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    long N = _fb / sizeof(__half); float *hf = (float*)malloc(N*4);
    for (long i=0;i<N;i++) hf[i] = __half2float(hb[i]);
    free(hb); CK(cudaMalloc(&{n}, N*4)); CK(cudaMemcpy({n}, hf, N*4, cudaMemcpyHostToDevice)); free(hf); }}\n"
                ));
            }
            CKind::F => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    h_{n} = (float*)malloc(_fb); if (fread(h_{n},1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    CK(cudaMalloc(&{n}, _fb)); CK(cudaMemcpy({n}, h_{n}, _fb, cudaMemcpyHostToDevice)); }}\n"
                ));
            }
            CKind::I => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    h_{n} = (int64_t*)malloc(_fb); if (fread(h_{n},1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    CK(cudaMalloc(&{n}, _fb)); CK(cudaMemcpy({n}, h_{n}, _fb, cudaMemcpyHostToDevice)); }}\n"
                ));
            }
            CKind::T => {}
        }
        let _ = es;
        idx += 1;
    }
    for x in &inter {
        let s = streams.get(x).unwrap();
        let n = s.elems().unwrap();
        let bytes = n * elem_size(s.kind, false);
        match s.kind {
            CKind::F => src.push_str(&format!("  CK(cudaMalloc(&{x}, {} * sizeof(float)));\n", n)),
            CKind::I => src.push_str(&format!(
                "  CK(cudaMalloc(&{x}, {} * sizeof(int64_t)));\n",
                n
            )),
            CKind::T => {}
        }
        let _ = bytes;
    }
    for o in &outputs {
        if in_names.contains(o) {
            continue;
        }
        let s = streams.get(o).unwrap();
        let n = s.elems().unwrap();
        match s.kind {
            CKind::F => src.push_str(&format!("  CK(cudaMalloc(&{o}, {} * sizeof(float)));\n", n)),
            CKind::I => src.push_str(&format!(
                "  CK(cudaMalloc(&{o}, {} * sizeof(int64_t)));\n",
                n
            )),
            CKind::T => {}
        }
    }

    for ins in &bound.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" => {
                src.push_str("  for (int64_t _it = 0; _it < 1; ++_it) {\n");
            }
            "__END__" => src.push_str("  }\n"),
            "__BREAK__" => {}
            _ => {
                let body = pattern(&ins.mn, &ins.outs, &ins.args, &c)?;
                src.push_str(&format!("  /* L{}: {} */\n{}\n", ins.line, ins.mn, body));
            }
        }
    }

    for o in &outputs {
        let s = streams.get(o).unwrap();
        let n = s.elems().unwrap();
        match s.kind {
            CKind::F => {
                src.push_str(&format!(
                    "  {{ float *ho = (float*)malloc({} * sizeof(float)); CK(cudaMemcpy(ho, {o}, {} * sizeof(float), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[{idx}], \"wb\"); if (!f) return 3; fwrite(ho, sizeof(float), {n}, f); fclose(f); free(ho); }}\n", n, n
                ));
            }
            CKind::I => {
                src.push_str(&format!(
                    "  {{ h_{o} = (int64_t*)malloc({} * sizeof(int64_t)); CK(cudaMemcpy(h_{o}, {o}, {} * sizeof(int64_t), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[{idx}], \"wb\"); if (!f) return 3; fwrite(h_{o}, sizeof(int64_t), {n}, f); fclose(f); free(h_{o}); }}\n", n, n
                ));
            }
            CKind::T => {}
        }
        idx += 1;
    }
    src.push_str("  return 0;\n}\n");

    Ok(Artifact {
        source: src,
        streams,
        inputs: in_names,
        outputs,
        config: bound.config.clone(),
        n_argv,
    })
}

/// Build a .cu source with nvcc.
pub fn build_cu(source: &str, workdir: &std::path::Path, name: &str) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(workdir)?;
    let src = workdir.join(format!("{name}.cu"));
    let exe = workdir.join(name);
    std::fs::write(&src, source)?;
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let nvcc_path = if std::path::Path::new(&nvcc).is_absolute() {
        std::path::PathBuf::from(&nvcc)
    } else {
        let path = std::env::var("PATH").unwrap_or_default();
        path.split(':')
            .map(|d| std::path::Path::new(d).join(&nvcc))
            .find(|p| p.is_file())
            .unwrap_or_else(|| std::path::PathBuf::from(&nvcc))
    };
    let real_nvcc = std::fs::canonicalize(&nvcc_path).unwrap_or_else(|_| nvcc_path.clone());
    let mut cmd = std::process::Command::new(&real_nvcc);
    // pip-installed CUDA toolkits do not export an include/lib prefix; derive
    // it from the nvcc location when present.
    if let Ok(real) = std::fs::canonicalize(&nvcc_path) {
        if let Some(root) = real.parent().and_then(|p| p.parent()) {
            let inc = root.join("include");
            let lib = root.join("lib");
            if inc.is_dir() {
                cmd.arg(format!("-I{}", inc.display()));
            }
            if let Some(site) = root.parent().and_then(|p| p.parent()) {
                let cccl = site.join("cuda/cccl/headers/include");
                if cccl.is_dir() {
                    cmd.arg(format!("-I{}", cccl.display()));
                }
            }
            if lib.is_dir() {
                cmd.arg(format!("-L{}", lib.display()));
                cmd.arg("-Xlinker");
                cmd.arg(format!("-rpath={}", lib.display()));
            }
            let bin = root.join("bin");
            if bin.is_dir() {
                let old = std::env::var("PATH").unwrap_or_default();
                let nvvm = root.join("nvvm/bin");
                cmd.env(
                    "PATH",
                    format!("{}:{}:{}", bin.display(), nvvm.display(), old),
                );
            }
            cmd.env("CUDA_HOME", root);
        }
    }
    let out = cmd
        .args([
            "-O2",
            "-std=c++14",
            "-Xcompiler",
            "-O2",
            "-gencode",
            "arch=compute_86,code=sm_86",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .args(["-l:libcublas.so.13", "-Xlinker", "--no-as-needed"])
        .output()?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "CUDA build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(exe)
}
