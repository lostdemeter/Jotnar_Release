//! CUDA backend (FP32 compute, naive correctness-first kernels).
//!
//! The reference CUDA backend is optimized (kernels + cuBLAS). This port is
//! deliberately simple: one launch per op, static shapes, a sync after each
//! op. It is a correctness backend first; the scheduling work is a separate
//! milestone. `CONV2D` uses im2col + cuBLAS SGEMM. T streams are refused
//! (CUDA is the F path).

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};
use crate::payload::Feeds;

use super::infer::{infer_shapes, CKind, StreamInfo};
use super::{sanitize_cnames, warm, Artifact, CompileOptions};

const CU_RUNTIME: &str = r#"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef JOTNAR_CUBLASLT
#include <cublasLt.h>
#endif
#ifdef JOTNAR_CUDNN
#include <cudnn.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(_e)); return 4; } } while (0)
#define CB(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS: %d\n", (int)_s); return 5; } } while (0)
#ifdef JOTNAR_CUDNN
#define CD(x) do { cudnnStatus_t _s = (x); if (_s != CUDNN_STATUS_SUCCESS) { \
  fprintf(stderr, "cuDNN: %s\n", cudnnGetErrorString(_s)); return 6; } } while (0)
#endif

static cublasHandle_t g_blas = NULL;
#ifdef JOTNAR_CUBLASLT
static cublasLtHandle_t g_lt = NULL;
#endif
static float *g_col = NULL;
static size_t g_col_n = 0;
#ifdef JOTNAR_CUDNN
static cudnnHandle_t g_cudnn = NULL;
static float *g_cudnn_ws = NULL;
static size_t g_cudnn_ws_n = 0;

static int ensure_cudnn_ws(size_t n) {
  if (n <= g_cudnn_ws_n) return 1;
  if (g_cudnn_ws) { cudaFree(g_cudnn_ws); g_cudnn_ws = NULL; g_cudnn_ws_n = 0; }
  if (cudaMalloc((void **)&g_cudnn_ws, n) != cudaSuccess) return 0;
  g_cudnn_ws_n = n;
  return 1;
}
#endif

static int ensure_col(size_t n) {
  if (n <= g_col_n) return 1;
  if (g_col) { cudaFree(g_col); g_col = NULL; g_col_n = 0; }
  if (cudaMalloc((void **)&g_col, n * sizeof(float)) != cudaSuccess) return 0;
  g_col_n = n;
  return 1;
}

/* One reusable N x N score tile for the head-aware SELF_ATTN path. Each head
 * is processed in order, so the tile is reused rather than allocating one per
 * head/call site. */
static float *g_self_score = NULL;
static size_t g_self_score_n = 0;
static int ensure_self_score(size_t n) {
  if (n <= g_self_score_n) return 1;
  if (g_self_score) { cudaFree(g_self_score); g_self_score = NULL; g_self_score_n = 0; }
  if (cudaMalloc((void **)&g_self_score, n * sizeof(float)) != cudaSuccess) return 0;
  g_self_score_n = n;
  return 1;
}

/* Shared HWC<->CHW scratch for the optional NCHW CONV2D path. Convolutions
 * execute sequentially, so one input and one output buffer can be reused by
 * every CONV2D call site. */
static float *g_nchw_x = NULL, *g_nchw_y = NULL;
static size_t g_nchw_xn = 0, g_nchw_yn = 0;
static int ensure_nchw_scratch(size_t xn, size_t yn) {
  if (xn > g_nchw_xn) {
    if (g_nchw_x) { cudaFree(g_nchw_x); g_nchw_x = NULL; g_nchw_xn = 0; }
    if (cudaMalloc((void **)&g_nchw_x, xn * sizeof(float)) != cudaSuccess) return 0;
    g_nchw_xn = xn;
  }
  if (yn > g_nchw_yn) {
    if (g_nchw_y) { cudaFree(g_nchw_y); g_nchw_y = NULL; g_nchw_yn = 0; }
    if (cudaMalloc((void **)&g_nchw_y, yn * sizeof(float)) != cudaSuccess) return 0;
    g_nchw_yn = yn;
  }
  return 1;
}

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
__global__ void k_matmul_t(const float *a, const float *b, float *o, int64_t m, int64_t k, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n; i += blockDim.x * gridDim.x) {
    int64_t r = i / n, c = i % n; float acc = 0.0f;
    for (int64_t t = 0; t < k; ++t) acc += a[r * k + t] * b[c * k + t];
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
  int64_t r = blockIdx.x;
  __shared__ float sh[256];
  float ss = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) ss += x[r * d + j] * x[r * d + j];
  sh[threadIdx.x] = ss; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
  float rs = rsqrtf(sh[0] / (float)d + eps);
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) o[r * d + j] = x[r * d + j] * rs * w[j];
}
__global__ void k_layernorm(const float *x, const float *w, const float *b, float *o, int64_t rows, int64_t d, float eps) {
  int64_t r = blockIdx.x;
  __shared__ float sh[256];
  float s = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) s += x[r * d + j];
  sh[threadIdx.x] = s; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
  float mu = sh[0] / (float)d;
  float ss = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) { float dv = x[r * d + j] - mu; ss += dv * dv; }
  __syncthreads(); sh[threadIdx.x] = ss; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
  float rs = rsqrtf(sh[0] / (float)d + eps);
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) o[r * d + j] = (x[r * d + j] - mu) * rs * w[j] + b[j];
}
__global__ void k_tshift(const float *x, float *o, int64_t rows, int64_t d) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float mx = x[r * d];
    for (int64_t j = 1; j < d; ++j) if (x[r * d + j] > mx) mx = x[r * d + j];
    for (int64_t j = 0; j < d; ++j) o[r * d + j] = x[r * d + j] - mx;
  }
}
__global__ void k_softmax(const float *x, float *o, int64_t rows, int64_t d) {
  int64_t r = blockIdx.x;
  __shared__ float sh[256];
  float mx = -3.402823466e+38f;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) mx = fmaxf(mx, x[r * d + j]);
  sh[threadIdx.x] = mx; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if ((int)threadIdx.x < st) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + st]); __syncthreads(); }
  mx = sh[0];
  float se = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) { float e = expf(x[r * d + j] - mx); o[r * d + j] = e; se += e; }
  __syncthreads(); sh[threadIdx.x] = se; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st]; __syncthreads(); }
  se = sh[0];
  for (int64_t j = threadIdx.x; j < d; j += blockDim.x) o[r * d + j] /= se;
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
// Row concat (axis 0): [ra, w] then [rb, w] -> [ra + rb, w]. Used by the
// DAV2 patch-embed prefix to prepend the CLS token.
__global__ void k_concat0(const float *a, const float *b, float *o, int64_t ra, int64_t rb, int64_t w) {
  int64_t na = ra * w, n = (ra + rb) * w;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    o[i] = (i < na) ? a[i] : b[i - na];
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
__global__ void k_relu(const float *x, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) o[i] = fmaxf(x[i], 0.0f);
}
// Abramowitz & Stegun 7.1.26, the same approximation the Rust target uses
// (|err| < 1.5e-7). Keeping the formula identical is what makes GELU parity
// meaningful across the two emitters.
__device__ float erf_approx_f(float x) {
  float s = x < 0.0f ? -1.0f : 1.0f;
  float ax = fabsf(x);
  float t = 1.0f / (1.0f + 0.3275911f * ax);
  float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf(-ax * ax);
  return s * y;
}
__global__ void k_gelu(const float *a, float *o, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float x = a[i];
    o[i] = 0.5f * x * (1.0f + erf_approx_f(x * 0.70710678118654752440f));
  }
}
__global__ void k_la_norm(const float *x, const float *w, const float *b, const float *temp,
                          float *o, int64_t rows, int64_t stride, int64_t dh,
                          int64_t head, int apply_temp) {
  int64_t r = blockIdx.x;
  int t = threadIdx.x;
  __shared__ float s1[64], s2[64];
  float sum = 0.0f, sq = 0.0f;
  for (int64_t i = t; i < dh; i += blockDim.x) {
    float v = x[r * stride + head * dh + i];
    sum += v; sq += v * v;
  }
  s1[t] = sum; s2[t] = sq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) { s1[t] += s1[t + s]; s2[t] += s2[t + s]; }
    __syncthreads();
  }
  float mean = s1[0] / (float)dh;
  float var = s2[0] / (float)dh - mean * mean;
  float rs = rsqrtf(var + 1e-5f);
  for (int64_t i = t; i < dh; i += blockDim.x) {
    float v = (x[r * stride + head * dh + i] - mean) * rs * w[i] + b[i];
    if (apply_temp) v *= expf(temp[head]);
    o[r * dh + i] = v;
  }
}
__global__ void k_la_softplus_bias(const float *x, const float *b, float *o,
                                   int64_t n, int64_t mod) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float v = x[i] + b[i % mod];
    o[i] = (v > 20.0f ? v : log1pf(expf(v))) + 1e-3f;
  }
}
__global__ void k_la_colsum(const float *x, float *o, int64_t rows, int64_t cols) {
  for (int64_t c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += blockDim.x * gridDim.x) {
    float acc = 0.0f;
    for (int64_t r = 0; r < rows; ++r) acc += x[r * cols + c];
    o[c] = acc;
  }
}
__global__ void k_la_den(const float *fq, const float *z, float *den,
                         int64_t rows, int64_t cols) {
  for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
    float acc = 0.0f;
    for (int64_t c = 0; c < cols; ++c) acc += fq[r * cols + c] * z[c];
    den[r] = fmaxf(acc, 1e-6f);
  }
}
__global__ void k_la_div(const float *num, const float *den, float *o,
                         int64_t n, int64_t stride, int64_t dh, int64_t head) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int64_t r = i / dh, d = i % dh;
    o[r * stride + head * dh + d] = num[i] / den[r];
  }
}
// Correctness-first transposed convolution, same loop order as the Rust
// `deconv2d_hwc` helper: x is HWC, the weight is (Cout,kh,kw,Cin), the bias is
// (Cout), and the output is HWC.
__global__ void k_deconv(const float *x, const float *wt, const float *b, float *o,
                         int64_t h, int64_t w, int64_t cin, int64_t cout,
                         int64_t kh, int64_t kw, int64_t ho, int64_t wo,
                         int64_t stride, int64_t pad) {
  int64_t total = ho * wo * cout;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    int64_t oy = i / (wo * cout), rem = i % (wo * cout), ox = rem / cout, co = rem % cout;
    float acc = b[co];
    for (int64_t ky = 0; ky < kh; ++ky) {
      int64_t iy = oy + pad - ky;
      if (iy < 0 || iy % stride) continue;
      iy /= stride;
      if (iy >= h) continue;
      for (int64_t kx = 0; kx < kw; ++kx) {
        int64_t ix = ox + pad - kx;
        if (ix < 0 || ix % stride) continue;
        ix /= stride;
        if (ix >= w) continue;
        int64_t wb = ((co * kh + ky) * kw + kx) * cin;
        int64_t x_base = (iy * w + ix) * cin;
        for (int64_t ci = 0; ci < cin; ++ci) acc += x[x_base + ci] * wt[wb + ci];
      }
    }
    o[i] = acc;
  }
}
__global__ void k_im2col(const float *x, float *col, int64_t h, int64_t w, int64_t cin,
                         int64_t kh, int64_t kw, int64_t ho, int64_t wo,
                         int64_t stride, int64_t pad) {
  int64_t K = kh * kw * cin;
  int64_t total = ho * wo * K;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    int64_t ci = i % cin, t = i / cin;
    int64_t kx = t % kw, t2 = t / kw;
    int64_t ky = t2 % kh, p = t2 / kh;
    int64_t oy = p / wo, ox = p % wo;
    int64_t iy = oy * stride + ky - pad;
    int64_t ix = ox * stride + kx - pad;
    col[i] = (iy >= 0 && iy < h && ix >= 0 && ix < w)
           ? x[(iy * w + ix) * cin + ci] : 0.0f;
  }
}
__global__ void k_add_bias(float *o, const float *b, int64_t n, int64_t cout) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    o[i] += b[i % cout];
  }
}
__global__ void k_add_bias_relu(float *o, const float *b, int64_t n, int64_t cout) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float v = o[i] + b[i % cout];
    o[i] = v > 0.0f ? v : 0.0f;
  }
}
__global__ void k_add_bias_residual(float *o, const float *b, const float *r, int64_t n, int64_t cout) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    o[i] += b[i % cout] + r[i];
  }
}
__global__ void k_add_bias_gelu(float *o, const float *b, int64_t n, int64_t cout) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float v = o[i] + b[i % cout];
    o[i] = 0.5f * v * (1.0f + erf_approx_f(v * 0.70710678118654752440f));
  }
}

/* Tiled 2D transpose: x is (r,c), o is (c,r). Used for HWC<->CHW and can
 * replace the naive TRANSPOSE kernel for conv layout conversion. */
__global__ void k_transpose_tile(const float *x, float *o, int64_t r, int64_t c) {
  __shared__ float t[32][33];
  int64_t bx = (int64_t)blockIdx.x * 32, by = (int64_t)blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  int64_t rr = by + ty, cc = bx + tx;
  if (rr < r && cc < c) t[ty][tx] = x[rr * c + cc];
  __syncthreads();
  int64_t orow = bx + ty, ocol = by + tx;
  if (orow < c && ocol < r) o[orow * r + ocol] = t[tx][ty];
}
/* CHW -> HWC transpose with the channel bias (and optional ReLU) applied in
 * the store. This replaces a separate full-tensor bias pass after every
 * NCHW conv. */
__global__ void k_transpose_tile_bias(const float *x, float *o, const float *b,
                                      int64_t r, int64_t c, int relu) {
  __shared__ float t[32][33];
  int64_t bx = (int64_t)blockIdx.x * 32, by = (int64_t)blockIdx.y * 32;
  int tx = threadIdx.x, ty = threadIdx.y;
  int64_t rr = by + ty, cc = bx + tx;
  if (rr < r && cc < c) t[ty][tx] = x[rr * c + cc];
  __syncthreads();
  int64_t orow = bx + ty, ocol = by + tx;
  if (orow < c && ocol < r) {
    float v = t[tx][ty] + b[ocol];
    if (relu && v < 0.0f) v = 0.0f;
    o[orow * r + ocol] = v;
  }
}

/* Conv weights are stored KCRS with the channel-last source layout
 * (Cout,kh,kw,Cin). cuDNN NCHW needs (Cout,Cin,kh,kw). */
__global__ void k_conv_w_nchw(const float *w, float *o,
                              int64_t cout, int64_t khkw, int64_t cin) {
  int64_t total = cout * cin * khkw;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
    int64_t co = i / (cin * khkw);
    int64_t rem = i % (cin * khkw);
    int64_t ci = rem / khkw;
    int64_t kk = rem % khkw;
    o[i] = w[(co * khkw + kk) * cin + ci];
  }
}
__global__ void k_resize2d(const float *x, float *o, int64_t h, int64_t w, int64_t c,
                           int64_t ho, int64_t wo, int64_t align) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ho * wo * c; i += blockDim.x * gridDim.x) {
    int64_t oy = i / (wo * c), rem = i % (wo * c), ox = rem / c, cc = rem % c;
    float sy = align ? (ho > 1 ? (float)oy * (float)(h - 1) / (float)(ho - 1) : 0.0f)
                     : ((float)oy + 0.5f) * (float)h / (float)ho - 0.5f;
    float sx = align ? (wo > 1 ? (float)ox * (float)(w - 1) / (float)(wo - 1) : 0.0f)
                     : ((float)ox + 0.5f) * (float)w / (float)wo - 0.5f;
    sy = sy < 0 ? 0 : (sy > (float)(h - 1) ? (float)(h - 1) : sy);
    sx = sx < 0 ? 0 : (sx > (float)(w - 1) ? (float)(w - 1) : sx);
    int64_t y0 = (int64_t)floorf(sy), x0 = (int64_t)floorf(sx);
    int64_t y1 = y0 + 1 >= h ? h - 1 : y0 + 1;
    int64_t x1 = x0 + 1 >= w ? w - 1 : x0 + 1;
    float wy = sy - (float)y0, wx = sx - (float)x0;
    o[i] = (1 - wy) * (1 - wx) * x[(y0 * w + x0) * c + cc]
         + (1 - wy) * wx * x[(y0 * w + x1) * c + cc]
         + wy * (1 - wx) * x[(y1 * w + x0) * c + cc]
         + wy * wx * x[(y1 * w + x1) * c + cc];
  }
}
__global__ void k_concat3(const float *a, const float *b, float *o,
                          int64_t h, int64_t w, int64_t ca, int64_t cb) {
  int64_t c = ca + cb;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < h * w * c; i += blockDim.x * gridDim.x) {
    int64_t cc = i % c, t = i / c, ox = t % w, oy = t / w;
    o[i] = cc < ca ? a[(oy * w + ox) * ca + cc]
                   : b[(oy * w + ox) * cb + (cc - ca)];
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
// Fused multi-head self-attention: out = softmax(scale * Q K^T) V.
// One block per (head, 32-query tile); one warp per query. For each key tile,
// each lane owns one key, so the online softmax needs one warp max/sum per
// tile rather than one per key. K is staged transposed (dh, BK) so the
// per-dimension reads are coalesced. dh is 64 (two output values per lane).
#define SA_BQ 32
#define SA_BK 32
#define SA_DH 64
__global__ __launch_bounds__(1024) void k_self_attn(const float *q, const float *k,
                                                    const float *v, float *o,
                                                    int64_t n, int nh, int dh,
                                                    float scale) {
  extern __shared__ float smem[];
  float *qs = smem;                       // SA_BQ * SA_DH
  float *ks = qs + SA_BQ * SA_DH;         // SA_DH * (SA_BK + 1), padded
  float *vs = ks + SA_DH * (SA_BK + 1);   // SA_BK * SA_DH
  float *ps = vs + SA_BK * SA_DH;         // SA_BQ * SA_BK
  int ntile = (int)((n + SA_BQ - 1) / SA_BQ);
  int head = blockIdx.x / ntile;
  int qt = blockIdx.x % ntile;
  int q0 = qt * SA_BQ;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int qi = q0 + warp;
  int H = nh * dh;
  int hb = head * dh;
  for (int idx = threadIdx.x; idx < SA_BQ * SA_DH; idx += blockDim.x) {
    int i = idx / SA_DH, d = idx % SA_DH;
    int qq = q0 + i;
    qs[idx] = (qq < n) ? q[(int64_t)qq * H + hb + d] : 0.f;
  }
  __syncthreads();
  float m = -3.402823466e+38f, l = 0.f, o0 = 0.f, o1 = 0.f;
  for (int kb = 0; kb < n; kb += SA_BK) {
    int kt = (int)((n - kb) < SA_BK ? (n - kb) : SA_BK);
    for (int idx = threadIdx.x; idx < SA_BK * SA_DH; idx += blockDim.x) {
      int j = idx / SA_DH, d = idx % SA_DH;
      int kg = kb + j;
      ks[d * (SA_BK + 1) + j] = (j < kt) ? k[(int64_t)kg * H + hb + d] : 0.f;
      vs[idx] = (j < kt) ? v[(int64_t)kg * H + hb + d] : 0.f;
    }
    __syncthreads();
    float s = -3.402823466e+38f;
    if (lane < kt) {
      s = 0.f;
      #pragma unroll
      for (int d = 0; d < SA_DH; ++d)
        s += qs[warp * SA_DH + d] * ks[d * (SA_BK + 1) + lane];
      s *= scale;
    }
    float tmax = s;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      tmax = fmaxf(tmax, __shfl_down_sync(0xffffffffu, tmax, off));
    tmax = __shfl_sync(0xffffffffu, tmax, 0);
    float mn = fmaxf(m, tmax);
    float alpha = expf(m - mn);
    float e = (lane < kt) ? expf(s - mn) : 0.f;
    float tsum = e;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      tsum += __shfl_down_sync(0xffffffffu, tsum, off);
    tsum = __shfl_sync(0xffffffffu, tsum, 0);
    l = l * alpha + tsum;
    o0 *= alpha;
    o1 *= alpha;
    ps[warp * SA_BK + lane] = e;
    __syncwarp();
    for (int j = 0; j < kt; ++j) {
      float p = ps[warp * SA_BK + j];
      o0 += p * vs[j * SA_DH + lane];
      o1 += p * vs[j * SA_DH + lane + 32];
    }
    m = mn;
    __syncthreads();
  }
  if (qi < n) {
    o[(int64_t)qi * H + hb + lane] = o0 / l;
    o[(int64_t)qi * H + hb + lane + 32] = o1 / l;
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

fn lt_epilogue(
    m: usize,
    k: usize,
    n: usize,
    a0: &str,
    a1: &str,
    b: &str,
    o: &str,
    ep: &str,
) -> String {
    format!(
        "    static cublasLtMatmulDesc_t _lt_op_{o} = NULL;\n\
         \x20   static cublasLtMatrixLayout_t _lt_a_{o} = NULL, _lt_b_{o} = NULL, _lt_c_{o} = NULL;\n\
         \x20   static cublasLtMatmulPreference_t _lt_pref_{o} = NULL;\n\
         \x20   static cublasLtMatmulHeuristicResult_t _lt_heur_{o};\n\
         \x20   static void *_lt_ws_{o} = NULL;\n\
         \x20   static size_t _lt_wsn_{o} = 0;\n\
         \x20   static int _lt_init_{o} = 0;\n\
         \x20   if (!g_lt) {{ CB(cublasLtCreate(&g_lt)); }}\n\
         \x20   if (!_lt_init_{o}) {{\n\
         \x20     CB(cublasLtMatrixLayoutCreate(&_lt_a_{o}, CUDA_R_32F, {n}, {k}, {n}));\n\
         \x20     CB(cublasLtMatrixLayoutCreate(&_lt_b_{o}, CUDA_R_32F, {k}, {m}, {k}));\n\
         \x20     CB(cublasLtMatrixLayoutCreate(&_lt_c_{o}, CUDA_R_32F, {n}, {m}, {n}));\n\
         \x20     CB(cublasLtMatmulDescCreate(&_lt_op_{o}, CUBLAS_COMPUTE_32F, CUDA_R_32F));\n\
         \x20     cublasOperation_t _tn_{o} = CUBLAS_OP_N;\n\
         \x20     CB(cublasLtMatmulDescSetAttribute(_lt_op_{o}, CUBLASLT_MATMUL_DESC_TRANSA, &_tn_{o}, sizeof(_tn_{o})));\n\
         \x20     CB(cublasLtMatmulDescSetAttribute(_lt_op_{o}, CUBLASLT_MATMUL_DESC_TRANSB, &_tn_{o}, sizeof(_tn_{o})));\n\
         \x20     cublasLtEpilogue_t _ep_{o} = {ep};\n\
         \x20     CB(cublasLtMatmulDescSetAttribute(_lt_op_{o}, CUBLASLT_MATMUL_DESC_EPILOGUE, &_ep_{o}, sizeof(_ep_{o})));\n\
         \x20     cudaDataType_t _bdt_{o} = CUDA_R_32F;\n\
         \x20     CB(cublasLtMatmulDescSetAttribute(_lt_op_{o}, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &_bdt_{o}, sizeof(_bdt_{o})));\n\
         \x20     CB(cublasLtMatmulPreferenceCreate(&_lt_pref_{o}));\n\
         \x20     size_t _maxws_{o} = 8u << 20;\n\
         \x20     CB(cublasLtMatmulPreferenceSetAttribute(_lt_pref_{o}, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &_maxws_{o}, sizeof(_maxws_{o})));\n\
         \x20     int _ret_{o} = 0;\n\
         \x20     CB(cublasLtMatmulAlgoGetHeuristic(g_lt, _lt_op_{o}, _lt_a_{o}, _lt_b_{o}, _lt_c_{o}, _lt_c_{o}, _lt_pref_{o}, 1, &_lt_heur_{o}, &_ret_{o}));\n\
         \x20     if (_ret_{o} == 0) return 5;\n\
         \x20     _lt_wsn_{o} = _lt_heur_{o}.workspaceSize;\n\
         \x20     if (_lt_wsn_{o}) CK(cudaMalloc(&_lt_ws_{o}, _lt_wsn_{o}));\n\
         \x20     _lt_init_{o} = 1;\n\
         \x20   }}\n\
         \x20   CB(cublasLtMatmulDescSetAttribute(_lt_op_{o}, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &{b}, sizeof({b})));\n\
         \x20   {{ float _alpha = 1.0f, _beta = 0.0f;\n\
         \x20     CB(cublasLtMatmul(g_lt, _lt_op_{o}, &_alpha, {a1}, _lt_a_{o}, {a0}, _lt_b_{o}, &_beta, {o}, _lt_c_{o}, {o}, _lt_c_{o}, &_lt_heur_{o}.algo, _lt_ws_{o}, _lt_wsn_{o}, 0));\n\
         \x20     CK(cudaGetLastError()); }}\n",
        m = m, k = k, n = n, a0 = a0, a1 = a1, b = b, o = o, ep = ep
    )
}

pub fn pattern(mn: &str, outs: &[String], args: &[String], c: &C) -> Result<String> {
    let o = outs.first().map(|s| s.as_str()).unwrap_or("");
    let a = |i: usize| args.get(i).map(|s| s.as_str()).unwrap_or("");
    let one = |kernel: &str, n: usize, args: &str| {
        launch(kernel, &grid_expr(&n.to_string()), "256", args)
    };
    match mn {
        "RELU" => {
            let n = c.n(o)?;
            Ok(one("k_relu", n, &format!("{}, {}, {}", a(0), o, n)))
        }
        "GELU" => {
            let n = c.n(o)?;
            Ok(one("k_gelu", n, &format!("{}, {}, {}", a(0), o, n)))
        }
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
                // cuBLAS is the fast path. The naive kernel is kept for tiny
                // matrices where the cuBLAS call/setup is not worth it.
                if m * n * k >= 4096 {
                    Ok(format!(
                        "  if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n  {{ float _alpha = 1.0f, _beta = 0.0f;\n    CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, {}, {}, &_beta, {}, {}));\n    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }}",
                        n, m, k, a(1), n, a(0), k, o, n
                    ))
                } else {
                    Ok(one(
                        "k_matmul",
                        m * n,
                        &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n),
                    ))
                }
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
        "MATMUL_T" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            if ash.len() != 2 || bsh.len() != 2 {
                return Err(Error::unimplemented("CUDA MATMUL_T needs 2D operands"));
            }
            let (Some(m), Some(k), Some(n)) = (ash[0], ash[1], bsh[0]) else {
                return Err(Error::unimplemented("CUDA MATMUL_T needs static shapes"));
            };
            if bsh[1] != Some(k) {
                return Err(Error::asm("CUDA MATMUL_T inner dims mismatch"));
            }
            if m * n * k >= 4096 {
                Ok(format!(
                    "  if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n  {{ float _alpha = 1.0f, _beta = 0.0f;\n    CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, {}, {}, &_beta, {}, {}));\n    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }}",
                    n, m, k, a(1), k, a(0), k, o, n
                ))
            } else {
                Ok(one(
                    "k_matmul_t",
                    m * n,
                    &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n),
                ))
            }
        }
        "MATMUL_BIAS" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            if ash.len() != 2 || bsh.len() != 2 {
                return Err(Error::unimplemented("CUDA MATMUL_BIAS needs 2D operands"));
            }
            let (Some(m), Some(k), Some(n)) = (ash[0], ash[1], bsh[1]) else {
                return Err(Error::unimplemented("CUDA MATMUL_BIAS needs static shapes"));
            };
            if bsh[0] != Some(k) {
                return Err(Error::asm("CUDA MATMUL_BIAS inner dims mismatch"));
            }
            let fallback = if m * n * k >= 4096 {
                format!(
                    "  if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n  {{ float _alpha = 1.0f, _beta = 0.0f;\n    CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, {}, {}, &_beta, {}, {}));\n    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }}\n  k_add_bias<<<{}, 256>>>({}, {}, {}, {});\n  CK(cudaGetLastError());",
                    n, m, k, a(1), n, a(0), k, o, n,
                    grid_expr(&(m * n).to_string()),
                    o, a(2), m * n, n
                )
            } else {
                format!(
                    "{}  k_add_bias<<<{}, 256>>>({}, {}, {}, {});\n  CK(cudaGetLastError());",
                    one(
                        "k_matmul",
                        m * n,
                        &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n)
                    ),
                    grid_expr(&(m * n).to_string()),
                    o,
                    a(2),
                    m * n,
                    n
                )
            };
            let cublaslt = c
                .config
                .get("cublaslt")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !cublaslt || m * n * k < 4096 {
                return Ok(fallback);
            }
            let lt = lt_epilogue(m, k, n, a(0), a(1), a(2), o, "CUBLASLT_EPILOGUE_BIAS");
            Ok(format!(
                "  {{\n#ifdef JOTNAR_CUBLASLT\n{lt}\n#else\n{fallback}\n#endif\n  }}"
            ))
        }
        "MATMUL_BIAS_RESIDUAL" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            if ash.len() != 2 || bsh.len() != 2 {
                return Err(Error::unimplemented(
                    "CUDA MATMUL_BIAS_RESIDUAL needs 2D operands",
                ));
            }
            let (Some(m), Some(k), Some(n)) = (ash[0], ash[1], bsh[1]) else {
                return Err(Error::unimplemented(
                    "CUDA MATMUL_BIAS_RESIDUAL needs static shapes",
                ));
            };
            if bsh[0] != Some(k) {
                return Err(Error::asm("CUDA MATMUL_BIAS_RESIDUAL inner dims mismatch"));
            }
            let gemm = if m * n * k >= 4096 {
                format!(
                    "  if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n  {{ float _alpha = 1.0f, _beta = 0.0f;\n    CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, {}, {}, &_beta, {}, {}));\n    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }}",
                    n, m, k, a(1), n, a(0), k, o, n
                )
            } else {
                one(
                    "k_matmul",
                    m * n,
                    &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n),
                )
            };
            Ok(format!(
                "{}  k_add_bias_residual<<<{}, 256>>>({}, {}, {}, {}, {});\n  CK(cudaGetLastError());",
                gemm, grid_expr(&(m * n).to_string()), o, a(2), a(3), m * n, n
            ))
        }
        "MATMUL_BIAS_GELU" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            if ash.len() != 2 || bsh.len() != 2 {
                return Err(Error::unimplemented(
                    "CUDA MATMUL_BIAS_GELU needs 2D operands",
                ));
            }
            let (Some(m), Some(k), Some(n)) = (ash[0], ash[1], bsh[1]) else {
                return Err(Error::unimplemented(
                    "CUDA MATMUL_BIAS_GELU needs static shapes",
                ));
            };
            if bsh[0] != Some(k) {
                return Err(Error::asm("CUDA MATMUL_BIAS_GELU inner dims mismatch"));
            }
            let fallback = if m * n * k >= 4096 {
                format!(
                    "  if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}
  {{ float _alpha = 1.0f, _beta = 0.0f;
    CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, {}, {}, &_beta, {}, {}));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }}
  k_add_bias_gelu<<<{}, 256>>>({}, {}, {}, {});
  CK(cudaGetLastError());",
                    n, m, k, a(1), n, a(0), k, o, n,
                    grid_expr(&(m * n).to_string()), o, a(2), m * n, n
                )
            } else {
                format!(
                    "{}  k_add_bias_gelu<<<{}, 256>>>({}, {}, {}, {});
  CK(cudaGetLastError());",
                    one(
                        "k_matmul",
                        m * n,
                        &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, m, k, n)
                    ),
                    grid_expr(&(m * n).to_string()),
                    o,
                    a(2),
                    m * n,
                    n
                )
            };
            let cublaslt = c
                .config
                .get("cublaslt")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !cublaslt || m * n * k < 4096 {
                return Ok(fallback);
            }
            let lt = lt_epilogue(m, k, n, a(0), a(1), a(2), o, "CUBLASLT_EPILOGUE_GELU_BIAS");
            Ok(format!(
                "  {{
#ifdef JOTNAR_CUBLASLT
{lt}
#else
{fallback}
#endif
  }}"
            ))
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
                Ok(format!(
                    "  k_rmsnorm<<<{}, 256>>>({}, {}, {}, {}, {}, {:?}f);\n  CK(cudaGetLastError());\n  CK(cudaDeviceSynchronize());",
                    rows, a(0), a(1), o, rows, d, eps
                ))
            } else {
                Ok(format!(
                    "  k_layernorm<<<{}, 256>>>({}, {}, {}, {}, {}, {}, {:?}f);\n  CK(cudaGetLastError());\n  CK(cudaDeviceSynchronize());",
                    rows, a(0), a(1), a(2), o, rows, d, eps
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
            Ok(format!(
                "  k_softmax<<<{}, 256>>>({}, {}, {}, {});\n  CK(cudaGetLastError());\n  CK(cudaDeviceSynchronize());",
                rows, a(0), o, rows, d
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
        "RESHAPE2" | "RESHAPE3" => {
            if c.s(o)?.kind != CKind::F {
                return Err(Error::unimplemented("CUDA RESHAPE needs F streams"));
            }
            let n = c.n(o)?;
            Ok(format!(
                "  CK(cudaMemcpy({o}, {}, {} * sizeof(float), cudaMemcpyDeviceToDevice));\n  CK(cudaDeviceSynchronize());",
                a(0),
                n
            ))
        }
        "CONCAT" => {
            let ash = c.s(a(0))?.shape.clone();
            let bsh = c.s(a(1))?.shape.clone();
            let axis = a(2)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA CONCAT: bad axis literal"))?;
            if ash.len() == 3 && bsh.len() == 3 && axis == 2 {
                let (h, w) = (ash[0].unwrap(), ash[1].unwrap());
                let ca = ash[2].unwrap();
                let cb = bsh[2].unwrap();
                return Ok(one(
                    "k_concat3",
                    h * w * (ca + cb),
                    &format!("{}, {}, {}, {}, {}, {}, {}", a(0), a(1), o, h, w, ca, cb),
                ));
            }
            let wa = c.w(a(0))?;
            let wb = c.w(a(1))?;
            let rows = c.rows(a(0))?;
            if axis == 0 {
                let ra = c.rows(a(0))?;
                let rb = c.rows(a(1))?;
                return Ok(one(
                    "k_concat0",
                    (ra + rb) * wa,
                    &format!("{}, {}, {}, {}, {}, {}", a(0), a(1), o, ra, rb, wa),
                ));
            }
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
        "CONV2D" | "CONV2D_RELU" => {
            let xsh = c.s(a(0))?.shape.clone();
            let wsh = c.s(a(1))?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented("CUDA CONV2D needs HWC + 4D weight"));
            }
            let (Some(h), Some(w), Some(cin)) = (xsh[0], xsh[1], xsh[2]) else {
                return Err(Error::unimplemented("CUDA CONV2D: dynamic input shape"));
            };
            let (Some(cout), Some(kh), Some(kw)) = (wsh[0], wsh[1], wsh[2]) else {
                return Err(Error::unimplemented("CUDA CONV2D: dynamic weight shape"));
            };
            let stride = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA CONV2D: bad stride literal"))?;
            let pad = a(4)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA CONV2D: bad pad literal"))?;
            let ho = ((h as i64 + 2 * pad - kh as i64) / stride + 1) as usize;
            let wo = ((w as i64 + 2 * pad - kw as i64) / stride + 1) as usize;
            let p = ho * wo;
            let k = kh * kw * cin;
            let n = p * cout;
            let bias_kernel = if mn == "CONV2D_RELU" {
                "k_add_bias_relu"
            } else {
                "k_add_bias"
            };
            // cuDNN NHWC is the fast path when the toolkit has cuDNN. The
            // fallback is the dependency-free im2col + cuBLAS correctness
            // path. Both produce HWC output. The f32 path stays on
            // IMPLICIT_GEMM: Winograd is unsupported for NHWC here and
            // IMPLICIT_PRECOMP_GEMM's fast mode is not f32-exact. Descriptors
            // and the workspace limit are cached per call site so the serve
            // loop does not recreate them every frame.
            let nchw = c
                .config
                .get("cudnn_nchw")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            // Opt-in winograd for the stride-1 3x3 f32 NCHW convolutions.
            // It changes the reduction order, so it is gated by the depth
            // parity card rather than enabled by default.
            let winograd = c
                .config
                .get("cudnn_winograd")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let algo = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
            let cudnn = format!(
                "    static cudnnTensorDescriptor_t _xd = NULL, _yd = NULL;\n\
                 \x20   static cudnnFilterDescriptor_t _wd = NULL;\n\
                 \x20   static cudnnConvolutionDescriptor_t _cd = NULL;\n\
                 \x20   static cudnnConvolutionFwdAlgo_t _algo = {algo};\n\
                 \x20   static size_t _ws = 0;\n\
                 \x20   static int _init = 0;\n\
                 \x20   if (!g_cudnn) {{ CD(cudnnCreate(&g_cudnn)); }}\n\
                 \x20   if (!_init) {{\n\
                 \x20     CD(cudnnCreateTensorDescriptor(&_xd)); CD(cudnnCreateTensorDescriptor(&_yd));\n\
                 \x20     CD(cudnnCreateFilterDescriptor(&_wd)); CD(cudnnCreateConvolutionDescriptor(&_cd));\n\
                 \x20     CD(cudnnSetTensor4dDescriptor(_xd, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, {}, {}, {}));\n\
                 \x20     CD(cudnnSetTensor4dDescriptor(_yd, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, {}, {}, {}));\n\
                 \x20     CD(cudnnSetFilter4dDescriptor(_wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, {}, {}, {}, {}));\n\
                 \x20     CD(cudnnSetConvolution2dDescriptor(_cd, {}, {}, {}, {}, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));\n\
                 \x20     cudnnStatus_t _st = cudnnGetConvolutionForwardWorkspaceSize(g_cudnn, _xd, _wd, _cd, _yd, _algo, &_ws);\n\
                 \x20     if (_st != CUDNN_STATUS_SUCCESS) {{ _algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; CD(cudnnGetConvolutionForwardWorkspaceSize(g_cudnn, _xd, _wd, _cd, _yd, _algo, &_ws)); }}\n\
                 \x20     _init = 1;\n\
                 \x20   }}\n\
                 \x20   if (!ensure_cudnn_ws(_ws)) return 4;\n\
                 \x20   float _a = 1.0f, _b = 0.0f;\n\
                 \x20   CD(cudnnConvolutionForward(g_cudnn, &_a, _xd, {}, _wd, {}, _cd, _algo, g_cudnn_ws, _ws, &_b, _yd, {}));\n\
                 \x20   CK(cudaDeviceSynchronize());\n\
                 \x20   {bias_kernel}<<<{}, 256>>>({}, {}, {}, {});\n\
                 \x20   CK(cudaGetLastError());",
                cin, h, w,
                cout, ho, wo,
                cout, cin, kh, kw,
                pad, pad, stride, stride,
                a(0), a(1), o,
                grid_expr(&n.to_string()),
                o, a(2), n, cout,
                bias_kernel = bias_kernel
            );
            let nchw_code = if nchw {
                let p = h * w;
                let pout = ho * wo;
                let nchw_algo = if winograd && kh == 3 && kw == 3 && stride == 1 {
                    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"
                } else if kh <= 3 && kw <= 3 {
                    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
                } else {
                    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
                };
                format!(
                    "    static float *_wn_{o} = NULL;\n\
                     \x20   static int _winit_{o} = 0;\n\
                     \x20   static cudnnTensorDescriptor_t _xd_{o} = NULL, _yd_{o} = NULL;\n\
                     \x20   static cudnnFilterDescriptor_t _wd_{o} = NULL;\n\
                     \x20   static cudnnConvolutionDescriptor_t _cd_{o} = NULL;\n\
                     \x20   static cudnnConvolutionFwdAlgo_t _algo_{o} = {nchw_algo};\n\
                     \x20   static size_t _ws_{o} = 0;\n\
                     \x20   static int _init_{o} = 0;\n\
                     \x20   if (!ensure_nchw_scratch({cin} * {p}, {cout} * {pout})) return 4;\n\
                     \x20   if (!_winit_{o}) {{\n\
                     \x20     if (cudaMalloc((void **)&_wn_{o}, {wn} * sizeof(float)) != cudaSuccess) return 4;\n\
                     \x20     k_conv_w_nchw<<<{wgrid}, 256>>>({w}, _wn_{o}, {cout}, {khkw}, {cin});\n\
                     \x20     CK(cudaGetLastError()); _winit_{o} = 1;\n\
                     \x20   }}\n\
                     \x20   k_transpose_tile<<<dim3(({cin} + 31) / 32, ({p} + 31) / 32), dim3(32, 32)>>>({x}, g_nchw_x, {p}, {cin});\n\
                     \x20   CK(cudaGetLastError());\n\
                     \x20   if (!g_cudnn) {{ CD(cudnnCreate(&g_cudnn)); }}\n\
                     \x20   if (!_init_{o}) {{\n\
                     \x20     CD(cudnnCreateTensorDescriptor(&_xd_{o})); CD(cudnnCreateTensorDescriptor(&_yd_{o}));\n\
                     \x20     CD(cudnnCreateFilterDescriptor(&_wd_{o})); CD(cudnnCreateConvolutionDescriptor(&_cd_{o}));\n\
                     \x20     CD(cudnnSetTensor4dDescriptor(_xd_{o}, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, {cin}, {h}, {w_in}));\n\
                     \x20     CD(cudnnSetTensor4dDescriptor(_yd_{o}, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, {cout}, {ho}, {wo}));\n\
                     \x20     CD(cudnnSetFilter4dDescriptor(_wd_{o}, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, {cout}, {cin}, {kh}, {kw}));\n\
                     \x20     CD(cudnnSetConvolution2dDescriptor(_cd_{o}, {pad}, {pad}, {stride}, {stride}, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));\n\
                     \x20     cudnnSetConvolutionMathType(_cd_{o}, CUDNN_FMA_MATH);\n\
                     \x20     cudnnStatus_t _st_{o} = cudnnGetConvolutionForwardWorkspaceSize(g_cudnn, _xd_{o}, _wd_{o}, _cd_{o}, _yd_{o}, _algo_{o}, &_ws_{o});\n\
                     \x20     if (_st_{o} != CUDNN_STATUS_SUCCESS) {{ _algo_{o} = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; CD(cudnnGetConvolutionForwardWorkspaceSize(g_cudnn, _xd_{o}, _wd_{o}, _cd_{o}, _yd_{o}, _algo_{o}, &_ws_{o})); }}\n\
                     \x20     _init_{o} = 1;\n\
                     \x20   }}\n\
                     \x20   if (!ensure_cudnn_ws(_ws_{o})) return 4;\n\
                     \x20   float _a_{o} = 1.0f, _b_{o} = 0.0f;\n\
                     \x20   CD(cudnnConvolutionForward(g_cudnn, &_a_{o}, _xd_{o}, g_nchw_x, _wd_{o}, _wn_{o}, _cd_{o}, _algo_{o}, g_cudnn_ws, _ws_{o}, &_b_{o}, _yd_{o}, g_nchw_y));\n\
                     \x20   k_transpose_tile_bias<<<dim3(({pout} + 31) / 32, ({cout} + 31) / 32), dim3(32, 32)>>>(g_nchw_y, {o}, {b}, {cout}, {pout}, {relu});\n\
                     \x20   CK(cudaGetLastError());",
                    relu = if mn == "CONV2D_RELU" { 1 } else { 0 },
                    o = o,
                    nchw_algo = nchw_algo,
                    p = p,
                    pout = pout,
                    wn = cout * kh * kw * cin,
                    wgrid = grid_expr(&(cout * kh * kw * cin).to_string()),
                    cout = cout,
                    khkw = kh * kw,
                    cin = cin,
                    x = a(0),
                    h = h,
                    w_in = w,
                    w = a(1),
                    ho = ho,
                    wo = wo,
                    kh = kh,
                    kw = kw,
                    pad = pad,
                    stride = stride,
                    b = a(2)
                )
            } else {
                cudnn
            };
            let fallback = format!(
                "    if (!ensure_col({})) return 4;\n\
                 \x20   k_im2col<<<{}, 256>>>({}, g_col, {}, {}, {}, {}, {}, {}, {}, {}, {});\n\
                 \x20   CK(cudaGetLastError());\n\
                 \x20   if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n\
                 \x20   float _alpha = 1.0f, _beta = 0.0f;\n\
                 \x20   CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, {}, {}, {}, &_alpha, {}, {}, g_col, {}, &_beta, {}, {}));\n\
                 \x20   {bias_kernel}<<<{}, 256>>>({}, {}, {}, {});\n\
                 \x20   CK(cudaGetLastError()); CK(cudaDeviceSynchronize());",
                p * k,
                grid_expr(&(p * k).to_string()),
                a(0), h, w, cin, kh, kw, ho, wo, stride, pad,
                cout, p, k, a(1), k, k, o, cout,
                grid_expr(&n.to_string()),
                o, a(2), n, cout,
                bias_kernel = bias_kernel
            );
            Ok(format!(
                "  {{\n#ifdef JOTNAR_CUDNN\n{}\n#else\n{}\n#endif\n  }}",
                nchw_code, fallback
            ))
        }
        "DECONV2D" => {
            let xsh = c.s(a(0))?.shape.clone();
            let wsh = c.s(a(1))?.shape.clone();
            if xsh.len() != 3 || wsh.len() != 4 {
                return Err(Error::unimplemented("CUDA DECONV2D needs HWC + 4D weight"));
            }
            let (Some(h), Some(w), Some(cin)) = (xsh[0], xsh[1], xsh[2]) else {
                return Err(Error::unimplemented("CUDA DECONV2D: dynamic input shape"));
            };
            let (Some(cout), Some(kh), Some(kw)) = (wsh[0], wsh[1], wsh[2]) else {
                return Err(Error::unimplemented("CUDA DECONV2D: dynamic weight shape"));
            };
            let stride = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA DECONV2D: bad stride literal"))?;
            let pad = a(4)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA DECONV2D: bad pad literal"))?;
            if stride <= 0 || pad < 0 {
                return Err(Error::asm("CUDA DECONV2D: bad stride/pad"));
            }
            let ho = ((h as i64 - 1) * stride - 2 * pad + kh as i64) as usize;
            let wo = ((w as i64 - 1) * stride - 2 * pad + kw as i64) as usize;
            let n = ho * wo * cout;
            Ok(one(
                "k_deconv",
                n,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                    a(0),
                    a(1),
                    a(2),
                    o,
                    h,
                    w,
                    cin,
                    cout,
                    kh,
                    kw,
                    ho,
                    wo,
                    stride,
                    pad
                ),
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
        "RESIZE2D" => {
            let sh = c.s(a(0))?.shape.clone();
            if sh.len() != 3 {
                return Err(Error::unimplemented("CUDA RESIZE2D needs HWC"));
            }
            let (h, w, cch) = (sh[0].unwrap(), sh[1].unwrap(), sh[2].unwrap());
            let ho = a(1)
                .parse::<usize>()
                .map_err(|_| Error::asm("CUDA RESIZE2D: bad ho literal"))?;
            let wo = a(2)
                .parse::<usize>()
                .map_err(|_| Error::asm("CUDA RESIZE2D: bad wo literal"))?;
            let align = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA RESIZE2D: bad align literal"))?;
            Ok(one(
                "k_resize2d",
                ho * wo * cch,
                &format!(
                    "{}, {}, {}, {}, {}, {}, {}, {}",
                    a(0),
                    o,
                    h,
                    w,
                    cch,
                    ho,
                    wo,
                    align
                ),
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
        "LINEAR_ATTN" => {
            let (q, k, v, o) = (a(0), a(1), a(2), o);
            let w1 = a(3);
            let b1 = a(4);
            let w2 = a(5);
            let b2 = a(6);
            let lnq_w = a(7);
            let lnq_b = a(8);
            let lnk_w = a(9);
            let lnk_b = a(10);
            let temp = a(11);
            let nh = a(12)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA LINEAR_ATTN nh"))?;
            let dh = a(13)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA LINEAR_ATTN dh"))?;
            if nh <= 0 || dh <= 0 {
                return Err(Error::asm("CUDA LINEAR_ATTN: bad nh/dh"));
            }
            let qsh = c.s(q)?.shape.clone();
            let ksh = c.s(k)?.shape.clone();
            let vsh = c.s(v)?.shape.clone();
            if qsh.len() != 2 || ksh != qsh || vsh != qsh {
                return Err(Error::unimplemented(
                    "CUDA LINEAR_ATTN needs equal 2D Q/K/V",
                ));
            }
            let (Some(n), Some(h)) = (qsh[0], qsh[1]) else {
                return Err(Error::unimplemented("CUDA LINEAR_ATTN: dynamic shape"));
            };
            if h != (nh * dh) as usize {
                return Err(Error::asm(format!(
                    "CUDA LINEAR_ATTN: width {h} != nh*dh {}",
                    nh * dh
                )));
            }
            let w1sh = c.s(w1)?.shape.clone();
            let w2sh = c.s(w2)?.shape.clone();
            if w1sh.len() != 2 || w1sh[0] != Some(dh as usize) {
                return Err(Error::unimplemented(
                    "CUDA LINEAR_ATTN: W1 must be (dh,hid)",
                ));
            }
            let hid =
                w1sh[1].ok_or_else(|| Error::unimplemented("CUDA LINEAR_ATTN: dynamic hid"))?;
            if w2sh.len() != 2 || w2sh[0] != Some(hid) {
                return Err(Error::unimplemented("CUDA LINEAR_ATTN: W2 must be (hid,m)"));
            }
            let m = w2sh[1].ok_or_else(|| Error::unimplemented("CUDA LINEAR_ATTN: dynamic m"))?;
            let qn = n * dh as usize;
            let a_sz = n * hid;
            let f_sz = n * m;
            let ft_sz = m * n;
            let s_sz = m * dh as usize;
            let num_sz = n * dh as usize;
            let g_nhid = grid_expr(&a_sz.to_string());
            let g_nm = grid_expr(&f_sz.to_string());
            let g_m = grid_expr(&m.to_string());
            let g_n = grid_expr(&n.to_string());
            let g_ndh = grid_expr(&num_sz.to_string());
            let template = r#"  {
    static float *la_qn = NULL, *la_kn = NULL, *la_a = NULL, *la_fq = NULL;
    static float *la_fk = NULL, *la_fkT = NULL, *la_s = NULL, *la_z = NULL;
    static float *la_num = NULL, *la_den = NULL;
    if (!la_qn) {
      CK(cudaMalloc(&la_qn, @QN@ * sizeof(float)));
      CK(cudaMalloc(&la_kn, @QN@ * sizeof(float)));
      CK(cudaMalloc(&la_a, @ASZ@ * sizeof(float)));
      CK(cudaMalloc(&la_fq, @FSZ@ * sizeof(float)));
      CK(cudaMalloc(&la_fk, @FSZ@ * sizeof(float)));
      CK(cudaMalloc(&la_fkT, @FTSZ@ * sizeof(float)));
      CK(cudaMalloc(&la_s, @SSZ@ * sizeof(float)));
      CK(cudaMalloc(&la_z, @M@ * sizeof(float)));
      CK(cudaMalloc(&la_num, @NUMSZ@ * sizeof(float)));
      CK(cudaMalloc(&la_den, @N@ * sizeof(float)));
    }
    if (!g_blas) { CB(cublasCreate(&g_blas)); }
    float _la_a = 1.0f, _la_b = 0.0f;
    for (int64_t head = 0; head < @NH@; ++head) {
      k_la_norm<<<@N@, 64>>>(@Q@, @LNQW@, @LNQB@, @TEMP@, la_qn, @N@, @H@, @DH@, head, 1);
      k_la_norm<<<@N@, 64>>>(@K@, @LNKW@, @LNKB@, @TEMP@, la_kn, @N@, @H@, @DH@, head, 0);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @HID@, @N@, @DH@, &_la_a, @W1@, @HID@, la_qn, @DH@, &_la_b, la_a, @HID@));
      k_gelu<<<@GNHID@, 256>>>(la_a, la_a, @ASZ@);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @M@, @N@, @HID@, &_la_a, @W2@, @M@, la_a, @HID@, &_la_b, la_fq, @M@));
      k_la_softplus_bias<<<@GNM@, 256>>>(la_fq, @B2@, la_fq, @FSZ@, @M@);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @HID@, @N@, @DH@, &_la_a, @W1@, @HID@, la_kn, @DH@, &_la_b, la_a, @HID@));
      k_gelu<<<@GNHID@, 256>>>(la_a, la_a, @ASZ@);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @M@, @N@, @HID@, &_la_a, @W2@, @M@, la_a, @HID@, &_la_b, la_fk, @M@));
      k_la_softplus_bias<<<@GNM@, 256>>>(la_fk, @B2@, la_fk, @FSZ@, @M@);
      k_transpose<<<@GNM@, 256>>>(la_fk, la_fkT, @N@, @M@);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @DH@, @M@, @N@, &_la_a, @V@ + head * @DH@, @H@, la_fkT, @N@, &_la_b, la_s, @DH@));
      k_la_colsum<<<@GM@, 256>>>(la_fk, la_z, @N@, @M@);
      CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, @DH@, @N@, @M@, &_la_a, la_s, @DH@, la_fq, @M@, &_la_b, la_num, @DH@));
      k_la_den<<<@GN@, 256>>>(la_fq, la_z, la_den, @N@, @M@);
      k_la_div<<<@GNDH@, 256>>>(la_num, la_den, @O@, @NDH@, @H@, @DH@, head);
    }
    CK(cudaGetLastError());
  }"#;
            let mut body = template.to_string();
            for (key, val) in [
                ("@QN@", qn.to_string()),
                ("@ASZ@", a_sz.to_string()),
                ("@FSZ@", f_sz.to_string()),
                ("@FTSZ@", ft_sz.to_string()),
                ("@SSZ@", s_sz.to_string()),
                ("@NUMSZ@", num_sz.to_string()),
                ("@NDH@", num_sz.to_string()),
                ("@N@", n.to_string()),
                ("@H@", h.to_string()),
                ("@DH@", dh.to_string()),
                ("@NH@", nh.to_string()),
                ("@HID@", hid.to_string()),
                ("@M@", m.to_string()),
                ("@Q@", q.to_string()),
                ("@K@", k.to_string()),
                ("@V@", v.to_string()),
                ("@O@", o.to_string()),
                ("@W1@", w1.to_string()),
                ("@B1@", b1.to_string()),
                ("@W2@", w2.to_string()),
                ("@B2@", b2.to_string()),
                ("@LNQW@", lnq_w.to_string()),
                ("@LNQB@", lnq_b.to_string()),
                ("@LNKW@", lnk_w.to_string()),
                ("@LNKB@", lnk_b.to_string()),
                ("@TEMP@", temp.to_string()),
                ("@GNHID@", g_nhid.clone()),
                ("@GNM@", g_nm.clone()),
                ("@GM@", g_m.clone()),
                ("@GN@", g_n.clone()),
                ("@GNDH@", g_ndh.clone()),
            ] {
                body = body.replace(key, &val);
            }
            Ok(body)
        }
        "SELF_ATTN" => {
            let (q, k, v, o) = (a(0), a(1), a(2), o);
            let nh = a(3)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SELF_ATTN nh"))?;
            let dh = a(4)
                .parse::<i64>()
                .map_err(|_| Error::asm("CUDA SELF_ATTN dh"))?;
            let scale = a(5)
                .parse::<f64>()
                .map_err(|_| Error::asm("CUDA SELF_ATTN scale"))?;
            if nh <= 0 || dh <= 0 || !scale.is_finite() {
                return Err(Error::asm("CUDA SELF_ATTN: bad nh/dh/scale"));
            }
            if dh != 64 {
                return Err(Error::unimplemented("CUDA SELF_ATTN currently needs dh=64"));
            }
            let qsh = c.s(q)?.shape.clone();
            let ksh = c.s(k)?.shape.clone();
            let vsh = c.s(v)?.shape.clone();
            if qsh.len() != 2 || ksh != qsh || vsh != qsh {
                return Err(Error::unimplemented("CUDA SELF_ATTN needs equal 2D Q/K/V"));
            }
            let (Some(n), Some(h)) = (qsh[0], qsh[1]) else {
                return Err(Error::unimplemented("CUDA SELF_ATTN: dynamic shape"));
            };
            if h != (nh * dh) as usize {
                return Err(Error::asm(format!(
                    "CUDA SELF_ATTN: width {h} != nh*dh {}",
                    nh * dh
                )));
            }
            // Head-aware per-head cuBLAS path. Q/K/V are the full (N, NH*DH)
            // streams; each head is an offset/strided view. One N*N score tile
            // is reused across heads, and PV writes directly into the full
            // output columns. This removes the 216 slices, 72 transposes and
            // 60 head-concat copies without changing the attention math.
            Ok(format!(
                "  if (!ensure_self_score((size_t){n} * (size_t){n})) return 4;\n\
                 \x20 if (!g_blas) {{ CB(cublasCreate(&g_blas)); }}\n\
                 \x20 for (int _h = 0; _h < {nh}; ++_h) {{\n\
                 \x20   float _alpha = {scale:?}f, _beta = 0.0f;\n\
                 \x20   CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,\n\
                 \x20     {n}, {n}, {dh}, &_alpha,\n\
                 \x20     {k} + (size_t)_h * {dh}, {h},\n\
                 \x20     {q} + (size_t)_h * {dh}, {h},\n\
                 \x20     &_beta, g_self_score, {n}));\n\
                 \x20   k_softmax<<<{n}, 256>>>(g_self_score, g_self_score, {n}, {n});\n\
                 \x20   CK(cudaGetLastError());\n\
                 \x20   float _pone = 1.0f, _pzero = 0.0f;\n\
                 \x20   CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,\n\
                 \x20     {dh}, {n}, {n}, &_pone,\n\
                 \x20     {v} + (size_t)_h * {dh}, {h},\n\
                 \x20     g_self_score, {n},\n\
                 \x20     &_pzero, {o} + (size_t)_h * {dh}, {h}));\n\
                 \x20 }}\n\
                 \x20 CK(cudaGetLastError());",
                n = n,
                nh = nh,
                dh = dh,
                h = h,
                q = q,
                k = k,
                v = v,
                o = o,
                scale = scale,
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

/// Optional liveness-based CUDA intermediate pool.
///
/// Mirrors the language-level stream arena: one allocation, offsets instead of
/// pointers, but with interval reuse so dead intermediates do not consume
/// peak memory. Opt-in via `CONFIG cuda_arena 1` or `JOTNAR_ARENA=1` at build
/// time. Inputs stay separate (they are loaded from argv and, for `--serve`,
/// overwritten from stdin); instruction outputs, including graph outputs, live
/// in the pool.
#[derive(Clone, Copy)]
struct ArenaSlot {
    off: usize,
    kind: CKind,
}

struct ArenaPlan {
    pool_bytes: usize,
    slots: HashMap<String, ArenaSlot>,
}

fn plan_cuda_arena(
    bound: &crate::ir::BoundProgram,
    streams: &HashMap<String, StreamInfo>,
    in_names: &[String],
    outputs: &[String],
    f16: bool,
) -> Option<ArenaPlan> {
    if f16 || bound.instrs.is_empty() {
        return None;
    }
    let n_instr = bound.instrs.len();
    let output_set: HashSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    let mut def: HashMap<&str, usize> = HashMap::new();
    let mut last: HashMap<&str, usize> = HashMap::new();
    for (i, ins) in bound.instrs.iter().enumerate() {
        if ins.is_repeat() || ins.is_end() || ins.is_break() {
            return None;
        }
        if ins.outs.len() > 1 {
            return None; // multi-output ops may expect contiguous outputs
        }
        for o in &ins.outs {
            if def.insert(o.as_str(), i).is_some() {
                return None; // reassignment: not a simple graph
            }
            if ins.args.iter().any(|a| a == o) {
                return None; // in-place op: liveness would alias
            }
        }
        for a in &ins.args {
            last.insert(a.as_str(), i);
        }
    }
    let mut intervals: Vec<(usize, usize, usize, &str, CKind)> = Vec::new();
    for (i, ins) in bound.instrs.iter().enumerate() {
        for o in &ins.outs {
            if in_names.iter().any(|x| x == o) {
                continue;
            }
            let s = streams.get(o)?;
            if s.kind == CKind::T {
                return None;
            }
            let elems = s.elems()?;
            let size = elems * elem_size(s.kind, false);
            let end = if output_set.contains(o.as_str()) {
                n_instr
            } else {
                last.get(o.as_str()).copied().unwrap_or(i) + 1
            };
            intervals.push((i, end, size, o.as_str(), s.kind));
        }
    }
    intervals.sort_by_key(|x| x.0);
    let align = |x: usize| (x + 15) & !15;
    let mut active: Vec<(usize, usize, usize)> = Vec::new(); // (end, off, size)
    let mut pool = 0usize;
    let mut slots = HashMap::new();
    for (d, e, size, name, kind) in intervals {
        active.retain(|(end, _, _)| *end > d);
        let mut occupied: Vec<(usize, usize)> = active
            .iter()
            .map(|(_, off, sz)| (*off, *off + *sz))
            .collect();
        occupied.sort_unstable();
        let mut gaps = Vec::new();
        let mut cursor = 0usize;
        for (a, b) in occupied {
            if a > cursor {
                gaps.push((cursor, a - cursor));
            }
            cursor = cursor.max(b);
        }
        if pool > cursor {
            gaps.push((cursor, pool - cursor));
        }
        let mut off = None;
        for (g, len) in &gaps {
            let a = align(*g);
            if a + size <= g + len {
                off = Some(a);
                break;
            }
        }
        let off = off.unwrap_or_else(|| {
            let a = align(pool);
            pool = a + size;
            a
        });
        if off + size > pool {
            pool = off + size;
        }
        active.push((e, off, size));
        slots.insert(name.to_string(), ArenaSlot { off, kind });
    }
    Some(ArenaPlan {
        pool_bytes: align(pool),
        slots,
    })
}

pub fn compile_cuda(text: &str, opts: &CompileOptions, sample: Feeds) -> Result<Artifact> {
    if opts.outputs.is_empty() {
        return Err(Error::asm(
            "compile_program needs outputs=[...] (OUT streams)",
        ));
    }
    let mut bound = crate::asm::assemble(text, &opts.asm)?;
    let source_inputs: Vec<String> = bound.inputs.iter().map(|d| d.name.clone()).collect();
    let source_outputs = opts.outputs.clone();
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
    let serve_in = bound.config.get("serve_input").map(str::to_string);
    let serve_out = bound.config.get("serve_output").map(str::to_string);
    if serve_in.is_some() != serve_out.is_some() {
        return Err(Error::asm(
            "CUDA serve mode needs both CONFIG serve_input and CONFIG serve_output",
        ));
    }
    if let Some(n) = &serve_in {
        if !in_names.contains(n) {
            return Err(Error::asm("CUDA serve_input must be an IN stream"));
        }
        let s = streams.get(n).unwrap();
        if s.kind != CKind::F || opts.use_fp16 {
            return Err(Error::unimplemented(
                "CUDA serve_input currently needs F32 (not fp16)",
            ));
        }
    }
    if let Some(n) = &serve_out {
        if !outputs.contains(n) {
            return Err(Error::asm("CUDA serve_output must be an output"));
        }
        let s = streams.get(n).unwrap();
        if s.kind != CKind::F {
            return Err(Error::unimplemented("CUDA serve_output must be F32"));
        }
    }
    let serve = serve_in.is_some();
    let profile = bound
        .config
        .get("profile")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let warm_frame = bound
        .config
        .get("warm_frame")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if warm_frame {
        if serve {
            return Err(Error::asm(
                "CUDA CONFIG warm_frame and serve_* are mutually exclusive",
            ));
        }
        if opts.use_fp16 {
            return Err(Error::unimplemented(
                "CUDA warm_frame does not support --fp16 yet",
            ));
        }
        if profile {
            return Err(Error::unimplemented(
                "CUDA warm_frame does not support CONFIG profile yet",
            ));
        }
        for (name, s) in &streams {
            if s.kind == CKind::T {
                return Err(Error::unimplemented(format!(
                    "CUDA warm_frame does not support T stream {name:?}"
                )));
            }
            if s.shape.iter().any(|d| d.is_none()) {
                return Err(Error::unimplemented(format!(
                    "CUDA warm_frame needs static shapes: {name:?}"
                )));
            }
        }
    }
    let nosync = bound
        .config
        .get("nosync")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut prof_names: Vec<String> = Vec::new();
    if profile {
        for ins in &bound.instrs {
            if ins.mn.starts_with("__") {
                continue;
            }
            if !prof_names.iter().any(|n| n == &ins.mn) {
                prof_names.push(ins.mn.clone());
            }
        }
    }
    let f16 = opts.use_fp16;
    let arena_enabled = bound
        .config
        .get("cuda_arena")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || std::env::var("JOTNAR_ARENA").ok().as_deref() == Some("1");
    if warm_frame && arena_enabled {
        return Err(Error::unimplemented(
            "CUDA warm_frame does not support cuda_arena yet",
        ));
    }
    let arena = if arena_enabled {
        plan_cuda_arena(&bound, &streams, &in_names, &outputs, f16)
    } else {
        None
    };
    let ftype = if f16 { "__half" } else { "float" };
    let mut c = C {
        streams: streams.clone(),
        config: bound.config.clone(),
    };

    let mut src = String::new();
    src.push_str(CU_RUNTIME);
    src.push('\n');
    if profile {
        src.push_str("#include <time.h>\n");
        src.push_str(
            "static double _jnow(void){ struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); return (double)_ts.tv_sec*1000.0 + (double)_ts.tv_nsec/1e6; }\n",
        );
        src.push_str(&format!(
            "static double _jacc[{}]; static long long _jcnt[{}];\n",
            prof_names.len(),
            prof_names.len()
        ));
        src.push_str("static const char *_jnames[] = {");
        for (i, n) in prof_names.iter().enumerate() {
            if i > 0 {
                src.push(',');
            }
            src.push_str(&format!("\"{n}\""));
        }
        src.push_str("};\n");
    }
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
    let serve_argc = 2 + in_names.len();
    let normal_argc = n_argv + 1;
    if warm_frame {
        src.push_str("int warm_main(int argc, char **argv);\n");
    }
    src.push_str("\nint main(int argc, char **argv) {\n");
    if warm_frame {
        src.push_str(
            "  if (argc >= 2 && strcmp(argv[1], \"--warm-frame\") == 0) return warm_main(argc, argv);\n",
        );
    }
    src.push_str("  bool __serve = argc >= 2 && strcmp(argv[1], \"--serve\") == 0;\n");
    src.push_str("  int argi = __serve ? 2 : 1;\n");
    if arena.is_some() {
        src.push_str("  char *g_cuda_arena = NULL;\n");
    }
    src.push_str(&format!(
        "  if ((__serve && argc != {serve_argc}) || (!__serve && argc != {normal_argc})) return 9;\n"
    ));
    for n in &in_names {
        let s = streams.get(n).unwrap();
        let es = elem_size(s.kind, f16);
        match s.kind {
            CKind::F if f16 => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[argi++], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    __half *hb = (__half*)malloc(_fb); if (fread(hb,1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    long N = _fb / sizeof(__half); float *hf = (float*)malloc(N*4);
    for (long i=0;i<N;i++) hf[i] = __half2float(hb[i]);
    free(hb); CK(cudaMalloc(&{n}, N*4)); CK(cudaMemcpy({n}, hf, N*4, cudaMemcpyHostToDevice)); free(hf); }}\n"
                ));
            }
            CKind::F => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[argi++], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    h_{n} = (float*)malloc(_fb); if (fread(h_{n},1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    CK(cudaMalloc(&{n}, _fb)); CK(cudaMemcpy({n}, h_{n}, _fb, cudaMemcpyHostToDevice)); }}\n"
                ));
            }
            CKind::I => {
                src.push_str(&format!(
                    "  {{ FILE *f = fopen(argv[argi++], \"rb\"); if (!f) return 1; fseek(f,0,SEEK_END); long _fb=ftell(f); fseek(f,0,SEEK_SET);
    h_{n} = (int64_t*)malloc(_fb); if (fread(h_{n},1,_fb,f)!=(size_t)_fb) return 2; fclose(f);
    CK(cudaMalloc(&{n}, _fb)); CK(cudaMemcpy({n}, h_{n}, _fb, cudaMemcpyHostToDevice)); }}\n"
                ));
            }
            CKind::T => {}
        }
        let _ = es;
    }
    if let Some(plan) = &arena {
        src.push_str(&format!(
            "  CK(cudaMalloc((void **)&g_cuda_arena, {}));\n",
            plan.pool_bytes
        ));
    }
    for x in &inter {
        let s = streams.get(x).unwrap();
        let n = s.elems().unwrap();
        if let Some(plan) = &arena {
            if let Some(slot) = plan.slots.get(x) {
                match slot.kind {
                    CKind::F => src.push_str(&format!(
                        "  {x} = (float *)(g_cuda_arena + {});\n",
                        slot.off
                    )),
                    CKind::I => src.push_str(&format!(
                        "  {x} = (int64_t *)(g_cuda_arena + {});\n",
                        slot.off
                    )),
                    CKind::T => {}
                }
                continue;
            }
        }
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
        if let Some(plan) = &arena {
            if let Some(slot) = plan.slots.get(o) {
                match slot.kind {
                    CKind::F => src.push_str(&format!(
                        "  {o} = (float *)(g_cuda_arena + {});\n",
                        slot.off
                    )),
                    CKind::I => src.push_str(&format!(
                        "  {o} = (int64_t *)(g_cuda_arena + {});\n",
                        slot.off
                    )),
                    CKind::T => {}
                }
                continue;
            }
        }
        match s.kind {
            CKind::F => src.push_str(&format!("  CK(cudaMalloc(&{o}, {} * sizeof(float)));\n", n)),
            CKind::I => src.push_str(&format!(
                "  CK(cudaMalloc(&{o}, {} * sizeof(int64_t)));\n",
                n
            )),
            CKind::T => {}
        }
    }

    let mut ops = String::new();
    for ins in &bound.instrs {
        match ins.mn.as_str() {
            "__REPEAT__" => {
                ops.push_str("  for (int64_t _it = 0; _it < 1; ++_it) {\n");
            }
            "__END__" => ops.push_str("  }\n"),
            "__BREAK__" => {}
            _ => {
                let body = pattern(&ins.mn, &ins.outs, &ins.args, &c)?;
                ops.push_str(&format!("  /* L{}: {} */\n", ins.line, ins.mn));
                if profile {
                    let idx = prof_names
                        .iter()
                        .position(|n| n == &ins.mn)
                        .expect("profiled op name");
                    ops.push_str("  { double _jt0 = _jnow();\n");
                    ops.push_str(&body);
                    ops.push_str(&format!(
                        "\n  _jacc[{idx}] += _jnow() - _jt0; _jcnt[{idx}]++; }}\n"
                    ));
                } else {
                    ops.push_str(&body);
                    ops.push('\n');
                }
            }
        }
    }

    if nosync && !profile {
        // One frame-boundary sync (the output memcpy) instead of one per op.
        // Host reads (output copies, ISAR/IDIV error checks) still synchronize
        // implicitly, so results are unchanged.
        ops = ops.replace("CK(cudaDeviceSynchronize());", "");
    }

    if serve {
        let si = serve_in.as_deref().unwrap();
        let so = serve_out.as_deref().unwrap();
        let n_in = streams.get(si).unwrap().elems().unwrap();
        let n_out = streams.get(so).unwrap().elems().unwrap();
        let pipe = format!(
            "  {{ float *__hin = (float*)malloc({n_in} * sizeof(float));\n\
             \x20 float *__hout = (float*)malloc({n_out} * sizeof(float));\n\
             \x20 while (fread(__hin, sizeof(float), {n_in}, stdin) == {n_in}) {{\n\
             \x20   CK(cudaMemcpy({si}, __hin, {n_in} * sizeof(float), cudaMemcpyHostToDevice));\n\
             {ops}\
             \x20   CK(cudaDeviceSynchronize());\n\
             \x20   CK(cudaMemcpy(__hout, {so}, {n_out} * sizeof(float), cudaMemcpyDeviceToHost));\n\
             \x20   fwrite(__hout, sizeof(float), {n_out}, stdout); fflush(stdout);\n\
             \x20 }}\n\
             \x20 free(__hin); free(__hout); }}\n"
        );
        let shm = c
            .config
            .get("serve_shm")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !shm {
            src.push_str(&pipe);
        } else {
            // Optional shared-memory frame protocol. The worker maps the two
            // files and sends one-byte ready tokens; the artifact keeps the
            // pipe only for the token handshake.
            let shm_loop = format!(
                "  {{ const char *__shm_in = getenv(\"JOTNAR_SHM_IN\");\n\
                 \x20 const char *__shm_out = getenv(\"JOTNAR_SHM_OUT\");\n\
                 \x20 if (__shm_in && __shm_out) {{\n\
                 \x20   int __sfi = open(__shm_in, O_RDWR);\n\
                 \x20   int __sfo = open(__shm_out, O_RDWR);\n\
                 \x20   if (__sfi < 0 || __sfo < 0) {{ fprintf(stderr, \"SHM open failed\\n\"); return 4; }}\n\
                 \x20   void *__smi = mmap(NULL, (size_t){n_in} * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, __sfi, 0);\n\
                 \x20   void *__smo = mmap(NULL, (size_t){n_out} * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, __sfo, 0);\n\
                 \x20   if (__smi == MAP_FAILED || __smo == MAP_FAILED) {{ fprintf(stderr, \"SHM mmap failed\\n\"); return 4; }}\n\
                 \x20   unsigned char __tok = 1;\n\
                 \x20   while (read(0, &__tok, 1) == 1) {{\n\
                 \x20     CK(cudaMemcpy({si}, __smi, {n_in} * sizeof(float), cudaMemcpyHostToDevice));\n\
                 {ops}\
                 \x20     CK(cudaDeviceSynchronize());\n\
                 \x20     CK(cudaMemcpy(__smo, {so}, {n_out} * sizeof(float), cudaMemcpyDeviceToHost));\n\
                 \x20     if (write(1, &__tok, 1) != 1) break;\n\
                 \x20   }}\n\
                 \x20   munmap(__smi, (size_t){n_in} * sizeof(float)); munmap(__smo, (size_t){n_out} * sizeof(float));\n\
                 \x20   close(__sfi); close(__sfo); return 0;\n\
                 \x20 }}\n\
                 \x20 {pipe}\n\
                 \x20 }}\n"
            );
            src.push_str(&shm_loop);
        }
    } else {
        src.push_str(&ops);
        src.push_str("  CK(cudaDeviceSynchronize());\n");
        for o in &outputs {
            let s = streams.get(o).unwrap();
            let n = s.elems().unwrap();
            match s.kind {
                CKind::F => {
                    src.push_str(&format!(
                        "  {{ float *ho = (float*)malloc({} * sizeof(float)); CK(cudaMemcpy(ho, {o}, {} * sizeof(float), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[argi++], \"wb\"); if (!f) return 3; fwrite(ho, sizeof(float), {n}, f); fclose(f); free(ho); }}\n", n, n
                    ));
                }
                CKind::I => {
                    src.push_str(&format!(
                        "  {{ h_{o} = (int64_t*)malloc({} * sizeof(int64_t)); CK(cudaMemcpy(h_{o}, {o}, {} * sizeof(int64_t), cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[argi++], \"wb\"); if (!f) return 3; fwrite(h_{o}, sizeof(int64_t), {n}, f); fclose(f); free(h_{o}); }}\n", n, n
                    ));
                }
                CKind::T => {}
            }
        }
    }
    if profile {
        src.push_str(&format!(
            "  for (int _ji = 0; _ji < {}; ++_ji) if (_jcnt[_ji]) fprintf(stderr, \"PROFILE %-12s calls=%lld ms=%.3f\\n\", _jnames[_ji], (long long)_jcnt[_ji], _jacc[_ji]);\n",
            prof_names.len()
        ));
    }
    src.push_str("  return 0;\n}\n");

    if warm_frame {
        let inputs: Vec<warm::WarmBuffer> = in_names
            .iter()
            .map(|n| {
                let s = streams.get(n).unwrap();
                warm::WarmBuffer {
                    name: n.clone(),
                    kind: s.kind,
                    elems: s.elems().unwrap(),
                }
            })
            .collect();
        let inters: Vec<warm::WarmBuffer> = inter
            .iter()
            .map(|n| {
                let s = streams.get(n).unwrap();
                warm::WarmBuffer {
                    name: n.clone(),
                    kind: s.kind,
                    elems: s.elems().unwrap(),
                }
            })
            .collect();
        let outs: Vec<warm::WarmBuffer> = outputs
            .iter()
            .map(|n| {
                let s = streams.get(n).unwrap();
                warm::WarmBuffer {
                    name: n.clone(),
                    kind: s.kind,
                    elems: s.elems().unwrap(),
                }
            })
            .collect();
        let api = warm::WarmApi {
            target: "cuda",
            malloc: "cudaMalloc",
            memcpy: "cudaMemcpy",
            host_to_device: "cudaMemcpyHostToDevice",
            device_to_host: "cudaMemcpyDeviceToHost",
            sync: "cudaDeviceSynchronize",
        };
        src.push_str(&warm::emit_main(&api, &inputs, &inters, &outs, &ops));
    }

    Ok(Artifact {
        source: src,
        streams,
        inputs: in_names,
        outputs,
        source_inputs,
        source_outputs,
        config: bound.config.clone(),
        n_argv,
        warm_frame,
    })
}

/// Normalize a CUDA architecture name for `-gencode`.
///
/// Accepts `sm_120`, `compute_120`, or a bare `120`; the default is `sm_86`
/// so existing hosts keep their current code generation.
fn cuda_arch_args_for(arch: &str) -> Vec<String> {
    let trimmed = arch.trim();
    let n = trimmed
        .strip_prefix("sm_")
        .or_else(|| trimmed.strip_prefix("compute_"))
        .unwrap_or(trimmed);
    let n = if n.is_empty() { "86" } else { n };
    vec![
        "-gencode".to_string(),
        format!("arch=compute_{n},code=sm_{n}"),
    ]
}

fn cuda_arch_args() -> Vec<String> {
    cuda_arch_args_for(&std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string()))
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
    let mut cudnn = false;
    let mut cublaslt = false;
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
            if inc.join("cublasLt.h").is_file() && lib.join("libcublasLt.so.13").is_file() {
                cublaslt = true;
            }
            if std::env::var("JOTNAR_CUDNN").ok().as_deref() != Some("0") {
                if let Some(croot) = root.parent().map(|p| p.join("cudnn")) {
                    let cinc = croot.join("include");
                    let clib = croot.join("lib");
                    if cinc.join("cudnn.h").is_file() && clib.join("libcudnn.so.9").is_file() {
                        cmd.arg(format!("-I{}", cinc.display()));
                        cmd.arg(format!("-L{}", clib.display()));
                        cmd.arg("-Xlinker");
                        cmd.arg(format!("-rpath={}", clib.display()));
                        cmd.arg("-DJOTNAR_CUDNN");
                        cudnn = true;
                    }
                }
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
    cmd.args(["-O2", "-std=c++14", "-Xcompiler", "-O2"]);
    cmd.args(cuda_arch_args());
    cmd.arg(&src)
        .arg("-o")
        .arg(&exe)
        .args(["-l:libcublas.so.13", "-Xlinker", "--no-as-needed"]);
    if cublaslt {
        cmd.arg("-DJOTNAR_CUBLASLT");
        cmd.arg("-l:libcublasLt.so.13");
    }
    if cudnn {
        cmd.arg("-l:libcudnn.so.9");
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(Error::asm(format!(
            "CUDA build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(exe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_arch_args_normalize() {
        assert_eq!(
            cuda_arch_args_for("sm_120"),
            vec![
                "-gencode".to_string(),
                "arch=compute_120,code=sm_120".to_string()
            ]
        );
        assert_eq!(
            cuda_arch_args_for("compute_90"),
            vec![
                "-gencode".to_string(),
                "arch=compute_90,code=sm_90".to_string()
            ]
        );
        assert_eq!(
            cuda_arch_args_for("86"),
            vec![
                "-gencode".to_string(),
                "arch=compute_86,code=sm_86".to_string()
            ]
        );
        assert_eq!(
            cuda_arch_args_for(""),
            vec![
                "-gencode".to_string(),
                "arch=compute_86,code=sm_86".to_string()
            ]
        );
    }
}
