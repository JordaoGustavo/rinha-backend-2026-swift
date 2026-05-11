#include "include/simd_distance.h"

/* ------------------------------------------------------------------ */
/*  Platform detection                                                 */
/* ------------------------------------------------------------------ */

#if defined(__x86_64__) || defined(_M_X64)
  #define CSIMD_X86 1
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define CSIMD_ARM64 1
  #include <arm_neon.h>
#endif

/* ================================================================== */
/*  int16 L2 squared  (16 elements)                                    */
/* ================================================================== */

#if CSIMD_X86 && defined(__AVX2__)

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b) {
    /* Load 16 x int16 each */
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);
    __m256i diff = _mm256_sub_epi16(va, vb);

    /*
     * _mm256_madd_epi16 multiplies adjacent pairs of int16 and adds
     * the products into int32 lanes.  With (diff, diff) this gives
     * the sum of squared differences packed into 8 x int32.
     */
    __m256i sq = _mm256_madd_epi16(diff, diff);

    /* Horizontal sum of 8 x int32 */
    __m128i lo = _mm256_castsi256_si128(sq);
    __m128i hi = _mm256_extracti128_si256(sq, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);               /* 4 x int32 */
    sum128 = _mm_hadd_epi32(sum128, sum128);               /* 2 x int32 */
    sum128 = _mm_hadd_epi32(sum128, sum128);               /* 1 x int32 */
    return _mm_cvtsi128_si32(sum128);
}

#elif CSIMD_ARM64

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b) {
    /* First 8 elements */
    int16x8_t va0 = vld1q_s16(a);
    int16x8_t vb0 = vld1q_s16(b);
    int16x8_t d0  = vsubq_s16(va0, vb0);
    int32x4_t acc = vmull_s16(vget_low_s16(d0), vget_low_s16(d0));
    acc = vmlal_high_s16(acc, d0, d0);

    /* Next 8 elements */
    int16x8_t va1 = vld1q_s16(a + 8);
    int16x8_t vb1 = vld1q_s16(b + 8);
    int16x8_t d1  = vsubq_s16(va1, vb1);
    acc = vmlal_s16(acc, vget_low_s16(d1), vget_low_s16(d1));
    acc = vmlal_high_s16(acc, d1, d1);

    return vaddvq_s32(acc);
}

#else /* scalar fallback */

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b) {
    int sum = 0;
    for (int i = 0; i < 16; i++) {
        int d = (int)a[i] - (int)b[i];
        sum += d * d;
    }
    return sum;
}

#endif

/* ================================================================== */
/*  int16 L2 squared  (first 8 elements only)                         */
/* ================================================================== */

#if CSIMD_X86 && defined(__AVX2__)

int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b) {
    __m128i va = _mm_loadu_si128((const __m128i *)a);
    __m128i vb = _mm_loadu_si128((const __m128i *)b);
    __m128i diff = _mm_sub_epi16(va, vb);
    __m128i sq = _mm_madd_epi16(diff, diff);   /* 4 x int32 */
    sq = _mm_hadd_epi32(sq, sq);               /* 2 x int32 */
    sq = _mm_hadd_epi32(sq, sq);               /* 1 x int32 */
    return _mm_cvtsi128_si32(sq);
}

#elif CSIMD_ARM64

int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b) {
    int16x8_t va = vld1q_s16(a);
    int16x8_t vb = vld1q_s16(b);
    int16x8_t d  = vsubq_s16(va, vb);
    int32x4_t acc = vmull_s16(vget_low_s16(d), vget_low_s16(d));
    acc = vmlal_high_s16(acc, d, d);
    return vaddvq_s32(acc);
}

#else /* scalar fallback */

int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b) {
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        int d = (int)a[i] - (int)b[i];
        sum += d * d;
    }
    return sum;
}

#endif

/* ================================================================== */
/*  int16 bounding-box lower bound  (16 dimensions)                    */
/*                                                                     */
/*  gap[d] = max(0, bboxMin[d] - query[d])                            */
/*         + max(0, query[d]   - bboxMax[d])                           */
/*  return  sum( gap[d]^2 )                                            */
/* ================================================================== */

#if CSIMD_X86 && defined(__AVX2__)

int csimd_int16_bbox_lower_bound(const int16_t *query,
                                 const int16_t *bboxMin,
                                 const int16_t *bboxMax) {
    __m256i vq   = _mm256_loadu_si256((const __m256i *)query);
    __m256i vmin = _mm256_loadu_si256((const __m256i *)bboxMin);
    __m256i vmax = _mm256_loadu_si256((const __m256i *)bboxMax);

    __m256i zero = _mm256_setzero_si256();

    /* gap_lo = max(0, bboxMin - query) */
    __m256i gap_lo = _mm256_max_epi16(_mm256_sub_epi16(vmin, vq), zero);
    /* gap_hi = max(0, query - bboxMax) */
    __m256i gap_hi = _mm256_max_epi16(_mm256_sub_epi16(vq, vmax), zero);

    __m256i gap = _mm256_add_epi16(gap_lo, gap_hi);

    /* Sum of squares via madd */
    __m256i sq = _mm256_madd_epi16(gap, gap);

    __m128i lo = _mm256_castsi256_si128(sq);
    __m128i hi = _mm256_extracti128_si256(sq, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

#elif CSIMD_ARM64

int csimd_int16_bbox_lower_bound(const int16_t *query,
                                 const int16_t *bboxMin,
                                 const int16_t *bboxMax) {
    /* First 8 dimensions */
    int16x8_t vq0   = vld1q_s16(query);
    int16x8_t vmin0  = vld1q_s16(bboxMin);
    int16x8_t vmax0  = vld1q_s16(bboxMax);

    int16x8_t zero = vdupq_n_s16(0);

    int16x8_t gap_lo0 = vmaxq_s16(vsubq_s16(vmin0, vq0), zero);
    int16x8_t gap_hi0 = vmaxq_s16(vsubq_s16(vq0, vmax0), zero);
    int16x8_t gap0    = vaddq_s16(gap_lo0, gap_hi0);

    int32x4_t acc = vmull_s16(vget_low_s16(gap0), vget_low_s16(gap0));
    acc = vmlal_high_s16(acc, gap0, gap0);

    /* Next 8 dimensions */
    int16x8_t vq1   = vld1q_s16(query + 8);
    int16x8_t vmin1  = vld1q_s16(bboxMin + 8);
    int16x8_t vmax1  = vld1q_s16(bboxMax + 8);

    int16x8_t gap_lo1 = vmaxq_s16(vsubq_s16(vmin1, vq1), zero);
    int16x8_t gap_hi1 = vmaxq_s16(vsubq_s16(vq1, vmax1), zero);
    int16x8_t gap1    = vaddq_s16(gap_lo1, gap_hi1);

    acc = vmlal_s16(acc, vget_low_s16(gap1), vget_low_s16(gap1));
    acc = vmlal_high_s16(acc, gap1, gap1);

    return vaddvq_s32(acc);
}

#else /* scalar fallback */

int csimd_int16_bbox_lower_bound(const int16_t *query,
                                 const int16_t *bboxMin,
                                 const int16_t *bboxMax) {
    int sum = 0;
    for (int i = 0; i < 16; i++) {
        int gap = 0;
        if (query[i] < bboxMin[i])
            gap = bboxMin[i] - query[i];
        else if (query[i] > bboxMax[i])
            gap = query[i] - bboxMax[i];
        sum += gap * gap;
    }
    return sum;
}

#endif

/* ================================================================== */
/*  int16 SoA block distances — 8 vectors at once (16 padded dims)     */
/*                                                                     */
/*  SoA layout: for each dim-pair kp, 16 consecutive int16 hold        */
/*  [v0d0, v0d1, v1d0, v1d1, ..., v7d0, v7d1].                        */
/*  Output: 8 x int32 L2 squared distances.                            */
/* ================================================================== */

#if CSIMD_X86 && defined(__AVX2__)

void csimd_int16_soa_block_distances(const int16_t *query,
                                     const int16_t *block,
                                     int32_t *distances) {
    __m256i acc = _mm256_setzero_si256();
    for (int kp = 0; kp < 8; kp++) {
        __m256i vq = _mm256_set1_epi32(*(const int32_t*)(query + kp * 2));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(block + kp * 16));
        __m256i diff = _mm256_sub_epi16(vb, vq);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
    }
    _mm256_storeu_si256((__m256i *)distances, acc);
}

#else

void csimd_int16_soa_block_distances(const int16_t *query,
                                     const int16_t *block,
                                     int32_t *distances) {
    for (int v = 0; v < 8; v++) distances[v] = 0;
    for (int kp = 0; kp < 8; kp++) {
        int q0 = query[kp * 2];
        int q1 = query[kp * 2 + 1];
        for (int v = 0; v < 8; v++) {
            int d0 = block[kp * 16 + v * 2] - q0;
            int d1 = block[kp * 16 + v * 2 + 1] - q1;
            distances[v] += d0 * d0 + d1 * d1;
        }
    }
}

#endif

/* ================================================================== */
/*  float32 L2 squared  (16 elements)                                  */
/* ================================================================== */

#if CSIMD_X86 && defined(__FMA__)

float csimd_float32_l2_squared(const float *a, const float *b) {
    /* Process 16 floats in two batches of 8 */
    __m256 va0 = _mm256_loadu_ps(a);
    __m256 vb0 = _mm256_loadu_ps(b);
    __m256 d0  = _mm256_sub_ps(va0, vb0);
    __m256 acc = _mm256_mul_ps(d0, d0);

    __m256 va1 = _mm256_loadu_ps(a + 8);
    __m256 vb1 = _mm256_loadu_ps(b + 8);
    __m256 d1  = _mm256_sub_ps(va1, vb1);
    acc = _mm256_fmadd_ps(d1, d1, acc);

    /* Horizontal sum of 8 floats */
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);                       /* 4 floats */
    sum = _mm_hadd_ps(sum, sum);                           /* 2 floats */
    sum = _mm_hadd_ps(sum, sum);                           /* 1 float  */
    return _mm_cvtss_f32(sum);
}

#elif CSIMD_ARM64

float csimd_float32_l2_squared(const float *a, const float *b) {
    /* First 4 floats */
    float32x4_t va0 = vld1q_f32(a);
    float32x4_t vb0 = vld1q_f32(b);
    float32x4_t d0  = vsubq_f32(va0, vb0);
    float32x4_t acc = vmulq_f32(d0, d0);

    /* Elements 4..7 */
    float32x4_t va1 = vld1q_f32(a + 4);
    float32x4_t vb1 = vld1q_f32(b + 4);
    float32x4_t d1  = vsubq_f32(va1, vb1);
    acc = vfmaq_f32(acc, d1, d1);

    /* Elements 8..11 */
    float32x4_t va2 = vld1q_f32(a + 8);
    float32x4_t vb2 = vld1q_f32(b + 8);
    float32x4_t d2  = vsubq_f32(va2, vb2);
    acc = vfmaq_f32(acc, d2, d2);

    /* Elements 12..15 */
    float32x4_t va3 = vld1q_f32(a + 12);
    float32x4_t vb3 = vld1q_f32(b + 12);
    float32x4_t d3  = vsubq_f32(va3, vb3);
    acc = vfmaq_f32(acc, d3, d3);

    return vaddvq_f32(acc);
}

#else /* scalar fallback */

float csimd_float32_l2_squared(const float *a, const float *b) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#endif

/* ================================================================== */
/*  float32 dot product  (variable length)                             */
/* ================================================================== */

#if CSIMD_X86 && defined(__FMA__)

float csimd_float32_dot(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

#elif CSIMD_ARM64

float csimd_float32_dot(const float *a, const float *b, int n) {
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float result = vaddvq_f32(acc);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

#else

float csimd_float32_dot(const float *a, const float *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

#endif

/* ================================================================== */
/*  Prefetch                                                           */
/* ================================================================== */

void csimd_prefetch(const void *ptr) {
#if CSIMD_X86
    _mm_prefetch((const char *)ptr, _MM_HINT_T0);
#elif CSIMD_ARM64
    __builtin_prefetch(ptr, 0, 3);   /* read, high temporal locality */
#else
    (void)ptr;  /* no-op on unsupported platforms */
#endif
}
