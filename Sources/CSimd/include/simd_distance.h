#ifndef SIMD_DISTANCE_H
#define SIMD_DISTANCE_H

#include <stdint.h>

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b);
int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b);
int csimd_int16_bbox_lower_bound(const int16_t *query, const int16_t *bboxMin, const int16_t *bboxMax);
float csimd_float32_l2_squared(const float *a, const float *b);
void csimd_int16_soa_block_distances(const int16_t *query, const int16_t *block, int32_t *distances);
float csimd_float32_dot(const float *a, const float *b, int n);
void csimd_prefetch(const void *ptr);

#endif
