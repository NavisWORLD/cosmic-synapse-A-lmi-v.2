#ifndef LIGHTTOKEN_ACCEL_H
#define LIGHTTOKEN_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LT_ACCEL_EXPORT __declspec(dllexport)
#else
#define LT_ACCEL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LT_ACCEL_ABI_VERSION 1u
#define LT_ACCEL_OK 0
#define LT_ACCEL_INVALID_ARGUMENT 1
#define LT_ACCEL_NUMERIC_ERROR 2

LT_ACCEL_EXPORT uint32_t lt_accel_abi_version(void);
LT_ACCEL_EXPORT int32_t lt_accel_self_test(void);
LT_ACCEL_EXPORT int32_t lt_accel_spectral_power(const float *real, const float *imag, size_t len, float *out);
LT_ACCEL_EXPORT int32_t lt_accel_cosine_many(const float *query, const float *candidates, size_t count, size_t len, float *out);
LT_ACCEL_EXPORT int32_t lt_accel_correlation_many(const float *query, const float *candidates, size_t count, size_t len, float *out);
LT_ACCEL_EXPORT int32_t lt_accel_euclidean_many(const float *query, const float *candidates, size_t count, size_t len, float *out);
LT_ACCEL_EXPORT int32_t lt_accel_top_k(const float *scores, size_t count, size_t k, size_t *out_indices);

#ifdef __cplusplus
}
#endif

#endif
