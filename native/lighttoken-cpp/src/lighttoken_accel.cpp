#include "lighttoken_accel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace {

bool finite_array(const float *values, std::size_t len) {
    if (values == nullptr) return false;
    for (std::size_t index = 0; index < len; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

bool equal_array(const float *left, const float *right, std::size_t len) {
    for (std::size_t index = 0; index < len; ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

double norm(const float *values, std::size_t len) {
    double sum = 0.0;
    for (std::size_t index = 0; index < len; ++index) {
        const double value = static_cast<double>(values[index]);
        sum += value * value;
    }
    return std::sqrt(sum);
}

float cosine(const float *left, const float *right, std::size_t len) {
    double numerator = 0.0;
    for (std::size_t index = 0; index < len; ++index) {
        numerator += static_cast<double>(left[index]) * static_cast<double>(right[index]);
    }
    const double denominator = norm(left, len) * norm(right, len);
    if (denominator == 0.0) return equal_array(left, right, len) ? 1.0f : 0.0f;
    return static_cast<float>(numerator / denominator);
}

float correlation(const float *left, const float *right, std::size_t len) {
    double mean_left = 0.0;
    double mean_right = 0.0;
    for (std::size_t index = 0; index < len; ++index) {
        mean_left += static_cast<double>(left[index]);
        mean_right += static_cast<double>(right[index]);
    }
    mean_left /= static_cast<double>(len);
    mean_right /= static_cast<double>(len);

    double numerator = 0.0;
    double square_left = 0.0;
    double square_right = 0.0;
    for (std::size_t index = 0; index < len; ++index) {
        const double centered_left = static_cast<double>(left[index]) - mean_left;
        const double centered_right = static_cast<double>(right[index]) - mean_right;
        numerator += centered_left * centered_right;
        square_left += centered_left * centered_left;
        square_right += centered_right * centered_right;
    }
    if (square_left == 0.0 || square_right == 0.0) {
        return equal_array(left, right, len) ? 1.0f : 0.0f;
    }
    return static_cast<float>(numerator / (std::sqrt(square_left) * std::sqrt(square_right)));
}

float euclidean(const float *left, const float *right, std::size_t len) {
    double squared = 0.0;
    for (std::size_t index = 0; index < len; ++index) {
        const double delta = static_cast<double>(left[index]) - static_cast<double>(right[index]);
        squared += delta * delta;
    }
    const double max_distance = norm(left, len) + norm(right, len);
    if (max_distance == 0.0) return 1.0f;
    return static_cast<float>(1.0 - std::sqrt(squared) / max_distance);
}

template <typename Function>
int32_t similarity_many(
    const float *query,
    const float *candidates,
    std::size_t count,
    std::size_t len,
    float *out,
    Function function
) {
    if (len == 0 || query == nullptr || out == nullptr) return LT_ACCEL_INVALID_ARGUMENT;
    if (count > 0 && candidates == nullptr) return LT_ACCEL_INVALID_ARGUMENT;
    if (!finite_array(query, len)) return LT_ACCEL_NUMERIC_ERROR;

    for (std::size_t row = 0; row < count; ++row) {
        const float *candidate = candidates + row * len;
        if (!finite_array(candidate, len)) return LT_ACCEL_NUMERIC_ERROR;
        const float score = function(query, candidate, len);
        if (!std::isfinite(score)) return LT_ACCEL_NUMERIC_ERROR;
        out[row] = score;
    }
    return LT_ACCEL_OK;
}

}  // namespace

extern "C" {

uint32_t lt_accel_abi_version(void) {
    return LT_ACCEL_ABI_VERSION;
}

int32_t lt_accel_spectral_power(const float *real, const float *imag, std::size_t len, float *out) {
    if (len == 0 || real == nullptr || imag == nullptr || out == nullptr) {
        return LT_ACCEL_INVALID_ARGUMENT;
    }
    if (!finite_array(real, len) || !finite_array(imag, len)) return LT_ACCEL_NUMERIC_ERROR;
    for (std::size_t index = 0; index < len; ++index) {
        const float magnitude = std::hypot(real[index], imag[index]);
        if (!std::isfinite(magnitude)) return LT_ACCEL_NUMERIC_ERROR;
        out[index] = magnitude;
    }
    return LT_ACCEL_OK;
}

int32_t lt_accel_cosine_many(
    const float *query,
    const float *candidates,
    std::size_t count,
    std::size_t len,
    float *out
) {
    return similarity_many(query, candidates, count, len, out, cosine);
}

int32_t lt_accel_correlation_many(
    const float *query,
    const float *candidates,
    std::size_t count,
    std::size_t len,
    float *out
) {
    return similarity_many(query, candidates, count, len, out, correlation);
}

int32_t lt_accel_euclidean_many(
    const float *query,
    const float *candidates,
    std::size_t count,
    std::size_t len,
    float *out
) {
    return similarity_many(query, candidates, count, len, out, euclidean);
}

int32_t lt_accel_top_k(
    const float *scores,
    std::size_t count,
    std::size_t k,
    std::size_t *out_indices
) {
    if (scores == nullptr || out_indices == nullptr || k > count) return LT_ACCEL_INVALID_ARGUMENT;
    if (!finite_array(scores, count)) return LT_ACCEL_NUMERIC_ERROR;

    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [scores](std::size_t left, std::size_t right) {
        if (scores[left] == scores[right]) return left < right;
        return scores[left] > scores[right];
    });
    for (std::size_t index = 0; index < k; ++index) out_indices[index] = indices[index];
    return LT_ACCEL_OK;
}

int32_t lt_accel_self_test(void) {
    const float real[] = {3.0f, 0.0f};
    const float imag[] = {4.0f, 5.0f};
    float power[2] = {};
    if (lt_accel_spectral_power(real, imag, 2, power) != LT_ACCEL_OK) return LT_ACCEL_NUMERIC_ERROR;
    if (std::fabs(power[0] - 5.0f) > 1.0e-6f || std::fabs(power[1] - 5.0f) > 1.0e-6f) {
        return LT_ACCEL_NUMERIC_ERROR;
    }

    const float query[] = {1.0f, 0.0f};
    const float candidates[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float scores[2] = {};
    if (lt_accel_cosine_many(query, candidates, 2, 2, scores) != LT_ACCEL_OK) {
        return LT_ACCEL_NUMERIC_ERROR;
    }
    if (std::fabs(scores[0] - 1.0f) > 1.0e-6f || std::fabs(scores[1]) > 1.0e-6f) {
        return LT_ACCEL_NUMERIC_ERROR;
    }

    const float ranking[] = {0.5f, 0.9f, 0.9f};
    std::size_t top[2] = {};
    if (lt_accel_top_k(ranking, 3, 2, top) != LT_ACCEL_OK || top[0] != 1 || top[1] != 2) {
        return LT_ACCEL_NUMERIC_ERROR;
    }
    return LT_ACCEL_OK;
}

}  // extern "C"
