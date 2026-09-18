#include "lighttoken_accel.h"

#include <cassert>
#include <cmath>
#include <cstddef>

static bool close(float left, float right, float tolerance = 1.0e-5f) {
    return std::fabs(left - right) <= tolerance;
}

int main() {
    assert(lt_accel_abi_version() == LT_ACCEL_ABI_VERSION);
    assert(lt_accel_self_test() == LT_ACCEL_OK);

    {
        const float real[] = {3.0f, 0.0f};
        const float imag[] = {4.0f, 5.0f};
        float output[2] = {};
        assert(lt_accel_spectral_power(real, imag, 2, output) == LT_ACCEL_OK);
        assert(close(output[0], 5.0f));
        assert(close(output[1], 5.0f));
    }

    {
        const float query[] = {1.0f, 0.0f};
        const float candidates[] = {1.0f, 0.0f, 0.0f, 1.0f};
        float output[2] = {};
        assert(lt_accel_cosine_many(query, candidates, 2, 2, output) == LT_ACCEL_OK);
        assert(close(output[0], 1.0f));
        assert(close(output[1], 0.0f));

        assert(lt_accel_euclidean_many(query, candidates, 2, 2, output) == LT_ACCEL_OK);
        assert(close(output[0], 1.0f));
        assert(close(output[1], 1.0f - std::sqrt(2.0f) / 2.0f));
    }

    {
        const float query[] = {1.0f, 2.0f, 3.0f};
        const float candidates[] = {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f};
        float output[2] = {};
        assert(lt_accel_correlation_many(query, candidates, 2, 3, output) == LT_ACCEL_OK);
        assert(close(output[0], 1.0f));
        assert(close(output[1], -1.0f));
    }

    {
        const float scores[] = {0.5f, 0.9f, 0.9f, 0.1f};
        size_t indices[2] = {};
        assert(lt_accel_top_k(scores, 4, 2, indices) == LT_ACCEL_OK);
        assert(indices[0] == 1);
        assert(indices[1] == 2);
    }

    {
        float output = 0.0f;
        assert(lt_accel_spectral_power(nullptr, nullptr, 1, &output) == LT_ACCEL_INVALID_ARGUMENT);
    }

    return 0;
}
