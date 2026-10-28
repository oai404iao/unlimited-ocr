#include <immintrin.h>
#include <cmath>
#include <cstdint>

__attribute__((target("avx2,fma")))
static float squares_avx2(const float* a, int64_t width) {
    __m256 sum = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j <= width - 8; j += 8) {
        const __m256 v = _mm256_loadu_ps(a + j);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    float partial[8];
    _mm256_storeu_ps(partial, sum);
    float s = 0;
    for (float lane : partial) s += lane;
    for (; j < width; ++j) s += a[j] * a[j];
    return s;
}

__attribute__((target("avx512f,fma")))
static float squares_avx512(const float* a, int64_t width) {
    __m512 sum = _mm512_setzero_ps();
    int64_t j = 0;
    for (; j <= width - 16; j += 16) {
        const __m512 v = _mm512_loadu_ps(a + j);
        sum = _mm512_fmadd_ps(v, v, sum);
    }
    float partial[16];
    _mm512_storeu_ps(partial, sum);
    float s = 0;
    for (float lane : partial) s += lane;
    for (; j < width; ++j) s += a[j] * a[j];
    return s;
}

__attribute__((target("avx2,fma")))
static void scale_avx2(const float* a, const float* w, float* b,
                       int64_t width, float inv) {
    const __m256 scale = _mm256_set1_ps(inv);
    int64_t j = 0;
    for (; j <= width - 8; j += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(a + j), scale);
        _mm256_storeu_ps(b + j, _mm256_mul_ps(v, _mm256_loadu_ps(w + j)));
    }
    for (; j < width; ++j) b[j] = (a[j] * inv) * w[j];
}

__attribute__((target("avx512f,fma")))
static void scale_avx512(const float* a, const float* w, float* b,
                         int64_t width, float inv) {
    const __m512 scale = _mm512_set1_ps(inv);
    int64_t j = 0;
    for (; j <= width - 16; j += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(a + j), scale);
        _mm512_storeu_ps(b + j, _mm512_mul_ps(v, _mm512_loadu_ps(w + j)));
    }
    for (; j < width; ++j) b[j] = (a[j] * inv) * w[j];
}

// Contiguous FP32 rows. No allocation or persistent global workspace.
// Callers select only variants supported by their CPU and OS.
template <auto squares, auto scale>
static void rmsnorm(const float* x, const float* w, float* y,
                    int64_t rows, int64_t width, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* a = x + r * width;
        const float inv = 1.0f / std::sqrt(squares(a, width) / width + eps);
        scale(a, w, y + r * width, width, inv);
    }
}

extern "C" void rmsnorm_avx2(const float* x, const float* w, float* y,
                             int64_t rows, int64_t width, float eps) {
    rmsnorm<squares_avx2, scale_avx2>(x, w, y, rows, width, eps);
}

extern "C" void rmsnorm_avx512(const float* x, const float* w, float* y,
                               int64_t rows, int64_t width, float eps) {
    rmsnorm<squares_avx512, scale_avx512>(x, w, y, rows, width, eps);
}
