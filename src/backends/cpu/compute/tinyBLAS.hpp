// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#ifndef MLLM_TINYBLAS_HPP
#define MLLM_TINYBLAS_HPP

#include "Types.hpp"
#include "backends/cpu/quantize/Quantize.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <atomic>
#include <array>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace {

inline float unhalf(mllm_fp16_t d) {
    return MLLM_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) {
    return _mm_add_ps(x, y);
}
inline __m128 sub(__m128 x, __m128 y) {
    return _mm_sub_ps(x, y);
}
inline __m128 mul(__m128 x, __m128 y) {
    return _mm_mul_ps(x, y);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 add(__m256 x, __m256 y) {
    return _mm256_add_ps(x, y);
}
inline __m256 sub(__m256 x, __m256 y) {
    return _mm256_sub_ps(x, y);
}
inline __m256 mul(__m256 x, __m256 y) {
    return _mm256_mul_ps(x, y);
}
#endif // __AVX__

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) {
    return _mm512_add_ps(x, y);
}
inline __m512 sub(__m512 x, __m512 y) {
    return _mm512_sub_ps(x, y);
}
inline __m512 mul(__m512 x, __m512 y) {
    return _mm512_mul_ps(x, y);
}
#endif // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) {
    return vaddq_f32(x, y);
}
inline float32x4_t sub(float32x4_t x, float32x4_t y) {
    return vsubq_f32(x, y);
}
inline float32x4_t mul(float32x4_t x, float32x4_t y) {
    return vmulq_f32(x, y);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) {
    return vaddq_f16(x, y);
}
inline float16x8_t sub(float16x8_t x, float16x8_t y) {
    return vsubq_f16(x, y);
}
inline float16x8_t mul(float16x8_t x, float16x8_t y) {
    return vmulq_f16(x, y);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__MMA__)
typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

#if defined(__FMA__)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512BF16__)
template <>
inline __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    return _mm512_dpbf16_ps(c, a, b);
}
template <>
inline __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    return _mm256_dpbf16_ps(c, a, b);
}
#endif
#endif

#if defined(__ARM_FEATURE_FMA)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, b, a);
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
template <>
inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vfmaq_f16(c, b, a);
}
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED SCALAR MULTIPLY-ADD

/*
 * compute a * b + c
 * where b is a scalar
 * */

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 smadd(__m128 a, float b, __m128 c) {
    return madd(a, _mm_set1_ps(b), c);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 smadd(__m256 a, float b, __m256 c) {
    return madd(a, _mm256_set1_ps(b), c);
}
#endif // __AVX__

#if defined(__AVX512F__)
inline __m512 smadd(__m512 a, float b, __m512 c) {
    return madd(a, _mm512_set1_ps(b), c);
}
#endif // __AVX512F__

#if defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FMA)
inline float32x4_t smadd(float32x4_t a, float b, float32x4_t c) {
    return vfmaq_n_f32(c, a, b);
}
#else
inline float32x4_t smadd(float32x4_t a, float b, float32x4_t c) {
    float32x4_t b_vec = vdupq_n_f32(b);
    return vaddq_f32(vmulq_f32(a, b_vec), c);
}
#endif // __ARM_FEATURE_FMA
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if defined(__ARM_FEATURE_FMA) && !defined(_MSC_VER)
inline float16x8_t smadd(float16x8_t a, __fp16 b, float16x8_t c) {
    return vfmaq_n_f16(c, a, b);
}
#else
inline float16x8_t smadd(float16x8_t a, __fp16 b, float16x8_t c) {
    float16x8_t b_vec = vdupq_n_f16(b);
    return vaddq_f16(vmulq_f16(a, b_vec), c);
}
#endif // __ARM_FEATURE_FMA
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
inline float hsum(float16x8_t x) {
    return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)),
                                vcvt_f32_f16(vget_high_f16(x))));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U>
T load(const U *);

#if defined(__ARM_NEON)
template <>
inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
#if !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float16x8_t load(const mllm_fp16_t *p) {
    return vld1q_f16((const float16_t *)p);
}
template <>
inline float32x4_t load(const mllm_fp16_t *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if (defined(__AVX2__) || defined(__AVX512F__)) && MLLM_BF16_ENABLED
template <>
inline __m256 load(const mllm_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif // __AVX2__

#if defined(__F16C__)
template <>
inline __m256 load(const mllm_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <>
inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <>
inline __m512 load(const mllm_fp16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#if MLLM_BF16_ENABLED
template <>
inline __m512 load(const mllm_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif // MLLM_BF16_ENABLED
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <>
inline __m512bh load(const mllm_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
template <>
inline __m256bh load(const mllm_bf16_t *p) {
    return (__m256bh)_mm256_loadu_ps((const float *)p);
}
template <>
inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <>
inline __m256bh load(const float *p) {
    return _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY STORING

template <typename T, typename U>
void store(U *, T);

#if defined(__ARM_NEON)
template <>
inline void store(float *p, float32x4_t v) {
    vst1q_f32(p, v);
}
#if !defined(_MSC_VER)
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline void store(mllm_fp16_t *p, float16x8_t v) {
    vst1q_f16((float16_t *)p, v);
}
template <>
inline void store(mllm_fp16_t *p, float32x4_t v) {
    vst1_f16((float16_t *)p, vcvt_f16_f32(v));
}
template <>
inline void store(float *p, float16x8_t v) {
    float32x4_t v_low  = vcvt_f32_f16(vget_low_f16(v));
    float32x4_t v_high = vcvt_f32_f16(vget_high_f16(v));

    vst1q_f32(p,     v_low);
    vst1q_f32(p + 4, v_high);
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline void store(float *p, __m128 v) {
    _mm_storeu_ps(p, v);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline void store(float *p, __m256 v) {
    _mm256_storeu_ps(p, v);
}
#endif // __AVX__

#if (defined(__AVX2__) || defined(__AVX512F__)) && MLLM_BF16_ENABLED
template <>
inline void store(mllm_bf16_t *p, __m256 v) {
    __m256i i32 = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    __m128i i16 = _mm256_cvtepi32_epi16(i32);
    _mm_storeu_si128((__m128i *)p, i16);
}
#endif // __AVX2__

#if defined(__F16C__)
template <>
inline void store(mllm_fp16_t *p, __m256 v) {
    _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(v, _MM_FROUND_CUR_DIRECTION));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <>
inline void store(float *p, __m512 v) {
    _mm512_storeu_ps(p, v);
}
template <>
inline void store(mllm_fp16_t *p, __m512 v) {
    _mm256_storeu_si256((__m256i *)p, _mm512_cvtps_ph(v, _MM_FROUND_CUR_DIRECTION));
}
#if MLLM_BF16_ENABLED
template <>
inline void store(mllm_bf16_t *p, __m512 v) {
    __m512i i32 = _mm512_srli_epi32(_mm512_castps_si512(v), 16);
    __m256i i16 = _mm512_cvtepi32_epi16(i32);
    _mm256_storeu_si256((__m256i *)p, i16);
}
#endif // MLLM_BF16_ENABLED
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <>
inline void store(mllm_bf16_t *p, __m512bh v) {
    _mm512_storeu_ps((float *)p, _mm512_castsi512_ps((__m512i)v));
}
template <>
inline void store(mllm_bf16_t *p, __m256bh v) {
    _mm256_storeu_ps((float *)p, _mm256_castsi256_ps((__m256i)v));
}
template <>
inline void store(float *p, __m512bh v) {
    _mm512_storeu_ps(p, _mm512_castsi512_ps((__m512i)v));
}
#endif // __AVX512BF16__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ASSIGNMENT

/**
 * Assigns scalar value to all elements of vector register.
 */
template <typename T, typename U>
T assign(U);

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m128 assign(float scalar) {
    return _mm_set1_ps(scalar);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 assign(float scalar) {
    return _mm256_set1_ps(scalar);
}
#endif // __AVX__

#if defined(__AVX512F__)
template <>
inline __m512 assign(float scalar) {
    return _mm512_set1_ps(scalar);
}
#endif // __AVX512F__

#if defined(__ARM_NEON)
template <>
inline float32x4_t assign(float scalar) {
    return vdupq_n_f32(scalar);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
inline float16x8_t assign(mllm_fp16_t scalar) {
    return vdupq_n_f16(scalar);
}
#endif // __ARM_FP16

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED SCALING

/**
 * Scales vector elements by scalar value.
 */
template <typename T, typename U>
inline T scale(T a, U scalar){
    return mul(a, assign(scalar));
}

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m128 scale(__m128 a, float scalar) {
    return _mm_mul_ps(a, _mm_set1_ps(scalar));
}

template <>
inline __m128 scale(__m128 a, mllm_fp16_t scalar) {
    return scale(a, unhalf(scalar));
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 scale(__m256 a, float scalar) {
    return _mm256_mul_ps(a, _mm256_set1_ps(scalar));
}

template <>
inline __m256 scale(__m256 a, mllm_bf16_t scalar) {
    return scale(a, unhalf(scalar));
}
#endif // __AVX__

#if defined(__AVX512F__)
template <>
inline __m512 scale(__m512 a, float scalar) {
    return _mm512_mul_ps(a, _mm512_set1_ps(scalar));
}

#if defined(__AVX512BF16__)
template <>
inline __m512bh scale(__m512bh a, mllm_bf16_t scalar) {
    // Convert __m512bh to __m512 for arithmetic operations
    __m512 a_f32 = _mm512_castsi512_ps(_mm512_castbh_si512(a));

    // Scale the float32 values
    __m512 scaled_f32 = _mm512_mul_ps(a_f32, _mm512_set1_ps(unhalf(scalar)));

    // Convert back to __m512bh
    return _mm512_castsi512_bh(_mm512_castps_si512(scaled_f32));
}
#endif // __AVX512BF16__

#endif // __AVX512F__

#if defined(__ARM_NEON)
template <>
inline float32x4_t scale(float32x4_t a, float scalar) {
    return vmulq_n_f32(a, scalar);
}

template <>
inline float32x4_t scale(float32x4_t a, mllm_fp16_t scalar) {
    return vmulq_n_f32(a, unhalf(scalar));
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
inline float16x8_t scale(float16x8_t a, mllm_fp16_t scalar) {
    return vmulq_n_f16(a, scalar);
}

template <>
inline float16x8_t scale(float16x8_t a, float scalar) {
    return scale(a, MLLM_FP32_TO_FP16(scalar));
}
#endif // __ARM_FP16

} // namespace
#endif // MLLM_TINYBLAS_HPP
