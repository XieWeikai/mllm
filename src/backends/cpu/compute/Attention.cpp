//
// Created by xwk on 25-2-27.
//
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>

#include "Attention.hpp"

#include <stddef.h>
#include <stdint.h>

#define USE_SIMD

#if defined(__SSE3__)
#include <pmmintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

#if defined(__AVX__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// 水平求和辅助函数
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX512F__)
static inline float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX__)
static inline float hsum_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    return hsum_ps_sse3(vlow);
}
#endif

#if defined(__ARM_NEON)
static inline float hsum_ps_neon(float32x4_t v) {
    float32x2_t sum = vadd_f32(vget_high_f32(v), vget_low_f32(v));
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
}
#endif

#if defined(USE_SIMD)

// 向量点积优化
void vec_dot_fp32(const float* a, const float* b, int dim, float* output) {
    float sum = 0.0f;
    int i = 0;

#if defined(__AVX512F__)
    __m512 sum512 = _mm512_setzero_ps();
    for (; i <= dim - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum512 = _mm512_fmadd_ps(va, vb, sum512);
    }
    sum += _mm512_reduce_add_ps(sum512);
#elif defined(__AVX__)
    __m256 sum256 = _mm256_setzero_ps();
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(va, vb));
    }
    sum += hsum_ps_avx(sum256);
#elif defined(__SSE__)
    __m128 sum128 = _mm_setzero_ps();
    for (; i <= dim - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum128 = _mm_add_ps(sum128, _mm_mul_ps(va, vb));
    }
    sum += hsum_ps_sse3(sum128);
#elif defined(__ARM_NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sumv = vmlaq_f32(sumv, va, vb);
    }
    sum += hsum_ps_neon(sumv);
#endif

    // 处理剩余元素
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    *output = sum;
}

// 向量缩放优化
void scale_fp32(const float* a, int dim, float scale, float* output) {
    int i = 0;

#if defined(__AVX512F__)
    __m512 scale512 = _mm512_set1_ps(scale);
    for (; i <= dim - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        _mm512_storeu_ps(output + i, _mm512_mul_ps(va, scale512));
    }
#elif defined(__AVX__)
    __m256 scale256 = _mm256_set1_ps(scale);
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(output + i, _mm256_mul_ps(va, scale256));
    }
#elif defined(__SSE__)
    __m128 scale128 = _mm_set1_ps(scale);
    for (; i <= dim - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(output + i, _mm_mul_ps(va, scale128));
    }
#elif defined(__ARM_NEON)
    float32x4_t scalev = vdupq_n_f32(scale);
    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(output + i, vmulq_f32(va, scalev));
    }
#endif

    // 处理剩余元素
    for (; i < dim; ++i) {
        output[i] = a[i] * scale;
    }
}

// 向量累加优化
void add_row_fp32(const float* a, int dim, float scale, float* output) {
    int i = 0;

#if defined(__AVX512F__)
    __m512 scale512 = _mm512_set1_ps(scale);
    for (; i <= dim - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vo = _mm512_loadu_ps(output + i);
        _mm512_storeu_ps(output + i, _mm512_fmadd_ps(va, scale512, vo));
    }
#elif defined(__AVX__)
    __m256 scale256 = _mm256_set1_ps(scale);
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vo = _mm256_loadu_ps(output + i);
        _mm256_storeu_ps(output + i, _mm256_add_ps(vo, _mm256_mul_ps(va, scale256)));
    }
#elif defined(__SSE__)
    __m128 scale128 = _mm_set1_ps(scale);
    for (; i <= dim - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vo = _mm_loadu_ps(output + i);
        _mm_storeu_ps(output + i, _mm_add_ps(vo, _mm_mul_ps(va, scale128)));
    }
#elif defined(__ARM_NEON)
    float32x4_t scalev = vdupq_n_f32(scale);
    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vo = vld1q_f32(output + i);
        vst1q_f32(output + i, vmlaq_f32(vo, va, scalev));
    }
#endif

    // 处理剩余元素
    for (; i < dim; ++i) {
        output[i] += a[i] * scale;
    }
}

#else

void vec_dot_fp32(const float *a, const float *b, int dim, float *output){
    float sum = 0;
    for(int i = 0; i < dim; i++){
        sum += a[i] * b[i];
    }
    *output = sum;
}

void scale_fp32(const float *a, int dim, float scale, float *output){
    for(int i = 0; i < dim; i++){
        output[i] = a[i] * scale;
    }
}

void add_row_fp32(const float *a, int dim, float scale, float *output){
    for(int i = 0; i < dim; i++){
        output[i] += a[i] * scale;
    }
}
#endif

// support MHA and GQA
void AttentionFP32_v2(
    // e.g. qo_indptr_length = 4
    // qo_indptr = [0, 2, 5, 7]
    // then q0 q1 is the first batch of queries
    // q2 q3 q4 is the second batch of queries
    // q5 q6 is the third batch of queries
    // where the start address of q_i is queries + i * num_qo_head * head_dim
    float *queries, const int *qo_indptr, int qo_indptr_length,
    float *kv_cache, // [max_tokens, 2, num_kv_head, head_dim]  in the dim=1, 0 means key, 1 means value
    // e.g. kv_indptr_length = 4
    // kv_indptr = [0, 4, 8, 16]
    // then the indices of the first batch of keys and values are kv_indices[0:4)
    // the indices of the second batch of keys and values are kv_indices[4:8)
    // ...
    int *kv_indices, const int *kv_indptr, int kv_indptr_length,
    bool is_causal, // only used when mask is nullptr

    // length can be inferred from qo_indptr and kv_indptr
    // assume a batch with qo_len queries and kv_len keys and values
    // then this mask should be of shape [qo_len, kv_len], i.e. [qo_len * kv_len]
    // and mask[i * kv_len + j] is true, then the ith query can only attend to the jth keys and values in this batch
    // the mask simply concat all the masks of all batches
    bool *mask, // the mask
    int num_qo_heads, int num_kv_heads, int head_dim,

    // o: [num_qo_heads, head_dim]
    // output + i * num_qo_heads *head_dim to output + j * num_qo_heads *head_dim is the output of a batch
    // where i to j is a batch
    float *output,
    int thread_count
){
#if defined(__AVX512F__)
    Attention<16, __m512, __m512, float, float, float> att(queries, qo_indptr, qo_indptr_length,
                                                            kv_cache, kv_indices, kv_indptr, kv_indptr_length,
                                                            is_causal, mask, num_qo_heads, num_kv_heads, head_dim, output);
    att.compute();
#elif defined(__AVX__) || defined(__AVX2__)
    Attention<8, __m256, __m256, float, float, float> att(queries, qo_indptr, qo_indptr_length,
                                                           kv_cache, kv_indices, kv_indptr, kv_indptr_length,
                                                           is_causal, mask, num_qo_heads, num_kv_heads, head_dim, output);
    att.compute();
#elif defined(__ARM_NEON)
    Attention<4, float32x4_t, float32x4_t, float, float, float> att(queries, qo_indptr, qo_indptr_length,
                                                                     kv_cache, kv_indices, kv_indptr, kv_indptr_length,
                                                                     is_causal, mask, num_qo_heads, num_kv_heads, head_dim, output);
    att.compute();
#else
    assert(false);
#endif
}

// support MHA and GQA
void AttentionFP32(
    // e.g. qo_indptr_length = 4
    // qo_indptr = [0, 2, 5, 7]
    // then q0 q1 is the first batch of queries
    // q2 q3 q4 is the second batch of queries
    // q5 q6 is the third batch of queries
    // where the start address of q_i is queries + qo_indptr[i] * num_qo_head * head_dim
    float *queries, const int *qo_indptr, int qo_indptr_length,
    float *kv_cache, // [max_tokens, 2, num_kv_head, head_dim]  in the dim=1, 0 means key, 1 means value
    // e.g. kv_indptr_length = 4
    // kv_indptr = [0, 4, 8, 16]
    // then the indices of the first batch of keys and values are kv_indices[0:4)
    // the indices of the second batch of keys and values are kv_indices[4:8)
    // ...
    int *kv_indices, const int *kv_indptr, int kv_indptr_length,
    bool is_causal, // only used when mask is nullptr

    // length can be inferred from qo_indptr and kv_indptr
    // assume a batch with qo_len queries and kv_len keys and values
    // then this mask should be of shape [qo_len, kv_len], i.e. [qo_len * kv_len]
    // and mask[i * kv_len + j] is true, then the ith query can only attend to the jth keys and values in this batch
    // the mask simply concat all the masks of all batches
    bool *mask, // the mask
    int num_qo_heads, int num_kv_heads, int head_dim,

    // o: [..., num_qo_heads, head_dim]
    // output + i * num_qo_heads *head_dim to output + j * num_qo_heads *head_dim is the output of a batch
    // where i to j is a batch
    float *output,
    int thread_cnt
){
    int q_stride = num_qo_heads * head_dim;
    int kv_stride = 2 * num_kv_heads * head_dim;
    int o_stride = num_qo_heads * head_dim;
    float *k_cache = kv_cache;
    float *v_cache = kv_cache + num_kv_heads * head_dim;

    // Precompute mask offsets
    int* mask_offsets = nullptr;
    if (mask) {
        mask_offsets = new int[qo_indptr_length];
        mask_offsets[0] = 0;
        for (int i = 0; i < qo_indptr_length - 1; ++i) {
            int q_len = qo_indptr[i+1] - qo_indptr[i];
            int kv_len = kv_indptr[i+1] - kv_indptr[i];
            mask_offsets[i+1] = mask_offsets[i] + q_len * kv_len;
        }
    }

// 合并batch和head的并行
    thread_cnt = std::min((qo_indptr_length - 1) * num_qo_heads, thread_cnt);
#pragma omp parallel for collapse(2) num_threads(thread_cnt) \
        schedule(dynamic) if (thread_cnt > 1)// 动态调度优化负载均衡
    for (int batch_idx = 0; batch_idx < qo_indptr_length - 1; ++batch_idx) {
        for (int head_idx = 0; head_idx < num_qo_heads; ++head_idx) {
            // Batch相关参数
            int q_start = qo_indptr[batch_idx];
            int q_end = qo_indptr[batch_idx + 1];
            int q_length = q_end - q_start;

            int kv_start = kv_indptr[batch_idx];
            int kv_end = kv_indptr[batch_idx + 1];
            int kv_length = kv_end - kv_start;

            // Head相关参数
            int kv_head_idx = head_idx * num_kv_heads / num_qo_heads;
            float *query_head = queries + q_start * q_stride + head_idx * head_dim;
            float *k_cache_head = k_cache + kv_head_idx * head_dim;
            float *v_cache_head = v_cache + kv_head_idx * head_dim;
            float *output_head = output + q_start * o_stride + head_idx * head_dim;

            // 处理当前batch和head
            AttentionFP32Head(
                query_head, q_stride, q_length,
                k_cache_head, kv_stride,
                v_cache_head, kv_stride,
                kv_indices + kv_start, kv_length,
                is_causal,
                mask ? mask + mask_offsets[batch_idx] : nullptr,
                head_dim,
                output_head, o_stride
            );
        }
    }

    if (mask_offsets) {
        delete[] mask_offsets;
    }
}

// this is a single head attention
void AttentionFP32Head(
    float *query, int q_stride, int q_length, // the address of ith query is query + i * q_stride
    float *k_cache, int k_stride, // the address of ith key is k_cache + i * k_stride
    float *v_cache, int v_stride, // the address of ith value is v_cache + i * v_stride

    // suppose kv_indices is [1,4,7,9,20] and kv_length is 5
    // then the first key is k_cache + 1 * k_stride, the first value is v_cache + 1 * v_stride
    // the second key is k_cache + 4 * k_stride, the second value is v_cache + 4 * v_stride
    // and so on
    const int *kv_indices, int kv_length, // the indices of the keys and values

    bool is_causal, // only used when mask is nullptr

    // length of the mask is q_length * kv_length
    // if mask[i * kv_length + j] is true, then the ith query can only attend to the jth keys and values
    bool *mask, // the mask
    int head_dim, // the dimension of the query, key and value

    float *output, int o_stride // the output of ith query is output + i * o_stride
){

    for(int q_i = 0; q_i < q_length; q_i++){
        auto query_i = query + q_i * q_stride;
        auto output_i = output + q_i * o_stride;

        // set the output to 0, use memset to speed up
        memset(output_i, 0, sizeof(float) * head_dim);

        // max value initialized to -inf
        float m = - std::numeric_limits<float>::infinity();
        // sum of exp
        float exp_sum = 0;

        bool *q_mask = mask ? mask + q_i * kv_length : nullptr;
        for(int kv_i = 0; kv_i < kv_length; kv_i++){
            if(mask == nullptr && is_causal && kv_i > kv_length - q_length + q_i){
                break;
            }

            // though this is the same as pytorch
            // I think it is not really causal when using kv_cache
            // if(mask == nullptr && is_causal && kv_i > q_i){
            //    // the same as scaled_dot_product_attention in pytorch
            //    break;
            // }

            if(q_mask != nullptr && !q_mask[kv_i]){
                continue;
            }

            auto key_i = k_cache + kv_indices[kv_i] * k_stride;
            auto value_i = v_cache + kv_indices[kv_i] * v_stride;

            float score;
            vec_dot_fp32(query_i, key_i, head_dim, &score);
            score = score / sqrtf((float)head_dim);

            float m_new = std::max(m, score);

            if(m_new > m && kv_i > 0){
                float scale = expf(m - m_new);
                scale_fp32(output_i, head_dim, scale, output_i);
                exp_sum *= scale;
            }

            m = m_new;

            float scale = expf(score - m);
            add_row_fp32(value_i, head_dim, scale, output_i);

            exp_sum += expf(score - m);
            //            printf("q_i: %d, kv_i: %d, score: %f\n", q_i, kv_i, score);
            //            printf("m: %f, exp_sum: %f\n\n", m, exp_sum);
        }

        if (exp_sum > 0)
            scale_fp32(output_i, head_dim, 1.0f / exp_sum, output_i);
    }
}

// this is a single head attention
void AttentionFP32Head_Simple(
    float *query, int q_stride, int q_length, // the address of ith query is query + i * q_stride
    float *k_cache, int k_stride, // the address of ith key is k_cache + i * k_stride
    float *v_cache, int v_stride, // the address of ith value is v_cache + i * v_stride

    // suppose kv_indices is [1,4,7,9,20] and kv_length is 5
    // then the first key is k_cache + 1 * k_stride, the first value is v_cache + 1 * v_stride
    // the second key is k_cache + 4 * k_stride, the second value is v_cache + 4 * v_stride
    // and so on
    const int *kv_indices, int kv_length, // the indices of the keys and values

    bool is_causal, // only used when mask is nullptr

    // length of the mask is q_length * kv_length
    // if mask[i * kv_length + j] is true, then the ith query can only attend to the jth keys and values
    bool *mask, // the mask
    int dim ,// the dimension of the query, key and value

    float *output, int o_stride // the output of ith query is output + i * o_stride
){

    for(int q_i = 0; q_i < q_length; q_i++){
        auto query_i = query + q_i * q_stride;
        auto output_i = output + q_i * o_stride;

        // set the output to 0, use memset to speed up
        memset(output_i, 0, sizeof(float) * dim);

        // max value initialized to -inf
        float m = - std::numeric_limits<float>::infinity();
        // sum of exp
        float exp_sum = 0;

        bool *q_mask = mask ? mask + q_i * kv_length : nullptr;
        // first pass to get the max value
        for(int kv_i = 0; kv_i < kv_length; kv_i++){
            if(mask == nullptr && is_causal && kv_i > q_i){
                // the same as scaled_dot_product_attention in pytorch
                break;
            }

            if(q_mask != nullptr && !q_mask[kv_i]){
                continue;
            }

            auto key_i = k_cache + kv_indices[kv_i] * k_stride;

            float score;
            vec_dot_fp32(query_i, key_i, dim, &score);
            score = score / sqrtf((float)dim);
            m = std::max(m, score);
        }

        // second pass to get the sum of exp
        for(int kv_i = 0; kv_i < kv_length; kv_i++){
            if(mask == nullptr && is_causal && kv_i > q_i){
                // the same as scaled_dot_product_attention in pytorch
                break;
            }

            if(q_mask != nullptr && !q_mask[kv_i]){
                continue;
            }

            auto key_i = k_cache + kv_indices[kv_i] * k_stride;

            float score;
            vec_dot_fp32(query_i, key_i, dim, &score);
            score = score / sqrtf((float)dim);
            exp_sum += expf(score - m);
        }

        // third pass to get the output
        for(int kv_i = 0; kv_length; kv_i++){
            if(mask == nullptr && is_causal && kv_i > q_i){
                // the same as scaled_dot_product_attention in pytorch
                break;
            }

            if(q_mask != nullptr && !q_mask[kv_i]){
                continue;
            }

            auto key_i = k_cache + kv_indices[kv_i] * k_stride;
            auto value_i = v_cache + kv_indices[kv_i] * v_stride;

            float score;
            vec_dot_fp32(query_i, key_i, dim, &score);
            score = score / sqrtf((float)dim);
            float scale = expf(score - m) / exp_sum;
            add_row_fp32(value_i, dim, scale, output_i);
        }
    }
}
