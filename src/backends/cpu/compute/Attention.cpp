//
// Created by xwk on 25-2-27.
//
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "Attention.hpp"

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

    // o: [num_qo_heads, head_dim]
    // output + i * num_qo_heads *head_dim to output + j * num_qo_heads *head_dim is the output of a batch
    // where i to j is a batch
    float *output
){
    int q_stride = num_qo_heads * head_dim;
    int kv_stride = 2 * num_kv_heads * head_dim;
    int o_stride = num_qo_heads * head_dim;
    float *k_cache = kv_cache;
    float *v_cache = kv_cache + num_kv_heads * head_dim;

    // Iterate over each batch
    bool *batch_mask = mask;
    for (int batch_idx = 0; batch_idx < qo_indptr_length - 1; batch_idx++) {
        // Get the start and end indices for queries in this batch
        int q_start = qo_indptr[batch_idx];
        int q_end = qo_indptr[batch_idx + 1];
        int q_length = q_end - q_start;

        // Get the start and end indices for keys and values in this batch
        int kv_start = kv_indptr[batch_idx];
        int kv_end = kv_indptr[batch_idx + 1];
        int kv_length = kv_end - kv_start;


        // Iterate over each query head
        for (int head_idx = 0; head_idx < num_qo_heads; head_idx++) {
            // Calculate the address of the queries for this head
            float *query_head = queries + q_start * q_stride + head_idx * head_dim;

            // considering GQA
            assert (num_qo_heads % num_kv_heads == 0);
            int kv_head_idx =  head_idx * num_kv_heads / num_qo_heads;  // head_idx / (num_qo_heads / num_kv_heads);

            // Calculate the address of the keys and values for this head
            float *k_cache_head = k_cache + kv_head_idx * head_dim; // TODO: this can be precomputed
            float *v_cache_head = v_cache + kv_head_idx * head_dim; // // TODO: this can be precomputed

            // Calculate the address of the output for this head
            float *output_head = output + q_start * o_stride + head_idx * head_dim;

            // Call AttentionFP32Head for this head
            AttentionFP32Head(
                query_head, q_stride, q_length,
                k_cache_head, kv_stride,
                v_cache_head, kv_stride,
                kv_indices + kv_start, kv_length,
                is_causal,
                batch_mask,
                head_dim,
                output_head, o_stride
            );
        }

        // Move the mask pointer to the next batch
        batch_mask += q_length * kv_length;
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
            //            if(mask == nullptr && is_causal && kv_i > kv_length - q_length + q_i){
            //                break;
            //            }

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
