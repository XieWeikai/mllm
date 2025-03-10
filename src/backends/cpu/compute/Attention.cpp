//
// Created by xwk on 25-2-27.
//
#include <cstring>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "Attention.hpp"
#include "tinyBLAS.hpp"

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

template <int RK, typename D, typename V, typename TKV, typename TQ, typename TO>
class Attention{
public:
    Attention(
        TQ *queries, int *qo_indptr, int qo_indptr_length,
        TKV *kv_cache, int *kv_indices, int *kv_indptr, int kv_indptr_length,
        bool is_causal,
        bool *mask,
        int num_qo_heads, int num_kv_heads, int head_dim,
        TO *output
    ): queries(queries), qo_indptr(qo_indptr), qo_indptr_length(qo_indptr_length),
       kv_cache(kv_cache), kv_indices(kv_indices), kv_indptr(kv_indptr), kv_indptr_length(kv_indptr_length),
       is_causal(is_causal), mask(mask), num_qo_heads(num_qo_heads), num_kv_heads(num_kv_heads), head_dim(head_dim),
       output(output)
    {
        o_stride = q_stride = num_qo_heads * head_dim;
        kv_stride = 2 * num_kv_heads * head_dim;
        k_cache = kv_cache;
        v_cache = kv_cache + num_kv_heads * head_dim;
        inv_sqrt_head_dim = 1.0f / std::sqrt(head_dim);
    }

    void compute() {
#if VECTOR_REGISTERS == 32
        constexpr int BM = 4;
        constexpr int BN = 6;
#else  // VECTOR_REGISTERS == 16
        constexpr int BM = 4;
        constexpr int BN = 3;
#endif
        bool *local_mask = mask;
        for (int batch_idx = 0; batch_idx < qo_indptr_length - 1; batch_idx++) {
            int q_start = qo_indptr[batch_idx];
            int q_end = qo_indptr[batch_idx + 1];
            int q_len = q_end - q_start;

            int kv_start = kv_indptr[batch_idx];
            int kv_end = kv_indptr[batch_idx + 1];
            int kv_len = kv_end - kv_start;

            for (int head_idx = 0; head_idx < num_qo_heads; head_idx++) {
                auto query_head = queries + q_start * q_stride + head_idx * head_dim;
                auto output_head = output + q_start * o_stride + head_idx * head_dim;
                for(int q_i = 0; q_i < q_len; q_i += BM){
                    int real_BM = std::min(BM, q_end - q_i);

                    computeO<real_BM, BN>(
                        query_head + q_i * q_stride,
                        k_cache, v_cache,
                        output_head + q_i * o_stride,
                        kv_start, kv_end,
                        local_mask + q_i * kv_len,
                        q_len, q_i
                        );

                }
            }
            local_mask += q_len * kv_len;
        }
    }

private:
    // i, j is the start and end idx of kv_indices
    // local_mask: [BM, kv_len]
    // k, v: start addr of k and v of a certain head
    // q: start addr of q of a certain head
    // ii: the q is the ii-th query in the batch (q_len queries totally)
    template <int BM, int BN>
    inline void computeO(TQ *q, TKV *k, TKV *v, TO *o, int kv_i, int kv_j, bool *local_mask, int q_len, int ii) {
        // TODO: for simplicity, we assume BM = RM and BN = RN, but later BM and BN should be bigger than RM and RN
        float S[BM * BN]; // attention scores
        float max_score[BM];
        float new_max_score[BM];
        float exp_sum[BM] = {};
        const int kv_len = kv_j - kv_i;

        // initialize max_score to -inf
        std::fill(max_score, max_score + BM, -std::numeric_limits<float>::infinity());
        std::fill(new_max_score, max_score + BM, -std::numeric_limits<float>::infinity());
        // set o to 0
        for (int i = 0; i < BM; i ++)
            memset((void *)(o + i * o_stride), 0, head_dim * sizeof(TO));

        for (int jj = 0; jj < kv_len; jj += BN) {
            if(!local_mask && is_causal && (jj > kv_len - q_len + ii + BM - 1))
                break; // if causal and all the queries are out of the range of kv, then break

            // mask leave in the r_gemm to handle

            // compute the actual BN
            auto RN = std::min(BN, kv_len - jj);
            // compute S
            if (!r_gemm<BM, RN>(q, k + kv_indices[kv_i + jj] * kv_stride, S, local_mask, kv_len, q_len, ii, jj))
                continue; // this block is all invalid

            // scale S and find max score
            for (int i = 0; i < BM; i++) {
                for (int j = 0; j < RN; j++) {
                    S[i * RN + j] *= inv_sqrt_head_dim;
                    new_max_score[i] = std::max(max_score[i], S[i * RN + j]);
                }
            }

            // scale o and update exp_sum
            for (int i = 0; i < BM; i++) {
                if (new_max_score[i] > max_score[i] && jj > 0) {
                    float scale = std::exp(max_score[i] - new_max_score[i]);
                    scale_row(o + i * o_stride, scale);
                    exp_sum[i] *= scale;
                }
                max_score[i] = new_max_score[i];
            }

            for (int i = 0; i < BM; i++){
                for (int j = 0; j < RN; j++){
                    S[i * RN + j] = std::exp(S[i * RN + j] - max_score[i]);
                }
            }

            // update o and exp_sum
            r_gemm_SV<BM, RN>(S, v, o, kv_i + jj);

            for (int i = 0; i < BM; i++){
                for (int j = 0; j < RN; j++){
                    exp_sum[i] += S[i * RN + j];
                }
            }
        }

        // scale o
        for (int i = 0; i < BM; i++){
            if (exp_sum[i] > 0)
                scale_row(o + i * o_stride, 1.0f / exp_sum[i]);
        }
    }

    inline void scale_row(TO *o, float scalar){
        for (int i = 0; i < head_dim; i += RK){
            V ov = load<V>(o + i);
            ov = scale(ov, scalar);
            store(o + i, ov);
        }
    }

    // micro kernel to compute C = A * B^T
    // A: [RM, head_dim]  B: [RN, head_dim]  C: [RM, RN]
    // this is a kernel done in registers
    // where RK is the dimension of the vectors
    // RM and RN is decided by the number of registers available
    // A is part of queries
    // B is part of keys
    // C is part of attn scores
    template <int RM, int RN>
    inline bool r_gemm(TQ *A, TKV *B, float *C, bool *local_mask, int kv_len, int q_len, int ii, int jj){
        // TODO: maybe explicitly unroll the loop will be faster?
        D Cv[RM][RN] = {};

        // Precompute compute mask for this tile
        bool compute_mask[RM][RN];
        bool all_invalid = true;
        for (int i = 0; i < RM; ++i) {
            for (int j = 0; j < RN; ++j) {
                const int q_i = ii + i;
                const int kv_i = jj + j;

                bool valid = true;
                // Handle causal mask
                if (!local_mask && is_causal) valid &= (kv_i > kv_len - q_len + q_i);
                // Handle local attention mask
                if (local_mask) valid &= local_mask[q_i * kv_len + kv_i];

                compute_mask[i][j] = valid;
                all_invalid &= !valid;
            }
        }

        for (int l = 0; l < head_dim; l += RK) {
            A += l;
            B += l;
            if constexpr (RM <= RN) {
                V Av[RM];
                for (int i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + i * q_stride);
                }
                for (int j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + j * kv_stride);
                    for (int i = 0; i < RM; ++i) {
                        if (compute_mask[i][j]) {
                            Cv[i][j] = madd(Av[i], Bv, Cv[i][j]);
                        }
                    }
                }
            } else {
                V Bv[RN];
                for (int j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + j * kv_stride);
                }
                for (int i = 0; i < RM; ++i) {
                    V Av = load<V>(A + i * q_stride);
                    for (int j = 0; j < RN; ++j) {
                        if (compute_mask[i][j]) {
                            Cv[i][j] = madd(Av, Bv[j], Cv[i][j]);
                        }
                    }
                }
            }
        }

        // Apply final mask and store results
        for (int i = 0; i < RM; ++i) {
            for (int j = 0; j < RN; ++j) {
                if (compute_mask[i][j]) {
                    C[i * RN + j] = hsum(Cv[i][j]) / std::sqrt(head_dim);
                } else {
                    C[i * RN + j] = -std::numeric_limits<float>::infinity();
                }
            }
        }

        return !all_invalid;
    }

    // compute S * V in registers
    // S is a small tile of the attention scores
    // S: [RM, RN]
    // i: start of kv_indices
    // v: start addr of v of a certain head
    template <int RM, int RN>
    inline void r_gemm_SV(float *S, TKV *v, TO *o, int i){
        for (int k = 0;k < head_dim; k += RK){
            V vv[RN];
            V dst[RN] = {};

            for(int ii = 0; ii < RN; ii++){
                vv[ii] = load<V>(v + kv_indices[i + ii] * kv_stride);
            }
            for(int ii = 0; ii < RM; ii++){
                for (int jj = 0; jj < RN; jj++){
                    dst[ii] = smadd(vv[jj], S[ii * RN + jj], dst[ii]);
                }
            }

            for(int ii = 0; ii < RM; ii++){
                store(o + ii * o_stride, dst[ii]);
            }
        }
    }

    // [n, num_qo_head, head_dim]
    TQ *queries;
    // e.g. qo_indptr_length = 4
    // qo_indptr = [0, 2, 5, 7]
    // then q0 q1 is the first batch of queries
    // q2 q3 q4 is the second batch of queries
    // q5 q6 is the third batch of queries
    // where the start address of q_i is queries + i * num_qo_head * head_dim
    int *qo_indptr, qo_indptr_length;
    // [max_tokens, 2, num_kv_head, head_dim]  in the dim=1, 0 means key, 1 means value
    TKV *kv_cache;
    // e.g. kv_indptr_length = 4
    // kv_indptr = [0, 4, 8, 16]
    // then the indices of the first batch of keys and values are kv_indices[0:4)
    // the indices of the second batch of keys and values are kv_indices[4:8)
    // ...
    int *kv_indices, *kv_indptr, kv_indptr_length;
    bool is_causal;
    // length can be inferred from qo_indptr and kv_indptr
    // assume a batch with qo_len queries and kv_len keys and values
    // then this mask should be of shape [qo_len, kv_len], i.e. [qo_len * kv_len]
    // and mask[i * kv_len + j] is true, then the ith query can only attend to the jth keys and values in this batch
    // the mask simply concat all the masks of all batches
    bool *mask;
    int num_qo_heads, num_kv_heads, head_dim;
    // o: [..., num_qo_head, head_dim]
    TO *output;

    int kv_stride, q_stride, o_stride;
    TKV *k_cache, *v_cache;

    float inv_sqrt_head_dim;
};

// support MHA and GQA
void AttentionFP32(
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
