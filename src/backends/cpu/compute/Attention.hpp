//
// Created by xwk on 25-2-27.
//

#ifndef MLLM_ATTENTION_HPP
#define MLLM_ATTENTION_HPP

#include <cassert>

#include "tinyBLAS.hpp"

template <int RK, typename D, typename V, typename TKV, typename TQ, typename TO>
class Attention{
public:
    Attention(
        const TQ *queries, const int *qo_indptr, int qo_indptr_length,
        const TKV *kv_cache, const int *kv_indices, const int *kv_indptr, const int kv_indptr_length,
        const bool is_causal,
        const bool *mask,
        const int num_qo_heads, const int num_kv_heads, const int head_dim,
        TO *const output
        ): queries(queries), qo_indptr(qo_indptr), qo_indptr_length(qo_indptr_length),
        kv_cache(kv_cache), kv_indices(kv_indices), kv_indptr(kv_indptr), kv_indptr_length(kv_indptr_length),
        is_causal(is_causal), mask(mask), num_qo_heads(num_qo_heads), num_kv_heads(num_kv_heads), head_dim(head_dim),
        output(output)
    {
        assert(head_dim % RK == 0);
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
        const bool *local_mask = mask;
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

                auto kv_head_idx = head_idx * num_kv_heads / num_qo_heads;
                auto k_cache_head = k_cache + kv_head_idx * head_dim;
                auto v_cache_head = v_cache + kv_head_idx * head_dim;

                for(int q_i = 0; q_i < q_len; q_i += BM){
                    int real_BM = std::min(BM, q_len - q_i);

                    switch (real_BM) {
                    case 4:
                        computeO<4, BN>(
                            query_head + q_i * q_stride,
                            k_cache_head, v_cache_head,
                            output_head + q_i * o_stride,
                            kv_start, kv_end,
                            mask ? (local_mask + q_i * kv_len) : nullptr,
                            q_len, q_i
                        );
                        break;

                    case 3:
                        computeO<3, BN>(
                            query_head + q_i * q_stride,
                            k_cache_head, v_cache_head,
                            output_head + q_i * o_stride,
                            kv_start, kv_end,
                            mask ? (local_mask + q_i * kv_len) : nullptr,
                            q_len, q_i
                        );
                        break;

                    case 2:
                        computeO<2, BN>(
                            query_head + q_i * q_stride,
                            k_cache_head, v_cache_head,
                            output_head + q_i * o_stride,
                            kv_start, kv_end,
                            mask ? (local_mask + q_i * kv_len) : nullptr,
                            q_len, q_i
                        );
                        break;

                    case 1:
                        computeO<1, BN>(
                            query_head + q_i * q_stride,
                            k_cache_head, v_cache_head,
                            output_head + q_i * o_stride,
                            kv_start, kv_end,
                            mask ? (local_mask + q_i * kv_len) : nullptr,
                            q_len, q_i
                        );
                        break;
                    }
                }
            }
            local_mask += q_len * kv_len;
        }
    }

    //private:
public:  // for debug now
    // kv_i, kv_j is the start and end idx of kv_indices
    // local_mask: [BM, kv_len]
    // k, v: start addr of k and v of a certain head
    // q: start addr of q of a certain row of a certain head
    // o: start addr of o of a certain row of a certain head
    // ii: the q is the ii-th query in the batch (q_len queries totally)
    template <int BM, int BN>
    inline void computeO(const TQ *q, const TKV *k, const TKV *v, TO *o, int kv_i, int kv_j, const bool *local_mask, int q_len, int ii) {
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

        auto k_indices = kv_indices + kv_i;
        for (int jj = 0; jj < kv_len; jj += BN) {
            if(!local_mask && is_causal && (jj > kv_len - q_len + ii + BM - 1))
                break; // if causal and all the queries are out of the range of kv, then break

            // mask leave in the r_gemm to handle

            // compute the actual BN
            auto RN = std::min(BN, kv_len - jj);
            // compute S
            if (!_r_gemm<BM>(RN, q, k, q_len, k_indices, kv_len, local_mask, ii, jj, S))
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
            _r_gemm_SV<BM>(RN, S, v, o, kv_i + jj);

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

    template <int RM>
    inline bool _r_gemm(int RN, const TQ *q, const TKV *k, int q_len, const int *k_indices, int kv_len, const bool *local_mask, int ii, int jj, float *C){
        switch (RN) {
        case 1:
            return r_gemm<RM, 1>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        case 2:
            return r_gemm<RM, 2>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        case 3:
            return r_gemm<RM, 3>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        case 4:
            return r_gemm<RM, 4>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        case 5:
            return r_gemm<RM, 5>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        case 6:
            return r_gemm<RM, 6>(q, q_len, k, k_indices, kv_len, local_mask, ii, jj, C);
        default:
            assert(false);
        }
    }

    // compute a small tile Q * K^T in registers
    // q: start of queries of this batch of a certain head
    // k: start of kv_cache of a certain head
    // k_indices: start of kv_indices of this batch
    // ii, jj: the start index of the tile.  i.e. tile row: [ii, ii + RM), tile col: [jj, jj + RN)
    // q_len: the length of queries of this batch
    // kv_len: the length of kv_indices of this batch
    // q_len and kv_len are used to handle causal mask
    // C: [RM, RN]  output of the tile (Q * K^T)
    template <int RM, int RN>
    inline bool r_gemm(const TQ *q, int q_len, const TKV *k, const int *k_indices, int kv_len, const bool *local_mask, int ii, int jj, float *C){
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

        if(all_invalid)
            return false;

        for (int l = 0; l < head_dim; l += RK) {
            if constexpr (RM <= RN) {
                V Av[RM];
                for (int i = 0; i < RM; ++i) {
                    Av[i] = load<V>(q + (ii + i) * q_stride);
                }
                for (int j = 0; j < RN; ++j) {
                    V Bv = load<V>(k + k_indices[jj + j] * kv_stride);
                    for (int i = 0; i < RM; ++i) {
                        if (compute_mask[i][j]) {
                            Cv[i][j] = madd(Av[i], Bv, Cv[i][j]);
                        }
                    }
                }
            } else {
                V Bv[RN];
                for (int j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(k + k_indices[jj + j] * kv_stride);
                }
                for (int i = 0; i < RM; ++i) {
                    V Av = load<V>(q + (ii + i) * q_stride);
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

        return true;
    }

    template <int RM>
    inline void _r_gemm_SV(int RN, float *S, const TKV *v, TO *o, int i){
        switch (RN) {
        case 1:
            r_gemm_SV<RM, 1>(S, v, o, i);
            break;
        case 2:
            r_gemm_SV<RM, 2>(S, v, o, i);
            break;
        case 3:
            r_gemm_SV<RM, 3>(S, v, o, i);
            break;
        case 4:
            r_gemm_SV<RM, 4>(S, v, o, i);
            break;
        case 5:
            r_gemm_SV<RM, 5>(S, v, o, i);
            break;
        case 6:
            r_gemm_SV<RM, 6>(S, v, o, i);
            break;
        default:
            assert(false);
        }
    }

    // compute S * V in registers
    // S is a small tile of the attention scores
    // S: [RM, RN]
    // i: start of kv_indices
    // v: start addr of v of a certain head
    template <int RM, int RN>
    inline void r_gemm_SV(float *S, const TKV *v, TO *o, int i){
        for (int k = 0;k < head_dim; k += RK){
            V vv[RN];
            V dst[RM] = {};

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
    const TQ *queries;
    // e.g. qo_indptr_length = 4
    // qo_indptr = [0, 2, 5, 7]
    // then q0 q1 is the first batch of queries
    // q2 q3 q4 is the second batch of queries
    // q5 q6 is the third batch of queries
    // where the start address of q_i is queries + i * num_qo_head * head_dim
    const int *qo_indptr, qo_indptr_length;
    // [max_tokens, 2, num_kv_head, head_dim]  in the dim=1, 0 means key, 1 means value
    const TKV *kv_cache;
    // e.g. kv_indptr_length = 4
    // kv_indptr = [0, 4, 8, 16]
    // then the indices of the first batch of keys and values are kv_indices[0:4)
    // the indices of the second batch of keys and values are kv_indices[4:8)
    // ...
    const int *kv_indices, *kv_indptr, kv_indptr_length;
    const bool is_causal;
    // length can be inferred from qo_indptr and kv_indptr
    // assume a batch with qo_len queries and kv_len keys and values
    // then this mask should be of shape [qo_len, kv_len], i.e. [qo_len * kv_len]
    // and mask[i * kv_len + j] is true, then the ith query can only attend to the jth keys and values in this batch
    // the mask simply concat all the masks of all batches
    const bool *mask;
    const int num_qo_heads, num_kv_heads, head_dim;
    // o: [..., num_qo_head, head_dim]
    TO *output;

    int kv_stride, q_stride, o_stride;
    const TKV *k_cache, *v_cache;

    float inv_sqrt_head_dim;
};

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
    int thread_count = 4);

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
    int thread_cnt = 4
);

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
    // if mask[i * kv_length + j] is true, then the ith query can only attend to the first j keys and values
    bool *mask, // the mask
    int head_dim ,// the dimension of the query, key and value

    float *output, int o_stride // the output of ith query is output + i * o_stride
);

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
);

#endif // MLLM_ATTENTION_HPP
