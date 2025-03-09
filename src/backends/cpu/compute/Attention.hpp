//
// Created by xwk on 25-2-27.
//

#ifndef MLLM_ATTENTION_HPP
#define MLLM_ATTENTION_HPP

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
    float *output
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
