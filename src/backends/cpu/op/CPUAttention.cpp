
#include "CPUAttention.hpp"
#include "backends/cpu/compute/Attention.hpp"

namespace mllm {

CPUAttention::CPUAttention(Backend *bn, string opName, int num_qo_heads, int num_kv_heads, int head_dim, bool causal, int threadCount) :
    Op(bn,opName),
    num_qo_heads_(num_qo_heads), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
    causal_(causal), thread_count(threadCount) {}

ErrorCode CPUAttention::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto queries = inputs[0]; // [q_len, num_qo_heads, head_dim]
    // NOTE: currently only support batch size 1
    // there is no batch scenario on edge device
    assert(queries->batch() == 1);
    assert(queries->head() == num_qo_heads_);
    assert(queries->dimension() == head_dim_);
    outputs[0]->reshape(1, num_qo_heads_, queries->sequence(), head_dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUAttention::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto queries = inputs[0]; // [q_len, num_qo_heads, head_dim]
    auto kv_cache = inputs[1];
    auto kv_indices = inputs[2]; // [kv_len]
    auto mask = inputs.size() == 4 ? inputs[3] : nullptr;
    auto mask_ptr = mask != nullptr ? mask->rawHostPtr() : nullptr;
    auto q_len = queries->sequence();
    int qo_indptr[2] = {0, q_len};
    int kv_indptr[2] = {0, kv_indices->sequence()};

    // currently support dtype
    assert(queries->dtype() == MLLM_TYPE_F32);
    assert(kv_cache->dtype() == MLLM_TYPE_F32);
    assert(kv_indices->dtype() == MLLM_TYPE_I32);
    assert(mask == nullptr || mask->dtype() == MLLM_TYPE_BOOL);

    AttentionFP32(
        (float *)queries->rawHostPtr(), qo_indptr, 2,
        (float *)kv_cache->rawHostPtr(),
        (int *)kv_indices->rawHostPtr(), kv_indptr, 2,
        causal_,
        (bool *)mask_ptr,
        num_qo_heads_, num_kv_heads_, head_dim_,
        (float*) outputs[0]->rawHostPtr(),
        thread_count
    );

    return Op::execute(inputs, outputs);
}
} // namespace mllm

