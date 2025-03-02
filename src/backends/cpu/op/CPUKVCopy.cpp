
#include "CPUKVCopy.hpp"
#include "Types.hpp"

#include "backends/cpu/compute/VecDotType.hpp"

namespace mllm {

CPUKVCopy::CPUKVCopy(Backend *bn,  string opName, int max_tokens, int num_kv_head, int head_dim, int threadCount) : thread_count(threadCount),
    Op(bn, opName), max_tokens_(max_tokens), head_dim_(head_dim), num_kv_head_(num_kv_head) {

}

ErrorCode CPUKVCopy::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCopy::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto kv_cache = inputs[0]; // [max_tokens, 2, num_kv_head, head_dim]
    auto k = inputs[1];        // [num_tokens, num_kv_head, head_dim]
    auto v = inputs[2];        // [num_tokens, num_kv_head, head_dim]
    auto pos = inputs[3];      // [num_tokens]

    assert(kv_cache->dtype() == k->dtype());
    assert(kv_cache->dtype() == v->dtype());

    auto kv_shape = kv_cache->shape();
    auto k_shape = k->shape();  // though logically the shape is [num_tokens, num_kv_head, head_dim] but in mllm, there is 4 dim, so its shape is actually [1, num_tokens, num_kv_head, head_dim]
    auto v_shape = v->shape();  // though logically the shape is [num_tokens, num_kv_head, head_dim] but in mllm, there is 4 dim, so its shape is actually [1, num_tokens, num_kv_head, head_dim]

    int num_tokens = k_shape[k_shape.size() - 3];
    assert(kv_shape[0] == max_tokens_);
    assert(kv_shape[1] == 2);
    assert(kv_shape[2] == num_kv_head_);
    assert(kv_shape[3] == head_dim_);
    assert(k_shape[k_shape.size() - 2] == num_kv_head_);
    assert(k_shape[k_shape.size() - 1] == head_dim_);
    assert(v_shape[v_shape.size() - 2] == num_kv_head_);
    assert(v_shape[v_shape.size() - 1] == head_dim_);
    assert(num_tokens == v_shape[v_shape.size() - 3]);
    assert(pos->count() == num_tokens);
    assert(pos->dtype() == MLLM_TYPE_I32);
    assert(kv_cache->dtype() == k->dtype());
    assert(kv_cache->dtype() == v->dtype());

    char *kv_cache_data = (char *)kv_cache->rawHostPtr();
    int *pos_data = (int *) pos->rawHostPtr();
    char *k_data = (char *)k->rawHostPtr();
    char *v_data = (char *)v->rawHostPtr();

    auto stride = num_kv_head_ * head_dim_ * type_size(kv_cache->dtype());
    auto kv_stride = 2 * stride;

    // TODO: optimize
    // num_tokens may usually be small(1) so omp will only bring overhead
    // should split into chunks and use omp according to the chunk size
    // when chunk size is large, overhead of omp will be covered
    // when chunk size is small, should not use omp
#pragma  omp parallel for num_threads(thread_count)
    for (int i = 0;i < num_tokens;i++) {
        auto dst = kv_cache_data + pos_data[i] * kv_stride;
        memcpy(dst, k_data + i * stride, stride);
        memcpy(dst + stride, v_data + i * stride, stride);
    }

    return MLLM_NO_ERROR;
}
} // namespace mllm

