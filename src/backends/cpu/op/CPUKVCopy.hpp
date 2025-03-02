
#ifndef MLLM_CPUKVCOPY_H
#define MLLM_CPUKVCOPY_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUKVCopy final : public Op {
public:
    CPUKVCopy(Backend *bn, string opName, int max_tokens, int num_kv_head, int head_dim, int threadCount);
    ~CPUKVCopy() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;

    // kv cache: [max_tokens, 2, num_kv_head, head_dim]
    int num_kv_head_;
    int head_dim_;
    int max_tokens_;
};

class CPUKVCopyCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int max_tokens = op_param["max_tokens"];
        int num_kv_head = op_param["num_kv_head"];
        int head_dim = op_param["head_dim"];

        return new CPUKVCopy(bn, name, max_tokens, num_kv_head, head_dim, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCOPY_H
