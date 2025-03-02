
#ifndef MLLM_CPUATTENTION_H
#define MLLM_CPUATTENTION_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUAttention final : public Op {
public:
    CPUAttention(Backend *bn, string opName, int num_qo_heads, int num_kv_heads, int head_dim, bool causal, int threadCount);
    ~CPUAttention() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;

    int num_qo_heads_;
    int num_kv_heads_;
    int head_dim_;
    bool causal_;
};

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int num_qo_heads = op_param["num_qo_heads"];
        int num_kv_heads = op_param["num_kv_heads"];
        int head_dim = op_param["head_dim"];
        bool causal = op_param["causal"] != 0.0;
        return new CPUAttention(bn, name, num_qo_heads, num_kv_heads, head_dim, causal, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUATTENTION_H
