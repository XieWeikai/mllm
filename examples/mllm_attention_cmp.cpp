//
// Created by xwk on 25-3-19.
//
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <functional>

#include "Layer.hpp"
#include "Module.hpp"

#define SEED 42
#define WARMUP 10
#define ITERATIONS 20

using namespace std;
using namespace mllm;

std::mt19937 gen(SEED);

struct TestCase {
    int q_len;
    int kv_len;
    int head_dim;
    int num_kv_heads;
    int num_qo_heads;
    int max_tokens;
    bool is_causal;
    int thread;
    int kv_gap = 1;
};

struct TestResult {
    double naive_time;
    double paged_time;
    double GFLOPS_naive;
    double GFLOPS_paged;
};

struct FieldDescriptor {
    string header;
    int width;
    function<string(const TestCase&, const TestResult&)> formatter;
};

const vector<FieldDescriptor> field_descriptors = {
    {"q_len",        8, [](auto& c, auto&) { return to_string(c.q_len); }},
    {"kv_len",       8, [](auto& c, auto&) { return to_string(c.kv_len); }},
    {"head_dim",     8, [](auto& c, auto&) { return to_string(c.head_dim); }},
    {"kv_heads",     8, [](auto& c, auto&) { return to_string(c.num_kv_heads); }},
    {"qo_heads",     8, [](auto& c, auto&) { return to_string(c.num_qo_heads); }},
    {"max_tokens",  10, [](auto& c, auto&) { return to_string(c.max_tokens); }},
    {"causal",       6, [](auto& c, auto&) { return c.is_causal ? "true" : "false"; }},
    {"thread",       6, [](auto& c, auto&) { return to_string(c.thread); }},
    {"kv_gap",       6, [](auto& c, auto&) { return to_string(c.kv_gap); }},
    {"Naive(us)",    10, [](auto&, auto& r) {
         ostringstream oss;
         oss << fixed << setprecision(2) << r.naive_time;
         return oss.str();
     }},
    {"Paged(us)",    10, [](auto&, auto& r) {
         ostringstream oss;
         oss << fixed << setprecision(2) << r.paged_time;
         return oss.str();
     }},
    {"NaiveGFLOPS",  12, [](auto&, auto& r) {
         ostringstream oss;
         oss << fixed << setprecision(4) << r.GFLOPS_naive;
         return oss.str();
     }},
    {"PagedGFLOPS",  12, [](auto&, auto& r) {
         ostringstream oss;
         oss << fixed << setprecision(4) << r.GFLOPS_paged;
         return oss.str();
     }},
    {"Speedup",      8, [](auto&, auto& r) {
         ostringstream oss;
         oss << fixed << setprecision(2) << (r.naive_time / r.paged_time) << "x";
         return oss.str();
     }},
};

void print_header() {
    for (const auto& fd : field_descriptors) {
        cout << setw(fd.width) << left << fd.header << " | ";
    }
    cout << "\n";

    for (const auto& fd : field_descriptors) {
        cout << string(fd.width, '-') << "-+-";
    }
    cout << "\n";
}

void print_results(const TestCase& cas, const TestResult& result) {
    for (const auto& fd : field_descriptors) {
        cout << setw(fd.width) << left << fd.formatter(cas, result) << " | ";
    }
    cout << "\n";
}

class NaiveAttention final : public Module {
    Softmax softmax;
    KVCache k_cache;
    KVCache v_cache;
    TestCase cas;

public:
    NaiveAttention(TestCase cas) : cas(cas) {
        softmax = Softmax(DIMENSION, cas.is_causal, "softmax");
        k_cache = KVCache(cas.num_qo_heads / cas.num_kv_heads, cas.max_tokens, "k_cache");
        v_cache = KVCache(cas.num_qo_heads / cas.num_kv_heads, cas.max_tokens, "v_cache");
    }

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor q = inputs[0];
        Tensor k = inputs[1];
        Tensor v = inputs[2];

        q = q.view(-1, cas.num_qo_heads, -1, cas.head_dim);
        k = k.view(-1, cas.num_kv_heads, -1, cas.head_dim);
        v = v.view(-1, cas.num_kv_heads, -1, cas.head_dim);

        if (k_cache.ready() && v_cache.ready()) {
            k = k_cache(k);
            v = v_cache(v);
        }

        k = k.transpose(SEQUENCE, DIMENSION);

        Tensor qk = Tensor::mm(q, k);
        qk = qk / std::sqrt(cas.head_dim);

        if (k_cache.ready() && v_cache.ready()) {
            qk = softmax(qk, k_cache.getCacheSeqLen());
        } else {
            qk = softmax(qk);
        }

        Tensor o = Tensor::mm(qk, v);
        o = o.view(-1, 1, -1, cas.num_qo_heads * cas.head_dim);

        return {o};
    }

    void clearCache() {
        k_cache.clearCache();
        v_cache.clearCache();
    }
};

class PagedAttention final : public Module {
    Attention attn;
    TestCase cas;

public:
    PagedAttention(TestCase cas) : cas(cas), attn(cas.num_qo_heads, cas.num_kv_heads,
                       cas.head_dim, cas.is_causal, "attenion") {}

    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Tensor q = inputs[0];
        Tensor kv_indices = inputs[1];
        Tensor kv_cache = inputs[2];

        q = q.view(-1, cas.num_qo_heads, -1, cas.head_dim);
        auto o = attn(q, kv_cache, kv_indices);

        return {o};
    }
};

void randn(float* array, int size, float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    for (int i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }
}

TestResult testCasePerformance(const TestCase& cas) {
    CPUBackend::cpu_threads = cas.thread;
    TestResult result;

    // Test NaiveAttention
    {
        auto naive = NaiveAttention(cas);
        naive.load("");

        Tensor q(1, cas.num_qo_heads, cas.q_len, cas.head_dim, Backend::global_backends[MLLM_CPU], true);
        q.setName("q");
        q.setTtype(INPUT_TENSOR);
        q.setDtype(MLLM_TYPE_F32);
        q.setModule(&naive);
        randn((float *)q.rawHostPtr(), cas.num_qo_heads * cas.q_len * cas.head_dim, 0.0, 1.0);

        Tensor k(1, cas.num_kv_heads, cas.kv_len, cas.head_dim, Backend::global_backends[MLLM_CPU], true);
        k.setName("k");
        k.setTtype(INPUT_TENSOR);
        k.setDtype(MLLM_TYPE_F32);
        k.setModule(&naive);
        randn((float *)k.rawHostPtr(), cas.num_kv_heads * cas.kv_len * cas.head_dim, 0.0, 1.0);

        Tensor v(1, cas.num_kv_heads, cas.kv_len, cas.head_dim, Backend::global_backends[MLLM_CPU], true);
        v.setName("v");
        v.setTtype(INPUT_TENSOR);
        v.setDtype(MLLM_TYPE_F32);
        v.setModule(&naive);
        randn((float *)v.rawHostPtr(), cas.num_kv_heads * cas.kv_len * cas.head_dim, 0.0, 1.0);

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            auto o = naive({q, k, v});
            naive.clearCache();
        }

        // Test iterations
        long long total_duration = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            auto start = chrono::high_resolution_clock::now();
            auto o = naive({q, k, v});
            auto end = chrono::high_resolution_clock::now();
            total_duration += chrono::duration_cast<chrono::microseconds>(end - start).count();
            naive.clearCache();
        }
        result.naive_time = static_cast<double>(total_duration) / ITERATIONS;

        // Calculate GFLOPS for naive attention
        uint64_t FLOPS = 4ULL * cas.q_len * cas.kv_len * cas.num_qo_heads * cas.head_dim;
        result.GFLOPS_naive = (static_cast<double>(FLOPS)) / (result.naive_time * 1000.0);
    }

    // Test PagedAttention
    {
        auto paged = PagedAttention(cas);
        paged.load("");

        Tensor q(1, cas.num_qo_heads, cas.q_len, cas.head_dim, Backend::global_backends[MLLM_CPU], true);
        q.setName("q");
        q.setTtype(INPUT_TENSOR);
        q.setDtype(MLLM_TYPE_F32);
        q.setModule(&paged);
        randn((float *)q.rawHostPtr(), cas.num_qo_heads * cas.q_len * cas.head_dim, 0.0, 1.0);

        Tensor kv_indices(1, 1, cas.kv_len, 1, Backend::global_backends[MLLM_CPU], true);
        kv_indices.setName("kv_indices");
        kv_indices.setTtype(INPUT_TENSOR);
        kv_indices.setDtype(MLLM_TYPE_I32);
        kv_indices.setModule(&paged);
        for(int i = 0; i < cas.kv_len; i++) {
            kv_indices.setDataAt(0, 0, i, 0, (i * cas.kv_gap) % cas.max_tokens);
        }

        Tensor kv_cache(cas.max_tokens, cas.num_kv_heads, 2, cas.head_dim, Backend::global_backends[MLLM_CPU], true);
        kv_cache.setName("kv_cache");
        kv_cache.setTtype(INPUT_TENSOR);
        kv_cache.setDtype(MLLM_TYPE_F32);
        kv_cache.setModule(&paged);
        randn((float *)kv_cache.rawHostPtr(), cas.max_tokens * cas.num_kv_heads * 2 * cas.head_dim, 0.0, 1.0);

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            auto o = paged({q, kv_indices, kv_cache});
        }

        // Test iterations
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; i++) {
            auto o = paged({q, kv_indices, kv_cache});
        }
        auto end = chrono::high_resolution_clock::now();
        result.paged_time = chrono::duration_cast<chrono::microseconds>(end - start).count() / ITERATIONS;

        // Calculate GFLOPS for paged attention
        uint64_t FLOPS = 4ULL * cas.q_len * cas.kv_len * cas.num_qo_heads * cas.head_dim;
        double scale = cas.is_causal ? (1.0 - (double)cas.q_len / (double)cas.kv_len / 2.0) : 1.0;
        result.GFLOPS_paged = (static_cast<double>(FLOPS) * scale) / (result.paged_time * 1000.0);
    }

    return result;
}

int main() {
    vector<TestCase> testCases = {
        // Prefill cases
        {32, 32, 64, 8, 32, 4096, true, 4, 1},
        {64, 64, 64, 8, 32, 4096, true, 4, 1},
        {128, 128, 64, 8, 32, 4096, true, 4, 1},
        {256, 256, 64, 8, 32, 4096, true, 4, 1},
        {512, 512, 64, 8, 32, 4096, true, 4, 1},

        // Decode cases
        {1, 32, 64, 8, 32, 4096, true, 4, 1},
        {1, 64, 64, 8, 32, 4096, true, 4, 1},
        {1, 128, 64, 8, 32, 4096, true, 4, 1},
        {1, 256, 64, 8, 32, 4096, true, 4, 1},
        {1, 512, 64, 8, 32, 4096, true, 4, 1},

        // KV gap cases
        {256, 256, 64, 8, 32, 4096, true, 4, 1},
        {256, 256, 64, 8, 32, 4096, true, 4, 13},
        {256, 256, 64, 8, 32, 4096, true, 4, 53},
        {256, 256, 64, 8, 32, 4096, true, 4, 137},
        {256, 256, 64, 8, 32, 4096, true, 4, 241}
    };

    cout << "Performance Comparison:\n";
    print_header();
    for (const auto& cas : testCases) {
        auto result = testCasePerformance(cas);
        print_results(cas, result);
    }

    return 0;
}
