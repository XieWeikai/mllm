#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <unordered_set>
#include <cstdlib> // for std::aligned_alloc and std::free
#include <functional>
#include <iomanip>

#include "backends/cpu/compute/Attention.hpp"

class AlignedAllocator {
public:
    // 构造函数
    AlignedAllocator() = default;

    // 析构函数：自动释放所有分配的内存
    ~AlignedAllocator() {
        free_all();
    }

    // 分配对齐的内存
    void* alloc_aligned(size_t size, size_t alignment) {
        // 检查对齐值是否是 2 的幂
        if ((alignment == 0) || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be a power of two");
        }

        // 分配额外的内存以确保对齐
        size_t extra = alignment - 1 + sizeof(void*); // 额外空间用于存储原始指针
        void* raw_memory = std::malloc(size + extra);
        if (!raw_memory) {
            throw std::bad_alloc();
        }

        // 计算对齐后的地址
        void* aligned_memory = reinterpret_cast<void*>(
            (reinterpret_cast<uintptr_t>(raw_memory) + extra) & ~(alignment - 1)
        );

        // 在对齐的内存前面存储原始指针，以便后续释放
        *(reinterpret_cast<void**>(aligned_memory) - 1) = raw_memory;

        // 记录分配的内存
        allocated_blocks.insert(aligned_memory);
        return aligned_memory;
    }

    // 释放单片内存
    void free(void* ptr) {
        if (!ptr) return;

        // 检查内存是否由本分配器分配
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            throw std::invalid_argument("Pointer not allocated by this allocator");
        }

        // 获取原始指针并释放
        void* raw_memory = *(reinterpret_cast<void**>(ptr) - 1);
        std::free(raw_memory);

        // 从记录中移除
        allocated_blocks.erase(it);
    }

    // 释放所有分配的内存
    void free_all() {
        for (void* ptr : allocated_blocks) {
            void* raw_memory = *(reinterpret_cast<void**>(ptr) - 1);
            std::free(raw_memory);
        }
        allocated_blocks.clear();
    }

    // 禁止拷贝和赋值
    AlignedAllocator(const AlignedAllocator&) = delete;
    AlignedAllocator& operator=(const AlignedAllocator&) = delete;

private:
    // 记录所有分配的内存块
    std::unordered_set<void*> allocated_blocks;
};

void randn(float* data, size_t size, float mean = 0.0f, float stddev = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);

    for (size_t i = 0; i < size; ++i) {
        data[i] = d(gen);
    }
}

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

struct TestInput {
    float* queries;
    int* qo_indptr;
    float* kv_cache;
    int* kv_indices;
    int* kv_indptr;
    float* attn_out;
    bool *mask;
};

struct TestResult {
    double milliseconds;
    double microseconds;
    double nanoseconds;
    double GFLOPS;
};

struct FieldDescriptor {
    std::string header;       // 列标题
    int width;                // 列宽
    std::function<std::string(const TestCase&, const TestResult&)> formatter; // 数据格式化方法
};

const std::vector<FieldDescriptor> field_descriptors = {
    {"q_len",        8, [](auto& c, auto&) { return std::to_string(c.q_len); }},
    {"kv_gap",       8, [](auto& c, auto&) { return std::to_string(c.kv_gap); }},
    {"kv_len",       8, [](auto& c, auto&) { return std::to_string(c.kv_len); }},
    {"head_dim",     8, [](auto& c, auto&) { return std::to_string(c.head_dim); }},
    {"num_kv_heads",12, [](auto& c, auto&) { return std::to_string(c.num_kv_heads); }},
    {"num_qo_heads",12, [](auto& c, auto&) { return std::to_string(c.num_qo_heads); }},
    {"max_tokens",  10, [](auto& c, auto&) { return std::to_string(c.max_tokens); }},
    {"is_causal",    8, [](auto& c, auto&) { return c.is_causal ? "true" : "false"; }},
    {"Time (ms)",    8, [](auto&, auto& r) {
         std::ostringstream oss;
         oss << std::fixed << std::setprecision(2) << r.milliseconds;
         return oss.str();
     }},
    {"GFLOPS",      10, [](auto& c, auto& r) {
         std::ostringstream oss;
         oss << std::fixed << std::setprecision(4) << r.GFLOPS;
         return oss.str();}},
    {"thread",       8, [](auto& c, auto&) { return std::to_string(c.thread); }},
    {"throughput",  12, [](auto& c, auto& r) {
         std::ostringstream oss;
         double bytes = c.q_len * c.num_qo_heads * c.head_dim * sizeof(float);
         double throughput_MB_ms = (bytes / r.nanoseconds) * ((1000.0 * 1000.0) / (1024.0 * 1024.0));
            oss << std::fixed << std::setprecision(4) << 1000.0 * throughput_MB_ms << "MB/s";
         return oss.str();
     }},
};

void print_header() {
    // 打印表头
    for (const auto& fd : field_descriptors) {
        std::cout << std::setw(fd.width) << fd.header << " | ";
    }
    std::cout << "\n";

    // 打印分隔线
    for (const auto& fd : field_descriptors) {
        std::cout << std::string(fd.width, '-') << "-+-";
    }
    std::cout << "\n";
}

void print_results(const TestCase& cas, const TestResult& result) {
    // 打印数据行
    for (const auto& fd : field_descriptors) {
        std::cout << std::setw(fd.width) << fd.formatter(cas, result) << " | ";
    }
    std::cout << "\n";
}

void genInput(AlignedAllocator &allocator, TestCase &c, TestInput &input) {
    const int alignment = 64;

    input.queries = (float *)allocator.alloc_aligned(c.q_len * c.num_qo_heads * c.head_dim * sizeof(float), alignment);
    input.qo_indptr = (int *)allocator.alloc_aligned(2 * sizeof(int), alignment);
    input.kv_cache = (float *)allocator.alloc_aligned(c.max_tokens * 2 * c.num_kv_heads * c.head_dim * sizeof(float), alignment);
    input.kv_indices = (int *)allocator.alloc_aligned(c.kv_len * sizeof(int), alignment);
    input.kv_indptr = (int *)allocator.alloc_aligned(2 * sizeof(int), alignment);
    input.attn_out = (float *)allocator.alloc_aligned(c.q_len * c.num_qo_heads * c.head_dim * sizeof(float), alignment);
    input.mask = nullptr;

    input.qo_indptr[0] = 0;
    input.qo_indptr[1] = c.q_len;
    input.kv_indptr[0] = 0;
    input.kv_indptr[1] = c.kv_len;

    randn(input.queries, c.q_len * c.num_qo_heads * c.head_dim);
    randn(input.kv_cache, c.max_tokens * 2 * c.num_kv_heads * c.head_dim);
    for (int i = 0; i < c.kv_len; i++)
        input.kv_indices[i] = (i * c.kv_gap) % c.max_tokens;
}

using AttentionFunc = std::function<void(
    float* queries, int* qo_indptr, int qo_indptr_size,
    float* kv_cache,
    int* kv_indices, int* kv_indptr, int kv_indptr_size,
    bool is_causal,
    bool* mask,
    int num_qo_heads, int num_kv_heads, int head_dim,
    float* attn_out, int thread_count)>;

TestResult Benchmark(AttentionFunc attention_func, TestCase &cas) {
    const int warmup_times = 10;
    const int infer_times = 20;

    AlignedAllocator allocator;

    TestInput input;
    genInput(allocator, cas, input);

    // Warmup
    for (int i = 0; i < warmup_times; ++i) {
        attention_func(
            input.queries, input.qo_indptr, 2,
            input.kv_cache,
            input.kv_indices, input.kv_indptr, 2,
            cas.is_causal,
            nullptr, // mask
            cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
            input.attn_out, cas.thread);
    }

    // Run and measure time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < infer_times; ++i) {
        attention_func(
            input.queries, input.qo_indptr, 2,
            input.kv_cache,
            input.kv_indices, input.kv_indptr, 2,
            cas.is_causal,
            nullptr, // mask
            cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
            input.attn_out, cas.thread);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate latency
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)infer_times;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)infer_times;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)infer_times;

    // Calculate GFLOPS
    uint64_t FLOPS = 4ULL * cas.q_len * cas.kv_len * cas.num_qo_heads * cas.head_dim;
    double scale = 1.0 - (double)cas.q_len / (double)cas.kv_len / 2.0;
    double GFLOPS = static_cast<double>(FLOPS) / nanoseconds;
    if(cas.is_causal)
        GFLOPS *= scale;

    TestResult res = {milliseconds, microseconds, nanoseconds, GFLOPS};
    return res;
}

void benchmark(std::vector <TestCase> &test_cases, std::vector<std::pair<std::string, AttentionFunc>> &attention_funcs) {
    for (const auto& func : attention_funcs) {
        std::cout << "Testing function: " << func.first << std::endl;
        print_header();
        for (auto& test_case : test_cases) {
            auto res = Benchmark(func.second, test_case);
            print_results(test_case, res);
        }
        std::cout << std::string(140, '-') << std::endl;
    }
}

bool compareFloatArray(const float* a, const float* b, size_t size, float epsilon = 1e-6) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void printFloatArray(const float *a, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << a[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main(){
    const int thread = 4;
    // Test different configurations
    std::vector<TestCase> prefill_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {  32,    32,     64,       8,            32,           10240, true,     thread},
        {  64,    64,     64,       8,            32,           10240, true,     thread},
        {  128,   128,    64,       8,            32,           10240, true,     thread},
        {  256,   256,    64,       8,            32,           10240, true,     thread},
        {  512,   512,    64,       8,            32,           10240, true,     thread},
        {  1024,   1024,    64,       8,            32,           10240, true,     thread}
    };

    // Test different Attention implementations
    std::vector<std::pair<std::string, AttentionFunc>> attention_funcs = {
        {"AttentionFP32", AttentionFP32},
        //        {"v2", AttentionFP32_v2},
    };

    printf("prefill_cases: %zu\n", prefill_cases.size());
    benchmark(prefill_cases, attention_funcs);

    std::vector<TestCase> decode_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {  1,    32,     64,       8,            32,           10240, true,     thread},
        {  1,    64,     64,       8,            32,           10240, true,     thread},
        {  1,   128,    64,       8,            32,           10240, true,     thread},
        {  1,   256,    64,       8,            32,           10240, true,     thread},
        {  1,   512,    64,       8,            32,           10240, true,     thread},
        {  1,   1024,    64,       8,            32,           10240, true,     thread}
    };

    printf("decode_cases: %zu\n", decode_cases.size());
    benchmark(decode_cases, attention_funcs);

    std::vector<TestCase> extend_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {  16,    32,     64,       8,            32,           10240, true,     thread},
        {  32,    64,     64,       8,            32,           10240, true,     thread},
        {  64,   128,    64,       8,            32,           10240, true,     thread},
        {  128,   256,    64,       8,            32,           10240, true,     thread},
        {  256,   512,    64,       8,            32,           10240, true,     thread},
        {  512,   1024,    64,       8,            32,           10240, true,     thread}
    };

    printf("extend_cases: %zu\n", extend_cases.size());
    benchmark(extend_cases, attention_funcs);

    std::vector<TestCase> kv_gap_prefill_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {256, 256, 64, 8, 32, 10240, true, thread, 1},
        {256, 256, 64, 8, 32, 10240, true, thread, 13},
        {256, 256, 64, 8, 32, 10240, true, thread, 53},
        {256, 256, 64, 8, 32, 10240, true, thread, 71},
        {256, 256, 64, 8, 32, 10240, true, thread, 137},
        {256, 256, 64, 8, 32, 10240, true, thread, 191},
        {256, 256, 64, 8, 32, 10240, true, thread, 241},
        {256, 256, 64, 8, 32, 10240, true, thread, 613},
    };

    printf("kv_gap_prefill_cases: %zu\n", kv_gap_prefill_cases.size());
    benchmark(kv_gap_prefill_cases, attention_funcs);

    std::vector<TestCase> kv_gap_decode_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {1, 256, 64, 8, 32, 10240, true, thread, 1},
        {1, 256, 64, 8, 32, 10240, true, thread, 13},
        {1, 256, 64, 8, 32, 10240, true, thread, 53},
        {1, 256, 64, 8, 32, 10240, true, thread, 71},
        {1, 256, 64, 8, 32, 10240, true, thread, 137},
        {1, 256, 64, 8, 32, 10240, true, thread, 191},
        {1, 256, 64, 8, 32, 10240, true, thread, 241},
        {1, 256, 64, 8, 32, 10240, true, thread, 613},
    };

    printf("kv_gap_decode_cases: %zu\n", kv_gap_decode_cases.size());
    benchmark(kv_gap_decode_cases, attention_funcs);

//    TestCase cas = {  32,
//                    32,
//                    64,
//                    8,
//                    32,
//                    1024,
//                    false,
//                    1};
//
//    TestInput input, input2;
//    AlignedAllocator allocator;
//    genInput(allocator, cas, input);
//    genInput(allocator, cas, input2);
//
//    AttentionFP32(
//        input.queries, input.qo_indptr, 2,
//        input.kv_cache,
//        input.kv_indices, input.kv_indptr, 2,
//        cas.is_causal,
//        nullptr, // mask
//        cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
//        input.attn_out, cas.thread);
//
//    AttentionFP32_v2(input.queries, input.qo_indptr, 2,
//                     input.kv_cache,
//                     input.kv_indices, input.kv_indptr, 2,
//                     cas.is_causal,
//                     nullptr, // mask
//                     cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
//                     input2.attn_out, cas.thread);
//
//    if (compareFloatArray(input.attn_out, input2.attn_out, cas.q_len * cas.num_qo_heads * cas.head_dim)) {
//        // print red color pass
//        std::cout << "\033[1;32mPass\033[0m" << std::endl;
//    } else {
//        // print green color fail and first 10 elements
//        std::cout << "\033[1;31mFail\033[0m" << std::endl;
//        for (int i = 0; i < 10; i++) {
//            std::cout << input.attn_out[i] << " ";
//        }
//        std::cout << std::endl;
//        for (int i = 0; i < 10; i++) {
//            std::cout << input2.attn_out[i] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//
//    Attention<16, __m512, __m512, float, float, float> att(
//        input.queries, input.qo_indptr, 2,
//        input.kv_cache, input.kv_indices, input.kv_indptr, 2,
//        cas.is_causal, input.mask, cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
//        input.attn_out);

    return 0;
}
