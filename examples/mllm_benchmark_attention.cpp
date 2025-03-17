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

        // 分配对齐的内存
        void* ptr = std::aligned_alloc(alignment, size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        // 记录分配的内存
        allocated_blocks.insert(ptr);
        return ptr;
    }

    // 释放单片内存
    void free(void* ptr) {
        if (!ptr) return;

        // 检查内存是否由本分配器分配
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) {
            throw std::invalid_argument("Pointer not allocated by this allocator");
        }

        // 释放内存并从记录中移除
        std::free(ptr);
        allocated_blocks.erase(it);
    }

    // 释放所有分配的内存
    void free_all() {
        for (void* ptr : allocated_blocks) {
            std::free(ptr);
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
        input.kv_indices[i] = i;
}

using AttentionFunc = std::function<void(
    float* queries, int* qo_indptr, int qo_indptr_size,
    float* kv_cache,
    int* kv_indices, int* kv_indptr, int kv_indptr_size,
    bool is_causal,
    bool* mask,
    int num_qo_heads, int num_kv_heads, int head_dim,
    float* attn_out, int thread_count)>;

void Benchmark(AttentionFunc attention_func, TestCase &cas) {
    const int warmup_times = 5;
    const int infer_times = 10;

    AlignedAllocator allocator;
    const int alignment = 64;

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
    double GFLOPS = static_cast<double>(FLOPS) / nanoseconds;

    // Print results in a table format
    std::cout << std::setw(8) << cas.q_len << " | "
              << std::setw(8) << cas.kv_len << " | "
              << std::setw(8) << cas.head_dim << " | "
              << std::setw(12) << cas.num_kv_heads << " | "
              << std::setw(12) << cas.num_qo_heads << " | "
              << std::setw(10) << cas.max_tokens << " | "
              << std::setw(8) << cas.is_causal << " | "
              << std::setw(8) << milliseconds << " | "
              << std::setw(14) << microseconds << " | "
              << std::setw(14) << nanoseconds << " | "
              << std::setw(10) << GFLOPS << std::endl;
}

void print_table_head(){
    // Print table header
    std::cout << std::setw(8) << "q_len" << " | "
              << std::setw(8) << "kv_len" << " | "
              << std::setw(8) << "head_dim" << " | "
              << std::setw(12) << "num_kv_heads" << " | "
              << std::setw(12) << "num_qo_heads" << " | "
              << std::setw(10) << "max_tokens" << " | "
              << std::setw(8) << "is_causal" << " | "
              << std::setw(8) << "Time (ms)" << " | "
              << std::setw(14) << "Time (us)" << " | "
              << std::setw(14) << "Time (ns)" << " | "
              << std::setw(10) << "GFLOPS" << std::endl;
    std::cout << std::string(140, '-') << std::endl;
}

void benchmark() {
    const int thread = 1;

    // Test different configurations
    std::vector<TestCase> test_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread
        {  32,    32,     64,       8,            32,           1024,       false,     thread},
        {  64,    64,     64,       8,            32,           1024,       false,     thread},
        {  128,   128,    64,       8,            32,           1024,       false,     thread},
        {  256,   256,    64,       8,            32,           1024,       false,     thread},
        {  512,   512,    64,       8,            32,           1024,       false,     thread},
        {  1024,   1024,    64,       8,            32,           1024,       false,     thread},
    };

    // Test different Attention implementations
    std::vector<std::pair<std::string, AttentionFunc>> attention_funcs = {
        {"AttentionFP32", AttentionFP32},
        {"v2", AttentionFP32_v2},
    };

    for (const auto& func : attention_funcs) {
        std::cout << "Testing function: " << func.first << std::endl;
        print_table_head();
        for (auto& test_case : test_cases) {
            Benchmark(func.second, test_case);
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
//    benchmark();

    TestCase cas = {  32,
                    32,
                    64,
                    8,
                    32,
                    1024,
                    false,
                    1};

    TestInput input, input2;
    AlignedAllocator allocator;
    genInput(allocator, cas, input);
    genInput(allocator, cas, input2);

    AttentionFP32(
        input.queries, input.qo_indptr, 2,
        input.kv_cache,
        input.kv_indices, input.kv_indptr, 2,
        cas.is_causal,
        nullptr, // mask
        cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
        input.attn_out, cas.thread);

    AttentionFP32_v2(input.queries, input.qo_indptr, 2,
                     input.kv_cache,
                     input.kv_indices, input.kv_indptr, 2,
                     cas.is_causal,
                     nullptr, // mask
                     cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
                     input2.attn_out, cas.thread);

    if (compareFloatArray(input.attn_out, input2.attn_out, cas.q_len * cas.num_qo_heads * cas.head_dim)) {
        // print red color pass
        std::cout << "\033[1;32mPass\033[0m" << std::endl;
    } else {
        // print green color fail and first 10 elements
        std::cout << "\033[1;31mFail\033[0m" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << input.attn_out[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << input2.attn_out[i] << " ";
        }
        std::cout << std::endl;
    }


    Attention<16, __m512, __m512, float, float, float> att(
        input.queries, input.qo_indptr, 2,
        input.kv_cache, input.kv_indices, input.kv_indptr, 2,
        cas.is_causal, input.mask, cas.num_qo_heads, cas.num_kv_heads, cas.head_dim,
        input.attn_out);

    return 0;
}
