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

using AttentionFunc = std::function<void(
    float* queries, int* qo_indptr, int qo_indptr_size,
    float* kv_cache,
    int* kv_indices, int* kv_indptr, int kv_indptr_size,
    bool is_causal,
    bool* mask,
    int num_qo_heads, int num_kv_heads, int head_dim,
    float* attn_out, int thread_count)>;

void Benchmark(AttentionFunc attention_func, int q_len, int kv_len, int head_dim, int num_kv_heads, int num_qo_heads, int max_tokens, bool is_causal, int thread) {
    const int warmup_times = 5;
    const int infer_times = 10;

    AlignedAllocator allocator;
    const int alignment = 64;

    // Allocate memory
    auto queries = (float *)allocator.alloc_aligned(q_len * num_qo_heads * head_dim * sizeof(float), alignment);
    int qo_indptr[] = {0, q_len};
    auto kv_cache = (float *)allocator.alloc_aligned(max_tokens * 2 * num_kv_heads * head_dim * sizeof(float), alignment);
    auto kv_indices = (int *)allocator.alloc_aligned(kv_len * sizeof(int), alignment);
    int kv_indptr[] = {0, kv_len};
    auto attn_out = (float *)allocator.alloc_aligned(q_len * num_qo_heads * head_dim * sizeof(float), alignment);

    // Initialize data
    randn(queries, q_len * num_qo_heads * head_dim);
    randn(kv_cache, max_tokens * 2 * num_kv_heads * head_dim);
    for (int i = 0; i < kv_len; i++)
        kv_indices[i] = i;

    // Warmup
    for (int i = 0; i < warmup_times; ++i) {
        attention_func(
            queries, qo_indptr, 2,
            kv_cache,
            kv_indices, kv_indptr, 2,
            is_causal,
            nullptr, // mask
            num_qo_heads, num_kv_heads, head_dim,
            attn_out, thread);
    }

    // Run and measure time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < infer_times; ++i) {
        attention_func(
            queries, qo_indptr, 2,
            kv_cache,
            kv_indices, kv_indptr, 2,
            is_causal,
            nullptr, // mask
            num_qo_heads, num_kv_heads, head_dim,
            attn_out, thread);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate latency
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)infer_times;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)infer_times;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)infer_times;

    // Calculate GFLOPS
    uint64_t FLOPS = 4ULL * q_len * kv_len * num_qo_heads * head_dim;
    double GFLOPS = static_cast<double>(FLOPS) / nanoseconds;

    // Print results in a table format
    std::cout << std::setw(8) << q_len << " | "
              << std::setw(8) << kv_len << " | "
              << std::setw(8) << head_dim << " | "
              << std::setw(12) << num_kv_heads << " | "
              << std::setw(12) << num_qo_heads << " | "
              << std::setw(10) << max_tokens << " | "
              << std::setw(8) << is_causal << " | "
              << std::setw(8) << milliseconds << "ms | "
              << std::setw(14) << microseconds << "us | "
              << std::setw(14) << nanoseconds << "ns | "
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

int main(){
    const int thread = 4;

    // Test different configurations
    std::vector<std::tuple<int, int, int, int, int, int, bool>> test_cases = {
        // q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal
        {  32,    32,     64,       8,            32,           1024,       false},
        {  64,    64,     64,       8,            32,           1024,       false},
        {  128,   128,    64,       8,            32,           1024,       false},
        {  256,   256,    64,       8,            32,           1024,       false},
        {  512,   512,    64,       8,            32,           1024,       false},
    };

    // Test different Attention implementations
    std::vector<std::pair<std::string, AttentionFunc>> attention_funcs = {
        {"AttentionFP32", AttentionFP32},
//        {"v2", AttentionFP32_v2},
    };

    for (const auto& func : attention_funcs) {
        std::cout << "Testing function: " << func.first << std::endl;
        print_table_head();
        for (const auto& test_case : test_cases) {
            auto [q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal] = test_case;
            Benchmark(func.second, q_len, kv_len, head_dim, num_kv_heads, num_qo_heads, max_tokens, is_causal, thread);
        }
        std::cout << std::string(140, '-') << std::endl;
    }

    return 0;
}
