#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

// 默认测试参数配置
int NUM_KV_HEAD = 8;              // 可调节的KV头数量
int HEAD_DIM = 64;                // 每个头的维度
size_t FILE_SIZE = 256 * 1024 * 1024; // 默认文件大小256MB
int TEST_ITERATIONS = 8192;       // 默认随机访问测试次数

// 解析文件大小字符串（如 "256MB", "128KB", "1G"）
size_t parse_file_size(const std::string& size_str) {
    size_t multiplier = 1;
    std::string num_str = size_str;
    std::transform(num_str.begin(), num_str.end(), num_str.begin(), ::toupper);

    if (num_str.find("KB") != std::string::npos) {
        multiplier = 1024;
        num_str = num_str.substr(0, num_str.size() - 2);
    } else if (num_str.find("MB") != std::string::npos) {
        multiplier = 1024 * 1024;
        num_str = num_str.substr(0, num_str.size() - 2);
    } else if (num_str.find("GB") != std::string::npos) {
        multiplier = 1024 * 1024 * 1024;
        num_str = num_str.substr(0, num_str.size() - 2);
    } else if (num_str.find('K') != std::string::npos) {
        multiplier = 1024;
        num_str = num_str.substr(0, num_str.size() - 1);
    } else if (num_str.find('M') != std::string::npos) {
        multiplier = 1024 * 1024;
        num_str = num_str.substr(0, num_str.size() - 1);
    } else if (num_str.find('G') != std::string::npos) {
        multiplier = 1024 * 1024 * 1024;
        num_str = num_str.substr(0, num_str.size() - 1);
    }

    try {
        size_t size = std::stoull(num_str) * multiplier;
        return size;
    } catch (const std::exception& e) {
        std::cerr << "Invalid file size format: " << size_str << "\n";
        exit(1);
    }
}

// 打印用法信息
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -h, --help            Show this help message\n"
              << "  -k, --num-kv-head     Number of KV heads (default: 8)\n"
              << "  -d, --head-dim        Dimension of each head (default: 64)\n"
              << "  -f, --file-size       File size (e.g., 256MB, 128KB, 1G) (default: 256MB)\n"
              << "  -i, --iterations      Number of test iterations (default: 8192)\n";
}

// 解析命令行参数
void parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-k" || arg == "--num-kv-head") {
            if (i + 1 < argc) {
                NUM_KV_HEAD = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing argument for " << arg << "\n";
                exit(1);
            }
        } else if (arg == "-d" || arg == "--head-dim") {
            if (i + 1 < argc) {
                HEAD_DIM = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing argument for " << arg << "\n";
                exit(1);
            }
        } else if (arg == "-f" || arg == "--file-size") {
            if (i + 1 < argc) {
                FILE_SIZE = parse_file_size(argv[++i]);
            } else {
                std::cerr << "Missing argument for " << arg << "\n";
                exit(1);
            }
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) {
                TEST_ITERATIONS = std::atoi(argv[++i]);
            } else {
                std::cerr << "Missing argument for " << arg << "\n";
                exit(1);
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }
}

// 生成随机测试数据
std::vector<float> generate_test_data() {
    std::vector<float> data(NUM_KV_HEAD * HEAD_DIM);
    for (auto& val : data) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
    return data;
}

// 连续写入测试
double test_sequential_write(const char* filename, double& tokens_per_ms) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        std::cerr << "Failed to open file for writing\n";
        return -1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    size_t total_blocks = FILE_SIZE / (NUM_KV_HEAD * HEAD_DIM * sizeof(float));
    std::vector<float> data = generate_test_data();

    for (size_t i = 0; i < total_blocks; ++i) {
        if (fwrite(data.data(), NUM_KV_HEAD * HEAD_DIM * sizeof(float), 1, file) != 1) {
            std::cerr << "Write failed at block " << i << "\n";
            fclose(file);
            return -1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    fclose(file);

    std::chrono::duration<double> diff = end - start;
    double mb_per_second = (FILE_SIZE / (1024.0 * 1024.0)) / diff.count(); // MB/s
    tokens_per_ms = (total_blocks / diff.count()) / 1000.0; // tokens/ms
    return mb_per_second;
}

// 顺序读取测试
double test_sequential_read(const char* filename, double& tokens_per_ms) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Failed to open file for reading\n";
        return -1.0;
    }

    std::vector<float> buffer(NUM_KV_HEAD * HEAD_DIM);
    size_t total_blocks = FILE_SIZE / (NUM_KV_HEAD * HEAD_DIM * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total_blocks; ++i) {
        if (fread(buffer.data(), NUM_KV_HEAD * HEAD_DIM * sizeof(float), 1, file) != 1) {
            std::cerr << "Read failed at block " << i << "\n";
            fclose(file);
            return -1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    fclose(file);

    std::chrono::duration<double> diff = end - start;
    double mb_per_second = (FILE_SIZE / (1024.0 * 1024.0)) / diff.count(); // MB/s
    tokens_per_ms = (total_blocks / diff.count()) / 1000.0; // tokens/ms
    return mb_per_second;
}

// 随机读取测试
double test_random_read(const char* filename, double& tokens_per_ms) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Failed to open file for reading\n";
        return -1.0;
    }

    std::vector<float> buffer(NUM_KV_HEAD * HEAD_DIM);
    size_t max_blocks = FILE_SIZE / (NUM_KV_HEAD * HEAD_DIM * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        size_t block_pos = rand() % max_blocks;
        if (fseek(file, block_pos * NUM_KV_HEAD * HEAD_DIM * sizeof(float), SEEK_SET) != 0) {
            std::cerr << "Fseek failed\n";
            fclose(file);
            return -1.0;
        }

        if (fread(buffer.data(), NUM_KV_HEAD * HEAD_DIM * sizeof(float), 1, file) != 1) {
            std::cerr << "Read failed at block " << block_pos << "\n";
            fclose(file);
            return -1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    fclose(file);

    std::chrono::duration<double> diff = end - start;
    double mb_per_second = (TEST_ITERATIONS * NUM_KV_HEAD * HEAD_DIM * sizeof(float) / (1024.0 * 1024.0)) / diff.count(); // MB/s
    tokens_per_ms = (TEST_ITERATIONS / diff.count()) / 1000.0; // tokens/ms
    return mb_per_second;
}

// 随机写入测试
double test_random_write(const char* filename, double& tokens_per_ms) {
    FILE* file = fopen(filename, "r+b");
    if (!file) {
        std::cerr << "Failed to open file for writing\n";
        return -1.0;
    }

    std::vector<float> data = generate_test_data();
    size_t max_blocks = FILE_SIZE / (NUM_KV_HEAD * HEAD_DIM * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        size_t block_pos = rand() % max_blocks;
        if (fseek(file, block_pos * NUM_KV_HEAD * HEAD_DIM * sizeof(float), SEEK_SET) != 0) {
            std::cerr << "Fseek failed\n";
            fclose(file);
            return -1.0;
        }

        if (fwrite(data.data(), NUM_KV_HEAD * HEAD_DIM * sizeof(float), 1, file) != 1) {
            std::cerr << "Write failed at block " << block_pos << "\n";
            fclose(file);
            return -1.0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    fclose(file);

    std::chrono::duration<double> diff = end - start;
    double mb_per_second = (TEST_ITERATIONS * NUM_KV_HEAD * HEAD_DIM * sizeof(float) / (1024.0 * 1024.0)) / diff.count(); // MB/s
    tokens_per_ms = (TEST_ITERATIONS / diff.count()) / 1000.0; // tokens/ms
    return mb_per_second;
}

int main(int argc, char* argv[]) {
    parse_arguments(argc, argv);

    const char* test_file = "/data/local/tmp/io_benchmark.bin";
    srand(time(nullptr));

    double tokens_per_ms;

    std::cout << "Starting sequential write test...\n";
    double write_speed = test_sequential_write(test_file, tokens_per_ms);
    if (write_speed > 0) {
        std::cout << "Sequential write speed: " << write_speed << " MB/s\n";
//        std::cout << "Sequential write throughput: " << tokens_per_ms << " tokens/ms\n";
    }

    std::cout << "Starting sequential read test...\n";
    double read_speed = test_sequential_read(test_file, tokens_per_ms);
    if (read_speed > 0) {
        std::cout << "Sequential read speed: " << read_speed << " MB/s\n";
//        std::cout << "Sequential read throughput: " << tokens_per_ms << " tokens/ms\n";
    }

    std::cout << "Starting random read test...\n";
    double random_read_speed = test_random_read(test_file, tokens_per_ms);
    if (random_read_speed > 0) {
        std::cout << "Random read speed: " << random_read_speed << " MB/s\n";
//        std::cout << "Random read throughput: " << tokens_per_ms << " tokens/ms\n";
    }

    std::cout << "Starting random write test...\n";
    double random_write_speed = test_random_write(test_file, tokens_per_ms);
    if (random_write_speed > 0) {
        std::cout << "Random write speed: " << random_write_speed << " MB/s\n";
//        std::cout << "Random write throughput: " << tokens_per_ms << " tokens/ms\n";
    }

    remove(test_file);
    return 0;
}
