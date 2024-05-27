#pragma once
#include <filesystem>
#include <string>

#include "consts/project_consts.hpp"

template <bool x = false>
void assert_false() {
    static_assert(x, "This should never be reached");
}

void output_count_file(std::string hash_code, long long total_count) {
    // 创建文件夹
    if (!std::filesystem::exists(RESULTS_DIR)) {
        std::filesystem::create_directory(RESULTS_DIR);
    }
    if (!std::filesystem::exists(RESULTS_DIR / hash_code)) {
        std::filesystem::create_directory(RESULTS_DIR / hash_code);
    }

    std::ofstream count_file{RESULTS_DIR / hash_code / COUNT_FILE_NAME};

    if (!count_file.is_open()) {
        throw std::runtime_error("Cannot open the time file");
    }

    count_file << total_count << std::endl;
}

void output_duration(std::string hash_code, double duration) {
    // 创建文件夹
    if (!std::filesystem::exists(RESULTS_DIR)) {
        std::filesystem::create_directory(RESULTS_DIR);
    }
    if (!std::filesystem::exists(RESULTS_DIR / hash_code)) {
        std::filesystem::create_directory(RESULTS_DIR / hash_code);
    }

    std::ofstream time_file{RESULTS_DIR / hash_code / TIME_FILE_NAME};

    if (!time_file.is_open()) {
        throw std::runtime_error("Cannot open the time file");
    }

    time_file << duration << std::endl;
}

void output_result_files(std::string hash_code, double duration,
                         long long total_count) {
    output_duration(hash_code, duration);
    output_count_file(hash_code, total_count);
}