#pragma once
#include <filesystem>
#include <string>

const std::filesystem::path PROJECT_ROOT{
    "/home/cqq/GraphMining/GraphAutoTuner"};

// 和 Python const.py 保持一致
const std::filesystem::path RESULTS_DIR{PROJECT_ROOT / "tuning_results"};

const std::string TIME_FILE_NAME = "time.txt";
const std::string COUNT_FILE_NAME = "count.txt";