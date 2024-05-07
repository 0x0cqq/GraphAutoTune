#pragma once
#include <filesystem>

const std::filesystem::path PROJECT_ROOT{
    "/home/cqq/GraphMining/GraphAutoTuner"};

// 和 Python const.py 保持一致
const std::filesystem::path TIME_RESULT_PATH{PROJECT_ROOT / "tuning_results" /
                                             "counting_time_cost.txt"};