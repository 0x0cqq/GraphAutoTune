import pathlib

FLOAT_INF = 1e38

# 项目根文件夹
PROJECT_PATH = pathlib.Path("/home/cqq/GraphMining/GraphAutoTuner")

# 数据集的位置
DATA_PATH = PROJECT_PATH / "data"
# 构建的目标位置
BUILD_PATH = PROJECT_PATH / "build_tuning"

# 生成的 Config 的位置
GENERATED_CONFIG_PATH = PROJECT_PATH / "include" / "generated" / "default_config.hpp"
GENERATED_CONFIG_TEMPLATE = """
#pragma once
#include "configs/config.hpp"

constexpr Config {};

"""  # 注意这里的 {} 是用来填充的，不是 C++ 的大括号
MAX_CONFIG_SPACE_SIZE = 1e6  # 最大的 Config Space 的大小

# Tuning 相关的文件的存储位置
TUNING_PATH = PROJECT_PATH / "tuning_results"

PARAM_PATH = TUNING_PATH / "param.json"
CONF_PATH = TUNING_PATH / "best_config.json"
RESULT_PATH = TUNING_PATH / "counting_time_cost.txt"
RECORD_PATH = TUNING_PATH / "record.json"

# 跑二进制文件的时候需要的前缀
RUN_COMMAND_PREFIX = "srun -p V100 --gres=gpu:v132p:1 --exclusive "

BINARY_NAME = "gpu_graph"
