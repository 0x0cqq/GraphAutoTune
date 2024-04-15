import pathlib

FLOAT_INF = 1e38

COMMAND_PREFIX = "srun -p V100 --gres=gpu:v132p:1 --exclusive "
# COMMAND_PREFIX = ""
DATA_PATH = pathlib.Path(__file__).parent.absolute() / ".." / "dataset"
BUILD_PATH = pathlib.Path(__file__).parent.absolute() / ".." / "build"
PARAM_PATH = pathlib.Path(__file__).parent.absolute() / "param.json"
CONF_PATH = pathlib.Path(__file__).parent.absolute() / "best_config.json"
RESULT_PATH = pathlib.Path(__file__).parent.absolute() / "counting_time_cost.txt"
RECORD_PATH = pathlib.Path(__file__).parent.absolute() / "record.json"
PARAM_VAL = {
    "USE_ARRAY": [1],
    "THREADS_PER_BLOCK": [32, 64, 128, 256, 512, 1024],
    "NUM_BLOCKS": [32, 64, 128, 256, 512, 1024],
    "IEP_BY_SM": [0, 1],
    "MAXREG": [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "SORT": [0, 1],
    "PARALLEL_INTERSECTION": [0, 1],
    "BINARY_SEARCH": [0, 1],
}

JOB_NAME = "gpu_graph"
