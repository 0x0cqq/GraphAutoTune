import json
import logging
import os
from typing import List

from ..common.const import *
from ..config.details import Config

logger = logging.getLogger("driver")


class Driver:
    @staticmethod
    def compile(config: Config):
        """根据Config生成可执行文件

        Args:
            config (Config): 配置 config 对象
        """

        definitions = ""

        config_hash = config.fingerprint()
        build_path = BUILD_PATH / config_hash
        logger.debug(f"build_path = {build_path}")

        if not os.path.exists(build_path):
            os.makedirs(build_path)
        with open(get_config_path(config_hash), "w") as f:
            f.write(json.dumps(config.get_flat_dict()))

        os.chdir(build_path)

        config.export()
        logger.info(f"Compile with config hash {config_hash}")

        cmake_command = "cmake " + definitions + " " + str(PROJECT_PATH)
        logger.info(
            f"Generating makefile with CMake: {cmake_command} for config {config}"
        )
        # without stdout
        ret_code = os.system(cmake_command + " > /dev/null 2>&1")

        assert ret_code == 0, f"CMake exited with non-zero code {ret_code}"

        make_command = "make -j"
        logger.info(f"Compiling with Make: {make_command} for config {config}")
        ret_code = os.system(make_command + " > /dev/null 2>&1")
        assert ret_code == 0, f"Make exited with non-zero code {ret_code}"

        logger.debug("Compilation finished")

    @staticmethod
    def run(job: str, options: List[str], config: Config) -> float:
        """
        根据 Config 运行程序

        Returns:
            float: 时间
        """

        if not os.path.exists(TUNING_PATH):
            os.makedirs(TUNING_PATH)

        config_hash = config.fingerprint()
        if not os.path.exists(TUNING_PATH / config_hash):
            os.makedirs(TUNING_PATH / config_hash)

        options = options + [config_hash]

        build_path = BUILD_PATH / config_hash
        Driver.compile(config)
        os.chdir(build_path)

        logger.info(f"Running Job: <{job}> with options: <{options}>")

        run_command = f"{RUN_COMMAND_PREFIX} ./{job} {' '.join(options)}"
        logger.debug(f"Running Command: {run_command}")
        result_path = get_result_path(config_hash)
        ret_code = os.system(run_command + f" > {result_path.as_posix()} 2>&1")

        if ret_code != 0:
            logger.warning(f"Graph mining program returned non-zero code {ret_code}")
            return FLOAT_INF

        with open(get_time_path(config_hash).as_posix(), "r") as f:
            time_cost = float(f.read().strip())

        with open(get_counting_path(config_hash).as_posix(), "r") as f:
            count = int(f.read().strip())

        logger.info(f"Time cost: {time_cost:.10f}s  count: {count}\n")

        return time_cost
