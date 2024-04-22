import logging
import os
from typing import List

from ..common.const import *

logger = logging.getLogger("tuner")


class Driver:
    @staticmethod
    def compile(param_dict: dict):
        """按照参数编译程序

        Args:
            param_dict (dict): 参数的词典

        """

        definitions = ""

        for key, val in param_dict.items():
            definitions += f"-D{key}={val} "

        logger.debug(f"BUILD_PATH = {BUILD_PATH}")
        os.chdir(BUILD_PATH)

        cmake_command = "cmake " + definitions + ".."
        logger.debug(f"Generating makefile with CMake: {cmake_command}")
        ret_code = os.system(cmake_command)

        assert ret_code == 0, f"CMake exited with non-zero code {ret_code}"

        make_command = "make -j"
        logger.debug(f"Compiling with Make: {make_command}")
        ret_code = os.system(make_command)
        assert ret_code == 0, f"Make exited with non-zero code {ret_code}"

    @staticmethod
    def run(job: str, options: List[str]) -> float:
        """运行程序

        Returns:
            float: 时间
        """

        os.chdir(BUILD_PATH / "bin")

        logger.debug(f"Running Job: {job} options: {options}")

        run_command = f"{COMMAND_PREFIX} ./{job} {' '.join(options)}"
        logger.debug(f"Running Command: {run_command}")
        ret_code = os.system(run_command)

        if ret_code != 0:
            logger.warning(f"Graph mining program returned non-zero code {ret_code}")
            return FLOAT_INF

        with open(RESULT_PATH, "r") as f:
            time_cost = float(f.readline())

        logger.info(f"Time cost: {time_cost:.2f} s\n")

        return time_cost
