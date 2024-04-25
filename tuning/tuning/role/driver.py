import logging
import os
from typing import List

from ..common.const import *
from ..config.details import Config

logger = logging.getLogger("driver")


class Driver:
    @staticmethod
    def compile(config: Config):

        definitions = ""

        build_path = BUILD_PATH / str(hash(config))

        logger.debug(f"build_path = {build_path}")

        if not os.path.exists(build_path):
            os.makedirs(build_path)
        else:
            logger.debug("Build path already exists, skipping compilation")
            return

        os.chdir(build_path)

        cmake_command = "cmake " + definitions + " " + PROJECT_PATH
        logger.debug(f"Generating makefile with CMake: {cmake_command}")
        # without stdout
        ret_code = os.system(cmake_command)

        assert ret_code == 0, f"CMake exited with non-zero code {ret_code}"

        make_command = "make -j"
        logger.debug(f"Compiling with Make: {make_command}")
        ret_code = os.system(make_command)
        assert ret_code == 0, f"Make exited with non-zero code {ret_code}"

    @staticmethod
    def run(job: str, options: List[str], config: Config) -> float:
        """运行程序

        Returns:
            float: 时间
        """

        build_path = BUILD_PATH / str(hash(config))

        if not os.path.exists(build_path):
            logger.warning("Build path does not exist, compiling...")
            Driver.compile(config)
        else:
            logger.debug("Build path already exists, skipping compilation")

        os.chdir(build_path)

        logger.debug(f"Running Job: <{job}> with options: <{options}>")

        run_command = f"{RUN_COMMAND_PREFIX} ./{job} {' '.join(options)}"
        logger.debug(f"Running Command: {run_command}")
        ret_code = os.system(run_command)

        if ret_code != 0:
            logger.warning(f"Graph mining program returned non-zero code {ret_code}")
            return FLOAT_INF

        with open(RESULT_PATH, "r") as f:
            time_cost = float(f.read().strip())

        logger.info(f"Time cost: {time_cost:.2f} s\n")

        return time_cost
