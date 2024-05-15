import json
import logging

from tuning.common.argparser import parser
from tuning.common.const import *
from tuning.config.details import Config
from tuning.role.driver import Driver
from tuning.role.manipulator import Manipulator
from tuning.role.modeling import Modeling
from tuning.role.tuner import Tuner

logger = logging.getLogger("main")

logging.basicConfig(level=logging.INFO)

JOB_NAME = "./bin/pm"


if __name__ == "__main__":

    args = parser.parse_args()

    if args.clear is True:
        logger.info("Clear the tuning result and start from scratch")
        # remove all files in build_tuning and tuning_results folder
        import os

        command = f"rm -rf {TUNING_PATH} {BUILD_PATH}"
        os.system(command)
        logger.info("Clear previous results, done.")

    # create context
    model = Modeling(Config)
    driver = Driver(Config, JOB_NAME, [args.data, args.pattern])
    manip = Manipulator(model, driver)
    tuner = Tuner(manip)

    mode: str = args.run_mode

    if mode == "tune":

        best_pair = tuner.tune()
        logger.info(
            f"Best configuration: {best_pair[0]}, estimated time cost: {best_pair[1]:.2f}s"
        )
    elif mode == "run_default":
        # run default configuration

        pass
    elif mode == "run_best":
        # run best configuration
        pass
