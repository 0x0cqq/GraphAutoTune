import json
import logging

from tuning.common.argparser import parser
from tuning.common.const import *
from tuning.config.details import Config
from tuning.role.manipulator import Manipulator
from tuning.role.modeling import Modeling
from tuning.role.tuner import Tuner

logger = logging.getLogger("main")


JOB_NAME = "./bin/pm"


if __name__ == "__main__":

    args = parser.parse_args()

    # create context
    model = Modeling()
    manip = Manipulator(model, Config)
    tuner = Tuner(
        JOB_NAME,
        [args.data, args.pattern],
        manip,
    )

    mode: str = args.run_mode
    if mode == "tune":
        best_pair = tuner.tune()
        # best_config = tuner.manipulator.find_maximums(5, 50, 1)
        logger.info(
            f"Best configuration: {best_pair[0]}, estimated time cost: {best_pair[1]:.2f}s"
        )
        with open(CONF_PATH, "w") as f:
            json.dump(best_pair[0], f, indent=4)
        logger.info("Best configuration dumped in ./best_config.json")
    elif mode == "run_default":
        # run default configuration
        pass
    elif mode == "run_best":
        # run best configuration
        pass
