import json
import logging

from tuning.argparser import parser
from tuning.const import *
from tuning.manipulator import Manipulator
from tuning.tuner import Tuner

if __name__ == "__main__":

    logger = logging.getLogger("main")

    args = parser.parse_args()

    """
    pattern string: 
        if(buffer[INDEX(i,j,size)] == '1')
            add_edge(i,j);
    """

    for file in [CONF_PATH, PARAM_PATH, RESULT_PATH, RECORD_PATH]:
        if not file.is_file():
            with open(file, "w"):
                pass

    manip = Manipulator()
    tuner = Tuner(
        JOB_NAME,
        f"{DATA_PATH}/{args.data} "
        + " ".join([args.graph_size, args.pattern_string, args.use_iep]),
        manip,
    )
    if args.run_best_config:
        param_dict = manip.read_params(CONF_PATH)
        tuner.compile_and_run(param_dict, True)
    elif args.run_default_config:
        param_dict = {
            "USE_ARRAY": 1,
            "THREADS_PER_BLOCK": 1024,
            "NUM_BLOCKS": 128,
            "IEP_BY_SM": 1,
            "LIMIT_REG": 1,
            "MAXREG": 64,
            "SORT": 0,
            "PARALLEL_INTERSECTION": 1,
            "BINARY_SEARCH": 1,
        }
        tuner.compile_and_run(param_dict, True)
    else:
        best_config = tuner.tune(10, 3, debug_msg=args.debug_msg)
        # best_config = tuner.manipulator.find_maximums(5, 50, 1)
        logger.info(
            f"Best configuration: {best_config[0]}, estimated time cost: {best_config[1]:.2f}s"
        )
        with open(CONF_PATH, "w") as f:
            json.dump(best_config[0], f, indent=4)
        logger.info("Best configuration dumped in ./best_config.json")
