import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-r", type=bool)
parser.add_argument("data", type=str)
parser.add_argument("graph_size", type=str)
parser.add_argument("pattern_string", type=str)
parser.add_argument("use_iep", type=str, help="<0/1> Use IEP or not")
parser.add_argument(
    "--debug_msg",
    default=False,
    action="store_true",
    help="Enable output of CMake and Make",
)
parser.add_argument(
    "--run_best_config",
    default=False,
    action="store_true",
    help="Run the best configuration only",
)
parser.add_argument(
    "--run_default_config",
    default=False,
    action="store_true",
    help="Run the default configuration only",
)
