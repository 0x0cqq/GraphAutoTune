import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "data", type=str, help="The data file path, relative to the project ROOT folder"
)
parser.add_argument(
    "pattern", type=str, help="The pattern string to search in the data file"
)
parser.add_argument(
    "-d",
    "--debug",
    default=False,
    action="store_true",
    help="Enable output of CMake and Make",
)
parser.add_argument(
    "-r",
    "--run_mode",
    type=str,
    default="tune",
    choices=["tune", "run_default", "run_best"],
    help="The run mode of the tuning program",
)
