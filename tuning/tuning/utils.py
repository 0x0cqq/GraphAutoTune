import json
import random

from .common.const import *


def dict2list(params: dict) -> list:
    """
    Convert `params` into a list
    """
    return [val for key, val in params.items()]


def config_random_walk(config: dict) -> dict:
    """
    Randomly walks to another configuration.
    """
    ret = {}
    for key, val in config.items():
        ret[key] = val
    key_list = list(PARAM_VAL.keys())
    from_i = random.choice(key_list)
    to_v = random.choice(PARAM_VAL[from_i])
    while to_v == config[from_i]:
        from_i = random.choice(key_list)
        to_v = random.choice(PARAM_VAL[from_i])
    ret[from_i] = to_v
    return ret


def read_params(param_path: str) -> dict:
    # read parameters
    with open(param_path) as f:
        param_dict = json.load(f)
        assert type(param_dict) == type(dict()), "Param JSON file error!"

    return param_dict
