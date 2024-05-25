import random

from ..common.const import *
from .base import ConfigClass, EnumParam, IntegerParam, ParamClass

# 这里是和 C++ 的实现一一对应的

INT_MAX = 2**31 - 1


class SetSearchType(IntegerParam):
    values = [0, 2, 4, 8, 16, 32, 64, INT_MAX]
    default_value = [0]


class SetIntersectionType(EnumParam):
    values = ["Parallel"]
    default_value = "Parallel"


class VertexStoreType(EnumParam):
    values = ["Array"]
    default_value = "Array"


class VertexSetConfig(ConfigClass):
    params = {
        "set_search_type": SetSearchType,
        "set_intersection_type": SetIntersectionType,
        "vertex_store_type": VertexStoreType,
    }


class GraphBackendType(EnumParam):
    values = ["InMemory"]
    default_value = "InMemory"


class InfraConfig(ConfigClass):
    params = {
        "graph_backend_type": GraphBackendType,
    }


class NumBlocks(IntegerParam):
    values = [128, 256, 512, 1024]
    default_value = 512


class ThreadsPerBlock(IntegerParam):
    values = [64, 128, 256]
    default_value = 128


class ThreadsPerWarp(IntegerParam):
    values = [1, 4, 16, 32]
    default_value = 32


class MaxRegs(IntegerParam):
    values = [32, 48, 64, 80, 96]
    default_value = 64


class LaunchConfig(ConfigClass):
    params = {
        "num_blocks": NumBlocks,
        "threads_per_block": ThreadsPerBlock,
        "threads_per_warp": ThreadsPerWarp,
        "max_regs": MaxRegs,
    }


class EngineConfig(ConfigClass):
    params = {
        "launch_config": LaunchConfig,
    }


class Config(ConfigClass):
    params = {
        "vertex_set_config": VertexSetConfig,
        "infra_config": InfraConfig,
        "engine_config": EngineConfig,
    }
