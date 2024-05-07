import random

from ..common.const import *
from .base import ConfigClass, EnumParam, IntegerParam, ParamClass

# 这里是和 C++ 的实现一一对应的


class SetSearchType(EnumParam):
    values = ["Binary", "Linear"]
    default_value = "Binary"


class SetIntersectionType(EnumParam):
    values = ["Parallel", "Serial"]
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


class MaxSetSize(IntegerParam):
    values = [5000, 10000]
    default_value = 5000


class NumUnits(IntegerParam):
    values = [5000, 10000, 20000, 50000]
    default_value = 10000


class EngineConfig(ConfigClass):
    params = {
        "max_set_size": MaxSetSize,
        "nums_unit": NumUnits,
    }


class Config(ConfigClass):
    params = {
        "vertex_set_config": VertexSetConfig,
        "infra_config": InfraConfig,
        "engine_config": EngineConfig,
    }
