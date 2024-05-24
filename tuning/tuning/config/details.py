import random

from ..common.const import *
from .base import ConfigClass, EnumParam, IntegerParam, ParamClass

# 这里是和 C++ 的实现一一对应的

INT_MAX = 2**31 - 1


class SetSearchType(IntegerParam):
    values = [0, 2, 4, 8, 16, 32, 64, INT_MAX]
    default_value = [0]


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


class EngineConfig(ConfigClass):
    params = {}


class Config(ConfigClass):
    params = {
        "vertex_set_config": VertexSetConfig,
        "infra_config": InfraConfig,
        "engine_config": EngineConfig,
    }
