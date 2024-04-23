import random

from ..common.const import *
from .base import ConfigClass, ParamClass


class SetSearchType(ParamClass):
    values = ["Binary", "Serial"]
    default_value = "Binary"


class SetIntersectionType(ParamClass):
    values = ["Parallel", "Sequential"]
    default_value = "Parallel"


class VertexStoreType(ParamClass):
    values = ["Array", "Bitmap"]
    default_value = "Array"


class VertexSetConfig(ConfigClass):
    params = {
        "set_search_type": SetSearchType,
        "set_intersection_type": SetIntersectionType,
        "vertex_store_type": VertexStoreType,
    }


class GraphBackendType(ParamClass):
    values = ["InMemory"]
    default_value = "InMemory"


class InfraConfig(ConfigClass):
    params = {
        "graph_backend_type": GraphBackendType,
    }


class Config(ConfigClass):
    params = {
        "vertex_set_config": VertexSetConfig,
        "infra_config": InfraConfig,
    }


def random_configuration(self) -> dict:
    """
    return a random configuration to be tested
    """
    ret = {}
    while True:
        for key, item in {}:  # TODO: 需要改成使用这里的 ConfigClass
            ret[key] = random.choice(item)
        if ret not in self.xs:
            break
    return ret
