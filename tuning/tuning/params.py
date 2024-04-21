from typing import Any, Dict, List, Type, Union

# 参数空间：有一个名字 name ，有一堆可选的值
# 参数选定：参数空间的一个点


# Enum 类
class ParamClass(object):
    values: List[Union[str, int]]
    default_value: Union[str, int]

    def __init__(self, name, value):
        self.name = name
        self.value = value
        assert (
            value in self.values
        ), f"Invalid value {value} for {self.__class__.__name__}, valid values are {self.values}"

    @classmethod
    def get_values(cls):
        return cls.values

    @classmethod
    def get_default_value(cls):
        return cls.default_value

    def get_name(self):
        return self.name

    def get_value(self):
        return self.value

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"

    @classmethod
    def get_decl_str(cls) -> str:
        ans = []
        ans.append(f"enum {cls.__name__}" + "{")
        ans += [f"    {value}," for value in cls.values]
        ans.append("};")
        return "\n".join(ans)


class ConfigClass(object):
    # key 是变量名，value 是组成 Config 的 ParamClass 或者 ConfigClass 的类本身
    params: Dict[str, Type[Union[ParamClass, "ConfigClass"]]]

    def __init__(self, name, **kwargs):
        self.name = name
        # 输入的 key 必须在 params 中，否则报错
        assert all(
            key in self.params for key in kwargs
        ), f"Invalid keys {kwargs.keys()} for {self.__class__.__name__}, valid keys are {self.params.keys()}"
        for key, value in self.params.items():
            if issubclass(value, ParamClass):
                if key in kwargs:
                    setattr(self, key, value(key, kwargs[key]))
                else:
                    setattr(self, key, value(key, value.get_default_value()))
            else:
                if key in kwargs:
                    setattr(self, key, value(key, **kwargs[key]))
                else:
                    setattr(self, key, value(key))

    @classmethod
    def get_params(cls):
        return cls.params

    @classmethod
    def get_default_value(cls):
        return {key: value.get_default_value() for key, value in cls.params.items()}

    def get_name(self):
        return self.name

    def get_value(self):
        return {key: getattr(self, key).get_value() for key in self.params.keys()}

    def __str__(self) -> str:
        ans = f"{self.name} = " + "{"
        ans += ", ".join([f".{getattr(self, key)}" for key in self.params.keys()])
        ans += "}"
        return ans

    @classmethod
    def get_decl_str(cls) -> str:
        ans = []
        ans.append(f"struct {cls.__name__}" + "{")
        for key, value in cls.params.items():
            ans.append(f"    {value.__name__} {key};")
        ans.append("};")
        return "\n".join(ans)


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


# 用法

config = Config("config")

print(config)