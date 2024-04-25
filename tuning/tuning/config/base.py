import random
from pathlib import Path
from typing import Dict, List, Type, TypeVar, Union

from ..common.const import *

# 参数空间：有一个名字 name ，有一堆可选的值
# 参数选定：参数空间的一个点

U = Union[str, int, float]


# 通用的 Param 类，我们目前只支持离散的变量
class ParamClass(object):
    T = TypeVar("T", str, int, float)

    values: List[T]
    default_value: T

    def __init__(self, name: str, value: T):
        self.name = name
        self.value = value
        assert (
            value in self.values
        ), f"Invalid value {value} for {self.__class__.__name__}, valid values are {self.values}"

    @classmethod
    def get_values(cls) -> List[T]:
        return cls.values

    @classmethod
    def get_default_value(cls) -> T:
        return cls.default_value

    def get_name(self) -> str:
        return self.name

    def get_value(self) -> T:
        return self.value

    def __str__(self) -> str:
        return f"{self.name} = {self.value}"

    @classmethod
    def get_decl_str(cls) -> str:
        raise NotImplementedError


# 类型是字符串
class EnumParam(ParamClass):
    T = str

    @classmethod
    def get_decl_str(cls) -> str:
        ans = []
        ans.append(f"enum {cls.__name__}" + "{")
        ans += [f"    {value}," for value in cls.values]
        ans.append("};")
        return "\n".join(ans)


# 类型是整数
class IntegerParam(ParamClass):
    T = int

    @classmethod
    def get_decl_str(cls) -> str:
        return f"using {cls.__name__} = int;"


# 类型是浮点数
class FloatParam(ParamClass):
    T = float

    @classmethod
    def get_decl_str(cls) -> str:
        return f"using {cls.__name__} = float;"


# Config 类，代表一个 C++ 的 Struct
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
    def get_flat_params(cls) -> Dict[str, ParamClass]:
        ret = {}
        for key, value in cls.params.items():
            if issubclass(value, ParamClass):
                # 如果是直接的 ParamClass，我们就直接加进去
                assert (
                    key not in ret
                ), f"Duplicate key {key} in {cls.__name__}'s flat params"
                ret[key] = value
            else:
                # 如果是 ConfigClass，我们就把它的 flat params 获取到，然后加一个前缀加进去
                sub_class_flat_params = value.get_flat_params()
                for sub_key, sub_value in sub_class_flat_params.items():
                    assert (
                        key + "." + sub_key not in ret
                    ), f"Duplicate key {key + '.' + sub_key} in {cls.__name__}'s flat params"
                    ret[key + "." + sub_key] = sub_value
        return ret

    @classmethod
    def get_params(cls) -> Dict[str, Type[Union[ParamClass, "ConfigClass"]]]:
        return cls.params

    @classmethod
    def get_default_value(cls) -> Dict[str, U]:
        return {key: value.get_default_value() for key, value in cls.params.items()}

    def get_name(self) -> str:
        return self.name

    def get_value(self) -> Dict[str, U]:
        return {key: getattr(self, key).get_value() for key in self.params.keys()}

    def export(self, path: Path = GENERATED_CONFIG_PATH) -> None:
        with open(path, "w") as f:
            f.write(GENERATED_CONFIG_TEMPLATE.format(self.__str__()))

    def __str__(self) -> str:
        ans = f"{self.name} = " + "{"
        ans += ", ".join([f".{getattr(self, key)}" for key in self.params.keys()])
        ans += "}"
        return ans

    # hash 函数，用于判断两个 Config 是否相等
    def __hash__(self) -> int:
        return hash(self.__str__())

    @classmethod
    def get_decl_str(cls) -> str:
        ans = []
        ans.append(f"struct {cls.__name__}" + "{")
        for key, value in cls.params.items():
            ans.append(f"    {value.__name__} {key};")
        ans.append("};")
        return "\n".join(ans)

    @classmethod
    def get_all_decl_str(cls) -> str:
        ans = []
        for key, value in cls.params.items():
            if issubclass(value, ConfigClass):
                ans.append(value.get_all_decl_str())
            elif issubclass(value, ParamClass):
                ans.append(value.get_decl_str())
            else:
                assert False, f"Invalid type {value}, must be ConfigClass or ParamClass"
        ans.append(cls.get_decl_str())
        return "\n".join(ans)


# 某个参数类的参数空间
# 在这里我们只会处理 Dict，而不是 ConfigClass
class ConfigSpace(object):
    def __init__(self, config: Type[ConfigClass]):
        self.config_class = config

    def config_space(self) -> List[Dict[str, Union[str, int]]]:
        flat_params = self.config_class.get_flat_params()
        # 展开所有的参数空间，可能会巨大，我们先判断一下到底有多大，如果太大了就不展开了
        total_size = 1
        for key, value in flat_params.items():
            total_size *= len(value.values)
        if total_size > MAX_CONFIG_SPACE_SIZE:
            assert False, f"Config Space is too large: {total_size}"
        # 使用乘法展开参数空间
        ret = [{}]
        for key, value in flat_params.items():
            new_ret = []
            for item in value.values:
                for old_item in ret:
                    new_item = old_item.copy()
                    new_item[key] = item
                    new_ret.append(new_item)
            ret = new_ret
        return ret

    # 从参数空间中随机取一个点出来
    def random_configuration(self) -> Dict[str, Union[str, int]]:
        flat_params = self.config_class.get_flat_params()
        return {key: random.choice(value.values) for key, value in flat_params.items()}

    # 从输入的 config 开始，随机走一步
    def random_walk(
        self, config: Dict[str, Union[str, int]]
    ) -> Dict[str, Union[str, int]]:
        # 随机选择一个 key
        random_key = random.choice(list(config.keys()))
        # 随机选择一个 value
        flat_params = self.config_class.get_flat_params()
        random_value = random.choice(flat_params[random_key].values)
        # 生成新的 config
        new_config = config.copy()
        new_config[random_key] = random_value
        return new_config
