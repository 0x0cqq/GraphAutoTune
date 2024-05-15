import hashlib
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

    def __init__(self, value: T):
        self.value = value
        assert (
            value in self.values
        ), f"Invalid value '{value}' for {self.__class__.__name__}, valid values are {self.values}"

    @classmethod
    def get_values(cls) -> List[T]:
        return cls.values

    @classmethod
    def random_choice(cls) -> "ParamClass":
        return cls(random.choice(cls.values))

    @classmethod
    def get_default_value(cls) -> "ParamClass":
        return cls(cls.default_value)

    @classmethod
    def get_all_values(cls) -> List["ParamClass"]:
        return [cls(value) for value in cls.values]

    @classmethod
    def get_decl_str(cls) -> str:
        raise NotImplementedError

    # copy method
    def copy(self):
        return self.__class__(self.value)

    def get_value(self) -> T:
        return self.value

    def get_index(self) -> int:
        return self.values.index(self.value)

    def __str__(self) -> str:
        return f"{self.value}"


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

    @classmethod
    def get_params(cls) -> Dict[str, Type[Union[ParamClass, "ConfigClass"]]]:
        return cls.params

    @classmethod
    def random_choice(cls) -> "ConfigClass":
        return cls(**{key: value.random_choice() for key, value in cls.params.items()})

    @classmethod
    def get_default_value(cls) -> "ConfigClass":
        return cls(
            **{key: value.get_default_value() for key, value in cls.params.items()}
        )

    @classmethod
    def get_all_values(cls) -> List["ConfigClass"]:
        # 做一个笛卡尔积
        ret = [cls()]
        for key, value in cls.params.items():
            new_ret = []
            for item in ret:
                for val in value.get_all_values():
                    new_item = item.copy()
                    setattr(new_item, key, val)
                    new_ret.append(new_item)
            ret = new_ret
        return ret

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

    def __init__(self, **kwargs):
        # 输入的 key 必须在 params 中，否则报错
        assert all(
            key in self.params for key in kwargs
        ), f"Invalid keys {kwargs.keys()} for {self.__class__.__name__}, valid keys are {self.params.keys()}"
        for key, value in self.params.items():
            # ParamClass
            assert (
                "." not in key
            ), f"Invalid key {key} in {self.__class__.__name__}, key should not contain '.'"
            if issubclass(value, ParamClass):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                else:
                    setattr(self, key, value.get_default_value())
            # ConfigClass
            else:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                else:
                    setattr(self, key, value.get_default_value())

    @classmethod
    def get_flat_params(cls) -> Dict[str, Type[ParamClass]]:
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

    # 这里获得的
    def get_flat_dict(self) -> Dict[str, ParamClass]:
        ret = {}
        for key, value in self.params.items():
            if issubclass(value, ParamClass):
                ret[key] = getattr(self, key)
            else:
                temp = getattr(self, key).get_flat_dict()
                for sub_key, sub_value in temp.items():
                    assert (
                        key + "." + sub_key not in ret
                    ), f"Duplicate key {key + '.' + sub_key} in {self.name}'s flat dict"
                    ret[key + "." + sub_key] = sub_value
        return ret

    def get_list(self) -> List[int]:
        flat_dict = self.get_flat_dict()
        # ordered by key
        flat_dict = sorted(flat_dict.items(), key=lambda x: x[0])
        return [item.get_index() for _, item in flat_dict]

    def export(self, path: Path = GENERATED_CONFIG_PATH) -> None:
        with open(path, "w") as f:
            f.write(GENERATED_CONFIG_TEMPLATE.format(self.__str__()))

    def copy(self):
        ret = {}
        for key in self.params.keys():
            ret[key] = getattr(self, key).copy()
        return self.__class__(**ret)

    def __str__(self) -> str:
        # sorted by key
        ans = "{"
        ans += ", ".join(
            [f".{key} = {getattr(self, key)}" for key in sorted(self.params.keys())]
        )
        ans += "}"
        return ans

    @classmethod
    def parse(cls, flat_dict: Dict[str, Union[str, int, float]]) -> "ConfigClass":
        ret = {}
        for key, value in cls.params.items():
            if issubclass(value, ParamClass):
                ret[key] = value(flat_dict[key])
            else:
                # 否则获得所有跟这个 ConfigClass 有关的参数
                sub_dict = {}
                for sub_key in flat_dict.keys():
                    if sub_key.startswith(key + "."):
                        sub_dict[sub_key[len(key) + 1 :]] = flat_dict[sub_key]
                ret[key] = value.parse(sub_dict)
        return cls(**ret)

    # hash 函数，用于判断两个 Config 是否相等
    def fingerprint(self) -> str:
        return hashlib.md5(self.__str__().encode()).hexdigest()


# 某个参数类的参数空间
# 在这里我们只会处理 Dict，而不是 ConfigClass
class ConfigSpace(object):
    def __init__(self, config: Type[ConfigClass]):
        self.config_class = config
        # filter 掉那些没得选的
        self.flat_params = {
            key: value
            for key, value in config.get_flat_params().items()
            if len(value.get_values()) > 1
        }

    def config_space_size(self) -> int:
        size = 1
        for key, value in self.flat_params.items():
            size *= len(value.get_values())
        return size

    def config_space(self) -> List[ConfigClass]:
        return self.config_class.get_all_values()

    # 从参数空间中随机取一个点出来
    def random_configuration(self) -> ConfigClass:
        return self.config_class.random_choice()

    def random_configurations(self, k: int) -> List[ConfigClass]:
        if k > self.config_space_size():
            raise ValueError(
                f"Cannot sample {k} configs from a space of size {self.config_space_size()}"
            )

        configs: List[ConfigClass] = []

        for _ in range(k):
            config = self.random_configuration()
            while config.fingerprint() in [x.fingerprint() for x in configs]:
                config = self.random_configuration()

            configs.append(config)

        return configs

    # 从输入的 config 开始，随机走一步
    def random_walk(self, config: ConfigClass) -> ConfigClass:
        assert isinstance(
            config, self.config_class
        ), f"Invalid config for {self.config_class.__name__}, which is a {type(config)}"

        while True:
            key = random.choice(list(self.flat_params.keys()))

            if len(self.flat_params[key].get_values()) == 1:
                continue
            else:
                break

        print(f"key = {key}")

        print(f"self.flat_params = {self.flat_params}")

        # deepcopy 一份 config
        new_config = config.copy()

        # 随机选择一个 value
        while True:
            value = self.flat_params[key].random_choice()

            # 逐层分析 key，然后赋值
            level_keys = key.split(".")
            cur = new_config
            for this_key in level_keys[:-1]:
                cur = getattr(cur, this_key)

            last_key = level_keys[-1]

            # 如果选的值和原来的值不一样，就赋值
            if getattr(cur, last_key).get_index() != value.get_index():
                print(f"Change {key} from {getattr(cur, last_key)} to {value}")
                setattr(cur, last_key, value)
                break
            else:
                continue
        return new_config
