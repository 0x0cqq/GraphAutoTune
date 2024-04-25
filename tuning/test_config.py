from tuning.config.base import ConfigSpace
from tuning.config.details import Config

# 用法

config = Config("default_config")

print(config)
print(Config.get_all_decl_str())


config_space = ConfigSpace(Config)

print(config_space.random_configuration())
