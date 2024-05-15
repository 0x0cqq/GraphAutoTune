from tuning.config.base import ConfigSpace
from tuning.config.details import Config

# 用法

config = Config()

print(config)
# print(Config.get_all_decl_str())
print(config.get_list())


config_space = ConfigSpace(Config)

random_config = config_space.random_configuration()

for i in range(10):
    print(random_config)
    print(random_config.get_list())
    random_config = config_space.random_walk(random_config)

# print("------")

# space = config_space.config_space()

# for item in space:
#     print(item)
