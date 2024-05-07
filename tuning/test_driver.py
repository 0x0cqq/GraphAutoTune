from tuning.config.details import Config
from tuning.role.driver import Driver

config = Config()


time = Driver.run(
    "./bin/pm", ["data/data_graph_5000.bin", "0100110110010110110010100"], config
)

print(time)
