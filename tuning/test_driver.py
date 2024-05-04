from tuning.config.details import Config
from tuning.role.driver import Driver

config = Config()


Driver.run("./bin/pm", [""], config)
