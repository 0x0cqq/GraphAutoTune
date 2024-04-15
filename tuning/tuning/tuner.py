import logging
import os

from .const import *
from .driver import Driver
from .manipulator import Manipulator

logger = logging.getLogger("tuner")


class Tuner:
    def __init__(self, job: str, options: str, manipulator: Manipulator) -> None:
        """
        :param: job cuda program
        :param: options data path and pattern size
        """
        self.job = job
        self.options = options
        self.manipulator = manipulator
        self.best_time = FLOAT_INF

    def tune(self, max_round: int, k: int):
        """
        The final autotune interface
        """
        logger.log(logging.INFO, "Start tuning.")

        if len(self.manipulator.trials) == 0:
            configs = [
                self.manipulator.random_configuration() for _ in range(k)
            ]  # 10 samples for warm-up trial
        else:
            configs = self.manipulator.trials

        for _ in range(max_round):
            logger.info(f"Tuning round {_+1}/{max_round}")
            valid_configs = []
            results = []
            for config in configs:
                Driver.compile(config)

                time_cost = Driver.run(self.job, self.options)

                if time_cost != FLOAT_INF:  # ERROR CONFIG
                    valid_configs.append(config)
                    results.append(time_cost)

            if len(valid_configs) == 0:
                logger.error("Too many resources required. Stopping...")
                break

            self.manipulator.update(k, valid_configs, results)
            configs = self.manipulator.trials
            logger.debug("len(valid_configs) =", len(valid_configs))
            logger.info(
                f"Round {_+1} / {max_round} Best performance: {self.manipulator.best_config[1]:.2f}s",
            )

        return self.manipulator.best_config
