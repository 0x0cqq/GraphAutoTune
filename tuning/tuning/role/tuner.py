import logging
import os

from ..common.const import *
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
        self.max_round = 10
        self.num_to_discover = 3

    def tune(self):
        """
        端到端的 AutoTune 过程
        """
        logger.info(f"Start tuning... max_round: {self.max_round}")

        for round_id in range(self.max_round):
            logger.info(f"Tuning round {round_id+1}/{self.max_round}")

            trials = self.manipulator.find_maximums(self.num_to_discover, 40, 5)

            valid_trials = []
            results = []
            for trial in trials:
                time_cost = Driver.run(self.job, self.options, trial)

                if time_cost != FLOAT_INF:  # ERROR CONFIG
                    valid_trials.append(trial)
                    results.append(time_cost)

                if len(valid_trials) == 0:
                    logger.error("Too many resources required. Stopping...")
                    break

            logger.debug(f"len(valid_configs) = {len(valid_trials)}")

            # send to manipulator
            self.manipulator.update(valid_trials, results)

            logger.info(
                f"Round {round_id+1} / {self.max_round}, best performance: {self.manipulator.best_config[1]:.2f}s",
            )

        logger.info(
            f"End tuning... best performance: {self.manipulator.best_config[1]:.2f}s"
        )

        return self.manipulator.best_config
