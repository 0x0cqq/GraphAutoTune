import logging
from typing import List, Tuple

from ..common.const import *
from ..config.base import ConfigClass
from .manipulator import Manipulator

logger = logging.getLogger("tuner")


class Tuner:
    def __init__(self, manipulator: Manipulator) -> None:
        """
        :param: job cuda program
        :param: options data path and pattern size
        """
        self.manipulator = manipulator
        self.best_time = FLOAT_INF
        self.max_round = 20
        self.batch_size = 5
        self.warmup_examples = 20

    def _warmup(self, warmup_examples: int) -> None:
        random_samples = self.manipulator.random_batch(warmup_examples)
        valid_trials, results = self.manipulator.run_batch(random_samples)
        self.manipulator.update_model(valid_trials, results)

    def tune(self) -> Tuple[ConfigClass, float]:
        """
        端到端的 AutoTune 过程
        """
        logger.info(f"Start tuning... max_round: {self.max_round}")

        # 需要先用随机 sample 热身，不然 predict 跑不通
        self._warmup(self.warmup_examples)

        for round_id in range(self.max_round):
            logger.info(f"Tuning round {round_id+1}/{self.max_round}")

            trials = self.manipulator.next_batch(self.batch_size)
            valid_trials, results = self.manipulator.run_batch(trials)

            # send to manipulator
            self.manipulator.update_model(valid_trials, results)

            best_config = self.manipulator.get_best_config()
            logger.info(
                f"Round {round_id + 1} / {self.max_round}, best performance: {best_config[1]:.4f}s",
            )

        best_config = self.manipulator.get_best_config()
        logger.info(f"End tuning... best performance: {best_config[1]:.4f}s")
        return best_config
