import heapq
import logging
import time
from typing import List

import numpy as np

from ..common.const import *
from ..config.details import random_configuration
from ..utils import config_random_walk, dict2list
from .modeling import Modeling

logger = logging.getLogger("manipulator")


class Manipulator:
    def __init__(self, model: Modeling, num_warmup_sample: int = 100):
        self.best_config = ({}, FLOAT_INF)
        self.model = model
        self.num_warmup_sample = num_warmup_sample
        self.batch_size = 10

    def update(self, inputs: List[dict], results: List[float]) -> None:
        """
        Add a test result to the manipulator.
        XGBoost does not support additional training, so re-train a model each time.
        """

        if len(inputs) == 0:
            return

        self.model.update_list(inputs, results)

        possible_vals = []
        logger.info("After update:")
        for tmp in CANSHUKONGJIAN:
            tmp_val = self.model.predict(tmp)
            if tmp_val not in possible_vals:
                possible_vals.append(tmp_val)
        logger.info(f"Possible values: {possible_vals}")

    def find_maximums(self, num: int, n_iter: int, log_interval: int):
        """
        Find the best `num` sets of parameters
        """

        class Pair:
            """
            class for heapifying tuple[float, dict]
            """

            def __init__(self, a: float, b: float) -> None:
                self.first = a
                self.second = b

            # reversed comparison to make max heap
            def __lt__(self, other: "Pair") -> bool:
                return self.first > other.first

            def __gt__(self, other: "Pair") -> bool:
                return self.first < other.first

        start = time.time()
        temp = 0.1

        points = [random_configuration() for _ in range(num)]

        scores = self.model.predict_list(points)

        heap_items = [Pair(scores[i], points[i]) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = [x.second for x in heap_items]

        for _ in range(n_iter):
            new_points = np.empty_like(points)
            for i in range(len(new_points)):
                new_points[i] = config_random_walk(points[i])
            new_scores = self.predict(new_points)

            ac_prob = np.exp((scores - new_scores) / temp)  # accept probability
            ac_index = np.random.random(len(ac_prob)) < ac_prob  # accepted index

            for idx in range(len(ac_prob)):  # update accepted points and scores
                if ac_index[idx]:
                    points[idx] = new_points[idx]
                    scores[idx] = new_scores[idx]

            for score, point in zip(new_scores, new_points):
                if score < heap_items[0].first and point not in in_heap:
                    pop = heapq.heapreplace(heap_items, Pair(score, point))
                    in_heap.remove(pop.second)
                    in_heap.append(point)

            temp *= 0.9

            end = time.time()

            if log_interval and _ % log_interval == 0:
                logger.info(
                    f"\rFinding maximums... {((_+1) / n_iter):.2f}%, time elapsed: {(end - start):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}",
                )

        logger.info("Finished finding maximums.")

        return [x.second for x in heap_items]
