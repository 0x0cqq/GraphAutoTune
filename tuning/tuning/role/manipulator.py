import heapq
import logging
import time
from typing import List, TypeVar

import numpy as np

from ..common.const import *
from ..config.base import ConfigClass, ConfigSpace
from .modeling import Modeling

logger = logging.getLogger("manipulator")

# Manipulator 中，我们使用 ConfigSpace 提供的能力操作参数空间，成为


class Manipulator:
    def __init__(self, model: Modeling):
        """构造 Manipulator

        Args:
            model (Modeling): 代价模型
        """
        self.model = model
        self.config_space = ConfigSpace(model.config_class)
        self.best_config = (None, FLOAT_INF)
        self.batch_size = 10

    def update(self, inputs: List[ConfigClass], results: List[float]) -> None:
        """添加若干的结果到代价模型中

        Args:
            inputs (List[ConfigClass]): x 点，参数空间中的点
            results (List[float]): y点，运行效率
        """

        assert len(inputs) == len(
            results
        ), f"inputs and results should have the same length, got {len(inputs)} and {len(results)}"
        # inputs 和 ConfigSpace 的类型应该是一致的
        assert all(
            isinstance(x, self.config_space.config_class) for x in inputs
        ), f"inputs: {inputs} are not of type {self.config_space.config_class.__name__}"

        if len(inputs) == 0:
            logger.debug("Provide empty inputs, skip update in manipulator.")
            return

        self.model.update_list(inputs, results)

        possible_vals: List[float] = []
        for tmp in self.config_space.config_space():
            tmp_val = self.model.predict(tmp)
            if tmp_val not in possible_vals:
                possible_vals.append(tmp_val)
        logger.info(f"After update, possible values: {possible_vals}")

    def find_maximums(
        self, num: int, n_iter: int, log_interval: int
    ) -> List[ConfigClass]:
        """
        Find the best `num` sets of parameters
        """

        class Pair:
            """
            class for heapifying tuple[float, dict]
            """

            def __init__(self, a: float, b: ConfigClass) -> None:
                self.first = a
                self.second = b

            # reversed comparison to make max heap
            def __lt__(self, other: "Pair") -> bool:
                return self.first > other.first

            def __gt__(self, other: "Pair") -> bool:
                return self.first < other.first

        start = time.time()
        temp = 0.1

        points = [self.config_space.random_configuration() for _ in range(num)]

        scores = self.model.predict_list(points)

        heap_items = [Pair(scores[i], points[i]) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = [x.second for x in heap_items]

        for _ in range(n_iter):
            new_points = [self.config_space.random_walk(point) for point in points]
            new_scores = self.model.predict_list(new_points)

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
