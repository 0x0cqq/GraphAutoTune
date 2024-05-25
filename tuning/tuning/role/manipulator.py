import heapq
import logging
import time
from typing import List, Tuple

import numpy as np

from ..common.const import *
from ..config.base import ConfigClass, ConfigSpace
from .driver import Driver
from .modeling import Modeling

logger = logging.getLogger("manipulator")

# Manipulator 中，我们使用 ConfigSpace 提供的能力操作参数空间，成为


class Manipulator:
    def __init__(self, model: Modeling, driver: Driver):
        """构造 Manipulator

        Args:
            model (Modeling): 代价模型
            driver (Driver): 运行程序的驱动器
        """
        self.model = model
        self.driver = driver
        self.config_space = ConfigSpace(model.config_class)

    def update_model(self, inputs: List[ConfigClass], results: List[float]) -> None:
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

    def random_batch(self, batch_size: int) -> List[ConfigClass]:
        """随机生成一批参数

        Args:
            batch_size (int): 参数个数

        Returns:
            List[ConfigClass]: 参数列表
        """
        return self.config_space.random_configurations(batch_size)

    def next_batch(
        self, batch_size: int, n_iter: int = 40, log_interval: int = -1
    ) -> List[ConfigClass]:
        """找到下一批最优的参数

        Args:
            batch_size (int): 一次找到的最优参数个数
            n_iter (int): 模拟退火的迭代次数
            log_interval (int): 日志输出间隔

        Returns:
            List[ConfigClass]: 下一批最优的参数
        """

        class Pair:
            """为了 heapify (float, ConfigClass)"""

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

        points = [self.config_space.random_configuration() for _ in range(batch_size)]

        scores = self.model.predict_list(points)

        heap_items = [Pair(scores[i], points[i]) for i in range(batch_size)]
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

            if log_interval != -1 and _ % log_interval == 0:
                logger.info(
                    f"\rFinding maximums... {((_+1) / n_iter):.2f}%, time elapsed: {(end - start):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}",
                )

        logger.info("Finished finding maximums.")

        return [x.second for x in heap_items]

    def run_batch(
        self, trials: List[ConfigClass]
    ) -> Tuple[List[ConfigClass], List[float]]:
        """测试一批参数的实际运行效率

        Args:
            trials (List[ConfigClass]): 参数空间中的点

        Returns:
            Tuple[List[ConfigClass], List[float]]: 有效的参数点和对应的运行效率
        """
        valid_trials = []
        results = []
        for trial in trials:
            time_cost = self.driver.run(trial)
            if time_cost != FLOAT_INF:
                valid_trials.append(trial)
                results.append(time_cost)
        return valid_trials, results

    def get_best_config(self) -> Tuple[ConfigClass, float]:
        """获取历史数据中最好的配置

        Returns:
            Tuple[ConfigClass, float]: 最好的配置和对应的运行时间
        """
        return self.model.get_best_config()
