import heapq
import json
import logging
import os
import random
import time
from typing import List

import numpy as np
import xgboost as xgb

from .const import *
from .utils import config_random_walk, dict2list

logger = logging.getLogger("manipulator")


class Manipulator:
    def __init__(self, num_warmup_sample: int = 100):
        self.xs = []
        self.ys = []
        self.best_config = ({}, FLOAT_INF)
        self.trials = []
        self.bst = None
        self.num_warmup_sample = num_warmup_sample
        self.xgb_params = {
            "max_depth": 10,
            "gamma": 0.001,
            "min_child_weight": 0,
            "eta": 0.2,
            "verbosity": 0,
            "disable_default_eval_metric": 1,
        }
        self.batch_size = 10

        with open(RECORD_PATH, "r") as f:
            if os.path.getsize(RECORD_PATH) != 0:
                records = json.load(f)
                for res in records:
                    self.xs.append(res[0])
                    self.ys.append(res[1])
        assert len(self.xs) == len(self.ys)
        logger.info(f"Loaded {len(self.ys)} records from file.")

    def read_params(self, param_path: str) -> dict:
        # read parameters
        with open(param_path) as f:
            param_dict = json.load(f)
            assert type(param_dict) == type(dict()), "Param JSON file error!"

        return param_dict

    def random_configuration(self) -> dict:
        """
        return a random configuration to be tested
        """
        ret = {}
        while True:
            for key, item in PARAM_VAL.items():
                ret[key] = random.choice(item)
            if ret not in self.xs:
                break
        return ret

    def next_batch(self, batch_size) -> List[dict]:
        """
        Return a batch of configurations to be tested.
        """
        ret = []
        trial_idx = 0
        while len(ret) < batch_size:
            while trial_idx < len(self.trials):
                trial = self.trials[trial_idx]
                if trial not in self.xs:
                    break
                trial_idx += 1
            else:
                trial_idx = -1

            chosen_config = (
                self.trials[trial_idx]
                if trial_idx != -1
                else self.random_configuration()
            )
            ret.append(chosen_config)

        return ret

    def update(self, k: int, inputs: List[dict], results: List[float]) -> None:
        """
        Add a test result to the manipulator.
        XGBoost does not support additional traning, so re-train a model each time.
        """
        logger.info(f"input: {inputs}, results: {results}")
        if len(inputs) == 0:
            return

        for params, res in zip(inputs, results):
            if res < self.best_config[1]:
                self.best_config = (params, res)

            self.xs.append(params)
            self.ys.append(res)

        with open(RECORD_PATH, "w") as f:
            json.dump([[x, y] for x, y in zip(self.xs, self.ys)], f)

        if self.bst is None:
            tmp_matrix = np.asanyarray([dict2list(item) for item in self.xs])
            logger.debug(f"tmp_matrix: {tmp_matrix}")
            perm = np.random.permutation(len(self.xs))
            dtrain = xgb.DMatrix(tmp_matrix[perm], np.asanyarray(self.ys)[perm])
            self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=1000)
        else:
            self.fit(self.xs, self.ys)

        possible_vals = []
        logger.info("After update:")
        for ua in PARAM_VAL["USE_ARRAY"]:
            for tpb in PARAM_VAL["THREADS_PER_BLOCK"]:
                for nb in PARAM_VAL["NUM_BLOCKS"]:
                    for ibs in PARAM_VAL["IEP_BY_SM"]:
                        for maxreg in PARAM_VAL["MAXREG"]:
                            tmp = [ua, tpb, nb, ibs, maxreg, 0]
                            tmp_matrix = xgb.DMatrix(np.asanyarray([tmp]))
                            tmp_val = self.bst.predict(tmp_matrix)[0]
                            if tmp_val not in possible_vals:
                                possible_vals.append(tmp_val)
        logger.info(f"Possible values: {possible_vals}")

        self.trials = self.find_maximums(k, 40, 5)

    def fit(self, data_x: list, data_y: list):
        if len(data_x) == 0:
            return

        tic = time.time()
        index = np.random.permutation(len(data_x))
        dx = [dict2list(data) for data in data_x]
        dy = data_y

        dtrain = xgb.DMatrix(np.asanyarray(dx)[index], np.array(dy)[index])
        self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=10000)

        logger.info(f"XGB train: {time.time() - tic: .2f}\tobs: {len(data_x)}")

    def predict(self, data_x: List[dict]):
        if len(self.xs) < self.num_warmup_sample:
            return np.random.uniform(
                0, 1, len(data_x)
            )  # TODO: add a better range of random score
        dtest = xgb.DMatrix(np.asanyarray([dict2list(item) for item in data_x]))
        return self.bst.predict(dtest)

    def find_maximums(self, num: int, n_iter: int, log_interval: int):
        """
        Find the best `num` sets of parameters
        """

        class Pair:
            """
            class for heapifying tuple[float, dict]
            """

            def __init__(self, a, b) -> None:
                self.first = a
                self.second = b

            # reversed comparison to make max heap
            def __lt__(self, other) -> bool:
                return self.first > other.first

            def __gt__(self, other) -> bool:
                return self.first < other.first

        tic = time.time()
        temp = 0.1

        points = [self.random_configuration() for _ in range(num)]

        scores = self.predict(points)

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

            if log_interval and _ % log_interval == 0:
                logger.info(
                    f"\rFinding maximums... {(_ / n_iter):.2f}%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}",
                )

        logger.info(
            f"\rFinding maximums... 100%, time elapsed: {(time.time() - tic):.2f}s, temp: {temp:.2f}, max: {heap_items[0].first}",
        )

        return [x.second for x in heap_items]
