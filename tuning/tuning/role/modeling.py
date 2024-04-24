import json
import logging
import os
import time
from typing import List

import numpy as np
import xgboost as xgb

from ..common.const import *
from ..utils import dict2list

logger = logging.getLogger("modeling")


# 调用 XGBoost 模块，维护与 XGBoost 模型相关的参数
# 维护当前运行的所有结果
# 负责提供参数空间的采样
class Modeling:
    xgb_params = {
        "max_depth": 10,
        "gamma": 0.001,
        "min_child_weight": 0,
        "eta": 0.2,
        "verbosity": 0,
        "disable_default_eval_metric": 1,
    }

    def __init__(self) -> None:
        self.xs = []
        self.ys = []
        self.bst: xgb.Booster | None = None

        self.__load()
        self.__fit()

    # 从文件中加载 xs 和 ys
    def __load(self, path=RECORD_PATH) -> None:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            logger.info("Empty record file, skipped.")
            return

        self.xs = []
        self.ys = []

        with open(path, "r") as f:
            records = json.load(f)
            for res in records:
                self.xs.append(res[0])
                self.ys.append(res[1])
        assert len(self.xs) == len(self.ys), "Length of xs and ys should be the same."
        logger.info(f"Loaded {len(self.ys)} records from file.")

    # 将 xs 和 ys 存储到文件中
    def __dump(self, path=RECORD_PATH) -> None:
        with open(path, "w") as f:
            json.dump([[x, y] for x, y in zip(self.xs, self.ys)], f)

    # 使用储存的 xs 和 ys 训练 XGBoost 模型
    def __fit(self):
        object_count = len(self.xs)

        if object_count == 0:
            logger.warning("Training with empty data. skipped.")
            return
        else:
            start = time.time()

            index = np.random.permutation(object_count)
            dx = [dict2list(item) for item in self.xs]
            dy = self.ys

            dtrain = xgb.DMatrix(np.asanyarray(dx)[index], np.array(dy)[index])
            self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=10000)

            end = time.time()

            logger.info(
                f"XGBoost train: {end - start: .2f} s \t objects: {object_count}"
            )

    # 用性能模型预测
    def predict_list(self, data_x: List[dict]):
        dtest = xgb.DMatrix(np.asanyarray([dict2list(item) for item in data_x]))
        return self.bst.predict(dtest)

    def predict(self, data_x: dict):
        return self.predict_list([data_x])[0]

    # 添加一组新的数据
    def update(self, new_x: dict, new_y: float):
        self.xs.append(new_x)
        self.ys.append(new_y)

        self.__dump()

        self.__fit(self.xs, self.ys)

    def update_list(self, new_xs: List[dict], new_ys: List[float]):
        self.xs.extend(new_xs)
        self.ys.extend(new_ys)

        self.__dump()

        self.__fit(self.xs, self.ys)
