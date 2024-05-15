import json
import logging
import os
import time
from typing import List, Type

import numpy as np
import xgboost as xgb

from ..common.const import *
from ..config.base import ConfigClass

logger = logging.getLogger("modeling")


class Modeling:
    """
    Autotune 中的 Cost Model。

    功能：

    1. 维护 History Data，以及和 History Data 对应的 XGBoost 模型
    2. 提供 Config -> Predicted 运行时间的接口 (Predict)
    3. 提供维护 History Data 的接口 (Update, Load, Dump)

    """

    xgb_params = {
        "max_depth": 10,
        "gamma": 0.001,
        "min_child_weight": 0,
        "eta": 0.2,
        "verbosity": 0,
        "disable_default_eval_metric": 1,
    }

    def __init__(self, config_class: Type[ConfigClass]) -> None:
        self.xs: List[ConfigClass] = []
        self.ys: List[float] = []
        self.bst: xgb.Booster | None = None
        self.config_class = config_class

        self.__load()
        self.__fit()

    def __load(self, path=RECORD_PATH) -> None:
        """从文件中加载历史数据

        Args:
            path (Path, optional): 历史数据的路径. Defaults to RECORD_PATH.
        """
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            logger.info("Empty record file, skipped.")
            return

        self.xs = []
        self.ys = []

        with open(path, "r") as f:
            records = json.load(f)
            for res in records:
                assert len(res) == 2, f"Invalid record: {res}"
                self.xs.append(self.config_class.parse(res[0]))
                self.ys.append(float(res[1]))

        logger.info(f"Loaded {len(self.ys)} records from file.")

    def __dump(self, path=RECORD_PATH) -> None:
        """将历史数据保存到文件

        Args:
            path (Path, optional): 将要存储的路径. Defaults to RECORD_PATH.
        """
        with open(path, "w") as f:
            json.dump([[x.get_value_dict(), y] for x, y in zip(self.xs, self.ys)], f)

        logger.info(f"Dumped {len(self.ys)} records to file.")

    def __fit(self):
        """利用存储的 self.xs 和 self.ys 训练 XGBoost 模型"""
        object_count = len(self.xs)

        if object_count == 0:
            logger.warning("Training with empty data. skipped.")
            return
        else:
            start = time.time()

            index = np.random.permutation(object_count)
            dx = [item.get_list() for item in self.xs]
            dy = self.ys

            dtrain = xgb.DMatrix(np.asanyarray(dx)[index], np.array(dy)[index])
            self.bst = xgb.train(self.xgb_params, dtrain, num_boost_round=10000)

            end = time.time()

            logger.info(
                f"XGBoost train: {end - start: .2f} s \t object count: {object_count}"
            )

    def predict_list(self, data_x: List[ConfigClass]) -> List[float]:
        """预测一组数据的运行时间

        Args:
            data_x (List[ConfigClass]): 参数空间中的点列表

        Returns:
            List[float]: 预测的运行时间列表
        """
        matrix_x = [item.get_list() for item in data_x]
        dtest = xgb.DMatrix(np.asanyarray(matrix_x))
        return self.bst.predict(dtest)

    def predict(self, data_x: ConfigClass) -> float:
        """预测单个数据的运行时间

        Args:
            data_x (ConfigClass): 参数空间中的点

        Returns:
            float: 预测的运行时间
        """
        return self.predict_list([data_x])[0]

    def update_list(self, new_xs: List[ConfigClass], new_ys: List[float]):
        """更新历史数据

        Args:
            new_xs (List[ConfigClass]): 新的参数空间中的点列表
            new_ys (List[float]): 对应的运行时间列表
        """
        self.xs.extend(new_xs)
        self.ys.extend(new_ys)
        self.__dump()
        self.__fit()
