import json
import random
from typing import Dict, List, Union

from .common.const import *


def dict2list(params: Dict[Union[str, int]]) -> List[int]:
    # 如果是字符串，对 val 做 Hash
    # 如果是数字，直接返回
    return [hash(val) if type(val) == str else val for val in params.values()]
