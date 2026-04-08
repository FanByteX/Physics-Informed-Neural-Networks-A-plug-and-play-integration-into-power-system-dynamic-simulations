"""
工具函数模块
timing: 用于测量函数执行时间的装饰器
dotdict: 支持点访问的字典类（可用 obj.key 替代 obj['key']）
list_to_str: 将数值列表格式化为字符串，用于控制台输出
"""
import sys
import time
from functools import wraps
import numpy as np


def timing(f):
    """测量函数执行时间的装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result
    return wrapper


class dotdict(dict):
    """支持点访问的字典类"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def list_to_str(nums, precision=3):
    """将数值列表格式化为字符串"""
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))
