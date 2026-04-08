import time
import sys
import numpy as np

from functools import wraps 

def timing(f):
    """ 
    用于测量方法执行时间的装饰器。
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r 花费 %f 秒\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result
    return wrapper

class dotdict(dict):
    """
    使用点符号访问字典属性。
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def list_to_str(nums, precision=3):
    """
    列表转字符串，用于显示错误和指标。
    """
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))