"""
激活函数模块 (MindSpore Cell 版本)
定义各种神经网络激活函数：sin, tanh, swish 等
每个激活函数都继承自 nn.Cell，实现 construct 方法
"""
import math
import mindspore.nn as nn
import mindspore.ops as ops


class sin_act(nn.Cell):
    """正弦激活函数"""
    def __init__(self):
        super(sin_act, self).__init__()
    
    def construct(self, x):
        return ops.sin(x)


class linear_act(nn.Cell):
    """线性激活函数（恒等映射）"""
    def __init__(self):
        super(linear_act, self).__init__()
    
    def construct(self, x):
        return x


class mish(nn.Cell):
    """Mish 激活函数: x * tanh(softplus(x))"""
    def __init__(self):
        super(mish, self).__init__()
    
    def construct(self, x):
        return x * ops.tanh(nn.functional.softplus(x))


class gelu_fast(nn.Cell):
    """快速 GELU 激活函数"""
    def __init__(self):
        super(gelu_fast, self).__init__()
    
    def construct(self, x):
        return 0.5 * x * (1.0 + ops.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class gelu_new(nn.Cell):
    """新版 GELU 激活函数"""
    def __init__(self):
        super(gelu_new, self).__init__()
    
    def construct(self, x):
        return 0.5 * x * (1.0 + ops.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * ops.pow(x, 3.0))))


def get_activation(identifier):
    """
    根据标识符获取激活函数实例
    
    Args:
        identifier: 激活函数名称
        
    Returns:
        对应的激活函数 Cell 实例
    """
    activations = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sin": sin_act(),
        "linear": linear_act(),
        "mish": mish(),
        "gelu-fast": gelu_fast(),
        "gelu-new": gelu_new(),
    }
    if identifier not in activations:
        raise ValueError(f"Unknown activation function: {identifier}")
    return activations[identifier]
