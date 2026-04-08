"""
模型模块
包含：RES-PINN 模型、物理方程、网络结构
"""
# 核心模型 - RES-PINN
from .three_bus_pn import ResPINN, three_bus_PN  # three_bus_PN 是 RES-PINN 的别名
from .power_net import power_net_dae, scipy_integrate, power_net_dae_scipy

# 网络结构
from .activations import sin_act, linear_act, mish, gelu_fast, gelu_new, get_activation
from .fnn import fnn
from .attention import attention
from .conv1d import Conv1D, dense_Conv1D

__all__ = [
    # 核心模型
    'ResPINN', 'three_bus_PN',  # ResPINN 是主名，three_bus_PN 是别名
    'power_net_dae', 'scipy_integrate', 'power_net_dae_scipy',
    # 网络
    'sin_act', 'linear_act', 'mish', 'gelu_fast', 'gelu_new', 'get_activation',
    'fnn', 'attention', 'Conv1D', 'dense_Conv1D'
]
