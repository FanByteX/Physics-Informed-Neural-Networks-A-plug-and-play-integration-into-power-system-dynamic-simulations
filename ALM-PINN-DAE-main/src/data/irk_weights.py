"""
IRK (隐式龙格-库塔) 权重加载模块
get_irk_weights_times: 读取 Butcher 表权重文件
"""
import os
import numpy as np


def get_irk_weights_times(num_stages, prefer_local=True):
    """
    返回 IRK Butcher 权重矩阵 [nu+1, nu] 和时间节点向量 [nu]
    稳健查找 Butcher_IRK{nu}.txt：依次在当前工作目录、模块所在目录、其上级(src)、项目根目录中查找。
    """
    fname = f"irk{num_stages}.txt"
    module_dir = os.path.dirname(__file__)
    candidates = [
        # 优先查找 data 目录
        os.path.join(module_dir, fname),
        os.path.join(os.path.abspath(os.path.join(module_dir, "..")), fname),           # src/
        os.path.join(os.getcwd(), fname),
    ]
    local_path = None
    for p in candidates:
        if os.path.exists(p):
            local_path = p
            break
    if local_path is None:
        raise FileNotFoundError(
            f"Missing IRK weights file '{fname}'. Searched paths: {', '.join(candidates)}"
        )
    tmp = np.float32(np.loadtxt(local_path, ndmin=2))
    IRK_weights = np.reshape(tmp[0:num_stages**2 + num_stages], (num_stages + 1, num_stages))
    IRK_times = tmp[num_stages**2 + num_stages:]
    return IRK_weights, IRK_times
