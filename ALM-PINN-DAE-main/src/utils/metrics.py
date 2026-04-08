"""
评估指标模块
l2_relative_error: 计算 L2 相对误差，用于评估预测精度
"""
import numpy as np


def l2_relative_error(y_true, y_pred):
    """计算 L2 相对误差"""
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
