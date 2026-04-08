"""
损失函数模块
MSE: 均方误差，用于 ResPINN 残差计算和 ALM 罚函数
"""
import mindspore.ops as ops
def MSE(y_pred, y_true=None):
    """均方误差损失函数"""
    return ops.mean(y_pred ** 2) if y_true is None else ops.mean((y_pred - y_true) ** 2)
