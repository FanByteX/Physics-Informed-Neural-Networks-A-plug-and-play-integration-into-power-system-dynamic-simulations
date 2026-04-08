import torch
import torch.nn as nn

def MSE(y_pred, y_true=None):
    """
    计算均方误差。
    """
    if y_true is None:
        return torch.mean(y_pred ** 2)
    else:
        return torch.mean((y_pred - y_true) ** 2)


def MSE_torch(y_pred, y_true=None):
    """
    使用torch函数计算均方误差。
    """
    if y_true is None:
        return nn.MSELoss(y_pred, torch.zeros_like(y_pred).to(y_pred.device))
    else:
        return nn.MSELoss(y_pred, y_true)