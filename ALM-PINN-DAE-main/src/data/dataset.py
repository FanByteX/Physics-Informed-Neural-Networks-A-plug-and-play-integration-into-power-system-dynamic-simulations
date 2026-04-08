"""
数据处理模块
包含数据加载器和变换
"""
import numpy as np
import mindspore
from mindspore import Tensor

from ..training.irk import get_irk_weights_times
from ..models.dae import power_net_dae
class Data(object):
    """基础数据类，定义数据加载接口"""
    def __init__(self): pass
    def loss_fn(self, targets, outputs, model): raise NotImplementedError
    def train_next_batch(self, batch_size=None): raise NotImplementedError
    def test(self): raise NotImplementedError
class dae_data(Data):
    """
    RK - DAE - 神经网络数据集
    参数:
        :x_train (numpy Tensor)
        :x_test (numpy Tensor)
        :func: IRK微分代数方程
    """
    def __init__(self, x_train, x_test, args, device="cpu", func=None):
        if x_train is not None:
            self.x_train = x_train
            self.x_test = x_test
        else:
            raise ValueError("训练数据不能为空 {}".format(x_train))

        self.nu = args.num_IRK_stages
        self.device = device
        # 收集RK数据 - 使用智能路径查找
        IRK_w_np, IRK_times_np = get_irk_weights_times(self.nu)
        # 使用MindSpore Tensor创建张量
        self.h = Tensor([args.h], mindspore.float32)
        self.IRK_weights = Tensor(IRK_w_np, mindspore.float32)    # \in [nu + 1, nu]
        self.IRK_times = IRK_times_np              # \in [nu, 1]
        self.pinn = func

    def loss_fn(self, inputs, model):
        """ 计算损失 """
        losses = []
        f, g = self.pinn(model, inputs, self.h, self.IRK_weights)

        # 动力方程的损失
        losses_dyn = [MSE(fi) for fi in f]
        losses.append(sum(losses_dyn))
        # 代数/扰动/刚性方程的损失
        losses_alg = [MSE(gi) for gi in g]
        losses.append(sum(losses_alg))
        return losses

    def train_next_batch(self, batch_size=None):
        y_train = np.zeros((self.x_train.shape[0], 1))        # 数据加载器所需，未使用
        if (batch_size is None) or (batch_size > self.x_train.shape[0]):
            return self.x_train, y_train, self.x_train.shape[0]
        else:
            return self.x_train, y_train, batch_size

    def test(self):
        y_test = np.zeros((self.x_test.shape[0], 1))        # 数据加载器所需，未使用
        return self.x_train, y_test


def MSE(y_pred, y_true=None):
    """均方误差损失"""
    if y_true is None:
        return mindspore.ops.mean(y_pred ** 2)
    else:
        return mindspore.ops.mean((y_pred - y_true) ** 2)
