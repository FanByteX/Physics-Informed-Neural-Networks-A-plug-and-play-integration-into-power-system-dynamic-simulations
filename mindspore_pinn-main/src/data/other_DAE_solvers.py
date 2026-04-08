import torch
import numpy as np

from .data import Data
from utils.losses import MSE

class dae_data_RK(Data):
    """
    RK - DAE - 神经网络数据集
    参数:
        :x_train (numpy Tensor)
        :x_test (numpy Tensor)
        :func: IRK微分代数方程
    """
    def __init__(self, x_train, x_test, args, RK="RK-4", device="cpu", func=None):
        # if func is None:
            # raise ValueError("{} 不是有效的DAE".format(func))
        if x_train is not None:
            self.x_train = x_train
            self.x_test = x_test
        else:
            raise ValueError("训练数据不能为空 {}".format(x_train))

        self.device = device
        if str(device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.h = TensorFloat([args.h])
        # 收集RK数据
        if RK == "RK":
            data = np.load("./data/IRK_weights/RK-3-8-tableau.npz")
            self.nu = 4
        elif RK == "Gauss-Legendre":
            data = np.load("./data/IRK_weights/Gauss-Legendre-tableau.npz")
            self.nu = 3
        
        self.IRK_weights = TensorFloat(data["IRK_weights"])    # \in [nu + 1, nu]
        self.IRK_times = data["IRK_times"]    # \in [nu, 1]
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

    
class dae_data_other(Data):
    """
    DAE - 神经网络数据集
    参数:
        :x_train (numpy Tensor)
        :x_test (numpy Tensor)
        :func: IRK微分代数方程
    """
    def __init__(self, x_train, x_test, args, device="cpu", func=None):
        # if func is None:
            # raise ValueError("{} 不是有效的DAE".format(func))
        if x_train is not None:
            self.x_train = x_train
            self.x_test = x_test
        else:
            raise ValueError("训练数据不能为空 {}".format(x_train))

        self.device = device
        if str(device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.h = TensorFloat([args.h])
        self.pinn = func

    def loss_fn(self, inputs, model):
        """ 计算损失 """
        losses = []
        f, g = self.pinn(model, inputs, self.h)

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