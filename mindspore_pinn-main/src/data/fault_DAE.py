"""
故障场景下的 DAE 数据类
扩展自原始 DAE.py，支持在 2s 时刻引入故障（电纳 b 从 10 变为 20）
"""

import numpy as np
import torch

from .data import Data
from utils.losses import MSE

class fault_dae_data(Data):
    """
    带故障的 RK - DAE - 神经网络数据集
    在 t >= fault_time 时，电纳 b 从 10 变为 20
    
    参数:
        :x_train (numpy Tensor)
        :x_test (numpy Tensor)
        :args: 参数配置
        :device: 计算设备
        :func: IRK微分代数方程（带故障参数）
        :fault_time: 故障发生时间（秒）
        :b_fault: 故障时的电纳值（默认20）
    """
    def __init__(self, x_train, x_test, args, device="cpu", func=None, fault_time=2.0, b_fault=20.0):
        if x_train is not None:
            self.x_train = x_train
            self.x_test = x_test
        else:
            raise ValueError("训练数据不能为空 {}".format(x_train))

        self.nu = args.num_IRK_stages
        self.device = device
        self.fault_time = fault_time  # 故障发生时间
        self.b_fault = b_fault        # 故障时的电纳值
        
        if str(device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.h = TensorFloat([args.h])
        
        # 收集RK数据
        tmp = np.float32(np.loadtxt('./data/IRK_weights/Butcher_IRK%d.txt' % (self.nu), ndmin=2))
        IRK_weights = np.reshape(tmp[0:self.nu**2+self.nu],(self.nu+1,self.nu))
        self.IRK_weights = TensorFloat(IRK_weights)    # \in [nu + 1, nu]
        self.IRK_times = tmp[self.nu**2 + self.nu:]    # \in [nu, 1]
        self.pinn = func

    def loss_fn(self, inputs, model):
        """ 
        计算损失（包含故障条件）
        在训练时，需要同时学习正常运行和故障条件下的系统行为
        """
        losses = []
        
        # 使用故障参数计算损失
        f, g = self.pinn(model, inputs, self.h, self.IRK_weights, 
                        fault_time=self.fault_time, b_fault=self.b_fault)

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
