"""
前馈神经网络模块 (Feedforward Neural Network)
支持 BatchNorm/LayerNorm 和 Dropout
"""
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

from .activations import get_activation


class fnn(nn.Cell):
    """前馈神经网络"""
    
    def __init__(self, layer_size, activation, kernel_initializer, 
                 dropout_rate=0.0, batch_normalization=None, layer_normalization=None, 
                 input_transform=None, output_transform=None, use_bias=True, print_net=False):
        super(fnn, self).__init__()
        self.layer_size = layer_size
        self.activation = get_activation(activation)
        self.initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_bias = use_bias

        # 构建神经网络
        if self.batch_normalization and self.layer_normalization:
            raise ValueError("不能同时应用批归一化和层归一化")
        self.net = nn.CellList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("神经网络未构建")
        
        if print_net:
            print("神经网络已构建...\n")
            print(self.net)

    def construct(self, input):
        y = input
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights_mindspore(self, m, initializer):
        """MindSpore权重初始化"""
        if isinstance(m, nn.Dense):
            if initializer == "Glorot normal":
                fan_in = m.in_channels
                fan_out = m.out_channels
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                m.weight.set_data(Tensor(np.random.uniform(-limit, limit, m.weight.shape), mindspore.float32))
            elif initializer == "Glorot uniform":
                fan_in = m.in_channels
                fan_out = m.out_channels
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                m.weight.set_data(Tensor(np.random.uniform(-limit, limit, m.weight.shape), mindspore.float32))
            if m.has_bias:
                m.bias.set_data(Tensor(np.zeros(m.bias.shape), mindspore.float32))
        elif isinstance(m, nn.LayerNorm):
            m.gamma.set_data(Tensor(np.ones(m.normalized_shape), mindspore.float32))
            if m.beta is not None:
                m.beta.set_data(Tensor(np.zeros(m.normalized_shape), mindspore.float32))

    def build_standard(self):
        """标准网络构建: 全连接 - 激活函数"""
        self.net.append(nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1], has_bias=self.use_bias))

    def build_before(self):
        """构建: 全连接 - 批归一化或层归一化 - 激活函数"""
        self.net.append(nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1], has_bias=self.use_bias))

    def build_after(self):
        """构建: 全连接 - 激活函数 - 批归一化或层归一化"""
        self.net.append(nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1], has_bias=self.use_bias))
