"""
Conv1D 神经网络模块
使用 1D 卷积层的前馈神经网络
"""
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .activations import get_activation


class dense_Conv1D(nn.Cell):
    """自定义 Conv1D 密集层"""
    
    def __init__(self, inputs, outputs, activation=None):
        super().__init__()
        self.n_out = outputs
        w = Tensor(np.random.normal(0, 0.02, (inputs, outputs)), mindspore.float32)
        self.weight = mindspore.Parameter(w)
        self.bias = mindspore.Parameter(Tensor(np.zeros(outputs), mindspore.float32))
        self.activation = activation
    
    def construct(self, input):
        size_out = input.shape[:-1] + (self.n_out,)
        y = ops.addmm(self.bias, input.reshape((-1, input.shape[-1])), self.weight)
        y = y.reshape(size_out)
        if self.activation is not None:
            y = self.activation(y)
        return y


class Conv1D(nn.Cell):
    """使用Conv1D密集层的前馈神经网络"""
    
    def __init__(self, layer_size, activation, dropout_rate=0.0,
                 batch_normalization=None, layer_normalization=None,
                 input_transform=None, output_transform=None):
        super(Conv1D, self).__init__()
        self.layer_size = layer_size
        self.activation = get_activation(activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        
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

    def construct(self, input):
        y = input
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def build_standard(self):
        """标准构建: 全连接 - 激活函数"""
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))

    def build_before(self):
        """构建: 全连接 - 批归一化或层归一化 - 激活函数"""
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1]))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1]))

    def build_after(self):
        """构建: 全连接 - 激活函数 - 批归一化或层归一化"""
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.batch_normalization is not None:
            self.net.append(nn.BatchNorm1d(self.layer_size[1]))
        elif self.layer_normalization is not None:
            self.net.append(nn.LayerNorm((self.layer_size[1],)))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        for i in range(1, len(self.layer_size) - 2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i+1]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i+1],)))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))
