"""
残差物理信息神经网络 (ResPINN) - 三母线电力系统模型

Residual Physics-Informed Neural Network for DAE (Differential-Algebraic Equations)
用于电力系统故障分析的 PINN 模型

核心特性：
- 采用隐式龙格库塔 (IRK) 时间离散方案解决刚度问题
- 残差跳跃连接缓解梯度消失，增强动态/代数变量协同学习
- 支持 fnn/attention/Conv1D 多种网络架构

变量说明：
- 动态变量 (Y): ω1, ω2 (角速度), δ2, δ3 (角度)
- 代数变量 (Z): V3 (电压)

模式：
- stacked: 堆叠式（4个独立网络）
- unstacked: 非堆叠式（单一网络输出4个变量）
"""
import mindspore.nn as nn

from .activations import get_activation
from .fnn import fnn
from .attention import attention
from .conv1d import Conv1D


class ResPINN(nn.Cell):
    """
    残差物理信息神经网络 (Residual Physics-Informed Neural Network)
    用于求解具有“无限刚度”特性的微分代数方程 (DAE)
    通过最小化残差驱动网络学习物理一致的解
    """
    
    def __init__(self, dynamic, algebraic, 
                 dyn_in_transform=None, dyn_out_transform=None, 
                 alg_in_transform=None, alg_out_transform=None, 
                 stacked=True):
        super().__init__()
        self.stacked = stacked
        self.dim = 4
        self.num_IRK_stages = dynamic.num_IRK_stages
        
        # 构建动态变量网络
        if dynamic.type == "fnn":
            if stacked:
                self.Y = nn.CellList([
                    fnn(dynamic.layer_size, dynamic.activation, dynamic.initializer, 
                        dropout_rate=dynamic.dropout_rate, 
                        batch_normalization=dynamic.batch_normalization, 
                        layer_normalization=dynamic.layer_normalization, 
                        input_transform=dyn_in_transform, 
                        output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = fnn(dynamic.layer_size, dynamic.activation, dynamic.initializer, 
                            dropout_rate=dynamic.dropout_rate, 
                            batch_normalization=dynamic.batch_normalization, 
                            layer_normalization=dynamic.layer_normalization, 
                            input_transform=dyn_in_transform, 
                            output_transform=dyn_out_transform)
        elif dynamic.type == "attention":
            if stacked:
                self.Y = nn.CellList([
                    attention(dynamic.layer_size, dynamic.activation, dynamic.initializer, 
                             dropout_rate=dynamic.dropout_rate, 
                             batch_normalization=dynamic.batch_normalization, 
                             layer_normalization=dynamic.layer_normalization, 
                             input_transform=dyn_in_transform, 
                             output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = attention(dynamic.layer_size, dynamic.activation, dynamic.initializer, 
                                  dropout_rate=dynamic.dropout_rate, 
                                  batch_normalization=dynamic.batch_normalization, 
                                  layer_normalization=dynamic.layer_normalization, 
                                  input_transform=dyn_in_transform, 
                                  output_transform=dyn_out_transform)
        elif dynamic.type == "Conv1D":
            if stacked:
                self.Y = nn.CellList([
                    Conv1D(dynamic.layer_size, dynamic.activation, 
                          dropout_rate=dynamic.dropout_rate, 
                          batch_normalization=dynamic.batch_normalization, 
                          layer_normalization=dynamic.layer_normalization, 
                          input_transform=dyn_in_transform, 
                          output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = Conv1D(dynamic.layer_size, dynamic.activation, 
                               dropout_rate=dynamic.dropout_rate, 
                               batch_normalization=dynamic.batch_normalization, 
                               layer_normalization=dynamic.layer_normalization, 
                               input_transform=dyn_in_transform, 
                               output_transform=dyn_out_transform)
        else:
            raise ValueError("{} type not implemented".format(dynamic.type))

        # 构建代数变量网络
        if algebraic.type == "fnn":
            self.Z = fnn(algebraic.layer_size, algebraic.activation, algebraic.initializer, 
                        dropout_rate=algebraic.dropout_rate, 
                        batch_normalization=algebraic.batch_normalization, 
                        layer_normalization=algebraic.layer_normalization, 
                        input_transform=alg_in_transform, 
                        output_transform=alg_out_transform)
        elif algebraic.type == "attention":
            self.Z = attention(algebraic.layer_size, algebraic.activation, algebraic.initializer, 
                              dropout_rate=algebraic.dropout_rate, 
                              batch_normalization=algebraic.batch_normalization, 
                              layer_normalization=algebraic.layer_normalization, 
                              input_transform=alg_in_transform, 
                              output_transform=alg_out_transform)
        elif algebraic.type == "Conv1D":
            self.Z = Conv1D(algebraic.layer_size, algebraic.activation, 
                           dropout_rate=algebraic.dropout_rate, 
                           batch_normalization=algebraic.batch_normalization, 
                           layer_normalization=algebraic.layer_normalization, 
                           input_transform=alg_in_transform, 
                           output_transform=alg_out_transform)
        else:
            raise ValueError("{} type not implemented".format(algebraic.type))

    def construct(self, input):
        dim_out = self.num_IRK_stages + 1
        if self.stacked:
            Y0 = self.Y[0](input)
            Y1 = self.Y[1](input)
            Y2 = self.Y[2](input)
            Y3 = self.Y[3](input)
        else:
            Y = self.Y(input)
            Y0 = Y[..., 0:dim_out]
            Y1 = Y[..., dim_out:2*dim_out]
            Y2 = Y[..., 2*dim_out:3*dim_out]
            Y3 = Y[..., 3*dim_out:4*dim_out]
        Z = self.Z(input)
        return Y0, Y1, Y2, Y3, Z


# 别名：保持向后兼容
three_bus_PN = ResPINN
