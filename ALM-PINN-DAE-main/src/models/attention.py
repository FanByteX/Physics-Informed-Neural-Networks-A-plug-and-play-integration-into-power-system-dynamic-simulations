"""
注意力机制神经网络模块
带有门控注意力机制的前馈神经网络
"""
import mindspore
import mindspore.nn as nn

from .activations import get_activation


class attention(nn.Cell):
    """具有注意力机制架构的前馈神经网络"""
    
    def __init__(self, layer_size, activation, kernel_initializer,
                 dropout_rate=0.0, batch_normalization=None, layer_normalization=None,
                 input_transform=None, output_transform=None, use_bias=True, print_net=False):
        super(attention, self).__init__()
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
        elif self.batch_normalization == "before":
            self.build_beforeBN()
        elif self.layer_normalization == "before":
            self.build_beforeLN()
        elif self.batch_normalization == "after":
            self.build_afterBN()
        elif self.layer_normalization == "after":
            self.build_afterLN()
        else:
            raise ValueError("神经网络未构建")
        
        # 初始化参数
        self.net.apply(self._init_weights)
        self.U.apply(self._init_weights)
        self.V.apply(self._init_weights)
        
        if print_net:
            print(self.net)
            print(self.U)
            print(self.V)

    def construct(self, input):
        y = input
        if self.input_transform is not None:
            y = self.input_transform(y)
        u = self.U(y)
        v = self.V(y)
        for i in range(len(self.net)-1):
            y = self.net[i](y)
            y = (1 - y) * u + y * v
        y = self.net[-1](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights(self, cell):
        """MindSpore中的参数初始化"""
        if isinstance(cell, nn.Dense):
            if self.initializer == "Glorot normal":
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.XavierNormal(), cell.weight.shape, cell.weight.dtype))
            elif self.initializer == "Glorot uniform":
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
            else:
                raise ValueError("初始化器 {} 未实现".format(self.initializer))
            if cell.bias is not None:
                cell.bias.set_data(mindspore.common.initializer.initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            if cell.beta is not None:
                cell.beta.set_data(mindspore.common.initializer.initializer('zeros', cell.beta.shape, cell.beta.dtype))
            if cell.gamma is not None:
                cell.gamma.set_data(mindspore.common.initializer.initializer('ones', cell.gamma.shape, cell.gamma.dtype))

    def build_standard(self):
        """标准构建：无归一化"""
        if self.dropout_rate > 0:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
        else:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation
            )
        for k in range(len(self.layer_size)-2):
            if self.dropout_rate > 0:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation,
                        nn.Dropout(keep_prob=1.0 - self.dropout_rate)
                    )
                )
            else:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation
                    )
                )
        # 输出层
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_beforeBN(self):
        """构建: 全连接 - 批归一化 - 激活函数"""
        if self.dropout_rate > 0:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.BatchNorm1d(self.layer_size[1]),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.BatchNorm1d(self.layer_size[1]),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
        else:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.BatchNorm1d(self.layer_size[1]),
                self.activation
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.BatchNorm1d(self.layer_size[1]),
                self.activation
            )
        for k in range(len(self.layer_size)-2):
            if self.dropout_rate > 0:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        nn.BatchNorm1d(self.layer_size[k+1]),
                        self.activation,
                        nn.Dropout(keep_prob=1.0 - self.dropout_rate)
                    )
                )
            else:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        nn.BatchNorm1d(self.layer_size[k+1]),
                        self.activation
                    )
                )
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_afterBN(self):
        """构建: 全连接 - 激活函数 - 批归一化"""
        if self.dropout_rate > 0:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.BatchNorm1d(self.layer_size[1]),
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.BatchNorm1d(self.layer_size[1]),
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
        else:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.BatchNorm1d(self.layer_size[1])
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.BatchNorm1d(self.layer_size[1])
            )
        for k in range(len(self.layer_size)-2):
            if self.dropout_rate > 0:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation,
                        nn.BatchNorm1d(self.layer_size[k+1]),
                        nn.Dropout(keep_prob=1.0 - self.dropout_rate)
                    )
                )
            else:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation,
                        nn.BatchNorm1d(self.layer_size[k+1])
                    )
                )
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_beforeLN(self):
        """构建: 全连接 - 层归一化 - 激活函数"""
        if self.dropout_rate > 0:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.LayerNorm((self.layer_size[1],)),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.LayerNorm((self.layer_size[1],)),
                self.activation,
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
        else:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.LayerNorm((self.layer_size[1],)),
                self.activation
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                nn.LayerNorm((self.layer_size[1],)),
                self.activation
            )
        for k in range(len(self.layer_size)-2):
            if self.dropout_rate > 0:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        nn.LayerNorm((self.layer_size[k+1],)),
                        self.activation,
                        nn.Dropout(keep_prob=1.0 - self.dropout_rate)
                    )
                )
            else:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        nn.LayerNorm((self.layer_size[k+1],)),
                        self.activation
                    )
                )
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_afterLN(self):
        """构建: 全连接 - 激活函数 - 层归一化"""
        if self.dropout_rate > 0:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.LayerNorm((self.layer_size[1],)),
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.LayerNorm((self.layer_size[1],)),
                nn.Dropout(keep_prob=1.0 - self.dropout_rate)
            )
        else:
            self.U = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.LayerNorm((self.layer_size[1],))
            )
            self.V = nn.SequentialCell(
                nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias),
                self.activation,
                nn.LayerNorm((self.layer_size[1],))
            )
        for k in range(len(self.layer_size)-2):
            if self.dropout_rate > 0:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation,
                        nn.LayerNorm((self.layer_size[k+1],)),
                        nn.Dropout(keep_prob=1.0 - self.dropout_rate)
                    )
                )
            else:
                self.net.append(
                    nn.SequentialCell(
                        nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias),
                        self.activation,
                        nn.LayerNorm((self.layer_size[k+1],))
                    )
                )
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))
