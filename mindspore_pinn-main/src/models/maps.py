import torch
import torch.nn as nn

from . import activations

class fnn(nn.Module):
    """
    前馈神经网络。
    """
    def __init__(
        self,
        layer_size, 
        activation,
        kernel_initializer,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
        use_bias=True,
        print_net=False,
        ):
        super(fnn, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
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
        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("神经网络未构建")
        
        # 初始化参数
        self.net.apply(self._init_weights)
        
        if print_net:
            print("神经网络已构建...\n")
            print(self.net)

    def forward(self, input):
        """
        前馈神经网络前向传播。
        参数:
            :input (Tensor): \in [B, d_in]
        返回:
            :y (Tensor): \in [B, d_out]
        """
        y = input 
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights(self, m):
        """
        初始化层参数。
        """ 
        if isinstance(m, nn.Linear):
            if self.initializer == "Glorot normal":
                nn.init.xavier_normal_(m.weight)
            elif self.initializer == "Glorot uniform":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError("初始化器 {} 未实现".format(self.initializer))
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def build_standard(self):
        # 全连接 - 激活函数
        # 输入层
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

    def build_before(self):
        # 全连接 - 批归一化或层归一化 - 激活函数
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

    def build_after(self):
        # 全连接 - 激活函数 - 批归一化或层归一化
        self.net.append(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(nn.Linear(self.layer_size[i], self.layer_size[i+1], bias=self.use_bias))

class attention(nn.Module):
    """
    具有注意力机制架构的前馈神经网络。
    """
    def __init__(
        self,
        layer_size, 
        activation,
        kernel_initializer,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
        use_bias=True,
        print_net=False,
        ):
        super(attention, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
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
        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()

        self.net = nn.ModuleList()

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

    def forward(self, input):
        """
        前馈神经网络前向传播
        参数:
            :input (Tensor): \in [B, d_in]
        返回:
            :y (Tensor): \in [B, d_out]
        """
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

    def _init_weights(self, m):
        """
        初始化层参数
        """ 
        if isinstance(m, nn.Linear):
            if self.initializer == "Glorot normal":
                nn.init.xavier_normal_(m.weight)
            elif self.initializer == "Glorot uniform":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError("初始化器 {} 未实现".format(self.initializer))
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def build_standard(self):
        # 构建U和V网络
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation
                )  
            )
        # 输出层
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

    def build_beforeBN(self):
        # 全连接 - 批归一化 - 激活函数
        # 构建U和V网络
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[1]),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    self.activation
                )  
            )
        # 输出层
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))
        
    def build_afterBN(self):
        # 全连接 - 激活函数 - 批归一化
        # 构建U和V网络
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[1]))
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[k+1]),
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.BatchNorm1d(self.layer_size[k+1]),
                )  
            )
        # 输出层
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

    def build_beforeLN(self):
        # 全连接 - 层归一化 - 激活函数
        # 构建U和V网络
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation)
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[1]),
                    self.activation)
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[k+1]),
                    self.activation,
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    nn.LayerNorm(self.layer_size[k+1]),
                    self.activation
                )  
            )
        # 输出层
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))
        
    def build_afterLN(self):
        # 全连接 - 激活函数 - 层归一化
        # 构建U和V网络
        if self.dropout_rate > 0:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]),
                    nn.Dropout(p=self.dropout_rate))
        else:
            self.U = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]))
            self.V = nn.Sequential(nn.Linear(self.layer_size[0], self.layer_size[1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[1]))
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[k+1]),
                    nn.Dropout(p=self.dropout_rate)
                ) if (self.dropout_rate > 0) else nn.Sequential(
                    nn.Linear(self.layer_size[k], self.layer_size[k+1], bias=self.use_bias),
                    self.activation,
                    nn.LayerNorm(self.layer_size[k+1]),
                )  
            )
        # 输出层
        self.net.append(nn.Linear(self.layer_size[-2], self.layer_size[-1], bias=self.use_bias))

class dense_Conv1D(nn.Module):
    """
    由Radford等人为OpenAI GPT定义的1D卷积层（GPT-2中也使用）。
    基本上类似于线性层，但权重是转置的。
    参数:
        :inputs (int): 输入特征的数量。
        :outputs (out): 输出特征的数量。
    代码来源于: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py
    """
    def __init__(
        self, 
        inputs, 
        outputs,
        activation=None,
        ):
        super(dense_Conv1D, self).__init__()
        self.n_out = outputs
        w = torch.empty(inputs, outputs)
        nn.init.normal_(w, std=.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(outputs))
        self.activation = activation

    def forward(self, input):
        """
        前向传播
        参数:
            :input (Tensor): \in [batch_size, inputs]
        返回:
            :(Tensor): \in [batch_size, outputs]
        """
        size_out = input.size()[:-1] + (self.n_out,)
        y = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight)
        y = y.view(*size_out)
        if self.activation is not None:
            y = self.activation(y)
        return y

class Conv1D(nn.Module):
    """
    使用Conv1D密集层的前馈神经网络。
    """
    def __init__(
        self,
        layer_size,
        activation,
        dropout_rate=0.0,
        batch_normalization=None,
        layer_normalization=None,
        input_transform=None,
        output_transform=None,
    ):
        super(Conv1D, self).__init__()
        self.layer_size = layer_size
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization 
        self.input_transform = input_transform
        self.output_transform = output_transform 
        
        if self.batch_normalization and self.layer_normalization:
            raise ValueError("不能同时应用批归一化和层归一化。")

        self.net = nn.ModuleList()

        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("神经网络未构建")
        
        print("神经网络已构建...\n")
        print(self.net)

    def forward(self, input):
        """
        前馈神经网络前向传播
        参数:
            :input (Tensor): \in [B, d_in]
        返回:
            :y (Tensor): \in [B, d_out]
        """
        y = input 
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def build_standard(self):
        # 全连接 - 激活函数
        # 输入层
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))

    def build_before(self):
        # 全连接 - 批归一化或层归一化 - 激活函数
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1]))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i]))
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1]))

    def build_after(self):
        # 全连接 - 激活函数 - 批归一化或层归一化
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.batch_normalization is not None:
            self.net.append(nn.BatchNorm1d(self.layer_size[1]))
        elif self.layer_normalization is not None:
            self.net.append(nn.LayerNorm(self.layer_size[1]))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(p=self.dropout_rate))
        for i in range(1, len(self.layer_size) - 2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i+1]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm(self.layer_size[i+1]))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(p=self.dropout_rate))
            
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))