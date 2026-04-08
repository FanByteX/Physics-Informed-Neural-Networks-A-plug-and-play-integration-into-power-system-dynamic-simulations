""" 
========================================================================
MindSpore 版本的 RES-PINN (残差物理信息神经网络)
========================================================================
MindSpore 框架
主要模块说明：
1. 工具函数 (utils): 计时器、字典包装、字符串格式化
2. 显示模块 (display): 训练过程的控制台输出
3. 损失函数 (losses): MSE 损失
4. 评估指标 (metrics): L2 相对误差
5. 事件系统 (events): 模型检查点保存等回调
6. 数据类 (data): DAE 数据加载和损失计算
7. 激活函数 (activations): sin, tanh, swish 等
8. RES-PINN 网络 (networks): fnn, attention, Conv1D 三种架构
9. DAE模型 (three_bus_PN): 电力系统三母线模型
10. 训练管理 (supervisor): 训练循环、优化器、保存/加载
11. 可视化 (plots): 损失曲线、轨迹图、误差图
12. 主函数 (main): 参数解析和完整训练流程
========================================================================
"""

import os
import sys
import time
import math
import argparse
from functools import wraps
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

# 禁用 DeepXDE 的 Horovod 检测（MindSpore 使用自己的分布式训练）
import os
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    # 暂时移除 MPI 环境变量，避免 DeepXDE 误判
    _ompi_vars = {}
    for key in list(os.environ.keys()):
        if key.startswith('OMPI_') or key.startswith('PMI_'):
            _ompi_vars[key] = os.environ.pop(key)
import deepxde as dde

# 恢复 MPI 环境变量供 MindSpore 使用
if '_ompi_vars' in locals():
    os.environ.update(_ompi_vars)
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# 工具函数模块
# timing: 用于测量函数执行时间的装饰器
# dotdict: 支持点访问的字典类（可用 obj.key 替代 obj['key']）
# list_to_str: 将数值列表格式化为字符串，用于控制台输出
# =========================
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result
    return wrapper

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def list_to_str(nums, precision=3):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))

# =========================
# 训练显示模块
# TrainingDisplay: 负责在控制台格式化输出训练过程信息
# 包括：训练步数、训练损失、测试损失、测试指标
# =========================
class TrainingDisplay:
    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4):
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                s1, s2, s3, s4, l1=10, l2=self.len_train, l3=self.len_test, l4=self.len_metric
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Step", "Train Loss", "Test Loss", "Test Metrics")
        self.is_header_print = True

    def __call__(self, state):
        if not self.is_header_print:
            self.len_train = len(state.loss_train) * 10 + 4
            self.len_test = len(state.loss_test) * 10 + 4
            self.len_metric = len(state.metrics_test) * 10 + 4
            self.header()
        self.print_one(
            str(state.step),
            list_to_str(state.loss_train),
            list_to_str(state.loss_test),
            list_to_str(state.metrics_test),
        )

    def summary(self, state):
        print("Best at step {:d}:".format(state.best_step))
        print("  Train Loss: {:.3e}".format(state.best_loss_train))
        print("  Test Loss: {:.3e}".format(state.best_loss_test))
        print("  Test Metrics: {:s}".format(list_to_str(state.best_metrics)))
        print("")
        self.is_header_print = False

training_display = TrainingDisplay()

# =========================
# 损失函数模块
# MSE: 均方误差损失，用于 PINN 训练
# =========================
def MSE(y_pred, y_true=None):
    return ops.mean(y_pred ** 2) if y_true is None else ops.mean((y_pred - y_true) ** 2)

# =========================
# 评估指标模块  
# l2_relative_error: 计算 L2 相对误差，用于评估预测精度
# =========================
def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

# =========================
# 事件系统模块
# Event: 基础事件类，定义训练过程中的回调接口
# EventList: 管理多个事件的列表
# ModelCheckPoint: 模型检查点保存，自动保存最优模型
# =========================
class Event:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()
    def init(self): pass
    def on_epoch_started(self): pass
    def on_epoch_completed(self): pass
    def on_train_started(self): pass
    def on_train_completed(self): pass
    def on_predict_started(self): pass
    def on_predict_completed(self): pass

class EventList(Event):
    def __init__(self, events=None):
        events = events or []
        self.events = [e for e in events]
        self.model = None
    def set_model(self, model):
        self.model = model
        for e in self.events: e.set_model(model)
    def on_epoch_started(self):
        for e in self.events: e.on_epoch_started()
    def on_epoch_completed(self):
        for e in self.events: e.on_epoch_completed()
    def on_train_started(self):
        for e in self.events: e.on_train_started()
    def on_train_completed(self):
        for e in self.events: e.on_train_completed()
    def on_predict_started(self):
        for e in self.events: e.on_predict_started()
    def on_predict_completed(self):
        for e in self.events: e.on_predict_completed()

class ModelCheckPoint(Event):
    def __init__(self, filepath, verbose=0, save_better_only=False, every=1, monitor="train loss"):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = every
        self.monitor = monitor
        self.epochs_since_last_save = 0
        self.best = np.inf

    def on_epoch_completed(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.model.state.best_loss_train if self.monitor == "train loss" else self.model.state.best_loss_test
            if current < self.best:
                if self.verbose > 0:
                    print("Epoch {epoch}: {m} improved {b:.2e} -> {c:.2e}, saving to {p}-{epoch} ...\n".format(
                        m=self.monitor, b=self.best, c=current, p=self.filepath, epoch=self.model.state.epoch
                    ))
                self.best = current
                self.model.save(self.filepath, verbose=0)
        else:
            self.model.save(self.filepath, verbose=self.verbose)

# =========================
# 数据基类和 DAE 数据加载器
# Data: 基础数据类，定义数据加载接口
# get_irk_weights_times: 读取 IRK (隐式龙格-库塔) 权重文件
# dae_data: DAE (微分代数方程) 专用数据类，用于 PINN 训练
# =========================
class Data(object):
    def __init__(self): pass
    def loss_fn(self, targets, outputs, model): raise NotImplementedError
    def train_next_batch(self, batch_size=None): raise NotImplementedError
    def test(self): raise NotImplementedError

def get_irk_weights_times(num_stages, prefer_local=True):
    """
    返回 IRK Butcher 权重矩阵 [nu+1, nu] 和时间节点向量 [nu]
    稳健查找 Butcher_IRK{nu}.txt：依次在当前工作目录、模块所在目录、其上级(src)、项目根目录中查找。
    """
    fname = f"Butcher_IRK{num_stages}.txt"
    module_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(os.getcwd(), fname),
        os.path.join(module_dir, fname),
        os.path.join(os.path.abspath(os.path.join(module_dir, "..")), fname),           # src/
        os.path.join(os.path.abspath(os.path.join(module_dir, "..", "..")), fname),     # project root
    ]
    local_path = None
    for p in candidates:
        if os.path.exists(p):
            local_path = p
            break
    if local_path is None:
        raise FileNotFoundError(
            f"Missing IRK weights file '{fname}'. Searched paths: {', '.join(candidates)}"
        )
    tmp = np.float32(np.loadtxt(local_path, ndmin=2))
    IRK_weights = np.reshape(tmp[0:num_stages**2 + num_stages], (num_stages + 1, num_stages))
    IRK_times = tmp[num_stages**2 + num_stages:]
    return IRK_weights, IRK_times

class dae_data(Data):
    """
    RK - DAE - RES-PINN 数据集
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

# =========================
# 激活函数模块 (MindSpore Cell 版本)
# 定义各种神经网络激活函数：sin, tanh, swish 等
# 每个激活函数都继承自 nn.Cell，实现 construct 方法
# =========================
class sin_act(nn.Cell):
    def __init__(self):
        super(sin_act, self).__init__()
    def construct(self, x):
        return ops.sin(x)

class linear_act(nn.Cell):
    def __init__(self):
        super(linear_act, self).__init__()
    def construct(self, x):
        return x

class mish(nn.Cell):
    def __init__(self):
        super(mish, self).__init__()
    def construct(self, x):
        return x * ops.tanh(nn.functional.softplus(x))

class gelu_fast(nn.Cell):
    def __init__(self):
        super(gelu_fast, self).__init__()
    def construct(self, x):
        return 0.5 * x * (1.0 + ops.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class gelu_new(nn.Cell):
    def __init__(self):
        super(gelu_new, self).__init__()
    def construct(self, x):
        return 0.5 * x * (1.0 + ops.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * ops.pow(x, 3.0))))

def get_activation(identifier):
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sin": sin_act(),
        "linear": linear_act(),
        "mish": mish(),
        "gelu-fast": gelu_fast(),
        "gelu-new": gelu_new(),
    }[identifier]

# =========================
# 神经网络架构模块
# 包含三种网络类型：
# 1. fnn: 普通前馈神经网络 (Feedforward Neural Network)
# 2. attention: 带有注意力机制的神经网络 (门控注意力)
# 3. Conv1D: 使用 1D 卷积层的神经网络
# 所有网络都支持 BatchNorm/LayerNorm 和 Dropout
# =========================
class fnn(nn.Cell):
    """前馈神经网络"""
    def __init__(self, layer_size, activation, kernel_initializer, dropout_rate=0.0, batch_normalization=None, layer_normalization=None, input_transform=None, output_transform=None, use_bias=True, print_net=False):
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
        
        # MindSpore 中参数初始化通过优化器进行，这里进行权重初始化
        
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
        # 全连接 - 激活函数
        self.net.append(nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(self.activation)
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1], has_bias=self.use_bias))

    def build_before(self):
        # 全连接 - 批归一化或层归一化 - 激活函数
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
        # 全连接 - 激活函数 - 批归一化或层归一化
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

class attention(nn.Cell):
    """具有注意力机制架构的前馈神经网络"""
    def __init__(self, layer_size, activation, kernel_initializer, dropout_rate=0.0, batch_normalization=None, layer_normalization=None, input_transform=None, output_transform=None, use_bias=True, print_net=False):
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
        # MindSpore中的参数初始化方式
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
        # 构建U和V网络
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
        # 全连接 - 批归一化 - 激活函数
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
        # 输出层
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_afterBN(self):
        # 全连接 - 激活函数 - 批归一化
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
        # 输出层
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_beforeLN(self):
        # 全连接 - 层归一化 - 激活函数
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
        # 输出层
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_afterLN(self):
        # 全连接 - 激活函数 - 层归一化
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
        # 输出层
        self.net.append(nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

class dense_Conv1D(nn.Cell):
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
    def __init__(self, layer_size, activation, dropout_rate=0.0, batch_normalization=None, layer_normalization=None, input_transform=None, output_transform=None):
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
        # 全连接 - 激活函数
        self.net.append(dense_Conv1D(self.layer_size[0], self.layer_size[1], activation=self.activation))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(dense_Conv1D(self.layer_size[i], self.layer_size[i+1], activation=self.activation))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1.0 - self.dropout_rate))
        self.net.append(dense_Conv1D(self.layer_size[-2], self.layer_size[-1]))

    def build_before(self):
        # 全连接 - 批归一化或层归一化 - 激活函数
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
        # 全连接 - 激活函数 - 批归一化或层归一化
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

# =========================
# DAE 主模型：三母线电力网络 (three_bus_PN)
# 这是用于电力系统故障分析的 PINN 模型
# 包含：
# - 动态变量 (Y): 4个角速度和角度变量
# - 代数变量 (Z): 电压变量
# 支持 stacked (堆叠式) 和 unstacked (非堆叠式) 两种模式
# =========================
class three_bus_PN(nn.Cell):
    def __init__(self, dynamic, algebraic, dyn_in_transform=None, dyn_out_transform=None, alg_in_transform=None, alg_out_transform=None, stacked=True):
        super().__init__()
        self.stacked = stacked
        self.dim = 4
        self.num_IRK_stages = dynamic.num_IRK_stages
        if dynamic.type == "fnn":
            if stacked:
                self.Y = nn.CellList([
                    fnn(dynamic.layer_size, dynamic.activation, dynamic.initializer, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = fnn(dynamic.layer_size, dynamic.activation, dynamic.initializer, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
        elif dynamic.type == "attention":
            if stacked:
                self.Y = nn.CellList([
                    attention(dynamic.layer_size, dynamic.activation, dynamic.initializer, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = attention(dynamic.layer_size, dynamic.activation, dynamic.initializer, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
        elif dynamic.type == "Conv1D":
            if stacked:
                self.Y = nn.CellList([
                    Conv1D(dynamic.layer_size, dynamic.activation, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
                    for _ in range(self.dim)
                ])
            else:
                self.Y = Conv1D(dynamic.layer_size, dynamic.activation, dropout_rate=dynamic.dropout_rate, batch_normalization=dynamic.batch_normalization, layer_normalization=dynamic.layer_normalization, input_transform=dyn_in_transform, output_transform=dyn_out_transform)
        else:
            raise ValueError("{} type not implemented".format(dynamic.type))

        if algebraic.type == "fnn":
            self.Z = fnn(algebraic.layer_size, algebraic.activation, algebraic.initializer, dropout_rate=algebraic.dropout_rate, batch_normalization=algebraic.batch_normalization, layer_normalization=algebraic.layer_normalization, input_transform=alg_in_transform, output_transform=alg_out_transform)
        elif algebraic.type == "attention":
            self.Z = attention(algebraic.layer_size, algebraic.activation, algebraic.initializer, dropout_rate=algebraic.dropout_rate, batch_normalization=algebraic.batch_normalization, layer_normalization=algebraic.layer_normalization, input_transform=alg_in_transform, output_transform=alg_out_transform)
        elif algebraic.type == "Conv1D":
            self.Z = Conv1D(algebraic.layer_size, algebraic.activation, dropout_rate=algebraic.dropout_rate, batch_normalization=algebraic.batch_normalization, layer_normalization=algebraic.layer_normalization, input_transform=alg_in_transform, output_transform=alg_out_transform)
        else:
            raise ValueError("{} type not implemented".format(dynamic.type))

    def construct(self, input):
        dim_out = self.num_IRK_stages + 1
        if self.stacked:
            Y0 = self.Y[0](input); Y1 = self.Y[1](input); Y2 = self.Y[2](input); Y3 = self.Y[3](input)
        else:
            Y = self.Y(input)
            Y0 = Y[..., 0:dim_out]; Y1 = Y[..., dim_out:2*dim_out]; Y2 = Y[..., 2*dim_out:3*dim_out]; Y3 = Y[..., 3*dim_out:4*dim_out]
        Z = self.Z(input)
        return Y0, Y1, Y2, Y3, Z

# =========================
# 训练管理器 (Supervisor)
# 负责整个训练流程，包括：
# 1. 编译 (compile): 配置优化器和损失权重
# 2. 训练 (train): 执行训练循环，使用 MindSpore 的 value_and_grad
# 3. 测试 (_test): 评估模型在测试集上的性能
# 4. 保存/加载 (save/restore): 模型检查点管理
# 5. 预测 (predict): 使用训练好的模型进行推理
# 6. 积分 (integrate): 时间序列积分，用于轨迹预测
# =========================
from tqdm import tqdm
class supervisor(object):
    def __init__(self, data, net, device="cpu"):
        self.data = data
        self.net = net
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.batch_size = None
        self.state = State(device=self.device)
        self.loss_history = LossHistory()
        self.stop_training = False
        self.events = None

    @timing
    def compile(self, optimizer, metrics=None, loss_weights=None, scheduler=None, scheduler_type=None):
        print("Compiling supervisor...\n")
        self.optimizer = optimizer
        self.metrics = []
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss_history.set_loss_weights(loss_weights)

    @timing
    def train(self, epochs=None, batch_size=None, test_every=1000, num_val=10, disregard_previous_best=False, events=None, model_restore_path=None, model_save_path=None, use_tqdm=True):
        self.batch_size = batch_size
        self.num_val = num_val
        self.events = EventList(events=events)
        self.events.set_model(self)
        if disregard_previous_best:
            self.state.disregard_best()
        if model_restore_path is not None and os.path.exists(model_restore_path):
            print(f"Loading model from: {model_restore_path}")
            check_point = self.restore(model_restore_path)
            state_dict = check_point['state_dict']
            # MindSpore中使用load_checkpoint加载模形
            mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(model_restore_path))
            print("Network weights loaded successfully.")
            # 恢复优化器状态（可选）
            if 'optimizer' in check_point and self.optimizer is not None:
                try:
                    # MindSpore中不直接推荣优化器状态恢复
                    print("Note: Optimizer state restoration not directly supported in MindSpore")
                except Exception as e:
                    print(f"Warning: Could not restore optimizer state: {e}")
            else:
                print("No optimizer state found in checkpoint.")
        print("Training model...\n")
        self.stop_training = False
        self.state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.state.set_data_test(*self.data.test())
        self.state.set_data_val(*self.data.train_next_batch(self.num_val))
        self.events.on_train_started()
        self._train(epochs, test_every, use_tqdm)
        self.events.on_train_completed()
        print("")
        training_display.summary(self.state)
        return self.loss_history, self.state

    def _train(self, epochs, test_every, use_tqdm):
        range_epochs = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in range_epochs:
            self.events.on_epoch_started()
            # MindSpore中网络囶态切换
            self.net.set_train(True)
            loss_record_epoch = []
            for batch in self.state.train_loader:
                x_batch = batch["X"]
                # MindSpore中使用value_and_grad与CellWithLossAndGrad
                def forward_fn(x_batch_input):
                    loss_list = self.data.loss_fn(x_batch_input, model=self.net)
                    if not isinstance(loss_list, list):
                        loss_list = [loss_list]
                    if self.loss_history.loss_weights is not None:
                        for k in range(len(loss_list)):
                            loss_list[k] = loss_list[k] * self.loss_history.loss_weights[k]
                    loss = ops.sum(ops.stack(loss_list))
                    return loss
                
                grad_fn = mindspore.value_and_grad(forward_fn, None, self.net.trainable_params())
                loss, grads = grad_fn(x_batch)
                
                # 直接更新，MindSpore的分布式训练会自动处理梯度同步
                self.optimizer(grads)
                
                # 检测桶度爆炸
                if loss.item() > 1e10:
                    print("Gradient explosion detected")
                    self.stop_training = True
                loss_record_epoch.append(float(loss))
            
            try:
                avg_loss_epoch = sum(loss_record_epoch) / len(loss_record_epoch)
            except ZeroDivisionError as e:
                print("Error:", e, "Batch size larger than training samples")
                avg_loss_epoch = np.inf
            self.state.loss_train = [avg_loss_epoch]

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if (epoch % 1 == 0):
                        self.net.set_train(False)
                        val_data_device, _ = self.state.get_val_data()
                        # MindSpore中不需要no_grad，直接网络模式推荣
                        loss_list = self.data.loss_fn(val_data_device, model=self.net)
                        if not isinstance(loss_list, list):
                            loss_list = [loss_list]
                        if self.loss_history.loss_weights is not None:
                            for k in range(len(loss_list)):
                                loss_list[k] = loss_list[k] * self.loss_history.loss_weights[k]
                        loss_val = ops.sum(ops.stack(loss_list))
                        self.scheduler.step(float(loss_val))
                else:
                    self.scheduler.step()

            self.state.epoch += 1
            self.state.step += 1
            # 测试模式
            if self.state.step % test_every == 0 or epoch + 1 == epochs:
                self._test()
            self.events.on_epoch_completed()

            if self.stop_training:
                break

    def _test(self):
        self.net.set_train(False)
        # MindSpore中不需要no_grad上下文，直接推荣
        loss_list = self.data.loss_fn(self.state.X_test, model=self.net)
        if not isinstance(loss_list, list):
            loss_list = [loss_list]
        if self.loss_history.loss_weights is not None:
            for k in range(len(loss_list)):
                loss_list[k] = loss_list[k] * self.loss_history.loss_weights[k]
        loss = ops.sum(ops.stack(loss_list))
        self.state.loss_test = [float(loss)]
        y_pred_test = None
        self.state.metrics_test = [
            m(self.state.y_test_np, y_pred_test) for m in self.metrics
        ] if y_pred_test is not None else []
        self.state.update_best()
        self.loss_history.append(
            self.state.step,
            self.state.loss_train,
            self.state.loss_test,
            self.state.metrics_test,
        )
        training_display(self.state)

    def predict(self, input, events=None, model_restore_path=None):
        """
        为给定的输入样本生成输出预测
        参数:
            :input (numpy Tensor 或 tensors列表)
            :events (事件实例列表)
            :model_restore_path (str) 之前保存model.parameters()的路径
        返回:
            :y (numpy Tensor)
        """
        X = Tensor(input, mindspore.float32)
        self.events = EventList(events=events)
        self.events.set_model(self)
        self.events.on_predict_started()

        if model_restore_path is not None:
            mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(model_restore_path))
        
        self.net.set_train(False)
        vel1, vel2, ang2, ang3, v3 = self.net(X)
        y_pred = np.vstack((vel1.asnumpy(), vel2.asnumpy(), ang2.asnumpy(), ang3.asnumpy(), v3.asnumpy()))
        self.events.on_predict_completed()
        return y_pred

    def integrate(self, X0, N=1, dyn_state_dim=4, model_restore_path=None):
        yn = X0
        soln = []
        for _ in range(N):
            y_pred_n = self.predict(yn.reshape(1, -1), model_restore_path=model_restore_path)
            soln.append(y_pred_n)
            yn = y_pred_n[:dyn_state_dim, -1]
        return np.hstack(soln)

    def save(self, save_path, verbose=0):
        if verbose > 0:
            print("Epoch {}: saving to {} ...\n".format(self.state.epoch, save_path))
        # MindSpore中使用save_checkpoint保存
        mindspore.save_checkpoint(self.net, save_path)

    def restore(self, restore_path, verbose=0):
        if verbose > 0:
            print("Restoring from {}".format(restore_path))
        # MindSpore中使用load_checkpoint恢复
        checkpoint = mindspore.load_checkpoint(restore_path)
        return {'state_dict': checkpoint}

class State(object):
    def __init__(self, device="cpu"):
        self.epoch, self.step = 0, 0
        self.device = device
        self.loss_train = None
        self.loss_test = None
        self.metrics_test = None
        self.loss_val = None
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_metrics = None
        self.train_loader = None

    def set_data_train(self, X, y, batch_size, shuffle=True):
        X = Tensor(X, mindspore.float32)
        y = Tensor(y, mindspore.float32)
        # MindSpore中使用Dataset类源
        dataset = mindspore.dataset.NumpySlicesDataset({"X": X.asnumpy(), "y": y.asnumpy()}, shuffle=shuffle)
        dataset = dataset.batch(batch_size)
        self.train_loader = dataset.create_dict_iterator()

    def set_data_val(self, X, y, num_val):
        self.num_val = num_val
        self.X_val = Tensor(X, mindspore.float32)
        self.y_val = Tensor(y, mindspore.float32)

    def get_val_data(self):
        num_train = self.X_val.shape[0]
        if self.num_val > num_train:
            self.num_val = num_train
        # MindSpore中使用随机模块
        val_indices = np.random.choice(num_train, self.num_val, replace=False)
        X_val_device = self.X_val[val_indices,:]
        y_val_device = self.y_val[val_indices,:]
        return X_val_device, y_val_device

    def set_data_test(self, X, y):
        self.y_test_np = y
        self.X_test = Tensor(X, mindspore.float32)
        self.y_test = Tensor(y, mindspore.float32)

    def disregard_best(self):
        self.best_loss_train = np.inf

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_metrics = self.metrics_test

class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)

# =========================
# 可视化模块
# 用于生成各种训练结果图表：
# 1. plot_loss_history: 损失曲线图 (对数坐标)
# 2. plot_three_bus_all: 电力系统轨迹图 (所有5个变量)
# 3. plot_L2relative_error: L2相对误差图
# 4. plot_regression: 回归分析图 (预测 vs 真实值)
# 所有图表使用英文标签，以避免中文编码问题
# =========================
def stylize_axes(ax, size=25, legend=True, xlabel=None, ylabel=None, title=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, top_spine=True, right_spine=True):
    ax.spines['top'].set_visible(top_spine)
    ax.spines['right'].set_visible(right_spine)
    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    if legend:
        leg = ax.legend(fontsize=14, framealpha=0.9)
    return ax

def custom_logplot(ax, x, y, label="Loss", xlims=None, ylims=None, color='blue', linestyle='solid', marker=None):
    ax.semilogy(x, y, color=color, label=label, linestyle=linestyle) if marker is None else ax.semilogy(x, y, color=color, label=label, linestyle=linestyle, marker=marker)
    if xlims is not None: ax.set_xlim(xlims)
    if ylims is not None: ax.set_ylim(ylims)
    return ax

def custom_lineplot(ax, x, y, label=None, xlims=None, ylims=None, color="red", linestyle="solid", linewidth=2.0, marker=None):
    if label is not None:
        ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth) if marker is None else ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth, marker=marker)
    else:
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth) if marker is None else ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker)
    if xlims is not None: ax.set_xlim(xlims)
    if ylims is not None: ax.set_ylim(ylims)
    return ax

def plot_loss_history(loss_history, fname="./logs/loss.png", size=25, figsize=(8,6)):
    loss_train = np.array(loss_history.loss_train)
    loss_test = np.array(loss_history.loss_test)
    all_losses = np.concatenate([loss_train, loss_test])
    finite_losses = all_losses[np.isfinite(all_losses)]
    if len(finite_losses) > 0:
        min_loss = max(np.min(finite_losses), 1e-8)
        max_loss = max(np.percentile(finite_losses, 99.9), 1e-5)
    else:
        min_loss = 1e-8; max_loss = 1e5
    fig, ax = plt.subplots(figsize=figsize)
    custom_logplot(ax, loss_history.steps, loss_train, label="Train", color='blue', linestyle='solid')
    custom_logplot(ax, loss_history.steps, loss_test, label="Test", color='red', linestyle='dashed')
    ax.set_ylim(min_loss, max_loss)
    stylize_axes(ax, size=size, xlabel="Iterations", ylabel="Mean Squared Error")
    fig.tight_layout(); fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True); plt.close(fig)

def plot_three_bus_all(t, y_eval, y_pred, fname="logs/trajectories.png", size=25, figsize=(12,16)):
    """
    Plot all five variables.
    Exact: deep-blue solid; Predicted: orange-red short-dashed (linewidth=1.5).
    """
    t = t.reshape(-1,)
    labels = [r"$\omega_1$", r"$\omega_2$", r"$\delta_2$", r"$\delta_3$", r"$V_3$"]
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=figsize)
    for i in range(5):
        y_pred_i = y_pred[i,...].reshape(-1,)
        # 对V3（代数变量）进行轻度平滑处理以消除数值毛刺
        if i == 4:  # V3 is the 5th variable (index 4)
            from scipy.ndimage import uniform_filter1d
            y_pred_i = uniform_filter1d(y_pred_i, size=3, mode='nearest')  # 3点移动平均
        # Exact: 深蓝色实线
        custom_lineplot(ax[i], t, y_eval[i,...].reshape(-1,), label="Exact", color="#00008B", linestyle="solid", linewidth=2.5)
        # Predicted: 橙红色短虚线，线宽1.5
        custom_lineplot(ax[i], t, y_pred_i, label="Predicted", color="#FF4500", linestyle=(0,(3,3)), linewidth=1.5)
        stylize_axes(ax[i], size=size, xlabel='Time (s)' if i==4 else None, ylabel=labels[i])
    fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

def plot_L2relative_error(N, error, fname="./logs/L2relative_error.png", size=20, figsize=(8,6), var_name=None):
    error = error.reshape(-1,)
    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, N, error, color="green", linestyle="dashed", linewidth=3.0, marker='s', label=None)
    # 根据变量名生成对应的ylabel
    if var_name is not None:
        ylabel_text = rf"$L_2$ Relative Error (${var_name}$)"
    else:
        ylabel_text = r"$L_2$ Relative Error"
    stylize_axes(ax, size=size, xlabel="Time Steps N", ylabel=ylabel_text, legend=False)
    fig.tight_layout(); fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True); plt.close(fig)

def plot_regression(predicted, y, fname="./logs/regression-voltage.png", size=20, figsize=(8,6), x_line=None, y_line=None):
    predicted = predicted.reshape(-1,)
    y = y.reshape(-1,)
    if x_line is None:
        x_line = [y.min(), y.max()]
        y_line = [y.min(), y.max()]
    fig, ax = plt.subplots(figsize=figsize)
    custom_lineplot(ax, x_line, y_line, color="#FF4500", linestyle="dashed", linewidth=3.0)
    ax.scatter(predicted, y, color='blue', marker='o', s=10, alpha=0.5)
    stylize_axes(ax, size=size, xlabel="Predicted", ylabel="Exact", legend=False)
    fig.tight_layout(); fig.savefig(fname, dpi=300, bbox_inches='tight', transparent=True); plt.close(fig)

# =========================
# DAE 物理模型核心函数
# power_net_dae: 电力网络 DAE 方程，定义物理约束
# scipy_integrate: 使用 SciPy 求解器计算真实解，用于对比
# =========================
def power_net_dae(model, y_n, h, IRK_weights):
    T = 1.0
    M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 5.
    V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1

    yn = y_n.copy()
    w1, w2, d2, d3, v3 = model(yn)
    # MindSpore中模式推荣不需要显式的设备移动
    w1 = w1; w2 = w2; d2 = d2; d3 = d3; v3 = v3
    xi_w1 = w1[...,:-1]
    xi_w2 = w2[...,:-1]
    xi_d2 = d2[...,:-1]
    xi_d3 = d3[...,:-1]
    zeta_v3 = v3[...,:-1]

    f_1 = b * V_1 * V_2 * ops.sin(xi_d2) + b * V_2 * zeta_v3 * ops.sin(xi_d2 - xi_d3) + P_g
    f_2 = b * V_1 * zeta_v3 * ops.sin(xi_d3) + b * V_2 * zeta_v3 * ops.sin(xi_d3 - xi_d2) + P_l

    F0 = T * (1 / M_1) * (- D * xi_w1 + f_1 + f_2)
    F1 = T * (1 / M_2) * (- D * xi_w2 - f_1)
    F2 = T * (xi_w2 - xi_w1)
    F3 = T * (- xi_w1 - (1 / D_d) * f_2)

    f0 = yn[...,0:1] -  (w1 - h*ops.matmul(F0, IRK_weights.T))
    f1 = yn[...,1:2] -  (w2 - h*ops.matmul(F1, IRK_weights.T))
    f2 = yn[...,2:3] -  (d2 - h*ops.matmul(F2, IRK_weights.T))
    f3 = yn[...,3:4] -  (d3 - h*ops.matmul(F3, IRK_weights.T))

    G = 2 * b * (v3 ** 2) - b * v3 * V_1 * ops.cos(d3) - b * v3 * V_2 * ops.cos(d3 - d2) + Q_l
    g = - (T / v3) * G

    return [f0, f1, f2, f3], [g]

def scipy_integrate(func, X0, args, IRK_times, N=0):
    # 计算故障瞬间(t=0+)的V3真实初始值，而非使用固定的0.7
    # 使用代数方程在故障参数b=5下求解V3的初始值
    T = 1.0
    M_1, M_2, D, D_d, b = .052, .0531, .05, .005, 5.  # b=5故障场景
    V_1, V_2, P_g, P_l, Q_l = 1.02, 1.05, -2.0, 3.0, .1
    
    # 在t=0时刻，动态变量的初值已知：X0 = [w1_0, w2_0, d2_0, d3_0]
    # 代数方程: 2*b*V3^2 - b*V3*V1*cos(d3) - b*V3*V2*cos(d3-d2) + Q_l = 0
    # 求解V3的初始值
    from scipy.optimize import fsolve
    def algebraic_eq(v3):
        return 2 * b * (v3 ** 2) - b * v3 * V_1 * np.cos(X0[3]) - b * v3 * V_2 * np.cos(X0[3] - X0[2]) + Q_l
    V0 = fsolve(algebraic_eq, 0.98)[0]  # 从0.98附近开始搜索（故障后电压约0.98）
    
    t_span = [0.0, args.h * N]
    t_sim = np.array([])
    # 只包含故障后的时间点，跳过t=0初始状态
    for k in range(1, N + 1):
        temp = (k - 1) * args.h + IRK_times * args.h
        if len(t_sim) == 0:
            t_sim = temp
        else:
            t_sim = np.vstack((t_sim, temp))
        t_next = np.array([k * args.h])
        t_sim = np.vstack((t_sim, t_next))
    
    # 求解时仍然从t=0开始，但输出时只返回t>0的点
    sol = solve_ivp(func, t_span, [X0[0], X0[1], X0[2], X0[3], V0], method=args.method, t_eval=np.concatenate([[0.0], t_sim.reshape(-1,)]))
    y_test = sol.y
    # 返回时跳过t=0的解，只返回故障后的状态
    return t_sim, y_test[:, 1:]

# =========================
# 主训练函数
# run_fault_b5: 完整的训练流程，包括：
# 1. 设备配置 (NPU/CPU)
# 2. 模型创建 (动态变量 + 代数变量)
# 3. 数据准备 (DeepXDE 几何采样)
# 4. 训练执行 (优化器 + 检查点)
# 5. 结果可视化 (损失、轨迹、误差图)
# 6. 误差评估 (L2 相对误差)
# =========================
def run_fault_b5(args):
    print("starting...\n")
    print("=" * 80)
    print("Lines fault training - b=5")
    print("=" * 80)
    print()

    # 注意：与原始 fault_powerNet_b5.py 完全一致，不设置任何全局随机种子
    # 这样模型初始化是完全随机的，每次运行都可能找到不同的局部最优解

    if os.path.isdir(args.log_dir) == False:
        os.makedirs(args.log_dir, exist_ok=True)

    # MindSpore NPU设备配置（支持单卡/多卡）
    from mindspore import context
    from mindspore.communication import init, get_rank, get_group_size
    # 检测是否为分布式训练环境
    distributed = args.distributed
    rank_id = 0
    device_num = 1
    
    if distributed:
        # 分布式训练初始化
        print("="*60)
        print("🚀 Initializing Distributed Training on Multi-NPU")
        print("="*60)
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        print(f"📍Rank {rank_id}/{device_num} initialized")
        
        # 设置当前进程使用的设备ID
        context.set_context(
            mode=context.PYNATIVE_MODE,  # 使用PYNATIVE模式避免GRAPH编译问题
            device_target="Ascend",
            device_id=rank_id,
            save_graphs=False
        )
        device = f"Ascend:{rank_id}"
        print(f"✅ Rank {rank_id}: Using device NPU-{rank_id}")
        
        # 修改日志目录，仅rank 0保存日志
        # 但所有rank都从同一个目录加载模型
        original_log_dir = args.log_dir  # 保存原始日志目录用于加载模型
        if rank_id != 0:
            # 非rank 0不保存日志，但需要一个临时目录避免None错误
            args.log_dir = f"/tmp/mindspore_rank_{rank_id}"
            os.makedirs(args.log_dir, exist_ok=True)
        print(f"📋 Rank {rank_id}: Logs directory: {args.log_dir}")
            
    else:
        # 单卡训练
        try:
            device_id = args.device_id if hasattr(args, 'device_id') else 0
            context.set_context(
                mode=context.PYNATIVE_MODE,
                device_target="Ascend",
                device_id=device_id
            )
            device = f"Ascend:{device_id}"
            print(f"✅ Using single NPU device: NPU-{device_id}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to set Ascend device: {e}")
            print("Falling back to CPU")
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
            device = "CPU"
            print(f"Using device: {device}")
    
    print("="*60)

    dynamic = dotdict()
    dynamic.num_IRK_stages = args.num_IRK_stages
    dynamic.state_dim = 4

    def dyn_input_feature_layer(x):
        return ops.cat((x, ops.cos(np.pi * x), ops.sin(np.pi * x), ops.cos(2 * np.pi * x), ops.sin(2 * np.pi * x)), axis=-1)
    def alg_output_feature_layer(x):
        return ops.softplus(x)

    dynamic.activation = args.dyn_activation
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = args.dropout_rate
    dynamic.batch_normalization = None if args.dyn_bn == "no-bn" else args.dyn_bn
    dynamic.layer_normalization = None if args.dyn_ln == "no-ln" else args.dyn_ln
    dynamic.type = args.dyn_type

    if args.unstacked:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1

    if args.use_input_layer:
        dynamic.layer_size = [dynamic.state_dim * 5] + [args.dyn_width] * args.dyn_depth + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + [args.dyn_width] * args.dyn_depth + [dim_out]
        # MindSpore GRAPH模式不支持内联lambda，定义为普通函数
        def identity_transform(x):
            return x
        dyn_input_feature_layer = identity_transform

    algebraic = dotdict()
    algebraic.num_IRK_stages = args.num_IRK_stages
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + [args.alg_width] * args.alg_depth + [dim_out_alg]
    algebraic.activation = args.alg_activation
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = args.dropout_rate
    algebraic.batch_normalization = None if args.alg_bn == "no-bn" else args.alg_bn
    algebraic.layer_normalization = None if args.alg_ln == "no-ln" else args.alg_ln
    algebraic.type = args.alg_type

    nn_model = three_bus_PN(
        dynamic,
        algebraic,
        dyn_in_transform=dyn_input_feature_layer,
        alg_out_transform=alg_output_feature_layer,
        stacked=not args.unstacked,
    )
    # MindSpore有不同的设备管理方式，此处无需显式设备移动

    # 数据生成：分布式训练时每个rank使用不同的数据
    geom = dde.geometry.Hypercube([-.5, -.5, -.5, -.5],[.5, .5, .5, .5])
    
    if distributed:
        # 分布式训练：每个rank生成独立的数据子集
        # 使用rank_id作为随机种子的偏移
        np.random.seed(1234 + rank_id * 1000)
        X_train = geom.random_points(args.num_train)
        np.random.seed(3456 + rank_id * 1000)
        X_test = geom.random_points(args.num_test)
        print(f"Rank {rank_id}: Generated {len(X_train)} training samples (独立数据分片)")
    else:
        # 单卡训练：使用固定种子
        np.random.seed(1234)
        X_train = geom.random_points(args.num_train)
        np.random.seed(3456)
        X_test = geom.random_points(args.num_test)
    
    # MindSpore GRAPH模式不支持内联lambda，定义为普通函数
    def pinn_func(model, y_n, h, IRK_weights):
        return power_net_dae(model, y_n, h, IRK_weights)
    
    data = dae_data(X_train, X_test, args, device=str(device), func=pinn_func)

    superv = supervisor(data, nn_model, device=device)
    # MindSpore中使用mindspore.nn.optim或mindspore.nn.optimizer
    optimizer = mindspore.nn.Adam(nn_model.trainable_params(), learning_rate=args.lr)
    if args.use_scheduler:
        # MindSpore中的learning rate scheduler
        if args.scheduler_type == "plateau":
            # MindSpore中没有直接的ReduceLROnPlateau，需要自定义
            print("Warning: LR scheduler not fully supported in MindSpore, using fixed LR")
            scheduler = None
        elif args.scheduler_type == "step":
            # 使用自定义scheduler
            scheduler = None
        else:
            scheduler = None
    else:
        scheduler = None

    superv.compile(optimizer, loss_weights=[args.dyn_weight, args.alg_weight], scheduler=scheduler, scheduler_type=args.scheduler_type)

    model_name = 'model.ckpt' if args.model_name == 'no-name' else ('model_' + args.model_name + '.ckpt')
    save_path = os.path.join(args.log_dir, model_name)
    chcker = ModelCheckPoint(save_path, save_better_only=True, every=1000)
    
    # 如果从最佳模型开始，需要恢复 checkpoint 的 best 值
    if args.start_from_best and os.path.exists(save_path):
        try:
            # 读取当前最佳损失
            loss_hist_path = os.path.join(args.log_dir, 'loss-history.npz')
            if os.path.exists(loss_hist_path):
                prev_history = np.load(loss_hist_path)
                prev_loss_test = prev_history['loss_test']
                best_prev_loss = min([sum(l) if isinstance(l, (list, np.ndarray)) else l for l in prev_loss_test])
                chcker.best = best_prev_loss
                print(f"✅ Checkpoint best loss initialized to: {best_prev_loss:.6e}")
        except Exception as e:
            print(f"⚠️  Warning: Could not restore checkpoint best value: {e}")
    
    restore_path = save_path if args.start_from_best else None
    # 分布式训练时，所有rank都从rank 0的日志目录加载模型
    if distributed and args.start_from_best and 'original_log_dir' in locals():
        restore_path = os.path.join(original_log_dir, model_name)
        if not os.path.exists(restore_path):
            print(f"⚠️  Rank {rank_id}: Model not found at {restore_path}, starting from scratch")
            restore_path = None
    if args.start_from_best:
        print("starting from best model so far...")

    loss_history, state = superv.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_every=args.test_every,
        num_val=args.num_val,
        events=[chcker],
        model_restore_path=restore_path,
        use_tqdm=args.use_tqdm,
    )

    print("plotting train and test loss...\n")
    plot_loss_history(loss_history, fname=os.path.join(args.log_dir, 'loss.png'))
    np.savez(os.path.join(args.log_dir, 'loss-history'),
             steps=np.array(loss_history.steps),
             loss_train=np.array(loss_history.loss_train),
             loss_test=np.array(loss_history.loss_test))

    X0 = [0., 0., .1, .1]
    X0_npy = np.array(X0)
    y_pred = superv.integrate(X0_npy, N=args.N, dyn_state_dim=4, model_restore_path=save_path)

    def power_net_dae_plot(t, x):
        eps = 0.0001
        m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 5.
        v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1
        w1, w2, d2, d3, v3 = x
        f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g
        f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
        g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - b * v3 * v_2 * np.cos(d3 - d2) + q_l
        F0 = (1 / m_1) * (- d * w1 + f_1 + f_2)
        F1 = (1 / m_2) * (- d * w2 - f_1)
        F2 = (w2 - w1)
        F3 = (- w1 - (1 / d_d) * f_2)
        F4 = (- (1 / (eps * v3)) * g)
        return F0, F1, F2, F3, F4

    t, y_eval = scipy_integrate(power_net_dae_plot, X0, args, superv.data.IRK_times, N=args.N)
    print("plotting trajectory...\n")
    plot_three_bus_all(t, y_eval, y_pred, fname=os.path.join(args.log_dir, 'trajectories.png'), size=20, figsize=(12,16))

    # L2 relative errors
    error_data = np.empty((args.N, 5))
    for k in range(1, args.N+1):
        y_pred_k = superv.integrate(X0_npy, N=k, dyn_state_dim=4, model_restore_path=save_path)
        _, y_eval_k = scipy_integrate(power_net_dae_plot, X0, args, superv.data.IRK_times, N=k)
        for i in range(5):
            error_data[k-1, i] = l2_relative_error(y_pred_k[i,...], y_eval_k[i,...])

    N_vec = np.arange(1, args.N + 0.1)
    var_names = [r'\omega_1', r'\omega_2', r'\delta_2', r'\delta_3', 'V_3']  # 5个变量的LaTeX名称
    for k in range(5):
        fname_k = 'L2relative_error_' + str(k) + '.png'
        fname = os.path.join(args.log_dir, fname_k)
        plot_L2relative_error(N_vec, error_data[:, k], fname=fname, size=20, figsize=(8,6), var_name=var_names[k])
    np.savez(os.path.join(args.log_dir, "L2Relative_error"), N=N_vec, error=error_data)

    # regression plot for voltage (d3)
    x_line = [-.5, .5]
    y_line = [-.5, .5]
    plot_regression(y_pred[-2,...], y_eval[-2,...], fname=os.path.join(args.log_dir, 'regression-voltage.png'), size=20, figsize=(8,6), x_line=x_line, y_line=y_line)

    # save prediction data
    np.savez(os.path.join(args.log_dir, "prediction-data"), y_pred=y_pred, y_eval=y_eval, time=t)

# =========================
# 命令行参数解析和程序入口
# 定义所有训练超参数：
# - 网络参数: 宽度、深度、激活函数、Dropout
# - 训练参数: 学习率、批次大小、epoch 数
# - IRK 参数: 龙格-库塔阶数、时间步长
# - 输出控制: 日志目录、模型保存等
# =========================
def main():
    parser = argparse.ArgumentParser(description="dae-pinns-fault-b5-allinone")
    # general
    parser.add_argument('--num-IRK-stages', type=int, default=100, help="number of RK stages")
    parser.add_argument('--log-dir', type=str, default="logs/mindspore_pinn/", help="log dir")
    # NPU/GPU 设备配置
    parser.add_argument('--distributed', action='store_true', default=False, help="enable distributed training on multi-NPU")
    parser.add_argument('--device-id', type=int, default=0, help="NPU device ID (0-3 for single card training)")
    parser.add_argument('--no-cuda', action='store_true', default=False, help="disable cuda training (legacy)")
    parser.add_argument('--gpu-number', type=int, default=0, help="GPU device number (legacy)")
    parser.add_argument('--num-train', type=int, default=1000, help="number of training examples")
    parser.add_argument('--num-val', type=int, default=200, help="number of validation examples")
    parser.add_argument('--num-test', type=int, default=400, help="number of test examples")
    parser.add_argument('--num-plot', type=int, default=1, help="number of ICs for plotting")
    # scheduler
    parser.add_argument('--use-scheduler', action='store_true', default=True, help='use lr scheduler')
    parser.add_argument('--scheduler-type', type=str, default="plateau", help="scheduler type")
    parser.add_argument('--patience', type=int, default=3000, help="patience for scheduler")
    parser.add_argument('--factor', type=float, default=.8, help="factor for scheduler")
    # optimizer
    parser.add_argument('--use-tqdm', action='store_true', default=True, help="use tqdm for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=5000, help="number of epochs (demo)")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--test-every', type=int, default=500, help="test and log every * steps")
    parser.add_argument('--start-from-best', action='store_true', default=False, help='start from best model so far')
    parser.add_argument('--model-name', type=str, default="no-name", help="model_ + model-name + .pth")
    # neural nets
    parser.add_argument('--dropout-rate', type=float, default=0.0, help="dropout rate")
    parser.add_argument('--dyn-bn', type=str, default="no-bn", help="dyn batch normalization {before, after}")
    parser.add_argument('--dyn-ln', type=str, default="no-ln", help="dyn layer normalization {before, after}")
    parser.add_argument('--dyn-type', type=str, default="attention", help="dyn net type {fnn, attention, Conv1D}")
    parser.add_argument('--unstacked', action='store_true', default=True, help="use unstaked nets for dynamic vars")
    parser.add_argument('--use-input-layer', action='store_true', default=False, help="use input feature layer for dynamic vars")
    parser.add_argument('--dyn-width', type=int, default=100, help="width of hidden layers - dynamic vars")
    parser.add_argument('--dyn-depth', type=int, default=4, help="depth of hidden layers - dynamic vars")
    parser.add_argument('--dyn-activation', type=str, default="sin", help="dynamic vars activation function")
    parser.add_argument('--dyn-weight', type=float, default=32.0, help="weight for dynamic residual loss")
    parser.add_argument('--alg-bn', type=str, default="no-bn", help="alg batch normalization {before, after}")
    parser.add_argument('--alg-ln', type=str, default="no-ln", help="alg layer normalization {before, after}")
    parser.add_argument('--alg-type', type=str, default="attention", help="alg net type {fnn, attention, Conv1D}")
    parser.add_argument('--alg-width', type=int, default=40, help="width of hidden layers - algebraic vars")
    parser.add_argument('--alg-depth', type=int, default=2, help="depth of hidden layers - algebraic vars")
    parser.add_argument('--alg-activation', type=str, default="sin", help="algebraic vars activation function")
    parser.add_argument('--alg-weight', type=float, default=1.0, help="weight for algebraic residual loss")
    # integration
    parser.add_argument('--h', type=float, default=.1, help="step size")
    parser.add_argument('--N', type=int, default=20, help="number of steps")
    parser.add_argument('--method', type=str, default='BDF', help="integration method")
    args = parser.parse_args()
    run_fault_b5(args)

if __name__ == "__main__":
    main()