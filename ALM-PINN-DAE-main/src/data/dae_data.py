"""
DAE 数据加载器模块
dae_data: DAE (微分代数方程) 专用数据类
采用增广拉格朗日 (ALM) 方法处理代数约束
ALM 损失: L = f(x) + λg(x) + (μ/2)||g(x)||²
"""
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from .base import Data
from .irk_weights import get_irk_weights_times
from utils.losses import MSE
class dae_data(Data):
    """
    RK-DAE 数据集 + 增广拉格朗日约束优化
    
    ALM参数:
        lambda_alg: 拉格朗日乘子 (代数约束)
        mu: 罚参数
        rho: 罚参数增长率
    """
    def __init__(self, x_train, x_test, args, device="cpu", func=None,
                 mu_init=1.0, rho=1.5, mu_max=1e6):
        if x_train is not None:
            self.x_train = x_train
            self.x_test = x_test
        else:
            raise ValueError("训练数据不能为空 {}".format(x_train))

        self.nu = args.num_IRK_stages
        self.device = device
        IRK_w_np, IRK_times_np = get_irk_weights_times(self.nu)
        self.h = Tensor([args.h], mindspore.float32)
        self.IRK_weights = Tensor(IRK_w_np, mindspore.float32)
        self.IRK_times = IRK_times_np
        self.pinn = func
        
        # ALM参数 - 拉格朗日乘子和罚参数
        self.lambda_alg = 0.0   # 拉格朗日乘子
        self.mu = mu_init       # 罚参数
        self.rho = rho          # 罚参数增长率
        self.mu_max = mu_max    # 罚参数上限
        self.g_history = []     # 约束违反历史

    def loss_fn(self, inputs, model):
        """计算损失 - 增广拉格朗日方法"""
        losses = []
        f, g = self.pinn(model, inputs, self.h, self.IRK_weights)

        # 动力方程损失 (ODE残差)
        losses_dyn = [MSE(fi) for fi in f]
        losses.append(sum(losses_dyn))
        
        # ALM代数约束损失: λg + (μ/2)g²
        g_residual = sum([MSE(gi) for gi in g])
        # 拉格朗日项 + 罚函数项
        loss_alg = self.lambda_alg * ops.sqrt(g_residual + 1e-8) + (self.mu / 2.0) * g_residual
        losses.append(loss_alg)
        
        return losses
    
    def update_alm_params(self, g_violation):
        """
        更新ALM参数 - 每个epoch后调用
        
        拉格朗日乘子更新: λ ← λ + μg
        罚参数更新: μ ← min(ρμ, μ_max)
        """
        self.g_history.append(g_violation)
        
        # 更新拉格朗日乘子
        self.lambda_alg = self.lambda_alg + self.mu * g_violation
        
        # 自适应增加罚参数
        if len(self.g_history) > 1:
            if self.g_history[-1] > 0.25 * self.g_history[-2]:
                self.mu = min(self.rho * self.mu, self.mu_max)
        
        return self.lambda_alg, self.mu
    
    def loss_fn_simple(self, inputs, model):
        """简化损失 - 仅用罚函数 (无拉格朗日乘子)"""
        losses = []
        f, g = self.pinn(model, inputs, self.h, self.IRK_weights)
        losses_dyn = [MSE(fi) for fi in f]
        losses.append(sum(losses_dyn))
        losses_alg = [MSE(gi) for gi in g]
        losses.append(sum(losses_alg))
        return losses
    def train_next_batch(self, batch_size=None):
        """获取下一个训练批次"""
        y_train = np.zeros((self.x_train.shape[0], 1))        # 数据加载器所需，未使用
        if (batch_size is None) or (batch_size > self.x_train.shape[0]):
            return self.x_train, y_train, self.x_train.shape[0]
        else:
            return self.x_train, y_train, batch_size
    def test(self):
        """获取测试数据"""
        y_test = np.zeros((self.x_test.shape[0], 1))        # 数据加载器所需，未使用
        return self.x_train, y_test
