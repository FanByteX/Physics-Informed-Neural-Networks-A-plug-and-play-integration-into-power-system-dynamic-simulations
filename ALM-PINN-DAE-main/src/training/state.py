"""
训练状态模块
State: 记录训练过程中的状态信息
"""
import numpy as np
import mindspore
from mindspore import Tensor


class State(object):
    """训练状态类"""
    
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
        """设置训练数据"""
        X = Tensor(X, mindspore.float32)
        y = Tensor(y, mindspore.float32)
        # MindSpore中使用Dataset类
        dataset = mindspore.dataset.NumpySlicesDataset(
            {"X": X.asnumpy(), "y": y.asnumpy()}, 
            shuffle=shuffle
        )
        dataset = dataset.batch(batch_size)
        self.train_loader = dataset.create_dict_iterator()

    def set_data_val(self, X, y, num_val):
        """设置验证数据"""
        self.num_val = num_val
        self.X_val = Tensor(X, mindspore.float32)
        self.y_val = Tensor(y, mindspore.float32)

    def get_val_data(self):
        """获取验证数据"""
        num_train = self.X_val.shape[0]
        if self.num_val > num_train:
            self.num_val = num_train
        val_indices = np.random.choice(num_train, self.num_val, replace=False)
        X_val_device = self.X_val[val_indices, :]
        y_val_device = self.y_val[val_indices, :]
        return X_val_device, y_val_device

    def set_data_test(self, X, y):
        """设置测试数据"""
        self.y_test_np = y
        self.X_test = Tensor(X, mindspore.float32)
        self.y_test = Tensor(y, mindspore.float32)

    def disregard_best(self):
        """重置最佳训练损失"""
        self.best_loss_train = np.inf

    def update_best(self):
        """更新最佳状态"""
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_metrics = self.metrics_test
