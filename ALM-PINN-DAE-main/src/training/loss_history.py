"""
损失历史记录模块
LossHistory: 记录训练过程中的损失变化
"""


class LossHistory(object):
    """损失历史记录类"""
    
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        """设置损失权重"""
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        """添加一条历史记录"""
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
