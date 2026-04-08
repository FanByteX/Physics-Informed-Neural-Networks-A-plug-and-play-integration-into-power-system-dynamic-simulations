"""
训练显示模块
TrainingDisplay: 负责在控制台格式化输出训练过程信息
包括：训练步数、训练损失、测试损失、测试指标
"""
import sys
from .utils import list_to_str


class TrainingDisplay:
    """训练过程控制台显示类"""
    
    def __init__(self):
        self.len_train = None
        self.len_test = None
        self.len_metric = None
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4):
        """打印一行信息"""
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                s1, s2, s3, s4, l1=10, l2=self.len_train, l3=self.len_test, l4=self.len_metric
            )
        )
        sys.stdout.flush()

    def header(self):
        """打印表头"""
        self.print_one("Step", "Train Loss", "Test Loss", "Test Metrics")
        self.is_header_print = True

    def __call__(self, state):
        """打印训练状态"""
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
        """打印训练总结"""
        print("Best at step {:d}:".format(state.best_step))
        print("  Train Loss: {:.3e}".format(state.best_loss_train))
        print("  Test Loss: {:.3e}".format(state.best_loss_test))
        print("  Test Metrics: {:s}".format(list_to_str(state.best_metrics)))
        print("")
        self.is_header_print = False


# 全局训练显示实例
training_display = TrainingDisplay()
