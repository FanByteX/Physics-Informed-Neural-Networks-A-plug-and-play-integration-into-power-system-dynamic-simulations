"""
模型检查点模块
ModelCheckPoint: 模型检查点保存，自动保存最优模型
"""
import numpy as np
from .events import Event


class ModelCheckPoint(Event):
    """模型检查点保存事件"""
    
    def __init__(self, filepath, verbose=0, save_better_only=False, every=1, monitor="train loss"):
        """
        初始化检查点保存器
        
        Args:
            filepath: 模型保存路径
            verbose: 是否打印保存信息
            save_better_only: 是否只保存更好的模型
            every: 每隔多少个 epoch 保存一次
            monitor: 监控的指标 ("train loss" 或 "test loss")
        """
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = every
        self.monitor = monitor
        self.epochs_since_last_save = 0
        self.best = np.inf

    def on_epoch_completed(self):
        """epoch 结束时检查是否需要保存模型"""
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
