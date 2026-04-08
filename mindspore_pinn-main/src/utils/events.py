import sys
import time
import numpy as np

class Event(object):
    """
    事件基类
    参数:
        :model 监督器实例
    """
    def __init__(self):
        self.model = None
    
    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """
        设置模型后初始化
        """

    def on_epoch_started(self):
        """
        在每个轮次开始时调用
        """
    
    def on_epoch_completed(self):
        """
        在轮次结束时调用
        """

    def on_train_started(self):
        """
        在训练开始时调用
        """

    def on_train_completed(self):
        """
        在训练结束时调用
        """

    def on_predict_started(self):
        """
        在预测开始时调用
        """

    def on_predict_completed(self):
        """
        在预测结束时调用
        """

class EventList(Event):
    """
    包含事件列表的抽象类
    """
    def __init__(self, events=None):
        events = events or []
        self.events = [e for e in events]
        self.model = None

    def set_model(self, model):
        self.model = model
        for event in self.events:
            event.set_model(model)

    def on_epoch_started(self):
        for event in self.events:
            event.on_epoch_started()

    def on_epoch_completed(self):
        for event in self.events:
            event.on_epoch_completed()

    def on_train_started(self):
        for event in self.events:
            event.on_train_started()

    def on_train_completed(self):
        for event in self.events:
            event.on_train_completed()
    
    def on_predict_started(self):
        for event in self.events:
            event.on_predict_started()

    def on_predict_completed(self):
        for event in self.events:
            event.on_predict_completed()

    def append(self, event):
        if not isinstance(event, Event):
            raise Exception(str(event) + " 是无效的事件对象")
        self.events.append(event)

class ModelCheckPoint(Event):
    """
    每个轮次后保存模型
    参数:
        :filepath
        :save_better_only
        :every: 检查点之间的间隔（轮次数）
    """
    def __init__(self, filepath, verbose=0, save_better_only=False, every=1, monitor="train loss"):
        super(ModelCheckPoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = every

        self.monitor = monitor
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_epoch_completed(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            if self.monitor == "train loss":
                current = self.model.state.best_loss_train
            elif self.monitor == "test loss":
                current = self.model.state.best_loss_test
            if current < self.best:
                if self.verbose > 0:
                    print(
                        "轮次 {epoch}: {} 从 {:.2e} 改进到 {:.2e}, 正在将模型保存到 {}-{epoch} ...\n".format(
                            self.monitor,
                            self.best,
                            current,
                            self.filepath,
                            epoch=self.model.state.epoch,
                    ))
                self.best = current
                self.model.save(self.filepath, verbose=0)
        else:
            self.model.save(self.filepath, verbose=self.verbose)

class EarlyStopping(Event):
    """
    当监控的数量（训练损失）停止改善时停止训练。
    参数
        :min_delta: 监控数量的最小变化以符合改进条件
        :patience: 没有改进的轮次数，之后训练将停止
        :baseline: 监控数量要达到的基准值。如果模型没有显示出超过基准的改进，训练将停止
    """
    def __init__(self, min_delta=0, patience=0, baseline=None, monitor="train loss"):
        super(EarlyStopping, self).__init__()

        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.min_delta *= -1
        self.monitor = monitor

    def on_train_started(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf

    def on_epoch_completed(self):
        if self.monitor == "train loss":
            current = self.model.state.loss_train
        elif self.monitor == "test loss":
            current = self.model.state.loss_test

        if current - self.min_delta < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.state.epoch
                self.model.stop_training = True

    def on_train_complete(self):
        if self.stopped_epoch > 0:
            print("轮次 {}: 提前停止".format(self.stopped_epoch))


class Timer(Event):
    """
    当训练时间达到阈值时停止训练
    参数
        :available_time (float): 可用于训练的总时间（分钟）
    """
    def __init__(self, available_time):
        super(Timer, self).__init__()

        self.threshold = available_time * 60  # 转换为秒
        self.t_start = None

    def on_train_started(self):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_completed(self):
        if time.time() - self.t_start > self.threshold:
            self.model.stop_training = True
            print(
                "\n由于时间用完而停止训练。已用时间: {:.1f} 分钟, 已训练轮次: {}".format(
                    (time.time() - self.t_start) / 60, self.model.train_state.epoch
                )
            )