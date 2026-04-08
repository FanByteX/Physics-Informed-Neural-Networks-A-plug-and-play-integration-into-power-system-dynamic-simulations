"""
事件系统基础模块
Event: 基础事件类，定义训练过程中的回调接口
EventList: 管理多个事件的列表
"""


class Event:
    """训练事件基类"""
    
    def __init__(self):
        self.model = None

    def set_model(self, model):
        """设置关联的模型"""
        if model is not self.model:
            self.model = model
            self.init()
    
    def init(self):
        """初始化事件"""
        pass
    
    def on_epoch_started(self):
        """epoch 开始时调用"""
        pass
    
    def on_epoch_completed(self):
        """epoch 结束时调用"""
        pass
    
    def on_train_started(self):
        """训练开始时调用"""
        pass
    
    def on_train_completed(self):
        """训练结束时调用"""
        pass
    
    def on_predict_started(self):
        """预测开始时调用"""
        pass
    
    def on_predict_completed(self):
        """预测结束时调用"""
        pass


class EventList(Event):
    """事件列表管理类"""
    
    def __init__(self, events=None):
        events = events or []
        self.events = [e for e in events]
        self.model = None
    
    def set_model(self, model):
        """设置所有事件的模型"""
        self.model = model
        for e in self.events:
            e.set_model(model)
    
    def on_epoch_started(self):
        for e in self.events:
            e.on_epoch_started()
    
    def on_epoch_completed(self):
        for e in self.events:
            e.on_epoch_completed()
    
    def on_train_started(self):
        for e in self.events:
            e.on_train_started()
    
    def on_train_completed(self):
        for e in self.events:
            e.on_train_completed()
    
    def on_predict_started(self):
        for e in self.events:
            e.on_predict_started()
    
    def on_predict_completed(self):
        for e in self.events:
            e.on_predict_completed()
