"""
数据基类模块
Data: 基础数据类，定义数据加载接口
"""


class Data(object):
    """数据基类"""
    
    def __init__(self):
        pass
    
    def loss_fn(self, targets, outputs, model):
        """损失函数"""
        raise NotImplementedError
    
    def train_next_batch(self, batch_size=None):
        """获取下一个训练批次"""
        raise NotImplementedError
    
    def test(self):
        """获取测试数据"""
        raise NotImplementedError
