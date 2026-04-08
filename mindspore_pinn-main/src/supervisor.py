import numpy as np
import torch
from tqdm import tqdm

from utils import display
from utils.utils import timing
from events import EventList
import metrics as metrics_module

class supervisor(object):
    """
    监督类用于在数据上训练映射
    参数:
        :data (数据实例)
        :map (映射或模型实例)
        :device (cuda 或 cpu)
    """
    def __init__(self, data, net, device="cpu"):
        self.data = data
        self.net = net
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.batch_size = None

        self.state = State(device=self.device)    # 训练状态
        self.loss_history = LossHistory()         # 损失历史
        self.stop_training = False
        self.events = None

    @timing 
    def compile(self, optimizer, metrics=None, loss_weights=None, scheduler=None, scheduler_type=None):
        """
        配置监督器进行训练
        参数:
            :optimizer: torch优化器
            :metrics: 指标列表（DAE-PINNs尚不支持）
            :loss_weights (list): 用于加权损失贡献的标量系数列表
            :scheduler: torch调度器
            :scheduler_type (str): 调度器类型
        """
        print("正在编译监督器...\n")
        self.optimizer = optimizer
        # DAE-PINNs尚不支持指标。
        # 为此，我们必须从MATLAB或Scipy收集模拟数据。
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss_history.set_loss_weights(loss_weights)

    @timing 
    def train(
        self, 
        epochs=None,
        batch_size=None,
        test_every=1000,
        num_val=10,
        disregard_previous_best=False,
        events=None,
        model_restore_path=None,
        model_save_path=None,
        use_tqdm=True,
        ):
        """
        训练函数
        参数:
            :epochs (int): 训练轮数
            :batch_size (int 或 None): 批次大小
            :test_every (int): 每隔多少次测试一次函数
            :num_val (int): 验证样本数量
            :disregard_previous_best (bool): 如果为True，则忽略之前的最佳模型
            :events (list): 事件实例列表
            :model_restore_path (str): 之前保存model.parameters()的路径
            :model_save_path (str): 检查点的文件名
            :use_tqdm (bool): 打印训练过程
        """
        self.batch_size = batch_size
        self.num_val = num_val
        self.events = EventList(events=events)
        self.events.set_model(self)
        # 忽略之前的最佳模型
        if disregard_previous_best:
            self.state.disregard_best()
        # 如果需要，从路径恢复模型
        if model_restore_path is not None:
            check_point = self.restore(model_restore_path)
            state_dict = check_point['state_dict']
            self.net.load_state_dict(state_dict)
            # 我们总是需要将训练数据和网络参数放在设备内存中
            self.net.to(self.device)
        # 开始训练
        print("正在训练模型...\n")
        self.stop_training = False
        # 获取训练、测试和验证数据 - 这应该设置数据加载器
        self.state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.state.set_data_test(*self.data.test())     # 测试时不使用批次
        self.state.set_data_val(*self.data.train_next_batch(self.num_val))

        self.events.on_train_started()
        self._train(epochs, test_every, use_tqdm)
        self.events.on_train_completed()

        # 打印训练结果
        print("")
        display.training_display.summary(self.state)

        # 保存模型 ### 检查这是否不会覆盖保存的最佳模型
        #if model_save_path is not None:
        #    self.save(model_save_path, verbose=1)
        return self.loss_history, self.state

    def _train(self, epochs, test_every, use_tqdm):
        if use_tqdm:
            range_epochs = tqdm(range(epochs))
        else:
            range_epochs = range(epochs) 

        for epoch in range_epochs:
            self.events.on_epoch_started()
            self.net.train()
            loss_record_epoch = []
            for x_batch, _ in self.state.train_loader:
                self.optimizer.zero_grad()
                # 计算损失
                loss_list = self.data.loss_fn(x_batch, model=self.net)
                if not isinstance(loss_list, list):
                    loss_list = [loss_list]       
                if self.loss_history.loss_weights is not None:                
                    for k in range(len(loss_list)):
                        loss_list[k] *= self.loss_history.loss_weights[k]
                loss = sum(loss_list)
                # 优化参数
                loss.backward()
                self.optimizer.step()
                # 检测梯度爆炸
                if loss.item() > 1e10:
                    print("检测到梯度爆炸")
                    self.stop_training = True
                # 保存当前批次损失
                loss_record_epoch.append(loss.item())
            try:
                avg_loss_epoch = sum(loss_record_epoch) / len(loss_record_epoch)
            except ZeroDivisionError as e:
                print("错误: ", e, "批次大小大于训练样本数量")
            # 保存轮次损失
            self.state.loss_train = [avg_loss_epoch]

            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    if (epoch % 1 == 0):
                        self.net.eval()
                        # 获取验证数据
                        val_data_device, _ = self.state.get_val_data()
                        with torch.no_grad():
                            # 计算损失
                            loss_list = self.data.loss_fn(val_data_device, model=self.net)
                        if not isinstance(loss_list, list):
                            loss_list = [loss_list]
                        if self.loss_history.loss_weights is not None:                
                            for k in range(len(loss_list)):
                                loss_list[k] *= self.loss_history.loss_weights[k]
                        loss_val = sum(loss_list)
                        self.scheduler.step(loss_val.cpu().data.numpy())
                else:
                    self.scheduler.step()

            self.state.epoch += 1
            self.state.step += 1
            # 测试模型
            if self.state.step % test_every == 0 or epoch + 1 == epochs:
                self._test()
            self.events.on_epoch_completed()

            if self.stop_training:
                break

    def _test(self):
        # 不支持批次测试
        self.net.eval()
        # 计算损失
        with torch.no_grad():
            loss_list = self.data.loss_fn(self.state.X_test, model=self.net)
        if not isinstance(loss_list, list):
            loss_list = [loss_list]
        if self.loss_history.loss_weights is not None:                
            for k in range(len(loss_list)):
                loss_list[k] *= self.loss_history.loss_weights[k]
        loss = sum(loss_list)
        self.state.loss_test = [loss.item()]
        # DAE-PINNs在测试期间尚不支持指标
        y_pred_test = None
        self.state.metrics_test = [
            m(self.state.y_test_np, y_pred_test.cpu().data.numpy()) for m in self.metrics
            ]
        self.state.update_best()
        self.loss_history.append(
            self.state.step,
            self.state.loss_train,
            self.state.loss_test,
            self.state.metrics_test,
        )
        display.training_display(self.state)

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
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        X = TensorFloat(input)
        self.events = EventList(events=events)
        self.events.set_model(self)
        self.events.on_predict_started()

        if model_restore_path is not None:
            check_point = self.restore(model_restore_path)
            state_dict = check_point['state_dict']
            self.net.load_state_dict(state_dict)
            # 我们总是需要将训练数据和网络参数放在设备内存中
            self.net.to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            # 前向传播
            vel1, vel2, ang2, ang3, v3 = self.net(X)
        y_pred = np.vstack((vel1.cpu().numpy(),vel2.cpu().numpy(), ang2.cpu().numpy(), ang3.cpu().numpy(), v3.cpu().numpy()))
        self.events.on_predict_completed()
        return y_pred

    # @timing
    def integrate(self, X0, N=1, dyn_state_dim=4, model_restore_path=None):
        """
        对N个时间步长的动力网络动力学进行积分
        参数:
            :X0 (numpy.array): \in [1, dyn_state_dim]
        返回:
            :y_pred (numpy.array): \in [1, dyn_state_dim + alg_state_dim]
        """
        #print("使用DAE-PINNs进行积分...")
        yn = X0
        soln = []
        for _ in range(N):
            y_pred_n = self.predict(yn.reshape(1, -1), model_restore_path=model_restore_path)
            soln.append(y_pred_n)
            yn = y_pred_n[:dyn_state_dim, -1]
        return np.hstack(soln)
            
    def save(self, save_path, verbose=0):
        """
        将模型保存到save_path
        """
        if verbose > 0:
            print(
                "轮次 {}: 正在将模型保存到 {} ...\n".format(
                    self.state.epoch, save_path
                )
            )
        state = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
                }
        torch.save(state, save_path)

    def restore(self, restore_path, verbose=0):
        """
        从restore_path恢复模型
        """
        if verbose > 0:
            print("正在从 {} 恢复模型...".format(restore_path))
        
        # 智能设备映射：自动将模型加载到当前设备
        if str(self.device) == "cpu":
            # 如果当前使用CPU，将模型映射到CPU
            checkpoint = torch.load(restore_path, map_location='cpu')
        else:
            # 如果当前使用GPU，将模型映射到当前GPU设备
            current_device = f'cuda:{torch.cuda.current_device()}'
            checkpoint = torch.load(restore_path, map_location=current_device)
            
        return checkpoint

class State(object):
    def __init__(self, device="cpu"):
        self.epoch, self.step = 0, 0
        self.device = device

        # 数据
        self.loss_train = None
        self.loss_test = None
        self.metrics_test = None
        self.loss_val = None

        # 最佳结果对应于最小训练损失
        # 我们可以改为最小测试损失
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_metrics = None

        # 数据加载器
        self.train_loader = None

    def set_data_train(self, X, y, batch_size, shuffle=True):
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        X, y = TensorFloat(X), TensorFloat(y)
        data = torch.utils.data.TensorDataset(X, y)
        self.train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def set_data_val(self, X, y, num_val):
        self.num_val = num_val
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.X_val, self.y_val = TensorFloat(X), TensorFloat(y)
    
    def get_val_data(self):
        num_train = self.X_val.shape[0]
        if self.num_val > num_train:
            self.num_val = num_train
        val_indices = torch.randperm(num_train)[:self.num_val].tolist()
        X_val_device = self.X_val[val_indices,:].to(self.device)
        y_val_device = self.y_val[val_indices,:].to(self.device)
        return X_val_device, y_val_device

    def set_data_test(self, X, y):
        self.y_test_np = y
        if str(self.device) == "cpu":
            TensorFloat = torch.FloatTensor
        else:
            TensorFloat = torch.cuda.FloatTensor
        self.X_test, self.y_test = TensorFloat(X), TensorFloat(y)

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