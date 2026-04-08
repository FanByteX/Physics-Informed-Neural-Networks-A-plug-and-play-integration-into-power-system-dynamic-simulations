"""
训练管理器模块 (Supervisor)
负责整个训练流程，包括：
1. 编译 (compile): 配置优化器和损失权重
2. 训练 (train): 执行训练循环，使用 MindSpore 的 value_and_grad
3. 测试 (_test): 评估模型在测试集上的性能
4. 保存/加载 (save/restore): 模型检查点管理
5. 预测 (predict): 使用训练好的模型进行推理
6. 积分 (integrate): 时间序列积分，用于轨迹预测
"""
import os
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
from tqdm import tqdm

from utils import timing, training_display
from .events import EventList
from .state import State
from .loss_history import LossHistory


class supervisor(object):
    """训练管理器"""
    
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
        """编译模型：配置优化器和损失权重"""
        print("Compiling supervisor...\n")
        self.optimizer = optimizer
        self.metrics = []
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.loss_history.set_loss_weights(loss_weights)

    @timing
    def train(self, epochs=None, batch_size=None, test_every=1000, num_val=10, 
              disregard_previous_best=False, events=None, model_restore_path=None, 
              model_save_path=None, use_tqdm=True):
        """执行训练"""
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
            mindspore.load_param_into_net(self.net, mindspore.load_checkpoint(model_restore_path))
            print("Network weights loaded successfully.")
            if 'optimizer' in check_point and self.optimizer is not None:
                try:
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
        """内部训练循环"""
        range_epochs = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in range_epochs:
            self.events.on_epoch_started()
            self.net.set_train(True)
            loss_record_epoch = []
            
            for batch in self.state.train_loader:
                x_batch = batch["X"]
                
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
                
                self.optimizer(grads)
                
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
            
            if self.state.step % test_every == 0 or epoch + 1 == epochs:
                self._test()
            self.events.on_epoch_completed()

            if self.stop_training:
                break

    def _test(self):
        """测试模型"""
        self.net.set_train(False)
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
        
        Args:
            input: numpy Tensor 或 tensors列表
            events: 事件实例列表
            model_restore_path: 之前保存model.parameters()的路径
            
        Returns:
            y: numpy Tensor
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
        """时间序列积分"""
        yn = X0
        soln = []
        for _ in range(N):
            y_pred_n = self.predict(yn.reshape(1, -1), model_restore_path=model_restore_path)
            soln.append(y_pred_n)
            yn = y_pred_n[:dyn_state_dim, -1]
        return np.hstack(soln)

    def save(self, save_path, verbose=0):
        """保存模型"""
        if verbose > 0:
            print("Epoch {}: saving to {} ...\n".format(self.state.epoch, save_path))
        mindspore.save_checkpoint(self.net, save_path)

    def restore(self, restore_path, verbose=0):
        """恢复模型"""
        if verbose > 0:
            print("Restoring from {}".format(restore_path))
        checkpoint = mindspore.load_checkpoint(restore_path)
        return {'state_dict': checkpoint}
