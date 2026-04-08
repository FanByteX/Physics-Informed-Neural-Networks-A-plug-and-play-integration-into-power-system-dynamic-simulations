"""
命令行参数解析模块
定义所有训练超参数：
- 网络参数: 宽度、深度、激活函数、Dropout
- 训练参数: 学习率、批次大小、epoch 数
- IRK 参数: 龙格-库塔阶数、时间步长
- 输出控制: 日志目录、模型保存等
"""
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="dae-pinns-fault-b5-allinone")
    
    # general
    parser.add_argument('--num-IRK-stages', type=int, default=100, help="number of RK stages")
    parser.add_argument('--log-dir', type=str, default="logs/mindspore_pinn/", help="log dir")
    
    # NPU/GPU 设备配置
    parser.add_argument('--distributed', action='store_true', default=False, 
                       help="enable distributed training on multi-NPU")
    parser.add_argument('--device-id', type=int, default=0, 
                       help="NPU device ID (0-3 for single card training)")
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                       help="disable cuda training (legacy)")
    parser.add_argument('--gpu-number', type=int, default=0, help="GPU device number (legacy)")
    parser.add_argument('--num-train', type=int, default=1000, help="number of training examples")
    parser.add_argument('--num-val', type=int, default=200, help="number of validation examples")
    parser.add_argument('--num-test', type=int, default=400, help="number of test examples")
    parser.add_argument('--num-plot', type=int, default=1, help="number of ICs for plotting")
    
    # scheduler
    parser.add_argument('--use-scheduler', action='store_true', default=True, help='use lr scheduler')
    parser.add_argument('--scheduler-type', type=str, default="plateau", help="scheduler type")
    parser.add_argument('--patience', type=int, default=3000, help="patience for scheduler")
    parser.add_argument('--factor', type=float, default=.8, help="factor for scheduler")
    
    # optimizer
    parser.add_argument('--use-tqdm', action='store_true', default=True, help="use tqdm for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=5000, help="number of epochs (demo)")
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--test-every', type=int, default=500, help="test and log every * steps")
    parser.add_argument('--start-from-best', action='store_true', default=False, 
                       help='start from best model so far')
    parser.add_argument('--model-name', type=str, default="no-name", help="model_ + model-name + .pth")
    
    # neural nets
    parser.add_argument('--dropout-rate', type=float, default=0.0, help="dropout rate")
    parser.add_argument('--dyn-bn', type=str, default="no-bn", 
                       help="dyn batch normalization {before, after}")
    parser.add_argument('--dyn-ln', type=str, default="no-ln", 
                       help="dyn layer normalization {before, after}")
    parser.add_argument('--dyn-type', type=str, default="attention", 
                       help="dyn net type {fnn, attention, Conv1D}")
    parser.add_argument('--unstacked', action='store_true', default=True, 
                       help="use unstaked nets for dynamic vars")
    parser.add_argument('--use-input-layer', action='store_true', default=False, 
                       help="use input feature layer for dynamic vars")
    parser.add_argument('--dyn-width', type=int, default=100, 
                       help="width of hidden layers - dynamic vars")
    parser.add_argument('--dyn-depth', type=int, default=4, 
                       help="depth of hidden layers - dynamic vars")
    parser.add_argument('--dyn-activation', type=str, default="sin", 
                       help="dynamic vars activation function")
    parser.add_argument('--dyn-weight', type=float, default=32.0, 
                       help="weight for dynamic residual loss")
    parser.add_argument('--alg-bn', type=str, default="no-bn", 
                       help="alg batch normalization {before, after}")
    parser.add_argument('--alg-ln', type=str, default="no-ln", 
                       help="alg layer normalization {before, after}")
    parser.add_argument('--alg-type', type=str, default="attention", 
                       help="alg net type {fnn, attention, Conv1D}")
    parser.add_argument('--alg-width', type=int, default=40, 
                       help="width of hidden layers - algebraic vars")
    parser.add_argument('--alg-depth', type=int, default=2, 
                       help="depth of hidden layers - algebraic vars")
    parser.add_argument('--alg-activation', type=str, default="sin", 
                       help="algebraic vars activation function")
    parser.add_argument('--alg-weight', type=float, default=1.0, 
                       help="weight for algebraic residual loss")
    
    # integration
    parser.add_argument('--h', type=float, default=.1, help="step size")
    parser.add_argument('--N', type=int, default=20, help="number of steps")
    parser.add_argument('--method', type=str, default='BDF', help="integration method")
    
    args = parser.parse_args()
    return args


def get_default_args():
    """获取默认参数（不通过命令行）"""
    class Args:
        def __init__(self):
            # general
            self.num_IRK_stages = 100
            self.log_dir = "logs/mindspore_pinn/"
            
            # device
            self.distributed = False
            self.device_id = 0
            self.no_cuda = False
            self.gpu_number = 0
            self.num_train = 1000
            self.num_val = 200
            self.num_test = 400
            self.num_plot = 1
            
            # scheduler
            self.use_scheduler = True
            self.scheduler_type = "plateau"
            self.patience = 3000
            self.factor = 0.8
            
            # optimizer
            self.use_tqdm = True
            self.lr = 1e-3
            self.epochs = 5000
            self.batch_size = 128
            self.test_every = 500
            self.start_from_best = False
            self.model_name = "no-name"
            
            # neural nets
            self.dropout_rate = 0.0
            self.dyn_bn = "no-bn"
            self.dyn_ln = "no-ln"
            self.dyn_type = "attention"
            self.unstacked = True
            self.use_input_layer = False
            self.dyn_width = 100
            self.dyn_depth = 4
            self.dyn_activation = "sin"
            self.dyn_weight = 32.0
            self.alg_bn = "no-bn"
            self.alg_ln = "no-ln"
            self.alg_type = "attention"
            self.alg_width = 40
            self.alg_depth = 2
            self.alg_activation = "sin"
            self.alg_weight = 1.0
            
            # integration
            self.h = 0.1
            self.N = 20
            self.method = 'BDF'
    
    return Args()
