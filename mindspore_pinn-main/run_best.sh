#!/usr/bin/env bash

python example_powerNet.py --log-dir ./logs/dae-pinns-best/ --num-test 500 --use-scheduler --patience 2000 --batch-size 1048 \
	--unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 --dyn-type attention --alg-type attention --dyn-activation sin \
	--alg-activation sin --test-every 1000 --scheduler-type plateau --alg-weight 1.0 --num-train 6000 --num-val 100 \
	--use-tqdm --num-test 500 \
	--dyn-weight 64.0 --epochs 30000 --start-from-best --lr 1e-4 
#重新训练 用上边参数  保存dae-pinns-new    仅是  --start-from-best的差距，是否从最佳参数开始训练
python example_powerNet.py --log-dir ./logs/dae-pinns-new/ --num-test 500 --use-scheduler --patience 2000 --batch-size 1048 \
	--unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 --dyn-type attention --alg-type attention --dyn-activation sin \
	--alg-activation sin --test-every 1000 --scheduler-type plateau --alg-weight 1.0 --num-train 6000 --num-val 100 \
	--use-tqdm --num-test 500 \
	--dyn-weight 64.0 --epochs 30000 --lr 1e-4


	
# --log-dir ./logs/dae-pinns-best/：指定日志文件存储目录
# --num-test 500：设置测试样本数量为 500
# --use-scheduler：启用学习率调度器
# --patience 2000：早停策略的耐心值，即性能不提升 2000 轮后停止
# --batch-size 1048：设置批次大小为 1048
# --unstacked：使用非堆叠结构
# --dyn-depth 4：动态网络的深度为 4 层
# --dyn-width 100：动态网络的宽度（每层神经元数）为 100
# --h 0.1：可能是时间步长或离散化步长
# --N 80：可能是时间步数或节点数量
# --dyn-type attention：动态网络使用注意力机制
# --alg-type attention：代数网络使用注意力机制
# --dyn-activation sin：动态网络使用正弦激活函数
# --alg-activation sin：代数网络使用正弦激活函数
# --test-every 1000：每 1000 个 epoch 进行一次测试
# --scheduler-type plateau：学习率调度器类型为 plateau（当指标不再提升时调整）
# --alg-weight 1.0：代数损失的权重为 1.0
# --num-train 6000：训练样本数量为 6000
# --num-val 100：验证样本数量为 100
# --use-tqdm：使用 tqdm 进度条
# --dyn-weight 64.0：动态损失的权重为 64.0
# --epochs 30000：训练总轮次为 30000
# --start-from-best：从最佳模型开始（可能指加载之前保存的最佳模型）
# --lr 1e-4：初始学习率为 0.0001

#从cmd.sh复制的第2个
# python example_powerNet.py --log-dir ./logs/dae-pinns-best-model/ --num-train 4000 --num-test 500 --use-scheduler \
#        --patience 2000 --use-tqdm --batch-size 1000 --unstacked --dyn-depth 5 --alg-width 100 --h 0.1 --N 80 \
#        --epochs 20000 --alg-type attention --dyn-type attention \
#        --dyn-weight 64.0 --lr 4e-6 --start-from-best 
#训练出网络的最佳参数
python example_powerNet.py --log-dir ./logs/dae-pinns-best/ --num-test 500 --use-scheduler --patience 2000 --batch-size 1048 \
	--unstacked --dyn-depth 4 --dyn-width 100 --h 0.1 --N 80 --dyn-type attention --alg-type attention --dyn-activation sin \
	--alg-activation sin --test-every 1000 --scheduler-type plateau --alg-weight 1.0 --num-train 6000 --num-val 100 \
	--use-tqdm --num-test 500 \
	--dyn-weight 64.0 --epochs 30000 --start-from-best --lr 1e-4 




	#--dyn-weight 1.0 --epochs 50000 
	#--dyn-weight 2.0 --epochs 20000 --start-from-best 
	#--dyn-weight 4.0 --epochs 20000 --start-from-best --lr 8e-4
	#--dyn-weight 8.0 --epochs 25000 --start-from-best --lr 5e-4
	#--dyn-weight 16.0 --epochs 25000 --start-from-best --lr 4e-4
	#--dyn-weight 32.0 --epochs 25000 --start-from-best --lr 3e-4
	#--dyn-weight 64.0 --epochs 30000 --start-from-best --lr 1e-4 --num-train from 4000 to 6000
	
