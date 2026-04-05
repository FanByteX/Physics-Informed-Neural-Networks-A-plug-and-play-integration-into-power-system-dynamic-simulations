## 参考论文：
Ignasi Ventura, Jochen Stiasny, Spyros Chatzivasileiadis. "Physics-Informed Neural Networks: a Plug and Play Integration into Power System Dynamic Simulations". 2024.
# 运行 main.py 时，可以通过以下命令行参数自定义仿真配置：
参数	            说明	                                           备注
--system	         研究的电力系统	                           目前支持 IEEE 9-bus
--machine	          使用 PINN 建模的发电机编号	             指定哪台机组由神经网络代替
--event_type	        故障/扰动类型	                        w_setpoint 或 p_setpoint
--event_location	    故障发生的地点	                              节点编号
--event_magnitude	     故障的幅值	                                   标幺值
--sim_time	             仿真总时长 	                              单位：秒
--time_step_size	       时间步长	                                  积分步长
--rk_scheme	               积分算法	                               trapezoidal 或 backward_euler
--compare_pure_RKscheme	   比较标志	                          对比“混合求解器”与“纯 RK 求解器”
--compare_ground_truth	   真实值对比	                            对比商业仿真软件的结果
--study_selection	      研究案例选择	                        对应论文中不同的 Figure 复现
--gpu	                   GPU 加速	                              是否使用 CUDA 推理
# 3. 支持的算法与场景
积分方案 (Runge-Kutta schemes):
trapezoidal: 梯形定则（隐式）。
backward_euler: 后向欧拉法（隐式）。
w_setpoint: 转子角速度相对变化Δω 的阶跃响应
p_setpoint: 机械输出功率P_m的阶跃响应
#  4. 核心代码结构
main.py: 项目入口，包含所有仿真的集成信息。
src/pinn_architecture.py: 定义全连接神经网络（PINN）的架构。
src/tds_dae_rk_schemes.py: 核心算法库，包含同时隐式算法及 PINN 的集成逻辑。
post_processing/: 用于处理仿真数据并生成可视化图表。
gt_simulations/: 存放传统电力系统仿真软件生成的“地面真值 (Ground Truth)”。
final_models/: 存放预训练好的 PINN 模型权重。
config_files/: 系统参数和动态元件配置。
# 5. 误差随时间演化图（示例说明）
在论文和示例图中，底部的“山丘状”阴影代表了仿真误差的演化过程：
## 橙色轨迹：20 种不同初始条件下，纯数值求解器产生的误差。
## 蓝色轨迹：相同初始条件下，集成 PINN 后的求解器产生的误差。
### 结论：蓝色阴影通常更低，证明了引入 PINN 后能有效降低复杂动态过程中的仿真误差。
# 6. 实验复现 (Reproducibility)
可以通过修改 --study_selection 参数复现论文中的图表：
全变量预览图: 设置 --study_selection = 1
图 2/4 (误差演化): 设置 --time_step_size = 8e-3 且 --study_selection = 2
大步长对比图: 设置 --time_step_size = 4e-2 且 --study_selection = 2
步长敏感度分析: 修改 --time_step_size 在 [1e-3:4e-2] 范围内，并设置 --study_selection = 3




## 1. 这个项目的核心逻辑是什么？（它在“替换”什么？）
在电力系统仿真（Power System Dynamic Simulations）中，工程师通常使用 Runge-Kutta（龙格-库塔）方法 来解微分代数方程组（DAEs）。

传统做法： 每走一步时间（time step），电脑都要用复杂的物理公式硬算一遍，这非常耗时。
PINNs 的做法： 如果我们可以用一个神经网络来“模拟”发电机（Generator）的行为，神经网络计算速度极快。
Plug-and-Play 的含义： 这个项目就像在传统仿真器里插了一个“插件”。当需要计算发电机部分的物理状态时，它不硬算了，而是问一下 .pth 文件里的神经网络：“基于现在的电压和电流，根据你学到的规律，下一个状态大概是多少？”
## 2. 为什么它是“Physics-Informed”（物理信息）的？
这是最关键的一点。

之前我们提到，普通的 AI 是通过数据（Input-Output）训练的。但在这个项目中，神经网络不仅看数据，还被“物理方程”锁住了。

普通 AI： 仅仅是拟合输入输出。如果遇到极端情况（比如电力系统故障），AI 可能会瞎猜。
PINNs： 它的损失函数（Loss Function）不仅包含“预测对不对”，还包含了“是否符合物理定律”。
它强制神经网络必须满足能量守恒、电磁方程等物理约束。
如果神经网络给出的预测违背了物理定律，Loss 就会变得极大，模型就不会被采纳。
这就回答了你之前的困惑：为什么 AI 没见过这个 x 也能输出精确的𝑦？
因为在这个项目中，它不仅依赖于过去的数据（经验），它还被植入了物理定律（真理）。即使遇到了没见过的x只要符合物理定律，它的预测就不会偏离轨道。
## 3. 如何理解这个项目里的 .pth 文件？
根据 README，你看到的 final_models 文件夹里的 .pth 文件，就是被“物理定律”训练过的发电机代理模型。

代码（pinn_architecture.py）： 定义了“弹珠台”的形状（神经网络的层数、神经元结构）。
.pth 文件（权重）： 里面存的不是随意的数字，而是经过物理定律打磨过的参数。
main.py 的作用： 它在跑仿真时，通过 pinn_boost 标志，把这个 .pth 文件加载进内存，实时计算电力系统的状态。
## 4. 为什么要这么做？（它的价值）
这个项目解决了一个很现实的痛点：

传统仿真太慢： 如果要模拟大规模电网的复杂动态，算起来可能要几分钟甚至几小时。
PINN 很快： 神经网络完成一次前向推理（Inference）只需毫秒级。
既要快，又要准： 这就是为什么要用 PINN。它用神经网络的速度，换取了接近物理仿真精度的结果。
总结给你的建议
如果你想在这个项目上动手：
不必重写仿真器： 你只需要理解它的 TDS_simulation 类是如何调用 .pth 文件的。
关注权重加载： 去看看 tds_dae_rk_schemes.py 中如何使用 torch.load 加载 final_models 里的 .pth。你会发现，这就是我们在最开始讨论的那个读取过程。
理解模型局限： 这类模型虽然好用，但它仅限于它所训练的“发电机模型”和“IEEE 9-bus 系统”。如果你把这个 .pth 文件直接放到另一个系统里，它大概率会输出“垃圾数据”，因为它的“物理经验”只适用于那个特定的系统环境。
你现在是不是对这些 .pth 文件不再觉得陌生了？ 它们其实就是这个电力仿真系统里的“物理规律计算器”。你想深入研究一下这个代码的哪一部分吗？或者想看看它具体怎么加载这些权重？