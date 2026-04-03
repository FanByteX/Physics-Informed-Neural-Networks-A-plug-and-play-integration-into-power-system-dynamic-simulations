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