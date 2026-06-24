# version2 moderate-full 实验设计

## 目标

在不覆盖 `quick` 既有成果的前提下，建立一个自包含的 `version2` 实验工程，用更长仿真时域、更丰富工况和更完整分析来验证 Transformer-MPC 在复杂低附着/高风险双移线跟踪中的优势。

## 边界

- `quick` 作为冻结基线，不修改其中源码、数据、图表和报告。
- `version2` 复制 quick 的稳定原型结构，但所有新数据、模型、结果写入 `version2/data`、`version2/checkpoints`、`version2/outputs`。
- 本轮不重构动力学模型、MPC 求解器或 Transformer 网络主体，只扩展实验组织、场景集合、训练/运行脚本和分析图表的适配能力。
- 当前 `/Users/dsacbery/Study/code` 不是 git 仓库，因此不执行规格技能中要求的 git commit。

## 方案

采用“自包含复制 + 定向扩展”的方案：

1. 从 `quick` 复制源码、测试、脚本和说明文档到 `version2`，排除旧的 `data`、`checkpoints`、`outputs`、`result.docx` 和缓存文件。
2. 在 `version2/trans_mpc/scenario_manager.py` 中新增 version2 专用训练/评估场景函数，保留 quick 原有函数以支持烟测。
3. 将 `version2/scripts/generate_dataset.py`、`train_transformer.py`、`run_experiments.py` 调整为 moderate-full 默认配置，同时保留 `--quick`。
4. 将 `generate_trans_result_figures.py` 中硬编码的高风险 zoom 场景改为自动选择风险最强的 Transformer-MPC 场景，使 8-10 个评估工况下仍能稳定出图。
5. 运行完整链路：单测、数据集生成、Transformer 训练、控制器对比实验、优势分析、trans_result 图组。

## 场景设计

训练场景约 12 个，覆盖：

- 高附着基准
- 中低附着阶跃
- 双移线入口低附着
- 双移线中段低附着
- 高速 DLC
- 严重低附着
- 初始横向偏置
- 初始航向偏置
- 测量噪声
- 扰动加速度
- 低附着后恢复
- 高速偏置噪声组合

评估场景约 9 个，覆盖：

- `baseline_high_mu_full`
- `mild_friction_step_full`
- `severe_friction_step_full`
- `lane_change_low_mu_full`
- `high_speed_dlc_full`
- `offset_noise_full`
- `yaw_offset_disturbed_full`
- `low_mu_recovery_full`
- `extreme_high_speed_low_mu_full`

## 训练与运行配置

- quick 模式：保持约 3.5 s，用于烟测。
- moderate-full 默认：仿真时域约 13 s，覆盖完整双移线路径的主要过程。
- 数据集生成：full 模式使用 PID、Fixed MPC、Rule-risk MPC 三类闭环数据；quick 模式保持轻量。
- Transformer 训练：默认 epoch 从 quick 的 10 提升到 16，并增加 `--epochs` 覆盖参数，便于后续复现实验。
- 控制器比较：PID、Fixed MPC、Rule-risk MPC、Transformer-MPC。

## 验收标准

- `version2` 单测通过。
- `version2/data/dataset.npz` 和 `version2/data/scaler.pkl` 生成成功。
- `version2/checkpoints/best_transformer.pt` 和 `training_history.csv` 生成成功。
- `version2/outputs/logs/experiment_log.csv` 覆盖所有评估工况和四类控制器。
- `version2/outputs/tables/metrics_summary.csv`、`advantage_summary.csv`、`advantage_intervals.csv` 生成成功。
- `version2/outputs/figures/trans_result` 生成 7 张核心结果图。
- 最终摘要用中文说明实验规模、关键结果和未解决风险。

## 风险

- MPC 求解是主要耗时来源；moderate-full 控制在约 8-10 个评估场景和约 12 个训练场景。
- 更丰富数据可能改善风险预测，但不保证 Transformer-MPC 在每个单项指标上都优于所有基线；分析应继续强调局部高风险区间、稳定性和综合代价。
- 若极端低附着导致 MPC 不可行，保留现有求解失败回退机制并在指标中报告 `mpc_failures`。
