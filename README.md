# Transformer-MPC version2 moderate-full 实验

本目录是 `quick` 原型之后的自包含 version2 实验工程，用于在更长时域和更丰富工况下验证动力学自感知 Transformer-MPC 轨迹跟踪框架。

## 运行环境

优先使用项目已有虚拟环境：

```bash
/Users/dsacbery/Study/code/.venv/bin/python
```

说明：该环境没有安装 `pytest` 和 `python-docx`。核心测试使用 `unittest`；`.docx` 报告生成使用 Codex 工作区文档运行时。

## 一键式运行顺序

```bash
cd /Users/dsacbery/Study/code/TRANS/version2
/Users/dsacbery/Study/code/.venv/bin/python -m unittest discover -s tests -v
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_dataset.py
/Users/dsacbery/Study/code/.venv/bin/python scripts/train_transformer.py
/Users/dsacbery/Study/code/.venv/bin/python scripts/run_experiments.py
/Users/dsacbery/Study/code/.venv/bin/python scripts/analyze_advantage.py
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_trans_result_figures.py
```

quick 烟测仍然保留：

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_dataset.py --quick
/Users/dsacbery/Study/code/.venv/bin/python scripts/train_transformer.py --quick
/Users/dsacbery/Study/code/.venv/bin/python scripts/run_experiments.py --quick
```

## version2 默认配置

- 训练场景：12 个，覆盖高附着、低附着阶跃、换道低附着、高速、初始偏置、噪声、扰动和低附着恢复。
- 评估场景：9 个，覆盖 baseline、mild/severe friction step、lane-change low-mu、high-speed DLC、offset/noise、yaw offset disturbed、low-mu recovery、extreme high-speed low-mu。
- 仿真时域：13 s。
- 数据集控制器：PID、Fixed MPC、Rule-risk MPC。
- 对比控制器：PID、Fixed MPC、Rule-risk MPC、Transformer-MPC。
- Transformer 训练：默认 16 epochs，可用 `scripts/train_transformer.py --epochs N` 覆盖。

## 核心输出

```text
data/dataset.npz
data/scaler.pkl
checkpoints/best_transformer.pt
checkpoints/training_history.csv
outputs/logs/experiment_log.csv
outputs/tables/metrics_summary.csv
outputs/tables/advantage_summary.csv
outputs/tables/advantage_intervals.csv
outputs/figures/
outputs/figures/trans_result/
```

`outputs/figures/trans_result/version2_trans_result_analysis.docx` 是基于 7 张核心图生成的中文分析报告。当前机器缺少 `soffice`，因此该 DOCX 未完成页面渲染视觉 QA。

## 结果解读边界

version2 的主要结论应表述为阶段性验证：Transformer 风险输出已经接入 MPC 权重、约束和速度参考自适应调节，并在多数场景的综合代价上优于 Fixed MPC 和 Rule-risk MPC。与此同时，极端低附着和高速工况下仍存在路径完成率不足、横向误差过大和 MPC 回退次数偏高的问题，后续需要继续校准风险标签、速度衰减和可行性约束。
