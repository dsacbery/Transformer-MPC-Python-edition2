# version2 moderate-full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `/Users/dsacbery/Study/code/TRANS/version2` 中建立并运行自包含的 moderate-full Transformer-MPC 实验链路。

**Architecture:** 复制 quick 的稳定 Python 原型作为 version2 基底，排除旧数据和输出；仅扩展场景、脚本默认配置、任意场景数量下的图表生成能力。核心动力学、MPC 求解器、Transformer 网络结构保持不变，减少变量。

**Tech Stack:** Python, NumPy, pandas, PyTorch, cvxpy/OSQP, matplotlib, unittest, joblib。

---

### Task 1: 复制 quick 工程骨架

**Files:**
- Create/Modify: `/Users/dsacbery/Study/code/TRANS/version2/trans_mpc/*`
- Create/Modify: `/Users/dsacbery/Study/code/TRANS/version2/scripts/*`
- Create/Modify: `/Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py`
- Preserve: `/Users/dsacbery/Study/code/TRANS/version2/docs/superpowers/*`

- [ ] **Step 1: 复制源码和脚本，排除旧成果**

Run:

```bash
rsync -a \
  --exclude data \
  --exclude checkpoints \
  --exclude outputs \
  --exclude result.docx \
  --exclude __pycache__ \
  --exclude .DS_Store \
  /Users/dsacbery/Study/code/TRANS/quick/ \
  /Users/dsacbery/Study/code/TRANS/version2/
```

Expected: `version2` 中出现 `trans_mpc`、`scripts`、`tests`、`README.md`，但不出现 quick 的旧数据/输出。

- [ ] **Step 2: 运行复制后基线测试**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest discover -s /Users/dsacbery/Study/code/TRANS/version2/tests -v
```

Expected: 10 tests pass。

### Task 2: 增加 version2 场景集合

**Files:**
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/trans_mpc/scenario_manager.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py`

- [ ] **Step 1: 写 failing test，要求 version2 场景更丰富**

Add to `tests/test_core.py`:

```python
    def test_version2_scenarios_cover_richer_conditions(self):
        from trans_mpc.scenario_manager import make_version2_evaluation_scenarios, make_version2_training_scenarios

        train = make_version2_training_scenarios()
        eval_scenarios = make_version2_evaluation_scenarios()
        self.assertGreaterEqual(len(train), 10)
        self.assertGreaterEqual(len(eval_scenarios), 8)
        self.assertLessEqual(len(eval_scenarios), 10)
        self.assertEqual(len({s.name for s in train}), len(train))
        self.assertEqual(len({s.name for s in eval_scenarios}), len(eval_scenarios))
        self.assertTrue(any(s.speed >= 17.0 for s in eval_scenarios))
        self.assertTrue(any(s.mu_low <= 0.24 for s in eval_scenarios))
        self.assertTrue(any(s.noise_std > 0.0 for s in eval_scenarios))
        self.assertTrue(any(s.disturbed for s in eval_scenarios))
        self.assertTrue(any(s.initial_y != 0.0 or s.initial_yaw != 0.0 for s in eval_scenarios))
```

- [ ] **Step 2: 验证 RED**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest /Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py -v
```

Expected: FAIL because `make_version2_evaluation_scenarios` is not defined。

- [ ] **Step 3: 实现 version2 场景函数**

Add to `scenario_manager.py`:

```python
def make_version2_training_scenarios() -> list[Scenario]:
    return [
        Scenario(name="v2_train_high_mu_10", speed=10.0, mu_high=0.95, mu_low=0.95),
        Scenario(name="v2_train_high_mu_12", speed=12.0, mu_high=0.90, mu_low=0.90),
        Scenario(name="v2_train_step_mid_12", speed=12.0, mu_high=0.90, mu_low=0.36, low_mu_start=34.0, low_mu_end=105.0),
        Scenario(name="v2_train_step_low_14", speed=14.0, mu_high=0.85, mu_low=0.26, low_mu_start=32.0, low_mu_end=108.0),
        Scenario(name="v2_train_lane_entry_low_13", speed=13.0, mu_high=0.90, mu_low=0.28, low_mu_start=22.0, low_mu_end=70.0),
        Scenario(name="v2_train_lane_mid_low_15", speed=15.0, mu_high=0.80, mu_low=0.30, low_mu_start=38.0, low_mu_end=88.0),
        Scenario(name="v2_train_high_speed_17", speed=17.0, mu_high=0.68, mu_low=0.34, low_mu_start=35.0, low_mu_end=95.0),
        Scenario(name="v2_train_offset_y", speed=12.0, mu_high=0.78, mu_low=0.32, low_mu_start=36.0, low_mu_end=94.0, initial_y=0.7),
        Scenario(name="v2_train_offset_yaw", speed=12.0, mu_high=0.82, mu_low=0.34, low_mu_start=36.0, low_mu_end=92.0, initial_yaw=0.035),
        Scenario(name="v2_train_noisy", speed=13.0, mu_high=0.78, mu_low=0.30, low_mu_start=35.0, low_mu_end=92.0, noise_std=0.012),
        Scenario(name="v2_train_disturbed", speed=14.0, mu_high=0.74, mu_low=0.29, low_mu_start=34.0, low_mu_end=90.0, disturbed=True),
        Scenario(name="v2_train_recovery", speed=13.0, mu_high=0.88, mu_low=0.27, low_mu_start=28.0, low_mu_end=62.0),
    ]


def make_version2_evaluation_scenarios() -> list[Scenario]:
    return [
        Scenario(name="baseline_high_mu_full", speed=12.0, mu_high=0.92, mu_low=0.92),
        Scenario(name="mild_friction_step_full", speed=13.0, mu_high=0.90, mu_low=0.38, low_mu_start=36.0, low_mu_end=115.0),
        Scenario(name="severe_friction_step_full", speed=14.0, mu_high=0.88, mu_low=0.24, low_mu_start=34.0, low_mu_end=115.0),
        Scenario(name="lane_change_low_mu_full", speed=13.5, mu_high=0.88, mu_low=0.26, low_mu_start=23.0, low_mu_end=74.0),
        Scenario(name="high_speed_dlc_full", speed=17.0, mu_high=0.66, mu_low=0.33, low_mu_start=34.0, low_mu_end=96.0),
        Scenario(name="offset_noise_full", speed=12.5, mu_high=0.76, mu_low=0.31, low_mu_start=35.0, low_mu_end=93.0, initial_y=-0.75, noise_std=0.012),
        Scenario(name="yaw_offset_disturbed_full", speed=13.0, mu_high=0.78, mu_low=0.30, low_mu_start=34.0, low_mu_end=92.0, initial_yaw=0.04, disturbed=True),
        Scenario(name="low_mu_recovery_full", speed=14.0, mu_high=0.90, mu_low=0.27, low_mu_start=26.0, low_mu_end=64.0),
        Scenario(name="extreme_high_speed_low_mu_full", speed=18.0, mu_high=0.62, mu_low=0.23, low_mu_start=32.0, low_mu_end=90.0, initial_y=0.35, noise_std=0.008),
    ]
```

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest /Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py -v
```

Expected: all tests pass。

### Task 3: 调整脚本默认 full 配置

**Files:**
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/trans_mpc/config.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/scripts/generate_dataset.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/scripts/train_transformer.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/scripts/run_experiments.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py`

- [ ] **Step 1: 写 failing test，约束 version2 训练 epoch 和场景导入**

Add to `tests/test_core.py`:

```python
    def test_version2_training_defaults_are_moderate_full(self):
        from trans_mpc.config import TrainingConfig
        from trans_mpc.scenario_manager import make_version2_evaluation_scenarios, make_version2_training_scenarios

        self.assertGreaterEqual(TrainingConfig().epochs, 16)
        self.assertGreaterEqual(len(make_version2_training_scenarios()), len(make_version2_evaluation_scenarios()))
```

- [ ] **Step 2: 验证 RED**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest /Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py -v
```

Expected: FAIL because `TrainingConfig().epochs` is still 10。

- [ ] **Step 3: 修改配置和脚本**

Required behavior:

```text
generate_dataset.py --quick -> 3.5 s, quick training scenarios, PID only
generate_dataset.py          -> 13.0 s, version2 training scenarios, PID + Fixed MPC + Rule-risk MPC
train_transformer.py --quick -> quick_epochs
train_transformer.py          -> TrainingConfig.epochs
train_transformer.py --epochs N -> N epochs
run_experiments.py --quick   -> first two legacy scenarios, 3.5 s
run_experiments.py           -> 13.0 s, version2 evaluation scenarios
```

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest discover -s /Users/dsacbery/Study/code/TRANS/version2/tests -v
```

Expected: all tests pass。

### Task 4: 让 trans_result 图组适配任意 version2 场景

**Files:**
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/scripts/generate_trans_result_figures.py`
- Modify: `/Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py`

- [ ] **Step 1: 写 failing test，要求 zoom 场景自动选择**

Add to `tests/test_core.py`:

```python
    def test_select_zoom_scenario_prefers_high_transformer_risk(self):
        import pandas as pd
        from scripts.generate_trans_result_figures import select_zoom_scenario

        df = pd.DataFrame(
            [
                {"scenario": "low", "controller": "Transformer-MPC", "time": 0.0, "r_low": 0.2, "r_stab": 0.1},
                {"scenario": "high", "controller": "Transformer-MPC", "time": 0.0, "r_low": 0.7, "r_stab": 0.8},
                {"scenario": "high", "controller": "PID", "time": 0.0, "r_low": 0.0, "r_stab": 0.0},
            ]
        )
        self.assertEqual(select_zoom_scenario(df), "high")
```

- [ ] **Step 2: 验证 RED**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest /Users/dsacbery/Study/code/TRANS/version2/tests/test_core.py -v
```

Expected: FAIL because `select_zoom_scenario` is not defined。

- [ ] **Step 3: 实现 `select_zoom_scenario` 并替换 hard-coded `friction_step`**

Required helper:

```python
def select_zoom_scenario(log_df: pd.DataFrame) -> str:
    transformer = log_df[log_df["controller"] == "Transformer-MPC"].copy()
    if transformer.empty:
        return str(sorted(log_df["scenario"].unique())[0])
    transformer["risk_score"] = transformer["r_low"].abs() + transformer["r_stab"].abs()
    scores = transformer.groupby("scenario")["risk_score"].max().sort_values(ascending=False)
    return str(scores.index[0])
```

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest discover -s /Users/dsacbery/Study/code/TRANS/version2/tests -v
```

Expected: all tests pass。

### Task 5: 运行 moderate-full 实验链路

**Files:**
- Generate: `/Users/dsacbery/Study/code/TRANS/version2/data/dataset.npz`
- Generate: `/Users/dsacbery/Study/code/TRANS/version2/checkpoints/best_transformer.pt`
- Generate: `/Users/dsacbery/Study/code/TRANS/version2/outputs/*`

- [ ] **Step 1: 生成数据集**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_dataset.py
```

Expected: prints generated scenario/controller rows and saves dataset with more windows than quick。

- [ ] **Step 2: 训练 Transformer**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/train_transformer.py
```

Expected: 16 epochs complete and `checkpoints/best_transformer.pt` exists。

- [ ] **Step 3: 运行控制器对比**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/run_experiments.py
```

Expected: 9 scenarios x 4 controllers logged, outputs saved under `version2/outputs`。

- [ ] **Step 4: 运行优势分析**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/analyze_advantage.py
```

Expected: `advantage_summary.csv` and `advantage_intervals.csv` generated。

- [ ] **Step 5: 生成 trans_result 图组**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python scripts/generate_trans_result_figures.py
```

Expected: `outputs/figures/trans_result/01_*.png` through `07_*.png` exist。

### Task 6: 最终验证和中文摘要

**Files:**
- Read: `/Users/dsacbery/Study/code/TRANS/version2/outputs/tables/metrics_summary.csv`
- Read: `/Users/dsacbery/Study/code/TRANS/version2/outputs/tables/advantage_summary.csv`

- [ ] **Step 1: 运行最终单测**

Run:

```bash
/Users/dsacbery/Study/code/.venv/bin/python -m unittest discover -s /Users/dsacbery/Study/code/TRANS/version2/tests -v
```

Expected: all tests pass。

- [ ] **Step 2: 检查核心产物**

Run:

```bash
find /Users/dsacbery/Study/code/TRANS/version2 -maxdepth 3 -type f | sort
```

Expected: data, checkpoints, outputs/tables, outputs/logs, outputs/figures all have generated files。

- [ ] **Step 3: 汇总结果**

Report in Chinese:

```text
- 实验规模：训练场景数、评估场景数、数据集窗口数、训练 epoch。
- 核心产物路径：dataset/checkpoint/log/tables/figures。
- 关键指标：Transformer-MPC 相对 Fixed MPC 和 Rule-risk MPC 的综合代价优势场景。
- 风险：若个别场景不完成或 MPC 失败次数偏高，明确说明。
```
