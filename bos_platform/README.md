# bos_platform — BOS 接口层

本目录是 **BMAC 集成用的 stub 实现**，用于 toy 演示和接线测试，**不是**生产级 `bos-platform` 本体。

## 默认行为

```python
from bos_platform.loader import load_bos_platform

bundle = load_bos_platform()
signal = bundle.SignalAPI()
```

`bundle.source` 为 `in_tree_stubs` 时表示使用本目录 stub。

## 接入外部 bos-platform

```bash
./scripts/setup_bos_platform_path.sh
eval "$(./scripts/setup_bos_platform_path.sh --export)"
cd /Users/lei/Desktop/bos-bmac
PYTHONPATH=. python3 -c "from bos_platform.loader import load_bos_platform; print(load_bos_platform().source)"
```

外部目录需包含 `bos_platform/signal_control.py` 等模块（见 `adapters/README.md`）。

**注意**：`~/Projects/bos-platform` 是全栈 HTTP 服务，不能直接把 `BOS_PLATFORM_PATH` 指到其根目录；需 `bmac_adapter` 薄封装。

## 必需接口（Phase 0 spec）

- `SignalAPI` / `ControlAPI`: `publish`, `get`, `apply_control`, `publish_quorum`, `export_alpha_seq_to_real`
- `Kalman`: `update(x_meas, L=None)`, `get_covariance`
- `TemporalWorkflow`: `start_workflow`, `advance`, `checkpoint(wf, state=...)`
- `OPA`: `enforce_policy`, `check_policy`, `get_violation_count`, `evolve_policy_from_dt`
- `DigitalTwin`: `sample_robust_parameters`, `simulate_trajectory`, `refine_from_observations`

## 验证

```bash
cd /Users/lei/Desktop/bos-bmac
PYTHONPATH=. /Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/test_integration_wiring.py tests/test_bos_platform_loader.py -q
PYTHONPATH=. /Library/Developer/CommandLineTools/usr/bin/python3 examples/run_all.py
```

`real_wiring_example.py` 提供 Real* 骨架，用于 swap 测试。
