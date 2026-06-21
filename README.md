# BOS-BMAC

将肖方舟 ROP（Reaction Order Polyhedron）+ FEC（Flux Exponent Control）形式化映射到 BOS 平台的生物分子多智能体控制研究原型。

**定位（诚实说明）**：这是 **研究/demo 原型**，有严格 toy gate，**不是**生产级全栈。`bos_platform/` 默认为 in-tree stub；真实 `bos-platform` 仓库需 adapter 才能接入。

Phase 0 规范：`docs/BOS-BMAC_Phase0_Spec_v1.0.{tex,md}`

## 唯一主线

```
/Users/lei/Desktop/bos-bmac
```

桌面重复快照已归档到 `~/Desktop/_bos_bmac_archive/`（勿并行维护）。

## 快速开始

推荐 Python（本机已验证）：

```bash
PY=/Library/Developer/CommandLineTools/usr/bin/python3
cd /Users/lei/Desktop/bos-bmac

# 全量 gate（pytest + 所有 demo）
PYTHONPATH=. $PY examples/run_all.py

# 仅单元测试
PYTHONPATH=. $PY -m pytest tests/ -q --tb=short
```

## 环境

| 组件 | 状态 |
|------|------|
| Python 3.9+ (CLT) | 必需 |
| numpy, scipy | 已用 |
| casadi + IPOPT | 可选，toy 上 ~100% 成功；失败时 scipy H-rep 后备 |
| matplotlib | 可选，用于图表 |

```bash
PY=/Library/Developer/CommandLineTools/usr/bin/python3
$PY -m pip install --user matplotlib
```

## 项目结构

| 路径 | 作用 |
|------|------|
| `bmac_engine/` | ROP、FEC、robust、multicell、BOS 接线 |
| `bos_platform/` | BOS 接口 stub + `loader.py` |
| `examples/` | demo、`run_all.py`、`correspondence_verification.py` |
| `tests/` | TDD + 集成不变量 + benchmark gate |
| `scripts/` | 桌面归档、`setup_bos_platform_path.sh` |

## 接入外部 bos-platform（BOS_PLATFORM_PATH）

### 1. 一键配置

```bash
chmod +x scripts/setup_bos_platform_path.sh
./scripts/setup_bos_platform_path.sh
eval "$(./scripts/setup_bos_platform_path.sh --export)"
```

### 2. 手动指定

```bash
# CI/开发：内置 external fixture
export BOS_PLATFORM_PATH=/Users/lei/Desktop/bos-bmac/tests/fixtures/external_bos_platform

PYTHONPATH=. $PY -c "from bos_platform.loader import load_bos_platform; print(load_bos_platform().source)"
# 期望: external:...
```

### 3. bos-platform adapter（已就绪）

已克隆：`~/Projects/bos-platform`，adapter 位于 `~/Projects/bos-platform/bmac_adapter/`。

```bash
export BMAC_HOME=/Users/lei/Desktop/bos-bmac
./scripts/setup_bos_platform_path.sh   # 自动选择 bmac_adapter
eval "$(./scripts/setup_bos_platform_path.sh --export)"
```

- **delegate 模式**（默认）：离线委托给 `BMAC_HOME` 的 stub，集成测试全绿
- **http 模式**（可选）：`BOS_API_BASE_URL` + `BOS_API_TOKEN`；发酵域 API 与 BMAC toy 语义尚未完全映射。当前 HTTP 路径实现为 **“真实端点 smoke-call + 本地 delegate 后备”**。

无 Docker 时可用 mock core 验证 HTTP 端到端（不假绿）：

```bash
PY=/Library/Developer/CommandLineTools/usr/bin/python3
PYTHONPATH=. $PY examples/run_mock_core_and_adapter_demo.py
```

```bash
PYTHONPATH=. $PY -m pytest tests/test_bmac_adapter.py tests/test_integration_wiring.py -q
PYTHONPATH=. $PY examples/run_all.py
```

详见 `~/Projects/bos-platform/bmac_adapter/README.md` 与 `bos_platform/adapters/README.md`。

## 可靠性硬化（2026-06-05）

已修复并加 gate 的问题：

- `robust_fec_alpha`：交替投影，保证多 scenario ROP 可行（移除 median 回归）
- multicell DT quorum：`ndarray` 不再触发 ambiguous truth value
- `bos_platform` 导入：`from bos_platform import DigitalTwin` 在 `PYTHONPATH=.` 下可用
- Temporal checkpoint：传入 workflow dict，记录 `x_meas/hat_x/f_star`
- `run_all.py`：失败即 `exit 1`，不再 `|| echo` 掩盖
- `correspondence_verification.py`：关键路径硬失败，去掉 `or True`
- `benchmarks.py`：FBA/MM vs ROP，`red_fba_vs_rop > 1.0` 为必过 gate
- `fec_solver`：长仿真数值裁剪，消除 overflow RuntimeWarning

### 测试覆盖

| 测试文件 | 验证内容 |
|----------|----------|
| `test_integration_wiring.py` | DT 导入、Temporal checkpoint、workflow ROP 可行 |
| `test_benchmarks.py` | FBA/MM benchmark gate |
| `test_bos_platform_loader.py` | 默认 stub + `BOS_PLATFORM_PATH` external |
| `test_fec_numerical_stability.py` | 200 步 multicell 无 overflow |
| `test_robust_extension.py` | robust 全样本可行回归 |

### 已知限制（不作 gate）

- toy 上 DT refine `improvement_pct` 可能为负
- scenario vs FBA 改善百分比可能为负（仅打印）
- CasADi 病态输入可能 fallback（scipy 后备可靠）
- 真实 `bos-platform` 需 REST adapter，尚未实现

## 实现状态 vs 规范

| 模块 | 状态 | 文件 |
|------|------|------|
| ROP H-rep | 完成 | `bmac_engine/rop_polyhedron.py` |
| FEC（显式 H-rep + CasADi） | 完成 | `bmac_engine/fec_solver.py` |
| BOS 接线 | 完成 | `bmac_engine/bos_integration.py` |
| Robust 扩展 | 完成 | `bmac_engine/robust_extension.py` |
| Multicell | 完成 | `bmac_engine/multicell_agent.py` |
| Benchmarks | 完成 | `bmac_engine/benchmarks.py` |
| bos_platform stub | 完成 | `bos_platform/*.py` |
| 外部 loader | 完成 | `bos_platform/loader.py` |

详细演进记录见 `BOS-BMAC_Phase0_Impl_Status.txt`。

## 常用命令

```bash
PY=/Library/Developer/CommandLineTools/usr/bin/python3
cd /Users/lei/Desktop/bos-bmac

PYTHONPATH=. $PY examples/bos_glue_example.py
PYTHONPATH=. $PY examples/phase1_digital_twin_robust_demo.py
PYTHONPATH=. $PY examples/correspondence_verification.py
PYTHONPATH=. $PY examples/end_to_end_toy.py

# 归档桌面重复快照（一次性）
./scripts/archive_desktop_snapshots.sh
```

编码时请对照 Phase 0 规范中的伪代码、矩阵维度和 CasADi `subject_to(A[i]@alpha <= b[i])` 注释。
