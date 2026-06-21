# 外部 bos-platform 接入说明

## 现状

已克隆的真实仓库位于：

`~/Projects/bos-platform`

它是 **FastAPI + 前端 + 部署** 的全栈项目，**没有**可直接 `import` 的 `bos_platform.SignalAPI` Python 包。因此 `BOS_PLATFORM_PATH` 不能指向该仓库根目录。

## BOS_PLATFORM_PATH 支持的布局

`loader.py` 会在以下路径查找 `bos_platform` 模块文件：

```
<path>/bos_platform/signal_control.py
<path>/bos_platform/kalman.py
<path>/bos_platform/temporal.py
<path>/bos_platform/opa.py
<path>/bos_platform/digital_twin.py
```

或扁平布局（`<path>/signal_control.py` 等）。

## 推荐做法

1. **bos-platform adapter（推荐）**：已实现在 `~/Projects/bos-platform/bmac_adapter/`
   ```bash
   export BMAC_HOME=/Users/lei/Desktop/bos-bmac
   export BOS_PLATFORM_PATH=~/Projects/bos-platform/bmac_adapter
   ```
   默认 **delegate** 模式：从 `BMAC_HOME/bos_platform/` 加载 stub；设 `BOS_API_BASE_URL` 可启用 HTTP 审计骨架。

2. **一键配置**：
   ```bash
   ./scripts/setup_bos_platform_path.sh   # 优先选 bmac_adapter
   eval "$(./scripts/setup_bos_platform_path.sh --export)"
   ```

3. **CI fixture**（无 Projects 克隆时）：
   ```bash
   export BOS_PLATFORM_PATH=/Users/lei/Desktop/bos-bmac/tests/fixtures/external_bos_platform
   ```

## 验证

```bash
cd /Users/lei/Desktop/bos-bmac
eval "$(./scripts/setup_bos_platform_path.sh --export)"
PYTHONPATH=. /Library/Developer/CommandLineTools/usr/bin/python3 -m pytest \
  tests/test_bos_platform_loader.py tests/test_integration_wiring.py -q
```

`load_bos_platform().source` 应以 `external:` 开头（fixture 或你的 adapter）。
