建立基准

```
python scripts/run_regression.py --pattern MA5 --update-baseline
```

运行全部策略：

```
python scripts/run_regression.py
```

# Strategy Registry / 回归使用说明

此目录存放所有需要纳入“回测结果回归测试”的策略。通过执行 `scripts/run_regression.py` 脚本批量跑回测并与基准结果 (baseline) 对比，快速发现引擎或数据变动导致的偏差。

---

## 1. 策略文件格式

支持两种：

1. 含 `strategy_code` 字符串（聚宽 initialize/handle_data 风格）
2. 定义 `UserStrategy` (Backtrader Strategy) 类

推荐在文件头写一个 YAML 风格元数据块（全部以注释开头），指定运行参数：

```
# ---
# name: basic_buy_and_hold
# symbol: 000008
# start: 2018-01-01
# end: 2020-01-01
# cash: 100000
# benchmark: 000300
# frequency: daily
# adjust: auto
# ---
```

字段说明：

- name: 基准文件名（省略则用文件名）
- symbol: 标的代码（可带/不带交易所后缀）
- start / end: 回测区间（YYYY-MM-DD）
- cash: 初始资金
- benchmark: 基准代码
- frequency: daily / weekly / monthly / 1min
- adjust: raw / qfq / hfq / auto

示例策略变量：

```python
strategy_code = """
def initialize(context):
    set_benchmark('000300')
def handle_data(context, data):
    pos = context.portfolio.positions.get('000008')
    if pos is None or getattr(pos, 'closeable_amount', 0) == 0:
        order_value('000008', 10000)
"""
```

---

## 2. 基准结果 (Baseline)

首次运行用 `--update-baseline` 将当前回测结果的核心指标与 equity 哈希写入 `baselines/<name>.json`。后续每次比较当前结果与 baseline 是否一致。

基准文件包含的核心字段：

- final_value
- pnl_pct
- max_drawdown
- total_trades
- win_rate
- benchmark_return
- excess_return
- use_real_price
- data_variant (选用的复权/数据变体)
- equity_hash (净值曲线 JSON 序列化后 MD5)

任意字段或 equity_hash 变化都会被标记为 DIFF。

---

## 3. 常用命令

所有命令在项目根目录执行 (PowerShell)：

创建/更新单个策略基准：

```
python scripts/run_regression.py --pattern MA5 --update-baseline
```

检查单策略是否发生变化（不更新基准）：

```
python scripts/run_regression.py --pattern MA5
```

运行全部策略：

```
python scripts/run_regression.py
```

列出已发现的策略及其运行配置：

```
python scripts/run_regression.py --list
```

首创基准后期望 SUMMARY 样例：

```
MA5: CREATED
Total=1 OK=0 DIFF=0 CREATED=1 NO_BASELINE=0
```

后续无变化：

```
MA5: OK
Total=1 OK=1 DIFF=0 CREATED=0 NO_BASELINE=0
```

出现变化（示例）：

```
== Running MA5 ==
  [DIFF] MA5 有 2 项差异:
    - final_value: baseline=123000.0 current=122500.0
    - equity_hash: baseline=abcd1234 current=ef567890
...
MA5: DIFF -> final_value,equity_hash
```

在 CI/自动化希望变动即失败：

```
python scripts/run_regression.py --fail-on-diff
```

确认变化是“预期”后更新基准：

```
python scripts/run_regression.py --update-baseline
```

只匹配部分名称（支持正则/子串）：

```
python scripts/run_regression.py --pattern ".*MA.*"
```

---

## 4. 如何判断策略结果是“对的”？

使用以下层次：

1. 稳定性：与上一次已确认正确的 baseline 指标完全一致（OK 状态）。
2. 关键指标：若存在轻微数值波动，检查是否由于：
   - 数据文件新增/修订
   - 调整撮合、手续费、滑点逻辑
   - 调整复权 / use_real_price 选择规则
3. 净值曲线哈希 (equity_hash)：
   - 变化意味着曲线结构或顺序发生变化；需核对日志、成交、数据源文件。
4. 辅助核查：
   - 查看 `jq_logs`（或策略里输出的[fill]/[fee]/[limit_check]）定位产生差异的具体日期。
   - 用相同参数单独运行 `smoke_test` 或写临时脚本调用 `run_backtest` 比对 equity_curve 的前后差异（行数、首末值）。
5. 预期变更判断：
   - 如果本次代码改动的目的正是调整撮合/费用/数据选择，则更新 baseline。
   - 若改动与交易逻辑/数据无关（例如仅日志或文档），出现 DIFF 代表引擎出现副作用，应先排查。

快速排查步骤：

```
1) 发现 DIFF -> 记录 diff_keys
2) 若包含 equity_hash -> 打印当前 run 的 first/last 5 个 equity_curve 点
3) 对比 jq_logs 中首次出现交易或费用差异的日期
4) 查该日期的数据文件是否被改动（git diff / 文件时间戳）
5) 确认原因合理后 --update-baseline
```

---

## 5. 新增策略流程

1. 在 `strategies/` 新建 `XXX.py`
2. 写元数据头；填好 `strategy_code` 或 `UserStrategy`
3. 运行：`python scripts/run_regression.py --pattern XXX --update-baseline`
4. 确认 SUMMARY 中出现 `CREATED`
5. 后续回归按普通策略处理

---

## 6. 常见问题 (FAQ)

| 问题                      | 说明与处理                                                                  |
| ------------------------- | --------------------------------------------------------------------------- | ------- | --- | --- | --------------- |
| SUMMARY 显示 NO_BASELINE  | 没有加 `--update-baseline`，或首次创建前脚本旧版本；重新加参数跑一次。      |
| 只有 equity_hash 变化     | 说明曲线有差别；检查是否撮合/费用/数据文件改动；再决定是否更新基准。        |
| 指标正常但 equity_hash 变 | 可能末尾浮点舍入差异导致曲线一点变动；确认日志中成交/数据无异常后可以更新。 |
| 频率想临时覆盖            | 暂未提供 CLI 覆盖；修改文件头 frequency or 复制一份新策略。                 |
| 想加浮点容差              | 可扩展脚本 diff_metrics：对特定字段设置阈值（如                             | new-old | /   | old | < 1e-4 忽略）。 |

---

## 7. 后续可增强（待选）

- 支持 `--rtol`/`--atol` 浮点容差
- 并行执行（多策略加速）
- 输出 equity 差异的首个偏离点
- 产出 HTML 报告 / 发送通知

---

## 8. 目录结构回顾

- `strategies/` 策略源文件
- `baselines/` 基准 JSON（自动创建）
- `scripts/run_regression.py` 回归驱动脚本

---

## 9. 快速参考 (Cheat Sheet)

| 目的             | 命令                                                               |
| ---------------- | ------------------------------------------------------------------ |
| 创建单策略基准   | `python scripts/run_regression.py --pattern MA5 --update-baseline` |
| 检查是否变动     | `python scripts/run_regression.py --pattern MA5`                   |
| 全量回归         | `python scripts/run_regression.py`                                 |
| 列出策略         | `python scripts/run_regression.py --list`                          |
| 有差异时失败     | `python scripts/run_regression.py --fail-on-diff`                  |
| 同时更新全部基准 | `python scripts/run_regression.py --update-baseline`               |

---

Baseline 结果 JSON 存放在 `baselines/` 目录，文件名 = `name.json`。
