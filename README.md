# 本地类聚宽回测平台 (基于 backtrader + FastAPI)

uvicorn backend.app.main:app --reload
http://127.0.0.1:8000/

## 目标

搭建一个轻量本地回测系统：在浏览器中粘贴/编写策略代码，设置时间区间、初始资金、标的等参数，点击运行后获得回测结果（收益曲线、核心指标、交易明细），类似聚宽的基本交互。

## 架构概览

```
Browser (HTML/JS + Chart)
   |
   |  HTTP (JSON)
   v
FastAPI Backend  ------------------------------
  /api/backtest  (POST)  -> 调用 backtest_engine.run_backtest()
                                  |
                                  v
                           Backtrader (Cerebro)
                                  |
                           动态加载用户策略 (exec 安全子命名空间)
                                  |
                           读取本地CSV数据  data/<symbol>.csv
                                  |
                           分析器 analyzers (Sharpe, DrawDown, Returns, TradeAnalyzer)
```

## 关键模块

- `backend/app/backtest_engine.py` 封装回测逻辑：
  - 动态编译用户提交的策略代码，要求定义 `UserStrategy(bt.Strategy)`。
  - 加载指定标的数据 (当前版本：从 `data/<symbol>.csv` 读取，CSV 需包含 `datetime,open,high,low,close,volume` 列)。
  - 运行 analyzers 收集指标：总收益、年化、夏普、最大回撤、胜率、交易次数等。
  - 使用 `TimeReturn` 构建每日收益 & 权益曲线。
  - 返回统一 JSON：`metrics`, `equity_curve`, `daily_returns`, `trades`, `log`。
- `backend/app/main.py` FastAPI 入口：
  - `POST /api/backtest` 接收参数与策略代码。
  - 提供静态文件与 `index.html` 前端页面。
- `frontend/index.html` 简单单页：
  - 表单输入：标的、开始/结束日期、初始资金、策略代码。
  - 提交后以 `fetch` 调用后端，渲染指标 + 图表。

## 数据格式

示例数据文件：`data/000001.csv` or `data/sample.csv`

```
datetime,open,high,low,close,volume
2025-02-03,10.2,10.5,10.1,10.3,123456
...
```

日期需为可被 `pd.to_datetime` 识别的字符串。

### （新增）统一数据加载与 `stockdata/` 目录支持

当前版本已将底层数据访问抽象到 `backend/app/data_loader.py`，支持两套目录结构：

1. 旧版：`data/` 平铺文件（仍然兼容）
2. 新版：`stockdata/stockdata/` 分层目录结构：

- 分钟线：`stockdata/stockdata/1min/sz000001.csv`
- 多周期：`stockdata/stockdata/1d_1w_1m/<code>/<code>_daily[_qfq|_hfq].csv` 及 `weekly/monthly` 文件
- 指标：`stockdata/stockdata/indicator/000001.SZ.csv`

回测引擎原先使用的 `load_csv_dataframe` / `load_csv_data` 已重定向到统一加载器，默认优先使用新版目录（存在则用），否则回退旧目录。这样可以逐步迁移数据而不影响历史脚本。

#### 环境变量控制

| 变量                      | 缺省                               | 说明                                                                      |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| `PREFER_STOCKDATA`        | `1`                                | 为 `1` 时优先在新版 `stockdata/` 中查找；设为 `0` 强制只用旧 `data/` 目录 |
| `ADJUST_TYPE`             | `auto`                             | 对应之前的 `adjust_type` 逻辑：`auto/raw/qfq/hfq`                         |
| `BACKTEST_DATA_ROOT`      | `项目根/data`                      | 指定旧版数据目录根路径                                                    |
| `BACKTEST_STOCKDATA_ROOT` | 自动探测                           | 指定新版 `stockdata/stockdata` 根路径                                     |
| `BACKTEST_BENCHMARK_ROOT` | `stockdata/stockdata/基准指数数据` | 指定基准指数数据目录（含 000300\_日.csv 等）；未设时自动扫描该内置目录    |

示例（Windows PowerShell）：

```powershell
$env:PREFER_STOCKDATA = '1'
$env:ADJUST_TYPE = 'qfq'
python scripts/smoke_test.py
```

#### 直接调用新加载器（可选）

```python
from backend.app import data_loader as dl
df = dl.load_price_dataframe('600008.XSHG', '2019-01-01', '2020-01-01', frequency='daily', adjust='auto')
feed = dl.load_bt_feed('600008.XSHG', '2019-01-01', '2020-01-01')
```

#### 支持的频率

- `daily` / `weekly` / `monthly` （来自 `1d_1w_1m/<code>/`）

  > 注意：已移除“自动补齐缺失交易日并前值填充”功能，现在只使用数据文件中真实存在的交易日行；如果源数据缺某些本应存在的交易日，该日将完全跳过（不会再出现 gap_fill 标记）。

- `1min` （来自 `1min/`）

> 目前回测主流程仍按日线节奏运行；分钟线接口可用于后续扩展到日内或聚合再回测。

#### 自动列名映射

加载器会统一将中文/变体列映射为：`datetime, open, high, low, close, volume, amount, code`；若缺少 `volume` 会自动补零以满足 backtrader 要求。

#### 代码规范化

输入代码可接受：`600008.XSHG`, `600008`, `000001.SZ` 等形式；内部统一拆分为六位代码 + 交易所，以匹配文件命名（分钟线需根据首位数字猜测交易所时：6=沪，其它=深）。

---

若需要将策略代码层面也暴露“频率”或“复权模式”选择（例如前端新增下拉框），只需在调用 `run_backtest` 前设置对应环境变量或扩展 `run_backtest` 参数并透传到 `data_loader`。

## 示例策略 (双均线)

```python
import backtrader as bt

class UserStrategy(bt.Strategy):
    params = dict(fast=5, slow=20)
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data, period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(self.data, period=self.p.slow)
    def next(self):
        if not self.position:
            if self.sma_fast[0] > self.sma_slow[0]:
                self.buy()
        else:
            if self.sma_fast[0] < self.sma_slow[0]:
                self.sell()
```

## 安全注意

当前版本直接用 `exec` 执行用户代码，仅用于本地研究。生产化需：

- 代码沙箱 / 子进程隔离
- 资源/时间限制
- 白名单模块导入控制

## 后续规划 (见 README 末尾 TODO)

- 多进程任务队列（Celery/RQ）
- 数据适配（Tushare / 聚宽数据接口等）
- 参数优化、Walk-Forward、组合回测、实时模拟
- 结果持久化与多回测比较

---

## 快速开始

1. (可选) 创建虚拟环境
2. 安装依赖
3. 准备数据: 将 CSV 放在 `data/` 下, 命名 `<symbol>.csv`；支持中文列名：`日期,开盘,最高,最低,收盘,成交量` (内部会自动映射)。
4. 启动 FastAPI 服务
5. 打开浏览器访问 http://127.0.0.1:8000/

Windows PowerShell 示例:

```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

页面左边填写:

- symbol: 如 `000516_daily` (对应文件 `data/000516_daily.csv`)

# Web Backtest Demo

## 对齐聚宽指标口径（Sharpe/Alpha）

为使本地回测的夏普比率与阿尔法尽量贴近聚宽（JoinQuant）的展示口径，后端已做如下处理：

- 阿尔法/贝塔：采用日度收益对齐基准后做线性回归，再按 Jensen α 进行 Rf 校正；阿尔法默认按年化（× 年化因子）输出。
  无风险利率：默认 4% 年化（0.04），可通过 set_option('risk_free_rate', 0.04) 调整。
  夏普计算：默认使用“CAGR 年化口径”（Rp 年化与波动年化的标准 Sharpe 形式），也支持“日度超额收益均值法”。
  可通过 set_option('sharpe_method', 'cagr'|'mean') 切换（默认 cagr）。

在策略代码（无论 backtrader 或聚宽风格）中，可以直接调用这些选项：

```python
# 设置聚宽兼容参数（在策略类定义前或 initialize 内）
set_benchmark('000300.XSHG')
set_option('risk_free_rate', 0.03)           # 年化无风险利率 3%
set_option('annualization_factor', 250)      # 年化交易日数 250
set_option('sharpe_method', 'mean')          # 夏普计算口径：mean/cagr
set_option('alpha_unit', 'annual')           # 阿尔法输出单位：annual/daily
```

后端在响应 JSON 的 `metrics` 字段中也会返回审计信息：

- `annualization_factor`, `sharpe_rf_annual`, `sharpe_method`, `alpha_unit` 等，便于和聚宽口径对比核对。

如仍与聚宽有差异，常见原因包括：

- 基准指数不同（或数据源起始点不同）。
- 成交价口径不同（`fill_price` 可选 `open` 或 `close`，默认 `open`）。
- 滑点/佣金设置不同。
- 收益序列是否包含首个样本、是否对齐缺失日期等细节。

- 开始 / 结束日期
- 初始资金
- 策略代码 (必须包含 `class UserStrategy(bt.Strategy)`)

点击 “运行回测” 后获得 JSON 指标 + 净值 / 基准 / 超额曲线。

返回结构示例:

```
{
  "metrics": {"final_value":..., "pnl_pct":..., "excess_return":...},
  "equity_curve": [{"date":"2025-02-03","equity":1.01}, ...],
  "daily_returns": [{"date":"2025-02-03","ret":0.01}, ...],
  "benchmark_curve": [{"date":"2025-02-03","equity":1.0}, ...],
  "excess_curve": [{"date":"2025-02-03","excess":0.01}, ...]
}
```

`excess_curve` 计算: 策略累计净值 / 基准累计净值 - 1。

当前未输出逐笔交易明细 (可通过自定义 `notify_trade` / Observer 扩展)。

## 后续改进建议

- 策略沙箱: 多进程 + 超时/内存限制 + 安全白名单。
- 交易明细与订单流水: 自定义 Observer 收集并返回前端。
- 多数据源: 支持 Akshare / Tushare / 聚宽本地数据统一接口。
- 组合 & 多标的: 同时加载多 feed, 组合权重调整。
- 参数优化: 网格/随机/遗传算法搜索, 提供最优参数报告。
- Walk-Forward 与滚动回测。
- 回测结果持久化与对比: SQLite/PostgreSQL + 历史记录页面。
- 性能: 数据缓存、并行多任务队列 (Celery / RQ)。
- 前端增强: Monaco Editor, 策略模板库, 图表联动, 指标勾选显示。
- 风险指标: Sortino, Calmar, 波动率、区间回撤分布。

## 聚宽(JQ) 风格代码兼容 (实验性)

现在可以直接粘贴包含以下结构的聚宽策略：

```
import jqdata

def initialize(context):
  g.security = '000001.XSHE'
  set_benchmark('000300.XSHG')
  set_option('use_real_price', True)

def handle_data(context, data):
  close_data = attribute_history(g.security, 5, '1d', ['close'])
  MA5 = close_data['close'].mean()
  current_price = close_data['close'][-1]
  cash = context.portfolio.available_cash
  if current_price > 1.05 * MA5 and cash > 0:
    order_value(g.security, cash)
  elif current_price < 0.95 * MA5 and context.portfolio.positions[g.security].closeable_amount > 0:
    order_target(g.security, 0)
```

限制 / 差异：

- 仅支持单标的 (使用 `g.security`)。
- `attribute_history` 只支持获取最近 n 根并且 unit='1d'，fields 需包含 backtrader 的字段 (close/open/high/low/volume)。
- `order_value` / `order_target` 无手续费、无滑点、无分笔撮合。
- `record` 暂存于内部 list，目前未返回前端。
- `set_benchmark` 目前只是记录，不自动替代前端 benchmark 参数；想使用基准曲线仍需在页面填 `benchmark_symbol`。
- 未实现：分钟级回测、分红送配、复权模式控制、context.portfolio 更丰富字段。

如果同一代码里既有 `UserStrategy` 类又有 `initialize/handle_data`，优先使用 `UserStrategy`。

### 新增实验性扩展

- attribute_history: 仅日级, 返回负索引 DataFrame, `df['close'][-1]` 可直接取最新值。
- record()/log.info(): 结果中以 `jq_records`, `jq_logs` 返回，并在前端显示最近记录。
- 手续费/滑点: 可在 initialize 里使用 `set_option('commission', 0.0003)` 与 `set_option('slippage_perc', 0.0005)`。

## 手续费模型（统一聚宽风格默认）

当前版本已简化为单一确定性手续费逻辑（不再需要 `fee_model='jq'`）：

默认规则：

- 佣金率 (commission)：0.0003 （双边）
- 单笔最小佣金 (min_commission)：5 元
- 卖出印花税 (stamp_duty)：0.001 （仅卖出单边征收）
- 滑点：可选 `slippage_perc` 百分比滑点（若设置则按成交金额 \* 百分比体现在成交价的偏移，或依据 backtrader 的内置实现）

可覆盖的选项（在策略 `initialize` 或顶层放置）：

```python
set_option('commission', 0.00025)      # 调整佣金率
set_option('min_commission', 3)        # 调整最小单笔佣金（>=0）
set_option('stamp_duty', 0.001)        # 调整卖出印花税（>=0）
set_option('slippage_perc', 0.0005)    # 成交滑点百分比（示例 5bp）
```

运行期日志中会出现：

```
[commission_setup] rate=0.0003 model=unified
[fee_config] min_commission=5.0 stamp_duty=0.001
[fee] BUY  000516_daily ... base_comm=xx adj_comm=xx stamp_duty=0.0000 final_comm=xx
[fee] SELL 000516_daily ... base_comm=xx adj_comm=xx stamp_duty=yy   final_comm=xx
```

计算说明：

1. Backtrader 先按 `value * commission` 生成基础佣金。
2. 二次调整：若 < `min_commission` 则提升到最小值。
3. 若为卖出单，则加上 `value * stamp_duty`。
4. 最终 `commission` 字段写回订单执行对象，影响 `pnlcomm`。

常见验证步骤：

- 极小金额买入：确认佣金被抬到 5 元。
- 卖出同等金额：确认在 5 元基础上再加印花税。
- 调整 `min_commission` 为 0：验证小额交易佣金按比例走。
- 调整 `stamp_duty` 为 0：验证卖出与买入费用对称。

和聚宽可能仍存在的差异：

- 聚宽可能对分笔成交、撮合时点、成本价四舍五入存在额外规则；这里按单笔一次性撮合。
- 未模拟融资融券、分红派息、手续费折扣等高级场景。

如果需要扩展更多费率层（如阶梯费率、不同市场不同费率），可在 `backtest_engine.py` 中对 `fee_config` 增加结构，再在 `OrderCapture.notify_order` 里细分逻辑。

## 数据源与复权模式 (adjust_type) ✅

为复刻聚宽在 `set_option('use_real_price', True/False)` 与不同复权数据（未复权 / 前复权 / 后复权）下的下单差异，后端实现了可配置复权类型解析：

支持的文件命名（放置于 `data/` 目录）：

```
<code>_daily.csv         # 未复权 (英文后缀)
<code>_日.csv            # 未复权 (中文后缀)
<code>_daily_qfq.csv     # 前复权
<code>_日_qfq.csv
<code>_daily_hfq.csv     # 后复权
<code>_日_hfq.csv
```

### 1. 选项说明

```python
set_option('adjust_type', 'auto')   # 'auto' | 'raw' | 'qfq' | 'hfq'
set_option('use_real_price', True)  # 是否使用真实未复权价撮合（聚宽语义）

# 可选：控制候选后缀额外顺序（会与 adjust_type 生成的序列 merge 去重）
set_option('data_source_preference', ['_daily','_daily_qfq','_日'])

# 强制覆盖：优先级最高（存在则直接使用）
set_option('force_data_variant', 'qfq')  # 'raw'|'daily'|'qfq'|'hfq'|'日'
```

### 2. 语义规则

| adjust_type | use_real_price=True 时候选顺序 | use_real_price=False 时候选顺序 | 说明                                                    |
| ----------- | ------------------------------ | ------------------------------- | ------------------------------------------------------- |
| raw         | raw > qfq > hfq                | raw > qfq > hfq                 | 始终尽量未复权价撮合                                    |
| qfq         | qfq > raw > hfq                | qfq > raw > hfq                 | 强制前复权优先                                          |
| hfq         | hfq > raw > qfq                | hfq > raw > qfq                 | 强制后复权优先                                          |
| auto (默认) | raw > qfq > hfq                | qfq > raw > hfq                 | (已实现) use_real_price=True 时优先 raw；否则前复权优先 |

内部会根据英文与中文两套命名族自动扩展：`_daily` 与 `_日` 同层级、`_daily_qfq` 与 `_日_qfq` 同层级，以提升命中率。

最终序列 = (adjust_type 推导序列) + (data_source_preference) 去重保序。

### 3. 强制指定

`force_data_variant` 若设置并且目标文件存在，会直接选中（并记录 `[data_source_force]` 日志），忽略其余顺序。不存在则回退正常流程。

### 4. 审计日志

运行时关键日志（新版）：

```
[data_loader] code=000514 freq=daily adjust=auto use_real_price=True rows=XXXX file=...\000514_daily.csv
```

含义：

- code: 标的基础代码
- freq / adjust: 加载频率与复权模式参数
- use_real_price: 是否开启真实价优先（影响 auto 排序）
- rows: 加载后的行数（含预热区间）
- file: 实际选中的物理 CSV 路径（可直接核对其 close 列）

`metrics` 中会新增：

```
"adjust_type": "auto",
"data_sources_used": {"000516.XSHE": "000516_daily"}
```

### 5. 下单与交易路径影响

第二笔及后续交易数量差异经常来自：

1. 复权价 vs 未复权价 导致单位股数成本不同，进而影响剩余现金。
2. 最小佣金（5 元）放大小额成交的费用相对比例。
3. order_value sizing 在逼近可用现金的迭代中，价格/费用口径变化。

通过设置：

```python
set_option('use_real_price', True)
set_option('adjust_type', 'auto')  # 或显式 'raw'
```

即可确保使用未复权真实价；若想复刻“前复权价下单”场景则：

```python
set_option('use_real_price', False)
set_option('adjust_type', 'qfq')   # 或 auto (默认) 也会落在前复权
```

### 6. 典型排障步骤

1. 查找 `[data_loader]` 日志确认 file= 路径是否指向期望的 `_daily` (raw) 或 `_daily_qfq` 文件。
2. metrics['data_sources_used'] 中查看每个代码的物理路径。
3. 对比第一笔与第二笔 `[order_value_calc]` 里的 price / comm / leftover 识别差异来源。
4. 切换 adjust_type = 'raw' 与 'qfq' 回放，验证股数与剩余现金差异是否收敛到预期。

### 7. 后续规划

- 引入前/后复权因子表，支持持仓数量 / 成本在除权除息日的自动调整（当前版本未处理派息送股）。
- 在 "auto" 模式下加入“若某日 raw 缺失而 qfq 存在” 的跨日动态回退（目前是初始化一次性选取）。

若需要扩展更多市场或分钟级数据，可在 `backtest_engine.py` 中扩展 `_resolve_adjust_pref`。欢迎继续反馈实际对齐聚宽时的差异场景。
