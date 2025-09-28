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
 夏普计算：默认使用“CAGR 年化口径”（Rp年化与波动年化的标准 Sharpe 形式），也支持“日度超额收益均值法”。
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
