import io
import json
import traceback
import types
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable

import backtrader as bt
import pandas as pd

# -----------------------------
# 数据加载器 (简化版本)
# -----------------------------

# 读取数据映射成英文
def _read_raw_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 兼容中文列名 -> 英文
    rename_map = {
        '日期': 'datetime',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '股票代码': 'code',
    }
    # 仅重命名存在的列
    cols = {c: rename_map[c] for c in df.columns if c in rename_map}
    if cols:
        df = df.rename(columns=cols)
    # 若没有 datetime 列尝试 date
    if 'datetime' not in df.columns:
        for candidate in ['date', 'Date']:
            if candidate in df.columns:
                df = df.rename(columns={candidate: 'datetime'})
                break
    if 'datetime' not in df.columns:
        raise ValueError('CSV 缺少日期列 (datetime / 日期)')
    return df

# 切片回测时间段
def load_csv_dataframe(symbol: str, start: str, end: str, datadir: str = 'data') -> pd.DataFrame:
    path = f"{datadir}/{symbol}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到数据文件: {path}')
    df = _read_raw_df(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    mask = (df['datetime'] >= pd.to_datetime(start)) & (df['datetime'] <= pd.to_datetime(end))
    df = df.loc[mask].sort_values('datetime').reset_index(drop=True)
    return df

# 转换DF为PD
def load_csv_data(symbol: str, start: str, end: str, datadir: str = 'data') -> bt.feeds.PandasData:
    """读取 CSV 并返回供 backtrader 使用的 DataFeed (兼容中文列)。"""
    df = load_csv_dataframe(symbol, start, end, datadir)
    # 只保留必要列
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'数据文件缺少必要列: {missing}')
    feed_df = df[required].copy()
    feed_df.set_index('datetime', inplace=True)
    return bt.feeds.PandasData(dataname=feed_df)

# -----------------------------
# 回测结果模型
# -----------------------------

@dataclass
class TradeRecord:
    datetime: str
    side: str
    size: float
    price: float
    value: float
    commission: float

@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]  # 策略累计净值
    daily_returns: List[Dict[str, Any]]  # 策略日收益
    benchmark_curve: List[Dict[str, Any]]  # 基准累计净值
    excess_curve: List[Dict[str, Any]]  # 超额累计净值 (策略/基准 -1)
    trades: List[TradeRecord]
    log: str
    jq_records: List[Dict[str, Any]] | None = None
    jq_logs: List[str] | None = None

# -----------------------------
# 策略动态加载
# -----------------------------

ALLOWED_GLOBALS = {
    '__builtins__': {
        'abs': abs,
        'min': min,
        'max': max,
        'range': range,
        'len': len,
        'sum': sum,
        'enumerate': enumerate,
        'zip': zip,
        'float': float,
        'int': int,
        'str': str,
        'print': print,
    },
    'bt': bt,
    'pd': pd,
}

# 允许的安全导入白名单
_IMPORT_WHITELIST = {
    'backtrader': bt,
    'pandas': pd,
    'math': __import__('math'),
    'statistics': __import__('statistics'),
    'numpy': None,  # 延迟导入，用户可能没用到
}

def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root in _IMPORT_WHITELIST:
        # 延迟导入 numpy
        if root == 'numpy' and _IMPORT_WHITELIST[root] is None:
            try:
                _IMPORT_WHITELIST[root] = __import__(root)
            except Exception:
                raise ImportError('numpy 不在当前环境或导入失败')
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f'不允许导入模块: {name}')

ALLOWED_GLOBALS['__builtins__']['__import__'] = _limited_import

class _G:  # 简易全局对象模拟聚宽 g
    pass


def _build_jq_compat_env(target_dict: Dict[str, Any]):
    """向执行环境注入最小聚宽兼容 API。"""
    g = _G()
    jq_state: Dict[str, Any] = {
        'benchmark': None,
        'options': {},
        'records': [],
        'log': [],
        'g': g,
    }

    class _Log:
        def info(self, msg):
            jq_state['log'].append(str(msg))
            print('[INFO]', msg)

    log_obj = _Log()

    def set_benchmark(code: str):
        jq_state['benchmark'] = code

    def set_option(name: str, value: Any):
        jq_state['options'][name] = value

    def record(**kwargs):
        jq_state['records'].append({'dt': None, **kwargs})

    # attribute_history 简化: 直接从已加载的 pandas 数据缓存里取；此时还未有数据引用，运行期在包装策略里实现
    def attribute_history(security: str, n: int, unit: str, fields: List[str]):
        raise RuntimeError('attribute_history 仅在聚宽兼容包装策略中可用')

    def order_value(security: str, value: float):
        raise RuntimeError('order_value 仅在聚宽兼容包装策略中可用')

    def order_target(security: str, target: float):
        raise RuntimeError('order_target 仅在聚宽兼容包装策略中可用')

    target_dict.update({
        'g': g,
        'set_benchmark': set_benchmark,
        'set_option': set_option,
        'record': record,
        'log': log_obj,
        'attribute_history': attribute_history,
        'order_value': order_value,
        'order_target': order_target,
        'jq_state': jq_state,
    })
    return jq_state


def compile_user_strategy(code: str):
    """执行用户策略代码，支持两种模式：
    1) 标准 backtrader 模式: 用户提供 UserStrategy 类。
    2) 聚宽兼容模式: 提供 initialize(context), handle_data(context, data) 函数与可选 g.security。
    """
    # 处理聚宽常见导入
    sanitized = []
    for line in code.splitlines():
        if line.strip().startswith('import jqdata'):
            continue  # 跳过
        sanitized.append(line)
    code = '\n'.join(sanitized)

    module = types.ModuleType('user_module')
    exec_env = dict(ALLOWED_GLOBALS)  # 复制
    jq_state = _build_jq_compat_env(exec_env)
    exec(code, exec_env, module.__dict__)

    # 模式1: 直接存在 UserStrategy
    if 'UserStrategy' in module.__dict__:
        return module.__dict__['UserStrategy'], jq_state

    # 模式2: 识别聚宽风格函数
    if 'initialize' in module.__dict__ and 'handle_data' in module.__dict__:
        init_func: Callable = module.__dict__['initialize']
        handle_func: Callable = module.__dict__['handle_data']

        class UserStrategy(bt.Strategy):  # type: ignore
            def __init__(self):
                # 构建 context 仿真对象
                class _Portfolio:
                    @property
                    def available_cash(inner_self):
                        return self.broker.getcash()

                    @property
                    def positions(inner_self):
                        # 返回 dict-like; 仅支持单标的
                        return {getattr(jq_state['g'], 'security', 'data0'): types.SimpleNamespace(
                            closeable_amount=int(self.position.size)
                        )}

                class _Context:
                    portfolio = _Portfolio()

                self._jq_context = _Context()
                # 运行 initialize
                init_func(self._jq_context)

            def prenext(self):
                # 允许 MA 等指标还未就绪也执行
                self.next()

            def next(self):  # 每个bar 调用 handle_data
                # 提供 attribute_history, order_value, order_target 实现
                def attribute_history(security: str, n: int, unit: str, fields: List[str]):
                    # 支持 unit 为 '1d' 或 '1D' 或 'day'
                    if unit.lower() not in ('1d', 'day', 'd'):
                        raise ValueError('attribute_history 目前仅支持日级 unit=1d')
                    length = min(n, len(self.data))
                    import pandas as _pd
                    if length <= 0:
                        return _pd.DataFrame({f: [] for f in fields})
                    data_dict: Dict[str, List[float]] = {}
                    for f in fields:
                        line = getattr(self.data, f, None)
                        if line is None:
                            # 用 NaN 填充
                            data_dict[f] = [float('nan')] * length
                            continue
                        seq = list(line.get(size=length))
                        data_dict[f] = seq[-length:]
                    idx = list(range(-length, 0))  # 负索引
                    return _pd.DataFrame(data_dict, index=idx)

                def order_value(security: str, value: float):
                    price = self.data.close[0]
                    size = int(value / price)
                    if size > 0:
                        self.buy(size=size)

                def order_target(security: str, target: float):
                    if target == 0 and self.position:
                        self.close()

                # 动态覆盖 exec 环境里的函数（只在第一次后就不再改变 jq_state）
                jq_state_ref = exec_env['jq_state']
                exec_env['attribute_history'] = attribute_history
                exec_env['order_value'] = order_value
                exec_env['order_target'] = order_target

                handle_func(self._jq_context, None)

        return UserStrategy, jq_state

    raise ValueError('策略代码需定义 UserStrategy(bt.Strategy) 或 initialize/handle_data 聚宽风格函数')

# -----------------------------
# Analyzer 助手
# -----------------------------

def extract_analyzers(cerebro: bt.Cerebro) -> Dict[str, Any]:
    analyzers = {}
    for name, analyzer in cerebro.runstrats[0][0].analyzers.items():
        try:
            analyzers[name] = analyzer.get_analysis()
        except Exception:
            analyzers[name] = {}
    return analyzers

# -----------------------------
# 回测执行
# -----------------------------


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    cash: float,
    strategy_code: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    benchmark_symbol: Optional[str] = None,
    datadir: str = 'data',
) -> BacktestResult:
    log_buffer = io.StringIO()
    try:
        StrategyCls, jq_state = compile_user_strategy(strategy_code)
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        # 基础佣金/滑点将于策略添加后设置

        # ------------------ 标的解析逻辑 ------------------
        # 允许三种来源优先级:
        # 1. 聚宽风格 g.security (str 或 list)
        # 2. 表单传入 symbol (逗号分隔)
        # 3. 默认 fallback 单一 symbol
        def _jq_code_to_filename(code: str) -> str:
            """将聚宽证券代码转换为本地 CSV 文件名。
            规则: 000001.XSHE -> 000001_daily (简化) ; 000001.XSHG -> 000001_daily
            若已包含 _daily / _日 等后缀则保持原样。
            """
            c = code.strip()
            lower = c.lower()
            # 已经是内部文件命名风格
            if lower.endswith('_daily') or lower.endswith('_日'):  # 兼容 _日
                return c.replace('.XSHE', '').replace('.XSHG', '')
            if '.xshe' in lower or '.xshg' in lower:
                base = c.split('.')[0]
                return f"{base}_daily"
            return c  # 原样返回, 例如用户直接写文件名

        symbols: List[str] = []
        g_sec = getattr(jq_state.get('g'), 'security', None) if 'jq_state' in locals() else None
        if g_sec:
            if isinstance(g_sec, (list, tuple)):
                symbols = [_jq_code_to_filename(s) for s in g_sec if str(s).strip()]
            elif isinstance(g_sec, str):
                symbols = [_jq_code_to_filename(g_sec)]

        if not symbols:  # 回退到表单 symbol
            if isinstance(symbol, str):
                symbols = [s.strip() for s in symbol.split(',') if s.strip()]
            else:
                symbols = list(symbol)

        # 防止重复
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            raise ValueError('未指定任何标的: 请在策略 g.security 或 表单 symbol 提供至少一个')

        # 若需要也对 benchmark_symbol 做同样转换 (允许用户输入 000300.XSHG)
        if benchmark_symbol:
            benchmark_symbol = _jq_code_to_filename(benchmark_symbol)
        data_feeds = []
        for idx, sym in enumerate(symbols):
            dfeed = load_csv_data(sym, start, end, datadir)
            cerebro.adddata(dfeed, name=sym)
            data_feeds.append(sym)

        # 策略
        strategy_params = strategy_params or {}
        cerebro.addstrategy(StrategyCls, **strategy_params)
        # 佣金 & 滑点 (聚宽兼容 set_option)
        commission = jq_state.get('options', {}).get('commission') if 'jq_state' in locals() else None
        if commission is not None:
            cerebro.broker.setcommission(commission=commission)
        slippage_perc = jq_state.get('options', {}).get('slippage_perc') if 'jq_state' in locals() else None
        if slippage_perc is not None:
            cerebro.broker.set_slippage_perc(perc=slippage_perc)

        # analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        results = cerebro.run()
        strat = results[0]

        # 指标汇总
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
        dd = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trade_ana = strat.analyzers.trades.get_analysis()
        timereturn = strat.analyzers.timereturn.get_analysis()

        # 策略权益曲线
        equity_curve = []
        cumulative = 1.0
        for dt, r in timereturn.items():
            cumulative *= (1 + r)
            equity_curve.append({'date': dt.strftime('%Y-%m-%d'), 'equity': cumulative})

        daily_returns = [{'date': dt.strftime('%Y-%m-%d'), 'ret': r} for dt, r in timereturn.items()]

        # 交易聚合数据
        total_trades = trade_ana.get('total', {}).get('total', 0)
        won = trade_ana.get('won', {}).get('total', 0)
        lost = trade_ana.get('lost', {}).get('total', 0)
        win_rate = (won / total_trades) if total_trades else 0

        metrics = {
            'final_value': cerebro.broker.getvalue(),
            'pnl_pct': (cerebro.broker.getvalue() / cash - 1),
            'sharpe': sharpe,
            'max_drawdown': dd.get('max', {}).get('drawdown'),
            'max_drawdown_len': dd.get('max', {}).get('len'),
            'drawdown_pct': dd.get('drawdown'),
            'drawdown_len': dd.get('len'),
            'rt_annual': returns.get('rtannual'),
            'rnorm': returns.get('rnorm'),
            'rnorm100': returns.get('rnorm100'),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'won_trades': won,
            'lost_trades': lost,
        }

        # 基准 & 超额
        benchmark_curve: List[Dict[str, Any]] = []
        excess_curve: List[Dict[str, Any]] = []
        if benchmark_symbol:
            bench_df = load_csv_dataframe(benchmark_symbol, start, end, datadir)
            bench_df = bench_df[['datetime', 'close']].copy()
            bench_df['ret'] = bench_df['close'].pct_change().fillna(0.0)
            bench_df['equity'] = (1 + bench_df['ret']).cumprod()
            bm_map = {d.strftime('%Y-%m-%d'): (r, eq) for d, r, eq in zip(bench_df['datetime'], bench_df['ret'], bench_df['equity'])}
            for ec in equity_curve:
                d = ec['date']
                if d in bm_map:
                    br, beq = bm_map[d]
                    benchmark_curve.append({'date': d, 'equity': beq})
                    excess_curve.append({'date': d, 'excess': ec['equity']/beq - 1 if beq != 0 else 0})
            if benchmark_curve:
                metrics['benchmark_final'] = benchmark_curve[-1]['equity']
                metrics['excess_return'] = equity_curve[-1]['equity']/benchmark_curve[-1]['equity'] - 1
            else:
                metrics['benchmark_final'] = None
                metrics['excess_return'] = None
        else:
            metrics['benchmark_final'] = None
            metrics['excess_return'] = None

        # 简化: 不逐笔构造 trades 列表 (需要 broker observers/notify) 这里只返回聚合
        # 交易记录: 使用 notify_trade 捕获
        trade_records: List[TradeRecord] = []

        class _TradeRecorder(bt.Strategy):
            def notify_trade(self, trade):
                if trade.isclosed:
                    trade_records.append(TradeRecord(
                        datetime=str(self.data.datetime.date(0)),
                        side='LONG' if trade.history[0].event.size > 0 else 'SHORT',
                        size=trade.history[0].event.size,
                        price=trade.price,
                        value=trade.value,
                        commission=trade.commission,
                    ))

        # 在策略后添加一个记录器 (仅监听)
        cerebro.addstrategy(_TradeRecorder)

        # 回测结束后 trade_records 填入

        trades: List[TradeRecord] = trade_records

        # 聚宽兼容记录
        # 给 record 补日期
        jq_records = jq_state.get('records') if 'jq_state' in locals() else None
        if jq_records and equity_curve:
            date_map = equity_curve[-len(jq_records):]  # 简单映射尾部
            for i, r in enumerate(jq_records):
                if r.get('dt') is None and i < len(date_map):
                    r['dt'] = date_map[i]['date']
        jq_logs = jq_state.get('log') if 'jq_state' in locals() else None

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_curve=benchmark_curve,
            excess_curve=excess_curve,
            trades=trades,
            log=log_buffer.getvalue(),
            jq_records=jq_records,
            jq_logs=jq_logs,
        )
    except Exception:
        tb = traceback.format_exc()
        return BacktestResult(
            metrics={'error': True},
            equity_curve=[],
            daily_returns=[],
            benchmark_curve=[],
            excess_curve=[],
            trades=[],
            log=tb,
            jq_records=None,
            jq_logs=None,
        )


if __name__ == '__main__':
    # 简单自测
    code = """\nimport backtrader as bt\nclass UserStrategy(bt.Strategy):\n    def next(self):\n        if not self.position: self.buy(size=10)\n        elif len(self) > 5: self.sell()\n"""
    # 注意: 需要先准备 data/sample.csv
    res = run_backtest('sample', '2025-01-01', '2025-03-01', 100000, code, benchmark_symbol='sample')
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
