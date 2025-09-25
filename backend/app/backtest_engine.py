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
    # 保留原字段 (兼容前端) —— datetime 代表平仓日期/或最后事件日期
    datetime: str
    side: str              # LONG / SHORT
    size: float            # 开仓手数(股数)
    price: float           # 兼容字段: 使用平仓价格(exit_price)
    value: float           # 兼容字段: 使用平仓市值(exit_value)
    commission: float
    # 新增更细字段
    open_datetime: str | None = None
    close_datetime: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    entry_value: float | None = None
    exit_value: float | None = None
    pnl: float | None = None
    pnl_comm: float | None = None

@dataclass
class OrderRecord:
    datetime: str
    symbol: str | None
    side: str            # BUY / SELL
    size: float          # 本次成交股数(带方向)
    price: float         # 本次成交价格
    value: float         # 本次成交金额(含方向)
    commission: float
    status: str          # Completed / Canceled / Rejected 等

@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]  # 策略累计净值
    daily_returns: List[Dict[str, Any]]  # 策略日收益
    benchmark_curve: List[Dict[str, Any]]  # 基准累计净值
    excess_curve: List[Dict[str, Any]]  # 超额累计净值 (策略/基准 -1)
    trades: List[TradeRecord]
    log: str
    orders: List[OrderRecord] | None = None
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
        # 运行辅助状态
        'history_df': None,      # 单标的完整历史（DataFrame，含 datetime 列）
        'history_df_map': {},    # 多标的缓存
        'current_dt': None,      # 当前 bar 展示时间（字符串）
        'user_start': None,      # 用户选择的开始日期（YYYY-MM-DD）
        'in_warmup': False,      # 暖场阶段（< user_start）
    }

    class _Log:
        def info(self, msg):
            # 暖场阶段不输出日志（仅用于准备历史数据）
            if jq_state.get('in_warmup'):
                return
            # 聚宽风格：首行包含时间与级别，续行缩进
            dt = jq_state.get('current_dt')
            if not dt:
                try:
                    from datetime import datetime as _dt
                    dt = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    dt = '0000-00-00 00:00:00'
            header = f"{dt} - INFO - "
            s = str(msg)
            lines = s.splitlines() or ['']
            first = header + lines[0]
            cont = ['    ' + ln for ln in lines[1:]]
            formatted = '\n'.join([first] + cont)
            jq_state['log'].append(formatted)
            print(formatted)

    log_obj = _Log()

    def set_benchmark(code: str):
        jq_state['benchmark'] = code

    def set_option(name: str, value: Any):
        jq_state['options'][name] = value
        jq_state['log'].append(f"[set_option] {name}={value}")

    def record(**kwargs):
        # 暖场阶段不记录（避免越界日期）
        if jq_state.get('in_warmup'):
            return
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

            def _run_handle(self):
                # 提供 attribute_history, order_value, order_target 实现
                def attribute_history(security: str, n: int, unit: str, fields: List[str]):
                    # 支持 unit 为 '1d' 或 '1D' 或 'day'
                    if unit.lower() not in ('1d', 'day', 'd'):
                        raise ValueError('attribute_history 目前仅支持日级 unit=1d')
                    import pandas as _pd
                    # JQ 兼容：Series/DF 支持使用负整数做“位置”索引，如 s[-1] 代表最后一项
                    class _JQSeries(_pd.Series):
                        @property
                        def _constructor(self):
                            return _JQSeries

                        def __getitem__(self, key):
                            # 如果是整数或整型切片，则按位置 iloc 处理（与聚宽兼容）
                            if isinstance(key, int):
                                return self.iloc[key]
                            if isinstance(key, slice):
                                is_int = lambda x: (x is None) or isinstance(x, int)
                                if is_int(key.start) and is_int(key.stop) and (key.step is None or isinstance(key.step, int)):
                                    return _JQSeries(self.iloc[key])
                            return super().__getitem__(key)

                    class _JQDataFrame(_pd.DataFrame):
                        @property
                        def _constructor(self):
                            return _JQDataFrame

                        @property
                        def _constructor_sliced(self):
                            # 让 df['col'] 返回的 Series 具备负整数位置索引语义
                            return _JQSeries

                        def __getitem__(self, key):
                            # 若 key 为整数或整型切片，则按位置 iloc 处理（与聚宽兼容）
                            if isinstance(key, int):
                                return self.iloc[key]
                            if isinstance(key, slice):
                                is_int = lambda x: (x is None) or isinstance(x, int)
                                if is_int(key.start) and is_int(key.stop) and (key.step is None or isinstance(key.step, int)):
                                    return _JQDataFrame(self.iloc[key])
                            return super().__getitem__(key)

                    jqst = exec_env['jq_state']
                    # 计算当前日期
                    try:
                        cur_date = bt.num2date(self.data.datetime[0]).date()
                    except Exception:
                        cur_date = None
                    # 根据传入 security 选择对应的完整历史
                    full_df = None
                    try:
                        sym_map = jqst.get('history_df_map') or {}
                        base = str(security).split('.')[0]
                        # 先精确匹配 key == base，再匹配以 base_ 开头
                        if base in sym_map:
                            full_df = sym_map[base]
                        else:
                            for k, v in sym_map.items():
                                if str(k).startswith(base + '_') or str(k).startswith(base):
                                    full_df = v
                                    break
                        if full_df is None:
                            full_df = jqst.get('history_df')
                    except Exception:
                        full_df = jqst.get('history_df')
                    if isinstance(full_df, _pd.DataFrame) and cur_date is not None:
                        # JQ 语义：不包含当日，返回“当前日之前”的 n 条
                        df = full_df[full_df['datetime'].dt.date < cur_date]
                        if df.empty:
                            return _JQDataFrame({f: [] for f in fields})
                        tail = df.tail(n)
                        out = tail[['datetime', *fields]].copy()
                        out.index = out['datetime'].dt.date.astype(str)
                        # 与聚宽展示一致，不显示索引名
                        try:
                            out.index.name = None
                        except Exception:
                            pass
                        out = out.drop(columns=['datetime'])
                        return _JQDataFrame(out)
                    # 回退：直接从数据行取，尽量用日期索引
                    # 不包含当前 bar（索引 0），所以最多使用 len(self.data) - 1 条
                    available = max(len(self.data) - 1, 0)
                    length = min(n, available)
                    if length <= 0:
                        return _JQDataFrame({f: [] for f in fields})
                    data_dict: Dict[str, List[float]] = {}
                    for f in fields:
                        line = getattr(self.data, f, None)
                        if line is None:
                            data_dict[f] = [float('nan')] * length
                            continue
                        # 取“之前”的 length 条（不包含索引 0 当日）：[-length, ..., -1]
                        vals = [line[i] for i in range(-length, 0)] if length > 0 else []
                        data_dict[f] = vals
                    # 构造日期索引（更稳健：逐条读取末尾 length 个 bar 的日期，排除当日）
                    try:
                        # 使用负索引逐条转换，避免某些环境下 get(size=) 异常
                        str_idx = []
                        for i in range(-length, 0):
                            dt_num = self.data.datetime[i]
                            dt_str = bt.num2date(dt_num).date().isoformat()
                            str_idx.append(dt_str)
                    except Exception:
                        # 仍然失败则退回负索引
                        str_idx = list(range(-length, 0))
                    return _JQDataFrame(_pd.DataFrame(data_dict, index=str_idx))

                def order_value(security: str, value: float):
                    # 暖场期不交易
                    if exec_env['jq_state'].get('in_warmup'):
                        return
                    # 成交定价策略：默认 'open'（聚宽日频），可通过 set_option('fill_price', 'close') 切换
                    fill_price = str(exec_env['jq_state']['options'].get('fill_price', 'open')).lower() if 'jq_state' in exec_env else 'open'
                    if fill_price == 'close':
                        chosen_price = getattr(self.data, 'close')[0]
                    else:
                        chosen_price = getattr(self.data, 'open')[0] if hasattr(self.data, 'open') else getattr(self.data, 'close')[0]
                    if chosen_price <= 0:
                        return
                    # 支持正负 value：正为买入金额，负为卖出金额
                    size_raw = int(value / chosen_price)
                    # A股规则：股数需为100的整数倍
                    lot = int(exec_env['jq_state']['options'].get('lot', 100)) if 'jq_state' in exec_env else 100
                    adj = (abs(size_raw) // lot) * lot
                    size = adj if size_raw >= 0 else -adj
                    if size_raw != 0 and abs(size) != abs(size_raw):
                        # 打印与聚宽类似的提示
                        try:
                            exec_env['log'].info(f"[order_value] 数量需为 {lot} 的整数倍，调整为 {abs(size)}")
                        except Exception:
                            pass
                    # 记录下单定价与数量，便于核对现金差异
                    try:
                        exec_env['log'].info(f"[order_value] price={chosen_price:.4f} final_size={size}")
                    except Exception:
                        pass
                    if size == 0:
                        return
                    if size > 0:
                        self.buy(size=size)
                    else:
                        self.sell(size=abs(size))

                def order_target(security: str, target: float):
                    # 暖场期不交易
                    if exec_env['jq_state'].get('in_warmup'):
                        return
                    """将当前持仓调整到目标股数 target。
                    - target 为目标持仓股数（可为浮点，将被取整）
                    - delta > 0 执行买入 delta 股；delta < 0 执行卖出 |delta| 股
                    - target == 0 等价于平仓
                    """
                    cur = int(getattr(self.position, 'size', 0) or 0)
                    tgt_raw = int(target or 0)
                    lot = int(exec_env['jq_state']['options'].get('lot', 100)) if 'jq_state' in exec_env else 100
                    # 目标同样按100股对齐
                    tgt = (abs(tgt_raw) // lot) * lot
                    tgt = tgt if tgt_raw >= 0 else -tgt
                    delta = tgt - cur
                    if delta == 0:
                        return
                    if delta > 0:
                        self.buy(size=delta)
                    else:
                        # delta < 0 -> 卖出 |delta|
                        # 若目标为 0 则等价于平仓
                        if tgt == 0 and self.position:
                            self.close()
                        else:
                            self.sell(size=abs(delta))

                # 动态覆盖 exec 环境里的函数（只在第一次后就不再改变 jq_state）
                jq_state_ref = exec_env['jq_state']
                # 设置当前 bar 的时间与暖场标记（用户选择开始日前为暖场）
                try:
                    cur_dt = bt.num2date(self.data.datetime[0])
                    cur_date_str = cur_dt.date().isoformat()
                    jq_state_ref['current_dt'] = f"{cur_date_str} 09:30:00"
                    user_start = jq_state_ref.get('user_start')
                    jq_state_ref['in_warmup'] = bool(user_start) and (cur_date_str < str(user_start))
                except Exception:
                    pass
                exec_env['attribute_history'] = attribute_history
                exec_env['order_value'] = order_value
                exec_env['order_target'] = order_target

                # 暖场阶段不调用用户 handle_data（仅用于准备历史数据），与聚宽一致
                if jq_state_ref.get('in_warmup'):
                    return

                # 调用用户 handle_data，并在出错时将堆栈写入 JQ Logs，便于排查
                try:
                    handle_func(self._jq_context, None)
                except Exception:
                    try:
                        import traceback as _tb
                        err = _tb.format_exc()
                        # 暖场期也输出异常，便于发现问题
                        exec_env['jq_state']['in_warmup'] = False
                        _logger = exec_env.get('log')
                        if _logger is not None:
                            _logger.info("handle_data 发生异常:\n" + err)
                    except Exception:
                        pass
                    raise

            def next(self):  # 每个bar 调用 handle_data
                # 当选择收盘成交(fill_price=close)时，在 next(收盘阶段)执行；否则跳过
                fillp = str(exec_env['jq_state']['options'].get('fill_price', 'open')).lower() if 'jq_state' in exec_env else 'open'
                if fillp == 'close':
                    self._run_handle()
                else:
                    # open 成交在 next_open 中执行
                    return

            def next_open(self):
                # 当选择开盘成交(fill_price=open)时，在 next_open(开盘阶段)执行；否则跳过
                fillp = str(exec_env['jq_state']['options'].get('fill_price', 'open')).lower() if 'jq_state' in exec_env else 'open'
                if fillp == 'open':
                    self._run_handle()
                else:
                    return

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
        # 1. 编译策略 & 初始化 Cerebro
        StrategyCls, jq_state = compile_user_strategy(strategy_code)
        # 记录用户开始日期，供暖场/日志用
        jq_state['user_start'] = start
        # 根据 fill_price 决定是否启用 cheat-on-open（保证当日开盘撮合）
        fill_price_opt = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
        try:
            cerebro = bt.Cerebro(cheat_on_open=(fill_price_opt == 'open'))
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open={fill_price_opt == 'open'}")
        except Exception:
            cerebro = bt.Cerebro()
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open=unsupported")
        cerebro.broker.setcash(cash)

        # 2. 标的解析与数据加载
        def _normalize_existing(name: str) -> str:
            base_lower = name.lower()
            patterns = [
                ('_daily_qfq_daily', '_daily_qfq'),
                ('_daily_hfq_daily', '_daily_hfq'),
                ('_日_qfq_日', '_日_qfq'),
                ('_日_hfq_日', '_日_hfq'),
                ('_daily_daily', '_daily'),
            ]
            for old, new in patterns:
                if base_lower.endswith(old):
                    return name[: -len(old)] + new
            return name

        preserved_suffixes = ('_daily', '_daily_qfq', '_daily_hfq', '_qfq', '_hfq', '_日', '_日_qfq', '_日_hfq')

        # 允许通过 set_option('data_source_preference', ['_daily_qfq','_daily','_日']) 指定文件优先级
        # 默认优先 qfq -> raw daily -> 中文“日”
        def _get_data_source_preference() -> List[str]:
            try:
                pref = jq_state.get('options', {}).get('data_source_preference')
                if isinstance(pref, (list, tuple)) and pref:
                    return list(pref)
            except Exception:
                pass
            return ['_daily_qfq', '_daily', '_日']

        def _map_security_code(code: str) -> str:
            c = code.strip()
            lower = c.lower()
            if any(lower.endswith(suf) for suf in preserved_suffixes):
                return _normalize_existing(c.replace('.XSHE', '').replace('.XSHG', ''))
            if '.xshe' in lower or '.xshg' in lower:
                base = c.split('.')[0]
                # 根据偏好顺序构建候选列表
                pref = _get_data_source_preference()
                candidates = [f"{base}{suf}" for suf in pref]
                for fn in candidates:
                    if os.path.exists(os.path.join(datadir, fn + '.csv')):
                        return fn
                return candidates[0]
            return c

        def _map_benchmark_code(code: str) -> str:
            c = code.strip()
            lower = c.lower()
            if any(lower.endswith(suf) for suf in preserved_suffixes):
                return _normalize_existing(c.replace('.XSHE', '').replace('.XSHG', ''))
            if '.xshe' in lower or '.xshg' in lower:
                base = c.split('.')[0]
                # 基准默认偏向中文“_日”，再 raw，再 qfq，除非用户有自定义
                bench_pref = jq_state.get('options', {}).get('benchmark_source_preference')
                if isinstance(bench_pref, (list, tuple)) and bench_pref:
                    candidates = [f"{base}{suf}" for suf in bench_pref]
                else:
                    candidates = [f"{base}_日", f"{base}_daily", f"{base}_daily_qfq"]
                for fn in candidates:
                    if os.path.exists(os.path.join(datadir, fn + '.csv')):
                        return fn
                return candidates[0]
            return c

        symbols: List[str] = []
        g_sec = getattr(jq_state.get('g'), 'security', None)
        if g_sec:
            if isinstance(g_sec, (list, tuple)):
                symbols = [_map_security_code(s) for s in g_sec if str(s).strip()]
            elif isinstance(g_sec, str):
                symbols = [_map_security_code(g_sec)]
        if not symbols:
            if isinstance(symbol, str):
                symbols = [s.strip() for s in symbol.split(',') if s.strip()]
            else:
                symbols = list(symbol)
        symbols = list(dict.fromkeys(symbols))  # 去重保持顺序
        if not symbols:
            raise ValueError('未指定任何标的: 请在策略 g.security 或 表单 symbol 提供至少一个')

        if benchmark_symbol:
            benchmark_symbol = _map_benchmark_code(benchmark_symbol)

        # 读取暖场天数（默认 250，可 set_option('history_lookback_days', N)）
        lookback_days = 250
        try:
            lb = jq_state.get('options', {}).get('history_lookback_days')
            if isinstance(lb, (int, float)) and lb >= 0:
                lookback_days = int(lb)
        except Exception:
            pass
        warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=lookback_days)).date().isoformat()

        # 建立原始输入代码与映射后文件名的对应关系，便于 attribute_history 精确匹配
        jq_state.setdefault('symbol_file_map', {})

        for i, sym in enumerate(symbols):
            dfeed = load_csv_data(sym, warmup_start, end, datadir)
            cerebro.adddata(dfeed, name=sym)
            # 缓存完整历史（用于 attribute_history）
            try:
                full_df = load_csv_dataframe(sym, warmup_start, end, datadir)
                jq_state['history_df_map'][sym] = full_df
                if i == 0:
                    jq_state['history_df'] = full_df
                # 记录映射关系与所用数据文件到日志
                try:
                    jq_state['symbol_file_map'][symbols[i] if i < len(symbols) else sym] = sym
                    jq_state['log'].append(f"[data_source] {symbols[i] if i < len(symbols) else sym} -> {sym}.csv")
                except Exception:
                    pass
            except Exception:
                pass

        # 3. 处理聚宽 set_option 影响 (commission / slippage / use_real_price)
        strategy_params = strategy_params or {}
        commission = jq_state.get('options', {}).get('commission')
        if commission is None:
            # 默认按照 0.03% 佣金，贴近聚宽
            commission = 0.0003
            jq_state['options']['commission'] = commission
        try:
            cerebro.broker.setcommission(commission=commission)
        except Exception:
            pass
        slippage_perc = jq_state.get('options', {}).get('slippage_perc')
        if slippage_perc is not None:
            try:
                cerebro.broker.set_slippage_perc(perc=slippage_perc)
            except Exception:
                pass
        use_real_price_opt = jq_state.get('options', {}).get('use_real_price', False)
        # 成交价格控制：fill_price=open/close（默认 open）。
        fill_price_opt = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
        # 优先尝试 cheat-on-open / cheat-on-close 与所选价格匹配
        try:
            if fill_price_opt == 'close' and hasattr(cerebro.broker, 'set_coc'):
                cerebro.broker.set_coc(True)
            elif fill_price_opt != 'close' and hasattr(cerebro.broker, 'set_coo'):
                cerebro.broker.set_coo(True)
        except Exception:
            pass

        # 4. 定义 TradeCapture Analyzer (捕获已平仓 & 未平仓)
        class TradeCapture(bt.Analyzer):
            def start(self):
                self.records: List[TradeRecord] = []
                self._open_cache: dict[int, TradeRecord] = {}

            def _fmt_date(self, num_dt):
                try:
                    return bt.num2date(num_dt).date().isoformat() if num_dt else None
                except Exception:
                    return None

            def notify_trade(self, trade):
                # 开仓记录/占位
                if trade.isopen and not trade.isclosed:
                    tid = id(trade)
                    if tid not in self._open_cache:
                        open_dt = self._fmt_date(getattr(trade, 'dtopen', None))
                        entry_price = getattr(trade, 'price', None) or getattr(trade, 'openprice', None)
                        size_raw = getattr(trade, 'size', 0)
                        size_hist = 0
                        hist = getattr(trade, 'history', None)
                        if hist:
                            try:
                                size_hist = abs(hist[0].event.size)
                            except Exception:
                                size_hist = 0
                        size = abs(size_raw) or size_hist
                        side = 'LONG' if size_raw > 0 else 'SHORT'
                        self._open_cache[tid] = TradeRecord(
                            datetime=open_dt or '',
                            side=side,
                            size=size,
                            price=entry_price or 0.0,
                            value=(entry_price * size) if entry_price else 0.0,
                            commission=getattr(trade, 'commission', 0.0),
                            open_datetime=open_dt,
                            close_datetime=None,
                            entry_price=entry_price,
                            exit_price=None,
                            entry_value=(entry_price * size) if entry_price else None,
                            exit_value=None,
                            pnl=None,
                            pnl_comm=None,
                        )
                    return
                # 未平仓继续
                if not trade.isclosed:
                    return
                # 平仓
                open_dt = self._fmt_date(getattr(trade, 'dtopen', None))
                close_dt = self._fmt_date(getattr(trade, 'dtclose', None))
                entry_price = None
                exit_price = None
                entry_size = 0
                hist = getattr(trade, 'history', None)
                try:
                    if hist:
                        first_ev = hist[0].event
                        last_ev = hist[-1].event
                        entry_price = getattr(first_ev, 'price', None)
                        exit_price = getattr(last_ev, 'price', None)
                        entry_size = getattr(first_ev, 'size', 0)
                except Exception:
                    pass
                size = abs(entry_size)
                side = 'LONG' if entry_size > 0 else 'SHORT'
                entry_value = (entry_price * size) if (entry_price is not None) else None
                exit_value = (exit_price * size) if (exit_price is not None) else None
                pnl = getattr(trade, 'pnl', None)
                pnl_comm = getattr(trade, 'pnlcomm', None)
                comm = getattr(trade, 'commission', 0.0)
                rec = TradeRecord(
                    datetime=close_dt or open_dt or '',
                    side=side,
                    size=size,
                    price=exit_price or 0.0,
                    value=exit_value or 0.0,
                    commission=comm,
                    open_datetime=open_dt,
                    close_datetime=close_dt,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_value=entry_value,
                    exit_value=exit_value,
                    pnl=pnl,
                    pnl_comm=pnl_comm,
                )
                self._open_cache.pop(id(trade), None)
                self.records.append(rec)

            def stop(self):
                for rec in self._open_cache.values():
                    self.records.append(rec)
                self._open_cache.clear()

            def get_analysis(self):
                return self.records

        # 4.2 订单级捕获（每笔成交）
        class OrderCapture(bt.Analyzer):
            def start(self):
                self.records: List[OrderRecord] = []

            def _fmt_date(self, dt):
                try:
                    return bt.num2date(dt).date().isoformat() if dt else None
                except Exception:
                    return None

            def notify_order(self, order):
                # 只记录完成的成交（可根据需求扩展到部分成交）
                if order.status not in [order.Completed, order.Canceled, order.Rejected, order.Margin]:
                    return
                dt = None
                try:
                    dt = self._fmt_date(order.executed.dt)
                except Exception:
                    dt = None
                # 推断多空方向（正为买，负为卖）
                size = getattr(order.executed, 'size', 0.0)
                price = getattr(order.executed, 'price', 0.0)
                value = getattr(order.executed, 'value', 0.0)
                comm = getattr(order.executed, 'comm', 0.0)
                side = 'BUY' if size >= 0 else 'SELL'
                name = None
                try:
                    data = order.data
                    name = getattr(data, '_name', None)
                except Exception:
                    name = None
                # 将实际成交写入 JQ 日志，方便与期望价格核对
                try:
                    # 构造聚宽风格时间头（09:30:00）
                    ts = f"{dt} 09:30:00" if dt else None
                    line = f"[fill] {side} {name} size={abs(size)} price={price} value={value} commission={comm}"
                    if ts:
                        jq_state['log'].append(f"{ts} - INFO - {line}")
                    else:
                        jq_state['log'].append(line)
                except Exception:
                    pass
                self.records.append(OrderRecord(
                    datetime=dt or '',
                    symbol=name,
                    side=side,
                    size=size,
                    price=price,
                    value=value,
                    commission=comm,
                    status=order.getstatusname(),
                ))

            def get_analysis(self):
                return self.records

        # 5. 注册 analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        cerebro.addanalyzer(TradeCapture, _name='trade_capture')
        cerebro.addanalyzer(OrderCapture, _name='order_capture')

        # 6. 添加策略并运行
        cerebro.addstrategy(StrategyCls, **strategy_params)
        results = cerebro.run()
        strat = results[0]

        # 指标汇总
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio') if hasattr(strat.analyzers, 'sharpe') else None
        dd = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trade_ana = strat.analyzers.trades.get_analysis()
        timereturn = strat.analyzers.timereturn.get_analysis()

        # 策略权益曲线（包含暖场），再切到用户起始日并归一
        tmp_eq = []
        cumulative = 1.0
        for dt, r in timereturn.items():
            cumulative *= (1 + r)
            tmp_eq.append({'date': dt.strftime('%Y-%m-%d'), 'equity': cumulative})
        # 切片到 >= start，并用首日值归一到 1
        eq_filtered = [p for p in tmp_eq if p['date'] >= start]
        if eq_filtered:
            base = eq_filtered[0]['equity'] or 1.0
            equity_curve = [{'date': p['date'], 'equity': (p['equity'] / base if base else p['equity'])} for p in eq_filtered]
        else:
            equity_curve = []

        daily_returns = [{'date': dt.strftime('%Y-%m-%d'), 'ret': r} for dt, r in timereturn.items() if dt.strftime('%Y-%m-%d') >= start]

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
            'symbols_used': symbols,
            'use_real_price': bool(use_real_price_opt),
            'jq_options': jq_state.get('options', {}) if 'jq_state' in locals() else {},
        }

        # 基准 & 超额
        benchmark_curve: List[Dict[str, Any]] = []
        excess_curve: List[Dict[str, Any]] = []
        # 若未显式提供 benchmark_symbol, 现在(策略 initialize 已执行完)再读取 jq_state 里的 benchmark
        if not benchmark_symbol:
            jq_bm_post = jq_state.get('benchmark') if 'jq_state' in locals() else None
            if jq_bm_post:
                benchmark_symbol = _map_benchmark_code(jq_bm_post)
                metrics['benchmark_detect_phase'] = 'post_run_initialize'
            else:
                # 默认基准回退: 若无 set_benchmark 且未传入, 尝试使用 000300_日.csv
                default_bm = '000300_日'
                default_path = os.path.join(datadir, default_bm + '.csv')
                if os.path.exists(default_path):
                    benchmark_symbol = default_bm
                    metrics['benchmark_detect_phase'] = 'fallback_default'
                    metrics['benchmark_fallback'] = True
                else:
                    metrics['benchmark_fallback'] = False
        else:
            metrics['benchmark_detect_phase'] = 'pre_run_explicit'

        if benchmark_symbol:
            bench_df = load_csv_dataframe(benchmark_symbol, start, end, datadir)
            if len(bench_df) == 0:
                metrics['benchmark_final'] = None
                metrics['excess_return'] = None
                metrics['benchmark_missing_reason'] = 'empty_file_or_no_rows_in_range'
            else:
                # 需要 close 列
                bench_df = bench_df[['datetime', 'close']].copy()
                bench_df['ret'] = bench_df['close'].pct_change().fillna(0.0)
                bench_df['equity'] = (1 + bench_df['ret']).cumprod()
                bm_map = {d.strftime('%Y-%m-%d'): (r, eq) for d, r, eq in zip(bench_df['datetime'], bench_df['ret'], bench_df['equity'])}
                overlap_cnt = 0
                for ec in equity_curve:
                    d = ec['date']
                    if d in bm_map:
                        br, beq = bm_map[d]
                        benchmark_curve.append({'date': d, 'equity': beq})
                        excess_curve.append({'date': d, 'excess': ec['equity']/beq - 1 if beq != 0 else 0})
                        overlap_cnt += 1
                if benchmark_curve:
                    metrics['benchmark_final'] = benchmark_curve[-1]['equity']
                    metrics['excess_return'] = equity_curve[-1]['equity']/benchmark_curve[-1]['equity'] - 1
                else:
                    metrics['benchmark_final'] = None
                    metrics['excess_return'] = None
                    # 诊断：无日期重叠
                    metrics['benchmark_missing_reason'] = 'no_overlap_with_strategy_dates'
        else:
            metrics['benchmark_final'] = None
            metrics['excess_return'] = None

        # 记录使用的基准信息
        metrics['benchmark_symbol_used'] = benchmark_symbol
        metrics['benchmark_code'] = jq_state.get('benchmark') if 'jq_state' in locals() else None
        if 'benchmark_detect_phase' not in metrics:
            metrics['benchmark_detect_phase'] = 'none'

        # 回测结束，从 analyzer 里取交易记录 (包含已平仓 + 未平仓) 及订单级别明细
        trades: List[TradeRecord] = strat.analyzers.trade_capture.get_analysis() if hasattr(strat.analyzers, 'trade_capture') else []
        orders: List[OrderRecord] = strat.analyzers.order_capture.get_analysis() if hasattr(strat.analyzers, 'order_capture') else []

        # 聚宽兼容记录日期对齐: 若用户 record 次数与 equity_curve 后段长度匹配, 补齐日期
        jq_records = jq_state.get('records') if 'jq_state' in locals() else None
        if jq_records and equity_curve:
            tail_len = min(len(jq_records), len(equity_curve))
            date_map = equity_curve[-tail_len:]
            # 只给缺失 dt 的填充
            offset = len(jq_records) - tail_len
            for i, r in enumerate(jq_records):
                if r.get('dt') is None and i >= offset:
                    map_idx = i - offset
                    if 0 <= map_idx < tail_len:
                        r['dt'] = date_map[map_idx]['date']
        jq_logs = jq_state.get('log') if 'jq_state' in locals() else None

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_curve=benchmark_curve,
            excess_curve=excess_curve,
            trades=trades,
            orders=orders,
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
