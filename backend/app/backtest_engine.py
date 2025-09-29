import io
import json
import traceback
import math as _math
import re
import types
import os
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Callable

import backtrader as bt
import pandas as pd
from .corporate_actions import load_corporate_actions, apply_event


def _round_to_tick(value: float, tick: float) -> float:
    """Quantize price to given tick using half-up rounding (A-share style)."""
    if tick is None or tick <= 0:
        return value
    try:
        return float(Decimal(str(value)).quantize(Decimal(str(tick)), rounding=ROUND_HALF_UP))
    except Exception:
        return value

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
    daily_pnl: List[Dict[str, Any]]  # 新增：每日盈亏（金额）
    daily_turnover: List[Dict[str, Any]]  # 新增：每日买卖额/开平仓计数
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
        'options': {
            'enable_limit_check': True,
            'limit_up_factor': 1.10,
            'limit_down_factor': 0.90,
            'price_tick': 0.01,
        },
        'records': [],
        'log': [],
        'g': g,
        # 运行辅助状态
        'history_df': None,      # 单标的完整历史（DataFrame，含 datetime 列）
        'history_df_map': {},    # 多标的缓存
        'current_dt': None,      # 当前 bar 展示时间（字符串）
        'user_start': None,      # 用户选择的开始日期（YYYY-MM-DD）
        'in_warmup': False,      # 暖场阶段（< user_start）
        'blocked_orders': [],  # 收集被涨跌停或规则阻断的订单（size=0）
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

    # ---- 聚宽风格滑点支持 ----
    # 语义：若用户未显式设置，则默认使用 PriceRelatedSlippage(0.00246) ≈ 0.246%
    # set_slippage(FixedSlippage(0.02))        -> 固定绝对价差 0.02 （按简单: open/close +/-0.02）
    # set_slippage(PriceRelatedSlippage(0.00246)) -> 百分比滑点，最终买入价 = 基准价*(1+perc/2) 卖出价 = 基准价*(1-perc/2) 的近似；
    # 这里我们简化：买入使用 price*(1+perc) 卖出使用 price*(1-perc)，并在订单估算中采用原价，撮合时通过 broker slippage 设置。
    # backtrader 自带 set_slippage_perc 为单方向百分比增量（默认对成交价格直接乘 (1 + perc)）；
    # 我们约定：PriceRelatedSlippage(p) -> 传入 perc=p；FixedSlippage 暂用 options.fixed_slippage 记录（可后续自定义执行层）。

    def set_slippage(obj=None, type=None, ref=None):  # noqa: A002 (聚宽接口命名保持)
        try:
            # 解析两种文本形式：FixedSlippage(x) / PriceRelatedSlippage(y)
            if obj is None:
                return
            s = str(obj)
            import re as _re
            m_fixed = _re.search(r'FixedSlippage\s*\(\s*([0-9eE\.+-]+)\s*\)', s)
            m_price = _re.search(r'PriceRelatedSlippage\s*\(\s*([0-9eE\.+-]+)\s*\)', s)
            if m_price:
                val = float(m_price.group(1))
                jq_state['options']['slippage_perc'] = val
                jq_state['log'].append(f"[set_slippage] PriceRelatedSlippage perc={val}")
            elif m_fixed:
                val = float(m_fixed.group(1))
                jq_state['options']['fixed_slippage'] = val
                jq_state['log'].append(f"[set_slippage] FixedSlippage value={val}")
            else:
                # 允许直接给数字（按百分比）
                try:
                    val = float(obj)
                    jq_state['options']['slippage_perc'] = val
                    jq_state['log'].append(f"[set_slippage] perc={val}")
                except Exception:
                    jq_state['log'].append(f"[set_slippage_warning] unrecognized={s}")
        except Exception:
            pass

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
        'set_slippage': set_slippage,
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
                # 绑定执行环境引用，供 _run_handle / next_open / next 使用
                try:
                    self._exec_env = globals().get('exec_env') or exec_env  # fallback
                except Exception:
                    self._exec_env = {'jq_state': jq_state}
                # 运行 initialize
                init_func(self._jq_context)

            def prenext(self):
                # 允许 MA 等指标还未就绪也执行
                self.next()

            def _run_handle(self):
                # 提供 attribute_history, order_value, order_target 实现
                # 先获取 exec_env/jq_state 以便后续 CA 应用
                exec_env = getattr(self, '_exec_env', None)
                if exec_env is None:
                    exec_env = {}
                jq_state = exec_env.get('jq_state', {}) if isinstance(exec_env, dict) else {}
                # 注入 get_price (简化版) 若未提供
                if isinstance(exec_env, dict) and 'get_price' not in exec_env:
                    def get_price(security: str,
                                  count: int = 1,
                                  end_date: str | None = None,
                                  frequency: str = 'daily',
                                  fields=None):
                        """简化版聚宽 get_price：仅支持 daily & 单标的。
                        security: 证券代码(忽略后缀)
                        count: 取最近 count 条（不含当前正在运行的当日 bar）
                        end_date: 忽略或与当前回测日不同则仍以当前回测日为结束(前一日)
                        fields: None/单字段/字段列表；默认返回 ['open','close','high','low','volume']。
                        返回 pandas.DataFrame 或 Series（当 fields 为单字段时）。
                        """
                        import pandas as _pd
                        if frequency != 'daily':
                            raise ValueError('get_price 简化实现仅支持 daily')
                        if count <= 0:
                            return _pd.DataFrame()
                        base = str(security).split('.')[0]
                        hist_map = jq_state.get('history_df_map') or {}
                        df_full = None
                        if base in hist_map:
                            df_full = hist_map[base]
                        else:
                            for k, v in hist_map.items():
                                if str(k).startswith(base):
                                    df_full = v
                                    break
                        if df_full is None:
                            df_full = jq_state.get('history_df')  # 可能是单一标的 DataFrame
                        # 需要 datetime 列
                        if df_full is None or 'datetime' not in df_full.columns:
                            return _pd.DataFrame()
                        # 当前模拟日
                        try:
                            cur_date = bt.num2date(self.data.datetime[0]).date()
                        except Exception:
                            cur_date = None
                        work = df_full
                        if cur_date is not None:
                            work = work[work['datetime'].dt.date < cur_date]
                        if work.empty:
                            return _pd.DataFrame()
                        tail = work.tail(count)
                        default_fields = ['open','close','high','low','volume']
                        if fields is None:
                            use_fields = default_fields
                        else:
                            if isinstance(fields, str):
                                use_fields = [fields]
                            else:
                                use_fields = list(fields)
                        cols_exist = [c for c in use_fields if c in tail.columns]
                        out = tail[['datetime', *cols_exist]].copy()
                        out.index = out['datetime'].dt.date.astype(str)
                        out = out.drop(columns=['datetime'])
                        if isinstance(fields, str):
                            # 返回 Series
                            s = out[fields] if fields in out.columns else _pd.Series([], dtype=float)
                            s.index.name = None
                            return s
                        out.index.name = None
                        return out
                    exec_env['get_price'] = get_price
                # --- Corporate Actions (simulate_corporate_actions) ---
                try:
                    if jq_state.get('corporate_actions'):
                        cur_dt = bt.num2date(self.data.datetime[0]).date().isoformat()
                        # 找出当天所有事件
                        todays = [e for e in jq_state['corporate_actions'] if e.date == cur_dt]
                        if todays:
                            logger = jq_state.get('logger')
                            for ev in todays:
                                from .corporate_actions import apply_event as _apply_ca
                                manual_override = getattr(ev, '_manual_shares', None)
                                if manual_override not in (None, 0):
                                    # Direct share delta injection; positive -> buy, negative -> sell
                                    delta = int(manual_override)
                                    try:
                                        if logger:
                                            logger.info(f"[ca_manual_apply] date={ev.date} type={ev.action_type} delta={delta}")
                                    except Exception:
                                        pass
                                    if delta > 0:
                                        self.buy(size=delta)
                                    elif delta < 0:
                                        self.sell(size=abs(delta))
                                else:
                                    res = _apply_ca(ev, self.position, self.broker, logger=logger)
                                    if res and isinstance(res, tuple):
                                        tag, payload = res
                                        if tag in ('BONUS_SHARES','SPLIT_ADJ') and isinstance(payload, int) and payload != 0:
                                            if payload > 0:
                                                self.buy(size=payload)
                                            else:
                                                self.sell(size=abs(payload))
                except Exception:
                    pass
                # 继续原有逻辑
                def attribute_history(security: str, n: int, unit: str, fields: List[str]):
                    # 支持 unit 为 '1d' 或 '1D' 或 'day'
                    if unit.lower() not in ('1d', 'day', 'd'):
                        raise ValueError('attribute_history 目前仅支持日级 unit=1d')
                    import pandas as _pd
                    # 方案A: 增加开关 attribute_history_include_current (默认 False)，
                    # True 时包含“当前bar”(即当前交易日)；False 保持聚宽语义(不含当日)
                    try:
                        _include_cur = bool(exec_env['jq_state']['options'].get('attribute_history_include_current', False))
                    except Exception:
                        _include_cur = False
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
                        # 根据开关决定是否包含当日
                        if _include_cur:
                            df = full_df[full_df['datetime'].dt.date <= cur_date]
                        else:
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
                    # 根据开关决定 available 是否包含当前 bar
                    if _include_cur:
                        available = max(len(self.data), 0)  # 允许包含当前bar
                    else:
                        available = max(len(self.data) - 1, 0)  # 不包含当前bar（索引0）
                    length = min(n, available)
                    if length <= 0:
                        return _JQDataFrame({f: [] for f in fields})
                    data_dict: Dict[str, List[float]] = {}
                    for f in fields:
                        line = getattr(self.data, f, None)
                        if line is None:
                            data_dict[f] = [float('nan')] * length
                            continue
                        # 取“之前”的 length 条；含当日则区间为 [-length+1, ..., 0]
                        if _include_cur:
                            start_idx = -length + 1 if length > 0 else 1
                            vals = [line[i] for i in range(start_idx, 1)] if length > 0 else []
                        else:
                            # 不含当日：[-length, ..., -1]
                            vals = [line[i] for i in range(-length, 0)] if length > 0 else []
                        data_dict[f] = vals
                    # 构造日期索引（更稳健：逐条读取末尾 length 个 bar 的日期，排除当日）
                    try:
                        # 使用索引逐条转换，包含/不包含当日由 _include_cur 控制
                        str_idx = []
                        if _include_cur:
                            for i in range(-length + 1, 1):
                                dt_num = self.data.datetime[i]
                                dt_str = bt.num2date(dt_num).date().isoformat()
                                str_idx.append(dt_str)
                        else:
                            for i in range(-length, 0):
                                dt_num = self.data.datetime[i]
                                dt_str = bt.num2date(dt_num).date().isoformat()
                                str_idx.append(dt_str)
                    except Exception:
                        # 仍然失败则退回负索引
                        if _include_cur:
                            str_idx = list(range(-length + 1, 1))
                        else:
                            str_idx = list(range(-length, 0))
                    return _JQDataFrame(_pd.DataFrame(data_dict, index=str_idx))

                def order_value(security: str, value: float):
                    """按金额下单（聚宽语义近似）：
                    value > 0  -> 期望用不超过 value 的现金买入；若 value 超过可用现金则用全部可用现金。
                    value < 0  -> 期望按金额卖出（减仓）不超过 abs(value)；受持仓限制。

                    与原简单实现差异：
                    1. 计算时考虑佣金率/最小佣金，确保 (成交额+佣金)<=目标金额；
                    2. 买入时不超出可用现金 (cash)；
                    3. 卖出时不超过当前持仓市值；
                    4. 使用 lot(默认100) 对齐；
                    5. 记录详细计算日志，便于对齐聚宽第二笔差异。
                    """
                    try:
                        # 严格暖场禁止交易（不记录 blocked，直接忽略）
                        if exec_env['jq_state'].get('in_warmup'):
                            return
                        jqst = exec_env['jq_state']
                        strict = bool(jqst['options'].get('jq_order_mode_strict', False))
                        enable_limit = bool(jqst['options'].get('enable_limit_check', False))
                        up_lim_fac = float(jqst['options'].get('limit_up_factor', 1.10))
                        down_lim_fac = float(jqst['options'].get('limit_down_factor', 0.90))
                        fill_price = str(jqst['options'].get('fill_price', 'open')).lower()
                        if fill_price == 'close':
                            base_price = float(getattr(self.data, 'close')[0])
                        else:
                            base_price = float(getattr(self.data, 'open')[0]) if hasattr(self.data, 'open') else float(getattr(self.data, 'close')[0])
                        if base_price <= 0:
                            return
                        slip_perc = float(jqst['options'].get('slippage_perc', 0.0) or 0.0)
                        # 方案2 半滑点：内部执行价 = base_price*(1±slip/2) 后买向下截断 / 卖向上进位到分
                        half = slip_perc / 2.0
                        # use module-level _math
                        def _round_buy(p: float) -> float:
                            return _math.floor(p * 100) / 100.0
                        def _round_sell(p: float) -> float:
                            return _math.ceil(p * 100) / 100.0
                        price = base_price
                        debug_trading = bool(jqst['options'].get('debug_trading', False))
                        buy_eff_price = _round_buy(base_price * (1 + half))
                        sell_eff_price = _round_sell(base_price * (1 - half))
                        # 涨跌停判定（简单：比较上一交易日收盘）
                        if enable_limit:
                            try:
                                prev_close = float(getattr(self.data, 'close')[-1])  # -1 为上一日 (backtrader 最后一根历史)
                            except Exception:
                                prev_close = None
                            if prev_close and prev_close > 0:
                                tick = float(jqst['options'].get('price_tick', 0.01) or 0.01)
                                up_lim = _round_to_tick(prev_close * up_lim_fac, tick)
                                down_lim = _round_to_tick(prev_close * down_lim_fac, tick)
                                side_tmp = 'BUY' if value >= 0 else 'SELL'
                                eff_price = buy_eff_price if side_tmp == 'BUY' else sell_eff_price
                                if side_tmp == 'BUY' and eff_price >= up_lim - 1e-9:
                                    exec_env['log'].info(
                                        f"[limit_check] BLOCK side=BUY price={eff_price:.4f} up_lim={up_lim:.4f} prev_close={prev_close:.4f}"
                                    )
                                    jqst['blocked_orders'].append(OrderRecord(
                                        datetime=jqst.get('current_dt','').split(' ')[0],
                                        symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                        side='BUY',
                                        size=0,
                                        price=eff_price,
                                        value=0.0,
                                        commission=0.0,
                                        status='BlockedLimitUp'
                                    ))
                                    return
                                if side_tmp == 'SELL' and eff_price <= down_lim + 1e-9:
                                    exec_env['log'].info(
                                        f"[limit_check] BLOCK side=SELL price={eff_price:.4f} down_lim={down_lim:.4f} prev_close={prev_close:.4f}"
                                    )
                                    jqst['blocked_orders'].append(OrderRecord(
                                        datetime=jqst.get('current_dt','').split(' ')[0],
                                        symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                        side='SELL',
                                        size=0,
                                        price=eff_price,
                                        value=0.0,
                                        commission=0.0,
                                        status='BlockedLimitDown'
                                    ))
                                    return
                        # ---- 真实下单计算 ----
                        lot = int(jqst['options'].get('lot', 100)) or 100
                        commission_rate = float(jqst['options'].get('commission', 0.0003))
                        min_comm = float(jqst['options'].get('min_commission', 5.0)) if 'min_commission' in jqst['options'] else 5.0
                        stamp_duty = float(jqst['options'].get('stamp_duty', 0.001)) if 'stamp_duty' in jqst['options'] else 0.001
                        available_cash = self.broker.getcash()
                        position_size = int(getattr(self.position, 'size', 0) or 0)
                        # 买入金额
                        if value > 0:
                            # 先用买入有效价估算股数
                            est_price = buy_eff_price
                            # 可选“保守模式”：若上一日收盘(prev_close)高于买入有效价，用 prev_close 作为 sizing 基准，避免实际撮合价>估价导致保证金 0。
                            # 聚宽本身按 (cash // buy_price) 取整再对齐 lot；这里通过开关控制：True=当前默认保守, False=严格 JQ 样式。
                            conservative_flag = True
                            try:
                                opt_cons = jqst['options'].get('order_value_conservative_prev_close')
                                if isinstance(opt_cons, bool):
                                    conservative_flag = opt_cons
                            except Exception:
                                conservative_flag = True
                            conservative_price = est_price
                            prev_close_for_size = None
                            try:
                                prev_close_for_size = float(getattr(self.data, 'close')[-1])
                            except Exception:
                                prev_close_for_size = None
                            if conservative_flag and prev_close_for_size and prev_close_for_size > conservative_price:
                                conservative_price = prev_close_for_size
                            raw_shares = int(value // (conservative_price if conservative_flag else est_price))
                            # 按 lot 对齐
                            shares = (raw_shares // lot) * lot
                            if shares <= 0:
                                if debug_trading:
                                    exec_env['log'].info(f"[sizing] BUY abort raw_shares={raw_shares} lot={lot} est={est_price} conservative={conservative_price} cash={available_cash}")
                                return
                            # 费用估算（迭代一次足够）
                            gross_price_for_fee = conservative_price if conservative_flag else est_price
                            gross = shares * gross_price_for_fee
                            comm = max(gross * commission_rate, min_comm)
                            total_cost = gross + comm  # 不含印花税（买入无印花税）
                            if total_cost > available_cash:
                                # 缩减 shares
                                max_afford = int((available_cash - min_comm) // gross_price_for_fee)
                                shares = (max_afford // lot) * lot
                            if shares <= 0:
                                if debug_trading:
                                    exec_env['log'].info(f"[sizing] BUY zero_after_cash max_afford={max_afford} cons_price={gross_price_for_fee} cash={available_cash}")
                                return
                            if debug_trading:
                                exec_env['log'].info(
                                    f"[sizing] BUY shares={shares} est={est_price} prev_close={prev_close_for_size} cons_flag={conservative_flag} cons_price={conservative_price} gross_price_used={gross_price_for_fee} gross={gross:.2f} comm_est={comm:.2f} total={total_cost:.2f} cash={available_cash:.2f}"
                                )
                            else:
                                # 非 debug 也记录一次模式
                                try:
                                    jqst['log'].append(f"[sizing_mode] conservative_prev_close={conservative_flag}")
                                except Exception:
                                    pass
                            self.buy(size=shares)
                        elif value < 0:
                            # 卖出金额：目标卖出 value_abs ，不超过持仓
                            est_price = sell_eff_price
                            value_abs = abs(value)
                            max_sell_shares = position_size if position_size > 0 else 0
                            if max_sell_shares <= 0:
                                return
                            # 需要的股数
                            raw_need = int(value_abs // est_price)
                            shares = min(max_sell_shares, (raw_need // lot) * lot if raw_need > 0 else 0)
                            if shares <= 0:
                                # 若希望彻底清仓且金额较小，允许直接全部平掉（与 value 小但有持仓的情况）
                                if value_abs >= est_price * lot and strict:
                                    return
                                shares = max_sell_shares
                            if shares <= 0:
                                if debug_trading:
                                    exec_env['log'].info(f"[sizing] SELL abort raw_need={raw_need} lot={lot} pos={position_size} est={est_price}")
                                return
                            if debug_trading:
                                exec_env['log'].info(f"[sizing] SELL shares={shares} est={est_price} pos={position_size}")
                            self.sell(size=shares)
                    except Exception as _e:
                        try:
                            exec_env['log'].info(f"[order_value_error] {type(_e).__name__}:{_e}")
                        except Exception:
                            pass
                    # end order_value
                def order_target(security: str, target: float):
                    # 暖场期不交易（严格忽略）
                    if exec_env['jq_state'].get('in_warmup'):
                        return
                    """将当前持仓调整到目标股数 target。
                    - target 为目标持仓股数（可为浮点，将被取整）
                    - delta > 0 执行买入 delta 股；delta < 0 执行卖出 |delta| 股
                    - target == 0 等价于平仓
                    """
                    try:
                        jqst = exec_env['jq_state']
                        strict = bool(jqst['options'].get('jq_order_mode_strict', False))
                        enable_limit = bool(jqst['options'].get('enable_limit_check', False))
                        up_lim_fac = float(jqst['options'].get('limit_up_factor', 1.10))
                        down_lim_fac = float(jqst['options'].get('limit_down_factor', 0.90))
                        fill_price = str(jqst['options'].get('fill_price', 'open')).lower()
                        if fill_price == 'close':
                            base_price = float(getattr(self.data, 'close')[0])
                        else:
                            base_price = float(getattr(self.data, 'open')[0]) if hasattr(self.data, 'open') else float(getattr(self.data, 'close')[0])
                        if base_price <= 0:
                            return
                        slip_perc = float(jqst['options'].get('slippage_perc', 0.0) or 0.0)
                        half = slip_perc / 2.0
                        # use module-level _math
                        def _round_buy(p: float) -> float:
                            return _math.floor(p * 100) / 100.0
                        def _round_sell(p: float) -> float:
                            return _math.ceil(p * 100) / 100.0
                        buy_eff_price = _round_buy(base_price * (1 + half))
                        sell_eff_price = _round_sell(base_price * (1 - half))
                        price = base_price
                        # limit check
                        prev_close = None
                        up_lim = down_lim = None
                        if enable_limit:
                            try:
                                prev_close = float(getattr(self.data, 'close')[-1])
                            except Exception:
                                prev_close = None
                            if prev_close and prev_close > 0:
                                tick = float(jqst['options'].get('price_tick', 0.01) or 0.01)
                                up_lim = _round_to_tick(prev_close * up_lim_fac, tick)
                                down_lim = _round_to_tick(prev_close * down_lim_fac, tick)
                        cur = int(getattr(self.position, 'size', 0) or 0)
                        tgt_raw = int(target or 0)
                        lot = int(jqst['options'].get('lot', 100)) or 100
                        tgt = (abs(tgt_raw) // lot) * lot
                        tgt = tgt if tgt_raw >= 0 else -tgt
                        delta = tgt - cur
                        if delta == 0:
                            return
                        if enable_limit and prev_close and prev_close > 0:
                            if delta > 0 and up_lim is not None and buy_eff_price >= up_lim - 1e-9:
                                exec_env['log'].info(
                                    f"[limit_check] BLOCK side=BUY price={buy_eff_price:.4f} up_lim={up_lim:.4f} prev_close={prev_close:.4f}"
                                )
                                jqst['blocked_orders'].append(OrderRecord(
                                    datetime=jqst.get('current_dt','').split(' ')[0],
                                    symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                    side='BUY',
                                    size=0,
                                    price=buy_eff_price,
                                    value=0.0,
                                    commission=0.0,
                                    status='BlockedLimitUp'
                                ))
                                return
                            if delta < 0 and down_lim is not None and sell_eff_price <= down_lim + 1e-9:
                                exec_env['log'].info(
                                    f"[limit_check] BLOCK side=SELL price={sell_eff_price:.4f} down_lim={down_lim:.4f} prev_close={prev_close:.4f}"
                                )
                                jqst['blocked_orders'].append(OrderRecord(
                                    datetime=jqst.get('current_dt','').split(' ')[0],
                                    symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                    side='SELL',
                                    size=0,
                                    price=sell_eff_price,
                                    value=0.0,
                                    commission=0.0,
                                    status='BlockedLimitDown'
                                ))
                                return
                        # --- 真正下单 ---
                        if delta > 0:
                            self.buy(size=delta)
                        else:
                            self.sell(size=abs(delta))
                    except Exception as _e:
                        try:
                            exec_env['log'].info(f"[order_target_error] {type(_e).__name__}:{_e}")
                        except Exception:
                            pass
                    # end order_target
                # 暴露函数到执行环境（供用户脚本中的全局方法调用）
                try:
                    exec_env['attribute_history'] = attribute_history
                    exec_env['order_value'] = order_value
                    exec_env['order_target'] = order_target
                except Exception:
                    pass
                # 记录执行模式（open/close）。fill_price!=close -> 在 nextopen 中执行用户逻辑，使下单在当日开盘撮合。
                try:
                    fp_mode = str(exec_env.get('jq_state', {}).get('options', {}).get('fill_price', 'open')).lower()
                except Exception:
                    fp_mode = 'open'
                self._fill_price_mode = fp_mode
                self._run_on_open = (fp_mode != 'close')  # 默认 open
                # 标记当前 bar 是否已在 open 阶段处理
                self._handled_today = False
                try:
                    exec_env['jq_state']['log'].append(f"[exec_mode] run_on_open={self._run_on_open} fill_price={fp_mode}")
                except Exception:
                    pass
                # 实际调用用户 handle_data
                try:
                    handle_func(self._jq_context, None)
                except Exception as _e:
                    try:
                        exec_env['log'].info(f"[handle_data_error] {type(_e).__name__}:{_e}")
                    except Exception:
                        pass

            # 在开盘阶段执行（Backtrader cheat-on-open 钩子: next_open）
            def next_open(self):  # type: ignore
                if not getattr(self, '_run_on_open', True):
                    return  # 仅 open 模式执行
                try:
                    jq_state = self._exec_env.get('jq_state') if hasattr(self, '_exec_env') else None
                    if isinstance(jq_state, dict):
                        cur_date_obj = bt.num2date(self.data.datetime[0]).date()
                        cur_dt = cur_date_obj.isoformat()
                        jq_state['current_dt'] = f"{cur_dt} 09:30:00"
                        user_start = jq_state.get('user_start')
                        in_warmup_before = jq_state.get('in_warmup')
                        if isinstance(user_start, str):
                            try:
                                from datetime import date as _d
                                start_date_obj = _d.fromisoformat(user_start)
                                jq_state['in_warmup'] = cur_date_obj < start_date_obj
                            except Exception:
                                jq_state['in_warmup'] = cur_dt < user_start
                        else:
                            jq_state['in_warmup'] = False
                        if jq_state['in_warmup'] and not in_warmup_before:
                            try:
                                jq_state['log'].append(f"[warmup] enter_warmup cur={cur_dt} start={user_start}")
                            except Exception:
                                pass
                        if (not jq_state['in_warmup']) and in_warmup_before:
                            try:
                                jq_state['log'].append(f"[warmup] leave_warmup cur={cur_dt} start={user_start}")
                            except Exception:
                                pass
                        if jq_state['in_warmup']:
                            return
                except Exception:
                    pass
                # 真正执行用户逻辑（基于前一日收盘生成信号，按当日开盘成交）
                try:
                    self._run_handle()
                    self._handled_today = True
                except Exception:
                    pass

            def next(self):  # 每个 bar 调用
                # close 模式或未在 open 阶段执行时才在这里运行用户逻辑
                if getattr(self, '_run_on_open', True) and getattr(self, '_handled_today', False):
                    return
                try:
                    jq_state = self._exec_env.get('jq_state') if hasattr(self, '_exec_env') else None
                    if isinstance(jq_state, dict):
                        cur_date_obj = bt.num2date(self.data.datetime[0]).date()
                        cur_dt = cur_date_obj.isoformat()
                        jq_state['current_dt'] = f"{cur_dt} 09:30:00"
                        user_start = jq_state.get('user_start')
                        in_warmup_before = jq_state.get('in_warmup')
                        if isinstance(user_start, str):
                            try:
                                from datetime import date as _d
                                start_date_obj = _d.fromisoformat(user_start)
                                jq_state['in_warmup'] = cur_date_obj < start_date_obj
                            except Exception:
                                jq_state['in_warmup'] = cur_dt < user_start
                        else:
                            jq_state['in_warmup'] = False
                        if jq_state['in_warmup'] and not in_warmup_before:
                            try:
                                jq_state['log'].append(f"[warmup] enter_warmup cur={cur_dt} start={user_start}")
                            except Exception:
                                pass
                        if (not jq_state['in_warmup']) and in_warmup_before:
                            try:
                                jq_state['log'].append(f"[warmup] leave_warmup cur={cur_dt} start={user_start}")
                            except Exception:
                                pass
                        if jq_state['in_warmup']:
                            return
                except Exception:
                    pass
                try:
                    self._run_handle()
                except Exception:
                    pass

        return UserStrategy, jq_state

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
        # 记录用户开始日期
        jq_state['user_start'] = start
        # 预扫描用户代码：因为 initialize 在数据加载之后才执行，若用户在 initialize 内才 set_option('use_real_price',True)
        # 会错过“选 raw 文件”阶段，这里用静态正则提前检测一次并预置 option；同理支持 adjust_type。
        try:
            # 仅当尚未显式设置 use_real_price 时才尝试推断
            if 'use_real_price' not in jq_state.get('options', {}):
                if re.search(r"set_option\(\s*['\"]use_real_price['\"]\s*,\s*True", strategy_code):
                    jq_state['options']['use_real_price'] = True
                    jq_state['log'].append('[preparse] detected use_real_price=True in source code')
            # adjust_type 同理（只解析 raw/qfq/hfq/auto 简单字面值）
            if 'adjust_type' not in jq_state.get('options', {}):
                m_adj = re.search(r"set_option\(\s*['\"]adjust_type['\"]\s*,\s*['\"](raw|qfq|hfq|auto)['\"]", strategy_code, re.IGNORECASE)
                if m_adj:
                    jq_state['options']['adjust_type'] = m_adj.group(1).lower()
                    jq_state['log'].append(f"[preparse] detected adjust_type={m_adj.group(1).lower()} in source code")
        except Exception:
            pass
        # 创建 cerebro（根据成交价类型决定 cheat_on_open）
        fill_price_opt = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
        try:
            cerebro = bt.Cerebro(cheat_on_open=(fill_price_opt == 'open'))
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open={fill_price_opt == 'open'}")
        except Exception:
            cerebro = bt.Cerebro()
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open=unsupported")
        cerebro.broker.setcash(cash)
        # 全局买卖限价拦截（支持 blocked_orders 记录）
        try:
            enable_limit_glob = bool(jq_state['options'].get('enable_limit_check', True))
            if enable_limit_glob:
                up_fac_glob = float(jq_state['options'].get('limit_up_factor', 1.10))
                down_fac_glob = float(jq_state['options'].get('limit_down_factor', 0.90))
                price_tick = float(jq_state['options'].get('price_tick', 0.01) or 0.01)
                if not jq_state.get('_global_limit_wrapped'):
                    jq_state['_global_limit_wrapped'] = True
                    _orig_buy = bt.Strategy.buy
                    _orig_sell = bt.Strategy.sell
                    def _limit_guard(strategy_self, *a, **kw):
                        try:
                            if jq_state.get('in_warmup'):
                                return None
                            data = strategy_self.data
                            fillp = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
                            cur_price = float(getattr(data, 'close')[0]) if fillp == 'close' else float(getattr(data, 'open')[0]) if hasattr(data,'open') else float(getattr(data,'close')[0])
                            try:
                                prev_close = float(getattr(data, 'close')[-1])
                            except Exception:
                                prev_close = None
                            if prev_close and prev_close > 0:
                                up_lim = _round_to_tick(prev_close * up_fac_glob, price_tick)
                                if cur_price >= up_lim - 1e-9:
                                    jq_state['log'].append(f"[limit_check_global] BLOCK BUY cur={cur_price:.4f} up={up_lim:.4f} prev_close={prev_close:.4f}")
                                    jq_state['blocked_orders'].append(OrderRecord(datetime=jq_state.get('current_dt','').split(' ')[0], symbol=getattr(jq_state.get('g'),'security',None) or 'data0', side='BUY', size=0, price=cur_price, value=0.0, commission=0.0, status='BlockedLimitUp'))
                                    return None
                        except Exception:
                            pass
                        return _orig_buy(strategy_self, *a, **kw)
                    def _limit_guard_sell(strategy_self, *a, **kw):
                        try:
                            if jq_state.get('in_warmup'):
                                return None
                            data = strategy_self.data
                            fillp = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
                            cur_price = float(getattr(data, 'close')[0]) if fillp == 'close' else float(getattr(data, 'open')[0]) if hasattr(data,'open') else float(getattr(data,'close')[0])
                            try:
                                prev_close = float(getattr(data, 'close')[-1])
                            except Exception:
                                prev_close = None
                            if prev_close and prev_close > 0:
                                down_lim = _round_to_tick(prev_close * down_fac_glob, price_tick)
                                if cur_price <= down_lim + 1e-9:
                                    jq_state['log'].append(f"[limit_check_global] BLOCK SELL cur={cur_price:.4f} down={down_lim:.4f} prev_close={prev_close:.4f}")
                                    jq_state['blocked_orders'].append(OrderRecord(datetime=jq_state.get('current_dt','').split(' ')[0], symbol=getattr(jq_state.get('g'),'security',None) or 'data0', side='SELL', size=0, price=cur_price, value=0.0, commission=0.0, status='BlockedLimitDown'))
                                    return None
                        except Exception:
                            pass
                        return _orig_sell(strategy_self, *a, **kw)
                    bt.Strategy.buy = _limit_guard  # type: ignore
                    bt.Strategy.sell = _limit_guard_sell  # type: ignore
                    jq_state['log'].append('[limit_check_global] monkeypatch buy/sell installed')
        except Exception:
            pass

        # 2. 标的解析与数据加载 (精简 & 明确)
        # 目标：
        #  1) 根据输入代码(可含 .XSHE/.XSHG 或带 *_daily_qfq 等后缀) 解析出核心代码 core (纯数字部分)。
        #  2) 按统一的“候选后缀优先级”生成文件名候选列表并选择第一个存在的。
        #  3) 优先级来源 = adjust_type + use_real_price + 用户 data_source_preference + force_data_variant + respect_symbol_suffix。
        #  4) use_real_price=True: 未复权(raw)优先；False: 前复权(qfq)优先（保持之前语义）。
        #  5) adjust_type 显式指定则覆盖 use_real_price 的默认推导。
        #  6) respect_symbol_suffix=True 时，如果用户直接给了带后缀名字，直接用（存在则用，不存在报 fallback 说明）。
        # 输出日志：
        #  [symbol_select] core=... adjust=... use_real_price=... respect_suffix=... candidates=... chosen=...
        #  [symbol_force]  若 force_data_variant 命中
        #  [symbol_warn]   未找到任何存在文件（将返回首个候选继续让后续 FileNotFound 暴露）

        _SUFFIX_RAW = ['_daily', '_日']
        _SUFFIX_QFQ = ['_daily_qfq', '_日_qfq']
        _SUFFIX_HFQ = ['_daily_hfq', '_日_hfq']
        _ALL_SUFFIXES = _SUFFIX_RAW + _SUFFIX_QFQ + _SUFFIX_HFQ

        def _opt(name: str, default=None):
            try:
                return jq_state.get('options', {}).get(name, default)
            except Exception:
                return default

        def _normalize_code(code: str) -> str:
            c = code.strip()
            c = c.replace('.XSHE', '').replace('.XSHG', '').replace('.xshe', '').replace('.xshg', '')
            for suf in sorted(_ALL_SUFFIXES, key=len, reverse=True):
                if c.lower().endswith(suf):
                    return c[:-len(suf)], suf  # (core, provided_suffix)
            return c, ''

        def _decide_adjust() -> str:
            adj = _opt('adjust_type')
            if isinstance(adj, str) and adj.lower() in ('raw','qfq','hfq','auto'):
                return adj.lower()
            return 'auto'

        def _candidate_suffix_sequence(use_real_price_flag: bool, adjust_type: str) -> List[str]:
            # adjust_type 显式优先
            if adjust_type == 'raw':
                base_seq = _SUFFIX_RAW + _SUFFIX_QFQ + _SUFFIX_HFQ
            elif adjust_type == 'qfq':
                base_seq = _SUFFIX_QFQ + _SUFFIX_RAW + _SUFFIX_HFQ
            elif adjust_type == 'hfq':
                base_seq = _SUFFIX_HFQ + _SUFFIX_RAW + _SUFFIX_QFQ
            else:  # auto
                if use_real_price_flag:
                    base_seq = _SUFFIX_RAW + _SUFFIX_QFQ + _SUFFIX_HFQ
                else:
                    base_seq = _SUFFIX_QFQ + _SUFFIX_RAW + _SUFFIX_HFQ
            # 用户 preference 追加（末尾去重保序）
            user_pref = _opt('data_source_preference')
            if isinstance(user_pref, (list, tuple)):
                seq = []
                seen = set()
                for suf in list(base_seq) + list(user_pref):
                    if suf not in seen and isinstance(suf, str):
                        seen.add(suf)
                        seq.append(suf)
                return seq
            return list(base_seq)

        def _apply_force_variant(core: str) -> Optional[str]:
            force = _opt('force_data_variant')
            if isinstance(force, str):
                fmap = {'daily':'_daily','raw':'_daily','qfq':'_daily_qfq','hfq':'_daily_hfq','日':'_日'}
                suf = fmap.get(force.lower().strip())
                if suf:
                    candidate = core + suf
                    path = os.path.join(datadir, candidate + '.csv')
                    if os.path.exists(path):
                        jq_state['log'].append(f"[symbol_force] using={candidate}.csv force_data_variant={force}")
                        return candidate
                    else:
                        jq_state['log'].append(f"[symbol_force] missing={candidate}.csv (will fallback)")
            return None

        def _select_one(raw_code: str) -> str:
            core, provided_suffix = _normalize_code(raw_code)
            respect = bool(_opt('respect_symbol_suffix', False))
            use_real_price_flag = bool(_opt('use_real_price', False))
            strict_real_price_flag = bool(_opt('strict_real_price', False))
            adjust = _decide_adjust()
            # 如果用户提供后缀且 respect=True 直接使用
            if provided_suffix and respect:
                direct = core + provided_suffix
                path = os.path.join(datadir, direct + '.csv')
                if os.path.exists(path):
                    jq_state['log'].append(
                        f"[symbol_select] mode=respect direct={direct} adjust={adjust} use_real_price={use_real_price_flag}"
                    )
                    return direct
                else:
                    jq_state['log'].append(
                        f"[symbol_select] mode=respect_missing direct={direct} adjust={adjust} use_real_price={use_real_price_flag}"
                    )
            # force_data_variant 优先
            forced = _apply_force_variant(core)
            if forced:
                return forced
            # 生成候选
            suffix_seq = _candidate_suffix_sequence(use_real_price_flag, adjust)
            candidates = [core + suf for suf in suffix_seq]
            chosen = None
            existence = []
            for name in candidates:
                exists = os.path.exists(os.path.join(datadir, name + '.csv'))
                existence.append(f"{name}:{'Y' if exists else 'N'}")
                if exists and chosen is None:
                    chosen = name
            if not chosen:
                jq_state['log'].append(f"[symbol_warn] no_candidate_exists core={core} first={candidates[0] if candidates else 'NONE'}")
                chosen = candidates[0]
            # strict_real_price 约束：若开启且 use_real_price=True，则必须选中 raw 后缀(_daily/_日)，否则抛错
            if strict_real_price_flag and use_real_price_flag:
                raw_ok = any(chosen.endswith(suf) for suf in _SUFFIX_RAW)
                if not raw_ok:
                    raise RuntimeError(
                        f"strict_real_price=True 但选中的数据文件 {chosen}.csv 不是 raw 类型(_daily/_日)。请添加原始未复权数据文件或关闭 strict_real_price。"
                    )
            jq_state['log'].append(
                f"[symbol_select] core={core} adjust={adjust} use_real_price={use_real_price_flag} strict={strict_real_price_flag} respect={respect} candidates={'|'.join(existence)} chosen={chosen}"
            )
            return chosen

        def _map_security_code(code: str) -> str:
            return _select_one(code)

        def _map_benchmark_code(code: str) -> str:
            # 基准简单：若 respect_symbol_suffix=True 同样生效；否则使用独立优先级：中文日 -> raw -> qfq
            core, provided_suffix = _normalize_code(code)
            respect = bool(_opt('respect_symbol_suffix', False))
            if provided_suffix and respect:
                direct = core + provided_suffix
                if os.path.exists(os.path.join(datadir, direct + '.csv')):
                    return direct
            bench_seq = ['_日','_daily','_daily_qfq','_daily_hfq']
            for suf in bench_seq:
                cand = core + suf
                if os.path.exists(os.path.join(datadir, cand + '.csv')):
                    jq_state['log'].append(f"[benchmark_select] core={core} candidates={bench_seq} chosen={cand}")
                    return cand
            jq_state['log'].append(f"[benchmark_select_warn] none_exists core={core} fallback={core+bench_seq[0]}")
            return core + bench_seq[0]

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
        # 记录原始输入
        try:
            jq_state['log'].append(f"[symbol_input] raw={symbols}")
        except Exception:
            pass
        # 若未设置 explicit 尊重后缀，则统一走映射流程（这样前端输入 '000514_daily_qfq' 仍可在 use_real_price/raw 场景下选到 _daily）。
        respect_suffix = False
        try:
            opt = jq_state.get('options', {}).get('respect_symbol_suffix')
            if isinstance(opt, bool):
                respect_suffix = opt
        except Exception:
            pass
        mapped_syms: List[str] = []
        for s in symbols:
            if respect_suffix:
                mapped_syms.append(s)
            else:
                # 剥后缀重选（允许用户直接传 qfq 仍由 adjust_type 决策）
                try:
                    mapped_syms.append(_map_security_code(s))
                except Exception:
                    mapped_syms.append(s)
        symbols = list(dict.fromkeys(mapped_syms))  # 去重保持顺序
        # use_real_price=True 时强制优先使用未复权(raw)数据文件（若存在），覆盖映射结果（除非用户显式 respect_suffix=True）
        try:
            if symbols and not respect_suffix and bool(jq_state.get('options', {}).get('use_real_price', False)):
                remapped = []
                detail_lines = []
                for sym in symbols:
                    original = sym
                    core = sym
                    stripped = False
                    for suf in ('_daily_qfq','_daily_hfq','_日_qfq','_日_hfq','_daily','_日'):
                        if core.endswith(suf):
                            core = core[:-len(suf)]
                            stripped = True
                            break
                    raw_candidates = [core + '_daily', core + '_日']
                    chosen_raw = None
                    for rc in raw_candidates:
                        exists = os.path.exists(os.path.join(datadir, rc + '.csv'))
                        detail_lines.append(f"candidate={rc} exists={exists}")
                        if exists and chosen_raw is None:
                            chosen_raw = rc
                    remapped.append(chosen_raw or sym)
                    jq_state['log'].append(
                        f"[use_real_price_scan] orig={original} stripped={stripped} core={core} raw_found={bool(chosen_raw)} chosen={chosen_raw or sym} details={'|'.join(detail_lines)}"
                    )
                symbols = list(dict.fromkeys(remapped))
                jq_state['log'].append(f"[use_real_price_remap] final_symbols={symbols}")
        except Exception as _e:
            try:
                jq_state['log'].append(f"[use_real_price_remap_error] {type(_e).__name__}:{_e}")
            except Exception:
                pass
        # 记录主 symbol 供后续 blocked 订单引用
        try:
            if symbols:
                jq_state['primary_symbol'] = symbols[0]
        except Exception:
            pass
        try:
            jq_state['log'].append(f"[symbol_mapped] final={symbols} respect_suffix={respect_suffix}")
        except Exception:
            pass
        if not symbols:
            raise ValueError('未指定任何标的: 请在策略 g.security 或 表单 symbol 提供至少一个')

        if benchmark_symbol:
            benchmark_symbol = _map_benchmark_code(benchmark_symbol)

        # 读取暖场天数（默认 250，可 set_option('history_lookback_days', N)）
        # 新增：jq_auto_history_preload=True (默认开启) && 用户未显式设置时，
        # 自动从策略源码中提取最大周期并放大，模拟聚宽“隐式加载历史”能力，使 start 当天即可产生信号。
        lookback_days = 250
        user_set_lb = False
        try:
            lb = jq_state.get('options', {}).get('history_lookback_days')
            if isinstance(lb, (int, float)) and lb >= 0:
                lookback_days = int(lb)
                user_set_lb = True
        except Exception:
            pass
        try:
            auto_flag = bool(jq_state.get('options', {}).get('jq_auto_history_preload', True))
        except Exception:
            auto_flag = True
        if (not user_set_lb) and auto_flag:
            try:
                code_txt = strategy_code
                periods: List[int] = []
                # pattern1: period=数字
                for m in re.finditer(r"period\s*=\s*(\d{1,4})", code_txt):
                    periods.append(int(m.group(1)))
                # pattern2: 常见指标函数第一个参数中出现数字（简化版）
                for m in re.finditer(r"\b(SMA|EMA|MA|ATR|RSI|WMA|TRIMA|KAMA|ADX|CCI)\s*\(\s*[^,\n]*?(\d{1,4})", code_txt, re.IGNORECASE):
                    try:
                        periods.append(int(m.group(2)))
                    except Exception:
                        pass
                periods = [p for p in periods if p >= 3]
                if periods:
                    max_p = max(periods)
                    auto_lb = min(max_p * 3, 600)  # 至少 3 倍，最大 600
                    if auto_lb > lookback_days:
                        lookback_days = auto_lb
                    jq_state['log'].append(f"[auto_history_preload] detected_periods={sorted(set(periods))} max={max_p} lookback_days={lookback_days}")
                else:
                    jq_state['log'].append(f"[auto_history_preload] none_detected use_default={lookback_days}")
            except Exception as _e:
                try:
                    jq_state['log'].append(f"[auto_history_preload_error] {type(_e).__name__}:{_e}")
                except Exception:
                    pass
        warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=lookback_days)).date().isoformat()

        # 建立原始输入代码与映射后文件名的对应关系，便于 attribute_history 精确匹配
        jq_state.setdefault('symbol_file_map', {})

        for i, sym in enumerate(symbols):
            # 不再无条件覆盖 qfq -> raw；完全由 adjust_type 和 force_data_variant 控制
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
                    jq_state['log'].append(f"[data_source] load={sym}.csv")
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
        # 统一手续费模型（默认即聚宽风格）：
        # 始终使用 rate=commission (默认0.0003)、单笔最小佣金5元、卖出印花税0.001；
        # 用户可通过 set_option 覆盖：commission / min_commission / stamp_duty。
        # 移除 fee_model 分支，保持逻辑简洁且确定性。
        try:
            jq_state['log'].append(f"[commission_setup] rate={commission} model=unified")
        except Exception:
            pass
        try:
            cerebro.broker.setcommission(commission=commission)
        except Exception:
            pass
        jq_state.setdefault('fee_config', {})
        cfg = jq_state['fee_config']
        cfg['rate'] = commission
        # 默认值
        min_comm = 5.0
        stamp_duty = 0.001
        # 用户覆盖
        try:
            user_min = jq_state.get('options', {}).get('min_commission')
            if isinstance(user_min, (int, float)) and user_min >= 0:
                min_comm = float(user_min)
        except Exception:
            pass
        try:
            user_sd = jq_state.get('options', {}).get('stamp_duty')
            if isinstance(user_sd, (int, float)) and user_sd >= 0:
                stamp_duty = float(user_sd)
        except Exception:
            pass
        cfg['min_commission'] = min_comm
        cfg['stamp_duty'] = stamp_duty
        try:
            jq_state['log'].append(f"[fee_config] min_commission={min_comm} stamp_duty={stamp_duty}")
        except Exception:
            pass
        slippage_perc = jq_state.get('options', {}).get('slippage_perc')
        # 方案2：内部半滑点（不再调用 broker.set_slippage_perc），默认仍用 0.00246 与聚宽对齐
        if slippage_perc is None and 'fixed_slippage' not in jq_state.get('options', {}):
            slippage_perc = 0.00246
            jq_state['options']['slippage_perc'] = slippage_perc
            try:
                jq_state['log'].append(f"[slippage_default] perc={slippage_perc}")
            except Exception:
                pass
        # 记录当前滑点模式（half_slip）便于审计
        try:
            jq_state['log'].append(f"[slippage_mode] scheme=half_slip half={(slippage_perc or 0)/2}")
        except Exception:
            pass
        # Corporate actions load (CSV optional) + manual merge (always if provided)
        try:
            jq_state['corporate_actions'] = []
            # Load from CSV only when simulate_corporate_actions True
            if bool(jq_state['options'].get('simulate_corporate_actions', False)):
                symbol = jq_state.get('symbol') or jq_state.get('security_code') or jq_state.get('raw_symbol')
                if symbol:
                    jq_state['corporate_actions'] = load_corporate_actions(symbol, jq_state.get('data_dir','data'), logger=jq_state.get('logger'))
            # Manual injection via set_option('manual_corporate_actions',[{...}])
            manual = jq_state['options'].get('manual_corporate_actions')
            if isinstance(manual, list) and manual:
                added = 0
                from .corporate_actions import CorporateActionEvent as _CAE
                for item in manual:
                    try:
                        if not isinstance(item, dict):
                            continue
                        date = str(item.get('date') or '').strip()
                        atype = str(item.get('type') or '').strip().upper()
                        if not date or not atype:
                            continue
                        ratio = item.get('ratio')
                        cash = item.get('cash')
                        shares = item.get('shares')
                        ev = _CAE(date=date, action_type=atype, ratio=(float(ratio) if ratio not in (None,'') else None), cash=(float(cash) if cash not in (None,'') else None), note='manual')
                        if shares not in (None,''):
                            try:
                                ev._manual_shares = int(shares)
                            except Exception:
                                pass
                        jq_state['corporate_actions'].append(ev)
                        try:
                            jq_state['log'].append(f"[ca_manual_merge] date={date} type={atype} shares={shares} ratio={ratio}")
                        except Exception:
                            pass
                        added += 1
                    except Exception:
                        continue
                if added:
                    try:
                        jq_state['log'].append(f"[ca_manual_load] count={added}")
                    except Exception:
                        pass
            # Sort after merge
            try:
                jq_state['corporate_actions'].sort(key=lambda e: e.date)
            except Exception:
                pass
        except Exception:
            jq_state['corporate_actions'] = []
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
                # Backtrader 的 order.executed.value 可能是根据价差/方向等内部逻辑计算的净值，
                # 为确保与聚宽风格一致，这里强制使用 size * price 作为成交金额（方向性金额）。
                orig_value = getattr(order.executed, 'value', 0.0)
                calc_value = size * price
                value = calc_value
                comm = getattr(order.executed, 'comm', 0.0)
                side = 'BUY' if size >= 0 else 'SELL'
                name = None
                try:
                    data = order.data
                    name = getattr(data, '_name', None)
                except Exception:
                    name = None
                # 二次费用调整（最小佣金 + 印花税）
                try:
                    fee_cfg = jq_state.get('fee_config', {})
                    rate = float(fee_cfg.get('rate', 0.0))
                    min_comm = float(fee_cfg.get('min_commission', 0.0))
                    stamp_duty = float(fee_cfg.get('stamp_duty', 0.0)) if side == 'SELL' else 0.0
                    # Margin 且零股数：不收取任何佣金，做一个明确日志
                    if order.status == order.Margin and abs(size) < 1e-9:
                        comm = 0.0
                        try:
                            jq_state['log'].append(f"[order_margin] status=Margin size=0 skip_fee orig_comm={getattr(order.executed,'comm',0.0):.4f}")
                        except Exception:
                            pass
                    elif value is not None:
                        gross = abs(value)
                        raw_comm = gross * rate
                        # Backtrader 已经按照 rate 计算过 comm（近似 raw_comm），我们做调整：
                        # 1) 重新计算标准佣金 raw_comm
                        # 2) 应用最小佣金
                        adj_comm = raw_comm
                        if min_comm > 0 and adj_comm < min_comm:
                            adj_comm = min_comm
                        total_fee = adj_comm + gross * stamp_duty
                        # 更新 order.executed 的 comm（注意可能无写权限，捕获异常）
                        delta = total_fee - comm
                        if abs(delta) > 1e-9:
                            try:
                                order.executed.comm = total_fee
                                comm = total_fee
                            except Exception:
                                comm = total_fee
                        # 费用日志
                        try:
                            # 与 [fill] 日志对齐：增加执行日期前缀，便于区分是否同日撮合
                            exec_date_prefix = ''
                            try:
                                exec_date_prefix = f"{dt} 09:30:00 - INFO - " if dt else ''
                            except Exception:
                                exec_date_prefix = ''
                            jq_state['log'].append(
                                f"{exec_date_prefix}[fee] {side} {name} value={value:.2f} price={price:.4f} base_comm={raw_comm:.4f} adj_comm={adj_comm:.4f} stamp_duty={(abs(value)*stamp_duty):.4f} final_comm={comm:.4f}"
                            )
                        except Exception:
                            pass
                        # 若原始 value 与重新计算差异较大，输出校正日志
                        try:
                            if orig_value is not None and abs(orig_value - value) > 1e-6:
                                jq_state['log'].append(
                                    f"[value_fix] {side} {name} orig_value={orig_value:.2f} recalculated={value:.2f} size={size} price={price:.4f}"
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
                # 将实际成交写入 JQ 日志，方便与期望价格核对
                try:
                    # 方案C: 始终使用 executed.dt (exec_date) 作为日志前缀日期，不依赖当前 bar
                    exec_date = dt  # already formatted date string
                    line = f"[fill] {side} {name} size={abs(size)} price={price} value={value} commission={comm}"
                    if exec_date:
                        jq_state['log'].append(f"{exec_date} 09:30:00 - INFO - {line}")
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

        # 指标汇总（先取 BT 自带，稍后会计算并覆盖 sharpe）
        sharpe_bt = strat.analyzers.sharpe.get_analysis().get('sharperatio') if hasattr(strat.analyzers, 'sharpe') else None
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

        # 依据日收益计算每日盈亏（金额），从初始资金开始滚动
        daily_pnl: List[Dict[str, Any]] = []
        try:
            prev_equity_val = float(cash)
            # 使用与 daily_returns 相同的日期顺序
            for dr in daily_returns:
                r = float(dr['ret'])
                pnl_amt = prev_equity_val * r
                eq_after = prev_equity_val + pnl_amt
                daily_pnl.append({'date': dr['date'], 'pnl': pnl_amt, 'equity': eq_after})
                prev_equity_val = eq_after
        except Exception:
            daily_pnl = []

        # 交易聚合数据
        total_trades = trade_ana.get('total', {}).get('total', 0)
        won = trade_ana.get('won', {}).get('total', 0)
        lost = trade_ana.get('lost', {}).get('total', 0)
        win_rate = (won / total_trades) if total_trades else 0

        # 推断数据变体（若仅单标的，取第一；多标的则列表）
        def _infer_variant(name: str) -> str:
            if any(name.endswith(s) for s in _SUFFIX_QFQ):
                return 'qfq'
            if any(name.endswith(s) for s in _SUFFIX_HFQ):
                return 'hfq'
            if any(name.endswith(s) for s in _SUFFIX_RAW):
                return 'raw'
            return 'unknown'
        data_variant = None
        if symbols:
            if len(symbols) == 1:
                data_variant = _infer_variant(symbols[0])
            else:
                data_variant = {s: _infer_variant(s) for s in symbols}

        metrics = {
            'final_value': cerebro.broker.getvalue(),
            'pnl_pct': (cerebro.broker.getvalue() / cash - 1),
            'sharpe': None,  # 将在下方以日收益计算年化夏普
            'sharpe_bt': sharpe_bt,
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
            # 追踪数据源：展示偏好与实际加载的文件映射
            'data_source_preference': None,
            'data_sources_used': None,
            'adjust_type': None,
            'data_variant': data_variant,
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

        # 计算基准与超额，并准备 alpha/beta 需要的对齐日收益
        # 将策略日收益映射为 {date_str: r}
        strat_ret_map = {dt.strftime('%Y-%m-%d'): r for dt, r in timereturn.items()}

        aligned_sr: List[float] = []
        aligned_br: List[float] = []
        if benchmark_symbol:
            # 为了与聚宽一致，基准收益以“回测开始日前最后一个可交易日”的收盘为基准
            # 因此这里向前多取几天，之后只在 >= start 的日期上展示，但 equity 的基准点可能在 start 之前
            bench_warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=10)).date().isoformat()
            bench_df = load_csv_dataframe(benchmark_symbol, bench_warmup_start, end, datadir)
            if len(bench_df) == 0:
                metrics['benchmark_final'] = None
                metrics['excess_return'] = None
                metrics['benchmark_missing_reason'] = 'empty_file_or_no_rows_in_range'
            else:
                # 需要 close 列
                bench_df = bench_df[['datetime', 'close']].copy()
                bench_df['ret'] = bench_df['close'].pct_change().fillna(0.0)
                bench_df['equity'] = (1 + bench_df['ret']).cumprod()
                # 找到回测开始日前（<= start）的最后一个基准日，记录基准点信息
                base_idx = bench_df[bench_df['datetime'] <= pd.to_datetime(start)].index
                base_row = None
                if len(base_idx) > 0:
                    base_row = bench_df.loc[base_idx.max()]
                # 按基准点将 equity 归一到 1
                try:
                    if base_row is not None:
                        base_equity = float(base_row['equity']) if 'equity' in base_row else None
                    else:
                        base_equity = float(bench_df['equity'].iloc[0]) if len(bench_df) else None
                except Exception:
                    base_equity = None
                if base_equity and base_equity != 0:
                    bench_df['equity_rebased'] = bench_df['equity'] / base_equity
                else:
                    bench_df['equity_rebased'] = bench_df['equity']
                # 记录日志，便于对齐核对
                try:
                    if base_row is not None:
                        jq_state['log'].append(
                            f"[benchmark_base] base_date={base_row['datetime'].date()} base_close={float(base_row['close']):.4f} start={start}"
                        )
                    else:
                        jq_state['log'].append(f"[benchmark_base] no_base_before_start start={start}")
                except Exception:
                    pass

                # 仅在 >= start 的日期用于展示/对齐，但 equity 已含有此前基准归一
                bm_map = {d.strftime('%Y-%m-%d'): (r, eq) for d, r, eq in zip(bench_df['datetime'], bench_df['ret'], bench_df['equity_rebased']) if d.strftime('%Y-%m-%d') >= start}
                overlap_cnt = 0
                for ec in equity_curve:
                    d = ec['date']
                    if d in bm_map:
                        br, beq = bm_map[d]
                        benchmark_curve.append({'date': d, 'equity': beq})
                        excess_curve.append({'date': d, 'excess': ec['equity']/beq - 1 if beq != 0 else 0})
                        overlap_cnt += 1
                        # 收集与基准对齐的策略/基准日收益
                        sr = strat_ret_map.get(d)
                        if sr is not None:
                            aligned_sr.append(float(sr))
                            aligned_br.append(float(br))
                if benchmark_curve:
                    metrics['benchmark_final'] = benchmark_curve[-1]['equity']
                    metrics['excess_return'] = equity_curve[-1]['equity']/benchmark_curve[-1]['equity'] - 1
                    # 新增：基准收益（终值-1）
                    metrics['benchmark_return'] = (benchmark_curve[-1]['equity'] - 1)
                else:
                    metrics['benchmark_final'] = None
                    metrics['excess_return'] = None
                    # 诊断：无日期重叠
                    metrics['benchmark_missing_reason'] = 'no_overlap_with_strategy_dates'
        else:
            metrics['benchmark_final'] = None
            metrics['excess_return'] = None
            metrics['benchmark_return'] = None

        # 记录使用的基准信息
        metrics['benchmark_symbol_used'] = benchmark_symbol
        metrics['benchmark_code'] = jq_state.get('benchmark') if 'jq_state' in locals() else None
        if 'benchmark_detect_phase' not in metrics:
            metrics['benchmark_detect_phase'] = 'none'
        # 回填数据源偏好与实际文件
        try:
            metrics['data_source_preference'] = jq_state.get('options', {}).get('data_source_preference') or ['_daily', '_daily_qfq', '_日']
            metrics['data_sources_used'] = jq_state.get('symbol_file_map')
            # 新结构：由 _decide_adjust() 决定
            try:
                metrics['adjust_type'] = _decide_adjust()
            except Exception:
                metrics['adjust_type'] = None
        except Exception:
            pass

        # 重新计算并覆盖夏普；计算 α/β（CAPM，默认 Rf=3% 年化，可通过 set_option('risk_free_rate') 年化设定）
        try:
            # use module-level _math
            # 交易日年化因子（聚宽口径常用 250，可通过 set_option('annualization_factor' 或 'trading_days') 配置）
            trading_days = 250.0
            try:
                td_opt = jq_state.get('options', {}).get('annualization_factor') or jq_state.get('options', {}).get('trading_days')
                if isinstance(td_opt, (int, float)) and td_opt > 0:
                    trading_days = float(td_opt)
            except Exception:
                pass
            # 年化无风险利率（例如 0.04 表示 4% 年化，若未配置默认采用 4% 贴近聚宽习惯）
            rf_annual = 0.04
            try:
                _opts = jq_state.get('options', {})
                if isinstance(_opts, dict) and ('risk_free_rate' in _opts):
                    rf_opt = _opts.get('risk_free_rate')
                    if isinstance(rf_opt, (int, float)):
                        rf_annual = float(rf_opt)
            except Exception:
                pass
            rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0

            # 策略日收益序列（≥start）。Sharpe 默认包含首个样本；可通过 set_option('sharpe_exclude_first', True) 排除。
            sr_all = [float(r) for dt, r in timereturn.items() if dt.strftime('%Y-%m-%d') >= start]
            sharpe_exclude_first = False
            try:
                opt_ex_first = jq_state.get('options', {}).get('sharpe_exclude_first')
                if isinstance(opt_ex_first, bool):
                    sharpe_exclude_first = opt_ex_first
            except Exception:
                pass
            sr_seq = (sr_all[1:] if len(sr_all) > 1 else list(sr_all)) if sharpe_exclude_first else list(sr_all)
            ex_sr_all = [r - rf_daily for r in sr_seq]
            # 夏普：sqrt(252) * mean(excess) / std(excess)
            def _mean(vals: List[float]) -> float:
                return sum(vals) / len(vals) if vals else 0.0
            def _std(vals: List[float]) -> float:
                n = len(vals)
                if n <= 1:
                    return 0.0
                m = _mean(vals)
                var = sum((x - m) ** 2 for x in vals) / (n - 1)
                return var ** 0.5
            # 使用样本标准差(ddof=1)作为波动，更贴近聚宽
            n_ex = len(ex_sr_all)
            sharpe_mean_excess = None
            if n_ex > 1:
                m_ex = _mean(ex_sr_all)
                var_samp_ex = sum((x - m_ex) ** 2 for x in ex_sr_all) / (n_ex - 1)
                std_ex_samp = (var_samp_ex ** 0.5)
                if std_ex_samp > 0:
                    sharpe_mean_excess = (_math.sqrt(trading_days) * m_ex / std_ex_samp)

            # 可选：CAGR 口径
            # 计算 CAGR
            # CAGR 法基于同样的 sr_seq（已剔除首样本）
            sharpe_jq = None
            if sr_seq:
                cum = 1.0
                for r in sr_seq:
                    cum *= (1.0 + r)
                n = float(len(sr_seq))
                r_annual_cagr = (cum ** (trading_days / n) - 1.0)
                # 年化波动（样本标准差，ddof=1）
                m = sum(sr_seq) / n
                if len(sr_seq) > 1:
                    var_samp = sum((r - m) ** 2 for r in sr_seq) / (len(sr_seq) - 1)
                else:
                    var_samp = 0.0
                vol_annual = (_math.sqrt(var_samp) * _math.sqrt(trading_days)) if var_samp > 0 else 0.0
                if vol_annual > 0:
                    sharpe_jq = (r_annual_cagr - rf_annual) / vol_annual

            # 选择输出口径：默认使用 CAGR 年化口径（与聚宽文档展示式一致）；可通过 set_option('sharpe_method','mean'|'cagr') 切换
            sharpe_method = None
            try:
                sm = jq_state.get('options', {}).get('sharpe_method')
                if isinstance(sm, str):
                    sharpe_method = sm.lower().strip()
            except Exception:
                sharpe_method = None
            if not sharpe_method:
                sharpe_method = 'cagr'
            if sharpe_method == 'cagr':
                metrics['sharpe'] = sharpe_jq if (sharpe_jq is not None) else sharpe_mean_excess
            else:
                metrics['sharpe'] = sharpe_mean_excess if (sharpe_mean_excess is not None) else sharpe_jq
            metrics['sharpe_method'] = sharpe_method
            try:
                jq_state['log'].append(
                    f"[sharpe] method={sharpe_method or 'mean'} jq={sharpe_jq} mean_excess={sharpe_mean_excess} n={len(sr_seq)} ddof=1 exclude_first={sharpe_exclude_first}"
                )
            except Exception:
                pass

            # 使用 250 天年化计算策略年化收益（覆盖 BT Returns 的 rt_annual）
            if sr_all:
                total_cum = 1.0
                for r in sr_all:
                    total_cum *= (1.0 + r)
                n = float(len(sr_all))
                metrics['rt_annual'] = (total_cum ** (trading_days / n) - 1.0)
            # 记录年化设置，便于审计
            try:
                jq_state['log'].append(f"[annualization] trading_days={int(trading_days)} rf_annual={rf_annual}")
            except Exception:
                pass

            # 计算 α / β
            # β：仍采用“日度收益回归”的样本协方差法
            # α：按聚宽展示公式（使用年化收益） Alpha = Rp - [ Rf + β (Rm - Rf) ]
            if aligned_sr and aligned_br and len(aligned_sr) == len(aligned_br):
                # α/β 使用“全部对齐样本”；Sharpe 根据上面选项决定是否排除首样本
                Rs = list(map(float, aligned_sr))
                Rb = list(map(float, aligned_br))
                mb = _mean(Rb)
                ms = _mean(Rs)
                # 方差与协方差（样本）
                def _cov(x: List[float], y: List[float]) -> float:
                    n = min(len(x), len(y))
                    if n <= 1:
                        return 0.0
                    mx = _mean(x)
                    my = _mean(y)
                    return sum((x[i]-mx)*(y[i]-my) for i in range(n)) / (n - 1)
                var_b = _cov(Rb, Rb)
                beta = (_cov(Rb, Rs) / var_b) if var_b > 0 else None
                # 年化收益（使用对齐样本）
                n_align = float(len(Rs))
                # 策略年化收益（CAGR）
                cum_s = 1.0
                for r in Rs:
                    cum_s *= (1.0 + r)
                rp_annual = (cum_s ** (trading_days / n_align) - 1.0) if n_align > 0 else None
                # 基准年化收益（CAGR）
                cum_b = 1.0
                for r in Rb:
                    cum_b *= (1.0 + r)
                rm_annual = (cum_b ** (trading_days / n_align) - 1.0) if n_align > 0 else None
                # Alpha（年化）: Rp - [ Rf + β (Rm - Rf) ]
                alpha_annual = None
                if (beta is not None) and (rp_annual is not None) and (rm_annual is not None):
                    alpha_annual = rp_annual - (rf_annual + beta * (rm_annual - rf_annual))
                # 额外提供日度口径（简单线性换算）
                alpha_daily = (alpha_annual / trading_days) if (alpha_annual is not None and trading_days > 0) else None
                metrics['beta'] = beta
                metrics['alpha_daily'] = alpha_daily
                metrics['alpha_annual'] = alpha_annual
                # 输出口径：默认按年化显示，更贴近聚宽页面 α；可通过 set_option('alpha_unit','daily'|'annual') 切换
                alpha_unit = 'annual'
                try:
                    au = jq_state.get('options', {}).get('alpha_unit')
                    if isinstance(au, str) and au.lower() in ('daily', 'annual'):
                        alpha_unit = au.lower()
                except Exception:
                    pass
                metrics['alpha'] = alpha_annual if alpha_unit == 'annual' else alpha_daily
                try:
                    jq_state['log'].append(
                        f"[alpha] model=annual_formula unit={alpha_unit} rp_annual={rp_annual} rm_annual={rm_annual} daily={alpha_daily} annual={alpha_annual} beta={beta} rf_annual={rf_annual} n_align={len(aligned_sr)}"
                    )
                except Exception:
                    pass
                metrics['alpha_unit'] = alpha_unit
            else:
                metrics['beta'] = None
                metrics['alpha'] = None
                metrics['alpha_daily'] = None
                metrics['alpha_annual'] = None
            # 补充导出审计字段，便于与聚宽核对
            metrics['annualization_factor'] = int(trading_days)
            metrics['sharpe_rf_annual'] = rf_annual
        except Exception:
            # 若计算失败，不影响回测其他结果
            if 'beta' not in metrics:
                metrics['beta'] = None
            if 'alpha' not in metrics:
                metrics['alpha'] = None

        # 计算最大回撤区间（后端权威输出，避免不同前端实现差异）
        try:
            # 使用已归一的策略累计净值 equity_curve（>= start）
            peak = -float('inf')
            peak_date = None
            min_dd = 0.0
            trough_date = None
            best_peak_date = None
            eps = 1e-12  # 浮点容差
            for p in (equity_curve or []):
                val = float(p.get('equity', 0.0) or 0.0)
                dt = p.get('date')
                if val > peak:
                    peak = val
                    peak_date = dt
                if peak > 0 and dt is not None:
                    dd = val / peak - 1.0
                    # 若出现相同的最小回撤（在容差内），选择更晚的谷底日期，贴近聚宽口径
                    if (dd < min_dd - eps) or (abs(dd - min_dd) <= eps and (trough_date is None or (dt and dt > trough_date))):
                        min_dd = dd
                        trough_date = dt
                        best_peak_date = peak_date
            if trough_date and best_peak_date:
                metrics['drawdown_interval'] = f"{best_peak_date} ~ {trough_date}"
            else:
                metrics['drawdown_interval'] = None
        except Exception:
            metrics['drawdown_interval'] = None

        # 回测结束，从 analyzer 里取交易记录 (包含已平仓 + 未平仓) 及订单级别明细
        trades: List[TradeRecord] = strat.analyzers.trade_capture.get_analysis() if hasattr(strat.analyzers, 'trade_capture') else []
        orders: List[OrderRecord] = strat.analyzers.order_capture.get_analysis() if hasattr(strat.analyzers, 'order_capture') else []
        # 合并被阻断订单
        try:
            blocked = jq_state.get('blocked_orders', [])
            if blocked:
                orders = blocked + orders
        except Exception:
            pass
        # 只保留 >= start 的订单 (Blocked 订单如果其日期 < start 也丢弃)
        try:
            orders = [o for o in orders if (o.datetime or '') >= start]
        except Exception:
            pass
        # 排序订单：按日期 + 原始插入顺序；若同日，保持 BlockedLimitUp 在前（便于展示拦截→成交）。
        try:
            def _ord_key(o: OrderRecord):
                d = o.datetime or '9999-99-99'
                # 加一个次序：Blocked 排前面，其余按 1
                pri = 0 if (o.status or '').startswith('Blocked') else 1
                return (d, pri)
            orders.sort(key=_ord_key)
        except Exception:
            pass
        # 只保留 >= start 的交易；并过滤 size=0 且无价格的伪记录
        try:
            trades = [t for t in trades if (t.datetime or '') >= start and (t.size or 0) != 0]
        except Exception:
            pass
        # 同样对交易按开/平仓日期排序（主要用 close_datetime, fallback datetime）
        try:
            def _trade_key(t: TradeRecord):
                d = t.close_datetime or t.datetime or '9999-99-99'
                return d
            trades.sort(key=_trade_key)
        except Exception:
            pass
        # 回退占位：当前版本尚未重新实现 daily_turnover / jq_records / jq_logs 采集逻辑，先以空值返回避免编译错误
        daily_turnover: List[Dict[str, Any]] = []
        jq_records = None
        try:
            jq_logs = list(jq_state.get('log', []))
        except Exception:
            jq_logs = []
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            daily_pnl=daily_pnl,
            daily_turnover=daily_turnover,
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
            daily_pnl=[],
            daily_turnover=[],
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
