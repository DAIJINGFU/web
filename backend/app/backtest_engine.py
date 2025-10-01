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
# 新增统一数据加载接口
from . import data_loader as _dl


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

def load_csv_dataframe(symbol: str, start: str, end: str, datadir: str = 'data') -> pd.DataFrame:  # backward compat
    """兼容旧接口：从 data/ 或 stockdata/ 中加载日线数据 (优先新 stockdata)."""
    prefer_stockdata = bool(int(os.environ.get('PREFER_STOCKDATA', '1')))
    # adjust 决策：若环境指定 ADJUST_TYPE 使用之，否则 auto
    adjust = os.environ.get('ADJUST_TYPE', 'auto').lower()
    try:
        return _dl.load_price_dataframe(symbol, start, end, frequency='daily', adjust=adjust, prefer_stockdata=prefer_stockdata,
                                        data_root=datadir, stockdata_root=None)
    except Exception as e:
        raise e

def load_csv_data(symbol: str, start: str, end: str, datadir: str = 'data') -> bt.feeds.PandasData:  # backward compat
    prefer_stockdata = bool(int(os.environ.get('PREFER_STOCKDATA', '1')))
    adjust = os.environ.get('ADJUST_TYPE', 'auto').lower()
    return _dl.load_bt_feed(symbol, start, end, frequency='daily', adjust=adjust, prefer_stockdata=prefer_stockdata,
                             data_root=datadir, stockdata_root=None)

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
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        'bool': bool,
        'isinstance': isinstance,
        'print': print,
        '__build_class__': __build_class__,  # 允许用户定义类（策略类等）
    '__name__': '__main__',
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
            'limit_pct': 0.10,
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
        'trading_calendar': None,
    }

    # 交易日历加载 (可选)
    try:
        import os as _os, pandas as _pd
        cal_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'data', 'trading_calendar.csv')
        cal_path = _os.path.abspath(cal_path)
        if _os.path.exists(cal_path):
            _cal_df = _pd.read_csv(cal_path)
            if 'date' in _cal_df.columns:
                col = 'date'
            else:
                col = _cal_df.columns[0]
            dates = set(_pd.to_datetime(_cal_df[col]).dt.date.astype(str))
            if dates:
                jq_state['trading_calendar'] = dates
                jq_state['log'].append(f"[trading_calendar_loaded] size={len(dates)}")
    except Exception:
        pass

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
                    def get_price(security,
                                  start_date=None,
                                  end_date=None,
                                  frequency: str = 'daily',
                                  fields=None,
                                  count=None,
                                  skip_paused: bool = False,
                                  fq: str = 'pre',
                                  panel: bool = False,
                                  fill_paused: bool = True):
                        """JoinQuant 风格 get_price（本地近似实现）
                        受限说明：
                        - 仅支持 frequency='daily'（分钟级暂未实现会抛异常）
                        - 不访问真实停牌日历；skip_paused/fill_paused 仅基于 volume==0 做近似
                        - fq: 'pre'|'post'|'none' 占位，当前数据假定已是所需复权口径（由外部文件选择决定），因此不做额外价格复权变换
                        - panel=True 时返回一个 PanelEmu 对象： panel['open'] -> DataFrame(index=日期, columns=证券)
                        参数组合规则（仿聚宽）：
                        1) (count, end_date) 或 (start_date, end_date) 二选一模式；不能同时指定 count 与 start_date
                        2) 若均未给 end_date，则默认使用当前回测日的前一交易日作为 end_date
                        3) count 模式：返回 end_date 之前（含 end_date）回溯 count 条
                        4) start/end 模式：返回区间 [start_date, end_date]（含端点）
                        返回：
                          - 单一证券: DataFrame(index=date, columns=fields)；若 fields 为单字段字符串则返回 Series
                          - 多证券 & panel=False: dict{sec -> DataFrame}
                          - 多证券 & panel=True: PanelEmu (panel['open'] -> DataFrame)
                        """
                        import datetime as _dt
                        import pandas as _pd
                        # 参数预处理 -------------------------------------------------
                        if frequency.lower() not in ('daily', 'd'):
                            raise ValueError('当前本地实现仅支持 frequency="daily"')
                        # 统一证券列表
                        if isinstance(security, (list, tuple, set)):
                            secs = list(security)
                        else:
                            secs = [security]
                        # 默认字段
                        all_possible_fields = ['open','close','high','low','volume','money','avg','pre_close','factor','paused']
                        if fields is None:
                            use_fields = ['open','close','high','low','volume']
                        else:
                            if isinstance(fields, str):
                                use_fields = [fields]
                            else:
                                use_fields = list(fields)
                        # 添加派生字段所需的基础字段
                        derived_need = []
                        if 'money' in use_fields or 'avg' in use_fields:
                            if 'close' not in use_fields: derived_need.append('close')
                            if 'volume' not in use_fields: derived_need.append('volume')
                        if 'pre_close' in use_fields:
                            if 'close' not in use_fields: derived_need.append('close')
                        base_field_set = set(use_fields + derived_need)
                        # 过滤非法字段
                        for f in list(base_field_set):
                            if f not in all_possible_fields:
                                base_field_set.remove(f)
                        # 解析日期 ---------------------------------------------------
                        def _to_date(x):
                            if x is None:
                                return None
                            if isinstance(x, _dt.datetime):
                                return x.date()
                            if isinstance(x, _dt.date):
                                return x
                            s = str(x)
                            try:
                                return _dt.datetime.fromisoformat(s.replace('/', '-')).date()
                            except Exception:
                                return None
                        cur_bt_date = None
                        try:
                            cur_bt_date = bt.num2date(self.data.datetime[0]).date()
                        except Exception:
                            pass
                        end_d = _to_date(end_date)
                        if end_d is None:
                            # end_date 默认 = 当前回测日的前一日
                            if cur_bt_date:
                                end_d = cur_bt_date - _dt.timedelta(days=1)
                        start_d = _to_date(start_date)
                        if count is not None and start_d is not None:
                            raise ValueError('count 与 start_date 不能同时指定')
                        if count is None and start_d is None:
                            # 默认取最近 1 条
                            count = 1
                        # 结果容器 ---------------------------------------------------
                        per_sec_frames = {}
                        hist_map = jq_state.get('history_df_map') or {}
                        global_df = jq_state.get('history_df')
                        for sec in secs:
                            base = str(sec).split('.')[0]
                            df_full = None
                            # 优先精确 key 命中
                            if base in hist_map:
                                df_full = hist_map[base]
                            else:
                                # 退化匹配
                                for k, v in hist_map.items():
                                    if str(k).startswith(base):
                                        df_full = v
                                        break
                            if df_full is None:
                                df_full = global_df if isinstance(global_df, _pd.DataFrame) else None
                            if df_full is None or 'datetime' not in df_full.columns:
                                per_sec_frames[sec] = _pd.DataFrame(columns=use_fields)
                                continue
                            df = df_full[['datetime'] + [c for c in ['open','close','high','low','volume'] if c in df_full.columns]].copy()
                            df['date'] = df['datetime'].dt.date
                            # 过滤未来：只取 < 当前回测日 的历史
                            if cur_bt_date is not None:
                                df = df[df['date'] < cur_bt_date]
                            if df.empty:
                                per_sec_frames[sec] = _pd.DataFrame(columns=use_fields)
                                continue
                            if count is not None:
                                # 回溯 count 条（含 end_d）
                                if end_d is not None:
                                    df = df[df['date'] <= end_d]
                                df = df.tail(int(count))
                            else:
                                # 区间模式 start_d & end_d
                                if start_d is not None:
                                    df = df[df['date'] >= start_d]
                                if end_d is not None:
                                    df = df[df['date'] <= end_d]
                            if df.empty:
                                per_sec_frames[sec] = _pd.DataFrame(columns=use_fields)
                                continue
                            df = df.sort_values('date')
                            # 去除自动补齐缺失交易日逻辑：只保留原始数据行
                            synthetic_col = '_synthetic'
                            if synthetic_col not in df.columns:
                                df[synthetic_col] = 0  # 占位，后续 paused 逻辑兼容
                            # 生成派生字段 -------------------------------------
                            if 'money' in base_field_set and 'money' not in df.columns:
                                try:
                                    df['money'] = df['close'] * df['volume']
                                except Exception:
                                    df['money'] = _pd.NA
                            if 'avg' in base_field_set and 'avg' not in df.columns:
                                try:
                                    df['avg'] = (df['close'] * df['volume']) / df['volume'].replace(0, _pd.NA)
                                except Exception:
                                    df['avg'] = _pd.NA
                            if 'pre_close' in base_field_set:
                                df['pre_close'] = df['close'].shift(1)
                            if 'factor' in base_field_set and 'factor' not in df.columns:
                                df['factor'] = 1.0  # 占位：真实复权因子需另行加载
                            if 'paused' in base_field_set:
                                # 仅对原始数据 volume==0 计为停牌
                                df['paused'] = (df['volume'] == 0).astype(int)
                            # 派生涨跌停 (若请求)
                            if {'high_limit','low_limit'} & base_field_set:
                                try:
                                    limit_pct = float(jq_state['options'].get('limit_pct', 0.1) or 0.1)
                                except Exception:
                                    limit_pct = 0.1
                                # 用 pre_close (若无则 close.shift(1))
                                if 'pre_close' in df.columns:
                                    ref_pc = df['pre_close']
                                else:
                                    ref_pc = df['close'].shift(1)
                                tick = float(jq_state['options'].get('price_tick', 0.01) or 0.01)
                                def _round_tick(v):
                                    return _math.floor(v / tick + 1e-9) * tick
                                hl = ref_pc * (1 + limit_pct)
                                ll = ref_pc * (1 - limit_pct)
                                df['high_limit'] = hl.map(lambda x: _round_tick(x) if _pd.notnull(x) else x)
                                df['low_limit'] = ll.map(lambda x: _round_tick(x) if _pd.notnull(x) else x)
                            # 应用停牌过滤/填充逻辑（近似实现）
                            if 'paused' in df.columns:
                                if skip_paused:
                                    df = df[df['paused'] == 0]
                                elif not fill_paused:
                                    # 不填充：将停牌行除 paused 字段外设为 NA
                                    paused_mask = df['paused'] == 1
                                    if paused_mask.any():
                                        for col in df.columns:
                                            if col not in ('datetime','date','paused'):
                                                df.loc[paused_mask, col] = _pd.NA
                            # 选择字段
                            out_cols = [f for f in use_fields if f in df.columns]
                            # 如果请求了 paused 但未请求 synthetic，可在调试需求下选择不暴露 synthetic
                            work = df[['date', *out_cols]].copy()
                            work.set_index('date', inplace=True)
                            per_sec_frames[sec] = work
                        # 单 / 多证券返回格式 -------------------------------------------
                        if len(secs) == 1:
                            single_df = per_sec_frames[secs[0]]
                            # Series 简化
                            if isinstance(fields, str):
                                series = single_df[fields] if fields in single_df.columns else _pd.Series([], dtype=float)
                                series.index.name = None
                                return series
                            single_df.index.name = None
                            return single_df
                        # 多证券
                        if panel:
                            # 构造一个简单 Panel 模拟器
                            class PanelEmu:
                                def __init__(self, data_map):  # data_map: sec -> DataFrame
                                    # 统一日期集合
                                    all_dates = sorted({d for df in data_map.values() for d in df.index})
                                    self._fields = set()
                                    self._dates = all_dates
                                    self._secs = list(data_map.keys())
                                    self._cube = {}
                                    for f in use_fields:
                                        # 构建 DataFrame 行=日期 列=证券
                                        mat = _pd.DataFrame(index=all_dates, columns=self._secs, dtype=float)
                                        for sec, df in data_map.items():
                                            if f in df.columns:
                                                mat.loc[df.index, sec] = df[f]
                                        self._cube[f] = mat
                                        self._fields.add(f)
                                def __getitem__(self, item):
                                    return self._cube.get(item, _pd.DataFrame())
                                @property
                                def fields(self):
                                    return list(self._fields)
                                @property
                                def symbols(self):
                                    return list(self._secs)
                            return PanelEmu(per_sec_frames)
                        # 默认: 返回 dict{sec: DataFrame}
                        return per_sec_frames
                    exec_env['get_price'] = get_price
                # --- history (JoinQuant style, daily only) ---
                if isinstance(exec_env, dict) and 'history' not in exec_env:
                    def history(count: int,
                                unit: str = '1d',
                                field: str = 'avg',
                                security_list=None,
                                df: bool = True,
                                skip_paused: bool = False,
                                fq: str = 'pre'):
                        """近似聚宽 history：仅支持日级。
                        参数:
                          count: 回溯条数（不含当前正在运行的当日）
                          unit: 仅支持 '1d'
                          field: 单字段，可选 open/close/high/low/volume/money/avg/pre_close/paused
                          security_list: 单个或列表；None 时尝试 jq_state['universe'] 或 g.security
                          df: True 返回 DataFrame(index=日期, columns=证券代码)；False 返回 dict{sec: np.ndarray}
                          skip_paused: True 删掉停牌( volume==0 ) 行（各列都删除）
                          fq: 复权标识，占位不做转换
                        行为：
                          - 不包含当前 bar 日期
                          - 若个别证券数据不足 count，会返回尽可能多的历史；不强制齐全
                          - 日期索引升序
                        差异/限制：
                          - 不支持分钟级
                          - 未实现 high_limit / low_limit / factor 真值；如需可扩展
                        """
                        import pandas as _pd
                        import numpy as _np
                        import datetime as _dt
                        if unit.lower() not in ('1d', 'd', 'day'):
                            raise ValueError('history 目前仅支持日级 unit=1d')
                        if count <= 0:
                            return _pd.DataFrame() if df else {}
                        # 解析证券列表
                        if security_list is None:
                            jqst = exec_env.get('jq_state', {})
                            universe = jqst.get('universe') or []
                            if universe:
                                secs = list(universe)
                            else:
                                # fallback: 尝试 g.security
                                gobj = exec_env.get('g')
                                primary = getattr(gobj, 'security', None) if gobj else None
                                secs = [primary] if primary else []
                        else:
                            if isinstance(security_list, (list, tuple, set)):
                                secs = list(security_list)
                            else:
                                secs = [security_list]
                        secs = [s for s in secs if s]
                        if not secs:
                            return _pd.DataFrame() if df else {}
                        # 当前回测日
                        try:
                            cur_date = bt.num2date(self.data.datetime[0]).date()
                        except Exception:
                            cur_date = None
                        # helper 取单证券 DataFrame
                        get_price_fn = exec_env.get('get_price')
                        results = {}
                        needed_field = field
                        # 若 field 为 avg/price 且数据里没有，将用 close 近似
                        for sec in secs:
                            try:
                                base_fields = None
                                if field in ('avg', 'price'):
                                    base_fields = ['close', 'volume']
                                elif field == 'money':
                                    base_fields = ['close', 'volume']
                                elif field == 'pre_close':
                                    base_fields = ['close']
                                elif field == 'paused':
                                    base_fields = ['paused']
                                else:
                                    base_fields = [field]
                                df_single = get_price_fn(sec, count=count, fields=base_fields, skip_paused=skip_paused, fq=fq, fill_paused=True)
                                if isinstance(df_single, _pd.Series):
                                    df_single = df_single.to_frame(name=base_fields[0])
                                # 计算派生字段
                                if field == 'avg' or field == 'price':
                                    if 'close' in df_single.columns:
                                        series_field = df_single['close']
                                    else:
                                        series_field = _pd.Series([], dtype=float)
                                elif field == 'money':
                                    if {'close','volume'} <= set(df_single.columns):
                                        series_field = df_single['close'] * df_single['volume']
                                    else:
                                        series_field = _pd.Series([], dtype=float)
                                elif field == 'pre_close':
                                    if 'close' in df_single.columns:
                                        series_field = df_single['close'].shift(1)
                                    else:
                                        series_field = _pd.Series([], dtype=float)
                                elif field == 'paused':
                                    if 'paused' in df_single.columns:
                                        series_field = df_single['paused']
                                    else:
                                        tmp_df = get_price_fn(sec, count=count, fields=['paused'], skip_paused=False, fq=fq, fill_paused=True)
                                        if isinstance(tmp_df, _pd.Series):
                                            series_field = tmp_df
                                        else:
                                            series_field = tmp_df.get('paused', _pd.Series([], dtype=int))
                                else:
                                    # 直接取字段
                                    if field in df_single.columns:
                                        series_field = df_single[field]
                                    else:
                                        # 尝试重新获取包含该字段
                                        tmp_df = get_price_fn(sec, count=count, fields=[field], skip_paused=skip_paused, fq=fq, fill_paused=True)
                                        if isinstance(tmp_df, _pd.Series):
                                            series_field = tmp_df
                                        else:
                                            series_field = tmp_df[field] if field in tmp_df.columns else _pd.Series([], dtype=float)
                                # 限制为 count 条（防止前值填充扩展超出）
                                series_field = series_field.tail(count)
                                series_field.name = sec
                                results[sec] = series_field
                            except Exception:
                                results[sec] = _pd.Series([], name=sec, dtype=float)
                        # 对齐索引（并集）
                        all_index = sorted({idx for s in results.values() for idx in s.index})
                        def _reindex(s):
                            return s.reindex(all_index)
                        if df:
                            data_map = {sec: _reindex(s) for sec, s in results.items()}
                            out_df = _pd.DataFrame(data_map, index=all_index)
                            # 自定义 Series 以支持负整数位置索引
                            class _HSeries(_pd.Series):
                                @property
                                def _constructor(self):
                                    return _HSeries
                                def __getitem__(self, key):
                                    if isinstance(key, int):
                                        return self.iloc[key]
                                    if isinstance(key, slice):
                                        ok = lambda x: (x is None) or isinstance(x, int)
                                        if ok(key.start) and ok(key.stop) and (key.step is None or isinstance(key.step, int)):
                                            return _HSeries(self.iloc[key])
                                    return super().__getitem__(key)
                            class _HDataFrame(_pd.DataFrame):
                                @property
                                def _constructor(self):
                                    return _HDataFrame
                                def __getitem__(self, key):
                                    obj = super().__getitem__(key)
                                    if isinstance(obj, _pd.Series):
                                        return _HSeries(obj)
                                    return obj
                            return _HDataFrame(out_df)
                        else:
                            return {sec: _reindex(s).to_numpy(dtype=float) for sec, s in results.items()}
                    exec_env['history'] = history
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
                def attribute_history(security, count, unit='1d', fields=None, skip_paused=True, df=True, fq='pre'):
                    """JoinQuant 风格 attribute_history 近似实现 (仅日级)。
                    兼容旧调用: attribute_history(sec, n, '1d', ['close']).
                    参数:
                      security: 证券代码
                      count: 回溯条数 (不含当前正在执行的当日，除非开启 attribute_history_include_current)
                      unit: 仅支持 '1d'
                      fields: list/tuple/str，默认 ['open','close','high','low','volume','money']
                      skip_paused: True 删除停牌日 (volume==0)；False 保留并前值填充（依赖 get_price fill_paused）
                      df: True 返回 DataFrame (index=日期升序)；False 返回 dict{field: np.ndarray}
                      fq: 'pre'|'post'|'None' 占位，不做真实复权转换（由数据文件决定）
                    差异: 未实现 factor/high_limit/low_limit 真值；factor 恒为1.0（若请求）。
                                        新增字段:
                                            (已移除 gap_fill：不再自动补齐缺失交易日)
                    """
                    import pandas as _pd
                    import numpy as _np
                    if unit.lower() not in ('1d','day','d'):
                        raise ValueError('attribute_history 目前仅支持日级 unit=1d')
                    # 包含当前 bar 开关
                    try:
                        _include_cur = bool(exec_env['jq_state']['options'].get('attribute_history_include_current', False))
                    except Exception:
                        _include_cur = False
                    # 规范 fields
                    if fields is None:
                        fields_list = ['open','close','high','low','volume','money']
                    else:
                        if isinstance(fields, str):
                            fields_list = [fields]
                        else:
                            fields_list = list(fields)
                    # 去重保持顺序
                    seen = set(); ordered = []
                    for f in fields_list:
                        if f not in seen:
                            ordered.append(f); seen.add(f)
                    fields_list = ordered
                    # 若请求 factor 且未在数据中存在 -> 占位
                    need_factor = 'factor' in fields_list
                    # 获取完整历史 (用 get_price 以利用其 fill_paused + 日期补齐)
                    get_price_fn = exec_env.get('get_price')
                    # get_price count 包含 end_date 自身；我们需要 count 条不含今日
                    # 先抓 (count + (1 if not _include_cur else 0)) 条，然后再裁剪
                    gp_extra = count if _include_cur else count
                    # 使用基础字段集合以便派生 money/factor
                    base_reqs = set(fields_list)
                    # Option C: 若用户请求 volume 且 skip_paused=False，自动附加 paused 字段用于区分真实停牌
                    auto_add_paused = False
                    if (not skip_paused) and ('volume' in base_reqs) and ('paused' not in base_reqs):
                        base_reqs.add('paused')
                        auto_add_paused = True
                    # money 需要 close + volume
                    if 'money' in base_reqs:
                        base_reqs.update(['close','volume'])
                    # 如果要 skip_paused，需要 paused 字段
                    if skip_paused:
                        base_reqs.add('paused')
                    # high_limit/low_limit 需要 pre_close (或 close.shift(1))
                    if 'high_limit' in base_reqs or 'low_limit' in base_reqs:
                        base_reqs.add('close')
                        base_reqs.add('pre_close')
                    # factor 占位不增加依赖；无特别处理
                    base_fields = list(base_reqs)
                    raw_df = get_price_fn(security, count=gp_extra+5, fields=base_fields, skip_paused=False, fq=fq, fill_paused=True)  # +5 冗余确保筛后足够
                    if isinstance(raw_df, _pd.Series):
                        raw_df = raw_df.to_frame(name=base_fields[0])
                    # 过滤未来日期（不含今日）除非 include_cur
                    try:
                        cur_date = bt.num2date(self.data.datetime[0]).date()
                    except Exception:
                        cur_date = None
                    if cur_date is not None:
                        if _include_cur:
                            raw_df = raw_df[raw_df.index <= cur_date]
                        else:
                            raw_df = raw_df[raw_df.index < cur_date]
                    # 卷尾截取最新 count 行
                    work = raw_df.tail(count)
                    # 生成派生字段 (money, factor, high_limit/low_limit)
                    if 'money' in fields_list and 'money' not in work.columns:
                        if {'close','volume'} <= set(work.columns):
                            work['money'] = work['close'] * work['volume']
                        else:
                            work['money'] = _pd.NA
                    if need_factor and 'factor' not in work.columns:
                        work['factor'] = 1.0
                    if 'high_limit' in fields_list or 'low_limit' in fields_list:
                        try:
                            limit_pct = float(exec_env['jq_state']['options'].get('limit_pct', 0.1) or 0.1)
                        except Exception:
                            limit_pct = 0.1
                        ref_pc = work['pre_close'] if 'pre_close' in work.columns else work['close'].shift(1)
                        tick = float(exec_env['jq_state']['options'].get('price_tick', 0.01) or 0.01)
                        def _round_tick(v):
                            return _math.floor(v / tick + 1e-9) * tick
                        if 'high_limit' in fields_list:
                            work['high_limit'] = ref_pc * (1 + limit_pct)
                            work['high_limit'] = work['high_limit'].map(lambda x: _round_tick(x) if _pd.notnull(x) else x)
                        if 'low_limit' in fields_list:
                            work['low_limit'] = ref_pc * (1 - limit_pct)
                            work['low_limit'] = work['low_limit'].map(lambda x: _round_tick(x) if _pd.notnull(x) else x)
                    # skip_paused: 使用 paused 字段（原始停牌），若无则回退到 volume
                    if skip_paused:
                        if 'paused' in work.columns:
                            work = work[work['paused'] == 0]
                        elif 'volume' in work.columns:
                            work = work[work['volume'] != 0]
                    # 只留请求字段顺序；若自动附加了 paused，则保留它方便用户统计真实停牌
                    keep_cols = [c for c in fields_list if c in work.columns]
                    if auto_add_paused and 'paused' in work.columns and 'paused' not in keep_cols:
                        keep_cols.append('paused')
                    work = work[keep_cols]
                    # 结果 DataFrame 升序，索引去 name
                    work = work.sort_index()
                    try:
                        work.index.name = None
                    except Exception:
                        pass
                    # 负整数位置索引支持
                    class _JQSeries(_pd.Series):
                        @property
                        def _constructor(self):
                            return _JQSeries
                        def __getitem__(self, key):
                            if isinstance(key, int):
                                return self.iloc[key]
                            if isinstance(key, slice):
                                ok = lambda x: (x is None) or isinstance(x, int)
                                if ok(key.start) and ok(key.stop) and (key.step is None or isinstance(key.step, int)):
                                    return _JQSeries(self.iloc[key])
                            return super().__getitem__(key)
                    class _JQDataFrame(_pd.DataFrame):
                        @property
                        def _constructor(self):
                            return _JQDataFrame
                        @property
                        def _constructor_sliced(self):
                            return _JQSeries
                        def __getitem__(self, key):
                            if isinstance(key, int):
                                return self.iloc[key]
                            if isinstance(key, slice):
                                ok = lambda x: (x is None) or isinstance(x, int)
                                if ok(key.start) and ok(key.stop) and (key.step is None or isinstance(key.step, int)):
                                    return _JQDataFrame(self.iloc[key])
                            return super().__getitem__(key)
                    if df:
                        return _JQDataFrame(work)
                    else:
                        return {c: work[c].to_numpy(dtype=float) for c in keep_cols}

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
                        # 半滑点：计算潜在“执行”价格（仅用于估算真实成交，不再影响 sizing）
                        half = slip_perc / 2.0
                        def _round_buy(p: float) -> float:  # use module-level _math
                            return _math.floor(p * 100) / 100.0
                        def _round_sell(p: float):
                            return _math.ceil(p * 100) / 100.0
                        price = base_price
                        debug_trading = bool(jqst['options'].get('debug_trading', False))
                        exec_buy_price = _round_buy(base_price * (1 + half)) if slip_perc else base_price
                        exec_sell_price = _round_sell(base_price * (1 - half)) if slip_perc else base_price
                        # 是否在 sizing 中忽略滑点（默认 True，更贴近聚宽：shares = floor(cash/open)）
                        sizing_use_raw = True
                        try:
                            opt_sz_raw = jqst['options'].get('sizing_use_raw_open_price')
                            if isinstance(opt_sz_raw, bool):
                                sizing_use_raw = opt_sz_raw
                        except Exception:
                            sizing_use_raw = True
                        if debug_trading:
                            jqst['log'].append(f"[sizing_slip_mode] sizing_use_raw_open_price={sizing_use_raw} slip_perc={slip_perc}")
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
                                eff_price = exec_buy_price if side_tmp == 'BUY' else exec_sell_price
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
                            # sizing 基准价：raw base 或（旧方案）含滑点执行价
                            sizing_price = base_price if sizing_use_raw else exec_buy_price
                            est_price = sizing_price
                            # 可选“保守模式”：若上一日收盘(prev_close)高于买入有效价，用 prev_close 作为 sizing 基准，避免实际撮合价>估价导致保证金 0。
                            # 聚宽本身按 (cash // buy_price) 取整再对齐 lot；这里通过开关控制：True=当前默认保守, False=严格 JQ 样式。
                            conservative_flag = False  # 默认关闭：与聚宽一致，按开盘价(或 sizing_price) 直接 sizing
                            try:
                                opt_cons = jqst['options'].get('order_value_conservative_prev_close')
                                if isinstance(opt_cons, bool):
                                    conservative_flag = opt_cons
                            except Exception:
                                conservative_flag = False
                            conservative_price = est_price
                            prev_close_for_size = None
                            try:
                                prev_close_for_size = float(getattr(self.data, 'close')[-1])
                            except Exception:
                                prev_close_for_size = None
                            if conservative_flag and prev_close_for_size and prev_close_for_size > conservative_price:
                                conservative_price = prev_close_for_size
                            raw_shares = int(value // (conservative_price if conservative_flag else sizing_price))
                            # 按 lot 对齐
                            shares = (raw_shares // lot) * lot
                            if shares <= 0:
                                if debug_trading:
                                    exec_env['log'].info(f"[sizing] BUY abort raw_shares={raw_shares} lot={lot} est={est_price} conservative={conservative_price} cash={available_cash}")
                                return
                            # 费用估算（迭代一次足够）
                            # 费用与现金占用使用“执行价格” (含滑点)，以避免超买；若无滑点则等于 base
                            exec_price_for_cost = exec_buy_price
                            gross_price_for_fee = exec_price_for_cost if (conservative_flag or sizing_use_raw) else exec_buy_price
                            gross = shares * gross_price_for_fee
                            comm = max(gross * commission_rate, min_comm)
                            total_cost = gross + comm  # 不含印花税（买入无印花税）
                            if total_cost > available_cash:
                                # 迭代减 lot 直到满足现金（保留简单循环，lot 数量通常极少）
                                while shares > 0:
                                    max_cost = shares * gross_price_for_fee + max(shares * gross_price_for_fee * commission_rate, min_comm)
                                    if max_cost <= available_cash:
                                        break
                                    shares -= lot
                                if shares < 0:
                                    shares = 0
                            if shares <= 0:
                                if debug_trading:
                                    exec_env['log'].info(f"[sizing] BUY zero_after_cash cons_price={gross_price_for_fee} cash={available_cash}")
                                return
                            if debug_trading:
                                exec_env['log'].info(
                                    f"[sizing] BUY shares={shares} sizing_price={sizing_price} exec_price={exec_buy_price} prev_close={prev_close_for_size} cons_flag={conservative_flag} cons_price={conservative_price} gross_price_used={gross_price_for_fee} gross={gross:.2f} comm_est={comm:.2f} total={total_cost:.2f} cash={available_cash:.2f}"
                                )
                            else:
                                # 非 debug 也记录一次模式
                                try:
                                    jqst['log'].append(
                                        f"[sizing_mode] conservative_prev_close={conservative_flag} sizing_use_raw={sizing_use_raw} raw_shares={raw_shares} final_shares={shares} base_price={base_price} prev_close={prev_close_for_size} exec_price={exec_buy_price}"
                                    )
                                except Exception:
                                    pass
                            self.buy(size=shares)
                        elif value < 0:
                            # 卖出金额：目标卖出 value_abs ，不超过持仓
                            est_price = exec_sell_price if not sizing_use_raw else base_price
                            value_abs = abs(value)
                            max_sell_shares = position_size if position_size > 0 else 0
                            if max_sell_shares <= 0:
                                return
                            # 需要的股数
                            raw_need = int(value_abs // (base_price if sizing_use_raw else exec_sell_price))
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
                                exec_env['log'].info(f"[sizing] SELL shares={shares} sizing_price={(base_price if sizing_use_raw else exec_sell_price)} exec_price={exec_sell_price} pos={position_size}")
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
                        exec_buy_price = _round_buy(base_price * (1 + half)) if slip_perc else base_price
                        exec_sell_price = _round_sell(base_price * (1 - half)) if slip_perc else base_price
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
                            if delta > 0 and up_lim is not None and exec_buy_price >= up_lim - 1e-9:
                                exec_env['log'].info(
                                    f"[limit_check] BLOCK side=BUY price={exec_buy_price:.4f} up_lim={up_lim:.4f} prev_close={prev_close:.4f}"
                                )
                                jqst['blocked_orders'].append(OrderRecord(
                                    datetime=jqst.get('current_dt','').split(' ')[0],
                                    symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                    side='BUY',
                                    size=0,
                                    price=exec_buy_price,
                                    value=0.0,
                                    commission=0.0,
                                    status='BlockedLimitUp'
                                ))
                                return
                            if delta < 0 and down_lim is not None and exec_sell_price <= down_lim + 1e-9:
                                exec_env['log'].info(
                                    f"[limit_check] BLOCK side=SELL price={exec_sell_price:.4f} down_lim={down_lim:.4f} prev_close={prev_close:.4f}"
                                )
                                jqst['blocked_orders'].append(OrderRecord(
                                    datetime=jqst.get('current_dt','').split(' ')[0],
                                    symbol=jqst.get('primary_symbol') or str(security).split('.')[0],
                                    side='SELL',
                                    size=0,
                                    price=exec_sell_price,
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
    frequency: str = 'daily',
    adjust_type: str = 'auto',
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
        # 若调用方传入 adjust_type / frequency 且用户未在策略 set_option 指定，则采用传入值
        if 'adjust_type' not in jq_state.get('options', {}):
            jq_state['options']['adjust_type'] = adjust_type
        jq_state['options']['api_frequency'] = frequency
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

        # 2. 统一标的解析与数据加载 (完全使用 data_loader)
        symbols: List[str] = []
        g_sec = getattr(jq_state.get('g'), 'security', None)
        if g_sec:
            if isinstance(g_sec, (list, tuple, set)):
                symbols = [str(s).strip() for s in g_sec if str(s).strip()]
            elif isinstance(g_sec, str):
                symbols = [g_sec.strip()]
        if not symbols:
            if isinstance(symbol, str):
                symbols = [s.strip() for s in symbol.split(',') if s.strip()]
            else:
                symbols = list(symbol)
        if not symbols:
            raise ValueError('未指定任何标的: 请在策略 g.security 或 表单 symbol 提供至少一个')
        # 规范代码（去掉交易所后缀 / 复权后缀标记）
        def _base_code(s: str) -> str:
            c = s.replace('.XSHE','').replace('.XSHG','').replace('.xshe','').replace('.xshg','')
            # 若带下划线如 000514_daily_qfq 只取数字前缀
            return re.split(r'[_ ]+', c)[0]
        base_symbols = list(dict.fromkeys([_base_code(s) for s in symbols]))
        jq_state['primary_symbol'] = base_symbols[0]
        jq_state.setdefault('symbol_file_map', {})
        jq_state['log'].append(f"[symbol_unified] input={symbols} base={base_symbols} freq={frequency} adjust={jq_state['options'].get('adjust_type')}")
        # 历史预热天数推断 & warmup_start 计算保持原逻辑 (下面代码依赖 lookback_days)

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

        final_adjust = jq_state['options'].get('adjust_type', adjust_type)
        use_real_price_flag = jq_state.get('options', {}).get('use_real_price')
        for i, base in enumerate(base_symbols):
            try:
                _path_holder_feed = {}
                feed = _dl.load_bt_feed(base, warmup_start, end, frequency=frequency, adjust=final_adjust, prefer_stockdata=True,
                                        use_real_price=use_real_price_flag, out_path_holder=_path_holder_feed)
                cerebro.adddata(feed, name=base)
                _path_holder_df = {}
                full_df = _dl.load_price_dataframe(base, warmup_start, end, frequency=frequency, adjust=final_adjust, prefer_stockdata=True,
                                                   use_real_price=use_real_price_flag, out_path_holder=_path_holder_df)
                jq_state['history_df_map'][base] = full_df
                if i == 0:
                    jq_state['history_df'] = full_df
                # 记录真实文件路径（feed 与 dataframe 应该一致；优先 dataframe 的记录）
                sel_path = _path_holder_df.get('path') or _path_holder_feed.get('path')
                jq_state['symbol_file_map'][base] = sel_path or f"{base}:{frequency}:{final_adjust}"
                jq_state['log'].append(f"[data_loader] code={base} freq={frequency} adjust={final_adjust} use_real_price={use_real_price_flag} rows={len(full_df)} file={sel_path}")
            except Exception as _e:
                jq_state['log'].append(f"[data_loader_error] code={base} err={type(_e).__name__}:{_e}")
                raise
        symbols = base_symbols
        # 处理基准：策略内 set_benchmark 优先于外部参数
        if jq_state.get('benchmark'):
            benchmark_symbol = jq_state['benchmark']
        # 1min 回测暂仍以日线基准 (向用户提示)
        if benchmark_symbol and frequency == '1min':
            jq_state['log'].append('[benchmark_notice] 1min 回测暂使用日线基准对齐')

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

        # 简化数据变体：直接使用最终 adjust_type（单标的字符串 / 多标的统一）
        data_variant = jq_state.get('options', {}).get('adjust_type')

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
                benchmark_symbol = jq_bm_post
                metrics['benchmark_detect_phase'] = 'post_run_initialize'
            else:
                # 默认基准回退: 使用 000300 (日线) 若存在
                benchmark_symbol = '000300'
                metrics['benchmark_detect_phase'] = 'fallback_default'
                metrics['benchmark_fallback'] = True
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
            bench_freq = frequency if frequency in ('daily','weekly','monthly') else 'daily'
            try:
                bench_df = _dl.load_price_dataframe(benchmark_symbol.replace('.XSHG','').replace('.XSHE',''), bench_warmup_start, end, frequency=bench_freq, adjust='auto', prefer_stockdata=True)
            except Exception:
                bench_df = pd.DataFrame()
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
            metrics['data_source_preference'] = jq_state.get('options', {}).get('data_source_preference')
            metrics['data_sources_used'] = jq_state.get('symbol_file_map')
            metrics['adjust_type'] = jq_state.get('options', {}).get('adjust_type')
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
