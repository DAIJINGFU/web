"""策略编译模块：负责执行用户脚本并构建聚宽兼容策略类。"""
import math as _math
import re
import types
from typing import Any, Callable, Dict, List

import backtrader as bt
import pandas as pd

from . import data_loader as _dl
from .corporate_actions import apply_event, load_corporate_actions
from .data_compat import load_csv_data, load_csv_dataframe, round_to_tick as _round_to_tick
from .jq_environment import (
    ALLOWED_GLOBALS,
    _build_jq_compat_env,
    _limited_import,
)
from .market_controls import is_stock_paused
from .models import OrderRecord


ALLOWED_GLOBALS['__builtins__']['__import__'] = _limited_import


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
    
    # 关键修复：将 exec_env 的内容合并到 module.__dict__ 中
    # 这样 initialize 和 handle_data 函数就能访问股票池函数了
    for key, value in exec_env.items():
        if key not in module.__dict__:
            module.__dict__[key] = value
    
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
                        # T+1制度：计算可卖数量 = 总持仓 - 当日买入数量
                        total_amount = int(self.position.size)
                        current_date = jq_state.get('current_dt', '').split(' ')[0]
                        today_bought = jq_state.get('_daily_bought', {}).get(current_date, 0)
                        closeable = max(0, total_amount - today_bought)
                        
                        return {getattr(jq_state['g'], 'security', 'data0'): types.SimpleNamespace(
                            closeable_amount=closeable,
                            total_amount=total_amount
                        )}

                class _Context:
                    portfolio = _Portfolio()
                    
                    @property
                    def current_dt(inner_self):
                        """返回当前日期时间对象"""
                        try:
                            import datetime as dt_module
                            current_dt_str = jq_state.get('current_dt', '')
                            if current_dt_str:
                                # current_dt 格式为 "YYYY-MM-DD HH:MM:SS"
                                return dt_module.datetime.strptime(current_dt_str, '%Y-%m-%d %H:%M:%S')
                            else:
                                # 回退到当前bar的时间
                                return bt.num2date(self.data.datetime[0])
                        except Exception:
                            return bt.num2date(self.data.datetime[0])

                self._jq_context = _Context()
                # 绑定执行环境引用，供 _run_handle / next_open / next 使用
                try:
                    self._exec_env = globals().get('exec_env') or exec_env  # 若全局未设定，则回退到当前执行环境
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
                            # 若源数据为分钟级（同一 date 多行）且当前仅支持日级查询，则聚合为日线
                            try:
                                if df['date'].nunique() < len(df):  # 存在重复日期 => 可能是分钟/更细粒度
                                    # 使用缓存：首次对该 symbol 聚合后保存，后续直接复用
                                    cache = jq_state.setdefault('minute_daily_cache', {})
                                    cache_key = base
                                    cached_daily = cache.get(cache_key)
                                    if cached_daily is None:
                                        # 聚合规则：open=第一条，close=最后一条，high=max，low=min，volume=sum
                                        agg_cols = {}
                                        for col in ['open','close','high','low','volume']:
                                            if col in df.columns:
                                                if col == 'open':
                                                    agg_cols[col] = 'first'
                                                elif col == 'close':
                                                    agg_cols[col] = 'last'
                                                elif col == 'high':
                                                    agg_cols[col] = 'max'
                                                elif col == 'low':
                                                    agg_cols[col] = 'min'
                                                elif col == 'volume':
                                                    agg_cols[col] = 'sum'
                                        gdf = df.groupby('date').agg(agg_cols).reset_index()
                                        import pandas as _pd
                                        gdf['datetime'] = _pd.to_datetime(gdf['date'])
                                        ordered_cols = ['datetime','open','high','low','close','volume']
                                        gdf = gdf[[c for c in ordered_cols if c in gdf.columns]]
                                        gdf['date'] = gdf['datetime'].dt.date
                                        cache[cache_key] = gdf  # 存缓存
                                        try:
                                            if not jq_state.get('_minute_source_aggregated_logged'):
                                                jq_state['_minute_source_aggregated_logged'] = True
                                                jq_state['log'].append(f"[minute_source_aggregate] detected minute-level history -> aggregated to daily for get_price/attribute_history")
                                            jq_state['log'].append(f"[minute_daily_cache] build sec={cache_key} rows={len(gdf)}")
                                        except Exception:
                                            pass
                                        df = gdf.copy()
                                    else:
                                        # 直接复用缓存（复制避免下游修改）
                                        df = cached_daily.copy()
                            except Exception:
                                pass
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
                                def __init__(self, data_map):  # data_map 代表 证券 -> DataFrame 的映射
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
                                # 回退：尝试使用 g.security
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
                        # 工具函数：提取单个证券的 DataFrame
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
                                    # 直接注入持仓增量，正数代表买入、负数代表卖出
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
                    # 方案 C：若用户请求 volume 且 skip_paused=False，则自动附加 paused 字段用于区分真实停牌
                    auto_add_paused = False
                    if (not skip_paused) and ('volume' in base_reqs) and ('paused' not in base_reqs):
                        base_reqs.add('paused')
                        auto_add_paused = True
                    # money 字段需要 close 与 volume 同时存在
                    if 'money' in base_reqs:
                        base_reqs.update(['close','volume'])
                    # 如果要 skip_paused，需要 paused 字段
                    if skip_paused:
                        base_reqs.add('paused')
                    # high_limit/low_limit 需要 pre_close（或 close.shift(1)）
                    if 'high_limit' in base_reqs or 'low_limit' in base_reqs:
                        base_reqs.add('close')
                        base_reqs.add('pre_close')
                    # factor 字段为占位，不增加额外依赖
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
                    # skip_paused：优先使用 paused 字段（原始停牌），若无则回退到 volume
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
                        def _round_buy(p: float) -> float:  # 使用模块级 _math，避免重复导入
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
                    # order_value 尾部逻辑
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
                        # 使用模块级 _math
                        def _round_buy(p: float) -> float:
                            return _math.floor(p * 100) / 100.0
                        def _round_sell(p: float) -> float:
                            return _math.ceil(p * 100) / 100.0
                        exec_buy_price = _round_buy(base_price * (1 + half)) if slip_perc else base_price
                        exec_sell_price = _round_sell(base_price * (1 - half)) if slip_perc else base_price
                        price = base_price
                        # 处理涨跌停检查
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
                    # order_target 尾部逻辑
                
                def order(security: str, amount: int):
                    """
                    聚宽风格的下单函数（按股数下单）
                    
                    Args:
                        security: 股票代码
                        amount: 股数（正数买入，负数卖出）
                    
                    Returns:
                        Order对象或None
                    """
                    try:
                        # 获取当前价格
                        data = self.data
                        fillp = str(exec_env.get('jq_state', {}).get('options', {}).get('fill_price', 'open')).lower()
                        price = float(getattr(data, 'close')[0]) if fillp == 'close' else float(getattr(data, 'open')[0]) if hasattr(data, 'open') else float(getattr(data, 'close')[0])
                        
                        # 计算金额
                        value = abs(amount) * price
                        
                        # 买入或卖出
                        if amount > 0:
                            # 买入：使用 order_value 逻辑
                            return order_value(security, value)
                        elif amount < 0:
                            # 卖出：使用 order_value 逻辑（负值）
                            return order_value(security, -value)
                        else:
                            return None
                    except Exception as _e:
                        try:
                            exec_env['log'].info(f"[order_error] {type(_e).__name__}:{_e}")
                        except Exception:
                            pass
                        return None
                    # order 函数尾部逻辑
                
                # 暴露函数到执行环境（供用户脚本中的全局方法调用）
                try:
                    exec_env['attribute_history'] = attribute_history
                    exec_env['order'] = order
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
                        
                        # T+1制度：清理过期的买入记录（保留最近2天，避免日期跳跃问题）
                        if '_daily_bought' in jq_state:
                            try:
                                from datetime import datetime, timedelta
                                current_date_obj = datetime.strptime(cur_dt, '%Y-%m-%d')
                                cutoff_date = (current_date_obj - timedelta(days=2)).strftime('%Y-%m-%d')
                                keys_to_delete = [k for k in jq_state['_daily_bought'].keys() if k < cutoff_date]
                                for k in keys_to_delete:
                                    del jq_state['_daily_bought'][k]
                            except Exception:
                                pass
                        
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

