"""数据加载管线：负责统一标的解析与历史数据装载。

该模块的职责是将调用方提供的任意形式标的（单字符串、多标的列表、含交易所/后缀）统一解析为回测可识别的基础代码，
并依据用户指定的频率、复权模式以及必要的预热长度，自动加载回测所需的历史行情。所有加载过程中产生的结果会写入
``jq_state``，供后续执行管线直接复用，避免重复 I/O 和数据清洗。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import backtrader as bt
import pandas as pd

from . import data_loader as _dl


def prepare_data_sources(
    cerebro: bt.Cerebro,
    jq_state: Dict[str, Any],
    symbol_input: Sequence[str] | str,
    start: str,
    end: str,
    frequency: str,
    adjust_type: str,
    strategy_code: str,
    benchmark_symbol: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """解析策略标的并加载历史数据，返回标准化证券列表与基准代码。

    参数说明：
    - ``cerebro``：Backtrader 核心对象，用于挂载数据源与策略；
    - ``jq_state``：聚宽兼容状态字典，包含策略运行时上下文；
    - ``symbol_input``：来自表单或策略的原始标的输入，允许逗号分隔字符串或列表；
    - ``start``/``end``：回测时间范围（ISO 日期字符串）；
    - ``frequency``：数据频率，例如 ``daily`` / ``weekly`` / ``1min``；
    - ``adjust_type``：复权模式初始值，将与策略选项合并；
    - ``strategy_code``：策略源码文本，用于自动推断所需预热期；
    - ``benchmark_symbol``：基准代码，允许 ``None`` 表示未设置。

    该函数会直接修改 ``jq_state``，以便后续策略执行阶段访问：
    - ``primary_symbol``：主交易标的，用于回测结果聚焦；
    - ``history_df`` / ``history_df_map``：加载后的行情 DataFrame 缓存，减少重复读取；
    - ``symbol_file_map``：标的到原始文件路径映射，便于日志追溯；
    - ``log``：字符串数组，详细记录数据加载发生的每个步骤与异常。

    返回值按顺序提供标准化后的标的列表与最终基准代码，供执行管线继续使用。
    """

    symbols: List[str] = []
    g_sec = getattr(jq_state.get("g"), "security", None)
    if g_sec:
        if isinstance(g_sec, (list, tuple, set)):
            symbols = [str(s).strip() for s in g_sec if str(s).strip()]
        elif isinstance(g_sec, str):
            symbols = [g_sec.strip()]

    if not symbols:
        if isinstance(symbol_input, str):
            symbols = [s.strip() for s in symbol_input.split(",") if s.strip()]
        elif isinstance(symbol_input, Iterable):
            symbols = [str(s).strip() for s in symbol_input if str(s).strip()]
        else:
            symbols = []

    if not symbols:
        raise ValueError("未指定任何标的: 请在策略 g.security 或 表单 symbol 提供至少一个")

    def _base_code(code: str) -> str:
        """提取六位基础代码。

        兼容聚宽常见写法（如 ``000001.XSHE``）、旧数据文件名后缀（``_daily_qfq``）以及用户输入的空格/下划线分隔符，
        确保最终用于文件定位的 key 是纯六位数字。
        """

        cleaned = code.replace(".XSHE", "").replace(".XSHG", "").replace(".xshe", "").replace(".xshg", "")
        return re.split(r"[_ ]+", cleaned)[0]

    base_symbols = list(dict.fromkeys([_base_code(s) for s in symbols]))
    jq_state["primary_symbol"] = base_symbols[0]
    jq_state.setdefault("symbol_file_map", {})
    jq_state.setdefault("history_df_map", {})
    jq_state.setdefault("log", []).append(
        f"[symbol_unified] input={symbols} base={base_symbols} freq={frequency} adjust={jq_state['options'].get('adjust_type')}"
    )

    # lookback_days 表示为了让指标顺利计算，需要额外加载的历史天数
    lookback_days = 250
    user_set_lb = False
    try:
        lb = jq_state.get("options", {}).get("history_lookback_days")
        if isinstance(lb, (int, float)) and lb >= 0:
            lookback_days = int(lb)
            user_set_lb = True
    except Exception:
        pass

    try:
        auto_flag = bool(jq_state.get("options", {}).get("jq_auto_history_preload", True))
    except Exception:
        auto_flag = True

    if (not user_set_lb) and auto_flag:
        try:
            periods: List[int] = []
            for m in re.finditer(r"period\s*=\s*(\d{1,4})", strategy_code):
                periods.append(int(m.group(1)))
            for m in re.finditer(
                r"\b(SMA|EMA|MA|ATR|RSI|WMA|TRIMA|KAMA|ADX|CCI)\s*\(\s*[^,\n]*?(\d{1,4})",
                strategy_code,
                re.IGNORECASE,
            ):
                try:
                    periods.append(int(m.group(2)))
                except Exception:
                    continue
            periods = [p for p in periods if p >= 3]
            if periods:
                max_period = max(periods)
                auto_lb = min(max_period * 3, 600)
                if auto_lb > lookback_days:
                    lookback_days = auto_lb
                jq_state["log"].append(
                    f"[auto_history_preload] detected_periods={sorted(set(periods))} max={max_period} lookback_days={lookback_days}"
                )
            else:
                jq_state["log"].append(f"[auto_history_preload] none_detected use_default={lookback_days}")
        except Exception as err:
            jq_state["log"].append(f"[auto_history_preload_error] {type(err).__name__}:{err}")

    warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=lookback_days)).date().isoformat()
    # symbol_file_map 在前面已初始化，这里重复 setdefault 以防调用方提前填入自定义值
    jq_state.setdefault("symbol_file_map", {})

    final_adjust = jq_state.get("options", {}).get("adjust_type", adjust_type)
    use_real_price_flag = jq_state.get("options", {}).get("use_real_price")

    for idx, base in enumerate(base_symbols):
        try:
            feed_holder: Dict[str, Any] = {}
            feed = _dl.load_bt_feed(
                base,
                warmup_start,
                end,
                frequency=frequency,
                adjust=final_adjust,
                prefer_stockdata=True,
                use_real_price=use_real_price_flag,
                out_path_holder=feed_holder,
            )
            cerebro.adddata(feed, name=base)

            df_holder: Dict[str, Any] = {}
            full_df = _dl.load_price_dataframe(
                base,
                warmup_start,
                end,
                frequency=frequency,
                adjust=final_adjust,
                prefer_stockdata=True,
                use_real_price=use_real_price_flag,
                out_path_holder=df_holder,
            )
            # 将完整数据缓存进 jq_state，后续指标或结果处理可以直接复用
            jq_state.setdefault("history_df_map", {})[base] = full_df
            if idx == 0:
                jq_state["history_df"] = full_df

            selected_path = df_holder.get("path") or feed_holder.get("path")
            jq_state["symbol_file_map"][base] = selected_path or f"{base}:{frequency}:{final_adjust}"
            jq_state["log"].append(
                f"[data_loader] code={base} freq={frequency} adjust={final_adjust} use_real_price={use_real_price_flag} rows={len(full_df)} file={selected_path}"
            )
        except Exception as err:
            jq_state["log"].append(f"[data_loader_error] code={base} err={type(err).__name__}:{err}")
            raise

    symbols = base_symbols
    if jq_state.get("benchmark"):
        benchmark_symbol = jq_state["benchmark"]

    if benchmark_symbol and frequency == "1min":
        jq_state["log"].append("[benchmark_notice] 1min 回测暂使用日线基准对齐")

    return symbols, benchmark_symbol
