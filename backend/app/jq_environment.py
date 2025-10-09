"""聚宽兼容执行环境：为用户策略提供仿真的全局变量与受限导入机制。"""
from __future__ import annotations

from typing import Any, Dict

import backtrader as bt
import pandas as pd

from . import stock_pool

__all__ = [
    "ALLOWED_GLOBALS",
    "_IMPORT_WHITELIST",
    "IMPORT_WHITELIST",
    "limited_import",
    "_limited_import",
    "build_jq_compat_env",
    "_build_jq_compat_env",
]


ALLOWED_GLOBALS: Dict[str, Any] = {
    "__builtins__": {
        "abs": abs,
        "min": min,
        "max": max,
        "range": range,
        "len": len,
        "sum": sum,
        "enumerate": enumerate,
        "zip": zip,
        "float": float,
        "int": int,
        "str": str,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "bool": bool,
        "isinstance": isinstance,
        "print": print,
        "__build_class__": __build_class__,
        "__name__": "__main__",
    },
    "bt": bt,
    "pd": pd,
    "get_index_stocks": stock_pool.get_index_stocks,
    "get_index_weights": stock_pool.get_index_weights,
    "get_industry_stocks": stock_pool.get_industry_stocks,
    "get_concept_stocks": stock_pool.get_concept_stocks,
    "get_all_securities": stock_pool.get_all_securities,
}

IMPORT_WHITELIST: Dict[str, Any] = {
    "backtrader": bt,
    "pandas": pd,
    "math": __import__("math"),
    "statistics": __import__("statistics"),
    "numpy": None,
}

_IMPORT_WHITELIST = IMPORT_WHITELIST


def limited_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: D401
    """受限导入函数：仅允许白名单模块被用户策略导入。"""

    root = name.split(".")[0]
    if root not in IMPORT_WHITELIST:
        raise ImportError(f"不允许导入模块: {name}")
    if root == "numpy" and IMPORT_WHITELIST[root] is None:
        IMPORT_WHITELIST[root] = __import__(root)
    return __import__(name, globals, locals, fromlist, level)


ALLOWED_GLOBALS["__builtins__"]["__import__"] = limited_import

_limited_import = limited_import


class _G:  # pragma: no cover - trivial container
    pass


def build_jq_compat_env(target_dict: Dict[str, Any]) -> Dict[str, Any]:
    """为目标字典填充聚宽兼容的工具函数，并返回 ``jq_state`` 状态字典。"""

    g = _G()
    jq_state: Dict[str, Any] = {
        "benchmark": None,
        "options": {
            "enable_limit_check": True,
            "limit_up_factor": 1.10,
            "limit_down_factor": 0.90,
            "price_tick": 0.01,
            "limit_pct": 0.10,
        },
        "records": [],
        "log": [],
        "g": g,
        "history_df": None,
        "history_df_map": {},
        "minute_daily_cache": {},
        "current_dt": None,
        "user_start": None,
        "in_warmup": False,
        "blocked_orders": [],
        "trading_calendar": None,
    }

    try:
        import os
        from datetime import datetime

        cal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading_calendar.csv")
        cal_path = os.path.abspath(cal_path)
        if os.path.exists(cal_path):
            cal_df = pd.read_csv(cal_path)
            col = "date" if "date" in cal_df.columns else cal_df.columns[0]
            dates = set(pd.to_datetime(cal_df[col]).dt.date.astype(str))
            if dates:
                jq_state["trading_calendar"] = dates
                jq_state["log"].append(f"[trading_calendar_loaded] size={len(dates)}")
    except Exception:
        pass

    class _Log:
        def info(self, msg):
            if jq_state.get("in_warmup"):
                return
            try:
                dt_value = jq_state.get("current_dt") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_value = "0000-00-00 00:00:00"
            header = f"{dt_value} - INFO - "
            lines = str(msg).splitlines() or [""]
            formatted = "\n".join([header + lines[0]] + ["    " + ln for ln in lines[1:]])
            jq_state["log"].append(formatted)
            print(formatted)

    log_obj = _Log()

    def set_benchmark(code: str):
        jq_state["benchmark"] = code

    def set_option(name: str, value: Any):
        jq_state["options"][name] = value
        jq_state["log"].append(f"[set_option] {name}={value}")

    def record(**kwargs):
        if jq_state.get("in_warmup"):
            return
        jq_state["records"].append({"dt": None, **kwargs})

    def set_slippage(obj=None, type=None, ref=None):  # noqa: A002, D401
        import re

        if obj is None:
            return
        s = str(obj)
        m_fixed = re.search(r"FixedSlippage\s*\(\s*([0-9eE\.+-]+)\s*\)", s)
        m_price = re.search(r"PriceRelatedSlippage\s*\(\s*([0-9eE\.+-]+)\s*\)", s)
        try:
            if m_price:
                val = float(m_price.group(1))
                jq_state["options"]["slippage_perc"] = val
                jq_state["log"].append(f"[set_slippage] PriceRelatedSlippage perc={val}")
            elif m_fixed:
                val = float(m_fixed.group(1))
                jq_state["options"]["fixed_slippage"] = val
                jq_state["log"].append(f"[set_slippage] FixedSlippage value={val}")
            else:
                val = float(obj)
                jq_state["options"]["slippage_perc"] = val
                jq_state["log"].append(f"[set_slippage] perc={val}")
        except Exception:
            jq_state["log"].append(f"[set_slippage_warning] unrecognized={s}")

    def attribute_history(security: str, n: int, unit: str, fields):  # noqa: D401
        raise RuntimeError("attribute_history 仅在聚宽兼容包装策略中可用")

    def order(security: str, amount: int):  # noqa: D401
        raise RuntimeError("order 仅在聚宽兼容包装策略中可用")

    def order_value(security: str, value: float):  # noqa: D401
        raise RuntimeError("order_value 仅在聚宽兼容包装策略中可用")

    def order_target(security: str, target: float):  # noqa: D401
        raise RuntimeError("order_target 仅在聚宽兼容包装策略中可用")

    target_dict.update(
        {
            "g": g,
            "set_benchmark": set_benchmark,
            "set_option": set_option,
            "set_slippage": set_slippage,
            "record": record,
            "log": log_obj,
            "attribute_history": attribute_history,
            "order": order,
            "order_value": order_value,
            "order_target": order_target,
            "jq_state": jq_state,
            "get_index_stocks": stock_pool.get_index_stocks,
            "get_index_weights": stock_pool.get_index_weights,
            "get_industry_stocks": stock_pool.get_industry_stocks,
            "get_concept_stocks": stock_pool.get_concept_stocks,
            "get_all_securities": stock_pool.get_all_securities,
        }
    )
    return jq_state


_build_jq_compat_env = build_jq_compat_env
