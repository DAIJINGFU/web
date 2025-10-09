"""对外兼容的数据加载辅助方法，方便历史项目或第三方集成复用。"""
from __future__ import annotations

import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import backtrader as bt
import pandas as pd

from . import data_loader as _dl

__all__ = [
    "round_to_tick",
    "_round_to_tick",
    "load_csv_dataframe",
    "load_csv_data",
]


def round_to_tick(value: float, tick: Optional[float]) -> float:
    """采用四舍五入方式，将价格量化到指定最小变动价位。"""

    if not tick or tick <= 0:
        return value
    try:
        return float(Decimal(str(value)).quantize(Decimal(str(tick)), rounding=ROUND_HALF_UP))
    except Exception:
        return value


_round_to_tick = round_to_tick


def load_csv_dataframe(symbol: str, start: str, end: str, datadir: str = "data") -> pd.DataFrame:
    """向后兼容的工具函数，以 DataFrame 形式加载历史 K 线数据。"""

    prefer_stockdata = bool(int(os.environ.get("PREFER_STOCKDATA", "1")))
    adjust = os.environ.get("ADJUST_TYPE", "auto").lower()
    return _dl.load_price_dataframe(
        symbol,
        start,
        end,
        frequency="daily",
        adjust=adjust,
        prefer_stockdata=prefer_stockdata,
        data_root=datadir,
        stockdata_root=None,
    )


def load_csv_data(symbol: str, start: str, end: str, datadir: str = "data") -> bt.feeds.PandasData:
    """向后兼容的工具函数，以 Backtrader 数据源格式加载历史 K 线。"""

    prefer_stockdata = bool(int(os.environ.get("PREFER_STOCKDATA", "1")))
    adjust = os.environ.get("ADJUST_TYPE", "auto").lower()
    return _dl.load_bt_feed(
        symbol,
        start,
        end,
        frequency="daily",
        adjust=adjust,
        prefer_stockdata=prefer_stockdata,
        data_root=datadir,
        stockdata_root=None,
    )
