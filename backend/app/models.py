"""核心数据模型：描述回测结果及审计记录在各环节中的结构。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


__all__ = [
    "TradeRecord",
    "OrderRecord",
    "BacktestResult",
]


@dataclass
class TradeRecord:
    """表示一笔成交记录（可为已平仓或仍持仓状态）。"""

    datetime: str
    side: str
    size: float
    price: float
    value: float
    commission: float
    open_datetime: Optional[str] = None
    close_datetime: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    entry_value: Optional[float] = None
    exit_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_comm: Optional[float] = None


@dataclass
class OrderRecord:
    """表示一次订单执行结果或被拦截的订单。"""

    datetime: str
    symbol: Optional[str]
    side: str
    size: float
    price: float
    value: float
    commission: float
    status: str


@dataclass
class BacktestResult:
    """``run_backtest`` 返回的汇总结果结构体。"""

    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]
    daily_returns: List[Dict[str, Any]]
    daily_pnl: List[Dict[str, Any]]
    daily_turnover: List[Dict[str, Any]]
    benchmark_curve: List[Dict[str, Any]]
    excess_curve: List[Dict[str, Any]]
    trades: List[TradeRecord]
    log: str
    orders: Optional[List[OrderRecord]] = None
    blocked_orders: Optional[List[OrderRecord]] = None
    jq_records: Optional[List[Dict[str, Any]]] = None
    jq_logs: Optional[List[str]] = None

    @property
    def final_value(self) -> float:
        """便捷属性：获取最终权益。"""

        return float(self.metrics.get("final_value", 0.0))

    @property
    def total_return(self) -> float:
        """便捷属性：获取总收益率。"""

        return float(self.metrics.get("total_return", 0.0))
