"""聚宽适配层复用的 Backtrader 分析器工具模块。"""
from __future__ import annotations

from typing import List

import backtrader as bt

from .models import OrderRecord, TradeRecord

__all__ = [
    "TradeCapture",
    "create_order_capture_analyzer",
]


class TradeCapture(bt.Analyzer):
    """同时记录已平仓与未平仓交易明细的分析器。"""

    def start(self):  # pragma: no cover - lifecycle method
        self.records: List[TradeRecord] = []
        self._open_cache: dict[int, TradeRecord] = {}

    def _fmt_date(self, num_dt):
        try:
            return bt.num2date(num_dt).date().isoformat() if num_dt else None
        except Exception:  # pragma: no cover - defensive
            return None

    def notify_trade(self, trade):  # pragma: no cover - runtime callback
        if trade.isopen and not trade.isclosed:
            tid = id(trade)
            if tid not in self._open_cache:
                open_dt = self._fmt_date(getattr(trade, "dtopen", None))
                entry_price = getattr(trade, "price", None) or getattr(trade, "openprice", None)
                size_raw = getattr(trade, "size", 0)
                size_hist = 0
                hist = getattr(trade, "history", None)
                if hist:
                    try:
                        size_hist = abs(hist[0].event.size)
                    except Exception:
                        size_hist = 0
                size = abs(size_raw) or size_hist
                side = "LONG" if size_raw > 0 else "SHORT"
                self._open_cache[tid] = TradeRecord(
                    datetime=open_dt or "",
                    side=side,
                    size=size,
                    price=entry_price or 0.0,
                    value=(entry_price * size) if entry_price else 0.0,
                    commission=getattr(trade, "commission", 0.0),
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

        if not trade.isclosed:
            return

        open_dt = self._fmt_date(getattr(trade, "dtopen", None))
        close_dt = self._fmt_date(getattr(trade, "dtclose", None))
        entry_price = None
        exit_price = None
        entry_size = 0
        hist = getattr(trade, "history", None)
        try:
            if hist:
                first_ev = hist[0].event
                last_ev = hist[-1].event
                entry_price = getattr(first_ev, "price", None)
                exit_price = getattr(last_ev, "price", None)
                entry_size = getattr(first_ev, "size", 0)
        except Exception:
            pass
        size = abs(entry_size)
        side = "LONG" if entry_size > 0 else "SHORT"
        entry_value = (entry_price * size) if (entry_price is not None) else None
        exit_value = (exit_price * size) if (exit_price is not None) else None
        pnl = getattr(trade, "pnl", None)
        pnl_comm = getattr(trade, "pnlcomm", None)
        comm = getattr(trade, "commission", 0.0)
        rec = TradeRecord(
            datetime=close_dt or open_dt or "",
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

    def stop(self):  # pragma: no cover - lifecycle method
        for rec in self._open_cache.values():
            self.records.append(rec)
        self._open_cache.clear()

    def get_analysis(self):  # pragma: no cover - API
        return self.records


def create_order_capture_analyzer(jq_state):
    """创建与 jq_state 状态字典绑定的订单级分析器工厂函数。"""

    class OrderCapture(bt.Analyzer):
        """订单捕获分析器：同步记录真实成交与被拦截的订单。"""

        def start(self):  # pragma: no cover - lifecycle method
            self.records: List[OrderRecord] = []

        def _fmt_date(self, dt):
            try:
                return bt.num2date(dt).date().isoformat() if dt else None
            except Exception:
                return None

        def notify_order(self, order):  # pragma: no cover - runtime callback
            if order.status not in [order.Completed, order.Canceled, order.Rejected, order.Margin]:
                return

            try:
                dt = self._fmt_date(order.executed.dt)
            except Exception:
                dt = None

            size = getattr(order.executed, "size", 0.0)
            price = getattr(order.executed, "price", 0.0)

            if order.status == order.Completed and size > 0:
                current_date = dt or jq_state.get("current_dt", "").split(" ")[0]
                daily_bought = jq_state.setdefault("_daily_bought", {})
                daily_bought[current_date] = daily_bought.get(current_date, 0) + abs(size)
                jq_state["log"].append(
                    f"[T+1_track] {current_date} bought {abs(size)} shares, total today: {daily_bought[current_date]}"
                )

            orig_value = getattr(order.executed, "value", 0.0)
            value = size * price
            comm = getattr(order.executed, "comm", 0.0)
            side = "BUY" if size >= 0 else "SELL"
            try:
                data = order.data
                name = getattr(data, "_name", None)
            except Exception:
                name = None

            try:
                fee_cfg = jq_state.get("fee_config", {})
                rate = float(fee_cfg.get("rate", 0.0))
                min_comm = float(fee_cfg.get("min_commission", 0.0))
                stamp_duty = float(fee_cfg.get("stamp_duty", 0.0)) if side == "SELL" else 0.0
                if order.status == order.Margin and abs(size) < 1e-9:
                    comm = 0.0
                    jq_state["log"].append(
                        f"[order_margin] status=Margin size=0 skip_fee orig_comm={getattr(order.executed,'comm',0.0):.4f}"
                    )
                elif value is not None:
                    gross = abs(value)
                    raw_comm = gross * rate
                    adj_comm = raw_comm if min_comm <= 0 or raw_comm >= min_comm else min_comm
                    total_fee = adj_comm + gross * stamp_duty
                    if abs(total_fee - comm) > 1e-9:
                        try:
                            order.executed.comm = total_fee
                        except Exception:
                            pass
                        comm = total_fee
                    try:
                        exec_prefix = f"{dt} 09:30:00 - INFO - " if dt else ""
                    except Exception:
                        exec_prefix = ""
                    jq_state["log"].append(
                        f"{exec_prefix}[fee] {side} {name} value={value:.2f} price={price:.4f} "
                        f"base_comm={raw_comm:.4f} adj_comm={adj_comm:.4f} stamp_duty={(abs(value)*stamp_duty):.4f} final_comm={comm:.4f}"
                    )
                    if orig_value is not None and abs(orig_value - value) > 1e-6:
                        jq_state["log"].append(
                            f"[value_fix] {side} {name} orig_value={orig_value:.2f} recalculated={value:.2f} size={size} price={price:.4f}"
                        )
            except Exception:
                pass

            try:
                exec_date = dt
                line = f"[fill] {side} {name} size={abs(size)} price={price} value={value} commission={comm}"
                if exec_date:
                    jq_state["log"].append(f"{exec_date} 09:30:00 - INFO - {line}")
                else:
                    jq_state["log"].append(line)
            except Exception:
                pass

            self.records.append(
                OrderRecord(
                    datetime=dt or "",
                    symbol=name,
                    side=side,
                    size=size,
                    price=price,
                    value=value,
                    commission=comm,
                    status=order.getstatusname(),
                )
            )

        def get_analysis(self):  # pragma: no cover - API
            return self.records

    return OrderCapture
