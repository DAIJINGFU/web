"""交易限制模块：集中处理涨跌停、停牌、T+1 等全局拦截逻辑。

核心思路：在 Backtrader ``Strategy`` 对象层面进行 monkeypatch，替换其 ``buy``/``sell`` 方法，实现统一的风控拦截。
拦截条件涵盖：
1. 交易日被标记为停牌；
2. 当前价格触及估算的涨跌停（结合 ``limit_up/down_factor`` 与价格最小变动单位）；
3. 当日买入的仓位无法在同日卖出（T+1 规则）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import backtrader as bt

from .data_compat import round_to_tick as _round_to_tick
from .market_controls import is_stock_paused
from .models import OrderRecord


def setup_global_limit_guards(jq_state: Dict[str, Any]) -> Optional[Callable[[], None]]:
    """为 ``bt.Strategy`` 安装全局买卖拦截器。

    返回一个可选的恢复函数，用于在需要时撤销 monkeypatch；若未启用或条件不满足则返回 ``None``。恢复函数只会在
    调用方显式执行时生效，默认情况下拦截器在整个回测周期内保持开启。
    """

    options = jq_state.setdefault("options", {})
    enable_limit = bool(options.get("enable_limit_check", True))
    if not enable_limit:
        return None

    up_factor = float(options.get("limit_up_factor", 1.10))
    down_factor = float(options.get("limit_down_factor", 0.90))
    price_tick = float(options.get("price_tick", 0.01) or 0.01)

    if jq_state.get("_global_limit_wrapped"):
        return None

    jq_state["_global_limit_wrapped"] = True

    original_buy = bt.Strategy.buy
    original_sell = bt.Strategy.sell

    def _current_price(data) -> float:
        fill_price = str(options.get("fill_price", "open")).lower()
        if fill_price == "close":
            return float(getattr(data, "close")[0])
        if hasattr(data, "open"):
            return float(getattr(data, "open")[0])
        return float(getattr(data, "close")[0])

    def _previous_close(data) -> Optional[float]:
        try:
            return float(getattr(data, "close")[-1])
        except Exception:
            return None

    def _record_block(side: str, status: str, jq_state: Dict[str, Any], price: float, size: float = 0.0) -> None:
        """记录被拦截的订单，方便最终结果返回给用户。

        函数会在 ``jq_state['blocked_orders']`` 中追加一条 ``OrderRecord``，并附带最基本的成交信息（方向、数量、价格）
        与拦截原因。
        """

        jq_state.setdefault("blocked_orders", [])
        jq_state.setdefault("log", [])
        jq_state["blocked_orders"].append(
            OrderRecord(
                datetime=jq_state.get("current_dt", "").split(" ")[0],
                symbol=getattr(jq_state.get("g"), "security", None) or "data0",
                side=side,
                size=size,
                price=price,
                value=0.0,
                commission=0.0,
                status=status,
            )
        )

    def _limit_guard(strategy_self, *args, **kwargs):
        try:
            if jq_state.get("in_warmup"):
                return None

            data = strategy_self.data
            current_date = jq_state.get("current_dt", "").split(" ")[0]
            stock_code = getattr(jq_state.get("g"), "security", None) or "data0"

            if is_stock_paused(stock_code, current_date, jq_state):
                jq_state.setdefault("log", []).append(f"[pause_check] BLOCK BUY {stock_code} paused on {current_date}")
                _record_block("BUY", "BlockedPaused", jq_state, 0.0)
                return None

            prev_close = _previous_close(data)
            cur_price = _current_price(data)
            if prev_close and prev_close > 0:
                up_limit = _round_to_tick(prev_close * up_factor, price_tick)
                if cur_price >= up_limit - 1e-9:
                    jq_state.setdefault("log", []).append(
                        f"[limit_check_global] BLOCK BUY cur={cur_price:.4f} up={up_limit:.4f} prev_close={prev_close:.4f}"
                    )
                    _record_block("BUY", "BlockedLimitUp", jq_state, cur_price)
                    return None
        except Exception:
            pass
        return original_buy(strategy_self, *args, **kwargs)

    def _limit_guard_sell(strategy_self, *args, **kwargs):
        try:
            if jq_state.get("in_warmup"):
                return None

            data = strategy_self.data
            current_date = jq_state.get("current_dt", "").split(" ")[0]
            stock_code = getattr(jq_state.get("g"), "security", None) or "data0"

            if is_stock_paused(stock_code, current_date, jq_state):
                jq_state.setdefault("log", []).append(f"[pause_check] BLOCK SELL {stock_code} paused on {current_date}")
                _record_block("SELL", "BlockedPaused", jq_state, 0.0)
                return None

            total_position = int(strategy_self.position.size)
            today_bought = jq_state.get("_daily_bought", {}).get(current_date, 0)
            closeable = max(0, total_position - today_bought)

            size = kwargs.get("size") if "size" in kwargs else (args[0] if len(args) > 0 else None)
            if size is None:
                size = total_position

            if size > closeable:
                jq_state.setdefault("log", []).append(
                    f"[T+1_check] BLOCK SELL {stock_code} size={size} > closeable={closeable} (total={total_position}, today_bought={today_bought})"
                )
                _record_block("SELL", "BlockedT+1", jq_state, 0.0, float(size))
                return None

            prev_close = _previous_close(data)
            cur_price = _current_price(data)
            if prev_close and prev_close > 0:
                down_limit = _round_to_tick(prev_close * down_factor, price_tick)
                if cur_price <= down_limit + 1e-9:
                    jq_state.setdefault("log", []).append(
                        f"[limit_check_global] BLOCK SELL cur={cur_price:.4f} down={down_limit:.4f} prev_close={prev_close:.4f}"
                    )
                    _record_block("SELL", "BlockedLimitDown", jq_state, cur_price)
                    return None
        except Exception:
            pass
        return original_sell(strategy_self, *args, **kwargs)

    bt.Strategy.buy = _limit_guard  # type: ignore
    bt.Strategy.sell = _limit_guard_sell  # type: ignore

    jq_state.setdefault("log", []).append("[limit_check_global] monkeypatch buy/sell installed")

    def _restore():
        bt.Strategy.buy = original_buy  # type: ignore
        bt.Strategy.sell = original_sell  # type: ignore
        jq_state["_global_limit_wrapped"] = False

    return _restore
