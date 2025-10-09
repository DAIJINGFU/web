"""选项配置模块：集中处理佣金、滑点与公司行动等聚宽兼容设置。

针对回测参数的常见需求（费用体系、滑点方案、公司行动模拟等），本模块给出与聚宽平台一致的默认值，并允许通过
``jq_state['options']`` 进行覆盖。调用顺序通常位于数据加载之后、策略运行之前。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .corporate_actions import CorporateActionEvent, load_corporate_actions


def apply_option_settings(cerebro, jq_state: Dict[str, Any]) -> Tuple[bool, str]:
    """根据 ``jq_state`` 中的选项配置经纪商、费用与公司行动。

    返回值：``(use_real_price, fill_price_mode)``，供回测主流程后续使用。``use_real_price`` 控制数据加载阶段是否优先
    使用未经复权的报价，``fill_price_mode`` 则决定订单成交价以开盘价还是收盘价计算。
    """

    options = jq_state.setdefault("options", {})

    commission = options.get("commission")
    if commission is None:
        commission = 0.0003
        options["commission"] = commission
    jq_state.setdefault("log", []).append(f"[commission_setup] rate={commission} model=unified")

    try:
        cerebro.broker.setcommission(commission=commission)
    except Exception:
        jq_state["log"].append("[commission_setup_warning] setcommission_failed")

    cfg = jq_state.setdefault("fee_config", {})
    cfg["rate"] = commission

    # 佣金及印花税限制：默认为券商常用配置，可由用户覆盖
    min_comm = 5.0
    user_min = options.get("min_commission")
    if isinstance(user_min, (int, float)) and user_min >= 0:
        min_comm = float(user_min)

    stamp_duty = 0.001
    user_sd = options.get("stamp_duty")
    if isinstance(user_sd, (int, float)) and user_sd >= 0:
        stamp_duty = float(user_sd)

    cfg["min_commission"] = min_comm
    cfg["stamp_duty"] = stamp_duty
    jq_state["log"].append(f"[fee_config] min_commission={min_comm} stamp_duty={stamp_duty}")

    slippage_perc = options.get("slippage_perc")
    if slippage_perc is None and "fixed_slippage" not in options:
        slippage_perc = 0.00246
        options["slippage_perc"] = slippage_perc
        jq_state["log"].append(f"[slippage_default] perc={slippage_perc}")

    jq_state["log"].append(f"[slippage_mode] scheme=half_slip half={(slippage_perc or 0)/2}")

    jq_state["corporate_actions"] = []
    try:
        if bool(options.get("simulate_corporate_actions", False)):
            symbol = jq_state.get("symbol") or jq_state.get("security_code") or jq_state.get("raw_symbol")
            if symbol:
                jq_state["corporate_actions"] = load_corporate_actions(
                    symbol,
                    jq_state.get("data_dir", "data"),
                    logger=jq_state.get("logger"),
                )

        manual = options.get("manual_corporate_actions")
        if isinstance(manual, list) and manual:
            added = 0
            for item in manual:
                if not isinstance(item, dict):
                    continue
                date = str(item.get("date") or "").strip()
                act_type = str(item.get("type") or "").strip().upper()
                if not date or not act_type:
                    continue
                ratio = item.get("ratio")
                cash = item.get("cash")
                shares = item.get("shares")
                event = CorporateActionEvent(
                    date=date,
                    action_type=act_type,
                    ratio=(float(ratio) if ratio not in (None, "") else None),
                    cash=(float(cash) if cash not in (None, "") else None),
                    note="manual",
                )
                if shares not in (None, ""):
                    try:
                        event._manual_shares = int(shares)
                    except Exception:
                        pass
                jq_state["corporate_actions"].append(event)
                jq_state["log"].append(
                    f"[ca_manual_merge] date={date} type={act_type} shares={shares} ratio={ratio}"
                )
                added += 1
            if added:
                jq_state["log"].append(f"[ca_manual_load] count={added}")

        jq_state["corporate_actions"].sort(key=lambda evt: evt.date)
    except Exception as err:
        jq_state["corporate_actions"] = []
        jq_state["log"].append(f"[ca_error] {type(err).__name__}:{err}")

    use_real_price = bool(options.get("use_real_price", False))
    fill_price = str(options.get("fill_price", "open")).lower()

    try:
        if fill_price == "close" and hasattr(cerebro.broker, "set_coc"):
            cerebro.broker.set_coc(True)
        elif fill_price != "close" and hasattr(cerebro.broker, "set_coo"):
            cerebro.broker.set_coo(True)
    except Exception:
        jq_state["log"].append("[fill_price_warning] broker_adjustment_failed")

    return use_real_price, fill_price
