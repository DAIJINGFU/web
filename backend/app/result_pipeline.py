"""回测结果汇总模块：负责指标统计、曲线生成与交易记录整理。

整体流程说明：
1. ``compute_metrics_and_curves`` 根据策略执行的分析器结果，计算净值曲线、收益率、最大回撤、夏普等指标；
2. 若指定了基准指数，会同步拉取基准行情并对齐日期，计算超额收益与因子（Alpha/Beta）；
3. ``collect_trade_and_order_details`` 汇总自定义分析器捕获到的成交、订单与日志，便于最终返回给前端。

模块内部大量采用 ``jq_state`` 中的共享选项，确保与聚宽原有配置保持一致。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import math as _math
import backtrader as bt

from . import data_loader as _dl
from .models import OrderRecord, TradeRecord


def compute_metrics_and_curves(
    cerebro: bt.Cerebro,
    strat: bt.Strategy,
    jq_state: Dict[str, Any],
    start: str,
    end: str,
    cash: float,
    frequency: str,
    benchmark_symbol: Optional[str],
    symbols: Iterable[str],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """基于回测结果计算核心指标与净值曲线。

    返回六个对象：指标字典、策略净值、日收益率、日盈亏、基准净值与超额收益序列。所有日期字段都统一格式化为
    ``YYYY-MM-DD``，确保与前端展示组件兼容。
    """

    # Backtrader 分析器集合，后续所有指标均基于此提取
    analyzers = strat.analyzers
    sharpe_bt = analyzers.sharpe.get_analysis().get("sharperatio") if hasattr(analyzers, "sharpe") else None
    dd = analyzers.drawdown.get_analysis() if hasattr(analyzers, "drawdown") else {}
    returns = analyzers.returns.get_analysis() if hasattr(analyzers, "returns") else {}
    trade_ana = analyzers.trades.get_analysis() if hasattr(analyzers, "trades") else {}
    timereturn = analyzers.timereturn.get_analysis() if hasattr(analyzers, "timereturn") else {}

    # 权益与日收益
    equity_curve: List[Dict[str, Any]] = []
    daily_returns: List[Dict[str, Any]] = []
    daily_pnl: List[Dict[str, Any]] = []

    tmp_eq: List[Dict[str, Any]] = []
    cumulative = 1.0
    for dt, r in timereturn.items():
        cumulative *= (1 + r)
        tmp_eq.append({"date": dt.strftime("%Y-%m-%d"), "equity": cumulative})
    eq_filtered = [p for p in tmp_eq if p["date"] >= start]
    if eq_filtered:
        base = eq_filtered[0]["equity"] or 1.0
        equity_curve = [{"date": p["date"], "equity": (p["equity"] / base if base else p["equity"]) } for p in eq_filtered]
    daily_returns = [{"date": dt.strftime("%Y-%m-%d"), "ret": r} for dt, r in timereturn.items() if dt.strftime("%Y-%m-%d") >= start]
    try:
        prev_equity_val = float(cash)
        for dr in daily_returns:
            r = float(dr["ret"])
            pnl_amt = prev_equity_val * r
            eq_after = prev_equity_val + pnl_amt
            daily_pnl.append({"date": dr["date"], "pnl": pnl_amt, "equity": eq_after})
            prev_equity_val = eq_after
    except Exception:
        daily_pnl = []

    total_trades = trade_ana.get("total", {}).get("total", 0)
    won = trade_ana.get("won", {}).get("total", 0)
    lost = trade_ana.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades) if total_trades else 0

    data_variant = jq_state.get("options", {}).get("adjust_type")

    metrics: Dict[str, Any] = {
        "final_value": cerebro.broker.getvalue(),
        "pnl_pct": (cerebro.broker.getvalue() / cash - 1),
        "sharpe": None,
        "sharpe_bt": sharpe_bt,
        "max_drawdown": dd.get("max", {}).get("drawdown") if isinstance(dd, dict) else None,
        "max_drawdown_len": dd.get("max", {}).get("len") if isinstance(dd, dict) else None,
        "drawdown_pct": dd.get("drawdown") if isinstance(dd, dict) else None,
        "drawdown_len": dd.get("len") if isinstance(dd, dict) else None,
        "rt_annual": returns.get("rtannual") if isinstance(returns, dict) else None,
        "rnorm": returns.get("rnorm") if isinstance(returns, dict) else None,
        "rnorm100": returns.get("rnorm100") if isinstance(returns, dict) else None,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "won_trades": won,
        "lost_trades": lost,
        "symbols_used": list(symbols),
        "use_real_price": bool(jq_state.get("options", {}).get("use_real_price", False)),
        "jq_options": jq_state.get("options", {}),
        "data_source_preference": None,
        "data_sources_used": None,
        "adjust_type": None,
        "data_variant": data_variant,
    }

    benchmark_curve: List[Dict[str, Any]] = []
    excess_curve: List[Dict[str, Any]] = []

    if not benchmark_symbol:
        jq_bm_post = jq_state.get("benchmark")
        if jq_bm_post:
            benchmark_symbol = jq_bm_post
            metrics["benchmark_detect_phase"] = "post_run_initialize"
        else:
            benchmark_symbol = "000300"
            metrics["benchmark_detect_phase"] = "fallback_default"
            metrics["benchmark_fallback"] = True
    else:
        metrics["benchmark_detect_phase"] = "pre_run_explicit"

    strat_ret_map = {dt.strftime("%Y-%m-%d"): r for dt, r in timereturn.items()}
    aligned_sr: List[float] = []
    aligned_br: List[float] = []

    if benchmark_symbol:
        bench_warmup_start = (pd.to_datetime(start) - pd.Timedelta(days=10)).date().isoformat()
        bench_freq = frequency if frequency in ("daily", "weekly", "monthly") else "daily"
        try:
            bench_df = _dl.load_price_dataframe(
                benchmark_symbol.replace(".XSHG", "").replace(".XSHE", ""),
                bench_warmup_start,
                end,
                frequency=bench_freq,
                adjust="auto",
                prefer_stockdata=True,
            )
        except Exception:
            bench_df = pd.DataFrame()
        if len(bench_df) == 0:
            metrics["benchmark_final"] = None
            metrics["excess_return"] = None
            metrics["benchmark_missing_reason"] = "empty_file_or_no_rows_in_range"
        else:
            bench_df = bench_df[["datetime", "close"]].copy()
            bench_df["ret"] = bench_df["close"].pct_change().fillna(0.0)
            bench_df["equity"] = (1 + bench_df["ret"]).cumprod()
            base_idx = bench_df[bench_df["datetime"] <= pd.to_datetime(start)].index
            base_row = bench_df.loc[base_idx.max()] if len(base_idx) > 0 else None
            try:
                if base_row is not None:
                    base_equity = float(base_row["equity"]) if "equity" in base_row else None
                else:
                    base_equity = float(bench_df["equity"].iloc[0]) if len(bench_df) else None
            except Exception:
                base_equity = None
            if base_equity and base_equity != 0:
                bench_df["equity_rebased"] = bench_df["equity"] / base_equity
            else:
                bench_df["equity_rebased"] = bench_df["equity"]
            try:
                if base_row is not None:
                    jq_state.setdefault("log", []).append(
                        f"[benchmark_base] base_date={base_row['datetime'].date()} base_close={float(base_row['close']):.4f} start={start}"
                    )
                else:
                    jq_state.setdefault("log", []).append(f"[benchmark_base] no_base_before_start start={start}")
            except Exception:
                pass

            bm_map = {
                d.strftime("%Y-%m-%d"): (r, eq)
                for d, r, eq in zip(bench_df["datetime"], bench_df["ret"], bench_df["equity_rebased"])
                if d.strftime("%Y-%m-%d") >= start
            }
            for ec in equity_curve:
                d = ec["date"]
                if d in bm_map:
                    br, beq = bm_map[d]
                    benchmark_curve.append({"date": d, "equity": beq})
                    excess_curve.append({"date": d, "excess": ec["equity"] / beq - 1 if beq != 0 else 0})
                    sr = strat_ret_map.get(d)
                    if sr is not None:
                        aligned_sr.append(float(sr))
                        aligned_br.append(float(br))
            if benchmark_curve:
                metrics["benchmark_final"] = benchmark_curve[-1]["equity"]
                metrics["excess_return"] = equity_curve[-1]["equity"] / benchmark_curve[-1]["equity"] - 1
                metrics["benchmark_return"] = benchmark_curve[-1]["equity"] - 1
            else:
                metrics["benchmark_final"] = None
                metrics["excess_return"] = None
                metrics["benchmark_missing_reason"] = "no_overlap_with_strategy_dates"
    else:
        metrics["benchmark_final"] = None
        metrics["excess_return"] = None
        metrics["benchmark_return"] = None

    metrics["benchmark_symbol_used"] = benchmark_symbol
    metrics["benchmark_code"] = jq_state.get("benchmark")
    metrics.setdefault("benchmark_detect_phase", "none")
    try:
        metrics["data_source_preference"] = jq_state.get("options", {}).get("data_source_preference")
        metrics["data_sources_used"] = jq_state.get("symbol_file_map")
        metrics["adjust_type"] = jq_state.get("options", {}).get("adjust_type")
    except Exception:
        pass

    try:
        trading_days = 250.0
        td_opt = jq_state.get("options", {}).get("annualization_factor") or jq_state.get("options", {}).get("trading_days")
        if isinstance(td_opt, (int, float)) and td_opt > 0:
            trading_days = float(td_opt)
    except Exception:
        trading_days = 250.0
    try:
        rf_annual = 0.04
        rf_opt = jq_state.get("options", {}).get("risk_free_rate")
        if isinstance(rf_opt, (int, float)):
            rf_annual = float(rf_opt)
    except Exception:
        rf_annual = 0.04
    rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0

    sr_all = [float(r) for dt, r in timereturn.items() if dt.strftime("%Y-%m-%d") >= start]
    sharpe_exclude_first = bool(jq_state.get("options", {}).get("sharpe_exclude_first", False))
    sr_seq = sr_all[1:] if (sharpe_exclude_first and len(sr_all) > 1) else list(sr_all)
    ex_sr_all = [r - rf_daily for r in sr_seq]

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    sharpe_mean_excess: Optional[float] = None
    if len(ex_sr_all) > 1:
        m_ex = _mean(ex_sr_all)
        var_samp_ex = sum((x - m_ex) ** 2 for x in ex_sr_all) / (len(ex_sr_all) - 1)
        std_ex_samp = (var_samp_ex ** 0.5)
        if std_ex_samp > 0:
            sharpe_mean_excess = (_math.sqrt(trading_days) * m_ex / std_ex_samp)

    sharpe_jq: Optional[float] = None
    if sr_seq:
        cum = 1.0
        for r in sr_seq:
            cum *= (1.0 + r)
        n = float(len(sr_seq))
        r_annual_cagr = (cum ** (trading_days / n) - 1.0)
        if len(sr_seq) > 1:
            var_samp = sum((r - _mean(sr_seq)) ** 2 for r in sr_seq) / (len(sr_seq) - 1)
        else:
            var_samp = 0.0
        vol_annual = (_math.sqrt(var_samp) * _math.sqrt(trading_days)) if var_samp > 0 else 0.0
        if vol_annual > 0:
            sharpe_jq = (r_annual_cagr - rf_annual) / vol_annual

    sharpe_method = jq_state.get("options", {}).get("sharpe_method")
    sharpe_method = sharpe_method.lower().strip() if isinstance(sharpe_method, str) else "cagr"
    if sharpe_method == "cagr":
        metrics["sharpe"] = sharpe_jq if (sharpe_jq is not None) else sharpe_mean_excess
    else:
        metrics["sharpe"] = sharpe_mean_excess if (sharpe_mean_excess is not None) else sharpe_jq
    metrics["sharpe_method"] = sharpe_method
    try:
        jq_state.setdefault("log", []).append(
            f"[sharpe] method={sharpe_method} jq={sharpe_jq} mean_excess={sharpe_mean_excess} n={len(sr_seq)} exclude_first={sharpe_exclude_first}"
        )
    except Exception:
        pass

    if sr_all:
        total_cum = 1.0
        for r in sr_all:
            total_cum *= (1.0 + r)
        metrics["rt_annual"] = (total_cum ** (trading_days / len(sr_all)) - 1.0) if len(sr_all) > 0 else None
    try:
        jq_state.setdefault("log", []).append(f"[annualization] trading_days={int(trading_days)} rf_annual={rf_annual}")
    except Exception:
        pass

    def _cov(x: List[float], y: List[float]) -> float:
        n = min(len(x), len(y))
        if n <= 1:
            return 0.0
        mx = _mean(x)
        my = _mean(y)
        return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)

    if aligned_sr and aligned_br and len(aligned_sr) == len(aligned_br):
        Rs = list(map(float, aligned_sr))
        Rb = list(map(float, aligned_br))
        var_b = _cov(Rb, Rb)
        beta = (_cov(Rb, Rs) / var_b) if var_b > 0 else None
        n_align = float(len(Rs))
        cum_s = 1.0
        for r in Rs:
            cum_s *= (1.0 + r)
        cum_b = 1.0
        for r in Rb:
            cum_b *= (1.0 + r)
        rp_annual = (cum_s ** (trading_days / n_align) - 1.0) if n_align > 0 else None
        rm_annual = (cum_b ** (trading_days / n_align) - 1.0) if n_align > 0 else None
        alpha_annual = None
        if (beta is not None) and (rp_annual is not None) and (rm_annual is not None):
            alpha_annual = rp_annual - (rf_annual + beta * (rm_annual - rf_annual))
        alpha_daily = (alpha_annual / trading_days) if (alpha_annual is not None and trading_days > 0) else None
        alpha_unit = jq_state.get("options", {}).get("alpha_unit")
        alpha_unit = alpha_unit.lower() if isinstance(alpha_unit, str) and alpha_unit.lower() in ("daily", "annual") else "annual"
        metrics["beta"] = beta
        metrics["alpha_daily"] = alpha_daily
        metrics["alpha_annual"] = alpha_annual
        metrics["alpha"] = alpha_annual if alpha_unit == "annual" else alpha_daily
        metrics["alpha_unit"] = alpha_unit
        try:
            jq_state.setdefault("log", []).append(
                f"[alpha] model=annual_formula unit={alpha_unit} rp_annual={rp_annual} rm_annual={rm_annual} daily={alpha_daily} annual={alpha_annual} beta={beta} rf_annual={rf_annual} n_align={len(aligned_sr)}"
            )
        except Exception:
            pass
    else:
        metrics["beta"] = None
        metrics["alpha"] = None
        metrics["alpha_daily"] = None
        metrics["alpha_annual"] = None
        metrics["alpha_unit"] = jq_state.get("options", {}).get("alpha_unit", "annual")

    metrics["annualization_factor"] = int(trading_days)
    metrics["sharpe_rf_annual"] = rf_annual

    try:
        peak = -float("inf")
        peak_date = None
        min_dd = 0.0
        trough_date = None
        best_peak_date = None
        eps = 1e-12
        for p in (equity_curve or []):
            val = float(p.get("equity", 0.0) or 0.0)
            dt = p.get("date")
            if val > peak:
                peak = val
                peak_date = dt
            if peak > 0 and dt is not None:
                dd = val / peak - 1.0
                if (dd < min_dd - eps) or (abs(dd - min_dd) <= eps and (trough_date is None or (dt and dt > trough_date))):
                    min_dd = dd
                    trough_date = dt
                    best_peak_date = peak_date
        if trough_date and best_peak_date:
            metrics["drawdown_interval"] = f"{best_peak_date} ~ {trough_date}"
        else:
            metrics["drawdown_interval"] = None
    except Exception:
        metrics["drawdown_interval"] = None

    return metrics, equity_curve, daily_returns, daily_pnl, benchmark_curve, excess_curve


def collect_trade_and_order_details(
    strat: bt.Strategy,
    jq_state: Dict[str, Any],
    start: str,
) -> Tuple[
    List[TradeRecord],
    List[OrderRecord],
    List[OrderRecord],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[str],
]:
    """整理交易、订单、日志等明细数据。

    返回内容依次为：
    1. ``trades``：成交记录列表，与聚宽成交导出格式一致；
    2. ``orders``：按照日期与状态排序的订单记录；
    3. ``blocked_orders``：触发交易限制的订单（如涨跌停、T+1 等）；
    4. ``daily_turnover``：占位字段，后续可扩展为每日成交额统计；
    5. ``jq_records``：聚宽原生 records 兼容数据，目前未使用；
    6. ``jq_logs``：在回测过程中积累的日志列表。
    """

    trades: List[TradeRecord] = strat.analyzers.trade_capture.get_analysis() if hasattr(strat.analyzers, "trade_capture") else []
    orders: List[OrderRecord] = strat.analyzers.order_capture.get_analysis() if hasattr(strat.analyzers, "order_capture") else []

    try:
        blocked = jq_state.get("blocked_orders", [])
        if blocked:
            orders = list(blocked) + orders
    except Exception:
        pass

    try:
        orders = [o for o in orders if (o.datetime or "") >= start]
    except Exception:
        pass

    try:
        def _ord_key(o: OrderRecord):
            d = o.datetime or "9999-99-99"
            pri = 0 if (o.status or "").startswith("Blocked") else 1
            return (d, pri)
        orders.sort(key=_ord_key)
    except Exception:
        pass

    try:
        trades = [t for t in trades if (t.datetime or "") >= start and (t.size or 0) != 0]
    except Exception:
        pass

    try:
        def _trade_key(t: TradeRecord):
            d = t.close_datetime or t.datetime or "9999-99-99"
            return d
        trades.sort(key=_trade_key)
    except Exception:
        pass

    daily_turnover: List[Dict[str, Any]] = []
    jq_records = None
    try:
        jq_logs = list(jq_state.get("log", []))
    except Exception:
        jq_logs = []

    try:
        blocked_orders = list(jq_state.get("blocked_orders", []))
    except Exception:
        blocked_orders = []

    return trades, orders, blocked_orders, daily_turnover, jq_records, jq_logs
