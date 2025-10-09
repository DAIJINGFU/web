"""回测执行模块：注册默认分析器并负责运行策略实例。

该模块主要承担两项职责：
1. 统一向 Cerebro 注入一组基础分析器，使得回测输出包含夏普比、最大回撤、交易概览等常见指标，同时挂载自定义的
    ``TradeCapture`` 与订单捕获器，便于复刻聚宽平台的成交记录；
2. 负责将策略类与参数添加到 Cerebro 并触发运行，返回实际使用的策略实例对象，为后续结果抽取与统计提供入口。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import backtrader as bt

from .analyzers import TradeCapture, create_order_capture_analyzer


def register_default_analyzers(cerebro: bt.Cerebro, jq_state: Dict[str, Any]) -> None:
    """向 Cerebro 注册一组默认分析器，便于聚宽风格的回测指标统计。

    参数 ``jq_state`` 主要用于构造订单捕获器，因为其中会存放共享的订单列表与日志缓冲区。分析器命名采用
    ``_name`` 形式，便于 ``results_pipeline`` 在后续统一读取。
    """

    # 夏普、回撤、交易汇总等标准指标，保证基础风险收益分析可用
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    # 自定义交易/订单捕获器（含中文日志记录），用于收集成交明细与订单生命周期
    cerebro.addanalyzer(TradeCapture, _name="trade_capture")
    cerebro.addanalyzer(create_order_capture_analyzer(jq_state), _name="order_capture")


def run_strategy(
    cerebro: bt.Cerebro,
    strategy_cls: type,
    strategy_params: Dict[str, Any],
) -> Tuple[bt.Strategy, Iterable[bt.Strategy]]:
    """将策略添加进 Cerebro 并执行，返回主策略实例与原始结果列表。

    返回的 ``strategy`` 是回测过程中实际运行的策略对象，便于提取内部自定义变量；``results`` 则保留 Backtrader
    原生的策略列表结构（通常只包含一个元素），以保持最大兼容性。
    """

    # 添加策略，运行并返回首个策略实例（当前仅支持单策略）
    cerebro.addstrategy(strategy_cls, **strategy_params)
    results = cerebro.run()
    strategy = results[0]
    return strategy, results
