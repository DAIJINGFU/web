"""回测引擎主入口：协调策略编译、数据加载、选项应用、执行与结果汇总。

整体流程分为以下阶段：
1. 调用 ``compile_user_strategy`` 执行用户脚本并识别策略类型；
2. 根据 ``jq_state`` 预解析 set_option 设置，确定数据加载与成交价模式；
3. 通过 ``prepare_data_sources`` 读取历史行情，并提前将原始文件路径、DataFrame 缓存写入状态字典；
4. 调用 ``apply_option_settings`` 配置佣金、滑点、公司行动等选项；
5. 注册分析器并运行策略，随后使用 ``result_pipeline``、``collect_trade_and_order_details`` 汇总结果。

该模块返回 ``BacktestResult`` 数据类，便于前端或 CLI 将回测指标、曲线与交易明细一次性取出。
"""

import io
import json
import os
import traceback
import math as _math
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import backtrader as bt
from .models import BacktestResult
from .market_controls import is_stock_paused
from .strategy_compiler import compile_user_strategy
from .data_pipeline import prepare_data_sources
from .option_handlers import apply_option_settings
from .execution_pipeline import register_default_analyzers, run_strategy
from .result_pipeline import compute_metrics_and_curves, collect_trade_and_order_details
from .trade_limits import setup_global_limit_guards

# -----------------------------
# 回测执行
# -----------------------------

def run_backtest(
    symbol: str,
    start: str,
    end: str,
    cash: float,
    strategy_code: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    benchmark_symbol: Optional[str] = None,
    frequency: str = 'daily',
    adjust_type: str = 'auto',
    datadir: str = 'data',
) -> BacktestResult:
    log_buffer = io.StringIO()
    try:
        # 1. 编译策略 & 初始化 Cerebro
        StrategyCls, jq_state = compile_user_strategy(strategy_code)
        # 记录用户开始日期
        jq_state['user_start'] = start
        # 预扫描用户代码：因为 initialize 在数据加载之后才执行，若用户在 initialize 内才 set_option('use_real_price',True)
        # 会错过“选 raw 文件”阶段，这里用静态正则提前检测一次并预置 option；同理支持 adjust_type。
        try:
            # 仅当尚未显式设置 use_real_price 时才尝试推断
            if 'use_real_price' not in jq_state.get('options', {}):
                # 更宽松的匹配：True / true / 1 / False / false / 0
                m_use = re.search(r"set_option\(\s*['\"]use_real_price['\"]\s*,\s*(True|False|true|false|1|0)\s*\)", strategy_code)
                if m_use:
                    raw_val = m_use.group(1)
                    val = True if raw_val in ('True','true','1') else False
                    jq_state['options']['use_real_price'] = val
                    jq_state['log'].append(f"[preparse] detected use_real_price={val} (token={raw_val}) in source code")
            # adjust_type 同理（只解析 raw/qfq/hfq/auto 简单字面值）
            if 'adjust_type' not in jq_state.get('options', {}):
                m_adj = re.search(r"set_option\(\s*['\"]adjust_type['\"]\s*,\s*['\"](raw|qfq|hfq|auto)['\"]", strategy_code, re.IGNORECASE)
                if m_adj:
                    jq_state['options']['adjust_type'] = m_adj.group(1).lower()
                    jq_state['log'].append(f"[preparse] detected adjust_type={m_adj.group(1).lower()} in source code")
        except Exception:
            pass
        # 若调用方传入 adjust_type / frequency 且用户未在策略 set_option 指定，则采用传入值
        if 'adjust_type' not in jq_state.get('options', {}):
            jq_state['options']['adjust_type'] = adjust_type
        jq_state['options']['api_frequency'] = frequency
        # 创建 cerebro（根据成交价类型决定 cheat_on_open）
        fill_price_opt = str(jq_state.get('options', {}).get('fill_price', 'open')).lower()
        try:
            cerebro = bt.Cerebro(cheat_on_open=(fill_price_opt == 'open'))
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open={fill_price_opt == 'open'}")
        except Exception:
            cerebro = bt.Cerebro()
            jq_state['log'].append(f"[exec_mode] fill_price={fill_price_opt} cheat_on_open=unsupported")
        cerebro.broker.setcash(cash)
        # 全局买卖限价拦截（支持 blocked_orders 记录）
        try:
            setup_global_limit_guards(jq_state)
        except Exception:
            pass

        # 2. 统一标的解析与数据加载
        symbols, benchmark_symbol = prepare_data_sources(
            cerebro=cerebro,
            jq_state=jq_state,
            symbol_input=symbol,
            start=start,
            end=end,
            frequency=frequency,
            adjust_type=adjust_type,
            strategy_code=strategy_code,
            benchmark_symbol=benchmark_symbol,
        )
        # 3. 处理聚宽 set_option 配置（佣金 / 滑点 / 公司行动等）
        strategy_params = strategy_params or {}
        apply_option_settings(cerebro, jq_state)

        # 4. 注册分析器并执行策略
        register_default_analyzers(cerebro, jq_state)
        strat, results = run_strategy(cerebro, StrategyCls, strategy_params)

        metrics, equity_curve, daily_returns, daily_pnl, benchmark_curve, excess_curve = compute_metrics_and_curves(
            cerebro=cerebro,
            strat=strat,
            jq_state=jq_state,
            start=start,
            end=end,
            cash=cash,
            frequency=frequency,
            benchmark_symbol=benchmark_symbol,
            symbols=symbols,
        )

        trades, orders, blocked_orders, daily_turnover, jq_records, jq_logs = collect_trade_and_order_details(
            strat=strat,
            jq_state=jq_state,
            start=start,
        )
        
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            daily_pnl=daily_pnl,
            daily_turnover=daily_turnover,
            benchmark_curve=benchmark_curve,
            excess_curve=excess_curve,
            trades=trades,
            orders=orders,
            blocked_orders=blocked_orders,
            log=log_buffer.getvalue(),
            jq_records=jq_records,
            jq_logs=jq_logs,
        )
    except Exception:
        tb = traceback.format_exc()
        return BacktestResult(
            metrics={'error': True},
            equity_curve=[],
            daily_returns=[],
            daily_pnl=[],
            daily_turnover=[],
            benchmark_curve=[],
            excess_curve=[],
            trades=[],
            blocked_orders=[],
            log=tb,
            jq_records=None,
            jq_logs=None,
        )


if __name__ == '__main__':
    # 简单自测
    code = """\nimport backtrader as bt\nclass UserStrategy(bt.Strategy):\n    def next(self):\n        if not self.position: self.buy(size=10)\n        elif len(self) > 5: self.sell()\n"""
    # 注意: 需要先准备 data/sample.csv
    res = run_backtest('sample', '2025-01-01', '2025-03-01', 100000, code, benchmark_symbol='sample')
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
