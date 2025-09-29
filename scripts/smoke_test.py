import json
import os
import sys
from dataclasses import asdict

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.backtest_engine import run_backtest

# 极简策略：首次买入，随后持有
strategy_code = (
    "import backtrader as bt\n"
    "def initialize(context):\n"
    "    set_benchmark('000300.XSHG')\n"
    "    set_option('risk_free_rate', 0.03)\n"
    "    set_option('annualization_factor', 250)\n"
    "    set_option('sharpe_method', 'mean')\n"
    "def handle_data(context, data):\n"
    "    pos = context.portfolio.positions.get('600008.XSHG')\n"
    "    if pos is None or getattr(pos, 'closeable_amount', 0) == 0:\n"
    "        order_value('600008.XSHG', 10000)\n"
)

res = run_backtest(
    symbol='600008.XSHG',
    start='2018-01-01',
    end='2020-01-01',
    cash=100000,
    strategy_code=strategy_code,
    strategy_params=None,
    benchmark_symbol='000300.XSHG',
)
print(json.dumps(asdict(res)['metrics'], ensure_ascii=False, indent=2))
