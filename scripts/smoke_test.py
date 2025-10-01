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
    "    set_benchmark('000300')\n"
    "    set_option('risk_free_rate', 0.03)\n"
    "    set_option('annualization_factor', 250)\n"
    "    set_option('sharpe_method', 'mean')\n"
    "def handle_data(context, data):\n"
    "    pos = context.portfolio.positions.get('000008')\n"
    "    if pos is None or getattr(pos, 'closeable_amount', 0) == 0:\n"
    "        order_value('000008', 10000)\n"
)

import traceback
try:
    res = run_backtest(
    symbol='000008',
        start='2018-01-01',
        end='2020-01-01',
        cash=100000,
        strategy_code=strategy_code,
        strategy_params=None,
    benchmark_symbol='000300',
    )
    data = asdict(res)
    if data['metrics'].get('error'):
        print('[SMOKE_ERROR] metrics.error True')
        # 打印回溯
        print('TRACEBACK:\n', data.get('log','')[:4000])
        # 打印最后 30 行 jq_logs 方便定位
        logs = data.get('jq_logs') or []
        tail = '\n'.join(logs[-30:])
        print('JQ_LOGS_TAIL:\n', tail)
    else:
        print(json.dumps(data['metrics'], ensure_ascii=False, indent=2))
except Exception as e:
    print('{"smoke_error": true, "type": "%s", "msg": "%s"}' % (type(e).__name__, str(e)))
    traceback.print_exc()
