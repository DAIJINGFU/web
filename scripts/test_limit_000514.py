import json
import os
import sys
from dataclasses import asdict

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.backtest_engine import run_backtest

# 目标: 验证 000514 在 2023-08-28 开盘价是否触发涨停板拦截 (上一交易日 2023-08-25 收盘价 * 1.10)
# 策略: 回测开始日设为 2023-08-28, 首日尝试按金额买入, 预期被 [limit_check] 或 [limit_check_global] 阻断, 不应生成成交订单。

strategy_code = (
    "def initialize(context):\n"
    "    set_benchmark('000300.XSHG')\n"
    "    # 保持默认 enable_limit_check=True\n"
    "    set_option('annualization_factor', 250)\n"
    "    set_option('sharpe_method', 'mean')\n"
    "def handle_data(context, data):\n"
    "    # 每日第一次如果尚无持仓尝试买入 10000 元\n"
    "    pos = context.portfolio.positions.get('000514.XSHE')\n"
    "    if pos is None or getattr(pos, 'closeable_amount', 0) == 0:\n"
    "        order_value('000514.XSHE', 10000)\n"
)

res = run_backtest(
    symbol='000514.XSHE',
    start='2023-08-28',
    end='2023-08-30',
    cash=100000,
    strategy_code=strategy_code,
    strategy_params=None,
    benchmark_symbol='000300.XSHG',
)

result = asdict(res)
metrics = result['metrics']
orders = result.get('orders') or []
logs = result.get('jq_logs') or []

# 抽取与涨停/跌停相关的日志行
limit_lines = [l for l in logs if '[limit_check' in l]
fill_lines = [l for l in logs if '[fill]' in l]

print('--- Metrics ---')
print(json.dumps({k: metrics[k] for k in list(metrics.keys())[:15]}, ensure_ascii=False, indent=2))
print('\n--- Limit Check Lines ---')
if limit_lines:
    for ln in limit_lines:
        print(ln)
else:
    print('(NO LIMIT LINES FOUND)')
print('\n--- Fill Lines (should be empty if blocked) ---')
if fill_lines:
    for ln in fill_lines:
        print(ln)
else:
    print('(NO FILLS)')
print('\n--- Orders Captured ---')
if orders:
    for od in orders:
        print(json.dumps(od, ensure_ascii=False))
else:
    print('(NO ORDERS)')
