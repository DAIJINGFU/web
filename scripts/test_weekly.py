import os, sys, json
from dataclasses import asdict
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.app.backtest_engine import run_backtest

code = """import backtrader as bt\nclass UserStrategy(bt.Strategy):\n    def next(self):\n        if not self.position: self.buy(size=100)\n"""

if __name__ == '__main__':
    res = run_backtest('000008', '2019-01-01', '2020-01-01', 100000, code, benchmark_symbol='000300', frequency='weekly', adjust_type='auto')
    d = asdict(res)
    print('[weekly] metrics final_value=', d['metrics'].get('final_value'))
    print('[weekly] equity points=', len(d['equity_curve']))
    if d['metrics'].get('error'):
        print('[weekly] ERROR')
        print(d['log'][:1000])
