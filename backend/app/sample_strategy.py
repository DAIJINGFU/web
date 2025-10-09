"""示例策略：使用快慢均线交叉演示回测流程。"""

import backtrader as bt

class UserStrategy(bt.Strategy):
    """简易策略：当快线穿越慢线向上时买入，反向穿越时平仓。"""

    params = dict(fast=5, slow=20)
    def __init__(self):
        # 预先计算两条均线，减少 next 周期中的重复创建开销
        self.sma_fast = bt.ind.SMA(self.data, period=self.p.fast)
        self.sma_slow = bt.ind.SMA(self.data, period=self.p.slow)
    def next(self):
        if not self.position and self.sma_fast[0] > self.sma_slow[0]:
            self.buy()
        elif self.position and self.sma_fast[0] < self.sma_slow[0]:
            self.close()
