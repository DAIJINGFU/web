import backtrader as bt

class UserStrategy(bt.Strategy):
    params = dict(fast=5, slow=20)
    def __init__(self):
        self.sma_fast = bt.ind.SMA(self.data, period=self.p.fast)
        self.sma_slow = bt.ind.SMA(self.data, period=self.p.slow)
    def next(self):
        if not self.position and self.sma_fast[0] > self.sma_slow[0]:
            self.buy()
        elif self.position and self.sma_fast[0] < self.sma_slow[0]:
            self.close()
