# ---
# name: MA5
# symbol: 000514.XSHG
# start: 2023-01-01
# end: 2024-01-01
# cash: 100000
# benchmark: 000300.XSHG
# frequency: daily
# adjust: auto
# ---
"""5日均线穿越策略示例 (聚宽风格).
保留原 initialize/handle_data 以兼容动态加载；同时暴露 strategy_code 给批量回归脚本使用。"""

# 回归脚本优先读取 strategy_code；也允许直接执行本文件（聚宽兼容模式）
strategy_code = r"""
def initialize(context):
    g.security = '000514.XSHG'
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

def handle_data(context, data):
    security = g.security
    close_data = attribute_history(security, 5, '1d', ['close'])
    MA5 = close_data['close'].mean()
    current_price = close_data['close'][-1]
    cash = context.portfolio.available_cash
    log.info(f"[MA5] cur={current_price} ma5={MA5} cash={cash}")
    if (current_price > 1.05*MA5) and (cash>0):
        order_value(security, cash)
        log.info(f"[MA5] Buying {security}")
    elif current_price < 0.95*MA5 and context.portfolio.positions[security].closeable_amount > 0:
        order_target(security, 0)
        log.info(f"[MA5] Selling {security}")
    record(stock_price=current_price, ma5=MA5)
"""

# 兼容：若用户直接 import 本文件并希望使用 initialize/handle_data
def initialize(context):
    g.security = '000514.XSHG'
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

def handle_data(context, data):
    security = g.security
    close_data = attribute_history(security, 5, '1d', ['close'])
    MA5 = close_data['close'].mean()
    current_price = close_data['close'][-1]
    cash = context.portfolio.available_cash
    if (current_price > 1.05*MA5) and (cash>0):
        order_value(security, cash)
    elif current_price < 0.95*MA5 and context.portfolio.positions[security].closeable_amount > 0:
        order_target(security, 0)
    record(stock_price=current_price, ma5=MA5)
    