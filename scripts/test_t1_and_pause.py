"""
测试T+1制度和停牌检测功能

本脚本验证：
1. T+1制度：当日买入的股票不能当日卖出
2. 停牌检测：停牌期间不能交易
3. closeable_amount 正确计算
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.backtest_engine import run_backtest


# 测试1: T+1制度测试
def test_t_plus_1():
    """测试当日买入不能当日卖出"""
    print("\n" + "="*60)
    print("测试1: T+1制度")
    print("="*60)
    
    strategy_code = """
def initialize(context):
    g.security = '000001.SZ'
    g.test_phase = 0
    
def handle_data(context, data):
    stock = g.security
    
    # 第一天：买入100股
    if context.current_dt.date().isoformat() == '2024-01-02':
        print(f"\\n[Day 1] {context.current_dt.date()}")
        order(stock, 100)
        print(f"买入 100 股")
        
    # 第一天晚些时候：尝试卖出（应该被T+1阻止）
    elif context.current_dt.date().isoformat() == '2024-01-02':
        pos = context.portfolio.positions.get(stock)
        if pos:
            print(f"持仓总数: {pos.total_amount}")
            print(f"可卖数量: {pos.closeable_amount}")
            if pos.closeable_amount > 0:
                order(stock, -50)  # 尝试卖出50股
                print(f"尝试卖出 50 股（应该被阻止）")
            else:
                print("可卖数量为0，T+1限制生效！✓")
    
    # 第二天：应该可以卖出
    elif context.current_dt.date().isoformat() == '2024-01-03':
        print(f"\\n[Day 2] {context.current_dt.date()}")
        pos = context.portfolio.positions.get(stock)
        if pos:
            print(f"持仓总数: {pos.total_amount}")
            print(f"可卖数量: {pos.closeable_amount}")
            if pos.closeable_amount > 0:
                order(stock, -50)
                print(f"卖出 50 股（应该成功）✓")
"""
    
    result = run_backtest(
        symbol='000001.SZ',
        start='2024-01-02',
        end='2024-01-05',
        cash=100000,
        strategy_code=strategy_code,
        frequency='daily'
    )
    
    print("\n回测结果:")
    print(f"最终净值: {result.final_value:.2f}")
    print(f"总收益率: {result.total_return:.2%}")
    
    # 检查是否有被T+1阻止的订单
    t_plus_1_blocks = [o for o in result.blocked_orders if o.status == 'BlockedT+1']
    if t_plus_1_blocks:
        print(f"\n✓ T+1制度生效，阻止了 {len(t_plus_1_blocks)} 笔订单:")
        for order in t_plus_1_blocks:
            print(f"  - {order.datetime} {order.side} {order.symbol} size={order.size}")
    else:
        print("\n✗ 未检测到T+1阻止记录（可能需要检查实现）")
    
    return result


# 测试2: 停牌检测测试
def test_pause_detection():
    """测试停牌期间不能交易"""
    print("\n" + "="*60)
    print("测试2: 停牌检测")
    print("="*60)
    
    strategy_code = """
def initialize(context):
    g.security = '000001.SZ'
    
def handle_data(context, data):
    stock = g.security
    current_date = context.current_dt.date().isoformat()
    
    # 在停牌日尝试买入（根据示例数据，2024-03-15到2024-03-20停牌）
    if current_date == '2024-03-15':
        print(f"\\n[停牌日] {current_date}")
        print("尝试买入 100 股（应该被阻止）")
        order(stock, 100)
    
    # 在停牌结束后尝试买入
    elif current_date == '2024-03-21':
        print(f"\\n[复牌日] {current_date}")
        print("尝试买入 100 股（应该成功）")
        order(stock, 100)
"""
    
    result = run_backtest(
        symbol='000001.SZ',
        start='2024-03-14',
        end='2024-03-22',
        cash=100000,
        strategy_code=strategy_code,
        frequency='daily'
    )
    
    print("\n回测结果:")
    print(f"最终净值: {result.final_value:.2f}")
    
    # 检查是否有被停牌阻止的订单
    pause_blocks = [o for o in result.blocked_orders if o.status == 'BlockedPaused']
    if pause_blocks:
        print(f"\n✓ 停牌检测生效，阻止了 {len(pause_blocks)} 笔订单:")
        for order in pause_blocks:
            print(f"  - {order.datetime} {order.side} {order.symbol}")
    else:
        print("\n⚠ 未检测到停牌阻止记录")
        print("  可能原因：")
        print("  1. 停牌数据文件不存在或路径不正确")
        print("  2. 停牌数据文件格式不正确")
        print("  3. 日期不在停牌区间内")
    
    return result


# 测试3: closeable_amount 计算测试
def test_closeable_amount():
    """测试可卖数量计算"""
    print("\n" + "="*60)
    print("测试3: closeable_amount 计算")
    print("="*60)
    
    strategy_code = """
def initialize(context):
    g.security = '000001.SZ'
    g.day_count = 0
    
def handle_data(context, data):
    stock = g.security
    g.day_count += 1
    
    pos = context.portfolio.positions.get(stock)
    total = pos.total_amount if pos else 0
    closeable = pos.closeable_amount if pos else 0
    
    print(f"\\n[Day {g.day_count}] {context.current_dt.date()}")
    print(f"持仓总数: {total}, 可卖数量: {closeable}")
    
    # Day 1: 买入100股
    if g.day_count == 1:
        order(stock, 100)
        print("买入 100 股")
        print("预期：total=0(尚未成交), closeable=0")
    
    # Day 2: 再买入50股，检查昨日的100股是否可卖
    elif g.day_count == 2:
        print(f"预期：total=100(昨日买入), closeable=100(昨日可卖)")
        order(stock, 50)
        print("买入 50 股")
    
    # Day 3: 检查是否正确计算
    elif g.day_count == 3:
        print(f"预期：total=150(100+50), closeable=100(仅昨日的100股可卖)")
        if closeable == 100:
            print("✓ closeable_amount 计算正确!")
        else:
            print(f"✗ closeable_amount 计算错误! 预期100，实际{closeable}")
    
    # Day 4: 卖出50股，检查剩余
    elif g.day_count == 4:
        print(f"预期：total=150, closeable=150(全部可卖)")
        order(stock, -50)
        print("卖出 50 股")
    
    # Day 5: 检查卖出后的状态
    elif g.day_count == 5:
        print(f"预期：total=100, closeable=100")
"""
    
    result = run_backtest(
        symbol='000001.SZ',
        start='2024-02-01',
        end='2024-02-08',
        cash=100000,
        strategy_code=strategy_code,
        frequency='daily'
    )
    
    print("\n回测结果:")
    print(f"成交订单数: {len(result.orders)}")
    print(f"最终净值: {result.final_value:.2f}")
    
    return result


# 测试4: 综合测试
def test_combined():
    """综合测试T+1和停牌"""
    print("\n" + "="*60)
    print("测试4: 综合测试（T+1 + 停牌）")
    print("="*60)
    
    strategy_code = """
def initialize(context):
    g.security = '000001.SZ'
    
def handle_data(context, data):
    stock = g.security
    date = context.current_dt.date().isoformat()
    
    # 2024-03-14: 买入（停牌前一天）
    if date == '2024-03-14':
        print(f"\\n[{date}] 停牌前一天")
        order(stock, 100)
        print("买入 100 股")
    
    # 2024-03-15: 尝试卖出（停牌 + T+1）
    elif date == '2024-03-15':
        print(f"\\n[{date}] 停牌期间")
        pos = context.portfolio.positions.get(stock)
        if pos:
            print(f"持仓: {pos.total_amount}, 可卖: {pos.closeable_amount}")
            order(stock, -50)
            print("尝试卖出（停牌，应该被阻止）")
    
    # 2024-03-21: 复牌后卖出
    elif date == '2024-03-21':
        print(f"\\n[{date}] 复牌")
        pos = context.portfolio.positions.get(stock)
        if pos:
            print(f"持仓: {pos.total_amount}, 可卖: {pos.closeable_amount}")
            order(stock, -50)
            print("卖出 50 股（应该成功）")
"""
    
    result = run_backtest(
        symbol='000001.SZ',
        start='2024-03-13',
        end='2024-03-25',
        cash=100000,
        strategy_code=strategy_code,
        frequency='daily'
    )
    
    print("\n回测结果:")
    print(f"成交订单: {len(result.orders)}")
    print(f"被阻止订单: {len(result.blocked_orders)}")
    
    for blocked in result.blocked_orders:
        print(f"  - {blocked.datetime} {blocked.status}: {blocked.side} {blocked.symbol}")
    
    return result


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" "*20 + "T+1 和停牌检测功能测试套件")
    print("="*80)
    
    tests = [
        ("T+1制度", test_t_plus_1),
        ("停牌检测", test_pause_detection),
        ("closeable_amount计算", test_closeable_amount),
        ("综合测试", test_combined),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
            print(f"\n✓ {name} 测试完成")
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" "*30 + "测试总结")
    print("="*80)
    print(f"完成测试: {len(results)}/{len(tests)}")
    
    for name, result in results.items():
        if result:
            print(f"\n{name}:")
            print(f"  - 成交订单: {len(result.orders)}")
            print(f"  - 被阻止订单: {len(result.blocked_orders)}")
            print(f"  - 最终净值: {result.final_value:.2f}")


if __name__ == '__main__':
    main()
