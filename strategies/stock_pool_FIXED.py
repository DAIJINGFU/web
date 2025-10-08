# 导入聚宽函数库
import jqdata

# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    """
    初始化策略
    
    重要提示：
    1. 在前端界面的"回测参数"中需要填写一个具体的股票代码（如 000001）
    2. 不要填写指数代码（如 399001.SZ）作为交易标的
    3. 策略中可以使用股票池函数获取指数成分股信息
    """
    # 设定深证成指作为基准
    set_benchmark('399001.SZ')
    
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    
    # 全局变量
    g.index_code = '399001.SZ'  # 深证成指
    g.tested = False
    
    log.info("=" * 60)
    log.info("策略初始化完成")
    log.info("=" * 60)


def handle_data(context, data):
    """每个交易日调用一次"""
    
    # 只在第一次运行时测试股票池功能
    if not g.tested:
        test_stock_pool()
        g.tested = True


def test_stock_pool():
    """测试股票池功能"""
    log.info("")
    log.info("=" * 60)
    log.info("开始测试股票池功能")
    log.info("=" * 60)
    
    try:
        # 正确用法 1: 不传日期参数（使用默认）
        log.info(f"\n【方法1】不指定日期")
        log.info(f"调用: get_index_stocks('{g.index_code}')")
        
        stocks = get_index_stocks(g.index_code)
        
        log.info(f"✅ 成功！获取到 {len(stocks)} 只成分股")
        log.info(f"前10只: {stocks[:10]}")
        
        # 正确用法 2: 使用关键字参数指定日期
        log.info(f"\n【方法2】使用关键字参数指定日期")
        log.info(f"调用: get_index_stocks('{g.index_code}', date='2025-08-29')")
        
        stocks_with_date = get_index_stocks(g.index_code, date='2025-08-29')
        
        log.info(f"✅ 成功！获取到 {len(stocks_with_date)} 只成分股")
        
        # 获取权重数据
        log.info(f"\n【方法3】获取权重数据")
        log.info(f"调用: get_index_weights('{g.index_code}')")
        
        weights_df = get_index_weights(g.index_code)
        
        log.info(f"✅ 成功！获取到 {len(weights_df)} 条权重数据")
        log.info(f"权重总和: {weights_df['weight'].sum():.2f}%")
        
        # 显示权重最大的前5只
        log.info("\n权重最大的前5只股票:")
        top5 = weights_df.nlargest(5, 'weight')
        for i in range(len(top5)):
            row = top5.iloc[i]
            log.info(f"  {i+1}. {row['code']}: {row['weight']:.2f}%")
        
        log.info("")
        log.info("=" * 60)
        log.info("✅ 股票池功能测试全部通过！")
        log.info("=" * 60)
        
    except Exception as e:
        log.info("")
        log.info("=" * 60)
        log.info(f"❌ 测试失败!")
        log.info(f"错误信息: {str(e)}")
        log.info("=" * 60)
        
        # 导入 traceback 查看详细错误
        import traceback
        log.info("\n详细错误堆栈:")
        log.info(traceback.format_exc())


# ============================================
# 错误示例（仅供参考，不要使用）
# ============================================
"""
# ❌ 错误用法1：使用位置参数传递日期
stocks = get_index_stocks('399001.SZ', '2025-08-29')  # 错误！

# ✅ 正确用法：使用关键字参数
stocks = get_index_stocks('399001.SZ', date='2025-08-29')  # 正确

# ✅ 或者不传日期
stocks = get_index_stocks('399001.SZ')  # 正确
"""
