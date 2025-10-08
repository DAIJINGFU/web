"""简单股票池测试策略

这是一个最简化的股票池功能测试策略
目的：验证 get_index_stocks 和 get_index_weights 函数是否正常工作

使用方法：
1. 在前端界面选择任意一只股票作为交易标的（如 000001）
2. 策略中会调用股票池函数获取深证成指成分股信息
3. 不进行实际交易，只打印信息
"""

# 导入聚宽函数库（会被回测引擎忽略）
import jqdata

def initialize(context):
    """初始化函数"""
    # 设定深证成指作为基准
    set_benchmark('399001.SZ')
    
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    
    # 定义全局变量
    g.index_code = '399001.SZ'  # 深证成指
    g.tested = False  # 是否已测试过股票池功能
    
    log.info("=" * 60)
    log.info("股票池功能测试策略已初始化")
    log.info(f"将测试指数: {g.index_code}")
    log.info("=" * 60)


def handle_data(context, data):
    """每日运行函数"""
    
    # 只在第一次运行时测试股票池功能
    if not g.tested:
        test_stock_pool_functions()
        g.tested = True


def test_stock_pool_functions():
    """测试股票池函数"""
    log.info("")
    log.info("=" * 60)
    log.info("开始测试股票池功能")
    log.info("=" * 60)
    
    try:
        # 测试 1: 获取指数成分股
        log.info(f"\n【测试1】获取 {g.index_code} 成分股列表")
        log.info("-" * 60)
        
        stocks = get_index_stocks(g.index_code)
        
        log.info(f"✅ 成功！获取到 {len(stocks)} 只成分股")
        log.info(f"前10只成分股: {stocks[:10]}")
        
        # 测试 2: 获取指数成分股权重
        log.info(f"\n【测试2】获取 {g.index_code} 成分股权重")
        log.info("-" * 60)
        
        weights_df = get_index_weights(g.index_code)
        
        log.info(f"✅ 成功！获取到 {len(weights_df)} 条权重数据")
        log.info(f"权重总和: {weights_df['weight'].sum():.2f}%")
        log.info(f"最大权重: {weights_df['weight'].max():.2f}%")
        log.info(f"最小权重: {weights_df['weight'].min():.2f}%")
        
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
        log.info(f"❌ 测试失败: {str(e)}")
        log.info("=" * 60)
        log.info("\n可能的原因:")
        log.info("1. 指数成分数据文件不存在")
        log.info("2. 数据文件格式不正确")
        log.info("3. 文件路径配置问题")
        log.info("\n解决方法:")
        log.info("请查看 STOCK_POOL_DATA_README.md 了解数据要求")
