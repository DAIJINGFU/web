# 导入聚宽函数库
import jqdata

# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    """
    注意：这个策略展示如何正确使用股票池功能
    
    关键点：
    1. 指数（如399001.SZ）只能用作基准或获取成分股，不能作为交易标的
    2. 必须选择具体的股票代码作为 context.set_universe 的参数
    3. 在聚宽兼容模式下，回测时会自动加载该股票的数据
    """
    
    # 设定深证成指作为基准（这是正确的用法）
    set_benchmark('399001.SZ')
    
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    
    # 定义全局变量
    g.index_code = '399001.SZ'  # 深证成指（用于获取成分股）
    g.tested = False  # 是否已测试
    
    log.info("策略初始化完成")


def handle_data(context, data):
    """每个单位时间调用一次"""
    
    # 只在第一次运行时测试
    if not g.tested:
        log.info("=" * 60)
        log.info("测试股票池功能")
        log.info("=" * 60)
        
        try:
            # 获取深证成指成分股
            stocks = get_index_stocks(g.index_code)
            log.info(f"✅ 成功获取 {len(stocks)} 只成分股")
            log.info(f"前5只: {stocks[:5]}")
            
            # 获取权重数据
            weights_df = get_index_weights(g.index_code)
            log.info(f"✅ 成功获取权重数据，共 {len(weights_df)} 条")
            
            # 选择权重最大的5只股票
            top5 = weights_df.nlargest(5, 'weight')
            log.info("权重最大的5只股票:")
            for i in range(len(top5)):
                row = top5.iloc[i]
                log.info(f"  {i+1}. {row['code']}: {row['weight']:.2f}%")
            
            log.info("=" * 60)
            
        except Exception as e:
            log.info(f"❌ 错误: {str(e)}")
        
        g.tested = True
