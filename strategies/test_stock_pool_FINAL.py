# 导入聚宽函数库
import jqdata

# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    """
    初始化策略
    
    ⚠️ 重要：
    本策略不设置 g.security，因此必须在前端界面填写交易标的！
    例如：在"回测参数"的"标的代码"中填写 000001
    """
    # 设定深证成指作为基准
    set_benchmark('399001.SZ')
    
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    
    # 全局变量
    g.index_code = '399001.SZ'  # 深证成指（用于获取成分股）
    g.tested = False  # 是否已测试
    
    log.info("=" * 60)
    log.info("✅ 策略初始化完成")
    log.info("=" * 60)


# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    """每个交易日调用一次"""
    
    # 只在第一次运行时测试股票池功能
    if not g.tested:
        log.info("")
        log.info("=" * 60)
        log.info("开始测试股票池功能")
        log.info("=" * 60)
        
        try:
            # 测试1: 获取成分股（使用关键字参数指定日期）
            log.info(f"\n【测试1】获取 {g.index_code} 成分股")
            log.info(f"调用: get_index_stocks('{g.index_code}', date='2025-08-29')")
            
            stocks = get_index_stocks(g.index_code, date='2025-08-29')
            
            log.info(f"✅ 成功！获取到 {len(stocks)} 只成分股")
            log.info(f"前10只成分股: {stocks[:10]}")
            
            # 测试2: 获取权重数据
            log.info(f"\n【测试2】获取 {g.index_code} 成分股权重")
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
            log.info(f"❌ 测试失败: {str(e)}")
            log.info("=" * 60)
            
            # 打印详细错误
            import traceback
            log.info("\n详细错误信息:")
            error_details = traceback.format_exc()
            for line in error_details.split('\n'):
                log.info(line)
        
        # 标记已测试
        g.tested = True
        log.info("\n✅ 股票池测试已完成，后续交易日不再重复测试")
