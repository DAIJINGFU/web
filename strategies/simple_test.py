# 最简单的股票池测试策略
# 只调用 get_index_stocks 和 get_index_weights 并打印结果

import jqdata

def initialize(context):
    # 设定基准
    set_benchmark('399001.SZ')
    # 全局变量
    g.tested = False

def handle_data(context, data):
    # 只执行一次
    if not g.tested:
        # 测试深证成指
        log.info("开始测试股票池函数")
        
        # 1. 获取成分股
        stocks = get_index_stocks('399001.SZ')
        log.info(f"成分股数量: {len(stocks)}")
        log.info(f"前10只: {stocks[:10]}")
        
        # 2. 获取权重
        weights_df = get_index_weights('399001.SZ')
        log.info(f"权重数据行数: {len(weights_df)}")
        log.info("前5行:")
        log.info(str(weights_df.head()))
        
        g.tested = True
