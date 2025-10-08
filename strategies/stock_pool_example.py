"""股票池使用示例策略

本策略演示如何使用类似聚宽的股票池功能：
1. 使用 get_index_stocks() 获取指数成分股
2. 使用 get_index_weights() 获取成分股权重
3. 动态调整股票池
4. 基于股票池进行轮动策略

策略逻辑：
- 获取深证成指(399001.SZ)成分股作为股票池
- 每月调仓一次
- 选择权重最大的前1只股票作为交易标的（示例简化）
- 采用简单的趋势跟踪策略

注意：
- 指数本身不能作为交易标的，需要选择成分股
- 本示例为了简化，只选择权重最大的1只股票进行演示
"""

def initialize(context):
    """初始化函数"""
    # 设置基准为深证成指
    set_benchmark('399001.SZ')
    
    # 设置滑点
    set_slippage(0.002)  # 0.2% 滑点
    
    # 开启真实价格模式
    set_option('use_real_price', True)
    
    # 初始化全局变量
    g.index_code = '399001.SZ'  # 指数代码（用于获取成分股）
    g.security = None  # 实际交易的股票（将从成分股中选择）
    g.stock_pool = []  # 当前股票池
    g.rebalance_days = 0  # 调仓天数计数器
    g.rebalance_period = 20  # 每20个交易日调仓一次（约1个月）
    g.initialized = False  # 标记是否已初始化股票池
    
    log.info("=" * 50)
    log.info("股票池示例策略初始化完成")
    log.info(f"股票池来源: {g.index_code} (深证成指)")
    log.info(f"调仓周期: 每 {g.rebalance_period} 个交易日")
    log.info("=" * 50)


def handle_data(context, data):
    """每日运行函数"""
    
    # 首次运行时初始化股票池并选择交易标的
    if not g.initialized:
        select_trading_security(context, data)
        g.initialized = True
    
    # 增加调仓计数器
    g.rebalance_days += 1
    
    # 判断是否需要调仓（重新选择股票）
    if g.rebalance_days >= g.rebalance_period:
        select_trading_security(context, data)
        g.rebalance_days = 0
    
    # 简单的交易逻辑示例
    # 注意：实际策略需要根据具体需求设计
    if g.security:
        log.info(f"当前交易标的: {g.security}")


def select_trading_security(context, data):
    """从股票池中选择交易标的"""
    log.info("=" * 50)
    log.info("开始选择交易标的")
    
    try:
        # 1. 获取指数成分股列表
        log.info(f"正在获取 {g.index_code} 成分股...")
        stocks = get_index_stocks(g.index_code)
        log.info(f"✅ 成功获取 {len(stocks)} 只成分股")
        
        # 2. 获取成分股权重
        log.info("正在获取成分股权重...")
        weights_df = get_index_weights(g.index_code)
        log.info(f"✅ 成功获取 {len(weights_df)} 条权重数据")
        
        # 显示前5个权重数据
        if len(weights_df) > 0:
            log.info("权重数据示例（前5个）:")
            for i in range(min(5, len(weights_df))):
                row = weights_df.iloc[i]
                log.info(f"  {row['code']}: {row['weight']:.2f}%")
        
        # 3. 选择权重最大的1只股票作为交易标的（示例简化）
        # 实际策略可以选择多只股票进行组合
        weights_df_sorted = weights_df.sort_values('weight', ascending=False)
        top_stock = weights_df_sorted.iloc[0]['code']
        top_weight = weights_df_sorted.iloc[0]['weight']
        
        # 更新交易标的
        g.security = top_stock
        g.stock_pool = stocks[:10]  # 保存前10只作为备选池
        
        log.info(f"✅ 选中交易标的: {g.security} (权重: {top_weight:.2f}%)")
        log.info(f"备选股票池大小: {len(g.stock_pool)}")
        
    except Exception as e:
        log.info(f"❌ 选择交易标的过程出现错误: {str(e)}")
        log.info("可能原因：")
        log.info("  1. 指数成分数据文件不存在")
        log.info("  2. 数据格式不正确")
        log.info("  3. 日期参数问题")
        
        # 发生错误时使用默认股票
        if not g.security:
            g.security = '000001.SZ'  # 默认使用平安银行
            log.info(f"⚠️ 使用默认股票: {g.security}")
    
    log.info("=" * 50)


# ============================================
# 以下是纯 Backtrader 风格的等效实现示例
# ============================================

"""
import backtrader as bt

class StockPoolStrategy(bt.Strategy):
    '''使用股票池的 Backtrader 策略示例'''
    
    params = (
        ('rebalance_period', 20),  # 调仓周期
        ('top_n', 10),  # 持仓数量
        ('index_code', '000300.XSHG'),  # 指数代码
    )
    
    def __init__(self):
        self.rebalance_counter = 0
        self.target_stocks = []
    
    def next(self):
        self.rebalance_counter += 1
        
        if self.rebalance_counter >= self.params.rebalance_period:
            self.rebalance()
            self.rebalance_counter = 0
    
    def rebalance(self):
        '''调仓逻辑'''
        try:
            # 获取指数成分股
            stocks = get_index_stocks(self.params.index_code)
            print(f'获取到 {len(stocks)} 只成分股')
            
            # 获取权重
            weights_df = get_index_weights(self.params.index_code)
            
            # 选择权重最大的N只
            weights_df_sorted = weights_df.sort_values('weight', ascending=False)
            self.target_stocks = weights_df_sorted.head(self.params.top_n)['code'].tolist()
            
            print(f'选出权重最大的 {len(self.target_stocks)} 只股票')
            
            # 执行调仓逻辑
            # ... 下单代码 ...
            
        except Exception as e:
            print(f'调仓错误: {e}')
"""
