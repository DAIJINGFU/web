"""股票池模块测试脚本

用于测试股票池功能是否正常工作
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.app import stock_pool

def test_data_status():
    """测试数据状态"""
    print("=" * 60)
    print("1. 检查本地数据状态")
    print("=" * 60)
    
    status = stock_pool.list_local_data_status()
    
    for key, value in status.items():
        print(f"\n【{key}】")
        print(f"  路径: {value['path']}")
        print(f"  可用: {'✅ 是' if value['available'] else '❌ 否'}")
        print(f"  文件数: {value['count']}")
        
        if key == 'index_components' and value.get('indexes'):
            print(f"  部分指数: {value['indexes'][:5]}")
    
    print("\n" + "=" * 60)


def test_available_indexes():
    """测试获取可用指数列表"""
    print("\n2. 获取可用指数列表")
    print("=" * 60)
    
    indexes = stock_pool.get_available_indexes()
    print(f"可用指数数量: {len(indexes)}")
    
    if len(indexes) > 0:
        print(f"\n前10个指数:")
        for idx in indexes[:10]:
            print(f"  - {idx}")
    else:
        print("⚠️ 没有找到任何指数数据")
    
    print("=" * 60)


def test_get_index_stocks():
    """测试获取指数成分股"""
    print("\n3. 测试获取指数成分股")
    print("=" * 60)
    
    test_indexes = ['399001.SZ', '399006.SZ', '000300.SH']
    
    for index_code in test_indexes:
        print(f"\n测试指数: {index_code}")
        try:
            stocks = stock_pool.get_index_stocks(index_code)
            print(f"  ✅ 成功获取 {len(stocks)} 只成分股")
            if len(stocks) > 0:
                print(f"  前5只: {stocks[:5]}")
        except stock_pool.StockPoolDataNotFound as e:
            print(f"  ❌ 数据未找到: {str(e)[:100]}")
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")
    
    print("\n" + "=" * 60)


def test_get_index_weights():
    """测试获取指数权重"""
    print("\n4. 测试获取指数成分股权重")
    print("=" * 60)
    
    test_indexes = ['399001.SZ']
    
    for index_code in test_indexes:
        print(f"\n测试指数: {index_code}")
        try:
            weights_df = stock_pool.get_index_weights(index_code)
            print(f"  ✅ 成功获取 {len(weights_df)} 条权重数据")
            
            if len(weights_df) > 0:
                print(f"\n  权重数据示例（前5条）:")
                print(weights_df.head())
                
                print(f"\n  权重统计:")
                print(f"    总权重: {weights_df['weight'].sum():.2f}")
                print(f"    最大权重: {weights_df['weight'].max():.2f}")
                print(f"    最小权重: {weights_df['weight'].min():.2f}")
                
        except stock_pool.StockPoolDataNotFound as e:
            print(f"  ❌ 数据未找到: {str(e)[:100]}")
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")
    
    print("\n" + "=" * 60)


def test_get_industry_stocks():
    """测试获取行业成分股"""
    print("\n5. 测试获取行业成分股")
    print("=" * 60)
    
    test_codes = ['I64']  # 计算机行业
    
    for code in test_codes:
        print(f"\n测试行业代码: {code}")
        try:
            stocks = stock_pool.get_industry_stocks(code)
            print(f"  ✅ 成功获取 {len(stocks)} 只成分股")
            if len(stocks) > 0:
                print(f"  前5只: {stocks[:5]}")
        except stock_pool.StockPoolDataNotFound as e:
            print(f"  ⚠️ 数据未找到（需要补充）: {str(e)[:100]}")
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")
    
    print("\n" + "=" * 60)


def test_get_concept_stocks():
    """测试获取概念板块成分股"""
    print("\n6. 测试获取概念板块成分股")
    print("=" * 60)
    
    test_codes = ['SC0084']  # 雄安概念
    
    for code in test_codes:
        print(f"\n测试概念代码: {code}")
        try:
            stocks = stock_pool.get_concept_stocks(code)
            print(f"  ✅ 成功获取 {len(stocks)} 只成分股")
            if len(stocks) > 0:
                print(f"  前5只: {stocks[:5]}")
        except stock_pool.StockPoolDataNotFound as e:
            print(f"  ⚠️ 数据未找到（需要补充）: {str(e)[:100]}")
        except Exception as e:
            print(f"  ❌ 错误: {str(e)[:100]}")
    
    print("\n" + "=" * 60)


def print_summary():
    """打印总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("""
✅ 已实现的功能:
  - get_index_stocks()      获取指数成分股
  - get_index_weights()     获取指数权重
  - get_industry_stocks()   获取行业成分股（需补充数据）
  - get_concept_stocks()    获取概念板块成分股（需补充数据）
  - get_all_securities()    获取全市场证券列表（需补充数据）

📁 现有数据:
  - ✅ 深交所指数成分数据（约300+个指数）
  - ✅ 股票日/周/月线数据
  - ✅ 分钟级数据
  - ✅ 基本面指标数据

⚠️ 需要补充的数据:
  1. 上交所指数成分数据（000300.SH等）
  2. 行业分类数据
  3. 概念板块数据
  4. 全市场股票列表
  5. 指数成分历史数据（时间序列）

📖 详细说明请查看: STOCK_POOL_DATA_README.md
    """)
    print("=" * 60)


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 18 + "股票池模块测试" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        test_data_status()
        test_available_indexes()
        test_get_index_stocks()
        test_get_index_weights()
        test_get_industry_stocks()
        test_get_concept_stocks()
        print_summary()
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程出现错误: {e}")
        import traceback
        traceback.print_exc()
