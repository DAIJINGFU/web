"""股票池数据加载模块 - 模拟聚宽 API

本模块提供以下功能（与聚宽 API 保持一致）：
1. get_index_stocks(index_symbol, date=None) - 获取指数成分股
2. get_index_weights(index_id, date=None) - 获取指数成分股权重
3. get_industry_stocks(industry_code, date=None) - 获取行业成分股
4. get_concept_stocks(concept_code, date=None) - 获取概念板块成分股

数据存储位置：
- 指数成分及权重：stockdata/stockdata/深交所指数成分_YYYYMMDD/
- 行业分类数据：stockdata/stockdata/行业分类/ (需补充)
- 概念板块数据：stockdata/stockdata/概念板块/ (需补充)
- 全市场股票列表：stockdata/stockdata/stock_list/ (需补充)

使用示例：
    from backend.app import stock_pool
    
    # 获取沪深300成分股列表
    stocks = stock_pool.get_index_stocks('000300.XSHG', date='2025-08-29')
    
    # 获取指数成分股权重
    weights_df = stock_pool.get_index_weights('000300.XSHG', date='2025-08-29')
    
    # 获取某行业所有股票
    stocks = stock_pool.get_industry_stocks('I64', date='2025-08-29')
    
    # 获取某概念板块所有股票
    stocks = stock_pool.get_concept_stocks('SC0084', date='2019-04-10')
"""
from __future__ import annotations
import os
import re
from typing import Optional, List
from datetime import datetime, date as date_type
import pandas as pd

# 项目根路径
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_STOCKDATA_ROOT = os.path.join(_PROJECT_ROOT, 'stockdata', 'stockdata')

# 数据子目录
_INDEX_COMPONENT_DIR = os.path.join(_STOCKDATA_ROOT, '深交所指数成分_20250829')
_INDUSTRY_DIR = os.path.join(_STOCKDATA_ROOT, '行业分类')
_CONCEPT_DIR = os.path.join(_STOCKDATA_ROOT, '概念板块')
_STOCK_LIST_DIR = os.path.join(_STOCKDATA_ROOT, 'stock_list')


class StockPoolDataNotFound(Exception):
    """股票池数据未找到异常"""
    pass


def _normalize_index_symbol(symbol: str) -> str:
    """标准化指数代码
    
    支持格式：
    - 000300.XSHG -> 000300.SH
    - 000300.SH -> 000300.SH
    - 399001.XSHE -> 399001.SZ
    - 399001.SZ -> 399001.SZ
    - 000300 -> 000300.SH (默认上交所)
    """
    symbol = symbol.strip().upper()
    
    # 替换交易所后缀
    symbol = symbol.replace('.XSHG', '.SH')
    symbol = symbol.replace('.XSHE', '.SZ')
    symbol = symbol.replace('.SS', '.SH')
    symbol = symbol.replace('.SZSE', '.SZ')
    
    # 如果没有后缀，根据代码判断
    if '.' not in symbol:
        if symbol.startswith('399'):
            symbol = f"{symbol}.SZ"
        else:
            symbol = f"{symbol}.SH"
    
    return symbol


def _parse_date(date_str: Optional[str | datetime | date_type]) -> Optional[str]:
    """解析日期参数为 YYYYMMDD 格式字符串"""
    if date_str is None:
        return None
    
    if isinstance(date_str, str):
        # 移除所有分隔符
        date_str = date_str.replace('-', '').replace('/', '').replace('.', '')
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
        raise ValueError(f"Invalid date format: {date_str}, expected YYYY-MM-DD or YYYYMMDD")
    
    if isinstance(date_str, (datetime, date_type)):
        return date_str.strftime('%Y%m%d')
    
    raise TypeError(f"date must be str, datetime or date, got {type(date_str)}")


def _find_index_component_file(index_symbol: str, date_str: Optional[str] = None) -> str:
    """查找指数成分文件
    
    Args:
        index_symbol: 标准化后的指数代码 (如 000300.SH)
        date_str: 日期 YYYYMMDD 格式，如果为 None 则使用最新可用数据
        
    Returns:
        文件完整路径
        
    Raises:
        StockPoolDataNotFound: 文件未找到
    """
    # 当前只有一个固定日期的目录
    if not os.path.exists(_INDEX_COMPONENT_DIR):
        raise StockPoolDataNotFound(
            f"Index component directory not found: {_INDEX_COMPONENT_DIR}\n"
            f"请补充指数成分数据到此目录"
        )
    
    # 查找对应的指数文件
    file_path = os.path.join(_INDEX_COMPONENT_DIR, f"{index_symbol}.csv")
    
    if not os.path.exists(file_path):
        raise StockPoolDataNotFound(
            f"Index component file not found: {file_path}\n"
            f"可用的指数代码请查看目录: {_INDEX_COMPONENT_DIR}"
        )
    
    return file_path


def get_index_stocks(index_symbol: str, date: Optional[str | datetime | date_type] = None) -> List[str]:
    """获取指数在指定日期的成分股列表
    
    模拟聚宽 API: get_index_stocks(index_symbol, date=None)
    
    Args:
        index_symbol: 指数代码，如 '000300.XSHG' (沪深300)、'399001.XSHE' (深证成指)
        date: 查询日期，格式如 '2015-10-15' 或 datetime 对象
              可以是 None，表示使用默认日期
              回测模块：默认值会随回测日期变化，等于 context.current_dt
              研究模块：默认是今天
              
    Returns:
        股票代码列表 (list)，如 ['000001.XSHE', '000002.XSHE', ...]
        
    Raises:
        StockPoolDataNotFound: 数据文件不存在
        TypeError: 参数类型错误
        
    Example:
        # 获取沪深300成分股（不指定日期）
        stocks = get_index_stocks('000300.XSHG')
        
        # 获取指定日期的成分股（使用关键字参数）
        stocks = get_index_stocks('000300.XSHG', date='2025-08-29')
        
        log.info(stocks)
    """
    # 参数验证
    if not isinstance(index_symbol, str):
        raise TypeError(f"index_symbol must be str, got {type(index_symbol).__name__}")
    
    normalized_symbol = _normalize_index_symbol(index_symbol)
    date_str = _parse_date(date) if date else None
    
    file_path = _find_index_component_file(normalized_symbol, date_str)
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 检查必要的列
        if '成分股票代码' not in df.columns:
            raise ValueError(f"Column '成分股票代码' not found in {file_path}")
        
        # 获取唯一的股票代码列表
        stocks = df['成分股票代码'].unique().tolist()
        
        return stocks
        
    except Exception as e:
        raise StockPoolDataNotFound(f"Error loading index stocks from {file_path}: {str(e)}")


def get_index_weights(index_id: str, date: Optional[str | datetime | date_type] = None) -> pd.DataFrame:
    """获取指数成分股权重
    
    模拟聚宽 API: get_index_weights(index_id, date=None)
    
    Args:
        index_id: 必选参数，代表指数的标准形式代码，形如：'000001.XSHG'
                 若代码格式错误或者传入不存在的指数代码，提报错信息
        date: 可选参数，查询权重值的日期，形如：'%Y-%m-%d'，例如'2018-05-03'
              除此之外还可以传入[datetime.date]/[datetime.datetime]对象
              当date为None时，在回测环境中，默认为context.current_dt.date()
              在研究环境中，默认为datetime.now().date()
              
    Returns:
        查询到过去日期，且有权重数据，返回 pandas.DataFrame
        列包括：code(股票代码), display_name(股票名称), date(日期), weight(权重)
        
        查询到过去日期，但无权重数据，返回距查询日期最近日期的权重信息
        找不到过去日期的权重信息，返回距查询日期最近日期的最近日期的权重信息
        
    Raises:
        StockPoolDataNotFound: 数据文件不存在
        
    Example:
        >>> get_index_weights(index_id="000001.XSHG", date="2018-05-09")
        
        ======== ======== ========== ========
        code     display_name  date   weight
        ======== ======== ========== ========
        000002.XSHG 万科A  2018-04-27  1.43
        000001.XSHG 平安银行  2018-04-27  0.93
        ======== ======== ========== ========
    """
    normalized_symbol = _normalize_index_symbol(index_id)
    date_str = _parse_date(date) if date else None
    
    file_path = _find_index_component_file(normalized_symbol, date_str)
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 检查必要的列
        required_cols = ['成分股票代码', '交易日期', '权重']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {file_path}")
        
        # 重命名列以匹配聚宽格式
        result = df.rename(columns={
            '成分股票代码': 'code',
            '交易日期': 'date',
            '权重': 'weight'
        })
        
        # 添加 display_name 列（如果原始数据没有，暂时设为空）
        if 'display_name' not in result.columns:
            result['display_name'] = ''
        
        # 选择需要的列
        result = result[['code', 'display_name', 'date', 'weight']]
        
        return result
        
    except Exception as e:
        raise StockPoolDataNotFound(f"Error loading index weights from {file_path}: {str(e)}")


def get_industry_stocks(industry_code: str, date: Optional[str | datetime | date_type] = None) -> List[str]:
    """获取在给定日期一个行业的所有股票
    
    模拟聚宽 API: get_industry_stocks(industry_code, date=None)
    
    Args:
        industry_code: 行业编码
        date: 查询日期，一个字符串(格式类似'2015-10-15')或者[datetime.date]/[datetime.datetime]对象
              可以是None，使用默认日期
              回测模块：默认值会随回测日期变化，等于context.current_dt
              研究模块：默认是今天
              
    Returns:
        股票代码列表 (list)
        
    Raises:
        StockPoolDataNotFound: 数据文件不存在
        
    Example:
        # 获取计算机/互联网行业的成分股
        stocks = get_industry_stocks('I64')
        
    Note:
        **数据缺失**：需要补充行业分类数据
        文件格式建议：行业代码.csv，包含列：股票代码, 交易日期, 行业名称
        存放目录：stockdata/stockdata/行业分类/
    """
    date_str = _parse_date(date) if date else None
    
    if not os.path.exists(_INDUSTRY_DIR):
        raise StockPoolDataNotFound(
            f"Industry data directory not found: {_INDUSTRY_DIR}\n"
            f"请补充行业分类数据到此目录\n"
            f"建议文件格式：行业代码.csv，包含列：股票代码, 交易日期, 行业名称"
        )
    
    file_path = os.path.join(_INDUSTRY_DIR, f"{industry_code}.csv")
    
    if not os.path.exists(file_path):
        raise StockPoolDataNotFound(
            f"Industry file not found: {file_path}\n"
            f"请补充行业代码 {industry_code} 的数据"
        )
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 假设列名为 '股票代码' 或 'code'
        stock_col = '股票代码' if '股票代码' in df.columns else 'code'
        
        if stock_col not in df.columns:
            raise ValueError(f"Column '股票代码' or 'code' not found in {file_path}")
        
        stocks = df[stock_col].unique().tolist()
        
        return stocks
        
    except Exception as e:
        raise StockPoolDataNotFound(f"Error loading industry stocks from {file_path}: {str(e)}")


def get_concept_stocks(concept_code: str, date: Optional[str | datetime | date_type] = None) -> List[str]:
    """获取在给定日期一个概念板块的所有股票
    
    模拟聚宽 API: get_concept_stocks(concept_code, date=None)
    
    Args:
        concept_code: 概念板块编码
        date: 查询日期，一个字符串(格式类似'2015-10-15')或者[datetime.date]/[datetime.datetime]对象
              可以是None，使用默认日期
              回测模块：默认值会随回测日期变化，等于context.current_dt
              研究模块：默认是今天
              
    Returns:
        股票代码列表 (list)
        
    Raises:
        StockPoolDataNotFound: 数据文件不存在
        
    Example:
        # 获取雄安概念板块的成分股
        stocks = get_concept_stocks('SC0084', date='2019-04-10')
        print(stocks)
        
    Note:
        **数据缺失**：需要补充概念板块数据
        
        申万概念板块调整说明：
        - 由于2014年2月21日做了调整，2014年2月21日有几个行业被删除了
        - 同时又增加了新的行业，2014年2月21日之前的行业是28个
        - 之前是23个，历史上总共有34个
        
        文件格式建议：概念代码.csv，包含列：股票代码, 交易日期, 概念名称
        存放目录：stockdata/stockdata/概念板块/
    """
    date_str = _parse_date(date) if date else None
    
    if not os.path.exists(_CONCEPT_DIR):
        raise StockPoolDataNotFound(
            f"Concept data directory not found: {_CONCEPT_DIR}\n"
            f"请补充概念板块数据到此目录\n"
            f"建议文件格式：概念代码.csv，包含列：股票代码, 交易日期, 概念名称"
        )
    
    file_path = os.path.join(_CONCEPT_DIR, f"{concept_code}.csv")
    
    if not os.path.exists(file_path):
        raise StockPoolDataNotFound(
            f"Concept file not found: {file_path}\n"
            f"请补充概念代码 {concept_code} 的数据"
        )
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 假设列名为 '股票代码' 或 'code'
        stock_col = '股票代码' if '股票代码' in df.columns else 'code'
        
        if stock_col not in df.columns:
            raise ValueError(f"Column '股票代码' or 'code' not found in {file_path}")
        
        stocks = df[stock_col].unique().tolist()
        
        return stocks
        
    except Exception as e:
        raise StockPoolDataNotFound(f"Error loading concept stocks from {file_path}: {str(e)}")


def get_all_securities(types: Optional[List[str]] = None, date: Optional[str | datetime | date_type] = None) -> pd.DataFrame:
    """获取全市场股票/基金/指数等的列表
    
    扩展功能（参考聚宽但需要补充数据）
    
    Args:
        types: 证券类型列表，默认为 ['stock']
               可选：'stock'(股票), 'fund'(基金), 'index'(指数), 'futures'(期货), 'etf'(ETF基金), 'lof'(LOF基金), 'fja'(分级A), 'fjb'(分级B)
        date: 查询日期
        
    Returns:
        pandas.DataFrame，包含列：code(代码), display_name(名称), name(简称), start_date(上市日期), end_date(退市日期), type(类型)
        
    Raises:
        StockPoolDataNotFound: 数据文件不存在
        
    Note:
        **数据缺失**：需要补充全市场股票列表数据
        文件格式建议：all_securities.csv 或按类型分文件
        存放目录：stockdata/stockdata/stock_list/
    """
    if types is None:
        types = ['stock']
    
    date_str = _parse_date(date) if date else None
    
    if not os.path.exists(_STOCK_LIST_DIR):
        raise StockPoolDataNotFound(
            f"Stock list directory not found: {_STOCK_LIST_DIR}\n"
            f"请补充全市场证券列表数据到此目录\n"
            f"建议文件格式：all_securities.csv 或按类型分文件 (stock.csv, fund.csv 等)"
        )
    
    # 尝试加载数据
    result_dfs = []
    
    for sec_type in types:
        file_path = os.path.join(_STOCK_LIST_DIR, f"{sec_type}.csv")
        
        if not os.path.exists(file_path):
            # 尝试通用文件
            file_path = os.path.join(_STOCK_LIST_DIR, "all_securities.csv")
            
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 如果有 type 列，过滤
                if 'type' in df.columns:
                    df = df[df['type'] == sec_type]
                    
                result_dfs.append(df)
            except Exception as e:
                continue
    
    if not result_dfs:
        raise StockPoolDataNotFound(
            f"No securities data found in {_STOCK_LIST_DIR}\n"
            f"请补充证券列表数据"
        )
    
    return pd.concat(result_dfs, ignore_index=True)


# 便捷函数：获取所有可用的指数列表
def get_available_indexes() -> List[str]:
    """获取本地可用的所有指数代码
    
    Returns:
        指数代码列表
    """
    if not os.path.exists(_INDEX_COMPONENT_DIR):
        return []
    
    indexes = []
    for filename in os.listdir(_INDEX_COMPONENT_DIR):
        if filename.endswith('.csv'):
            index_code = filename.replace('.csv', '')
            indexes.append(index_code)
    
    return sorted(indexes)


# 便捷函数：列出本地数据情况
def list_local_data_status() -> dict:
    """列出本地数据的完整性状态
    
    Returns:
        字典，包含各类数据的状态信息
    """
    status = {
        'index_components': {
            'available': os.path.exists(_INDEX_COMPONENT_DIR),
            'path': _INDEX_COMPONENT_DIR,
            'count': 0,
            'indexes': []
        },
        'industry_data': {
            'available': os.path.exists(_INDUSTRY_DIR),
            'path': _INDUSTRY_DIR,
            'count': 0
        },
        'concept_data': {
            'available': os.path.exists(_CONCEPT_DIR),
            'path': _CONCEPT_DIR,
            'count': 0
        },
        'stock_list': {
            'available': os.path.exists(_STOCK_LIST_DIR),
            'path': _STOCK_LIST_DIR,
            'count': 0
        }
    }
    
    # 统计指数成分数据
    if status['index_components']['available']:
        indexes = get_available_indexes()
        status['index_components']['count'] = len(indexes)
        status['index_components']['indexes'] = indexes[:10]  # 只显示前10个
    
    # 统计行业数据
    if status['industry_data']['available']:
        status['industry_data']['count'] = len([f for f in os.listdir(_INDUSTRY_DIR) if f.endswith('.csv')])
    
    # 统计概念数据
    if status['concept_data']['available']:
        status['concept_data']['count'] = len([f for f in os.listdir(_CONCEPT_DIR) if f.endswith('.csv')])
    
    # 统计股票列表数据
    if status['stock_list']['available']:
        status['stock_list']['count'] = len([f for f in os.listdir(_STOCK_LIST_DIR) if f.endswith('.csv')])
    
    return status
