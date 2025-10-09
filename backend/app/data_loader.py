"""统一的数据加载模块：兼容旧版平铺目录 `data/` 与新版分层目录 `stockdata/`。

功能职责：
- 规范化输入的证券代码（示例：600008.XSHG -> 600008；000001.SZ -> 000001）。
- 依据频率(frequency)、复权模式(adjust_type) 和目录存在情况选择最合适的文件路径。
- 提供返回 Pandas DataFrame 的加载函数与 Backtrader 可直接使用的 DataFeed 包装。
- 保持向后兼容：旧代码仍可通过 `backtest_engine` 的 `load_csv_dataframe/load_csv_data` 间接调用本模块。

支持的频率：
- daily（日线，默认）
- weekly / monthly （来源：1d_1w_1m/<code>/ 下的 code_weekly[_qfq|_hfq].csv / code_monthly...）
- 1min （来源：stockdata/stockdata/1min/sz000001.csv 等）

复权模式 (adjust types)：
- raw ：未复权（文件名一般为 *_daily.csv / *_日.csv）
- qfq ：前复权（*_daily_qfq.csv / *_日_qfq.csv）
- hfq ：后复权（*_daily_hfq.csv / *_日_hfq.csv）
- auto：自动；默认优先前复权，若调用方指定“真实价格”逻辑可转向 raw（由调用层策略选项决定）。

环境变量覆盖：
- BACKTEST_DATA_ROOT：旧版数据根目录（默认 项目根/data）
- BACKTEST_STOCKDATA_ROOT：新版 stockdata 根目录（若不显式设置则按默认路径尝试）

对外暴露函数：
- resolve_price_file(symbol, ...)：解析并返回最终文件路径
- load_price_dataframe(...): 返回裁剪到指定起止日期的 DataFrame
- load_bt_feed(...): 返回 backtrader 的 PandasData feed
"""
from __future__ import annotations
import os
import re
from typing import Optional, Tuple, List
import pandas as pd
import backtrader as bt

# 探测项目根路径（当前文件位于 backend/app/ 下，两级回退）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_DEFAULT_STOCKDATA_DIR = os.path.join(_PROJECT_ROOT, 'stockdata', 'stockdata')  # 与仓库结构一致的内层 stockdata/stockdata

# 中英文 / 同义列名统一映射
_RENAME_MAP = {
    '日期': 'datetime',
    '交易日期': 'datetime',
    '开盘': 'open', '开盘价': 'open',
    '最高': 'high', '最高价': 'high',
    '最低': 'low', '最低价': 'low',
    '收盘': 'close', '收盘价': 'close',
    '成交量': 'volume',
    '成交额': 'amount',
    '股票代码': 'code', '代码': 'code',
}

class DataNotFound(Exception):
    pass

_symbol_ex_pattern = re.compile(r"^(?P<code>\d{6})(?:\.(?P<ex>[A-Za-z]{2,4}))?$")

_exchange_map = {
    'XSHG': 'SH', 'XSHG2': 'SH', 'SH': 'SH', 'SS': 'SH',
    'XSHE': 'SZ', 'SZ': 'SZ', 'SZSE': 'SZ'
}

_minute_prefix = {'SH': 'sh', 'SZ': 'sz'}


def normalize_symbol(symbol: str) -> Tuple[str, Optional[str]]:
    s = symbol.strip()
    s = s.replace('.xshg', '.XSHG').replace('.xshe', '.XSHE')
    m = _symbol_ex_pattern.match(s)
    if not m:
        # 尝试剥离诸如 _daily_qfq 等后缀得到核心代码，确保能够识别旧文件名中的代码部分
        core = re.sub(r"[_\.].*$", "", s)
        if len(core) == 6 and core.isdigit():
            return core, None
        return s, None
    code = m.group('code')
    ex = m.group('ex')
    if ex:
        exu = _exchange_map.get(ex.upper())
    else:
        exu = None
    return code, exu


def _candidate_daily_files(code: str, adjust: str, use_real_price: bool | None = None) -> List[str]:
    """生成候选文件序列。
    auto 语义：
      - use_real_price=True  -> raw > qfq > hfq (真实价优先)
      - 否则                 -> qfq > raw > hfq (默认前复权优先)
    其余模式显式固定。
    """
    seq_raw = [f"{code}_daily.csv", f"{code}_日.csv"]
    seq_qfq = [f"{code}_daily_qfq.csv", f"{code}_日_qfq.csv"]
    seq_hfq = [f"{code}_daily_hfq.csv", f"{code}_日_hfq.csv"]
    if adjust == 'raw':
        return seq_raw + seq_qfq + seq_hfq
    if adjust == 'qfq':
        return seq_qfq + seq_raw + seq_hfq
    if adjust == 'hfq':
        return seq_hfq + seq_raw + seq_qfq
    # auto
    if use_real_price:
        return seq_raw + seq_qfq + seq_hfq
    return seq_qfq + seq_raw + seq_hfq


def resolve_price_file(symbol: str,
                        start: str,
                        end: str,
                        frequency: str = 'daily',
                        adjust: str = 'auto',
                        prefer_stockdata: bool = True,
                        data_root: Optional[str] = None,
                        stockdata_root: Optional[str] = None,
                        use_real_price: bool | None = None) -> str:
    code, ex = normalize_symbol(symbol)
    data_root = data_root or os.environ.get('BACKTEST_DATA_ROOT', _DEFAULT_DATA_DIR)
    stockdata_root = stockdata_root or os.environ.get('BACKTEST_STOCKDATA_ROOT', _DEFAULT_STOCKDATA_DIR)

    freq = frequency.lower()
    # 优先新版目录：在尝试 1d_1w_1m / 1min 之前，先检查是否是“基准指数”代码并存在专用目录
    benchmark_root_env = os.environ.get('BACKTEST_BENCHMARK_ROOT')
    benchmark_dir_candidates: List[str] = []
    # 1) 显式环境变量
    if benchmark_root_env and os.path.isdir(benchmark_root_env):
        benchmark_dir_candidates.append(benchmark_root_env)
    # 2) 新版 stockdata 下的 "基准指数数据" 目录
    builtin_benchmark_dir = os.path.join(stockdata_root, '基准指数数据')
    if os.path.isdir(builtin_benchmark_dir):
        benchmark_dir_candidates.append(builtin_benchmark_dir)
    # 若是典型指数代码（000300 / 000001 等）尝试直接匹配 *_日 / *_daily 系列
    if freq == 'daily' and benchmark_dir_candidates:
        # 指数文件命名目前示例：000300_日.csv / 000001_日.csv
        for bench_dir in benchmark_dir_candidates:
            # 复权模式对指数通常不适用，直接尝试几种常见后缀
            bench_candidates = [
                f"{code}_日.csv", f"{code}_daily.csv",
                f"{code}_daily_qfq.csv", f"{code}_daily_hfq.csv",  # 兼容可能存在的复权形式
            ]
            for fname in bench_candidates:
                p = os.path.join(bench_dir, fname)
                if os.path.exists(p):
                    return p
    if prefer_stockdata and os.path.isdir(stockdata_root):
    # 新版 stockdata 目录结构
        if freq == '1min':
            # --- 分钟线命名兼容说明 ---
            #  为了兼容历史与新采集两种风格：
            #   1) 旧: sh600025.csv / sz000001_qfq.csv
            #   2) 新: 600025.SH.csv / 000001.SZ_qfq.csv
            #  这里的策略：优先按“新风格” code.EX[_{qfq|hfq}].csv 搜索；若不存在再尝试旧前缀式。
            #  调用方可以继续传入 600025.SH 或 600025.XSHG / 600025 皆可（normalize_symbol 会补充交易所）。
            # 额外说明：部分历史数据采集脚本可能将交易所写为小写，normalize_symbol 已统一大小写以降低歧义。
            # --------------------------------
            # 分钟线文件命名兼容：
            #  旧版: sh600025.csv / sz000001_qfq.csv
            #  新版: 600025.SH.csv / 000001.SZ_qfq.csv
            # 规则：优先尝试新版 (code.EX) 命名，再回退旧前缀式 (sh|sz+code)
            if not ex:
                ex = 'SH' if code.startswith('6') else 'SZ'
            prefix = _minute_prefix.get(ex, ex.lower())
            base_old = f"{prefix}{code}"      # sh600025
            base_new = f"{code}.{ex}"          # 600025.SH

            def _minute_name_candidates(base: str) -> List[str]:
                if adjust == 'raw':
                    return [base + '.csv', base + '_qfq.csv', base + '_hfq.csv']
                if adjust == 'qfq':
                    return [base + '_qfq.csv', base + '.csv', base + '_hfq.csv']
                if adjust == 'hfq':
                    return [base + '_hfq.csv', base + '.csv', base + '_qfq.csv']
                # auto
                if use_real_price:
                    return [base + '.csv', base + '_qfq.csv', base + '_hfq.csv']
                return [base + '_qfq.csv', base + '.csv', base + '_hfq.csv']

            minute_dir = os.path.join(stockdata_root, '1min')
            tried: List[str] = []
            # 组合候选：新版优先，其次旧版
            for fname in _minute_name_candidates(base_new) + _minute_name_candidates(base_old):
                path = os.path.join(minute_dir, fname)
                tried.append(path)
                if os.path.exists(path):
                    return path
            # 兜底扫描：查找任何前缀匹配 (新版/旧版) 的文件
            if os.path.isdir(minute_dir):
                for f in os.listdir(minute_dir):
                    if f.startswith(base_new) or f.startswith(base_old):
                        return os.path.join(minute_dir, f)
            raise DataNotFound(f"分钟数据缺失: code={code} ex={ex} tried={tried}")
        elif freq in ('daily','weekly','monthly'):
            subdir = os.path.join(stockdata_root, '1d_1w_1m', code)
            if not os.path.isdir(subdir):
                # 若新版不存在该股票子目录，退回旧 data 逻辑
                return _resolve_legacy_daily(code, adjust, data_root)
            if freq == 'daily':
                for fname in _candidate_daily_files(code, adjust, use_real_price):
                    p = os.path.join(subdir, fname)
                    if os.path.exists(p):
                        return p
            else:
                # 周 / 月线命名模式: code_weekly[_qfq].csv / code_monthly_hfq.csv 等
                base = f"{code}_{'weekly' if freq=='weekly' else 'monthly'}"
                # 针对日/周/月聚合文件，实际命名可能只存在一个 base.csv 或 base_qfq.csv 或 base_hfq.csv
                # 这里根据 adjust 模式生成更宽松、且包含 auto 情况的候选顺序：
                if adjust == 'qfq':
                    seq = [base + '_qfq.csv', base + '_hfq.csv', base + '.csv']
                elif adjust == 'hfq':
                    seq = [base + '_hfq.csv', base + '_qfq.csv', base + '.csv']
                elif adjust == 'raw':  # raw 优先非复权
                    seq = [base + '.csv', base + '_qfq.csv', base + '_hfq.csv']
                else:  # auto: 需要与日线保持一致，并结合 use_real_price
                    if use_real_price:
                        # 真实价优先 raw > qfq > hfq
                        seq = [base + '.csv', base + '_qfq.csv', base + '_hfq.csv']
                    else:
                        # 默认前复权优先 qfq > raw > hfq
                        seq = [base + '_qfq.csv', base + '.csv', base + '_hfq.csv']
                tried = []
                for fname in seq:
                    p = os.path.join(subdir, fname)
                    tried.append(p)
                    if os.path.exists(p):
                        return p
                # 兜底：扫描目录中任意以 base 前缀开头的文件
                for f in os.listdir(subdir):
                    if f.startswith(base):
                        return os.path.join(subdir, f)
                # 最终仍未找到，抛出并附带尝试路径方便排查
                raise DataNotFound(f"未找到{freq}文件: {subdir}; tried={tried}")
        else:
            raise ValueError(f"不支持频率: {frequency}")
    # 旧目录或回退逻辑
    if freq != 'daily':
        raise DataNotFound("旧 data 目录仅支持日线 (daily)")
    return _resolve_legacy_daily(code, adjust, data_root)


def _resolve_legacy_daily(code: str, adjust: str, data_root: str) -> str:
    # 扫描候选文件列表（旧目录没有 use_real_price 语义，直接按 adjust 顺序）
    for fname in _candidate_daily_files(code, adjust, use_real_price=None):
        path = os.path.join(data_root, fname)
        if os.path.exists(path):
            return path
    # 再次兜底：尝试 <code>.csv 简单命名
    raw = os.path.join(data_root, f"{code}.csv")
    if os.path.exists(raw):
        return raw
    raise DataNotFound(f"未找到任何可用日线文件: {code}")


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 列名标准化重命名
    cols = {c: _RENAME_MAP[c] for c in df.columns if c in _RENAME_MAP}
    if cols:
        df = df.rename(columns=cols)
    # 统一日期列名为 datetime
    if 'datetime' not in df.columns:
        for cand in ('date','Date','日期','交易日期'):
            if cand in df.columns:
                df = df.rename(columns={cand:'datetime'})
                break
    if 'datetime' not in df.columns:
        raise ValueError(f"文件缺少日期列: {path}")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    # 兼容含时区的数据：统一去除时区，内部使用本地 naive 时间
    try:
        if hasattr(df['datetime'].dt, 'tz') and df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_convert(None)
    except Exception:
        try:
            # 某些版本 pandas 需要 tz_localize(None)
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        except Exception:
            pass
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def load_price_dataframe(symbol: str,
                          start: str,
                          end: str,
                          frequency: str = 'daily',
                          adjust: str = 'auto',
                          prefer_stockdata: bool = True,
                          data_root: Optional[str] = None,
                          stockdata_root: Optional[str] = None,
                          use_real_price: bool | None = None,
                          out_path_holder: dict | None = None) -> pd.DataFrame:
    try:
        path = resolve_price_file(symbol, start, end, frequency, adjust, prefer_stockdata, data_root, stockdata_root, use_real_price)
        df = _read_csv(path)
        if out_path_holder is not None:
            out_path_holder['path'] = path
    except DataNotFound:
        # 若是周/月线文件缺失，尝试用日线聚合构造
        if frequency in ('weekly','monthly'):
            daily_df = load_price_dataframe(symbol, start, end, 'daily', adjust, prefer_stockdata, data_root, stockdata_root)
            if daily_df.empty:
                return daily_df
            daily_df = daily_df.set_index('datetime')
            rule = 'W-FRI' if frequency == 'weekly' else 'M'
            agg = daily_df.resample(rule).agg({
                'open':'first','high':'max','low':'min','close':'last',
                'volume':'sum' if 'volume' in daily_df.columns else 'first',
                'amount':'sum' if 'amount' in daily_df.columns else 'first',
            })
            agg = agg.dropna(subset=['close'])
            agg.reset_index(inplace=True)
            df = agg
        else:
            raise
    mask = (df['datetime'] >= pd.to_datetime(start)) & (df['datetime'] <= pd.to_datetime(end))
    return df.loc[mask].reset_index(drop=True)


def load_bt_feed(symbol: str,
                 start: str,
                 end: str,
                 frequency: str = 'daily',
                 adjust: str = 'auto',
                 prefer_stockdata: bool = True,
                 data_root: Optional[str] = None,
                 stockdata_root: Optional[str] = None,
                 use_real_price: bool | None = None,
                 out_path_holder: dict | None = None) -> bt.feeds.PandasData:
    df = load_price_dataframe(symbol, start, end, frequency, adjust, prefer_stockdata, data_root, stockdata_root, use_real_price, out_path_holder)
    required = ['datetime','open','high','low','close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"数据缺少必要列: {r}")
    if 'volume' not in df.columns:
    # 若缺失成交量列，填充 0（避免 backtrader 对 volume 的依赖报错）
        df['volume'] = 0
    feed_df = df[['datetime','open','high','low','close','volume']].set_index('datetime')
    return bt.feeds.PandasData(dataname=feed_df)

__all__ = [
    'normalize_symbol','resolve_price_file','load_price_dataframe','load_bt_feed','DataNotFound'
]
