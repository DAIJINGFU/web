"""对比 000514 在 adjust=auto 情况下 use_real_price=True 与 False 所选文件与最近5日收盘价。
运行: python scripts/test_000514_auto_compare.py
"""
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.app import data_loader as dl

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

CODE = '000514'
START = '2024-12-01'
END = '2025-03-01'


def fetch(use_real):
    holder = {}
    df = dl.load_price_dataframe(CODE, START, END, frequency='daily', adjust='auto', prefer_stockdata=True,
                                 data_root=DATA_ROOT, stockdata_root=None, use_real_price=use_real, out_path_holder=holder)
    tail = df.tail(5)[['datetime','close']].copy()
    tail['datetime'] = tail['datetime'].dt.strftime('%Y-%m-%d')
    return holder.get('path'), tail

if __name__ == '__main__':
    p_true, tail_true = fetch(True)
    p_false, tail_false = fetch(False)
    print('use_real_price=True  ->', p_true)
    print(tail_true.to_string(index=False))
    print('\nuse_real_price=False ->', p_false)
    print(tail_false.to_string(index=False))
    # 直接加载 raw / qfq 文件尾部以人工核对
    raw_path = os.path.join(DATA_ROOT, f'{CODE}_daily.csv')
    qfq_path = os.path.join(DATA_ROOT, f'{CODE}_daily_qfq.csv')
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        print('\nRaw file tail:')
        print(raw_df.tail(5))
    if os.path.exists(qfq_path):
        qfq_df = pd.read_csv(qfq_path)
        print('\nQFQ file tail:')
        print(qfq_df.tail(5))
