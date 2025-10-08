import os, sys
# 确保项目根目录进入 sys.path 以便脚本直接运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.data_loader import resolve_price_file, load_price_dataframe

if __name__ == '__main__':
    for sym in ['600025.SH','600028.SH']:
        try:
            path = resolve_price_file(sym,'2024-01-01','2024-12-31',frequency='1min')
            df = load_price_dataframe(sym,'2024-01-01','2024-01-03',frequency='1min')
            print(f'{sym} path={path} rows={len(df)} head=')
            print(df.head())
        except Exception as e:
            print('ERROR', sym, e)
