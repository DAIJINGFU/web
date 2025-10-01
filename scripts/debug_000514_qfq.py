import os, sys, pandas as pd
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.app import data_loader as dl

df = dl.load_price_dataframe('000514', '2022-12-01','2023-01-10', frequency='daily', adjust='qfq', prefer_stockdata=True)
print(df.tail(15)[['datetime','close']])