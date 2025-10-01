import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.app.data_loader import resolve_price_file

print('PROJECT ROOT', ROOT)
try:
    p = resolve_price_file('000300.XSHG', '2018-01-01','2020-01-01', frequency='daily', adjust='auto', prefer_stockdata=True)
    print('RESOLVED:', p)
except Exception as e:
    import traceback; traceback.print_exc()
