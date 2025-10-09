"""FastAPI 服务入口：提供本地回测接口与静态页面托管。"""

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from dataclasses import asdict
from .backtest_engine import run_backtest

app = FastAPI(title="Local Backtest Platform", version="0.1.0")

# 挂载静态前端(稍后生成) - 假设在项目根 frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))
if os.path.isdir(FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')

class BacktestRequest(BaseModel):
    symbol: str = "sample"
    start: str
    end: str
    cash: float = 100000
    benchmark_symbol: Optional[str] = None
    strategy_code: str
    strategy_params: Optional[Dict[str, Any]] = None
    frequency: str = 'daily'           # 新增: '1min' | 'daily' | 'weekly' | 'monthly'
    adjust_type: str = 'auto'          # 新增: 'auto' | 'raw' | 'qfq' | 'hfq'

@app.get('/')
async def index():
    # 简单重定向/返回说明
    if os.path.exists(os.path.join(FRONTEND_DIR, 'index.html')):
        with open(os.path.join(FRONTEND_DIR, 'index.html'), 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse('<h3>前端尚未创建</h3>')

@app.post('/api/backtest')
async def api_backtest(req: BacktestRequest):
    result = run_backtest(
        req.symbol,
        req.start,
        req.end,
        req.cash,
        req.strategy_code,
        req.strategy_params,
        benchmark_symbol=req.benchmark_symbol,
        frequency=req.frequency,
        adjust_type=req.adjust_type,
    )
    # 使用 asdict 递归转换 dataclass, 否则 TradeRecord 列表无法直接 JSON 序列化
    return JSONResponse(asdict(result))

# 方便 uvicorn 直接运行: uvicorn backend.app.main:app --reload
