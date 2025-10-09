"""交易规则辅助模块：当前主要提供停牌判断工具函数。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

__all__ = ["is_stock_paused", "_is_stock_paused"]


def is_stock_paused(stock_code: str, check_date: str, jq_state: Dict[str, Any]) -> bool:
    """判断指定日期内标的是否处于停牌状态，返回布尔结果。"""

    try:
        data_root = Path(jq_state.get("options", {}).get("datadir", "stockdata/stockdata"))
        base_code = stock_code.replace(".XSHE", ".SZ").replace(".XSHG", ".SH")
        base_code = base_code.replace(".xshe", ".SZ").replace(".xshg", ".SH")
        possible_paths = [
            data_root / "停牌数据" / f"{base_code}.csv",
            data_root / "pause_data" / f"{base_code}.csv",
            data_root / "停牌数据" / f"{base_code.split('.')[0]}.csv",
        ]

        for pause_file in possible_paths:
            if not pause_file.exists():
                continue
            df = pd.read_csv(pause_file)
            date_cols = [c for c in df.columns if "日期" in c or "date" in c.lower()]
            start_cols = [c for c in df.columns if "开始" in c or "start" in c.lower()]
            end_cols = [c for c in df.columns if "结束" in c or "end" in c.lower()]
            if not (start_cols and end_cols):
                continue
            start_col = start_cols[0]
            end_col = end_cols[0]
            for _, row in df.iterrows():
                start_date = str(row[start_col])[:10]
                end_date = str(row[end_col])[:10]
                if start_date <= check_date <= end_date:
                    jq_state.setdefault("log", []).append(
                        f"[pause_check] {stock_code} paused: {start_date} to {end_date}"
                    )
                    return True
        return False
    except Exception as exc:  # pragma: no cover - defensive
        jq_state.setdefault("log", []).append(f"[pause_check] 停牌检测失败: {exc}")
        return False


_is_stock_paused = is_stock_paused
