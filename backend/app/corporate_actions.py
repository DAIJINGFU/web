"""公司行为处理模块（简化版），用于支持送转股、拆股与现金分红。

CSV 文件格式（位于 ``data/CA_<symbol>.csv`` 的示例）：
        date,action_type,ratio,cash,note
        2024-07-26,BONUS,0.01,,示例：ratio=0.10 表示 10 股转 1 股，新股数 = 持仓 * ratio
        2024-08-15,CASH_DIVIDEND,,0.25,示例：每股派息 0.25 元
        2024-09-10,SPLIT,0.5,,示例：每 1 股拆为 2 股 -> ratio=2.0（小于 1 表示合股）

字段说明：
        date: ISO 日期（YYYY-MM-DD）
        action_type: 事件类型，可选 BONUS | SPLIT | CASH_DIVIDEND
        ratio: 浮点数。BONUS 表示送/转股比例；SPLIT 表示拆分后每股得到的新股数；现金分红无需填写
        cash: 浮点数，仅 CASH_DIVIDEND 使用，表示每股派发现金
        note: 选填备注

实现简化：
    - BONUS：仅增加股数，不立即影响现金；引擎可据此生成等额买入指令。
    - SPLIT：按比例调整股数（pos_size *= ratio），假设行情数据已做前复权；否则会看到价格跳变。
    - CASH_DIVIDEND：增加现金账户，不考虑税费；如需复杂逻辑可自行扩展。

日志标签约定：
    [ca_load] 读取文件路径及事件数量
    [ca_event] 每条事件解析结果
    [ca_apply] 应用事件前后持仓与差异

扩展方式：实现 apply_event 时新增分支即可支持新的公司行为类型。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import csv
import os

@dataclass
class CorporateActionEvent:
    date: str               # ISO 日期，例如 2024-01-01
    action_type: str        # 事件类型：BONUS | SPLIT | CASH_DIVIDEND
    ratio: Optional[float] = None  # BONUS 对应送转比例；SPLIT 对应拆分后的换股比
    cash: Optional[float] = None   # 每股现金分红金额
    note: Optional[str] = None

SUPPORTED_TYPES = {"BONUS", "SPLIT", "CASH_DIVIDEND"}


def load_corporate_actions(symbol: str, datadir: str, logger=None) -> List[CorporateActionEvent]:
    """从 CSV 文件加载公司行为，按日期排序后返回。优先文件名：``CA_<symbol>.csv`` 或 ``<symbol>_CA.csv``。"""
    candidates = [f"CA_{symbol}.csv", f"{symbol}_CA.csv"]
    path = None
    for fname in candidates:
        p = os.path.join(datadir, fname)
        if os.path.exists(p):
            path = p
            break
    events: List[CorporateActionEvent] = []
    if not path:
        return events
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    atype = (row.get('action_type') or '').strip().upper()
                    if atype not in SUPPORTED_TYPES:
                        continue
                    date = (row.get('date') or '').strip()
                    ratio = row.get('ratio')
                    cash = row.get('cash')
                    ratio_f = float(ratio) if ratio not in (None, '') else None
                    cash_f = float(cash) if cash not in (None, '') else None
                    ev = CorporateActionEvent(date=date, action_type=atype, ratio=ratio_f, cash=cash_f, note=row.get('note'))
                    events.append(ev)
                    if logger:
                        logger.info(f"[ca_event] date={date} type={atype} ratio={ratio_f} cash={cash_f}")
                except Exception:
                    continue
        events.sort(key=lambda e: e.date)
        if logger:
            logger.info(f"[ca_load] file={path} events={len(events)}")
    except Exception as e:
        if logger:
            logger.info(f"[ca_load_error] file={path} err={type(e).__name__}:{e}")
    return events


def apply_event(event: CorporateActionEvent, position, broker, logger=None):
    """将单条公司行为应用于当前持仓。

    :param event: 公司行为事件对象
    :param position: Backtrader 持仓对象，需具备 size/price 等属性
    :param broker: Backtrader 经纪商实例，用于现金调整
    :param logger: 可选日志记录器
    """
    try:
        before = int(getattr(position, 'size', 0) or 0)
        if event.action_type == 'BONUS':
            if event.ratio and event.ratio > 0 and before > 0:
                add_sh = int(before * event.ratio)
                after = before + add_sh
                # Backtrader 不允许直接修改 position.size，因此通过返回差值提示引擎生成等额买单。
                # 引擎接收到 ('BONUS_SHARES', delta) 后自行下单，保证账面一致。
                if logger:
                    logger.info(f"[ca_apply] date={event.date} type=BONUS before={before} after={after} delta={add_sh}")
                return ('BONUS_SHARES', add_sh)
        elif event.action_type == 'SPLIT':
            if event.ratio and event.ratio > 0 and event.ratio != 1 and before > 0:
                # 与送转逻辑类似，返回拆分后应调整的持仓量，由引擎执行实际操作。
                new_sh = int(before * event.ratio)
                delta = new_sh - before
                if logger:
                    logger.info(f"[ca_apply] date={event.date} type=SPLIT before={before} after={new_sh} delta={delta}")
                return ('SPLIT_ADJ', delta)
        elif event.action_type == 'CASH_DIVIDEND':
            if event.cash and event.cash > 0 and before > 0:
                amount = before * event.cash
                try:
                    broker.add_cash(amount)
                except Exception:
                    # 若经纪商不允许直接加现金则忽略，避免报错中断。
                    pass
                if logger:
                    logger.info(f"[ca_apply] date={event.date} type=CASH_DIVIDEND shares={before} cash_received={amount:.2f}")
                return ('CASH_DIVIDEND', amount)
    except Exception as _e:
        if logger:
            logger.info(f"[ca_apply_error] date={event.date} err={type(_e).__name__}:{_e}")
    return None
