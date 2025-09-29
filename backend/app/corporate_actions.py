"""Corporate actions framework (simplified) for bonus shares / splits / cash dividends.

File Format (CSV expected example placed under data/CA_<symbol>.csv):
    date,action_type,ratio, cash, note
    2024-07-26,BONUS,0.01,,10转1 示例: ratio=0.10 表示送(转)10股配1股 -> new_shares = pos_size * ratio
    2024-08-15,CASH_DIVIDEND,,0.25,每股派0.25元
    2024-09-10,SPLIT,0.5,,每1股拆为2股 -> ratio=2.0 (or reverse split <1)

Columns:
    date: ISO date (YYYY-MM-DD)
    action_type: BONUS | SPLIT | CASH_DIVIDEND
    ratio: (float) For BONUS: bonus_ratio per existing share; For SPLIT: new_per_old (e.g., 2.0 means 1 -> 2) ; unused for CASH_DIVIDEND
    cash: (float) per-share cash dividend (only for CASH_DIVIDEND)
    note: optional free text

Simplifications:
  - BONUS: Increase share count, no immediate cash impact.
  - SPLIT: Adjust share count by ratio (pos_size *= ratio). Price back-adjustment is assumed already in data (qfq/hfq). If not, this will cause a jump.
  - CASH_DIVIDEND: Record cash received (adds to broker cash) without tax; user can extend.

Logging Tags:
  [ca_load] file= path events= n
  [ca_event] date= type= ratio= cash= parsed
  [ca_apply] date= type= before= after= delta= reason=...

Extensibility: add new action types by extending apply_event.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import csv
import os

@dataclass
class CorporateActionEvent:
    date: str               # YYYY-MM-DD
    action_type: str        # BONUS | SPLIT | CASH_DIVIDEND
    ratio: Optional[float] = None  # used for BONUS (bonus_ratio) / SPLIT (new_per_old)
    cash: Optional[float] = None   # per-share cash dividend
    note: Optional[str] = None

SUPPORTED_TYPES = {"BONUS", "SPLIT", "CASH_DIVIDEND"}


def load_corporate_actions(symbol: str, datadir: str, logger=None) -> List[CorporateActionEvent]:
    """Load corporate actions from CSV: priority file names:
        CA_<symbol>.csv, <symbol>_CA.csv
    Returns list sorted by date.
    """
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
    """Apply a single corporate action event to current position.
    position: backtrader position object (has size / price attributes)
    broker: backtrader broker (for cash adjustments)
    """
    try:
        before = int(getattr(position, 'size', 0) or 0)
        if event.action_type == 'BONUS':
            if event.ratio and event.ratio > 0 and before > 0:
                add_sh = int(before * event.ratio)
                after = before + add_sh
                # We cannot directly mutate position.size (managed internally). Instead, we emulate via cash-neutral buy() order.
                # Caller should interpret a synthetic buy order injection; here we just return delta for engine to place order.
                if logger:
                    logger.info(f"[ca_apply] date={event.date} type=BONUS before={before} after={after} delta={add_sh}")
                return ('BONUS_SHARES', add_sh)
        elif event.action_type == 'SPLIT':
            if event.ratio and event.ratio > 0 and event.ratio != 1 and before > 0:
                # Similar limitation as above; we signal required adjustment.
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
                    # Fallback: ignore if broker disallows direct add
                    pass
                if logger:
                    logger.info(f"[ca_apply] date={event.date} type=CASH_DIVIDEND shares={before} cash_received={amount:.2f}")
                return ('CASH_DIVIDEND', amount)
    except Exception as _e:
        if logger:
            logger.info(f"[ca_apply_error] date={event.date} err={type(_e).__name__}:{_e}")
    return None
