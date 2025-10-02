import os, sys, re, json, argparse, importlib.util, hashlib, textwrap
from dataclasses import asdict
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.backtest_engine import run_backtest

STRATEGY_DIR = os.path.join(ROOT, 'strategies')
BASELINE_DIR = os.path.join(ROOT, 'baselines')
DEFAULTS = dict(start='2018-01-01', end='2020-01-01', cash=100000, benchmark='000300', frequency='daily', adjust='auto')

META_PATTERN = re.compile(r"^#\s*---(.*?)#\s*---", re.DOTALL | re.MULTILINE)
KV_PATTERN = re.compile(r"^#\s*([A-Za-z_]+)\s*:\s*(.+)$", re.MULTILINE)


def parse_meta(text: str) -> Dict[str, str]:
    m = META_PATTERN.search(text)
    if not m:
        return {}
    block = m.group(1)
    meta = {}
    for kv in KV_PATTERN.finditer(block):
        key = kv.group(1).strip().lower()
        val = kv.group(2).strip()
        meta[key] = val
    return meta


def load_strategy_file(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    meta = parse_meta(content)
    name = meta.get('name') or os.path.splitext(os.path.basename(path))[0]
    # 动态加载 python 文件，取 strategy_code 或 UserStrategy 源码文本
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    strategy_code = getattr(module, 'strategy_code', None)
    if strategy_code is None:
        # 备选：若定义 UserStrategy 类，则将文件全文作为 source 传入
        if 'UserStrategy' in dir(module):
            strategy_code = content
        else:
            raise RuntimeError(f'{path} 未找到 strategy_code 或 UserStrategy')
    cfg = dict(DEFAULTS)
    for k in ['symbol','start','end','cash','benchmark','frequency','adjust']:
        if k in meta:
            cfg[k] = meta[k]
    # 整理类型
    cfg['cash'] = float(cfg.get('cash', DEFAULTS['cash']))
    return dict(name=name, code=strategy_code, config=cfg, file=path)


def run_one(entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = entry['config']
    res = run_backtest(
        symbol=cfg['symbol'],
        start=cfg['start'],
        end=cfg['end'],
        cash=cfg['cash'],
        strategy_code=entry['code'],
        strategy_params=None,
        benchmark_symbol=cfg.get('benchmark'),
        frequency=cfg.get('frequency','daily'),
        adjust_type=cfg.get('adjust','auto'),
    )
    d = asdict(res)
    metrics = d['metrics']
    # 只保留核心可回归字段
    core = {k: metrics.get(k) for k in [
        'final_value','pnl_pct','max_drawdown','total_trades','win_rate','benchmark_return','excess_return','use_real_price','data_variant'
    ]}
    core['equity_hash'] = hashlib.md5(json.dumps(d['equity_curve'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return dict(name=entry['name'], metrics=core, raw=d)


def load_baseline(name: str) -> Dict[str, Any] | None:
    path = os.path.join(BASELINE_DIR, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)


def save_baseline(name: str, metrics: Dict[str, Any]):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    path = os.path.join(BASELINE_DIR, f'{name}.json')
    with open(path,'w',encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)


def diff_metrics(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    keys = sorted(set(old.keys()) | set(new.keys()))
    for k in keys:
        if old.get(k) != new.get(k):
            out[k] = {'baseline': old.get(k), 'current': new.get(k)}
    return out


def main():
    ap = argparse.ArgumentParser(description='批量回归回测')
    ap.add_argument('--pattern', help='只匹配策略名子串/正则', default=None)
    ap.add_argument('--update-baseline', action='store_true', help='更新基准 (无 diff)')
    ap.add_argument('--fail-on-diff', action='store_true', help='出现差异时退出码!=0')
    ap.add_argument('--list', action='store_true', help='仅列出策略')
    args = ap.parse_args()

    if not os.path.isdir(STRATEGY_DIR):
        print(f'[ERROR] strategies 目录不存在: {STRATEGY_DIR}')
        return 2

    # 收集策略文件
    files = [os.path.join(STRATEGY_DIR,f) for f in os.listdir(STRATEGY_DIR) if f.endswith('.py')]
    entries = []
    for fp in files:
        try:
            entry = load_strategy_file(fp)
            if args.pattern:
                if re.search(args.pattern, entry['name']) is None:
                    continue
            entries.append(entry)
        except Exception as e:
            print(f'[LOAD_ERROR] {fp}: {e}')

    if args.list:
        for e in entries:
            cfg = e['config']
            print(f"{e['name']} -> {cfg['symbol']} {cfg['start']}~{cfg['end']} freq={cfg.get('frequency')} adjust={cfg.get('adjust')}")
        print(f'Total strategies: {len(entries)}')
        return 0

    if not entries:
        print('[WARN] 没有匹配的策略文件')
        return 0

    os.makedirs(BASELINE_DIR, exist_ok=True)
    summary = []
    exit_code = 0

    for e in entries:
        print(f'== Running {e['name']} ==')
        result = run_one(e)
        new_metrics = result['metrics']
        baseline = load_baseline(e['name'])
        if baseline is None:
            print(f'  [BASELINE_MISSING] {e['name']} -> 将创建新的 baseline')
            created = False
            if args.update_baseline:
                save_baseline(e['name'], new_metrics)
                print('  [BASELINE_SAVED]')
                created = True
            else:
                print('  使用 --update-baseline 保存基准')
            summary.append({'name': e['name'], 'status': ('CREATED' if created else 'NO_BASELINE')})
            continue
        diff = diff_metrics(baseline, new_metrics)
        if diff:
            print(f'  [DIFF] {e['name']} 有 {len(diff)} 项差异:')
            for k,v in diff.items():
                print(f"    - {k}: baseline={v['baseline']} current={v['current']}")
            if args.update_baseline:
                save_baseline(e['name'], new_metrics)
                print('  [BASELINE_UPDATED]')
            else:
                summary.append({'name': e['name'], 'status': 'DIFF', 'diff_keys': list(diff.keys())})
                if args.fail_on_diff:
                    exit_code = 1
        else:
            print(f'  [OK] {e['name']} 无差异')
            summary.append({'name': e['name'], 'status': 'OK'})

    print('\n== SUMMARY ==')
    for s in summary:
        if s['status'] == 'DIFF':
            print(f"{s['name']}: DIFF -> {','.join(s.get('diff_keys', []))}")
        else:
            print(f"{s['name']}: {s['status']}")
    print(f"Total={len(summary)} OK={sum(1 for x in summary if x['status']=='OK')} DIFF={sum(1 for x in summary if x['status']=='DIFF')} CREATED={sum(1 for x in summary if x['status']=='CREATED')} NO_BASELINE={sum(1 for x in summary if x['status']=='NO_BASELINE')}")

    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
