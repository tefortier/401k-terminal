"""
401k Dual Momentum — Period Optimization
=========================================
Tests all meaningful lookback period combinations against real historical data.
Weights fixed at baseline 30/40/30 (validated against ETFReplay).

Run locally:  python scripts/optimize_periods.py
Requires:     pip install yfinance numpy pandas
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Run: pip install yfinance numpy pandas")

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ['AMFFX','IWB','SHY','VBMFX','VEIEX','VGTSX',
           'VINIX','VISGX','VMFXX','VSNGX','VUSTX','VWEHX']

W_A    = 0.30   # RetA weight  (keep at baseline)
W_B    = 0.40   # RetB weight
W_VOL  = 0.30   # Volatility weight

TOP_N  = 2      # Funds held per month
START  = '2006-01-01'

# Periods to test (months)
RETA_PERIODS = [1, 2, 3, 6, 9, 12]   # longer lookback
RETB_PERIODS = [1, 2, 3, 6, 9]       # shorter lookback
VOL_PERIODS  = [1, 3, 6]             # volatility window

BASELINE = (6, 3, 3)   # current ETFReplay setting


# ── Download ──────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print("  401k Momentum — Period Optimization")
print(f"  Weights fixed: RetA={W_A*100:.0f}%  RetB={W_B*100:.0f}%  Vol={W_VOL*100:.0f}%")
print(f"{'='*68}\n")
print("Downloading price data (Yahoo Finance)...")

raw = yf.download(TICKERS, start=START, interval='1mo',
                  progress=False, auto_adjust=True)

# Handle both single and multi-level column structures
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Close']
else:
    prices = raw[['Close']] if 'Close' in raw.columns else raw

prices = prices.dropna(how='all').ffill().dropna()
monthly_rets = prices.pct_change()

loaded = list(prices.columns)
missing = [t for t in TICKERS if t not in loaded]
print(f"Loaded {len(loaded)}/{len(TICKERS)} funds | "
      f"{prices.index[0].strftime('%Y-%m')} → {prices.index[-1].strftime('%Y-%m')}")
if missing:
    print(f"Missing: {missing}")
print()


# ── Backtest engine ───────────────────────────────────────────────────────────
def backtest(lb_a, lb_b, lb_vol):
    """
    Monthly rotation: score all funds, hold equal-weight top 2.
    Score = RetA * W_A + RetB * W_B - Volatility * W_VOL
    """
    min_lb = max(lb_a, lb_b, lb_vol) + 2
    n = len(prices)
    port_rets = []

    for t in range(min_lb, n):
        scores = {}
        for tk in loaded:
            try:
                if (t - 1 - lb_a) < 0 or (t - 1 - lb_b) < 0:
                    continue
                ret_a = prices[tk].iloc[t-1] / prices[tk].iloc[t-1-lb_a] - 1
                ret_b = prices[tk].iloc[t-1] / prices[tk].iloc[t-1-lb_b] - 1
                vol_s = monthly_rets[tk].iloc[t-lb_vol:t].dropna()
                if len(vol_s) < 2:
                    continue
                vol   = vol_s.std() * np.sqrt(12)
                scores[tk] = ret_a * W_A + ret_b * W_B - vol * W_VOL
            except Exception:
                continue

        if len(scores) < TOP_N:
            continue

        top_funds = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
        month_ret = np.nanmean([monthly_rets[tk].iloc[t] for tk in top_funds])
        if not np.isnan(month_ret):
            port_rets.append(month_ret)

    if len(port_rets) < 24:
        return None

    p   = np.array(port_rets)
    cum = np.cumprod(1 + p)
    dd  = (cum / np.maximum.accumulate(cum) - 1).min()
    yrs = len(p) / 12

    cagr   = cum[-1] ** (1 / yrs) - 1
    sharpe = (p.mean() * 12) / (p.std() * np.sqrt(12)) if p.std() > 0 else 0
    calmar = cagr / abs(dd) if dd < 0 else 0
    win_rt = (p > 0).mean()

    return dict(cagr=cagr, max_dd=dd, sharpe=sharpe,
                calmar=calmar, win_rate=win_rt, n=len(p))


# ── Run all combinations ──────────────────────────────────────────────────────
results = []
combos  = [
    (la, lb, lv)
    for la in RETA_PERIODS
    for lb in RETB_PERIODS
    for lv in VOL_PERIODS
    if lb < la                     # RetB must be shorter than RetA
]

print(f"Testing {len(combos)} period combinations...", end='', flush=True)
for la, lb, lv in combos:
    r = backtest(la, lb, lv)
    if r:
        results.append({'RetA(mo)': la, 'RetB(mo)': lb, 'Vol(mo)': lv, **r})
print(f" done.\n")

df = pd.DataFrame(results)

# ── Results: Top 20 by Sharpe ─────────────────────────────────────────────────
print(f"{'='*68}")
print(f"  TOP 20 by Sharpe Ratio  (weights: {W_A*100:.0f}/{W_B*100:.0f}/{W_VOL*100:.0f})")
print(f"{'='*68}")
hdr = f"{'RetA':>6} {'RetB':>6} {'Vol':>5}  {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'WinRate':>8}"
print(hdr)
print('─' * 68)

top20 = df.nlargest(20, 'sharpe')
for _, row in top20.iterrows():
    la, lb, lv = int(row['RetA(mo)']), int(row['RetB(mo)']), int(row['Vol(mo)'])
    marker = '  ← baseline' if (la, lb, lv) == BASELINE else ''
    if (la, lb, lv) == (int(top20.iloc[0]['RetA(mo)']),
                        int(top20.iloc[0]['RetB(mo)']),
                        int(top20.iloc[0]['Vol(mo)'])):
        marker = '  ← BEST'
    print(f"{la:>6}mo {lb:>4}mo {lv:>3}mo  "
          f"{row['cagr']*100:>+7.2f}% {row['max_dd']*100:>7.2f}%"
          f"  {row['sharpe']:>7.3f}  {row['calmar']:>7.3f}  {row['win_rate']*100:>6.1f}%"
          f"{marker}")

# ── Results: Top 20 by Calmar ─────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  TOP 20 by Calmar Ratio  (return per unit of drawdown)")
print(f"{'='*68}")
print(hdr)
print('─' * 68)

top20c = df.nlargest(20, 'calmar')
for _, row in top20c.iterrows():
    la, lb, lv = int(row['RetA(mo)']), int(row['RetB(mo)']), int(row['Vol(mo)'])
    marker = '  ← baseline' if (la, lb, lv) == BASELINE else ''
    if (la, lb, lv) == (int(top20c.iloc[0]['RetA(mo)']),
                        int(top20c.iloc[0]['RetB(mo)']),
                        int(top20c.iloc[0]['Vol(mo)'])):
        marker = '  ← BEST'
    print(f"{la:>6}mo {lb:>4}mo {lv:>3}mo  "
          f"{row['cagr']*100:>+7.2f}% {row['max_dd']*100:>7.2f}%"
          f"  {row['sharpe']:>7.3f}  {row['calmar']:>7.3f}  {row['win_rate']*100:>6.1f}%"
          f"{marker}")

# ── Baseline comparison ───────────────────────────────────────────────────────
print(f"\n{'='*68}")
print("  Baseline vs Best")
print(f"{'='*68}")
base_row = df[(df['RetA(mo)'] == BASELINE[0]) &
              (df['RetB(mo)'] == BASELINE[1]) &
              (df['Vol(mo)']  == BASELINE[2])]

if not base_row.empty:
    b = base_row.iloc[0]
    best_sh  = df.loc[df['sharpe'].idxmax()]
    best_cal = df.loc[df['calmar'].idxmax()]
    best_cagr= df.loc[df['cagr'].idxmax()]

    rows = [
        ('Baseline (ETFReplay default)', BASELINE[0], BASELINE[1], BASELINE[2], b),
        (f'Best Sharpe',  int(best_sh['RetA(mo)']),  int(best_sh['RetB(mo)']),  int(best_sh['Vol(mo)']),  best_sh),
        (f'Best Calmar',  int(best_cal['RetA(mo)']), int(best_cal['RetB(mo)']), int(best_cal['Vol(mo)']), best_cal),
        (f'Best CAGR',    int(best_cagr['RetA(mo)']),int(best_cagr['RetB(mo)']),int(best_cagr['Vol(mo)']),best_cagr),
    ]
    print(f"\n  {'Label':<32} {'Periods':>12} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print(f"  {'─'*32} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for label, la, lb, lv, row in rows:
        print(f"  {label:<32} {la}M/{lb}M/{lv}M vol  "
              f"{row['cagr']*100:>+7.2f}%  {row['max_dd']*100:>7.2f}%  "
              f"{row['sharpe']:>7.3f}  {row['calmar']:>7.3f}")

# ── Heatmap: Sharpe by RetA vs RetB (best vol period per combo) ───────────────
print(f"\n{'='*68}")
print("  Sharpe Heatmap — RetA vs RetB  (best vol period per cell)")
print(f"{'='*68}")

best_per_ab = (df.groupby(['RetA(mo)', 'RetB(mo)'])
                 .apply(lambda g: g.loc[g['sharpe'].idxmax()])
                 .reset_index(drop=True))
pivot = best_per_ab.pivot(index='RetA(mo)', columns='RetB(mo)', values='sharpe')
pivot.index.name = 'RetA \\ RetB'
print(pivot.round(3).to_string())

print(f"\n{'='*68}")
print("  Done. Verify top settings on ETFReplay before updating the terminal.")
print(f"{'='*68}\n")
