"""
401k Momentum Terminal — Monthly Picks Calculator + Backtest Builder
Runs via GitHub Actions on the 1st of each month.

Scoring:  ETFReplay rank-based methodology
  - Rank all ETFs by 6M return (1 = highest)
  - Rank all ETFs by 3M return (1 = highest)
  - Rank all ETFs by 3M volatility (1 = lowest)
  - Weighted rank = rank_6M x 30% + rank_3M x 40% + rank_vol x 30%
  - Pick 2 ETFs with lowest weighted rank

Vol method:  std of trailing 3 monthly returns x sqrt(12)
Returns:     month-end close to month-end close (institutional standard)
Data source: Yahoo Finance monthly bars (auto-adjusted)
Cash proxy:  SHY | Top 2 funds held equal-weight
"""

import json
import sys
import numpy as np
from datetime import datetime

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    sys.exit("ERROR: Run: pip install yfinance numpy pandas")

# ── Fund universe ─────────────────────────────────────────────────────────────
FUNDS = [
    {"ticker": "SPY", "name": "S&P 500",                    "category": "U.S. Equity"},
    {"ticker": "IWF", "name": "U.S. Large Cap Growth",      "category": "U.S. Equity"},
    {"ticker": "IWD", "name": "U.S. Large Cap Value",       "category": "U.S. Equity"},
    {"ticker": "IWM", "name": "U.S. Small Cap",             "category": "U.S. Equity"},
    {"ticker": "IJH", "name": "U.S. Mid Cap",               "category": "U.S. Equity"},
    {"ticker": "EFA", "name": "Intl Developed Markets",     "category": "International Equity"},
    {"ticker": "EEM", "name": "Emerging Markets",           "category": "Emerging Markets"},
    {"ticker": "TLT", "name": "Long-Term Treasury",         "category": "Treasury"},
    {"ticker": "HYG", "name": "High Yield Bonds",           "category": "Corp / Credit Bonds"},
    {"ticker": "AGG", "name": "Total Bond Market",          "category": "Bonds"},
    {"ticker": "SHY", "name": "Short-Term Treasury (Cash)", "category": "Treasury / Cash"},
]

CASH_TICKER = "SHY"
W_6M        = 0.30
W_3M        = 0.40
W_VOL       = 0.30
TOP_N       = 2

# ── Benchmark definitions ─────────────────────────────────────────────────────
BENCHMARKS = {
    "allworld6040": {
        "name":        "All-World 60/40",
        "description": "IEF 40% / EFA 30% / VTI 30%, monthly rebalanced",
        "components":  [("IEF", 0.40), ("EFA", 0.30), ("VTI", 0.30)],
    },
    "spy": {
        "name":        "S&P 500 (SPY)",
        "description": "SPDR S&P 500 ETF",
        "components":  [("SPY", 1.0)],
    },
    "agg": {
        "name":        "US Bonds (AGG)",
        "description": "iShares Core US Aggregate Bond ETF",
        "components":  [("AGG", 1.0)],
    },
}

BACKTEST_START       = "2005-01-01"
BACKTEST_CHART_START = "2006-01"


# ── Download ──────────────────────────────────────────────────────────────────
def download_monthly(tickers, start):
    """Monthly adjusted-close bars from Yahoo Finance."""
    print(f"  Downloading {len(tickers)} tickers (monthly)...", flush=True)
    raw = yf.download(tickers, start=start, interval="1mo",
                      progress=False, auto_adjust=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.ffill(limit=2)


# ── Rank-based scoring (ETFReplay methodology) ────────────────────────────────
def rank_dict(values, ascending=False):
    """Rank a dict of {ticker: value}. ascending=True means lower value = rank 1."""
    tickers = list(values.keys())
    ranked  = sorted(tickers, key=lambda x: values[x], reverse=not ascending)
    return {tk: i + 1 for i, tk in enumerate(ranked)}


def weighted_ranks(monthly, t, fund_tickers):
    """
    Score all tickers at bar index t-1 (score date = end of previous month).
    Returns {ticker: weighted_rank}. Lower = better. Pick lowest.

    6M return : rank 1 = highest
    3M return : rank 1 = highest
    3M vol    : rank 1 = lowest (vol is a penalty)
    """
    if t < 7:
        return {}

    r6_raw, r3_raw, vol_raw = {}, {}, {}

    for tk in fund_tickers:
        if tk not in monthly.columns:
            continue
        try:
            p0 = monthly[tk].iloc[t - 1]
            p6 = monthly[tk].iloc[t - 7]
            p3 = monthly[tk].iloc[t - 4]
            if any(pd.isna(x) for x in [p0, p6, p3]):
                continue
            r6_raw[tk] = (p0 - p6) / p6
            r3_raw[tk] = (p0 - p3) / p3
            # 3-month vol: std of trailing 3 monthly returns x sqrt(12)
            window = monthly[tk].iloc[max(0, t - 4):t].pct_change().dropna()
            vol_raw[tk] = float(window.std() * np.sqrt(12)) if len(window) >= 2 else 0.10
        except Exception:
            continue

    common = set(r6_raw) & set(r3_raw) & set(vol_raw)
    if len(common) < 2:
        return {}

    rank_6m  = rank_dict({tk: r6_raw[tk]  for tk in common}, ascending=False)
    rank_3m  = rank_dict({tk: r3_raw[tk]  for tk in common}, ascending=False)
    rank_vol = rank_dict({tk: vol_raw[tk] for tk in common}, ascending=True)

    return {
        tk: rank_6m[tk] * W_6M + rank_3m[tk] * W_3M + rank_vol[tk] * W_VOL
        for tk in common
    }


# ── Stats ─────────────────────────────────────────────────────────────────────
def backtest_stats(monthly_rets):
    r      = np.array(monthly_rets)
    cum    = np.cumprod(1 + r)
    years  = len(r) / 12
    cagr   = float(cum[-1] ** (1 / years) - 1) if years > 0 else 0.0
    sharpe = float((r.mean() * 12) / (r.std() * np.sqrt(12))) if r.std() > 0 else 0.0
    max_dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    total  = float(cum[-1] - 1)
    return dict(cagr=round(cagr, 4), sharpe=round(sharpe, 4),
                max_dd=round(max_dd, 4), total_return=round(total, 4))


# ── Model backtest ────────────────────────────────────────────────────────────
def run_model_backtest(monthly, fund_tickers):
    """
    Score at month-end t-1, earn month-end-to-month-end return in period t.
    Rank-based scoring matches ETFReplay methodology.
    """
    m_rets = monthly[fund_tickers].pct_change()
    dates, index_vals, monthly_rets = [], [], []
    cum = 100.0

    for t in range(1, len(monthly)):
        date_str = monthly.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START:
            continue

        wr = weighted_ranks(monthly, t, fund_tickers)
        if not wr:
            dates.append(date_str)
            index_vals.append(round(cum, 2))
            monthly_rets.append(0.0)
            continue

        picks  = sorted(wr, key=wr.get)[:TOP_N]
        rets   = [m_rets[tk].iloc[t] for tk in picks
                  if not pd.isna(m_rets[tk].iloc[t])]
        port_r = float(np.mean(rets)) if rets else 0.0
        cum   *= (1 + port_r)

        dates.append(date_str)
        index_vals.append(round(cum, 2))
        monthly_rets.append(port_r)

    return dates, index_vals, monthly_rets


# ── Benchmark backtest ────────────────────────────────────────────────────────
def run_benchmark(monthly, components, model_dates):
    date_set = set(model_dates)
    tickers  = [tk for tk, w in components if tk in monthly.columns]
    if not tickers:
        return []

    weights = {tk: w for tk, w in components if tk in tickers}
    total_w = sum(weights.values())
    weights = {tk: w / total_w for tk, w in weights.items()}

    rets       = monthly[tickers].pct_change()
    index_vals = []
    cum        = 100.0

    for t in range(1, len(monthly)):
        date_str = monthly.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START or date_str not in date_set:
            continue
        r = sum(rets[tk].iloc[t] * w for tk, w in weights.items()
                if not pd.isna(rets[tk].iloc[t]))
        cum *= (1 + r)
        index_vals.append(round(cum, 2))

    return index_vals


# ── Current picks ─────────────────────────────────────────────────────────────
def current_picks(monthly):
    """Score using the latest complete month-end as the signal date."""
    fund_tickers = [f["ticker"] for f in FUNDS]
    t = len(monthly)

    wr = weighted_ranks(monthly, t, fund_tickers)
    if not wr:
        return [], ["insufficient data"]

    results = []
    for fund in FUNDS:
        tk = fund["ticker"]
        if tk not in wr:
            continue
        try:
            idx = t - 1
            p0  = float(monthly[tk].iloc[idx])
            p6  = float(monthly[tk].iloc[idx - 6])
            p3  = float(monthly[tk].iloc[idx - 3])
            r6  = (p0 - p6) / p6
            r3  = (p0 - p3) / p3
            win = monthly[tk].iloc[max(0, idx - 3):idx + 1].pct_change().dropna()
            vol = float(win.std() * np.sqrt(12)) if len(win) >= 2 else 0.10
            results.append({
                **fund,
                "ret_6m":        round(r6,  6),
                "ret_3m":        round(r3,  6),
                "vol_3m":        round(vol, 6),
                "weighted_rank": round(wr[tk], 4),
                "latest_price":  round(p0,  4),
                "score":         round(r6,  6),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["weighted_rank"])
    for i, r in enumerate(results):
        r["rank"] = i + 1

    errors = [f["ticker"] for f in FUNDS if f["ticker"] not in wr]
    return results, errors


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now()
    print(f"\n401k momentum model — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Rank-based | 6M x{W_6M:.0%} + 3M x{W_3M:.0%} - Vol x{W_VOL:.0%} | "
          f"Top {TOP_N} | Cash: {CASH_TICKER}")

    fund_tickers  = [f["ticker"] for f in FUNDS]
    bench_tickers = list({tk for b in BENCHMARKS.values() for tk, _ in b["components"]})
    all_tickers   = list(set(fund_tickers + bench_tickers))

    monthly = download_monthly(all_tickers, BACKTEST_START)

    avail = [tk for tk in all_tickers if tk in monthly.columns]
    print(f"  {len(avail)}/{len(all_tickers)} tickers | "
          f"{monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')}")

    # ── Current picks ──────────────────────────────────────────────────────────
    print("\nComputing current picks...")
    fund_monthly     = monthly[[tk for tk in fund_tickers if tk in monthly.columns]]
    rankings, errors = current_picks(fund_monthly)

    cash_row  = next((r for r in rankings if r["ticker"] == CASH_TICKER), None)
    cash_rank = cash_row["weighted_rank"] if cash_row else 99.0
    picks     = rankings[:TOP_N]
    above     = sum(1 for r in rankings if r["weighted_rank"] < cash_rank)

    for p in picks:
        print(f"  #{p['rank']} {p['ticker']}  wrank={p['weighted_rank']:.2f}  "
              f"6M={p['ret_6m']*100:+.1f}%  3M={p['ret_3m']*100:+.1f}%  "
              f"vol={p['vol_3m']*100:.1f}%")

    # ── Model backtest ─────────────────────────────────────────────────────────
    print("\nRunning model backtest...")
    avail_funds = [tk for tk in fund_tickers if tk in monthly.columns]
    model_dates, model_vals, model_monthly = run_model_backtest(monthly, avail_funds)
    model_stats = backtest_stats(model_monthly)
    print(f"  {len(model_dates)} months | CAGR {model_stats['cagr']*100:.1f}% | "
          f"Sharpe {model_stats['sharpe']:.2f} | MaxDD {model_stats['max_dd']*100:.1f}%")

    # ── Benchmark backtests ────────────────────────────────────────────────────
    bench_out = {}
    for key, bench in BENCHMARKS.items():
        print(f"  Benchmark {bench['name']}...", end=" ", flush=True)
        components = [(tk, w) for tk, w in bench["components"] if tk in monthly.columns]
        if not components:
            print("SKIPPED (tickers missing)")
            continue
        vals    = run_benchmark(monthly, components, model_dates)
        idx     = np.array([100.0] + vals)
        b_rets  = np.diff(idx) / idx[:-1]
        b_stats = backtest_stats(b_rets)
        bench_out[key] = {
            "name":        bench["name"],
            "description": bench["description"],
            "stats":       b_stats,
            "data":        vals,
        }
        print(f"CAGR {b_stats['cagr']*100:.1f}% | MaxDD {b_stats['max_dd']*100:.1f}%")

    # ── Assemble output ────────────────────────────────────────────────────────
    output = {
        "updated":         now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_display": now.strftime("%B %d, %Y"),
        "month":           now.strftime("%B %Y"),
        "scoring": {
            "method":        "rank-based",
            "ret_6m_weight": W_6M,
            "ret_3m_weight": W_3M,
            "vol_weight":    W_VOL,
            "top_n":         TOP_N,
            "cash_proxy":    CASH_TICKER,
        },
        "picks": [
            {"rank": p["rank"], "ticker": p["ticker"],
             "name": p["name"], "category": p["category"],
             "weighted_rank": p["weighted_rank"], "allocation": 1.0 / TOP_N}
            for p in picks
        ],
        "rankings":         rankings,
        "cash_rank":        round(cash_rank, 4),
        "funds_above_cash": above,
        "errors":           errors,
        "backtest": {
            "dates":       model_dates,
            "model":       model_vals,
            "model_stats": model_stats,
            "benchmarks":  bench_out,
        },
    }

    with open("data.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = len(json.dumps(output)) / 1024
    print(f"\n  data.json written ({size_kb:.1f} KB)")
    if errors:
        print(f"  Skipped: {', '.join(errors)}")


if __name__ == "__main__":
    main()
