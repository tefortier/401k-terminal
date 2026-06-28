"""
401k Momentum Terminal — Monthly Picks Calculator + Backtest Builder
Runs via GitHub Actions on the 1st of each month.

Scoring: (6M Return x 40%) + (3M Return x 40%) - (3M Volatility x 20%)
Cash proxy: SHY | Validated: CAGR +11.9%, Sharpe 0.79, MaxDD -17.3% (ETFReplay 2006-2026)
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
    {"ticker": "AMFFX", "name": "American Mutual Fund Growth & Income", "category": "U.S. Equity"},
    {"ticker": "IWB",   "name": "iShares Russell 1000 Index Fund",      "category": "U.S. Equity"},
    {"ticker": "SHY",   "name": "iShares 1-3 Year Treasury Bond ETF",   "category": "Treasury / Cash"},
    {"ticker": "VBMFX", "name": "Vanguard Total Bond Market",           "category": "Corp / Credit Bonds"},
    {"ticker": "VEIEX", "name": "Vanguard Emerging Markets MF",         "category": "Emerging Markets"},
    {"ticker": "VGTSX", "name": "Vanguard Total Int'l Stock Index MF",  "category": "International Equity"},
    {"ticker": "VINIX", "name": "Vanguard Instl Index Fund (S&P 500)",  "category": "U.S. Equity"},
    {"ticker": "VISGX", "name": "Vanguard Small Cap Growth Index Inv",  "category": "U.S. Equity"},
    {"ticker": "VMFXX", "name": "Vanguard Money Market Mutual Fund",    "category": "Treasury / Cash"},
    {"ticker": "VSNGX", "name": "JP Morgan Mid Cap Equity Instl",       "category": "U.S. Equity"},
    {"ticker": "VUSTX", "name": "Vanguard Long-Term Treasury",          "category": "Treasury"},
    {"ticker": "VWEHX", "name": "Vanguard High-Yield Bond MF",          "category": "Corp / Credit Bonds"},
]

CASH_TICKER  = "SHY"
W_6M  = 0.40
W_3M  = 0.40
W_VOL = 0.20
TOP_N = 2

# ── Benchmark definitions ─────────────────────────────────────────────────────
BENCHMARKS = {
    "allworld6040": {
        "name": "All-World 60/40",
        "description": "IEF 40% / EFA 30% / VTI 30%, monthly rebalanced",
        "components": [("IEF", 0.40), ("EFA", 0.30), ("VTI", 0.30)],
    },
    "spy": {
        "name": "S&P 500 (SPY)",
        "description": "SPDR S&P 500 ETF",
        "components": [("SPY", 1.0)],
    },
    "agg": {
        "name": "US Bonds (AGG)",
        "description": "iShares Core US Aggregate Bond ETF",
        "components": [("AGG", 1.0)],
    },
}

BACKTEST_START = "2005-01-01"   # extra year for 6M lookback warmup
BACKTEST_CHART_START = "2006-01"


# ── Helpers ───────────────────────────────────────────────────────────────────
def download_all(tickers, start):
    print(f"  Downloading {len(tickers)} tickers from {start}...", flush=True)
    raw = yf.download(tickers, start=start, interval="1mo",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.ffill(limit=2)
    return prices


def backtest_stats(monthly_rets):
    """Compute CAGR, Sharpe, MaxDD, total return from a monthly return series."""
    r = np.array(monthly_rets)
    cum = np.cumprod(1 + r)
    years = len(r) / 12
    cagr = float(cum[-1] ** (1 / years) - 1) if years > 0 else 0.0
    sharpe = float((r.mean() * 12) / (r.std() * np.sqrt(12))) if r.std() > 0 else 0.0
    drawdowns = cum / np.maximum.accumulate(cum) - 1
    max_dd = float(drawdowns.min())
    total = float(cum[-1] - 1)
    return dict(cagr=round(cagr, 4), sharpe=round(sharpe, 4),
                max_dd=round(max_dd, 4), total_return=round(total, 4))


def run_model_backtest(prices, fund_tickers):
    """
    Monthly rotation backtest: score funds at month-end, hold top 2 next month.
    Returns (dates, cumulative_index, monthly_returns).
    """
    rets = prices[fund_tickers].pct_change()
    dates, index_vals, monthly_rets = [], [], []
    cum = 100.0

    for t in range(1, len(prices)):
        date_str = prices.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START:
            continue

        # Score using prices known at end of month t-1
        scores = {}
        for tk in fund_tickers:
            try:
                if t < 7:
                    continue
                p0 = prices[tk].iloc[t - 1]
                p6 = prices[tk].iloc[t - 7]
                p3 = prices[tk].iloc[t - 4]
                if any(pd.isna(x) for x in [p0, p6, p3]):
                    continue
                r6 = (p0 - p6) / p6
                r3 = (p0 - p3) / p3
                vol_w = rets[tk].iloc[max(0, t-4):t-1].dropna()
                vol = float(vol_w.std() * np.sqrt(12)) if len(vol_w) >= 2 else 0.10
                scores[tk] = r6 * W_6M + r3 * W_3M - vol * W_VOL
            except Exception:
                continue

        if not scores:
            dates.append(date_str)
            index_vals.append(round(cum, 2))
            monthly_rets.append(0.0)
            continue

        # Top N by score (no cash filter — matches ETFReplay Cash Filter OFF setting)
        picks = sorted(scores, key=scores.get, reverse=True)[:TOP_N]

        # Return this month (month t)
        month_r = [rets[tk].iloc[t] for tk in picks
                   if not pd.isna(rets[tk].iloc[t])]
        port_r = float(np.mean(month_r)) if month_r else 0.0
        cum *= (1 + port_r)

        dates.append(date_str)
        index_vals.append(round(cum, 2))
        monthly_rets.append(port_r)

    return dates, index_vals, monthly_rets


def run_benchmark(prices, components, model_dates):
    """
    Compute monthly-rebalanced benchmark cumulative return.
    Aligned to model_dates.
    """
    date_set = set(model_dates)
    tickers = [t for t, w in components if t in prices.columns]
    if not tickers:
        return []

    weights = {t: w for t, w in components if t in prices.columns}
    # Renormalise weights if any component is missing
    total_w = sum(weights.values())
    weights = {t: w / total_w for t, w in weights.items()}

    rets = prices[tickers].pct_change()
    index_vals, cum = [], 100.0

    for t in range(1, len(prices)):
        date_str = prices.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START:
            continue
        if date_str not in date_set:
            continue

        r = sum(rets[tk].iloc[t] * w
                for tk, w in weights.items()
                if not pd.isna(rets[tk].iloc[t]))
        cum *= (1 + r)
        index_vals.append(round(cum, 2))

    return index_vals


# ── Current picks (today's scores) ───────────────────────────────────────────
def current_picks(prices):
    rets = prices.pct_change()
    results, errors = [], []
    n = len(prices)

    for fund in FUNDS:
        tk = fund["ticker"]
        if tk not in prices.columns:
            errors.append(tk)
            continue
        try:
            p0 = float(prices[tk].iloc[-1])
            p6 = float(prices[tk].iloc[max(0, n - 7)])
            p3 = float(prices[tk].iloc[max(0, n - 4)])
            if any(np.isnan(x) for x in [p0, p6, p3]):
                errors.append(tk)
                continue
            r6  = (p0 - p6) / p6
            r3  = (p0 - p3) / p3
            vol = float(rets[tk].dropna().iloc[-3:].std() * np.sqrt(12))
            score = r6 * W_6M + r3 * W_3M - vol * W_VOL
            results.append({**fund,
                            "ret_6m": round(r6, 6), "ret_3m": round(r3, 6),
                            "vol_3m": round(vol, 6), "score": round(score, 6),
                            "latest_price": round(p0, 4)})
        except Exception:
            errors.append(tk)

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results, errors


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now()
    print(f"\n401k momentum model — {now.strftime('%Y-%m-%d %H:%M')}")

    fund_tickers  = [f["ticker"] for f in FUNDS]
    bench_tickers = list({tk for b in BENCHMARKS.values()
                          for tk, _ in b["components"]})
    all_tickers   = list(set(fund_tickers + bench_tickers))

    prices = download_all(all_tickers, BACKTEST_START)
    avail  = [t for t in all_tickers if t in prices.columns]
    print(f"  Loaded {len(avail)}/{len(all_tickers)} tickers | "
          f"{prices.index[0].strftime('%Y-%m')} → {prices.index[-1].strftime('%Y-%m')}")

    # ── Current picks ──────────────────────────────────────────────────────────
    print("\nComputing current picks...")
    fund_prices = prices[[t for t in fund_tickers if t in prices.columns]]
    rankings, errors = current_picks(fund_prices)

    cash_row   = next((r for r in rankings if r["ticker"] == CASH_TICKER), None)
    cash_score = cash_row["score"] if cash_row else 0.0
    picks      = rankings[:TOP_N]
    above      = sum(1 for r in rankings if r["score"] > cash_score)

    for p in picks:
        print(f"  #{p['rank']} {p['ticker']} score={p['score']:+.4f}")

    # ── Model backtest ─────────────────────────────────────────────────────────
    print("\nRunning model backtest...")
    avail_funds = [t for t in fund_tickers if t in prices.columns]
    model_dates, model_vals, model_monthly = run_model_backtest(prices, avail_funds)
    model_stats = backtest_stats(model_monthly)
    print(f"  {len(model_dates)} months | CAGR {model_stats['cagr']*100:.1f}% "
          f"| Sharpe {model_stats['sharpe']:.2f} | MaxDD {model_stats['max_dd']*100:.1f}%")

    # ── Benchmark backtests ────────────────────────────────────────────────────
    bench_out = {}
    for key, bench in BENCHMARKS.items():
        print(f"Computing {bench['name']}...", end=" ", flush=True)
        components = [(t, w) for t, w in bench["components"] if t in prices.columns]
        if not components:
            print("SKIPPED (no data)")
            continue

        vals = run_benchmark(prices, components, model_dates)

        # Monthly returns from index for stats
        idx = np.array([100.0] + vals)
        b_rets = np.diff(idx) / idx[:-1]
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
            "ret_6m_weight": W_6M,
            "ret_3m_weight": W_3M,
            "vol_weight":    W_VOL,
            "cash_proxy":    CASH_TICKER,
        },
        "picks": [
            {"rank": p["rank"], "ticker": p["ticker"],
             "name": p["name"], "category": p["category"], "score": p["score"]}
            for p in picks
        ],
        "rankings":        rankings,
        "cash_score":      round(cash_score, 6),
        "funds_above_cash": above,
        "errors":          errors,
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
    print(f"\n✓ data.json written ({size_kb:.1f} KB)")
    if errors:
        print(f"  Errors: {', '.join(errors)}")


if __name__ == "__main__":
    main()
