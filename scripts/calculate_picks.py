"""
401k Momentum Terminal — Monthly Picks Calculator + Backtest Builder
Runs via GitHub Actions on the 1st of each month.

Scoring:  (6M Return × 30%) + (3M Return × 40%) - (Volatility × 30%)
Vol method: daily std × sqrt(252) over trailing 63 trading days (matches ETFReplay)

Execution methodology:
  - SCORE at close of last trading day of month M
  - BUY   at open  of first trading day of month M+1
  - SELL  at close of last  trading day of month M+1
  - RETURN = last_day_close(M+1) / first_day_open(M+1) - 1

Cash proxy: SHY | Top 2 funds held equal-weight | No vol targeting overlay
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


# ── Download & resample ───────────────────────────────────────────────────────
def download_daily(tickers, start):
    print(f"  Downloading {len(tickers)} tickers (daily OHLC)...", flush=True)
    raw = yf.download(tickers, start=start, interval="1d",
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].ffill(limit=5)
        open_ = raw["Open"].ffill(limit=5)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}).ffill(limit=5) \
                if len(tickers) == 1 else raw.ffill(limit=5)
        open_ = raw[["Open"]].rename(columns={"Open": tickers[0]}).ffill(limit=5) \
                if len(tickers) == 1 else raw.ffill(limit=5)
    return close, open_


def to_month_end_close(close_prices):
    """Last trading day CLOSE of each month — for scoring lookbacks and sell prices."""
    try:
        out = close_prices.resample('ME').last()
    except ValueError:
        out = close_prices.resample('M').last()
    return out.ffill(limit=2)


def to_month_start_open(open_prices):
    """First trading day OPEN of each month — buy price."""
    try:
        out = open_prices.resample('MS').first()
    except ValueError:
        out = open_prices.resample('BMS').first()
    return out.ffill(limit=2)


# ── Vol helper ────────────────────────────────────────────────────────────────
def daily_vol(daily_rets, ticker, as_of_date, n=63):
    """std(daily returns) × sqrt(252) over trailing n trading days. Matches ETFReplay."""
    if ticker not in daily_rets.columns:
        return 0.10
    series = daily_rets[ticker][daily_rets.index <= as_of_date].dropna().iloc[-n:]
    if len(series) < 20:
        return 0.10
    return float(series.std() * np.sqrt(252))


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
def run_model_backtest(me_close, fd_open, daily_close, fund_tickers):
    """
    Score at month-end close. Hold from first-day open to last-day close.
    me_close and fd_open both indexed monthly (one row per month).
    me_close.iloc[t]  = last trading day close of month t  (sell price / scoring price)
    fd_open.iloc[t]   = first trading day open of month t  (buy price)
    Return(t) = me_close.iloc[t] / fd_open.iloc[t] - 1
    """
    d_rets = daily_close[[tk for tk in fund_tickers
                          if tk in daily_close.columns]].pct_change()

    avail = [tk for tk in fund_tickers
             if tk in me_close.columns and tk in fd_open.columns]

    # Align fd_open to me_close by position (both monthly, one row per month)
    # Trim fd_open to match me_close length if needed
    fd_aligned = fd_open[avail].iloc[:len(me_close)].values  # numpy array for fast lookup

    dates, index_vals, monthly_rets = [], [], []
    cum = 100.0

    for t in range(1, len(me_close)):
        date_str = me_close.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START:
            continue

        as_of  = me_close.index[t - 1]   # score at end of previous month
        scores = {}

        for tk in avail:
            try:
                if t < 7:
                    continue
                p0 = me_close[tk].iloc[t - 1]   # current month-end close (score date)
                p6 = me_close[tk].iloc[t - 7]   # 6 months ago month-end close
                p3 = me_close[tk].iloc[t - 4]   # 3 months ago month-end close
                if any(pd.isna(x) for x in [p0, p6, p3]):
                    continue
                r6  = (p0 - p6) / p6
                r3  = (p0 - p3) / p3
                vol = daily_vol(d_rets, tk, as_of)
                scores[tk] = r6 * W_6M + r3 * W_3M - vol * W_VOL
            except Exception:
                continue

        if not scores:
            dates.append(date_str)
            index_vals.append(round(cum, 2))
            monthly_rets.append(0.0)
            continue

        picks = sorted(scores, key=scores.get, reverse=True)[:TOP_N]

        # Return = last-day close / first-day open — 1 (same month t)
        rets = []
        for tk in picks:
            try:
                sell = me_close[tk].iloc[t]    # last trading day close of month t
                buy  = fd_open[tk].iloc[t]     # first trading day open of month t
                if pd.isna(sell) or pd.isna(buy) or buy == 0:
                    continue
                rets.append((sell - buy) / buy)
            except Exception:
                continue

        port_r = float(np.mean(rets)) if rets else 0.0
        cum   *= (1 + port_r)

        dates.append(date_str)
        index_vals.append(round(cum, 2))
        monthly_rets.append(port_r)

    return dates, index_vals, monthly_rets


# ── Benchmark backtest ────────────────────────────────────────────────────────
def run_benchmark(me_close, fd_open, components, model_dates):
    """Same open-to-close execution as model, monthly rebalanced."""
    date_set = set(model_dates)
    tickers  = [tk for tk, w in components
                if tk in me_close.columns and tk in fd_open.columns]
    if not tickers:
        return []

    weights = {tk: w for tk, w in components if tk in tickers}
    total_w = sum(weights.values())
    weights = {tk: w / total_w for tk, w in weights.items()}

    index_vals = []
    cum        = 100.0

    for t in range(1, len(me_close)):
        date_str = me_close.index[t].strftime("%Y-%m")
        if date_str < BACKTEST_CHART_START or date_str not in date_set:
            continue
        r = 0.0
        for tk, w in weights.items():
            try:
                sell = me_close[tk].iloc[t]
                buy  = fd_open[tk].iloc[t]
                if pd.isna(sell) or pd.isna(buy) or buy == 0:
                    continue
                r += ((sell - buy) / buy) * w
            except Exception:
                continue
        cum *= (1 + r)
        index_vals.append(round(cum, 2))

    return index_vals


# ── Current picks ─────────────────────────────────────────────────────────────
def current_picks(me_close, daily_close):
    """Score all funds using latest month-end close prices."""
    fund_tickers = [f["ticker"] for f in FUNDS]
    d_rets       = daily_close[[tk for tk in fund_tickers
                                if tk in daily_close.columns]].pct_change()
    results, errors = [], []
    n     = len(me_close)
    as_of = me_close.index[-1]

    for fund in FUNDS:
        tk = fund["ticker"]
        if tk not in me_close.columns:
            errors.append(tk)
            continue
        try:
            p0 = float(me_close[tk].iloc[-1])
            p6 = float(me_close[tk].iloc[max(0, n - 7)])
            p3 = float(me_close[tk].iloc[max(0, n - 4)])
            if any(np.isnan(x) for x in [p0, p6, p3]):
                errors.append(tk)
                continue
            r6    = (p0 - p6) / p6
            r3    = (p0 - p3) / p3
            vol   = daily_vol(d_rets, tk, as_of)
            score = r6 * W_6M + r3 * W_3M - vol * W_VOL
            results.append({
                **fund,
                "ret_6m":       round(r6,    6),
                "ret_3m":       round(r3,    6),
                "vol_3m":       round(vol,   6),
                "score":        round(score, 6),
                "latest_price": round(p0,    4),
            })
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
    print(f"  6M×{W_6M:.0%} + 3M×{W_3M:.0%} - Vol×{W_VOL:.0%} | Top {TOP_N} | Cash: {CASH_TICKER}")
    print(f"  Execution: score month-end close | buy first-day open | sell last-day close")

    fund_tickers  = [f["ticker"] for f in FUNDS]
    bench_tickers = list({tk for b in BENCHMARKS.values() for tk, _ in b["components"]})
    all_tickers   = list(set(fund_tickers + bench_tickers))

    daily_close, daily_open = download_daily(all_tickers, BACKTEST_START)

    me_close = to_month_end_close(daily_close)    # last day close — score + sell
    fd_open  = to_month_start_open(daily_open)    # first day open — buy

    avail = [tk for tk in all_tickers if tk in me_close.columns]
    print(f"  {len(avail)}/{len(all_tickers)} tickers | "
          f"{me_close.index[0].strftime('%Y-%m')} → {me_close.index[-1].strftime('%Y-%m')}")

    # ── Current picks ──────────────────────────────────────────────────────────
    print("\nComputing current picks...")
    fund_me          = me_close[[tk for tk in fund_tickers if tk in me_close.columns]]
    rankings, errors = current_picks(fund_me, daily_close)

    cash_row   = next((r for r in rankings if r["ticker"] == CASH_TICKER), None)
    cash_score = cash_row["score"] if cash_row else 0.0
    picks      = rankings[:TOP_N]
    above      = sum(1 for r in rankings if r["score"] > cash_score)

    for p in picks:
        print(f"  #{p['rank']} {p['ticker']}  score={p['score']:+.4f}  "
              f"6M={p['ret_6m']*100:+.1f}%  3M={p['ret_3m']*100:+.1f}%  vol={p['vol_3m']*100:.1f}%")

    # ── Model backtest ─────────────────────────────────────────────────────────
    print("\nRunning model backtest...")
    avail_funds = [tk for tk in fund_tickers
                   if tk in me_close.columns and tk in fd_open.columns]
    model_dates, model_vals, model_monthly = run_model_backtest(
        me_close, fd_open, daily_close, avail_funds
    )
    model_stats = backtest_stats(model_monthly)
    print(f"  {len(model_dates)} months | CAGR {model_stats['cagr']*100:.1f}% | "
          f"Sharpe {model_stats['sharpe']:.2f} | MaxDD {model_stats['max_dd']*100:.1f}%")

    # ── Benchmark backtests ────────────────────────────────────────────────────
    bench_out = {}
    for key, bench in BENCHMARKS.items():
        print(f"  Benchmark {bench['name']}...", end=" ", flush=True)
        components = [(tk, w) for tk, w in bench["components"]
                      if tk in me_close.columns and tk in fd_open.columns]
        if not components:
            print("SKIPPED")
            continue
        vals    = run_benchmark(me_close, fd_open, components, model_dates)
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
            "ret_6m_weight": W_6M,
            "ret_3m_weight": W_3M,
            "vol_weight":    W_VOL,
            "top_n":         TOP_N,
            "cash_proxy":    CASH_TICKER,
        },
        "picks": [
            {"rank": p["rank"], "ticker": p["ticker"],
             "name": p["name"], "category": p["category"],
             "score": p["score"], "allocation": 1.0 / TOP_N}
            for p in picks
        ],
        "rankings":         rankings,
        "cash_score":       round(cash_score, 6),
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
    print(f"\n✓ data.json written ({size_kb:.1f} KB)")
    if errors:
        print(f"  Skipped: {', '.join(errors)}")


if __name__ == "__main__":
    main()
