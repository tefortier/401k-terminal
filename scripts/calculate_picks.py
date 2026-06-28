"""
401k Momentum Terminal — Monthly Picks Calculator
Runs via GitHub Actions on the 1st of each month.
Fetches price data, calculates dual momentum scores, writes data.json.

Scoring: (6M Return × 40%) + (3M Return × 40%) − (3M Volatility × 20%)
Cash proxy: SHY (iShares 1-3 Year Treasury Bond ETF)

Weights validated via ETFReplay backtest 2006-2026:
  CAGR +11.9% | Sharpe 0.79 | Max DD -17.3% vs benchmark -35.4%
"""

import json
import sys
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance numpy", file=sys.stderr)
    sys.exit(1)

# ── Universe ──────────────────────────────────────────────────────────────────
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
W_RETURN_6M  = 0.40
W_RETURN_3M  = 0.40
W_VOLATILITY = 0.20
LOOKBACK_DAYS = 300   # ~10 months of buffer for monthly data


# ── Calculation ───────────────────────────────────────────────────────────────
def calculate_metrics(ticker):
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)

    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1mo",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"  WARN {ticker}: download failed — {e}", file=sys.stderr)
        return None

    if raw.empty:
        print(f"  WARN {ticker}: no data returned", file=sys.stderr)
        return None

    closes = raw["Close"].dropna()
    n = len(closes)

    if n < 4:
        print(f"  WARN {ticker}: only {n} months of data (need ≥4)", file=sys.stderr)
        return None

    latest   = float(closes.iloc[-1])
    price_6m = float(closes.iloc[max(0, n - 7)])
    price_3m = float(closes.iloc[max(0, n - 4)])

    ret_6m = (latest - price_6m) / price_6m
    ret_3m = (latest - price_3m) / price_3m

    # Annualised 3-month volatility (std dev of last 3 monthly returns × √12)
    monthly_rets = closes.pct_change().dropna().iloc[-3:]
    vol_3m = float(monthly_rets.std() * np.sqrt(12)) if len(monthly_rets) >= 2 else 0.10

    score = ret_6m * W_RETURN_6M + ret_3m * W_RETURN_3M - vol_3m * W_VOLATILITY

    return {
        "ret_6m":  round(ret_6m,  6),
        "ret_3m":  round(ret_3m,  6),
        "vol_3m":  round(vol_3m,  6),
        "score":   round(score,   6),
        "latest_price": round(latest, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Running 401k momentum model — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Universe: {len(FUNDS)} funds | Weights: 6M={W_RETURN_6M} 3M={W_RETURN_3M} Vol={W_VOLATILITY}\n")

    results = []
    errors  = []

    for fund in FUNDS:
        print(f"  Fetching {fund['ticker']}...", end=" ")
        metrics = calculate_metrics(fund["ticker"])
        if metrics:
            results.append({**fund, **metrics})
            print(f"score={metrics['score']:+.4f}")
        else:
            errors.append(fund["ticker"])
            print("FAILED")

    if not results:
        print("\nERROR: No fund data retrieved. Check network / ticker symbols.", file=sys.stderr)
        sys.exit(1)

    # Rank by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    cash     = next((r for r in results if r["ticker"] == CASH_TICKER), None)
    cash_score = cash["score"] if cash else 0.0
    picks    = results[:2]
    above    = sum(1 for r in results if r["score"] > cash_score)

    now = datetime.now()
    output = {
        "updated":         now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_display": now.strftime("%B %d, %Y"),
        "month":           now.strftime("%B %Y"),
        "scoring": {
            "ret_6m_weight":  W_RETURN_6M,
            "ret_3m_weight":  W_RETURN_3M,
            "vol_weight":     W_VOLATILITY,
            "cash_proxy":     CASH_TICKER,
        },
        "picks": [
            {"rank": p["rank"], "ticker": p["ticker"], "name": p["name"],
             "category": p["category"], "score": p["score"]}
            for p in picks
        ],
        "rankings": results,
        "cash_score":       round(cash_score, 6),
        "funds_above_cash": above,
        "errors":           errors,
    }

    with open("data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ data.json written")
    print(f"  Top picks: #{picks[0]['rank']} {picks[0]['ticker']} ({picks[0]['score']:+.4f})", end="")
    if len(picks) > 1:
        print(f"  |  #{picks[1]['rank']} {picks[1]['ticker']} ({picks[1]['score']:+.4f})")
    if errors:
        print(f"  Errors: {', '.join(errors)}")


if __name__ == "__main__":
    main()
