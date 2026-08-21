from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

INPUT = Path("data/research/historical_price_download_failures.csv")
OUT = Path("data/research")
STOOQ = "https://stooq.com/q/d/l/?s={symbol}.us&d1=20170101&d2=20260630&i=d"


def stooq_symbol(symbol: str) -> str:
    return str(symbol).strip().lower().replace("-", ".")


def probe(symbol: str) -> dict:
    url = STOOQ.format(symbol=stooq_symbol(symbol))
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "regime-factor-investing research"},
            )
            status = r.status_code
            text = r.text.strip()
            if status == 200 and text and "No data" not in text:
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception:
                    df = pd.DataFrame()
                if not df.empty and "Date" in df.columns and "Close" in df.columns:
                    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
                    close = pd.to_numeric(df["Close"], errors="coerce")
                    valid = dates.notna() & close.notna()
                    if valid.any():
                        return {
                            "yahoo_symbol": symbol,
                            "stooq_symbol": stooq_symbol(symbol) + ".us",
                            "http_status": status,
                            "rows": int(valid.sum()),
                            "first_date": dates[valid].min().date().isoformat(),
                            "last_date": dates[valid].max().date().isoformat(),
                            "available": True,
                            "note": "coverage_only; adjustment semantics not yet approved for backtest",
                        }
            last = {"http_status": status, "body_prefix": text[:80]}
        except Exception as exc:
            last = {"http_status": None, "body_prefix": str(exc)[:80]}
        time.sleep(1.5 + attempt)
    return {
        "yahoo_symbol": symbol,
        "stooq_symbol": stooq_symbol(symbol) + ".us",
        "http_status": last.get("http_status"),
        "rows": 0,
        "first_date": "",
        "last_date": "",
        "available": False,
        "note": last.get("body_prefix", "unavailable"),
    }


def main() -> None:
    failures = pd.read_csv(INPUT)
    symbols = sorted(set(failures["yahoo_symbol"].dropna().astype(str)))
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        row = probe(symbol)
        rows.append(row)
        print(
            f"{i}/{len(symbols)} {symbol}: available={row['available']} rows={row['rows']}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "alternate_price_stooq_probe.csv", index=False)
    pd.DataFrame([{
        "yahoo_failures_tested": len(out),
        "stooq_available": int(out["available"].sum()),
        "stooq_coverage_of_yahoo_failures": float(out["available"].mean()) if len(out) else 0.0,
        "approved_for_backtest": False,
        "reason": "Coverage probe only. Need to verify corporate-action adjustment semantics and historical identity before mixing with Yahoo adjusted prices.",
    }]).to_csv(OUT / "alternate_price_stooq_summary.csv", index=False)


if __name__ == "__main__":
    main()
