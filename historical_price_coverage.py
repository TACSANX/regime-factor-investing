from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT = Path("data/research")
PIERRE_RAW = "https://raw.githubusercontent.com/pierrebrunelle/sp500-historical-constituents/main/data/spy{yyyymm}"
START = pd.Period("2018-01", freq="M")
END = pd.Period("2026-04", freq="M")


def norm_symbol(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def members(period: pd.Period) -> set[str]:
    url = PIERRE_RAW.format(yyyymm=period.strftime("%Y%m"))
    r = requests.get(url, timeout=30, headers={"User-Agent": "regime-factor-investing research"})
    r.raise_for_status()
    out = set()
    for line in r.content.decode("utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        ticker = line.split("\t", 1)[0].split(",", 1)[0].strip().strip('"')
        if ticker:
            out.add(norm_symbol(ticker))
    return out


def download_close(symbols: list[str], start: str, end: str, batch: int = 60) -> pd.DataFrame:
    frames = []
    for i in range(0, len(symbols), batch):
        b = symbols[i:i + batch]
        print(f"Yahoo coverage batch {i + 1}-{min(i + batch, len(symbols))}/{len(symbols)}", flush=True)
        try:
            d = yf.download(b, start=start, end=end, auto_adjust=True, progress=False, threads=True, group_by="column", timeout=30)
        except Exception as exc:
            print(f"batch failed: {exc}", flush=True)
            continue
        if d.empty:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            if "Close" in d.columns.get_level_values(0):
                c = d["Close"].copy()
            else:
                continue
        else:
            c = pd.DataFrame({b[0]: d["Close"]}, index=d.index)
        frames.append(c)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.columns = [norm_symbol(c) for c in out.columns]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = {p: members(p) for p in pd.period_range(START, END, freq="M")}
    universe = sorted(set().union(*monthly.values()))
    print(f"Historical ticker union: {len(universe)}", flush=True)
    close = download_close(universe, "2017-11-01", "2026-06-15")

    rows = []
    missing_rows = []
    for p, mset in monthly.items():
        month_end = p.end_time.normalize()
        window = close.loc[(close.index >= month_end - pd.Timedelta(days=10)) & (close.index <= month_end + pd.Timedelta(days=3))] if not close.empty else pd.DataFrame()
        available = set()
        for s in mset:
            if s in window.columns and window[s].notna().any():
                available.add(s)
        missing = sorted(mset - available)
        rows.append({
            "month": str(p),
            "members": len(mset),
            "price_available": len(available),
            "missing": len(missing),
            "coverage": len(available) / len(mset) if mset else np.nan,
        })
        for s in missing:
            missing_rows.append({"month": str(p), "symbol": s})

    summary = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows, columns=["month", "symbol"])
    summary.to_csv(OUT / "historical_price_coverage.csv", index=False)
    missing.to_csv(OUT / "historical_price_missing.csv", index=False)
    persistent = missing.groupby("symbol").size().rename("missing_months").sort_values(ascending=False).reset_index() if not missing.empty else pd.DataFrame(columns=["symbol", "missing_months"])
    persistent.to_csv(OUT / "historical_price_missing_summary.csv", index=False)
    pd.DataFrame([{
        "months": len(summary),
        "unique_historical_tickers": len(universe),
        "mean_monthly_coverage": summary["coverage"].mean(),
        "median_monthly_coverage": summary["coverage"].median(),
        "min_monthly_coverage": summary["coverage"].min(),
        "months_ge_99pct": int((summary["coverage"] >= 0.99).sum()),
        "months_ge_98pct": int((summary["coverage"] >= 0.98).sum()),
        "usable_without_alternate_price_source": bool((summary["coverage"] >= 0.98).all()),
    }]).to_csv(OUT / "historical_price_coverage_summary.csv", index=False)


if __name__ == "__main__":
    main()
