from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT = Path("data/research")
PIERRE_RAW = "https://raw.githubusercontent.com/pierrebrunelle/sp500-historical-constituents/main/data/spy{yyyymm}"
START = pd.Period("2018-01", freq="M")
END = pd.Period("2026-04", freq="M")


def norm_symbol(x: str) -> str:
    return str(x).strip().upper().replace(".", "/")


def yahoo_symbol(x: str) -> str:
    # Yahoo uses '-' for share classes (BRK-B, BF-B), while the historical
    # constituent source uses Wikipedia-style '/'.
    return norm_symbol(x).replace("/", "-")


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


def _extract_close(d: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        level0 = set(map(str, d.columns.get_level_values(0)))
        level1 = set(map(str, d.columns.get_level_values(1)))
        if "Close" in level0:
            c = d["Close"].copy()
        elif "Close" in level1:
            c = d.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" not in d.columns or len(requested) != 1:
            return pd.DataFrame()
        c = pd.DataFrame({requested[0]: d["Close"]}, index=d.index)
    c.index = pd.to_datetime(c.index).tz_localize(None)
    c.columns = [str(x).upper().replace(".", "-").replace("/", "-") for x in c.columns]
    return c


def _download(symbols: list[str], start: str, end: str, threads: bool = True) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    try:
        d = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=threads,
            group_by="column",
            timeout=30,
        )
    except Exception as exc:
        print(f"download failed for {symbols[:3]}...: {exc}", flush=True)
        return pd.DataFrame()
    return _extract_close(d, symbols)


def download_close(symbols: list[str], start: str, end: str, batch: int = 50) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    expected = sorted(set(yahoo_symbol(s) for s in symbols))

    for i in range(0, len(expected), batch):
        b = expected[i:i + batch]
        print(f"Yahoo coverage batch {i + 1}-{min(i + batch, len(expected))}/{len(expected)}", flush=True)
        c = _download(b, start, end, threads=True)
        if not c.empty:
            frames.append(c)

    close = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not close.empty:
        close = close.loc[:, ~close.columns.duplicated()].sort_index()

    # A single bad/delisted ticker or a transient Yahoo failure can poison part
    # of a bulk response. Retry symbols with no usable series individually so
    # transport failures are not mislabeled as historical data gaps.
    no_series = [s for s in expected if s not in close.columns or close[s].notna().sum() == 0]
    print(f"Bulk pass missing entire series: {len(no_series)}; retrying individually", flush=True)
    recovered: list[pd.DataFrame] = []
    still_failed: list[str] = []
    for j, symbol in enumerate(no_series, start=1):
        got = pd.DataFrame()
        for attempt in range(3):
            got = _download([symbol], start, end, threads=False)
            if not got.empty and symbol in got.columns and got[symbol].notna().any():
                break
            time.sleep(1.0 + attempt)
        if not got.empty and symbol in got.columns and got[symbol].notna().any():
            recovered.append(got[[symbol]])
        else:
            still_failed.append(symbol)
        if j % 25 == 0:
            print(f"Individual retries: {j}/{len(no_series)}", flush=True)

    if recovered:
        close = pd.concat([close] + recovered, axis=1)
        close = close.loc[:, ~close.columns.duplicated()].sort_index()
    return close, still_failed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = {p: members(p) for p in pd.period_range(START, END, freq="M")}
    universe = sorted(set().union(*monthly.values()))
    print(f"Historical ticker union: {len(universe)}", flush=True)
    close, download_failures = download_close(universe, "2017-11-01", "2026-06-15")

    rows = []
    missing_rows = []
    for p, mset in monthly.items():
        month_end = p.end_time.normalize()
        window = close.loc[
            (close.index >= month_end - pd.Timedelta(days=10))
            & (close.index <= month_end + pd.Timedelta(days=3))
        ] if not close.empty else pd.DataFrame()
        available = set()
        for source_symbol in mset:
            ys = yahoo_symbol(source_symbol)
            if ys in window.columns and window[ys].notna().any():
                available.add(source_symbol)
        missing = sorted(mset - available)
        rows.append({
            "month": str(p),
            "members": len(mset),
            "price_available": len(available),
            "missing": len(missing),
            "coverage": len(available) / len(mset) if mset else np.nan,
        })
        for source_symbol in missing:
            ys = yahoo_symbol(source_symbol)
            missing_rows.append({
                "month": str(p),
                "symbol": source_symbol,
                "yahoo_symbol": ys,
                "download_failed_entire_series": ys in download_failures,
            })

    summary = pd.DataFrame(rows)
    missing = pd.DataFrame(
        missing_rows,
        columns=["month", "symbol", "yahoo_symbol", "download_failed_entire_series"],
    )
    summary.to_csv(OUT / "historical_price_coverage.csv", index=False)
    missing.to_csv(OUT / "historical_price_missing.csv", index=False)
    persistent = (
        missing.groupby(["symbol", "yahoo_symbol"])
        .agg(missing_months=("month", "size"), download_failed_entire_series=("download_failed_entire_series", "max"))
        .sort_values("missing_months", ascending=False)
        .reset_index()
        if not missing.empty
        else pd.DataFrame(columns=["symbol", "yahoo_symbol", "missing_months", "download_failed_entire_series"])
    )
    persistent.to_csv(OUT / "historical_price_missing_summary.csv", index=False)
    pd.DataFrame({"yahoo_symbol": sorted(download_failures)}).to_csv(
        OUT / "historical_price_download_failures.csv", index=False
    )

    effective_missing = missing[~missing["download_failed_entire_series"]] if not missing.empty else missing
    effective_counts = effective_missing.groupby("month").size() if not effective_missing.empty else pd.Series(dtype=float)
    adjusted_coverage = []
    for _, row in summary.iterrows():
        effective = int(effective_counts.get(row["month"], 0))
        adjusted_coverage.append((row["members"] - effective) / row["members"] if row["members"] else np.nan)
    summary["coverage_excluding_transport_failures"] = adjusted_coverage
    summary.to_csv(OUT / "historical_price_coverage.csv", index=False)

    pd.DataFrame([{
        "months": len(summary),
        "unique_historical_tickers": len(universe),
        "bulk_and_retry_download_failures": len(download_failures),
        "mean_monthly_coverage": summary["coverage"].mean(),
        "median_monthly_coverage": summary["coverage"].median(),
        "min_monthly_coverage": summary["coverage"].min(),
        "mean_coverage_excluding_transport_failures": summary["coverage_excluding_transport_failures"].mean(),
        "months_ge_99pct": int((summary["coverage"] >= 0.99).sum()),
        "months_ge_98pct": int((summary["coverage"] >= 0.98).sum()),
        "usable_without_alternate_price_source": bool((summary["coverage"] >= 0.98).all() and not download_failures),
    }]).to_csv(OUT / "historical_price_coverage_summary.csv", index=False)


if __name__ == "__main__":
    main()
