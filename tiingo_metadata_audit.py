from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("data/research")
FAILURES = OUT / "historical_price_download_failures.csv"
MISSING = OUT / "historical_price_missing.csv"
TIINGO_SUPPORTED = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
LOOKBACK_DAYS = 430
EXIT_BUFFER_DAYS = 45


def norm_symbol(value: str) -> str:
    return str(value).strip().upper().replace(".", "-")


def load_tiingo_supported() -> pd.DataFrame:
    r = requests.get(
        TIINGO_SUPPORTED,
        timeout=90,
        headers={"User-Agent": "regime-factor-investing research metadata audit"},
    )
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        target = next((n for n in names if n.endswith("supported_tickers.csv")), None)
        if target is None:
            raise RuntimeError(f"supported_tickers.csv not found in Tiingo archive: {names[:10]}")
        with z.open(target) as f:
            df = pd.read_csv(f, dtype=str)
    required = {"ticker", "startDate", "endDate"}
    if missing := required - set(df.columns):
        raise RuntimeError(f"Unexpected Tiingo metadata columns, missing={sorted(missing)}")
    df["ticker_norm"] = df["ticker"].map(norm_symbol)
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce")
    return df


def required_windows() -> pd.DataFrame:
    failures = pd.read_csv(FAILURES, dtype=str)
    symbols = failures["yahoo_symbol"].dropna().map(norm_symbol).drop_duplicates().tolist()
    missing = pd.read_csv(MISSING, dtype=str)
    missing["yahoo_symbol"] = missing["yahoo_symbol"].map(norm_symbol)
    missing["month"] = pd.PeriodIndex(missing["month"], freq="M")
    if "download_failed_entire_series" in missing:
        flag = missing["download_failed_entire_series"].astype(str).str.lower().isin({"true", "1", "yes"})
        missing = missing[flag]

    rows = []
    for symbol in symbols:
        g = missing[missing["yahoo_symbol"] == symbol]
        if g.empty:
            rows.append({"yahoo_symbol": symbol})
            continue
        first = g["month"].min()
        last = g["month"].max()
        first_month_end = first.end_time.normalize()
        last_month_end = last.end_time.normalize()
        rows.append({
            "yahoo_symbol": symbol,
            "first_missing_month": str(first),
            "last_missing_month": str(last),
            "required_membership_start": first.start_time.normalize(),
            "required_membership_end": last_month_end,
            "required_signal_lookback_start": first_month_end - pd.Timedelta(days=LOOKBACK_DAYS),
            "required_exit_buffer_end": last_month_end + pd.Timedelta(days=EXIT_BUFFER_DAYS),
        })
    return pd.DataFrame(rows)


def best_metadata_row(candidates: pd.DataFrame, req: pd.Series) -> pd.Series | None:
    if candidates.empty:
        return None
    x = candidates.copy()
    req_start = req.get("required_signal_lookback_start")
    req_end = req.get("required_exit_buffer_end")
    if pd.notna(req_start) and pd.notna(req_end):
        overlap_start = x["startDate"].apply(lambda d: max(d, req_start) if pd.notna(d) else req_end)
        overlap_end = x["endDate"].apply(lambda d: min(d, req_end) if pd.notna(d) else req_start)
        x["overlap_days"] = (overlap_end - overlap_start).dt.days.clip(lower=0)
    else:
        x["overlap_days"] = 0
    x["dated"] = x["startDate"].notna().astype(int) + x["endDate"].notna().astype(int)
    return x.sort_values(["overlap_days", "dated", "endDate"], ascending=[False, False, False]).iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    supported = load_tiingo_supported()
    reqs = required_windows()
    rows = []

    for _, req in reqs.iterrows():
        symbol = req["yahoo_symbol"]
        candidates = supported[supported["ticker_norm"] == symbol]
        best = best_metadata_row(candidates, req)
        if best is None:
            rows.append({
                **req.to_dict(),
                "tiingo_rows_for_symbol": 0,
                "tiingo_metadata_candidate": False,
                "covers_membership_window": False,
                "covers_signal_lookback": False,
                "covers_exit_buffer": False,
                "covers_full_backtest_window": False,
            })
            continue

        t_start = best.get("startDate")
        t_end = best.get("endDate")
        membership_start = req.get("required_membership_start")
        membership_end = req.get("required_membership_end")
        lookback_start = req.get("required_signal_lookback_start")
        exit_end = req.get("required_exit_buffer_end")

        covers_membership = bool(
            pd.notna(t_start) and pd.notna(t_end)
            and pd.notna(membership_start) and pd.notna(membership_end)
            and t_start <= membership_start and t_end >= membership_end
        )
        covers_lookback = bool(pd.notna(t_start) and pd.notna(lookback_start) and t_start <= lookback_start)
        covers_exit = bool(pd.notna(t_end) and pd.notna(exit_end) and t_end >= exit_end)
        rows.append({
            **req.to_dict(),
            "tiingo_rows_for_symbol": int(len(candidates)),
            "tiingo_metadata_candidate": True,
            "tiingo_ticker": best.get("ticker"),
            "tiingo_exchange": best.get("exchange", best.get("exchangeCode", "")),
            "tiingo_asset_type": best.get("assetType", ""),
            "tiingo_price_currency": best.get("priceCurrency", ""),
            "tiingo_start_date": t_start,
            "tiingo_end_date": t_end,
            "covers_membership_window": covers_membership,
            "covers_signal_lookback": covers_lookback,
            "covers_exit_buffer": covers_exit,
            "covers_full_backtest_window": bool(covers_membership and covers_lookback and covers_exit),
        })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "tiingo_metadata_coverage.csv", index=False)
    n = len(detail)
    metadata = int(detail["tiingo_metadata_candidate"].sum()) if n else 0
    membership = int(detail["covers_membership_window"].sum()) if n else 0
    lookback = int(detail["covers_signal_lookback"].sum()) if n else 0
    exit_ok = int(detail["covers_exit_buffer"].sum()) if n else 0
    full = int(detail["covers_full_backtest_window"].sum()) if n else 0
    ambiguous = int((detail.get("tiingo_rows_for_symbol", pd.Series(dtype=int)).fillna(0) > 1).sum()) if n else 0
    summary = pd.DataFrame([{
        "yahoo_failures_tested": n,
        "tiingo_metadata_candidates": metadata,
        "metadata_candidate_rate": metadata / n if n else np.nan,
        "covers_membership_window": membership,
        "covers_signal_lookback": lookback,
        "covers_exit_buffer": exit_ok,
        "covers_full_backtest_window": full,
        "multiple_tiingo_rows_same_symbol": ambiguous,
        "requires_api_token_for_price_validation": True,
        "approved_for_backtest": False,
        "note": "Metadata-only audit using Tiingo supported_tickers.zip. Actual adjusted OHLC coverage, identity, delisting proceeds/corporate actions, and licensing must be validated before mixing with Yahoo data.",
    }])
    summary.to_csv(OUT / "tiingo_metadata_coverage_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("\nPotential metadata gaps / delisting-exit cases:", flush=True)
    print(
        detail.loc[~detail["covers_full_backtest_window"], [
            "yahoo_symbol", "tiingo_metadata_candidate", "tiingo_start_date", "tiingo_end_date",
            "covers_membership_window", "covers_signal_lookback", "covers_exit_buffer",
        ]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
