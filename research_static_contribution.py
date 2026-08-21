from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import research_static_equal as rse

ROOT = Path("data/research_backtest")
SOURCE = rse.SOURCE_STRATEGY


def main() -> None:
    holdings = pd.read_csv(
        ROOT / "holdings.csv",
        usecols=["strategy", "signal_date", "Symbol", "Security", "GICS Sector"],
        parse_dates=["signal_date"],
    )
    monthly = pd.read_csv(
        ROOT / "monthly.csv",
        usecols=["strategy", "signal_date", "execution_date", "exit_execution_date"],
        parse_dates=["signal_date", "execution_date", "exit_execution_date"],
    )
    h = holdings[holdings["strategy"] == SOURCE].copy()
    periods = monthly[monthly["strategy"] == SOURCE][
        ["signal_date", "execution_date", "exit_execution_date"]
    ].drop_duplicates("signal_date").sort_values("signal_date")
    if h.empty or len(periods) != 102:
        raise RuntimeError(f"Expected full 102-month static source; holdings={len(h)}, periods={len(periods)}")

    start, end = periods["execution_date"].min(), periods["exit_execution_date"].max()
    symbols = sorted({rse.yahoo_symbol(s) for s in h["Symbol"].astype(str)})
    open_px = rse.download_opens(symbols, start, end)
    required = rse.required_price_dates(h, periods)
    open_px, unresolved = rse.repair_missing_opens(open_px, required, start, end)
    if unresolved:
        raise RuntimeError(f"Unresolved price histories: {unresolved[:20]}")

    period_map = periods.set_index("signal_date").to_dict("index")
    detail = []
    for date, g in h.groupby("signal_date"):
        p = period_map[pd.Timestamp(date)]
        n = len(g)
        if n != 10:
            raise RuntimeError(f"Expected 10 names at {date.date()}, got {n}")
        weight = 1.0 / n
        for _, row in g.iterrows():
            symbol = str(row["Symbol"])
            ret = rse.price_return(open_px, symbol, pd.Timestamp(p["execution_date"]), pd.Timestamp(p["exit_execution_date"]))
            if not np.isfinite(ret):
                raise RuntimeError(f"Missing return {symbol} at {date.date()}")
            detail.append({
                "signal_date": date.date().isoformat(),
                "Symbol": symbol,
                "Security": row["Security"],
                "GICS Sector": row["GICS Sector"],
                "weight": weight,
                "stock_return": ret,
                "gross_return_contribution": weight * ret,
            })
    d = pd.DataFrame(detail)
    d.to_csv(ROOT / "static_equal_contribution_detail.csv", index=False)

    years = len(periods) / 12.0
    sym = d.groupby(["Symbol", "Security", "GICS Sector"], as_index=False).agg(
        months_held=("signal_date", "count"),
        arithmetic_gross_contribution=("gross_return_contribution", "sum"),
        avg_stock_return_when_held=("stock_return", "mean"),
    )
    sym["annualized_arithmetic_gross_contribution"] = sym["arithmetic_gross_contribution"] / years
    sym = sym.sort_values("arithmetic_gross_contribution", ascending=False)
    sym.to_csv(ROOT / "static_equal_symbol_contribution.csv", index=False)

    sector = d.groupby("GICS Sector", as_index=False).agg(
        holding_observations=("Symbol", "count"),
        arithmetic_gross_contribution=("gross_return_contribution", "sum"),
    )
    sector["annualized_arithmetic_gross_contribution"] = sector["arithmetic_gross_contribution"] / years
    sector["holding_observation_share"] = sector["holding_observations"] / len(d)
    sector = sector.sort_values("arithmetic_gross_contribution", ascending=False)
    sector.to_csv(ROOT / "static_equal_sector_contribution.csv", index=False)

    total = float(sym["arithmetic_gross_contribution"].sum())
    positives = sym[sym["arithmetic_gross_contribution"] > 0]
    positive_total = float(positives["arithmetic_gross_contribution"].sum())
    top5 = float(sym.head(5)["arithmetic_gross_contribution"].sum())
    top10 = float(sym.head(10)["arithmetic_gross_contribution"].sum())
    top_sector = float(sector.iloc[0]["arithmetic_gross_contribution"])
    summary = pd.DataFrame([{
        "months": len(periods),
        "holding_observations": len(d),
        "distinct_symbols": int(d["Symbol"].nunique()),
        "distinct_sectors": int(d["GICS Sector"].nunique()),
        "total_arithmetic_gross_contribution": total,
        "annualized_arithmetic_gross_contribution": total / years,
        "top5_symbol_contribution_fraction_of_net_total": top5 / total if total else np.nan,
        "top10_symbol_contribution_fraction_of_net_total": top10 / total if total else np.nan,
        "top5_share_of_all_positive_contributions": top5 / positive_total if positive_total else np.nan,
        "top_sector": sector.iloc[0]["GICS Sector"],
        "top_sector_contribution_fraction_of_net_total": top_sector / total if total else np.nan,
        "top_sector_holding_observation_share": float(sector.iloc[0]["holding_observation_share"]),
    }])
    summary.to_csv(ROOT / "static_equal_contribution_summary.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print("\nTop symbols:\n" + sym.head(15).to_string(index=False), flush=True)
    print("\nSectors:\n" + sector.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
