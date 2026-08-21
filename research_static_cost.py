from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if x.empty:
        return np.nan
    wealth = float((1.0 + x).prod())
    return wealth ** (12.0 / len(x)) - 1.0 if wealth > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    wealth = (1.0 + x).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min()) if len(wealth) else np.nan


def net_at_bps(g: pd.DataFrame, bps: float) -> pd.Series:
    return g["gross_return"].astype(float) - g["traded_notional"].astype(float) * (bps / 10000.0)


def break_even_bps(g: pd.DataFrame, benchmark_cagr: float, high: float = 300.0) -> float:
    if cagr(net_at_bps(g, 0.0)) <= benchmark_cagr:
        return 0.0
    if cagr(net_at_bps(g, high)) > benchmark_cagr:
        return np.nan
    lo = 0.0
    hi = high
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cagr(net_at_bps(g, mid)) > benchmark_cagr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main() -> None:
    g = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"]).sort_values("signal_date")
    b = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])[["signal_date", "spy_return"]]
    x = g.merge(b, on="signal_date", how="inner")
    if len(x) != 102:
        raise RuntimeError(f"Expected 102 months, got {len(x)}")

    spy_cagr = cagr(x["spy_return"])
    breakeven = break_even_bps(x, spy_cagr)
    rows = []
    for bps in (0, 5, 10, 25, 50, 100):
        r = net_at_bps(x, float(bps))
        rows.append({
            "cost_bps_per_dollar_traded": bps,
            "cagr": cagr(r),
            "max_drawdown": max_drawdown(r),
            "spy_cagr": spy_cagr,
            "cagr_excess_vs_spy": cagr(r) - spy_cagr,
            "avg_monthly_traded_notional": float(x["traded_notional"].mean()),
            "estimated_break_even_cost_bps_vs_spy_cagr": breakeven,
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "static_equal_cost_sensitivity.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
