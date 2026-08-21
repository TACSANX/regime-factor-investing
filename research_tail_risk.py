from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
STRATEGIES = [
    "hyp_no_growth_low_vol_macro_sector_n10_score",
    "hyp_static_neutral_no_growth_low_vol_n10",
    "full_dynamic_n10_score",
]


def drawdown_stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("signal_date").copy()
    wealth = (1.0 + g["net_return"].astype(float)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough_i = int(np.argmin(dd.to_numpy()))
    trough_date = pd.Timestamp(g.iloc[trough_i]["signal_date"])
    max_dd = float(dd.iloc[trough_i])

    peak_level = float(peak.iloc[trough_i])
    pre = wealth.iloc[: trough_i + 1]
    peak_candidates = pre[np.isclose(pre.to_numpy(), peak_level, rtol=1e-10, atol=1e-12)]
    if peak_candidates.empty:
        peak_i = int(np.argmax(pre.to_numpy()))
    else:
        peak_i = int(peak_candidates.index[-1] - g.index[0]) if isinstance(g.index, pd.RangeIndex) else int(np.where(pre.index == peak_candidates.index[-1])[0][0])
    peak_date = pd.Timestamp(g.iloc[peak_i]["signal_date"])

    recovery_date = pd.NaT
    after = wealth.iloc[trough_i + 1 :]
    recovered = after[after >= peak_level]
    if not recovered.empty:
        recovery_pos = int(np.where(wealth.index == recovered.index[0])[0][0])
        recovery_date = pd.Timestamp(g.iloc[recovery_pos]["signal_date"])

    underwater = dd < -1e-12
    longest = 0
    current = 0
    for flag in underwater:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)

    return {
        "max_drawdown": max_dd,
        "drawdown_peak_signal_date": peak_date.date().isoformat(),
        "drawdown_trough_signal_date": trough_date.date().isoformat(),
        "recovery_signal_date": recovery_date.date().isoformat() if pd.notna(recovery_date) else "",
        "peak_to_trough_months": int(trough_i - peak_i),
        "longest_underwater_months": int(longest),
    }


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    benchmark = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    spy = benchmark[["signal_date", "spy_return"]]

    summary_rows = []
    worst_rows = []
    concentration_rows = []

    for strategy in STRATEGIES:
        g = monthly[monthly["strategy"] == strategy].sort_values("signal_date").copy()
        if g.empty:
            continue
        g = g.merge(spy, on="signal_date", how="left")
        g["excess"] = g["net_return"] - g["spy_return"]

        row = {"strategy": strategy, "months": len(g), **drawdown_stats(g)}
        row.update({
            "worst_month": float(g["net_return"].min()),
            "worst_month_spy": float(g.loc[g["net_return"].idxmin(), "spy_return"]),
            "worst_12m_excess": float(g["excess"].rolling(12).sum().min()),
            "best_12m_excess": float(g["excess"].rolling(12).sum().max()),
            "return_corr_max_sector_weight": float(g["net_return"].corr(g["max_sector_weight"])),
            "return_corr_max_stock_weight": float(g["net_return"].corr(g["max_stock_weight"])),
            "return_corr_traded_notional": float(g["net_return"].corr(g["traded_notional"])),
        })
        summary_rows.append(row)

        for _, r in g.nsmallest(8, "net_return").iterrows():
            worst_rows.append({
                "strategy": strategy,
                "signal_date": r["signal_date"].date().isoformat(),
                "net_return": r["net_return"],
                "spy_return": r["spy_return"],
                "excess": r["excess"],
                "regime": r["regime"],
                "max_stock_weight": r["max_stock_weight"],
                "max_sector_weight": r["max_sector_weight"],
                "traded_notional": r["traded_notional"],
            })

        for metric in ["max_sector_weight", "max_stock_weight", "traded_notional"]:
            med = float(g[metric].median())
            for label, subset in [("low", g[g[metric] <= med]), ("high", g[g[metric] > med])]:
                concentration_rows.append({
                    "strategy": strategy,
                    "metric": metric,
                    "bucket": label,
                    "threshold_median": med,
                    "months": len(subset),
                    "avg_monthly_return": float(subset["net_return"].mean()),
                    "avg_monthly_excess_vs_spy": float(subset["excess"].mean()),
                    "negative_month_rate": float((subset["net_return"] < 0).mean()),
                })

    pd.DataFrame(summary_rows).to_csv(ROOT / "tail_risk_summary.csv", index=False)
    pd.DataFrame(worst_rows).to_csv(ROOT / "worst_months.csv", index=False)
    pd.DataFrame(concentration_rows).to_csv(ROOT / "concentration_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
