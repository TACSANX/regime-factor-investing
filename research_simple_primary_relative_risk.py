from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")


def compound(x: pd.Series) -> float:
    s = pd.Series(x, dtype=float).dropna()
    return float((1.0 + s).prod() - 1.0) if len(s) else np.nan


def longest_true_streak(mask: pd.Series) -> int:
    best = cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main() -> None:
    simple = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])[["signal_date", "spy_return"]]
    x = simple[["signal_date", "net_return"]].merge(bench, on="signal_date", how="inner").sort_values("signal_date")
    if len(x) != 102:
        raise RuntimeError(f"Expected 102 common months, got {len(x)}")

    x["active_return"] = x["net_return"] - x["spy_return"]
    x["strategy_wealth"] = (1.0 + x["net_return"]).cumprod()
    x["spy_wealth"] = (1.0 + x["spy_return"]).cumprod()
    x["relative_wealth"] = x["strategy_wealth"] / x["spy_wealth"]
    x["relative_drawdown"] = x["relative_wealth"] / x["relative_wealth"].cummax() - 1.0
    x.to_csv(ROOT / "simple_primary_relative_path.csv", index=False)

    rows = []
    for window in (12, 24, 36, 60):
        values = []
        end_dates = []
        for i in range(window - 1, len(x)):
            g = x.iloc[i - window + 1:i + 1]
            sr = compound(g["net_return"])
            br = compound(g["spy_return"])
            values.append(sr - br)
            end_dates.append(g["signal_date"].iloc[-1])
        a = pd.Series(values, dtype=float)
        worst_i = int(a.idxmin()) if len(a) else -1
        rows.append({
            "window_months": window,
            "windows": len(a),
            "outperformance_rate": float((a > 0).mean()) if len(a) else np.nan,
            "median_compounded_excess": float(a.median()) if len(a) else np.nan,
            "worst_compounded_excess": float(a.min()) if len(a) else np.nan,
            "best_compounded_excess": float(a.max()) if len(a) else np.nan,
            "worst_window_end": pd.Timestamp(end_dates[worst_i]).date().isoformat() if worst_i >= 0 else "",
        })
    rolling = pd.DataFrame(rows)
    rolling.to_csv(ROOT / "simple_primary_relative_rolling.csv", index=False)

    dd_idx = int(x["relative_drawdown"].idxmin())
    summary = pd.DataFrame([{
        "months": len(x),
        "max_relative_drawdown_vs_spy": float(x["relative_drawdown"].min()),
        "max_relative_drawdown_date": x.loc[dd_idx, "signal_date"].date().isoformat(),
        "longest_consecutive_months_underperforming_spy": longest_true_streak(x["active_return"] < 0),
        "monthly_outperformance_rate": float((x["active_return"] > 0).mean()),
        "final_relative_wealth_multiple": float(x["relative_wealth"].iloc[-1]),
        "positive_12m_windows": float(rolling.loc[rolling["window_months"] == 12, "outperformance_rate"].iloc[0]),
        "positive_24m_windows": float(rolling.loc[rolling["window_months"] == 24, "outperformance_rate"].iloc[0]),
        "positive_36m_windows": float(rolling.loc[rolling["window_months"] == 36, "outperformance_rate"].iloc[0]),
        "positive_60m_windows": float(rolling.loc[rolling["window_months"] == 60, "outperformance_rate"].iloc[0]),
    }])
    summary.to_csv(ROOT / "simple_primary_relative_risk.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print("\nRolling relative results:\n" + rolling.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
