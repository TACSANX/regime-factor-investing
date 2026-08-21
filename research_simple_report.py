from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
SHORTLIST = [
    "hyp_no_growth_low_vol_macro_sector_n10_score",
    "hyp_static_neutral_no_growth_low_vol_n10",
    "full_dynamic_n10_score",
]


def beta(y: pd.Series, x: pd.Series) -> float:
    pair = pd.concat([y, x], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    xv = float(pair.iloc[:, 1].var(ddof=1))
    if xv <= 0:
        return float("nan")
    return float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / xv)


def compound(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    return float((1.0 + x).prod() - 1.0) if len(x) else float("nan")


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    bench = bench[["signal_date", "spy_return", "rsp_return"]]

    market_rows = []
    regime_rows = []
    yearly_rows = []
    stability_rows = []

    for strategy in SHORTLIST:
        s = monthly[monthly["strategy"] == strategy].copy()
        if s.empty:
            continue
        x = s.merge(bench, on="signal_date", how="inner").sort_values("signal_date")
        x["excess_vs_spy"] = x["net_return"] - x["spy_return"]
        x["year"] = x["signal_date"].dt.year

        down = x[x["spy_return"] < 0]
        up = x[x["spy_return"] >= 0]
        q20 = float(x["spy_return"].quantile(0.20))
        stress = x[x["spy_return"] <= q20]

        market_rows.append({
            "strategy": strategy,
            "months": len(x),
            "correlation_with_spy": float(x["net_return"].corr(x["spy_return"])),
            "beta_to_spy": beta(x["net_return"], x["spy_return"]),
            "down_months": len(down),
            "down_month_avg_strategy": float(down["net_return"].mean()),
            "down_month_avg_spy": float(down["spy_return"].mean()),
            "down_month_avg_excess": float(down["excess_vs_spy"].mean()),
            "down_month_outperform_rate": float((down["excess_vs_spy"] > 0).mean()),
            "down_month_beta": beta(down["net_return"], down["spy_return"]),
            "up_months": len(up),
            "up_month_avg_strategy": float(up["net_return"].mean()),
            "up_month_avg_spy": float(up["spy_return"].mean()),
            "up_month_avg_excess": float(up["excess_vs_spy"].mean()),
            "up_month_outperform_rate": float((up["excess_vs_spy"] > 0).mean()),
            "worst_spy_quintile_months": len(stress),
            "worst_spy_quintile_avg_strategy": float(stress["net_return"].mean()),
            "worst_spy_quintile_avg_spy": float(stress["spy_return"].mean()),
            "worst_spy_quintile_avg_excess": float(stress["excess_vs_spy"].mean()),
        })

        for regime, g in x.groupby("regime", dropna=False):
            regime_rows.append({
                "strategy": strategy,
                "regime": regime,
                "months": len(g),
                "avg_monthly_strategy_return": float(g["net_return"].mean()),
                "avg_monthly_spy_return": float(g["spy_return"].mean()),
                "annualized_arithmetic_excess_vs_spy": float(g["excess_vs_spy"].mean() * 12.0),
                "outperform_rate_vs_spy": float((g["excess_vs_spy"] > 0).mean()),
                "positive_month_rate": float((g["net_return"] > 0).mean()),
            })

        year_excess = []
        for year, g in x.groupby("year"):
            sr = compound(g["net_return"])
            br = compound(g["spy_return"])
            ex = sr - br
            year_excess.append(ex)
            yearly_rows.append({
                "strategy": strategy,
                "year": int(year),
                "months": len(g),
                "strategy_return": sr,
                "spy_return": br,
                "excess_return": ex,
            })

        regime_summary = pd.DataFrame([r for r in regime_rows if r["strategy"] == strategy])
        stability_rows.append({
            "strategy": strategy,
            "years": len(year_excess),
            "years_outperforming_spy": int(sum(v > 0 for v in year_excess)),
            "year_outperformance_rate": float(np.mean(np.array(year_excess) > 0)) if year_excess else np.nan,
            "worst_year_excess": float(min(year_excess)) if year_excess else np.nan,
            "best_year_excess": float(max(year_excess)) if year_excess else np.nan,
            "regimes_observed": len(regime_summary),
            "regimes_positive_mean_excess": int((regime_summary["annualized_arithmetic_excess_vs_spy"] > 0).sum()) if len(regime_summary) else 0,
            "worst_regime_annualized_mean_excess": float(regime_summary["annualized_arithmetic_excess_vs_spy"].min()) if len(regime_summary) else np.nan,
            "best_regime_annualized_mean_excess": float(regime_summary["annualized_arithmetic_excess_vs_spy"].max()) if len(regime_summary) else np.nan,
        })

    pd.DataFrame(market_rows).to_csv(ROOT / "simple_market_states.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(ROOT / "simple_regime_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(ROOT / "simple_yearly.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(ROOT / "simple_stability.csv", index=False)

    print(pd.DataFrame(market_rows).to_string(index=False), flush=True)
    print("\nStability:\n" + pd.DataFrame(stability_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
