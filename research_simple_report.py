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
PRIMARY_ALLOCATIONS = [
    "hyp_no_growth_low_vol_macro_sector_n10_score",
    "hyp_no_growth_low_vol_macro_sector_n10_equal",
]
PAIRWISE = [
    (
        "score_vs_equal_best_factor_set",
        "hyp_no_growth_low_vol_macro_sector_n10_score",
        "hyp_no_growth_low_vol_macro_sector_n10_equal",
    ),
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


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if not len(x):
        return float("nan")
    wealth = float((1.0 + x).prod())
    return wealth ** (12.0 / len(x)) - 1.0 if wealth > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    wealth = (1.0 + x).cumprod()
    if wealth.empty:
        return float("nan")
    return float((wealth / wealth.cummax() - 1.0).min())


def circular_block_means(values: np.ndarray, block: int = 12, reps: int = 20000, seed: int = 20260821) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for i in range(reps):
        starts = rng.integers(0, n, size=blocks_needed)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        out[i] = float(values[idx].mean())
    return out


def net_at_cost(g: pd.DataFrame, bps: float) -> pd.Series:
    return g["gross_return"].astype(float) - g["traded_notional"].astype(float) * (bps / 10000.0)


def break_even_cost_bps(g: pd.DataFrame, benchmark_cagr: float, hi: float = 250.0) -> float:
    if cagr(net_at_cost(g, 0.0)) <= benchmark_cagr:
        return 0.0
    if cagr(net_at_cost(g, hi)) > benchmark_cagr:
        return float("nan")
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if cagr(net_at_cost(g, mid)) > benchmark_cagr:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def main() -> None:
    monthly = pd.read_csv(ROOT / "monthly.csv", parse_dates=["signal_date"])
    bench = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])
    bench = bench[["signal_date", "spy_return", "rsp_return"]]

    market_rows = []
    regime_rows = []
    yearly_rows = []
    stability_rows = []
    pairwise_rows = []
    cost_rows = []
    loo_rows = []
    loo_summary_rows = []

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

    for label, left, right in PAIRWISE:
        l = monthly[monthly["strategy"] == left][["signal_date", "net_return"]].rename(columns={"net_return": "left_return"})
        r = monthly[monthly["strategy"] == right][["signal_date", "net_return"]].rename(columns={"net_return": "right_return"})
        x = l.merge(r, on="signal_date", how="inner").sort_values("signal_date")
        if x.empty:
            continue
        d = (x["left_return"] - x["right_return"]).to_numpy(dtype=float)
        boot = circular_block_means(d)
        pairwise_rows.append({
            "comparison": label,
            "left_strategy": left,
            "right_strategy": right,
            "months": len(d),
            "annualized_mean_return_delta": float(np.mean(d) * 12.0),
            "ci025": float(np.quantile(boot, 0.025) * 12.0),
            "ci50": float(np.quantile(boot, 0.50) * 12.0),
            "ci975": float(np.quantile(boot, 0.975) * 12.0),
            "bootstrap_probability_left_gt_right": float(np.mean(boot > 0.0)),
            "block_months": 12,
            "bootstrap_reps": len(boot),
        })

    spy_cagr = cagr(bench.sort_values("signal_date")["spy_return"])
    for strategy in PRIMARY_ALLOCATIONS:
        g = monthly[monthly["strategy"] == strategy].sort_values("signal_date").copy()
        if g.empty:
            continue
        breakeven = break_even_cost_bps(g, spy_cagr)
        for bps in [0, 5, 10, 15, 20, 25, 50]:
            r = net_at_cost(g, float(bps))
            cost_rows.append({
                "strategy": strategy,
                "cost_bps_per_dollar_traded": bps,
                "cagr": cagr(r),
                "max_drawdown": max_drawdown(r),
                "spy_cagr": spy_cagr,
                "cagr_excess_vs_spy": cagr(r) - spy_cagr,
                "estimated_break_even_cost_bps_vs_spy_cagr": breakeven,
            })

        x = g.merge(bench[["signal_date", "spy_return"]], on="signal_date", how="inner")
        x["year"] = x["signal_date"].dt.year
        full_years = [int(y) for y, n in x.groupby("year").size().items() if int(n) >= 10]
        strategy_loo = []
        for omitted_year in full_years:
            z = x[x["year"] != omitted_year]
            scagr = cagr(z["net_return"])
            bcagr = cagr(z["spy_return"])
            excess = scagr - bcagr
            strategy_loo.append(excess)
            loo_rows.append({
                "strategy": strategy,
                "omitted_year": omitted_year,
                "remaining_months": len(z),
                "strategy_cagr": scagr,
                "spy_cagr": bcagr,
                "cagr_excess_vs_spy": excess,
            })
        if strategy_loo:
            loo_summary_rows.append({
                "strategy": strategy,
                "omissions": len(strategy_loo),
                "min_cagr_excess_vs_spy": float(np.min(strategy_loo)),
                "median_cagr_excess_vs_spy": float(np.median(strategy_loo)),
                "max_cagr_excess_vs_spy": float(np.max(strategy_loo)),
                "positive_excess_rate": float(np.mean(np.array(strategy_loo) > 0)),
                "all_omissions_positive_excess": bool(np.all(np.array(strategy_loo) > 0)),
            })

    pd.DataFrame(market_rows).to_csv(ROOT / "simple_market_states.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(ROOT / "simple_regime_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(ROOT / "simple_yearly.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(ROOT / "simple_stability.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(ROOT / "simple_pairwise.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(ROOT / "simple_cost_sensitivity.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(ROOT / "simple_leave_one_year_out.csv", index=False)
    pd.DataFrame(loo_summary_rows).to_csv(ROOT / "simple_loo_summary.csv", index=False)

    print(pd.DataFrame(market_rows).to_string(index=False), flush=True)
    print("\nStability:\n" + pd.DataFrame(stability_rows).to_string(index=False), flush=True)
    print("\nPairwise:\n" + pd.DataFrame(pairwise_rows).to_string(index=False), flush=True)
    print("\nCost sensitivity:\n" + pd.DataFrame(cost_rows).to_string(index=False), flush=True)
    print("\nLeave-one-year-out:\n" + pd.DataFrame(loo_summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
