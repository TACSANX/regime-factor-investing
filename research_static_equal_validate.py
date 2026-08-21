from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/research_backtest")
STRATEGY = "hyp_static_neutral_no_growth_low_vol_n10_equal_reconstructed"


def cagr(r: pd.Series) -> float:
    x = pd.Series(r, dtype=float).dropna()
    if x.empty:
        return float("nan")
    wealth = float((1.0 + x).prod())
    return wealth ** (12.0 / len(x)) - 1.0 if wealth > 0 else float("nan")


def circular_block_means(values: np.ndarray, block: int = 12, reps: int = 20000, seed: int = 20260821) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=float)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block)
    for i in range(reps):
        starts = rng.integers(0, n, size=blocks_needed)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        out[i] = float(values[idx].mean())
    return out


def rolling_outperformance(x: pd.DataFrame, window: int) -> float:
    wins = []
    for i in range(window - 1, len(x)):
        g = x.iloc[i - window + 1:i + 1]
        wins.append(cagr(g["net_return"]) > cagr(g["spy_return"]))
    return float(np.mean(wins)) if wins else np.nan


def main() -> None:
    s = pd.read_csv(ROOT / "static_equal_monthly.csv", parse_dates=["signal_date"])
    b = pd.read_csv(ROOT / "benchmarks.csv", parse_dates=["signal_date"])[["signal_date", "spy_return"]]
    x = s.merge(b, on="signal_date", how="inner").sort_values("signal_date").reset_index(drop=True)
    x["active"] = x["net_return"] - x["spy_return"]
    x["year"] = x["signal_date"].dt.year
    if len(x) < 90:
        raise RuntimeError(f"Too few months for validation: {len(x)}")

    boot = circular_block_means(x["active"].to_numpy())
    mid = len(x) // 2
    first, second = x.iloc[:mid], x.iloc[mid:]
    down = x[x["spy_return"] < 0]
    up = x[x["spy_return"] >= 0]

    validation = pd.DataFrame([{
        "strategy": STRATEGY,
        "months": len(x),
        "cagr": cagr(x["net_return"]),
        "spy_cagr": cagr(x["spy_return"]),
        "annualized_active_mean": float(x["active"].mean() * 12.0),
        "active_mean_ci025": float(np.quantile(boot, 0.025) * 12.0),
        "active_mean_ci50": float(np.quantile(boot, 0.50) * 12.0),
        "active_mean_ci975": float(np.quantile(boot, 0.975) * 12.0),
        "bootstrap_prob_active_gt_0": float(np.mean(boot > 0.0)),
        "rolling_36_outperformance_rate": rolling_outperformance(x, 36),
        "rolling_60_outperformance_rate": rolling_outperformance(x, 60),
        "first_half_strategy_cagr": cagr(first["net_return"]),
        "first_half_spy_cagr": cagr(first["spy_return"]),
        "second_half_strategy_cagr": cagr(second["net_return"]),
        "second_half_spy_cagr": cagr(second["spy_return"]),
        "down_month_avg_excess": float(down["active"].mean()),
        "down_month_outperform_rate": float((down["active"] > 0).mean()),
        "up_month_avg_excess": float(up["active"].mean()),
        "up_month_outperform_rate": float((up["active"] > 0).mean()),
    }])
    validation.to_csv(ROOT / "static_equal_validation.csv", index=False)

    yearly_rows = []
    loo_excess = []
    for year, g in x.groupby("year"):
        sr, br = cagr(g["net_return"]), cagr(g["spy_return"])
        yearly_rows.append({"year": int(year), "months": len(g), "strategy_cagr": sr, "spy_cagr": br, "cagr_excess": sr - br})
        if len(g) >= 10:
            z = x[x["year"] != year]
            loo_excess.append({"omitted_year": int(year), "strategy_cagr": cagr(z["net_return"]), "spy_cagr": cagr(z["spy_return"]), "cagr_excess": cagr(z["net_return"]) - cagr(z["spy_return"])})
    pd.DataFrame(yearly_rows).to_csv(ROOT / "static_equal_yearly.csv", index=False)
    loo = pd.DataFrame(loo_excess)
    loo.to_csv(ROOT / "static_equal_loo.csv", index=False)
    pd.DataFrame([{
        "omissions": len(loo),
        "min_cagr_excess_vs_spy": float(loo["cagr_excess"].min()),
        "median_cagr_excess_vs_spy": float(loo["cagr_excess"].median()),
        "max_cagr_excess_vs_spy": float(loo["cagr_excess"].max()),
        "all_omissions_positive_excess": bool((loo["cagr_excess"] > 0).all()),
    }]).to_csv(ROOT / "static_equal_loo_summary.csv", index=False)

    regime = x.groupby("regime", dropna=False).agg(
        months=("active", "size"),
        avg_monthly_return=("net_return", "mean"),
        avg_monthly_spy=("spy_return", "mean"),
        avg_monthly_active=("active", "mean"),
    ).reset_index()
    regime["annualized_active_mean"] = regime["avg_monthly_active"] * 12.0
    regime.to_csv(ROOT / "static_equal_regime.csv", index=False)

    print(validation.to_string(index=False), flush=True)
    print("\nLOO:\n" + pd.read_csv(ROOT / "static_equal_loo_summary.csv").to_string(index=False), flush=True)
    print("\nRegimes:\n" + regime.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
