from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path("data/backtest")
OUT = Path("data/research")


def perf_stats(r: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    r = pd.Series(r, dtype=float).dropna()
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    years = len(r) / periods_per_year
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    std = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    vol = std * math.sqrt(periods_per_year) if np.isfinite(std) else np.nan
    sharpe = float(r.mean() / std * math.sqrt(periods_per_year)) if np.isfinite(std) and std > 0 else np.nan
    downside = float(r[r < 0].std(ddof=1)) if (r < 0).sum() > 1 else np.nan
    sortino = float(r.mean() / downside * math.sqrt(periods_per_year)) if np.isfinite(downside) and downside > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    mdd = float(dd.min())
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe_0rf": sharpe,
        "sortino_0rf": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "positive_month_rate": float((r > 0).mean()),
        "months": int(len(r)),
    }


def longest_underwater_months(r: pd.Series) -> int:
    eq = (1 + pd.Series(r, dtype=float).fillna(0)).cumprod()
    high = eq.cummax()
    underwater = eq < high
    longest = current = 0
    for v in underwater:
        if bool(v):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def bootstrap_active(active: pd.Series, block: int = 12, reps: int = 10000, seed: int = 20260821) -> dict[str, float]:
    x = pd.Series(active, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < block:
        return {}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    base = np.arange(n)
    blocks_needed = math.ceil(n / block)
    for i in range(reps):
        starts = rng.integers(0, n, size=blocks_needed)
        sample_idx = np.concatenate([(base[s:s + block] if s + block <= n else np.r_[base[s:], base[:(s + block) % n]]) for s in starts])[:n]
        means[i] = x[sample_idx].mean()
    return {
        "bootstrap_block_months": block,
        "bootstrap_reps": reps,
        "annualized_active_mean_p025": float(np.quantile(means, 0.025) * 12),
        "annualized_active_mean_p50": float(np.quantile(means, 0.50) * 12),
        "annualized_active_mean_p975": float(np.quantile(means, 0.975) * 12),
        "bootstrap_probability_active_mean_gt_0": float((means > 0).mean()),
    }


def rsp_returns(perf: pd.DataFrame) -> pd.Series | None:
    if yf is None:
        return None
    start = pd.to_datetime(perf["date"]).min() - pd.Timedelta(days=10)
    end = pd.to_datetime(perf["next_date"]).max() + pd.Timedelta(days=10)
    try:
        data = yf.download("RSP", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False, threads=False)
        if data.empty:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        vals = []
        for _, row in perf.iterrows():
            d0, d1 = pd.Timestamp(row["date"]), pd.Timestamp(row["next_date"])
            s0, s1 = close.loc[:d0].dropna(), close.loc[:d1].dropna()
            if s0.empty or s1.empty:
                vals.append(np.nan)
            else:
                vals.append(float(s1.iloc[-1] / s0.iloc[-1] - 1.0))
        return pd.Series(vals, index=perf.index, name="rsp_return")
    except Exception as exc:
        print(f"RSP benchmark unavailable: {exc}", flush=True)
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(ROOT / "equity_curve.csv", parse_dates=["date", "next_date"])
    holdings = pd.read_csv(ROOT / "holdings.csv", parse_dates=["date"])

    # Core active-return significance.
    active = perf["strategy_return"] - perf["spy_return"]
    active_std = float(active.std(ddof=1))
    active_mean = float(active.mean())
    t_stat = active_mean / (active_std / math.sqrt(len(active))) if active_std > 0 else np.nan
    normal_two_sided_p = math.erfc(abs(t_stat) / math.sqrt(2)) if np.isfinite(t_stat) else np.nan
    sig = {
        "months": len(active),
        "annualized_active_mean_arithmetic": active_mean * 12,
        "annualized_tracking_error": active_std * math.sqrt(12),
        "information_ratio": active_mean / active_std * math.sqrt(12) if active_std > 0 else np.nan,
        "iid_t_stat": t_stat,
        "normal_approx_two_sided_p": normal_two_sided_p,
    }
    sig.update(bootstrap_active(active))
    pd.DataFrame([sig]).to_csv(OUT / "significance.csv", index=False)

    # Calendar-year performance.
    annual_rows = []
    perf["return_year"] = perf["next_date"].dt.year
    for year, g in perf.groupby("return_year"):
        sr = float((1 + g["strategy_return"]).prod() - 1)
        br = float((1 + g["spy_return"]).prod() - 1)
        annual_rows.append({"year": int(year), "strategy_return": sr, "spy_return": br, "excess_return": sr - br, "months": len(g)})
    pd.DataFrame(annual_rows).to_csv(OUT / "annual_returns.csv", index=False)

    # Regime-conditional behavior.
    regime_rows = []
    for regime, g in perf.groupby("regime"):
        a = g["strategy_return"] - g["spy_return"]
        regime_rows.append({
            "regime": regime,
            "months": len(g),
            "strategy_mean_month": g["strategy_return"].mean(),
            "spy_mean_month": g["spy_return"].mean(),
            "active_mean_month": a.mean(),
            "active_annualized_arithmetic": a.mean() * 12,
            "strategy_positive_month_rate": (g["strategy_return"] > 0).mean(),
            "active_positive_month_rate": (a > 0).mean(),
            "strategy_vol_annualized": g["strategy_return"].std(ddof=1) * math.sqrt(12),
            "active_vol_annualized": a.std(ddof=1) * math.sqrt(12),
        })
    pd.DataFrame(regime_rows).sort_values("months", ascending=False).to_csv(OUT / "regime_performance.csv", index=False)

    # Rolling windows.
    roll_rows = []
    for window in (36, 60):
        if len(perf) < window:
            continue
        for end_i in range(window - 1, len(perf)):
            g = perf.iloc[end_i - window + 1:end_i + 1]
            years = window / 12
            s_growth = float((1 + g["strategy_return"]).prod())
            b_growth = float((1 + g["spy_return"]).prod())
            s_cagr = s_growth ** (1 / years) - 1
            b_cagr = b_growth ** (1 / years) - 1
            roll_rows.append({
                "window_months": window,
                "start": g["date"].iloc[0],
                "end": g["next_date"].iloc[-1],
                "strategy_cagr": s_cagr,
                "spy_cagr": b_cagr,
                "excess_cagr": s_cagr - b_cagr,
            })
    rolling = pd.DataFrame(roll_rows)
    rolling.to_csv(OUT / "rolling_windows.csv", index=False)
    rolling_summary = []
    for window, g in rolling.groupby("window_months"):
        rolling_summary.append({
            "window_months": int(window),
            "windows": len(g),
            "pct_windows_strategy_outperformed": float((g["excess_cagr"] > 0).mean()),
            "min_excess_cagr": g["excess_cagr"].min(),
            "median_excess_cagr": g["excess_cagr"].median(),
            "max_excess_cagr": g["excess_cagr"].max(),
        })
    pd.DataFrame(rolling_summary).to_csv(OUT / "rolling_summary.csv", index=False)

    # Cost sensitivity. Existing backtest uses half-L1 turnover * bps once. If bps is a
    # per-dollar one-way trading cost, post-initial rebalances trade ~2*turnover notional.
    traded_notional = 2.0 * perf["turnover"].astype(float)
    if len(traded_notional):
        traded_notional.iloc[0] = 1.0  # initial portfolio is buys only from cash
    cost_rows = []
    for bps in (0, 5, 10, 20, 50):
        net = perf["gross_return"] - (bps / 10000.0) * traded_notional
        st = perf_stats(net)
        st.update({
            "one_way_cost_bps": bps,
            "cost_definition": "bps_per_dollar_traded; first build=1x NAV, later traded_notional=2*half_L1_turnover",
            "excess_cagr_vs_spy": st.get("cagr", np.nan) - perf_stats(perf["spy_return"]).get("cagr", np.nan),
            "annualized_cost_drag_approx": float(((bps / 10000.0) * traded_notional).mean() * 12),
        })
        cost_rows.append(st)
    pd.DataFrame(cost_rows).to_csv(OUT / "cost_sensitivity.csv", index=False)

    # Drawdown and tail behavior.
    tail_rows = []
    for name, col in (("strategy", "strategy_return"), ("SPY", "spy_return")):
        st = perf_stats(perf[col])
        tail_rows.append({"portfolio": name, **st, "longest_underwater_months": longest_underwater_months(perf[col])})
    pd.DataFrame(tail_rows).to_csv(OUT / "drawdown_stats.csv", index=False)

    worst = perf.nsmallest(10, "strategy_return")[["date", "next_date", "regime", "strategy_return", "spy_return", "turnover"]].copy()
    worst["active_return"] = worst["strategy_return"] - worst["spy_return"]
    worst.to_csv(OUT / "worst_months.csv", index=False)
    best = perf.nlargest(10, "strategy_return")[["date", "next_date", "regime", "strategy_return", "spy_return", "turnover"]].copy()
    best["active_return"] = best["strategy_return"] - best["spy_return"]
    best.to_csv(OUT / "best_months.csv", index=False)

    # Holdings / concentration diagnostics.
    sector = holdings.groupby(["date", "GICS Sector"], as_index=False)["weight"].sum()
    sector_pivot = sector.pivot(index="date", columns="GICS Sector", values="weight").fillna(0)
    sector_pivot.to_csv(OUT / "sector_weights_monthly.csv")
    sector_summary = pd.DataFrame({
        "avg_weight": sector_pivot.mean(),
        "max_weight": sector_pivot.max(),
        "months_present": (sector_pivot > 0).sum(),
    }).sort_values("avg_weight", ascending=False)
    sector_summary.to_csv(OUT / "sector_summary.csv")
    ticker_summary = holdings.groupby(["Symbol", "Security", "GICS Sector"], as_index=False).agg(
        months_selected=("date", "nunique"), avg_weight=("weight", "mean"), max_weight=("weight", "max"), avg_score=("score", "mean")
    ).sort_values(["months_selected", "avg_weight"], ascending=False)
    ticker_summary.to_csv(OUT / "ticker_persistence.csv", index=False)
    monthly_hhi = holdings.groupby("date")["weight"].apply(lambda w: float(np.square(w).sum())).rename("stock_weight_hhi")
    monthly_hhi.to_csv(OUT / "concentration_monthly.csv")

    # Add RSP equal-weight benchmark when Yahoo is available.
    rsp = rsp_returns(perf)
    benchmark_rows = [{"portfolio": "regime_factor", **perf_stats(perf["strategy_return"])}, {"portfolio": "SPY", **perf_stats(perf["spy_return"])}]
    if rsp is not None and rsp.notna().sum() >= 90:
        perf["rsp_return"] = rsp
        benchmark_rows.append({"portfolio": "RSP", **perf_stats(perf["rsp_return"])})
        perf[["date", "next_date", "strategy_return", "spy_return", "rsp_return"]].to_csv(OUT / "benchmark_returns.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(OUT / "benchmark_summary.csv", index=False)

    # Explicit methodology audit; these are reasons to rerun before trusting the headline CAGR.
    audit = pd.DataFrame([
        {"severity": "HIGH", "finding": "same_close_execution", "detail": "Signals use month-end close/volume through the signal date while returns start from that same close. This assumes execution at a close only known after the signal is computed.", "next_action": "Execute at next trading session open (preferred) or next close."},
        {"severity": "HIGH", "finding": "monthly_macro_release_timing", "detail": "Monthly FRED observation dates are reference-period dates, not publication timestamps. Existing fixed lags can admit CPI/UNRATE/Sahm/CFNAI before actual release.", "next_action": "Use release-date-aware availability or conservative reference-date lags; rerun regimes."},
        {"severity": "HIGH", "finding": "survivorship_bias", "detail": "Historical tests use current S&P 500 constituents; removed, acquired and failed constituents are absent.", "next_action": "Reconstruct historical membership, preferably from historical IVV holdings / index-change history, and quantify missing delisted-price coverage."},
        {"severity": "MEDIUM", "finding": "transaction_cost_convention", "detail": "Current cost uses half-L1 turnover * bps. If bps is a one-way per-dollar trading cost, later rebalances trade about 2x the half-L1 turnover in gross notional.", "next_action": "Report both conventions; default to full traded-notional cost."},
        {"severity": "MEDIUM", "finding": "macro_revision_bias", "detail": "Most FRED history is latest vintage and may contain revisions; SAHMREALTIME is designed as a real-time series but other macro inputs remain revised.", "next_action": "Use ALFRED vintages where feasible or stress-test with longer release lags / fewer revised macro inputs."},
        {"severity": "MEDIUM", "finding": "statistical_power", "detail": "103 monthly observations are a small sample for validating a multi-factor model with regime switching.", "next_action": "Evaluate rolling windows, bootstrap active returns, ablations and out-of-sample parameter freezes."},
    ])
    audit.to_csv(OUT / "methodology_audit.csv", index=False)

    print("Research analysis complete", flush=True)
    print(pd.DataFrame(benchmark_rows).to_string(index=False), flush=True)
    print(pd.DataFrame([sig]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
