from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import backtest_v2 as v2
import screener as sc
from portfolio_constraints import build_portfolio_strict


BASE_STRATEGY = "full_dynamic_n10_invvol"


def build_uncapped(ranked: pd.DataFrame, n: int, allocation: str) -> pd.DataFrame:
    picks = ranked[ranked["eligible"]].head(n).copy()
    if picks.empty:
        return picks
    if allocation == "equal":
        raw = pd.Series(1.0, index=picks.index)
    elif allocation == "score":
        raw = (picks["score"] / 100.0).clip(lower=0.01)
    elif allocation == "invvol":
        raw = (1.0 / picks["vol_63d"].replace(0, np.nan)) * (picks["score"] / 100.0).clip(lower=0.01)
    else:
        raise ValueError(f"Unknown allocation: {allocation}")
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float)
    picks["weight"] = raw / raw.sum()
    return picks.sort_values("weight", ascending=False)


def concentration_metrics(port: pd.DataFrame) -> dict:
    if port.empty:
        return {
            "max_stock_weight": np.nan,
            "top3_stock_weight": np.nan,
            "stock_hhi": np.nan,
            "effective_stock_count": np.nan,
            "max_sector_weight": np.nan,
            "sector_hhi": np.nan,
            "effective_sector_count": np.nan,
        }
    w = port["weight"].astype(float)
    sector_w = port.groupby("GICS Sector")["weight"].sum().astype(float)
    stock_hhi = float((w * w).sum())
    sector_hhi = float((sector_w * sector_w).sum())
    return {
        "max_stock_weight": float(w.max()),
        "top3_stock_weight": float(w.nlargest(min(3, len(w))).sum()),
        "stock_hhi": stock_hhi,
        "effective_stock_count": 1.0 / stock_hhi if stock_hhi > 0 else np.nan,
        "max_sector_weight": float(sector_w.max()),
        "sector_hhi": sector_hhi,
        "effective_sector_count": 1.0 / sector_hhi if sector_hhi > 0 else np.nan,
    }


def _zero_factor(config: dict, factor: str) -> dict:
    c = copy.deepcopy(config)
    for weights in c["regime_weights"].values():
        weights[factor] = 0.0
    if factor == "macro_sector":
        c["macro_sector_tilts"] = {regime: {} for regime in c["regime_weights"]}
    return c


def score_variants(config: dict) -> dict[str, dict]:
    variants = {"full_dynamic": copy.deepcopy(config)}

    static = copy.deepcopy(config)
    neutral = dict(static["regime_weights"]["NEUTRAL"])
    neutral["macro_sector"] = 0.0
    for regime in static["regime_weights"]:
        static["regime_weights"][regime] = dict(neutral)
    static["macro_sector_tilts"] = {regime: {} for regime in static["regime_weights"]}
    variants["static_neutral"] = static

    variants["dynamic_no_macro_sector"] = _zero_factor(config, "macro_sector")
    for factor in ["sector_rotation", "momentum", "quality", "value", "growth", "low_vol"]:
        variants[f"no_{factor}"] = _zero_factor(config, factor)
    return variants


def strategy_specs() -> list[dict]:
    specs = [
        {"strategy": BASE_STRATEGY, "score_variant": "full_dynamic", "n": 10, "allocation": "invvol", "category": "base"},
        {"strategy": "full_dynamic_n10_equal", "score_variant": "full_dynamic", "n": 10, "allocation": "equal", "category": "allocation"},
        {"strategy": "full_dynamic_n10_score", "score_variant": "full_dynamic", "n": 10, "allocation": "score", "category": "allocation"},
        {"strategy": "full_dynamic_n5_invvol", "score_variant": "full_dynamic", "n": 5, "allocation": "invvol", "category": "breadth"},
        {"strategy": "full_dynamic_n20_invvol", "score_variant": "full_dynamic", "n": 20, "allocation": "invvol", "category": "breadth"},
        {"strategy": "full_dynamic_strict_12_30", "score_variant": "full_dynamic", "n": 10, "allocation": "strict", "category": "constraint_baseline"},
        {"strategy": "static_neutral_n10_invvol", "score_variant": "static_neutral", "n": 10, "allocation": "invvol", "category": "macro_ablation"},
        {"strategy": "dynamic_no_macro_sector_n10", "score_variant": "dynamic_no_macro_sector", "n": 10, "allocation": "invvol", "category": "macro_ablation"},
        {"strategy": "no_sector_rotation_n10", "score_variant": "no_sector_rotation", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
        {"strategy": "no_momentum_n10", "score_variant": "no_momentum", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
        {"strategy": "no_quality_n10", "score_variant": "no_quality", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
        {"strategy": "no_value_n10", "score_variant": "no_value", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
        {"strategy": "no_growth_n10", "score_variant": "no_growth", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
        {"strategy": "no_low_vol_n10", "score_variant": "no_low_vol", "n": 10, "allocation": "invvol", "category": "factor_ablation"},
    ]
    return specs


def _portfolio(ranked: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if spec["allocation"] == "strict":
        return build_portfolio_strict(ranked, int(spec["n"]), 0.12, 0.30)
    return build_uncapped(ranked, int(spec["n"]), str(spec["allocation"]))


def _turnover(target: dict[str, float], prev_end: dict[str, float]) -> tuple[float, float]:
    if not prev_end:
        return 1.0, 1.0
    half_l1 = 0.5 * sum(abs(target.get(s, 0.0) - prev_end.get(s, 0.0)) for s in set(target) | set(prev_end))
    return float(half_l1), float(2.0 * half_l1)


def _end_weights(target: dict[str, float], returns: dict[str, float]) -> dict[str, float]:
    raw = {s: target[s] * (1.0 + returns.get(s, 0.0)) for s in target}
    den = float(sum(raw.values()))
    return {s: v / den for s, v in raw.items()} if den > 0 else dict(target)


def _perf_stats(r: pd.Series) -> dict:
    return v2.perf_stats(pd.Series(r, dtype=float))


def _summary_row(name: str, group: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    stats = _perf_stats(group["net_return"])
    merged = group[["signal_date", "net_return"]].merge(
        benchmark[["signal_date", "spy_return"]], on="signal_date", how="inner"
    )
    active = merged["net_return"] - merged["spy_return"]
    active_std = float(active.std(ddof=1))
    stats.update({
        "portfolio": name,
        "avg_monthly_turnover_half_l1": float(group["turnover_half_l1"].mean()),
        "avg_monthly_traded_notional": float(group["traded_notional"].mean()),
        "annualized_excess_return_arith_vs_spy": float(active.mean() * 12) if len(active) else np.nan,
        "information_ratio_vs_spy": float(active.mean() / active_std * math.sqrt(12)) if active_std > 0 else np.nan,
        "active_t_stat_iid": float(active.mean() / (active_std / math.sqrt(len(active)))) if active_std > 0 and len(active) > 1 else np.nan,
        "avg_max_stock_weight": float(group["max_stock_weight"].mean()),
        "peak_max_stock_weight": float(group["max_stock_weight"].max()),
        "avg_top3_stock_weight": float(group["top3_stock_weight"].mean()),
        "avg_effective_stock_count": float(group["effective_stock_count"].mean()),
        "avg_max_sector_weight": float(group["max_sector_weight"].mean()),
        "peak_max_sector_weight": float(group["max_sector_weight"].max()),
        "avg_effective_sector_count": float(group["effective_sector_count"].mean()),
        "avg_valid_weight_fraction": float(group["valid_weight_fraction"].mean()),
    })
    return stats


def _benchmark_rows(benchmark: pd.DataFrame) -> list[dict]:
    rows = []
    for name, col in [("SPY", "spy_return"), ("RSP", "rsp_return")]:
        stats = _perf_stats(benchmark[col])
        stats["portfolio"] = name
        rows.append(stats)
    return rows


def _period_summary(monthly: pd.DataFrame, benchmark: pd.DataFrame, start: str, end: str, label: str) -> list[dict]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    b = benchmark[(benchmark["signal_date"] >= lo) & (benchmark["signal_date"] <= hi)]
    out = []
    for strategy, g in monthly.groupby("strategy"):
        x = g[(g["signal_date"] >= lo) & (g["signal_date"] <= hi)]
        if len(x) < 12:
            continue
        row = _summary_row(strategy, x, b)
        row["period"] = label
        out.append(row)
    return out


def run(args) -> None:
    config = sc.load_config(args.config)
    cache = sc.Cache(Path(config.get("cache_dir", ".cache_sp500_screener")))
    universe = sc.get_universe(cache, 87600, None)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc).date())

    symbols = universe["Yahoo Symbol"].tolist() + ["SPY", "RSP"] + list(config["sector_etfs"].values())
    open_px, close, volume = v2.download_ohlcv(
        list(dict.fromkeys(symbols)), start - pd.Timedelta(days=430), end, int(config.get("price_batch_size", 80))
    )
    dates = bt.month_ends(close.index, start, end)
    if len(dates) < 13:
        raise RuntimeError("Need at least 13 monthly observations")
    signal_dates = dates[:-1]

    fred = {name: sc.fred_series(cache, name, sid, 87600) for name, sid in sc.FRED_SERIES.items()}
    bt.MACRO_LAG_DAYS = v2.MACRO_LAG_DAYS_V2.copy()
    fund_panel = v2.point_in_time_fundamentals_fast(universe, signal_dates, cache)
    fund_panel["date"] = pd.to_datetime(fund_panel["date"])
    yahoo = dict(zip(universe["Symbol"], universe["Yahoo Symbol"]))

    configs = score_variants(config)
    specs = strategy_specs()
    state: dict[str, dict[str, float]] = {s["strategy"]: {} for s in specs}
    monthly_rows: list[dict] = []
    holding_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    cost_rate = args.cost_bps / 10000.0

    for i, signal_date in enumerate(signal_dates):
        next_signal = dates[i + 1]
        exec_start = v2.next_session(open_px.index, signal_date)
        exec_end = v2.next_session(open_px.index, next_signal)
        if exec_start is None or exec_end is None:
            continue

        tech = sc.price_features(close.loc[:signal_date], volume.loc[:signal_date], universe)
        sector = sc.sector_rotation(close.loc[:signal_date], universe, tech, config["sector_etfs"])
        fund = fund_panel[fund_panel["date"] == signal_date].drop(columns="date")
        features = sc.build_features(universe, tech, fund, sector)
        macro = bt.macro_asof(fred, signal_date)
        ranked = {name: sc.score_stocks(features, macro, cfg) for name, cfg in configs.items()}

        spy_ret = v2.price_return_at_open(open_px, "SPY", exec_start, exec_end)
        rsp_ret = v2.price_return_at_open(open_px, "RSP", exec_start, exec_end)
        benchmark_rows.append({
            "signal_date": signal_date,
            "execution_date": exec_start,
            "exit_execution_date": exec_end,
            "regime": macro.regime,
            "spy_return": spy_ret,
            "rsp_return": rsp_ret,
        })

        for spec in specs:
            strategy = spec["strategy"]
            port = _portfolio(ranked[spec["score_variant"]], spec)
            if port.empty:
                continue
            target = dict(zip(port["Symbol"], port["weight"]))
            half_l1, traded_notional = _turnover(target, state[strategy])

            indiv: dict[str, float] = {}
            for sym in target:
                ys = yahoo.get(sym)
                if ys:
                    ret = v2.price_return_at_open(open_px, ys, exec_start, exec_end)
                    if np.isfinite(ret):
                        indiv[sym] = ret
            valid_weight = float(sum(target[s] for s in indiv))
            if valid_weight < args.min_valid_weight:
                print(f"SKIP {signal_date.date()} {strategy}: valid weight {valid_weight:.1%}", flush=True)
                continue

            gross = float(sum(target[s] * indiv[s] for s in indiv) / valid_weight)
            cost = float(traded_notional * cost_rate)
            net = gross - cost
            state[strategy] = _end_weights(target, indiv)
            conc = concentration_metrics(port)

            monthly_rows.append({
                "strategy": strategy,
                "category": spec["category"],
                "score_variant": spec["score_variant"],
                "n": spec["n"],
                "allocation": spec["allocation"],
                "signal_date": signal_date,
                "execution_date": exec_start,
                "exit_execution_date": exec_end,
                "regime": macro.regime,
                "gross_return": gross,
                "net_return": net,
                "turnover_half_l1": half_l1,
                "traded_notional": traded_notional,
                "cost": cost,
                "valid_weight_fraction": valid_weight,
                **conc,
            })
            for _, h in port.iterrows():
                holding_rows.append({
                    "strategy": strategy,
                    "signal_date": signal_date,
                    "execution_date": exec_start,
                    "Symbol": h["Symbol"],
                    "Security": h["Security"],
                    "GICS Sector": h["GICS Sector"],
                    "score": h["score"],
                    "weight": h["weight"],
                })

        base = [r for r in monthly_rows if r["strategy"] == BASE_STRATEGY and pd.Timestamp(r["signal_date"]) == signal_date]
        if base:
            b = base[-1]
            print(
                f"{signal_date.date()} {macro.regime:16s} base={b['net_return']:+.2%} SPY={spy_ret:+.2%} "
                f"max_stock={b['max_stock_weight']:.1%} max_sector={b['max_sector_weight']:.1%}",
                flush=True,
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(monthly_rows)
    holdings = pd.DataFrame(holding_rows)
    benchmark = pd.DataFrame(benchmark_rows)
    if monthly.empty:
        raise RuntimeError("No research backtest observations")
    monthly["signal_date"] = pd.to_datetime(monthly["signal_date"])
    benchmark["signal_date"] = pd.to_datetime(benchmark["signal_date"])

    summary_rows = [_summary_row(strategy, g, benchmark) for strategy, g in monthly.groupby("strategy")]
    summary_rows.extend(_benchmark_rows(benchmark))
    summary = pd.DataFrame(summary_rows).sort_values("cagr", ascending=False, na_position="last")

    base_stats = summary[summary["portfolio"] == BASE_STRATEGY].iloc[0]
    effects = summary[~summary["portfolio"].isin(["SPY", "RSP", BASE_STRATEGY])].copy()
    effects["delta_cagr_vs_base"] = effects["cagr"] - float(base_stats["cagr"])
    effects["delta_sharpe_vs_base"] = effects["sharpe_0rf"] - float(base_stats["sharpe_0rf"])
    effects["delta_max_drawdown_vs_base"] = effects["max_drawdown"] - float(base_stats["max_drawdown"])

    base_monthly = monthly[monthly["strategy"] == BASE_STRATEGY].copy()
    cost_rows = []
    for bps in [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        r = base_monthly["gross_return"] - base_monthly["traded_notional"] * (bps / 10000.0)
        row = _perf_stats(r)
        row["cost_bps_per_dollar_traded"] = bps
        cost_rows.append(row)
    cost_sensitivity = pd.DataFrame(cost_rows)

    regime_rows = []
    for (strategy, regime), g in monthly.groupby(["strategy", "regime"]):
        if len(g) < 3:
            continue
        stats = _perf_stats(g["net_return"])
        stats.update({"strategy": strategy, "regime": regime})
        regime_rows.append(stats)
    regime_summary = pd.DataFrame(regime_rows)

    subperiod = pd.DataFrame(
        _period_summary(monthly, benchmark, "2018-01-01", "2021-12-31", "2018-2021")
        + _period_summary(monthly, benchmark, "2022-01-01", "2026-12-31", "2022-2026")
    )

    monthly.to_csv(out / "monthly.csv", index=False)
    holdings.to_csv(out / "holdings.csv", index=False)
    benchmark.to_csv(out / "benchmarks.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    effects.to_csv(out / "ablation_effects.csv", index=False)
    cost_sensitivity.to_csv(out / "cost_sensitivity.csv", index=False)
    regime_summary.to_csv(out / "regime_summary.csv", index=False)
    subperiod.to_csv(out / "subperiod_summary.csv", index=False)
    pd.DataFrame(specs).to_csv(out / "strategy_specs.csv", index=False)
    pd.DataFrame([
        ["universe", "Current S&P 500 constituents; survivorship bias remains until historical-universe price coverage is resolved"],
        ["signal", "Month-end close/volume, computed after close"],
        ["execution", "Next trading-session adjusted open"],
        ["fundamentals", "SEC facts filed on/before signal date only"],
        ["macro", f"Point-in-time proxy lags {v2.MACRO_LAG_DAYS_V2}; latest-vintage FRED revision bias remains"],
        ["base_allocation", "Top 10, inverse-volatility x score, NO stock or sector hard cap"],
        ["constraint_baseline", "Separate comparison strategy keeps 12% stock / 30% sector caps"],
        ["cost", f"{args.cost_bps} bps per dollar traded; sensitivity also reported at 0/5/10/25/50/100 bps"],
        ["research_warning", "Ablations are diagnostic, not a license to select the best in-sample variant; confirmation requires historical constituents and out-of-sample testing"],
    ], columns=["assumption", "value"]).to_csv(out / "assumptions.csv", index=False)

    print("\n=== RESEARCH BACKTEST SUMMARY ===\n" + summary.to_string(index=False), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--min-valid-weight", type=float, default=0.95)
    p.add_argument("--output-dir", default="data/research_backtest")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
